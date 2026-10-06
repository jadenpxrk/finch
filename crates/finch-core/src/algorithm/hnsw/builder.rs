use super::*;

/// Builds an HNSW graph over dense f32 vectors.
pub struct HnswBuilder {
    pub(super) params: HnswIndexParams,
    pub(super) dim: usize,
    pub(super) heuristic_dim: usize,
    pub(super) vec_stride_floats: usize,
    pub(super) normalize_cosine: bool,
    pub(super) normalize_buf: Vec<f32>,
    // Flat row-major vector storage: node_id -> vectors[(node_id*dim)..((node_id+1)*dim)]
    pub(super) vectors: Vec<f32>,
    pub(super) keys: Vec<u64>,
    pub(super) levels: Vec<u32>,
    // L0: flat [count × (2*m)]
    pub(super) l0_neighbors: Vec<u32>,
    // Upper levels (level>=1): sparse-per-level chunks, aligned by node_id via offsets.
    // lv_idx = level-1. Each present node has exactly `m` neighbors in `neighbors` (filled with u32::MAX).
    pub(super) upper_neighbors: Vec<UpperLevelNeighbors>,
    pub(super) entry_point: u32,
    pub(super) max_level: usize,
    pub(super) metric_type: MetricType,
    // Scratch space to avoid per-insert allocations during beam search.
    pub(super) beam_candidates: BinaryHeap<MinDistNode>,
    pub(super) beam_results: BinaryHeap<DistNode>,
    pub(super) beam_out: Vec<(f32, u32)>,
    // Scratch space to avoid per-backlink allocations during pruning.
    pub(super) backlink_scored: Vec<(f32, u32)>,
    pub(super) backlink_selected: Vec<u32>,
    pub(super) backlink_selected_is_new: Vec<bool>,
}

/// Where a node's neighbor slot lives: the flat L0 array or one upper level.
struct SlotRef {
    upper_idx: Option<usize>,
    range: std::ops::Range<usize>,
}

/// Deduplicated L0 repair candidates of one node; `marks[id] == mark` means picked.
struct RepairCandidates {
    list: Vec<u32>,
    marks: Vec<u32>,
    mark: u32,
    cap: usize,
}

impl RepairCandidates {
    fn start_node(&mut self) {
        self.mark = self.mark.wrapping_add(1);
        if self.mark == 0 {
            self.marks.fill(0);
            self.mark = 1;
        }
        self.list.clear();
    }

    fn is_full(&self) -> bool {
        self.list.len() >= self.cap
    }

    /// Adds `candidate` unless it is empty, `self_node`, or already picked; returns whether the list is full.
    #[inline]
    fn push(&mut self, self_node: u32, candidate: u32) -> bool {
        if self.list.len() >= self.cap || candidate == u32::MAX || candidate == self_node {
            return self.list.len() >= self.cap;
        }
        let idx = candidate as usize;
        let Some(slot) = self.marks.get_mut(idx) else {
            return false;
        };
        if *slot == self.mark {
            return false;
        }
        *slot = self.mark;
        self.list.push(candidate);
        self.list.len() >= self.cap
    }

    /// Direct links, reverse links, then two-hop links of `node`, up to `cap`.
    fn collect_graph(&mut self, current: &[u32], incoming: &[u32], node: usize, l0_m: usize) {
        let node_id = node as u32;
        let n = self.marks.len();
        let base = node * l0_m;
        for &nb in &current[base..base + l0_m] {
            if self.push(node_id, nb) {
                break;
            }
        }
        if !self.is_full() {
            for &nb in incoming {
                if self.push(node_id, nb) {
                    break;
                }
            }
        }

        let direct_len = self.list.len();
        for idx in 0..direct_len {
            if self.is_full() {
                break;
            }
            let nb = self.list[idx] as usize;
            if nb >= n {
                continue;
            }
            let nb_base = nb * l0_m;
            for &hop2 in &current[nb_base..nb_base + l0_m] {
                if self.push(node_id, hop2) {
                    break;
                }
            }
        }
    }
}

struct L0RepairSearch {
    visited: VisitedList,
    candidates: BinaryHeap<MinDistNode>,
    results: BinaryHeap<DistNode>,
}

/// Double-buffered L0 graph plus per-node scratch for `repair_l0_neighbors`.
struct L0RepairState {
    current: Vec<u32>,
    next: Vec<u32>,
    incoming: Vec<Vec<u32>>,
    picks: RepairCandidates,
    scored: Vec<(f32, u32)>,
    selected: Vec<u32>,
    search: L0RepairSearch,
}

impl L0RepairState {
    fn new(current: Vec<u32>, n: usize, l0_m: usize) -> Self {
        let cap = L0_REPAIR_CANDIDATE_CAP.max(l0_m);
        L0RepairState {
            next: vec![u32::MAX; current.len()],
            current,
            incoming: (0..n).map(|_| Vec::new()).collect(),
            picks: RepairCandidates {
                list: Vec::with_capacity(cap),
                marks: vec![0u32; n],
                mark: 1,
                cap,
            },
            scored: Vec::with_capacity(cap),
            selected: Vec::with_capacity(l0_m),
            search: L0RepairSearch {
                visited: VisitedList::new(),
                candidates: BinaryHeap::new(),
                results: BinaryHeap::new(),
            },
        }
    }

    fn rebuild_incoming(&mut self, l0_m: usize) {
        let n = self.incoming.len();
        for links in self.incoming.iter_mut() {
            links.clear();
        }
        for (node, links) in self.current.chunks_exact(l0_m).enumerate().take(n) {
            for &nb in links {
                if nb == u32::MAX || nb as usize >= n {
                    continue;
                }
                self.incoming[nb as usize].push(node as u32);
            }
        }
    }
}

impl HnswBuilder {
    /// Creates an empty builder for `dim`-dimensional vectors.
    pub fn new(dim: usize, params: HnswIndexParams) -> Self {
        let metric_type = params.metric;
        let heuristic_dim = heuristic_dim(&params.build_tuning, dim);
        // align the in-memory vector stride and avoid 1024-byte
        // multiples which can cause severe cache conflict slowdowns (notably at
        // 1536-dim f32 = 6144 bytes).
        let vec_stride_floats = aligned_vector_stride_floats(dim);
        HnswBuilder {
            params,
            dim,
            heuristic_dim,
            vec_stride_floats,
            normalize_cosine: metric_type == MetricType::Cosine,
            normalize_buf: Vec::new(),
            vectors: Vec::new(),
            keys: Vec::new(),
            levels: Vec::new(),
            l0_neighbors: Vec::new(),
            upper_neighbors: Vec::new(),
            entry_point: u32::MAX,
            max_level: 0,
            metric_type,
            beam_candidates: BinaryHeap::new(),
            beam_results: BinaryHeap::new(),
            beam_out: Vec::new(),
            backlink_scored: Vec::new(),
            backlink_selected: Vec::new(),
            backlink_selected_is_new: Vec::new(),
        }
    }

    /// Builder distances run on normalized vectors, so cosine is `1 - ip`.
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        hnsw_distance(self.metric_type, a, b)
    }

    /// Assign a random level to a new node
    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        random_level_with_scaling(&mut rng, self.params.scaling_factor)
    }

    #[inline]
    fn vec_slice<'a>(&self, vectors: &'a [f32], node_id: u32) -> &'a [f32] {
        vec_slice_raw(vectors, self.vec_stride_floats, self.dim, node_id)
    }

    #[inline]
    fn heuristic_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        hnsw_heuristic_distance(self.metric_type, self.heuristic_dim, self.dim, a, b)
    }

    fn view(&self) -> VectorView<'_> {
        VectorView {
            metric_type: self.metric_type,
            vectors: &self.vectors,
            vec_stride_floats: self.vec_stride_floats,
            dim: self.dim,
            heuristic_dim: self.heuristic_dim,
            tuning: self.params.build_tuning,
        }
    }

    #[inline]
    fn upper_slot(&self, node: u32, level: usize) -> &[u32] {
        match self.upper_neighbors.get(level - 1) {
            Some(ul) => ul.slot(node, self.params.m),
            None => &[],
        }
    }

    fn slot_ref(&self, node: u32, level: usize) -> Option<SlotRef> {
        if level == 0 {
            let l0_m = self.params.m * 2;
            let start = (node as usize) * l0_m;
            return Some(SlotRef {
                upper_idx: None,
                range: start..start + l0_m,
            });
        }
        let lv_idx = level - 1;
        let range = self
            .upper_neighbors
            .get(lv_idx)?
            .slot_range(node, self.params.m)?;
        Some(SlotRef {
            upper_idx: Some(lv_idx),
            range,
        })
    }

    /// Adds `vector` under `key` and links it into the graph.
    pub fn add(&mut self, key: u64, vector: &[f32]) -> ZResult<()> {
        check_vector_dim(vector.len(), self.dim)?;

        let level = self.random_level();
        let node_id = (self.vectors.len() / self.vec_stride_floats) as u32;
        let normalized = self.normalize_cosine.then(|| {
            let mut buf = std::mem::take(&mut self.normalize_buf);
            buf.clear();
            buf.extend_from_slice(vector);
            normalize_l2(&mut buf);
            buf
        });
        let query = normalized.as_deref().unwrap_or(vector);
        self.append_node(key, query, node_id, level);
        self.link_new_node(query, node_id, level);
        if let Some(buf) = normalized {
            self.normalize_buf = buf;
        }
        Ok(())
    }

    /// Stores the vector, key, and level, and allocates empty neighbor slots.
    fn append_node(&mut self, key: u64, vector: &[f32], node_id: u32, level: usize) {
        self.vectors.extend_from_slice(vector);
        if self.vec_stride_floats > self.dim {
            self.vectors.resize(
                self.vectors.len() + (self.vec_stride_floats - self.dim),
                0.0,
            );
        }
        self.keys.push(key);
        self.levels.push(level as u32);

        // Initialize L0 neighbors (2*m slots, initialized to u32::MAX = invalid)
        let l0_m = self.params.m * 2;
        self.l0_neighbors
            .resize(self.l0_neighbors.len() + l0_m, u32::MAX);
        reserve_upper_slots(&mut self.upper_neighbors, node_id, level, self.params.m);
    }

    fn link_new_node(&mut self, query: &[f32], node_id: u32, level: usize) {
        let entry = self.entry_point;
        if entry == u32::MAX {
            self.entry_point = node_id;
            self.max_level = level;
            return;
        }
        let current_max_level = self.max_level;

        // Phase 1: Greedy descent from max_level to level+1
        let mut ep = Entry {
            node: entry,
            dist: self.distance(query, self.vec_slice(&self.vectors, entry)),
        };
        for lv in (level + 1..=current_max_level).rev() {
            ep = self.descend_level(query, ep, lv);
        }

        // Phase 2: Beam search at each level from min(level, max_level) down to 0
        let m = self.params.m;
        let ef = self.params.ef_construction;
        let mut ep = ep.node;
        let mut selected: Vec<u32> = Vec::with_capacity(m * 2);
        for lv in (0..=level.min(current_max_level)).rev() {
            self.beam_search_into(query, ep, ef, lv);
            let next_ep = self.beam_out.first().map(|(_, id)| *id);
            let max_m = if lv == 0 { m * 2 } else { m };
            selected.clear();
            select_neighbors_query_into(&self.view(), &self.beam_out, max_m, query, &mut selected);

            self.set_neighbors(node_id, lv, &selected);
            for &nb in &selected {
                if nb == u32::MAX {
                    continue;
                }
                self.add_backlink(nb, lv, node_id, max_m);
            }

            if let Some(next_ep) = next_ep {
                ep = next_ep;
            }
        }

        // Update entry point if new node has higher level
        if level > current_max_level {
            self.entry_point = node_id;
            self.max_level = level;
        }
    }

    /// Greedy walk on one upper level until no neighbor is closer to `query`.
    fn descend_level(&self, query: &[f32], mut ep: Entry, level: usize) -> Entry {
        let mut changed = true;
        while changed {
            changed = false;
            for &nb in self.upper_slot(ep.node, level) {
                if nb == u32::MAX {
                    continue;
                }
                let d = self.distance(query, self.vec_slice(&self.vectors, nb));
                if d < ep.dist {
                    ep = Entry { node: nb, dist: d };
                    changed = true;
                }
            }
        }
        ep
    }

    fn beam_search_into(&mut self, query: &[f32], entry: u32, ef: usize, level: usize) {
        let entry_dist = self.distance(query, self.vec_slice(&self.vectors, entry));
        let count = self.vectors.len() / self.vec_stride_floats;

        VISITED_LIST.with(|visited_cell| {
            let mut visited = visited_cell.borrow_mut();
            visited.reset_for_len(count);

            self.beam_candidates.clear();
            self.beam_results.clear();
            // Ensure capacity for typical ef sizes without reallocating on every call.
            self.beam_candidates.reserve(ef + 8);
            self.beam_results.reserve(ef + 8);

            self.beam_candidates.push(MinDistNode(entry_dist, entry));
            self.beam_results.push(DistNode(entry_dist, entry));
            visited.insert(entry);
            self.expand_beam(query, ef, level, &mut visited);
        });

        self.beam_out.clear();
        self.beam_out.reserve(self.beam_results.len());
        while let Some(d) = self.beam_results.pop() {
            self.beam_out.push((d.0, d.1));
        }
        self.beam_out
            .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// Best-first expansion of the seeded beam heaps at `level`.
    fn expand_beam(&mut self, query: &[f32], ef: usize, level: usize, visited: &mut VisitedList) {
        while let Some(MinDistNode(dist, node)) = self.beam_candidates.pop() {
            if let Some(DistNode(worst, _)) = self.beam_results.peek() {
                if dist > *worst && self.beam_results.len() >= ef {
                    break;
                }
            }

            let neighbors = graph_neighbors(
                &self.l0_neighbors,
                &self.upper_neighbors,
                self.params.m,
                node,
                level,
            );
            for &nb in neighbors {
                if nb == u32::MAX || !visited.insert_if_absent(nb) {
                    continue;
                }
                let nb_dist = self.distance(query, self.vec_slice(&self.vectors, nb));

                let worst = self
                    .beam_results
                    .peek()
                    .map(|d| d.0)
                    .unwrap_or(f32::INFINITY);
                let improves = nb_dist < worst || self.beam_results.len() < ef;
                if !improves {
                    continue;
                }
                self.beam_candidates.push(MinDistNode(nb_dist, nb));
                self.beam_results.push(DistNode(nb_dist, nb));
                if self.beam_results.len() > ef {
                    self.beam_results.pop();
                }
            }
        }
    }

    #[cfg(test)]
    pub(super) fn select_neighbors_heuristic<F>(
        candidates: &[(f32, u32)],
        m: usize,
        mut dist_between: F,
    ) -> Vec<u32>
    where
        F: FnMut(u32, u32) -> f32,
    {
        // Standard HNSW heuristic: keep a candidate only if it is not "shadowed"
        // by an already-selected neighbor (diversification).
        //
        // For candidates sorted by increasing distance-to-query:
        // keep c iff for all selected s: dist(c, s) >= dist(c, query).
        let mut selected: Vec<u32> = Vec::with_capacity(m);
        for &(dist_to_query, cand) in candidates.iter() {
            if cand == u32::MAX {
                continue;
            }
            let mut ok = true;
            for &s in selected.iter() {
                let d = dist_between(cand, s);
                if should_prune_neighbor(1.0, d, dist_to_query) {
                    ok = false;
                    break;
                }
            }
            if ok {
                selected.push(cand);
                if selected.len() >= m {
                    break;
                }
            }
        }
        selected
    }

    /// Add a backlink to `node` at `level`, pruning to `max_m` when full.
    fn add_backlink(&mut self, node: u32, level: usize, new_node: u32, max_m: usize) {
        let Some(slot) = self.slot_ref(node, level) else {
            return;
        };
        let HnswBuilder {
            vectors,
            l0_neighbors,
            upper_neighbors,
            backlink_scored,
            backlink_selected,
            backlink_selected_is_new,
            metric_type,
            vec_stride_floats,
            dim,
            heuristic_dim,
            params,
            ..
        } = self;
        let links = match slot.upper_idx {
            None => &mut l0_neighbors[slot.range],
            Some(lv_idx) => &mut upper_neighbors[lv_idx].neighbors[slot.range],
        };
        if links.contains(&new_node) {
            return;
        }
        if let Some(pos) = links.iter().position(|&x| x == u32::MAX) {
            links[pos] = new_node;
            return;
        }

        let view = VectorView {
            metric_type: *metric_type,
            vectors,
            vec_stride_floats: *vec_stride_floats,
            dim: *dim,
            heuristic_dim: *heuristic_dim,
            tuning: params.build_tuning,
        };
        let vectors = view.vectors;
        let node_vec = vec_slice_raw(vectors, view.vec_stride_floats, view.dim, node);
        let score = |nb: u32| {
            let nb_vec = vec_slice_raw(vectors, view.vec_stride_floats, view.dim, nb);
            hnsw_heuristic_distance(
                view.metric_type,
                view.heuristic_dim,
                view.dim,
                node_vec,
                nb_vec,
            )
        };
        backlink_scored.clear();
        backlink_scored.reserve(max_m + 1);
        for &nb in links.iter().take(max_m) {
            if nb == u32::MAX {
                continue;
            }
            backlink_scored.push((score(nb), nb));
        }
        let d_new = score(new_node);

        let buf = BacklinkBuffers {
            scored: backlink_scored,
            selected: backlink_selected,
            selected_is_new: backlink_selected_is_new,
        };
        select_backlinks_into(&view, (d_new, new_node), max_m, buf);
        write_links(links, backlink_selected, max_m);
    }

    fn set_neighbors(&mut self, node: u32, level: usize, neighbors: &[u32]) {
        let Some(slot) = self.slot_ref(node, level) else {
            return;
        };
        let links = match slot.upper_idx {
            None => &mut self.l0_neighbors[slot.range],
            Some(lv_idx) => &mut self.upper_neighbors[lv_idx].neighbors[slot.range],
        };
        let len = links.len();
        write_links(links, neighbors, len);
    }

    /// Scores `candidates` against `node_vec` and runs the pruning heuristic into `selected`.
    fn rescore_select(
        &self,
        node_vec: &[f32],
        candidates: &[u32],
        scored: &mut Vec<(f32, u32)>,
        selected: &mut Vec<u32>,
    ) {
        scored.clear();
        for &cand in candidates {
            let d = self.heuristic_distance(node_vec, self.vec_slice(&self.vectors, cand));
            scored.push((d, cand));
        }
        scored.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        selected.clear();
        select_neighbors_precomputed_into(&self.view(), scored, self.params.m * 2, selected);
    }

    pub(super) fn refine_l0_neighbors(&mut self) {
        let Some(candidate_cap) = self.params.build_tuning.l0_refine_candidate_cap else {
            return;
        };
        let candidate_cap = candidate_cap.max(1);
        let n = self.keys.len();
        if n <= 1 {
            return;
        }
        let l0_m = self.params.m * 2;
        let original = self.l0_neighbors.clone();
        let mut candidates: Vec<u32> = Vec::with_capacity(candidate_cap.min(l0_m * l0_m));
        let mut scored: Vec<(f32, u32)> = Vec::with_capacity(candidate_cap.min(l0_m * l0_m));
        let mut selected: Vec<u32> = Vec::with_capacity(l0_m);

        for node in 0..n {
            let node_id = node as u32;
            let base = node * l0_m;
            collect_two_hop(&original, node, l0_m, candidate_cap, &mut candidates);

            let node_vec = self.vec_slice(&self.vectors, node_id);
            self.rescore_select(node_vec, &candidates, &mut scored, &mut selected);
            write_links(&mut self.l0_neighbors[base..base + l0_m], &selected, l0_m);
        }
    }

    fn repair_l0_entry_for_query(&self, query: &[f32]) -> u32 {
        let ep = self.entry_point;
        if ep == u32::MAX {
            return ep;
        }

        let mut ep = Entry {
            node: ep,
            dist: self.distance(query, self.vec_slice(&self.vectors, ep)),
        };
        for lv in (1..=self.max_level).rev() {
            ep = self.descend_level(query, ep, lv);
        }
        ep.node
    }

    /// Adds the nodes an L0 beam search from `entry` finds for `query` to `picks`.
    fn collect_l0_repair_search_candidates(
        &self,
        current: &[u32],
        query: &[f32],
        entry: u32,
        self_node: u32,
        picks: &mut RepairCandidates,
        search: &mut L0RepairSearch,
    ) {
        let ef = self.params.ef_construction;
        let n = self.keys.len();
        if entry == u32::MAX || entry as usize >= n || ef == 0 || picks.is_full() {
            return;
        }

        let L0RepairSearch {
            visited,
            candidates: search_candidates,
            results: search_results,
        } = search;
        visited.reset_for_len(n);
        search_candidates.clear();
        search_results.clear();
        search_candidates.reserve(ef + 8);
        search_results.reserve(ef + 8);

        let entry_dist = self.distance(query, self.vec_slice(&self.vectors, entry));
        search_candidates.push(MinDistNode(entry_dist, entry));
        search_results.push(DistNode(entry_dist, entry));
        visited.insert(entry);
        self.expand_repair_search(current, query, ef, search);

        while let Some(DistNode(_, node)) = search.results.pop() {
            if picks.push(self_node, node) {
                break;
            }
        }
    }

    /// Best-first expansion over the in-progress L0 graph `current`.
    fn expand_repair_search(
        &self,
        current: &[u32],
        query: &[f32],
        ef: usize,
        search: &mut L0RepairSearch,
    ) {
        let n = self.keys.len();
        let l0_m = self.params.m * 2;
        let L0RepairSearch {
            visited,
            candidates: search_candidates,
            results: search_results,
        } = search;
        while let Some(MinDistNode(dist, node)) = search_candidates.pop() {
            if let Some(DistNode(worst, _)) = search_results.peek() {
                if dist > *worst && search_results.len() >= ef {
                    break;
                }
            }

            let base = node as usize * l0_m;
            if base + l0_m > current.len() {
                continue;
            }
            for &nb in &current[base..base + l0_m] {
                if nb == u32::MAX || nb as usize >= n || !visited.insert_if_absent(nb) {
                    continue;
                }
                let nb_dist = self.distance(query, self.vec_slice(&self.vectors, nb));
                let worst = search_results.peek().map(|d| d.0).unwrap_or(f32::INFINITY);
                let improves = nb_dist < worst || search_results.len() < ef;
                if !improves {
                    continue;
                }
                search_candidates.push(MinDistNode(nb_dist, nb));
                search_results.push(DistNode(nb_dist, nb));
                if search_results.len() > ef {
                    search_results.pop();
                }
            }
        }
    }

    pub(super) fn repair_l0_neighbors(&mut self) {
        let n = self.keys.len();
        if n <= 1 || L0_REPAIR_PASSES == 0 {
            return;
        }

        let l0_m = self.params.m * 2;
        let mut state = L0RepairState::new(self.l0_neighbors.clone(), n, l0_m);
        for pass in 0..L0_REPAIR_PASSES {
            state.rebuild_incoming(l0_m);
            state.next.fill(u32::MAX);
            let use_search = pass + L0_REPAIR_SEARCH_PASSES >= L0_REPAIR_PASSES;
            for node in 0..n {
                self.repair_node(&mut state, node, use_search);
            }
            std::mem::swap(&mut state.current, &mut state.next);
        }

        self.l0_neighbors = state.current;
    }

    /// Re-picks `node`'s L0 links from its graph neighborhood (and optionally a search) into `state.next`.
    fn repair_node(&self, state: &mut L0RepairState, node: usize, use_search: bool) {
        let l0_m = self.params.m * 2;
        let node_id = node as u32;
        let base = node * l0_m;
        let node_vec = self.vec_slice(&self.vectors, node_id);

        state.picks.start_node();
        state
            .picks
            .collect_graph(&state.current, &state.incoming[node], node, l0_m);
        if use_search && !state.picks.is_full() {
            let entry = self.repair_l0_entry_for_query(node_vec);
            self.collect_l0_repair_search_candidates(
                &state.current,
                node_vec,
                entry,
                node_id,
                &mut state.picks,
                &mut state.search,
            );
        }

        self.rescore_select(
            node_vec,
            &state.picks.list,
            &mut state.scored,
            &mut state.selected,
        );
        write_links(&mut state.next[base..base + l0_m], &state.selected, l0_m);
    }
}

#[inline]
fn graph_neighbors<'a>(
    l0_neighbors: &'a [u32],
    upper_neighbors: &'a [UpperLevelNeighbors],
    m: usize,
    node: u32,
    level: usize,
) -> &'a [u32] {
    if level == 0 {
        return l0_links(l0_neighbors, m, node);
    }
    match upper_neighbors.get(level - 1) {
        Some(ul) => ul.slot(node, m),
        None => &[],
    }
}

/// Direct then two-hop L0 links of `node` in `graph`, deduplicated, up to `cap`.
fn collect_two_hop(graph: &[u32], node: usize, l0_m: usize, cap: usize, out: &mut Vec<u32>) {
    let node_id = node as u32;
    let base = node * l0_m;
    out.clear();
    for &nb in &graph[base..base + l0_m] {
        if nb != u32::MAX && nb != node_id && !out.contains(&nb) {
            out.push(nb);
        }
    }

    let direct_len = out.len();
    for i in 0..direct_len {
        if out.len() >= cap {
            break;
        }
        let nb_base = out[i] as usize * l0_m;
        for &hop2 in &graph[nb_base..nb_base + l0_m] {
            if hop2 == u32::MAX || hop2 == node_id || out.contains(&hop2) {
                continue;
            }
            out.push(hop2);
            if out.len() >= cap {
                break;
            }
        }
    }
}

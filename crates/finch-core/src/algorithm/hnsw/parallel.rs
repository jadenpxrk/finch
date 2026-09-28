use super::*;

/// Per-thread scratch buffers for parallel HNSW insertion.
pub(super) struct BuildScratch {
    pub(super) candidates: BinaryHeap<MinDistNode>,
    pub(super) results: BinaryHeap<DistNode>,
    pub(super) out: Vec<(f32, u32)>,
    pub(super) visited: VisitedList,
    pub(super) backlink_scored: Vec<(f32, u32)>,
    pub(super) backlink_selected: Vec<u32>,
    pub(super) backlink_selected_is_new: Vec<bool>,
    pub(super) selected: Vec<u32>,
    pub(super) unvisited: Vec<u32>,
}

impl BuildScratch {
    pub(super) fn new() -> Self {
        Self {
            candidates: BinaryHeap::new(),
            results: BinaryHeap::new(),
            out: Vec::new(),
            visited: VisitedList::new(),
            backlink_scored: Vec::new(),
            backlink_selected: Vec::new(),
            backlink_selected_is_new: Vec::new(),
            selected: Vec::new(),
            unvisited: Vec::new(),
        }
    }
}

/// Upper-level neighbor data for one level, shared across threads.
pub(super) struct SharedUpperLevel {
    /// offsets[node_id] = start index into `neighbors` (in AtomicU32 units),
    /// or u32::MAX when node has no slot at this level.
    pub(super) offsets: Vec<u32>,
    /// Concatenated fixed-size chunks (m entries per present node).
    pub(super) neighbors: Vec<AtomicU32>,
    /// Per-node locks for backlink pruning at this level.
    pub(super) locks: Vec<Mutex<()>>,
}

/// Thread-safe graph structure for parallel HNSW construction.
pub(super) struct SharedGraph {
    pub(super) vectors: Vec<f32>,
    pub(super) keys: Vec<u64>,
    pub(super) levels: Vec<u32>,
    pub(super) l0_neighbors: Vec<AtomicU32>,
    pub(super) l0_locks: Vec<Mutex<()>>,
    pub(super) upper_neighbors: Vec<SharedUpperLevel>,
    pub(super) ready: Vec<AtomicBool>,
    pub(super) entry_point: AtomicU32,
    pub(super) max_level: AtomicUsize,
    pub(super) promotion_lock: Mutex<()>,
    // Immutable params
    pub(super) dim: usize,
    pub(super) vec_stride_floats: usize,
    pub(super) heuristic_dim: usize,
    pub(super) tuning: HnswBuildTuning,
    pub(super) m: usize,
    pub(super) ef_construction: usize,
    pub(super) l0_m: usize,
    pub(super) metric_type: MetricType,
}

struct EntrySnapshot {
    node: u32,
    max_level: usize,
}

/// A node's slot on one upper level.
struct UpperSlot<'a> {
    level: &'a SharedUpperLevel,
    links: &'a [AtomicU32],
}

/// Links a pending node publishes on one level after all its levels are connected.
struct PendingBacklinks {
    level: usize,
    max_m: usize,
    selected: Vec<u32>,
}

/// How much of a full slot is re-scored and how many links survive pruning.
struct PruneLimits {
    score_len: usize,
    select_m: usize,
}

impl SharedGraph {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        hnsw_distance(self.metric_type, a, b)
    }

    #[inline]
    fn heuristic_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        hnsw_heuristic_distance(self.metric_type, self.heuristic_dim, self.dim, a, b)
    }

    #[inline]
    fn vec_slice(&self, node_id: u32) -> &[f32] {
        vec_slice_raw(&self.vectors, self.vec_stride_floats, self.dim, node_id)
    }

    /// Raw pointer to the start of a node's vector data, for prefetching.
    #[inline]
    fn vec_ptr(&self, node_id: u32) -> *const u8 {
        let start = (node_id as usize) * self.vec_stride_floats;
        (self.vectors.as_ptr() as *const u8).wrapping_add(start * std::mem::size_of::<f32>())
    }

    fn view(&self) -> VectorView<'_> {
        VectorView {
            metric_type: self.metric_type,
            vectors: &self.vectors,
            vec_stride_floats: self.vec_stride_floats,
            dim: self.dim,
            heuristic_dim: self.heuristic_dim,
            tuning: self.tuning,
        }
    }

    fn node_count(&self) -> usize {
        self.keys.len()
    }

    #[inline]
    fn is_ready(&self, node: u32) -> bool {
        self.ready
            .get(node as usize)
            .is_some_and(|ready| ready.load(Ordering::Acquire))
    }

    fn l0_links(&self, node: u32) -> &[AtomicU32] {
        let start = (node as usize) * self.l0_m;
        &self.l0_neighbors[start..start + self.l0_m]
    }

    fn upper_slot(&self, node: u32, level: usize) -> Option<UpperSlot<'_>> {
        let ul = self.upper_neighbors.get(level - 1)?;
        let off = ul.offsets.get(node as usize).copied().unwrap_or(u32::MAX);
        if off == u32::MAX {
            return None;
        }
        let off = off as usize;
        Some(UpperSlot {
            level: ul,
            links: &ul.neighbors[off..off + self.m],
        })
    }

    /// Read a node's ready neighbors at `level` (relaxed atomic loads: may see
    /// partial updates from concurrent writers, but each individual u32 is always valid).
    fn read_neighbors(&self, node: u32, level: usize, buf: &mut Vec<u32>) {
        buf.clear();
        let links = if level == 0 {
            self.l0_links(node)
        } else {
            match self.upper_slot(node, level) {
                Some(slot) => slot.links,
                None => return,
            }
        };
        for link in links {
            let v = link.load(Ordering::Relaxed);
            if v != u32::MAX && self.is_ready(v) {
                buf.push(v);
            }
        }
    }

    /// Set neighbors for a newly inserted node (only the inserting thread writes to its own slots).
    fn set_neighbors(&self, node: u32, level: usize, neighbors: &[u32]) {
        if level == 0 {
            store_links(self.l0_links(node), neighbors);
        } else if let Some(slot) = self.upper_slot(node, level) {
            store_links(slot.links, neighbors);
        }
    }

    /// Entry point and max level read as a consistent pair.
    fn entry_snapshot(&self) -> EntrySnapshot {
        loop {
            let max_level_before = self.max_level.load(Ordering::Acquire);
            let node = self.entry_point.load(Ordering::Acquire);
            let max_level_after = self.max_level.load(Ordering::Acquire);
            if max_level_before == max_level_after {
                return EntrySnapshot {
                    node,
                    max_level: max_level_after,
                };
            }
        }
    }
}

/// Overwrites `links` with `values`, padding with u32::MAX.
fn store_links(links: &[AtomicU32], values: &[u32]) {
    for (i, link) in links.iter().enumerate() {
        let val = values.get(i).copied().unwrap_or(u32::MAX);
        link.store(val, Ordering::Relaxed);
    }
}

/// Insert a single node into the shared graph (used by both sequential seed and parallel phases).
pub(super) fn par_insert_node(graph: &SharedGraph, node_id: u32, scratch: &mut BuildScratch) {
    let level = graph.levels[node_id as usize] as usize;
    let query = graph.vec_slice(node_id);

    let snapshot = graph.entry_snapshot();
    let (ep_id, current_max_level) = (snapshot.node, snapshot.max_level);
    if ep_id == u32::MAX {
        // Should not happen in normal flow (first node handled separately).
        return;
    }

    // Phase 1: Greedy descent from max_level to level+1
    let mut ep = Entry {
        node: ep_id,
        dist: graph.distance(query, graph.vec_slice(ep_id)),
    };
    let mut nb_buf: Vec<u32> = Vec::with_capacity(graph.m * 2);
    for lv in (level + 1..=current_max_level).rev() {
        ep = par_descend_level(graph, query, ep, lv, &mut nb_buf);
    }

    // Phase 2: Beam search + connect at each level from min(level, max_level) down to 0.
    // Defer backlink publication until all levels of the new node are connected so
    // concurrent inserts do not route through a partially constructed node.
    let top_level = level.min(current_max_level);
    let pending = par_connect_levels(graph, node_id, query, ep.node, top_level, scratch);

    // Publish reverse links only after the new node has all of its levels connected.
    for links in pending.into_iter().rev() {
        for nb in links.selected {
            if nb == u32::MAX {
                continue;
            }
            par_add_backlink(graph, nb, links.level, node_id, links.max_m, scratch);
        }
    }

    graph.ready[node_id as usize].store(true, Ordering::Release);

    if level > current_max_level {
        let _lock = graph.promotion_lock.lock();
        let cur = graph.max_level.load(Ordering::Acquire);
        if level > cur {
            graph.entry_point.store(node_id, Ordering::Release);
            graph.max_level.store(level, Ordering::Release);
        }
    }
}

/// Greedy walk on one level of the shared graph until no neighbor is closer to `query`.
fn par_descend_level(
    graph: &SharedGraph,
    query: &[f32],
    mut ep: Entry,
    level: usize,
    nb_buf: &mut Vec<u32>,
) -> Entry {
    let mut changed = true;
    while changed {
        changed = false;
        graph.read_neighbors(ep.node, level, nb_buf);
        // Prefetch all valid neighbor vectors before computing distances.
        for &nb in nb_buf.iter() {
            if nb != u32::MAX {
                prefetch_read(graph.vec_ptr(nb));
            }
        }
        for &nb in nb_buf.iter() {
            if nb == u32::MAX {
                continue;
            }
            let d = graph.distance(query, graph.vec_slice(nb));
            if d < ep.dist {
                ep = Entry { node: nb, dist: d };
                changed = true;
            }
        }
    }
    ep
}

/// Connects `node_id` on levels `top_level..=0` and returns the backlinks still to publish.
fn par_connect_levels(
    graph: &SharedGraph,
    node_id: u32,
    query: &[f32],
    mut ep: u32,
    top_level: usize,
    scratch: &mut BuildScratch,
) -> Vec<PendingBacklinks> {
    let m = graph.m;
    scratch.selected.clear();
    let mut pending: Vec<PendingBacklinks> = Vec::with_capacity(top_level + 1);

    for lv in (0..=top_level).rev() {
        par_beam_search(graph, query, ep, graph.ef_construction, lv, scratch);

        let next_ep = scratch.out.first().map(|(_, id)| *id);

        // Select best M neighbors using heuristic
        let max_m = if lv == 0 { m * 2 } else { m };
        scratch.selected.clear();
        select_neighbors_query_into(
            &graph.view(),
            &scratch.out,
            max_m,
            query,
            &mut scratch.selected,
        );

        // Connect new node to selected neighbors
        graph.set_neighbors(node_id, lv, &scratch.selected);
        pending.push(PendingBacklinks {
            level: lv,
            max_m,
            selected: scratch.selected.clone(),
        });

        if let Some(next_ep) = next_ep {
            ep = next_ep;
        }
    }
    pending
}

/// Beam search on the shared graph, writing results into scratch.out (sorted by ascending distance).
pub(super) fn par_beam_search(
    graph: &SharedGraph,
    query: &[f32],
    entry: u32,
    ef: usize,
    level: usize,
    scratch: &mut BuildScratch,
) {
    let entry_dist = graph.distance(query, graph.vec_slice(entry));
    let count = graph.node_count();

    scratch.visited.reset_for_len(count);
    scratch.candidates.clear();
    scratch.results.clear();
    scratch.candidates.reserve(ef + 8);
    scratch.results.reserve(ef + 8);

    scratch.candidates.push(MinDistNode(entry_dist, entry));
    scratch.results.push(DistNode(entry_dist, entry));
    scratch.visited.insert(entry);

    let mut nb_buf: Vec<u32> = Vec::with_capacity(graph.l0_m);

    while let Some(MinDistNode(dist, node)) = scratch.candidates.pop() {
        if let Some(DistNode(worst, _)) = scratch.results.peek() {
            if dist > *worst && scratch.results.len() >= ef {
                break;
            }
        }

        graph.read_neighbors(node, level, &mut nb_buf);

        // Phase 1: filter unvisited neighbors and issue prefetches.
        scratch.unvisited.clear();
        for &nb in &nb_buf {
            if nb != u32::MAX && scratch.visited.insert_if_absent(nb) {
                scratch.unvisited.push(nb);
                prefetch_read(graph.vec_ptr(nb));
            }
        }

        // Phase 2: compute distances (data should now be in cache).
        for &nb in &scratch.unvisited {
            let nb_dist = graph.distance(query, graph.vec_slice(nb));

            let worst = scratch.results.peek().map(|d| d.0).unwrap_or(f32::INFINITY);
            if nb_dist < worst || scratch.results.len() < ef {
                scratch.candidates.push(MinDistNode(nb_dist, nb));
                scratch.results.push(DistNode(nb_dist, nb));
                if scratch.results.len() > ef {
                    scratch.results.pop();
                }
            }
        }
    }

    scratch.out.clear();
    scratch.out.reserve(scratch.results.len());
    while let Some(d) = scratch.results.pop() {
        scratch.out.push((d.0, d.1));
    }
    scratch
        .out
        .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
}

/// Add a backlink from `node` to `new_node` at `level`, pruning if full. Holds per-node lock.
pub(super) fn par_add_backlink(
    graph: &SharedGraph,
    node: u32,
    level: usize,
    new_node: u32,
    max_m: usize,
    scratch: &mut BuildScratch,
) {
    if level == 0 {
        let _lock = graph.l0_locks[node as usize].lock();
        let limits = PruneLimits {
            score_len: graph.l0_m.min(max_m),
            select_m: max_m,
        };
        link_or_prune(
            graph,
            graph.l0_links(node),
            node,
            new_node,
            &limits,
            scratch,
        );
        return;
    }
    let Some(slot) = graph.upper_slot(node, level) else {
        return;
    };
    let _lock = slot.level.locks[node as usize].lock();
    let limits = PruneLimits {
        score_len: graph.m,
        select_m: graph.m,
    };
    link_or_prune(graph, slot.links, node, new_node, &limits, scratch);
}

/// Adds `new_node` to `node`'s locked slot, re-running the heuristic when the slot is full.
fn link_or_prune(
    graph: &SharedGraph,
    links: &[AtomicU32],
    node: u32,
    new_node: u32,
    limits: &PruneLimits,
    scratch: &mut BuildScratch,
) {
    // Check if already linked.
    for link in links {
        if link.load(Ordering::Relaxed) == new_node {
            return;
        }
    }

    // Try to find an empty slot.
    for link in links {
        if link.load(Ordering::Relaxed) == u32::MAX {
            link.store(new_node, Ordering::Relaxed);
            return;
        }
    }

    // Slot is full: prune using heuristic.
    let node_vec = graph.vec_slice(node);
    scratch.backlink_scored.clear();
    scratch.backlink_scored.reserve(limits.select_m + 1);

    // Prefetch neighbor vectors before scoring.
    let scored_links = &links[..limits.score_len];
    for link in scored_links {
        let nb = link.load(Ordering::Relaxed);
        if nb != u32::MAX {
            prefetch_read(graph.vec_ptr(nb));
        }
    }
    for link in scored_links {
        let nb = link.load(Ordering::Relaxed);
        if nb == u32::MAX {
            continue;
        }
        let d = graph.heuristic_distance(node_vec, graph.vec_slice(nb));
        scratch.backlink_scored.push((d, nb));
    }
    let d_new = graph.heuristic_distance(node_vec, graph.vec_slice(new_node));

    let buf = BacklinkBuffers {
        scored: &mut scratch.backlink_scored,
        selected: &mut scratch.backlink_selected,
        selected_is_new: &mut scratch.backlink_selected_is_new,
    };
    select_backlinks_into(&graph.view(), (d_new, new_node), limits.select_m, buf);

    // Write back.
    store_links(links, &scratch.backlink_selected);
}

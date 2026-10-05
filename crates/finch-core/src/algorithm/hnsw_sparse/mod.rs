//! HNSW sparse vector index.
//!
//! This is the sparse sibling of `algorithm::hnsw`:
//! - metric: InnerProduct over sparse vectors (distance = -dot)
//! - levels + L0/upper neighbor graph persistence
//! - load/search paths support mmap-backed segment bytes (neighbors) while
//!   computing dot products directly over encoded bytes (mmap-friendly)

use std::cell::RefCell;
use std::collections::BinaryHeap;
use std::io::Cursor;

use rand::Rng;

use finch_types::{HnswIndexParams, QuantizeType, Status, ZResult};

use super::codec::{u32s_to_le, UpperRun};
use super::flat::{segment_array_u32, SegmentArray, SegmentBytes, StorageReader, StorageWriter};
use super::flat_sparse::{dump_sparse_rows, SparseRows, SparseSegments, SparseVector};
use super::hnsw::{l0_links, reserve_upper_slots, DistNode, MinDistNode, UpperLevelNeighbors};
use super::{doc_filter_allows, DocFilter, TopkHeap};

const SEG_KEYS: &str = "HNSW_SPARSE_KEYS";
const SEG_VECTORS: &str = "HNSW_SPARSE_VECTORS";
const SEG_OFFSETS: &str = "HNSW_SPARSE_OFFSETS";
const SEG_LEVELS: &str = "HNSW_SPARSE_LEVELS";
const SEG_L0_NEIGHBORS: &str = "HNSW_SPARSE_L0";
const SEG_UPPER_NEIGHBORS: &str = "HNSW_SPARSE_UPPER";
const SEG_META: &str = "HNSW_SPARSE_META";

const SEGMENTS: SparseSegments = SparseSegments {
    label: "HNSW_SPARSE",
    keys: SEG_KEYS,
    vectors: SEG_VECTORS,
    offsets: SEG_OFFSETS,
};

/// Negative sparse dot product (so smaller = more similar).
#[inline]
fn sparse_neg_ip(a: &SparseVector, b: &SparseVector) -> f32 {
    -a.dot(b)
}

/// Like `sparse_neg_ip` against an encoded vector; malformed or empty input scores 0.
#[inline]
fn sparse_neg_ip_encoded(query: &SparseVector, encoded: &[u8], quantize: QuantizeType) -> f32 {
    query
        .dot_encoded_checked(encoded, quantize)
        .map_or(0.0, |dot| -dot)
}

struct VisitedList {
    marks: Vec<u32>,
    tag: u32,
}

impl VisitedList {
    fn new() -> Self {
        Self {
            marks: Vec::new(),
            tag: 1,
        }
    }

    fn reset_for_len(&mut self, len: usize) {
        if self.marks.len() < len {
            self.marks.resize(len, 0);
        }
        self.tag = self.tag.wrapping_add(1);
        if self.tag == 0 {
            self.marks.fill(0);
            self.tag = 1;
        }
    }

    #[inline]
    fn contains(&self, node: u32) -> bool {
        self.marks.get(node as usize).copied().unwrap_or(0) == self.tag
    }

    #[inline]
    fn insert(&mut self, node: u32) {
        if let Some(slot) = self.marks.get_mut(node as usize) {
            *slot = self.tag;
        }
    }
}

thread_local! {
    static VISITED_LIST: RefCell<VisitedList> = RefCell::new(VisitedList::new());
    static HNSW_SPARSE_SEARCH_SCRATCH: RefCell<SearchScratch> = RefCell::new(SearchScratch::new());
}

impl Default for VisitedList {
    fn default() -> Self {
        Self::new()
    }
}

struct SearchScratch {
    candidates: BinaryHeap<MinDistNode>,
    results: BinaryHeap<DistNode>,
    out: Vec<(f32, u32)>,
}

impl SearchScratch {
    fn new() -> Self {
        Self {
            candidates: BinaryHeap::new(),
            results: BinaryHeap::new(),
            out: Vec::new(),
        }
    }

    fn beam<'a>(&'a mut self, visited: &'a mut VisitedList) -> Beam<'a> {
        Beam {
            visited,
            candidates: &mut self.candidates,
            results: &mut self.results,
            out: &mut self.out,
        }
    }
}

#[derive(Default)]
struct BuilderScratch {
    visited: VisitedList,
    candidates: BinaryHeap<MinDistNode>,
    results: BinaryHeap<DistNode>,
    out: Vec<(f32, u32)>,
    selected: Vec<u32>,
    backlink_scored: Vec<(f32, u32)>,
    backlink_selected: Vec<u32>,
}

impl BuilderScratch {
    fn beam(&mut self) -> Beam<'_> {
        Beam {
            visited: &mut self.visited,
            candidates: &mut self.candidates,
            results: &mut self.results,
            out: &mut self.out,
        }
    }
}

/// Heaps and output buffer of one sparse beam search.
struct Beam<'a> {
    visited: &'a mut VisitedList,
    candidates: &'a mut BinaryHeap<MinDistNode>,
    results: &'a mut BinaryHeap<DistNode>,
    out: &'a mut Vec<(f32, u32)>,
}

/// Best-first search from `entry`; leaves `beam.out` sorted nearest first.
fn beam_search<'g, D, N>(
    beam: Beam<'_>,
    entry: u32,
    ef: usize,
    count: usize,
    mut distance: D,
    neighbors: N,
) where
    D: FnMut(u32) -> f32,
    N: Fn(u32) -> &'g [u32],
{
    let entry_dist = distance(entry);
    beam.visited.reset_for_len(count);
    beam.candidates.clear();
    beam.results.clear();
    beam.out.clear();
    beam.candidates.reserve(ef + 8);
    beam.results.reserve(ef + 8);
    beam.out.reserve(ef);

    beam.candidates.push(MinDistNode(entry_dist, entry));
    beam.results.push(DistNode(entry_dist, entry));
    beam.visited.insert(entry);

    while let Some(MinDistNode(dist, node)) = beam.candidates.pop() {
        if let Some(DistNode(worst, _)) = beam.results.peek() {
            if dist > *worst && beam.results.len() >= ef {
                break;
            }
        }

        for &nb in neighbors(node) {
            if nb == u32::MAX || beam.visited.contains(nb) {
                continue;
            }
            beam.visited.insert(nb);
            let d = distance(nb);
            if let Some(DistNode(worst, _)) = beam.results.peek() {
                if d > *worst && beam.results.len() >= ef {
                    continue;
                }
            }
            beam.candidates.push(MinDistNode(d, nb));
            beam.results.push(DistNode(d, nb));
            if beam.results.len() > ef {
                beam.results.pop();
            }
        }
    }

    while let Some(DistNode(d, n)) = beam.results.pop() {
        beam.out.push((d, n));
    }
    beam.out
        .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
}

/// Greedy walk from `entry` through `levels` (highest first); returns the closest node reached.
fn greedy_descend<'g, D, N>(
    entry: u32,
    levels: impl Iterator<Item = usize>,
    mut distance: D,
    neighbors: N,
) -> u32
where
    D: FnMut(u32) -> f32,
    N: Fn(u32, usize) -> &'g [u32],
{
    let mut ep = entry;
    let mut ep_dist = distance(entry);
    for lv in levels {
        let mut changed = true;
        while changed {
            changed = false;
            for &nb in neighbors(ep, lv) {
                if nb == u32::MAX {
                    continue;
                }
                let d = distance(nb);
                if d < ep_dist {
                    ep_dist = d;
                    ep = nb;
                    changed = true;
                }
            }
        }
    }
    ep
}

pub struct HnswSparseBuilder {
    params: HnswIndexParams,
    keys: Vec<u64>,
    vectors: Vec<SparseVector>,
    levels: Vec<u32>,
    l0_neighbors: Vec<u32>,                    // flat: count × (2*m)
    upper_neighbors: Vec<UpperLevelNeighbors>, // levels 1..=max_level (index = level-1)
    entry_point: u32,
    max_level: usize,
    scratch: BuilderScratch,
}

impl HnswSparseBuilder {
    pub fn new(params: HnswIndexParams) -> Self {
        Self {
            params,
            keys: Vec::new(),
            vectors: Vec::new(),
            levels: Vec::new(),
            l0_neighbors: Vec::new(),
            upper_neighbors: Vec::new(),
            entry_point: u32::MAX,
            max_level: 0,
            scratch: BuilderScratch::default(),
        }
    }

    #[inline]
    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let mut level = 0;
        // At scaling_factor <= e, 1/ln is >= 1: every draw promotes and every node reaches level 32.
        let scale = (1.0 / (self.params.scaling_factor.max(2) as f64).ln()).min(0.5);
        while rng.gen::<f64>() < scale && level < 32 {
            level += 1;
        }
        level
    }

    #[inline]
    fn get_neighbors(&self, node: u32, level: usize) -> &[u32] {
        if level == 0 {
            return l0_links(&self.l0_neighbors, self.params.m, node);
        }
        match self.upper_neighbors.get(level - 1) {
            Some(ul) => ul.slot(node, self.params.m),
            None => &[],
        }
    }

    #[inline]
    fn neighbors_mut(&mut self, node: u32, level: usize) -> Option<&mut [u32]> {
        if level == 0 {
            let l0_m = self.params.m * 2;
            let start = node as usize * l0_m;
            let end = start + l0_m;
            return Some(&mut self.l0_neighbors[start..end]);
        }
        let m = self.params.m;
        let ul = self.upper_neighbors.get_mut(level - 1)?;
        let range = ul.slot_range(node, m)?;
        Some(&mut ul.neighbors[range])
    }

    fn beam_search_into(
        &self,
        query: &SparseVector,
        entry: u32,
        ef: usize,
        level: usize,
        scratch: &mut BuilderScratch,
    ) {
        beam_search(
            scratch.beam(),
            entry,
            ef,
            self.vectors.len(),
            |nb| sparse_neg_ip(query, &self.vectors[nb as usize]),
            |node| self.get_neighbors(node, level),
        );
    }

    fn select_neighbors_heuristic(
        &self,
        candidates: &[(f32, u32)],
        max_neighbor_cnt: usize,
        prune_cnt: usize,
        out: &mut Vec<u32>,
    ) {
        out.clear();
        out.reserve(max_neighbor_cnt);

        // when the candidate set is small enough, no diversity
        // pruning is applied; the neighbor list becomes the nearest-N list.
        //
        // Algorithm: update_neighbors:
        // - if candidates <= prune_cnt AND <= max_neighbor_cnt: keep all.
        if candidates.len() <= prune_cnt && candidates.len() <= max_neighbor_cnt {
            for &(_dist_to_query, cand) in candidates.iter() {
                if cand == u32::MAX {
                    continue;
                }
                out.push(cand);
            }
            return;
        }

        for &(dist_to_query, cand) in candidates.iter() {
            if cand == u32::MAX {
                continue;
            }
            let cand_vec = &self.vectors[cand as usize];
            let mut ok = true;
            for &s in out.iter() {
                let s_vec = &self.vectors[s as usize];
                let d = sparse_neg_ip(cand_vec, s_vec);
                // reject ties (<=), not just strict-improvement.
                if d <= dist_to_query {
                    ok = false;
                    break;
                }
            }
            if ok {
                out.push(cand);
                if out.len() >= max_neighbor_cnt {
                    break;
                }
            }
        }
    }

    fn set_neighbors(&mut self, node: u32, level: usize, neighbors: &[u32]) {
        let Some(slot) = self.neighbors_mut(node, level) else {
            return;
        };
        slot.fill(u32::MAX);
        for (i, &nb) in neighbors.iter().enumerate().take(slot.len()) {
            slot[i] = nb;
        }
    }

    fn add_backlink(
        &mut self,
        node: u32,
        level: usize,
        new_node: u32,
        max_m: usize,
        scratch: &mut BuilderScratch,
    ) {
        // backlink insertion only triggers pruning when the
        // neighbor list is already full. Otherwise, the new backlink is simply
        // appended (first empty slot).
        if self.get_neighbors(node, level).contains(&u32::MAX) {
            let Some(slot) = self.neighbors_mut(node, level) else {
                return;
            };
            if slot.contains(&new_node) {
                return;
            }
            if let Some(pos) = slot.iter().position(|&x| x == u32::MAX) {
                slot[pos] = new_node;
            }
            return;
        }

        scratch.backlink_scored.clear();

        let node_vec = &self.vectors[node as usize];
        for &nb in self.get_neighbors(node, level).iter() {
            if nb == u32::MAX || nb == new_node {
                continue;
            }
            let d = sparse_neg_ip(node_vec, &self.vectors[nb as usize]);
            scratch.backlink_scored.push((d, nb));
        }
        scratch.backlink_scored.push((
            sparse_neg_ip(node_vec, &self.vectors[new_node as usize]),
            new_node,
        ));
        scratch
            .backlink_scored
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        scratch.backlink_selected.clear();
        // reverse_update_neighbors always uses the diversity prune
        // step when the list is full (no prune_cnt bypass here).
        self.select_neighbors_heuristic(
            &scratch.backlink_scored,
            max_m,
            0,
            &mut scratch.backlink_selected,
        );
        self.set_neighbors(node, level, &scratch.backlink_selected);
    }

    pub fn add(&mut self, key: u64, vec: SparseVector) -> ZResult<()> {
        let vec = match self.params.quantize {
            QuantizeType::Undefined => vec,
            QuantizeType::Fp16 => vec.quantize_to_f16(),
            _ => {
                return Err(Status::invalid_argument(
                    "sparse quantize only supports Fp16",
                ))
            }
        };

        let level = self.random_level();
        let node_id = self.keys.len() as u32;

        self.keys.push(key);
        self.vectors.push(vec);
        self.levels.push(level as u32);

        let l0_m = self.params.m * 2;
        self.l0_neighbors
            .resize(self.l0_neighbors.len() + l0_m, u32::MAX);
        reserve_upper_slots(&mut self.upper_neighbors, node_id, level, self.params.m);

        let entry = self.entry_point;
        if entry == u32::MAX {
            self.entry_point = node_id;
            self.max_level = level;
            return Ok(());
        }

        let current_max_level = self.max_level;
        let mut scratch = std::mem::take(&mut self.scratch);

        // Phase 1: greedy descent from max_level to level+1.
        let query = &self.vectors[node_id as usize];
        let ep = greedy_descend(
            entry,
            (level + 1..=current_max_level).rev(),
            |nb| sparse_neg_ip(query, &self.vectors[nb as usize]),
            |node, lv| self.get_neighbors(node, lv),
        );

        self.connect_levels(node_id, ep, level.min(current_max_level), &mut scratch);

        if level > current_max_level {
            self.entry_point = node_id;
            self.max_level = level;
        }

        self.scratch = scratch;
        Ok(())
    }

    /// Phase 2: beam search + connect from `top_level` down to 0.
    fn connect_levels(
        &mut self,
        node_id: u32,
        mut ep: u32,
        top_level: usize,
        scratch: &mut BuilderScratch,
    ) {
        let m = self.params.m;
        let ef = self.params.ef_construction.max(m * 2);
        for lv in (0..=top_level).rev() {
            self.beam_search_into(&self.vectors[node_id as usize], ep, ef, lv, scratch);
            let next_ep = scratch.out.first().map(|(_, id)| *id);

            scratch.selected.clear();
            self.select_neighbors_heuristic(
                &scratch.out,
                if lv == 0 { m * 2 } else { m },
                m,
                &mut scratch.selected,
            );

            self.set_neighbors(node_id, lv, &scratch.selected);

            for i in 0..scratch.selected.len() {
                let nb = scratch.selected[i];
                if nb == u32::MAX {
                    continue;
                }
                let max_m = if lv == 0 { m * 2 } else { m };
                self.add_backlink(nb, lv, node_id, max_m, scratch);
            }

            if let Some(nep) = next_ep {
                ep = nep;
            }
        }
    }

    pub fn dump(&self, storage: &mut dyn StorageWriter) -> ZResult<()> {
        let n = self.keys.len();
        dump_sparse_rows(
            storage,
            &SEGMENTS,
            &self.keys,
            &self.vectors,
            self.params.quantize,
        )?;
        storage.write_segment(SEG_LEVELS, &u32s_to_le(&self.levels))?;
        storage.write_segment(SEG_L0_NEIGHBORS, &u32s_to_le(&self.l0_neighbors))?;
        storage.write_segment(SEG_UPPER_NEIGHBORS, &self.encode_upper())?;

        let mut meta = Vec::new();
        meta.extend_from_slice(&(n as u64).to_le_bytes());
        meta.extend_from_slice(&self.entry_point.to_le_bytes());
        meta.extend_from_slice(&(self.max_level as u32).to_le_bytes());
        meta.extend_from_slice(&(self.params.m as u32).to_le_bytes());
        meta.extend_from_slice(&(self.params.quantize as u32).to_le_bytes());
        meta.extend_from_slice(&(self.params.ef_construction as u32).to_le_bytes());
        meta.extend_from_slice(&(self.params.scaling_factor as u32).to_le_bytes());
        storage.write_segment(SEG_META, &meta)?;

        Ok(())
    }

    /// Upper levels in the `[num_levels]([node_count]([count][ids...])*)*` layout.
    fn encode_upper(&self) -> Vec<u8> {
        let m = self.params.m;
        let mut upper_buf: Vec<u8> = Vec::new();
        upper_buf.extend_from_slice(&(self.max_level as u32).to_le_bytes());
        for level in &self.upper_neighbors {
            upper_buf.extend_from_slice(&(level.offsets.len() as u32).to_le_bytes());
            for &off in &level.offsets {
                if off == u32::MAX {
                    upper_buf.extend_from_slice(&0u32.to_le_bytes());
                    continue;
                }
                upper_buf.extend_from_slice(&(m as u32).to_le_bytes());
                let off = off as usize;
                for &nb in level.neighbors[off..off + m].iter() {
                    upper_buf.extend_from_slice(&nb.to_le_bytes());
                }
            }
        }
        upper_buf
    }
}

struct UpperLevelIndex {
    node_offsets: Vec<UpperRun>,
}

mod searcher;

pub use searcher::HnswSparseSearcher;

#[cfg(test)]
mod tests;

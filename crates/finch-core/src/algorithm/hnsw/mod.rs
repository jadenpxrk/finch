//! HNSW (Hierarchical Navigable Small World) dense vector index

use super::flat::{
    segment_array_f32, segment_array_u32, segment_array_u64, SegmentArray, SegmentBytes,
    StorageReader, StorageWriter,
};
use super::{check_vector_dim, doc_filter_allows, DocFilter, TopkHeap};
use crate::metric::normalize_l2;
use crate::quantizer::{
    bytes_per_vector, distance_to_quantized_with_query_sq_norm, quantize_append,
    quantize_type_from_u32,
};
use crate::simd;
use finch_types::{HnswBuildTuning, HnswIndexParams, MetricType, QuantizeType, Status, ZResult};
use parking_lot::Mutex;
use rand::Rng;
use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::BinaryHeap;
use std::io::Cursor;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};

mod build_parallel;
mod builder;
mod parallel;
mod persistence;
mod searcher;
mod select;

pub use builder::HnswBuilder;
use parallel::*;
use searcher::*;
pub use searcher::{HnswSearchParams, HnswSearcher};
use select::*;

// Segment names
const SEG_KEYS: &str = "HNSW_KEYS";
const SEG_VECTORS: &str = "HNSW_VECTORS";
const SEG_L0_NEIGHBORS: &str = "HNSW_L0_NEIGHBORS";
const SEG_UPPER_NEIGHBORS: &str = "HNSW_UPPER_NEIGHBORS";
const SEG_UPPER_INDEX: &str = "HNSW_UPPER_INDEX";
const SEG_META: &str = "HNSW_META";
const SEG_LEVELS: &str = "HNSW_LEVELS";
const SEG_HEADER: &str = "HNSW_HEADER";

const HEADER_MAGIC: [u8; 8] = *b"FINCHHNS";
const HEADER_VERSION: u32 = 1;

#[inline]
fn random_level_with_scaling<R: Rng + ?Sized>(rng: &mut R, scaling_factor: usize) -> usize {
    // Match Qdrant's HNSW level distribution:
    // level = round(-ln(U) / ln(M)).
    //
    // This keeps the first upper layer at about 1/sqrt(M) of points while
    // thinning higher layers much more aggressively than a repeated Bernoulli
    // sampler. With M=16 that means P(level >= 2) is ~1/64 rather than 1/16,
    // which avoids overpopulated upper routing layers.
    let level_factor = 1.0 / (scaling_factor.max(2) as f64).ln();
    let sample = rng.gen::<f64>().max(f64::MIN_POSITIVE);
    ((-sample.ln() * level_factor).round() as usize).min(32)
}

/// Dimension prefix for neighbor-selection distances; keeps high-dim builds cheap, search exact.
fn heuristic_dim(tuning: &HnswBuildTuning, dim: usize) -> usize {
    tuning.heuristic_dim.unwrap_or(dim).min(dim)
}

#[inline]
fn should_prune_neighbor(prune_alpha: f32, dist_between: f32, dist_to_query: f32) -> bool {
    dist_between * prune_alpha < dist_to_query
}

const L0_REPAIR_PASSES: usize = 3;
const L0_REPAIR_CANDIDATE_CAP: usize = 1024;
const L0_REPAIR_SEARCH_PASSES: usize = 1;

#[inline]
fn hnsw_distance(metric_type: MetricType, a: &[f32], b: &[f32]) -> f32 {
    match metric_type {
        MetricType::InnerProduct => -simd::ip_f32(a, b),
        MetricType::Cosine => 1.0 - simd::ip_f32(a, b),
        MetricType::MipsL2 => simd::mips_l2_f32(a, b),
        MetricType::L2 | MetricType::Undefined | MetricType::Hamming => simd::l2_f32(a, b),
    }
}

#[inline]
fn vec_slice_raw(vectors: &[f32], vec_stride_floats: usize, dim: usize, node_id: u32) -> &[f32] {
    let start = (node_id as usize) * vec_stride_floats;
    &vectors[start..start + dim]
}

#[inline]
fn hnsw_heuristic_distance(
    metric_type: MetricType,
    heuristic_dim: usize,
    dim: usize,
    a: &[f32],
    b: &[f32],
) -> f32 {
    if heuristic_dim >= dim {
        hnsw_distance(metric_type, a, b)
    } else {
        hnsw_distance(metric_type, &a[..heuristic_dim], &b[..heuristic_dim])
    }
}

/// Current greedy-descent position and its distance to the query.
#[derive(Clone, Copy)]
struct Entry {
    node: u32,
    dist: f32,
}

/// Upper-level (level >= 1) neighbor slots, aligned by node id.
pub(super) struct UpperLevelNeighbors {
    // offsets[node_id] = start index into `neighbors` (in u32 units), or u32::MAX when node has no slot at this level.
    pub(super) offsets: Vec<u32>,
    // Concatenated fixed-size chunks (m entries per present node).
    pub(super) neighbors: Vec<u32>,
}

impl UpperLevelNeighbors {
    #[inline]
    pub(super) fn slot_range(&self, node: u32, m: usize) -> Option<std::ops::Range<usize>> {
        let off = self.offsets.get(node as usize).copied().unwrap_or(u32::MAX);
        if off == u32::MAX {
            return None;
        }
        let off = off as usize;
        Some(off..off + m)
    }

    #[inline]
    pub(super) fn slot(&self, node: u32, m: usize) -> &[u32] {
        match self.slot_range(node, m) {
            Some(range) => &self.neighbors[range],
            None => &[],
        }
    }
}

/// Adds `node_id` to every upper level and gives it an empty `m`-slot on levels `1..=level`.
pub(super) fn reserve_upper_slots(
    upper: &mut Vec<UpperLevelNeighbors>,
    node_id: u32,
    level: usize,
    m: usize,
) {
    while upper.len() < level {
        upper.push(UpperLevelNeighbors {
            offsets: vec![u32::MAX; node_id as usize],
            neighbors: Vec::new(),
        });
    }
    for ul in upper.iter_mut() {
        ul.offsets.push(u32::MAX);
    }
    for ul in upper.iter_mut().take(level) {
        let off = ul.neighbors.len() as u32;
        let new_len = ul.neighbors.len() + m;
        ul.neighbors.resize(new_len, u32::MAX);
        ul.offsets[node_id as usize] = off;
    }
}

/// A node's `2 * m` level-0 links in the flat L0 array.
#[inline]
pub(super) fn l0_links(l0: &[u32], m: usize, node: u32) -> &[u32] {
    let l0_m = m * 2;
    let start = node as usize * l0_m;
    &l0[start..start + l0_m]
}

/// Clears `slot` and copies in up to `limit` links.
#[inline]
fn write_links(slot: &mut [u32], links: &[u32], limit: usize) {
    slot.fill(u32::MAX);
    for (i, &nb) in links.iter().enumerate().take(slot.len().min(limit)) {
        slot[i] = nb;
    }
}

/// Ordered pair for BinaryHeap (max-heap by distance)
#[derive(PartialEq)]
pub(super) struct DistNode(pub(super) f32, pub(super) u32);
impl Eq for DistNode {}
impl PartialOrd for DistNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for DistNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Min-heap variant (smallest distance first)
#[derive(PartialEq)]
pub(super) struct MinDistNode(pub(super) f32, pub(super) u32);
impl Eq for MinDistNode {}
impl PartialOrd for MinDistNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MinDistNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .0
            .partial_cmp(&self.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Software prefetch for read access. Fires a non-blocking cache-line load.
#[inline(always)]
fn prefetch_read(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: a prefetch is a hint and never faults, whatever the address.
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(ptr as *const i8);
    }
    #[cfg(target_arch = "aarch64")]
    // SAFETY: PRFM PLDL1KEEP is an L1 read hint and never faults, whatever the address.
    unsafe {
        std::arch::asm!("prfm pldl1keep, [{ptr}]", ptr = in(reg) ptr, options(nostack, preserves_flags));
    }
}

/// Generation-counter visited list.
///
/// Instead of zeroing the entire bitset on every search (~125KB for 1M vectors),
/// we store a generation stamp per node and bump the generation each search.
/// `contains` / `insert` are O(1) with no per-search clearing cost.
struct VisitedList {
    stamps: Vec<u32>,
    generation: u32,
}

impl VisitedList {
    fn new() -> Self {
        Self {
            stamps: Vec::new(),
            generation: 0,
        }
    }

    fn reset_for_len(&mut self, len: usize) {
        self.stamps.resize(len, 0);
        self.generation = self.generation.wrapping_add(1);
        // On the rare wrap-around (every ~4 billion searches), clear all stamps.
        if self.generation == 0 {
            self.stamps.fill(0);
            self.generation = 1;
        }
    }

    #[inline]
    fn insert(&mut self, node: u32) {
        if let Some(s) = self.stamps.get_mut(node as usize) {
            *s = self.generation;
        }
    }

    #[inline]
    fn insert_if_absent(&mut self, node: u32) -> bool {
        let Some(s) = self.stamps.get_mut(node as usize) else {
            return false;
        };
        if *s == self.generation {
            return false;
        }
        *s = self.generation;
        true
    }
}

thread_local! {
    static VISITED_LIST: RefCell<VisitedList> = RefCell::new(VisitedList::new());
    static HNSW_SEARCH_SCRATCH: RefCell<SearchScratch> = RefCell::new(SearchScratch::new());
    static HNSW_QUERY_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

struct SearchScratch {
    candidates: BinaryHeap<MinDistNode>,
    results: BinaryHeap<DistNode>,
    out: Vec<(f32, u32)>,
    unvisited: Vec<u32>,
    /// Reusable buffer for scatter-batch vector pointers.
    ptr_buf: Vec<*const f32>,
    /// Reusable buffer for scatter-batch distance results.
    dist_buf: Vec<f32>,
    /// Reusable top-k heap for final filtered result collection.
    topk_heap: TopkHeap,
    /// Entry candidates for optional multi-seed level-0 search.
    entry_seeds: Vec<u32>,
}

impl SearchScratch {
    fn new() -> Self {
        Self {
            candidates: BinaryHeap::new(),
            results: BinaryHeap::new(),
            out: Vec::new(),
            unvisited: Vec::new(),
            ptr_buf: Vec::new(),
            dist_buf: Vec::new(),
            topk_heap: TopkHeap::new(0),
            entry_seeds: Vec::new(),
        }
    }

    #[inline]
    fn reset(&mut self, _ef: usize) {
        self.candidates.clear();
        self.results.clear();
        self.out.clear();
    }
}

#[cfg(test)]
mod tests;

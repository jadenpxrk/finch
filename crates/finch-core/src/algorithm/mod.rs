pub mod cluster;
mod codec;
pub mod flat;
pub mod flat_sparse;
pub mod hnsw;
pub mod hnsw_sparse;
pub mod ivf;

use finch_types::ZResult;

/// Document filter trait for vector search
pub trait DocFilter: Send + Sync {
    fn is_valid(&self, doc_id: u64) -> ZResult<bool>;
}

/// Allow-all filter (no filtering)
pub struct AllowAllFilter;
impl DocFilter for AllowAllFilter {
    fn is_valid(&self, _doc_id: u64) -> ZResult<bool> {
        Ok(true)
    }
}

pub(crate) fn check_query_dim(query_len: usize, dim: usize) -> ZResult<()> {
    if query_len != dim {
        return Err(finch_types::Status::invalid_argument(format!(
            "query dim {} != index dim {}",
            query_len, dim
        )));
    }
    Ok(())
}

pub(crate) fn check_vector_dim(len: usize, dim: usize) -> ZResult<()> {
    if len != dim {
        return Err(finch_types::Status::invalid_argument(format!(
            "dim mismatch: expected {}, got {}",
            dim, len
        )));
    }
    Ok(())
}

pub(crate) fn doc_filter_allows(filter: Option<&dyn DocFilter>, doc_id: u64) -> ZResult<bool> {
    match filter {
        Some(filter) => filter.is_valid(doc_id),
        None => Ok(true),
    }
}

/// Top-K result accumulator using a bounded max-heap
pub struct TopkHeap {
    pub k: usize,
    heap: std::collections::BinaryHeap<OrderedPair>,
}

#[derive(PartialEq)]
struct OrderedPair(f32, u64);

impl Eq for OrderedPair {}

impl PartialOrd for OrderedPair {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedPair {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl TopkHeap {
    pub fn new(k: usize) -> Self {
        TopkHeap {
            k,
            heap: std::collections::BinaryHeap::with_capacity(k + 1),
        }
    }

    pub fn reset(&mut self, k: usize) {
        self.k = k;
        self.heap.clear();
        let needed = k + 1;
        if self.heap.capacity() < needed {
            self.heap.reserve(needed - self.heap.capacity());
        }
    }

    pub fn push(&mut self, dist: f32, key: u64) {
        if self.heap.len() < self.k {
            self.heap.push(OrderedPair(dist, key));
        } else if let Some(mut top) = self.heap.peek_mut() {
            if dist < top.0 {
                *top = OrderedPair(dist, key);
                // PeekMut::drop sifts down automatically: one sift instead of two
            }
        }
    }

    pub fn top_dist(&self) -> f32 {
        self.heap.peek().map(|p| p.0).unwrap_or(f32::INFINITY)
    }

    /// Drain into sorted (ascending distance) result list
    pub fn into_sorted(self) -> Vec<(u64, f32)> {
        let mut v: Vec<(u64, f32)> = self.heap.into_iter().map(|p| (p.1, p.0)).collect();
        v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        v
    }

    pub fn drain_sorted(&mut self) -> Vec<(u64, f32)> {
        let mut v: Vec<(u64, f32)> = self.heap.drain().map(|p| (p.1, p.0)).collect();
        v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        v
    }
}

//! Vector index algorithms (flat, IVF, and HNSW, with sparse flat and sparse HNSW), distance
//! metrics, quantizers, and SIMD kernels. Indexes read and write their files through the
//! `StorageReader` and `StorageWriter` traits. Used by `finch-db`.

#![deny(missing_docs)]

/// Index builders and searchers, their storage traits, and the top-k heap.
pub mod algorithm;
pub mod metric;
/// Fusion of ranked result lists from several vector fields.
pub mod mixed_reducer;
pub mod quantizer;
pub mod simd;

pub use algorithm::flat::{
    FlatBinary32Builder, FlatBinary32Searcher, FlatBinary64Builder, FlatBinary64Searcher,
    FlatBuilder, FlatSearcher, MemoryStorage, StorageReader, StorageWriter,
};
pub use algorithm::flat_sparse::{FlatSparseBuilder, FlatSparseSearcher, SparseVector};
pub use algorithm::hnsw::{HnswBuilder, HnswSearchParams, HnswSearcher};
pub use algorithm::hnsw_sparse::{HnswSparseBuilder, HnswSparseSearcher};
pub use algorithm::ivf::{IvfBuilder, IvfSearcher};
pub use algorithm::{DocFilter, TopkHeap};
pub use metric::{make_metric, Metric};

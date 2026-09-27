//! Vector index algorithms (flat, IVF, and HNSW, with sparse flat and sparse HNSW), distance
//! metrics, quantizers, and SIMD kernels. Indexes read and write their files through the
//! `StorageReader` and `StorageWriter` traits. Used by `finch-db`.

pub mod algorithm;
pub mod metric;
pub mod mixed_reducer;
pub mod quantizer;
pub mod simd;

pub use algorithm::cluster::KmeansCluster;
pub use algorithm::flat::{
    FlatBinary32Builder, FlatBinary32Searcher, FlatBinary64Builder, FlatBinary64Searcher,
    FlatBuilder, FlatSearcher, MemoryStorage, StorageReader, StorageWriter,
};
pub use algorithm::flat_sparse::{FlatSparseBuilder, FlatSparseSearcher, SparseVector};
pub use algorithm::hnsw::{HnswBuilder, HnswSearcher};
pub use algorithm::hnsw_sparse::{HnswSparseBuilder, HnswSparseSearcher};
pub use algorithm::ivf::{IvfBuilder, IvfSearcher};
pub use algorithm::{AllowAllFilter, DocFilter, TopkHeap};
pub use metric::{
    l2_norm, make_metric, normalize_l2, CosineMetric, HammingMetric, IpMetric, L2Metric, Metric,
    MipsL2Metric,
};
pub use quantizer::{
    make_converter, Converter, CosineConverter, HalfFloatConverter, HalfFloatReformer,
    Int4Dequantizer, Int4Quantizer, Int8Dequantizer, Int8Quantizer, MipsConverter, NullConverter,
    NullReformer, Reformer,
};

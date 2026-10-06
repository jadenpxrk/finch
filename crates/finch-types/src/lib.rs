//! Collection schemas, documents, query and index parameters, the process-wide config, and
//! the `Status` error type. Every other Finch crate depends on it.

#![deny(missing_docs)]

/// Process-wide settings: memory limit, threads, log, and WAL.
pub mod config;
/// Documents and field values.
pub mod doc;
/// Index build parameters.
pub mod index_params;
/// Vector queries, collection options, and statistics.
pub mod query;
/// Field and collection schemas and their validation.
pub mod schema;
/// The `Status` error type and its codes.
pub mod status;
pub mod system_columns;
/// Data types, metrics, quantization, operators, and storage modes.
pub mod types;

// Re-export most commonly used items

pub use config::GlobalConfigData;
pub use config::LogLevel;
pub use doc::{Doc, Value};
pub use index_params::{
    FlatIndexParams, HnswBuildTuning, HnswIndexParams, IndexParams, InvertIndexParams,
    IvfIndexParams,
};
pub use query::{
    AddColumnOptions, AlterColumnOptions, CollectionOptions, CollectionStats, CreateIndexOptions,
    GroupByVectorQuery, GroupResult, OptimizeOptions, QueryParams, VectorFieldStats, VectorQuery,
};
pub use schema::{
    CollectionSchema, FieldSchema, MAX_DOC_COUNT_PER_SEGMENT,
    MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD,
};
pub use status::{Status, StatusCode, ZResult};
pub use system_columns::{
    FINCH_IPC_DOC_ID, SYS_GLOBAL_DOC_ID, SYS_LOCAL_ROW_ID, SYS_SCORE, SYS_USER_ID,
};
pub use types::{CompareOp, DataType, FileFormat, MetricType, Operator, QuantizeType, StorageType};

//! Collection schemas, documents, query and index parameters, the process-wide config, and
//! the `Status` error type. Every other Finch crate depends on it.

pub mod agent;
pub mod config;
pub mod doc;
pub mod index_params;
pub mod query;
pub mod schema;
pub mod status;
pub mod system_columns;
pub mod types;

// Re-export most commonly used items
pub use agent::{
    AgentContext, AgentPermissions, AuditEntry, AuditOperation, AuditResult, ContextValue,
    DbPermissions, FsPermissions, NetworkPermissions, OperationBudget, SessionId, ToolCallRecord,
    ToolDefinition,
};
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
    FINCH_IPC_DOC_ID, FINCH_IPC_PK, SYS_GLOBAL_DOC_ID, SYS_INTERNAL_GROUP_ID,
    SYS_INTERNAL_IS_VALID, SYS_INTERNAL_SPARSE_INDICES, SYS_INTERNAL_SPARSE_VALUES,
    SYS_INTERNAL_VECTOR, SYS_LOCAL_ROW_ID, SYS_SCORE, SYS_USER_ID,
};
pub use types::{
    BlockType, ColumnOp, CompareOp, DataType, FileFormat, IndexType, MetricType, Operator,
    QuantizeType, RelationOp, StorageType,
};

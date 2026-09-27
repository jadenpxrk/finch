#![allow(dead_code, unused_imports)]

pub use finch_db::invert::InvertIndex;
pub use finch_db::Collection;
pub use finch_types::{
    AddColumnOptions, AlterColumnOptions, CollectionOptions, CollectionSchema, CreateIndexOptions,
    DataType, Doc, FieldSchema, FlatIndexParams, GroupByVectorQuery, HnswIndexParams, IndexParams,
    InvertIndexParams, IvfIndexParams, MetricType, OptimizeOptions, QuantizeType, QueryParams,
    StatusCode, Value, VectorQuery, FINCH_IPC_DOC_ID, MAX_DOC_COUNT_PER_SEGMENT,
    MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD, SYS_GLOBAL_DOC_ID, SYS_LOCAL_ROW_ID, SYS_SCORE,
    SYS_USER_ID,
};
pub use half::f16;
pub use std::sync::atomic::{AtomicU64, Ordering};
pub use std::sync::Arc;

pub fn temp_dir(name: &str) -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir =
        std::env::temp_dir().join(format!("finch_test_{}_{}_{}", name, std::process::id(), n));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

pub fn basic_schema(dim: usize) -> CollectionSchema {
    CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(dim))
        .with_field(FieldSchema::new("label", DataType::String))
}

pub fn make_doc(pk: &str, vec: Vec<f32>, label: &str) -> Doc {
    Doc::new(pk).set("emb", vec).set("label", label)
}

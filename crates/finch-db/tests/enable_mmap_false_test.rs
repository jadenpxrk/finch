use finch_db::Collection;
use finch_types::{
    CollectionOptions, CollectionSchema, CreateIndexOptions, DataType, Doc, FieldSchema,
    FlatIndexParams, IndexParams, MetricType, VectorQuery,
};
use std::sync::atomic::{AtomicU64, Ordering};

fn temp_dir(name: &str) -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir =
        std::env::temp_dir().join(format!("finch_test_{}_{}_{}", name, std::process::id(), n));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

#[test]
fn test_open_and_query_with_enable_mmap_false() {
    let path = temp_dir("enable_mmap_false");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("label", DataType::String));

    let opts = CollectionOptions {
        read_only: false,
        enable_mmap: false,
        index_storage: None,
        forward_storage: None,
        max_buffer_size: CollectionOptions::DEFAULT_MAX_BUFFER_SIZE,
        forward_file_format: None,
    };

    {
        let col = Collection::create_and_open(&path, schema, opts).unwrap();
        col.insert(vec![
            Doc::new("d0")
                .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
                .set("label", "d0"),
            Doc::new("d1")
                .set("emb", vec![0.0f32, 1.0, 0.0, 0.0])
                .set("label", "d1"),
        ])
        .unwrap();
        col.flush().unwrap();

        col.create_index(
            "emb",
            IndexParams::Flat(FlatIndexParams::new(MetricType::L2)),
            CreateIndexOptions::default(),
        )
        .unwrap();
    }

    // Reopen: options.enable_mmap is taken from the manifest (created with false).
    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    let res = col
        .query(VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 2))
        .unwrap();
    assert_eq!(res[0].pk, "d0");
}

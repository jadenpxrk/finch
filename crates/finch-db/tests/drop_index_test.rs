mod common;
use common::*;

#[test]
fn test_drop_index() {
    let path = temp_dir("drop_idx");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![
        Doc::new("d1").set("emb", vec![1.0f32, 0.0, 0.0, 0.0]),
        Doc::new("d2").set("emb", vec![0.0f32, 1.0, 0.0, 0.0]),
    ])
    .unwrap();

    col.create_index(
        "emb",
        IndexParams::Flat(FlatIndexParams {
            metric: MetricType::L2,
            quantize: Default::default(),
            column_major: false,
        }),
        CreateIndexOptions::default(),
    )
    .unwrap();

    col.drop_index("emb").unwrap();

    // Schema should have no index_params for emb
    let info = col.schema_info();
    assert!(
        info.get_field("emb")
            .map(|f| f.index_params.is_none())
            .unwrap_or(false),
        "index_params should be cleared after drop_index"
    );

    // Query should still work (brute-force fallback)
    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 2);
    let results = col.query(q).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].pk, "d1");

    // Dropping a non-indexed field is a no-op
    assert!(col.drop_index("emb").is_ok());

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_drop_invert_index_keeps_filter_correctness() {
    let path = temp_dir("drop_invert_idx");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(
            FieldSchema::new("id", DataType::Int64)
                .not_null()
                .with_index(IndexParams::Invert(InvertIndexParams::default())),
        );

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![
        Doc::new("d1")
            .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
            .set("id", 1i64),
        Doc::new("d2")
            .set("emb", vec![0.0f32, 1.0, 0.0, 0.0])
            .set("id", 2i64),
        Doc::new("d3")
            .set("emb", vec![0.0f32, 0.0, 1.0, 0.0])
            .set("id", 3i64),
    ])
    .unwrap();

    // Make sure docs are persisted so we exercise persisted-segment filter paths.
    col.flush().unwrap();

    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("id >= 2");
    let results = col.query(q).unwrap();
    let pks: std::collections::HashSet<_> = results.iter().map(|d| d.pk.as_str()).collect();
    assert_eq!(pks.len(), 2);
    assert!(pks.contains("d2"));
    assert!(pks.contains("d3"));

    // Drop invert index and ensure correctness is preserved (scan fallback).
    col.drop_index("id").unwrap();
    let info = col.schema_info();
    assert!(
        info.get_field("id")
            .map(|f| f.index_params.is_none())
            .unwrap_or(false),
        "invert index_params should be cleared after drop_index"
    );

    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("id >= 2");
    let results = col.query(q).unwrap();
    let pks: std::collections::HashSet<_> = results.iter().map(|d| d.pk.as_str()).collect();
    assert_eq!(pks.len(), 2);
    assert!(pks.contains("d2"));
    assert!(pks.contains("d3"));

    // Ensure on-disk invert dirs are removed.
    for ent in std::fs::read_dir(&path).unwrap() {
        let ent = ent.unwrap();
        let name = ent.file_name().to_string_lossy().to_string();
        if !name.starts_with("seg_") {
            continue;
        }
        assert!(
            !ent.path().join("id_invert").exists(),
            "drop_index should remove seg_*/id_invert"
        );
    }

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_drop_index_persists_across_reopen() {
    let path = temp_dir("drop_idx_reopen");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4));

    {
        let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

        col.insert(vec![
            Doc::new("d1").set("emb", vec![1.0f32, 0.0, 0.0, 0.0]),
            Doc::new("d2").set("emb", vec![0.0f32, 1.0, 0.0, 0.0]),
        ])
        .unwrap();

        col.create_index(
            "emb",
            IndexParams::Flat(FlatIndexParams {
                metric: MetricType::L2,
                quantize: Default::default(),
                column_major: false,
            }),
            CreateIndexOptions::default(),
        )
        .unwrap();

        col.drop_index("emb").unwrap();
    }

    // Reopen: index must remain dropped.
    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    let info = col.schema_info();
    assert!(
        info.get_field("emb")
            .map(|f| f.index_params.is_none())
            .unwrap_or(false),
        "index_params should remain cleared after reopen"
    );

    // On-disk index dirs should be gone too.
    for ent in std::fs::read_dir(&path).unwrap() {
        let ent = ent.unwrap();
        let name = ent.file_name().to_string_lossy().to_string();
        if !name.starts_with("seg_") {
            continue;
        }
        let idx_dir = ent.path().join("idx_emb");
        assert!(!idx_dir.exists(), "idx_emb must not exist after drop_index");
    }

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_drop_sparse_index_persists_across_reopen() {
    let path = temp_dir("drop_sparse_idx_reopen");
    let schema = CollectionSchema::new("test").with_field(
        FieldSchema::new("sparse_emb", DataType::SparseFp32)
            .with_dimension(1000)
            .with_index(IndexParams::FlatSparse(FlatIndexParams {
                metric: MetricType::InnerProduct,
                quantize: QuantizeType::Undefined,
                column_major: false,
            })),
    );

    {
        let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

        col.insert(vec![
            Doc::new("d1").set(
                "sparse_emb",
                Value::SparseF32 {
                    indices: vec![0, 5, 10],
                    values: vec![1.0, 0.5, 0.3],
                },
            ),
            Doc::new("d2").set(
                "sparse_emb",
                Value::SparseF32 {
                    indices: vec![1, 6, 100],
                    values: vec![1.0, 0.5, 0.3],
                },
            ),
            Doc::new("d3").set(
                "sparse_emb",
                Value::SparseF32 {
                    indices: vec![0, 5, 10],
                    values: vec![0.8, 0.4, 0.2],
                },
            ),
        ])
        .unwrap();

        col.flush().unwrap();

        let mut q = VectorQuery::new("sparse_emb", Vec::new(), 2);
        q.sparse_indices = vec![0, 5, 10];
        q.sparse_values = vec![1.0, 0.5, 0.3];
        let results = col.query(q).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].pk, "d1");

        // Drop the index and ensure brute-force fallback remains correct.
        col.drop_index("sparse_emb").unwrap();
        assert!(
            col.schema_info()
                .get_field("sparse_emb")
                .map(|f| f.index_params.is_none())
                .unwrap_or(false),
            "index_params should be cleared after drop_index"
        );

        let mut q = VectorQuery::new("sparse_emb", Vec::new(), 2);
        q.sparse_indices = vec![0, 5, 10];
        q.sparse_values = vec![1.0, 0.5, 0.3];
        let results = col.query(q).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].pk, "d1");
    }

    // Reopen: index must remain dropped and on-disk index dirs removed.
    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    let info = col.schema_info();
    assert!(
        info.get_field("sparse_emb")
            .map(|f| f.index_params.is_none())
            .unwrap_or(false),
        "index_params should remain cleared after reopen"
    );
    for ent in std::fs::read_dir(&path).unwrap() {
        let ent = ent.unwrap();
        let name = ent.file_name().to_string_lossy().to_string();
        if !name.starts_with("seg_") {
            continue;
        }
        let idx_dir = ent.path().join("idx_sparse_emb");
        assert!(
            !idx_dir.exists(),
            "idx_sparse_emb must not exist after drop_index"
        );
    }

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_create_index_rebuild_semantics() {
    let path = temp_dir("create_idx_rebuild");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![
        Doc::new("d1").set("emb", vec![1.0f32, 0.0, 0.0, 0.0]),
        Doc::new("d2").set("emb", vec![0.0f32, 1.0, 0.0, 0.0]),
        Doc::new("d3").set("emb", vec![0.0f32, 0.0, 1.0, 0.0]),
    ])
    .unwrap();

    // Initial index: FLAT.
    col.create_index(
        "emb",
        IndexParams::Flat(FlatIndexParams {
            metric: MetricType::L2,
            quantize: Default::default(),
            column_major: false,
        }),
        CreateIndexOptions::default(),
    )
    .unwrap();
    assert!(matches!(
        col.schema_info()
            .get_field("emb")
            .and_then(|f| f.index_params.clone()),
        Some(IndexParams::Flat(_))
    ));

    // Attempt to change index type without rebuild: should be a no-op.
    col.create_index(
        "emb",
        IndexParams::Hnsw(HnswIndexParams::new(MetricType::L2)),
        CreateIndexOptions::default(),
    )
    .unwrap();
    assert!(matches!(
        col.schema_info()
            .get_field("emb")
            .and_then(|f| f.index_params.clone()),
        Some(IndexParams::Flat(_))
    ));

    // Now rebuild with HNSW.
    col.create_index(
        "emb",
        IndexParams::Hnsw(HnswIndexParams::new(MetricType::L2)),
        CreateIndexOptions {
            rebuild: true,
            concurrency: None,
        },
    )
    .unwrap();
    assert!(matches!(
        col.schema_info()
            .get_field("emb")
            .and_then(|f| f.index_params.clone()),
        Some(IndexParams::Hnsw(_))
    ));

    // Query still works post-rebuild.
    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 1);
    let results = col.query(q).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].pk, "d1");

    drop(col);

    // Reopen preserves index params.
    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    assert!(matches!(
        col.schema_info()
            .get_field("emb")
            .and_then(|f| f.index_params.clone()),
        Some(IndexParams::Hnsw(_))
    ));

    // is_linear should still allow brute-force fallback even if index exists.
    let qp = QueryParams {
        is_linear: Some(true),
        ..QueryParams::default()
    };
    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 1).with_params(qp);
    let results = col.query(q).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].pk, "d1");

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_create_index_rebuild_updates_writing_segment_metric_even_with_invert_index() {
    let path = temp_dir("create_idx_rebuild_invert_writing_metric");
    let schema = CollectionSchema::new("test")
        .with_field(
            FieldSchema::new("id", DataType::Int64)
                .not_null()
                .with_index(IndexParams::Invert(InvertIndexParams {
                    enable_range_optimization: true,
                    enable_extended_wildcard: false,
                })),
        )
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(1));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    // Switch vector index params to L2 HNSW. This should update the active writing segment's
    // in-memory vector store metric without re-opening invert fjall handles.
    col.create_index(
        "emb",
        IndexParams::Hnsw(HnswIndexParams::new(MetricType::L2)),
        CreateIndexOptions {
            rebuild: true,
            concurrency: None,
        },
    )
    .unwrap();

    // Insert into the active writing segment (no flush). Query must use L2, not IP.
    col.insert(vec![
        Doc::new("low").set("id", 1i64).set("emb", vec![1.0f32]),
        Doc::new("high").set("id", 2i64).set("emb", vec![100.0f32]),
    ])
    .unwrap();

    let q = VectorQuery::new("emb", vec![1.1f32], 1);
    let results = col.query(q).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].pk, "low");

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

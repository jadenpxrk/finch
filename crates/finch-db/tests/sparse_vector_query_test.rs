mod common;
use common::*;

// ── Sparse vector query ───────────────────────────────────────────────────────

#[test]
fn test_sparse_vector_query() {
    let path = temp_dir("sparse");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("sparse_emb", DataType::SparseFp32).with_dimension(1000));

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

    // Sparse query matching d1 and d3 (indices 0, 5, 10)
    let mut q = VectorQuery::new("sparse_emb", Vec::new(), 2);
    q.sparse_indices = vec![0, 5, 10];
    q.sparse_values = vec![1.0, 0.5, 0.3];

    let results = col.query(q).unwrap();
    assert_eq!(results.len(), 2);
    // d1 has the highest inner product with the query
    assert_eq!(results[0].pk, "d1");
    assert!(
        results.iter().all(|d| d.pk != "d2"),
        "d2 has no shared indices"
    );

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_sparse_vectors_are_canonicalized_sort_and_dedup_on_write_and_query() {
    let path = temp_dir("sparse_normalize");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("sparse_emb", DataType::SparseFp32).with_dimension(10_000));

    {
        let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

        // Unsorted indices with duplicates.
        col.insert(vec![Doc::new("d1").set(
            "sparse_emb",
            Value::SparseF32 {
                indices: vec![10, 5, 10, 0],
                values: vec![1.0, 2.0, 3.0, 4.0],
            },
        )])
        .unwrap();

        let fetched = col.fetch(vec!["d1".to_string()]).unwrap();
        let d1 = fetched.get("d1").expect("d1 fetch");
        match d1.get("sparse_emb") {
            Some(Value::SparseF32 { indices, values }) => {
                assert_eq!(indices, &vec![0, 5, 10]);
                assert_eq!(values, &vec![4.0, 2.0, 4.0]); // 10 merged: 1.0 + 3.0
            }
            other => panic!("expected SparseF32, got {other:?}"),
        }

        // Query with unsorted indices and duplicates should behave identically.
        let mut q = VectorQuery::new("sparse_emb", Vec::new(), 1);
        q.sparse_indices = vec![10, 0, 10, 5];
        q.sparse_values = vec![1.0, 4.0, 3.0, 2.0];
        let got = col.query(q).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].pk, "d1");

        col.flush().unwrap();
    }

    // Reopen: canonicalization persists through forward-store roundtrips.
    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    let fetched = col.fetch(vec!["d1".to_string()]).unwrap();
    let d1 = fetched.get("d1").expect("d1 fetch after reopen");
    match d1.get("sparse_emb") {
        Some(Value::SparseF32 { indices, values }) => {
            assert_eq!(indices, &vec![0, 5, 10]);
            assert_eq!(values, &vec![4.0, 2.0, 4.0]);
        }
        other => panic!("expected SparseF32 after reopen, got {other:?}"),
    }

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_insert_sparse_f16_is_accepted_and_queryable() {
    let path = temp_dir("sparse_f16_insert");
    let schema = CollectionSchema::new("test").with_field(
        FieldSchema::new("sparse_emb", DataType::SparseFp16).with_index(IndexParams::FlatSparse(
            FlatIndexParams {
                metric: MetricType::InnerProduct,
                quantize: QuantizeType::Undefined,
                column_major: false,
            },
        )),
    );

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![
        Doc::new("d1").set(
            "sparse_emb",
            Value::SparseF16 {
                indices: vec![0, 5, 10],
                values: vec![f16::from_f32(1.0), f16::from_f32(0.5), f16::from_f32(0.3)],
            },
        ),
        Doc::new("d2").set(
            "sparse_emb",
            Value::SparseF16 {
                indices: vec![1, 6, 100],
                values: vec![f16::from_f32(1.0), f16::from_f32(0.5), f16::from_f32(0.3)],
            },
        ),
    ])
    .unwrap();

    col.flush().unwrap();

    let mut q = VectorQuery::new("sparse_emb", Vec::new(), 2);
    q.sparse_indices = vec![0, 5, 10];
    q.sparse_values = vec![1.0, 0.5, 0.3];
    let results = col.query(q).unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].pk, "d1");

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_sparse_index_persisted_flat_fp16() {
    let path = temp_dir("sparse_idx_flat_fp16");
    let schema = CollectionSchema::new("test").with_field(
        FieldSchema::new("sparse_emb", DataType::SparseFp32)
            .with_dimension(1000)
            .with_index(IndexParams::FlatSparse(FlatIndexParams {
                metric: MetricType::InnerProduct,
                quantize: QuantizeType::Fp16,
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

        // Flush to persisted segments; sparse vectors must remain queryable and
        // the sparse index must build/load successfully.
        col.flush().unwrap();

        let mut q = VectorQuery::new("sparse_emb", Vec::new(), 2);
        q.sparse_indices = vec![0, 5, 10];
        q.sparse_values = vec![1.0, 0.5, 0.3];

        let results = col.query(q).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].pk, "d1");
        assert!(results.iter().all(|d| d.pk != "d2"));
    }

    // Reopen: index must reload and queries must still work.
    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    let mut q = VectorQuery::new("sparse_emb", Vec::new(), 2);
    q.sparse_indices = vec![0, 5, 10];
    q.sparse_values = vec![1.0, 0.5, 0.3];
    let results = col.query(q).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].pk, "d1");

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

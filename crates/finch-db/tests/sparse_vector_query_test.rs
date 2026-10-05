mod common;
use common::*;

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

#[test]
fn flushed_sparse_index_excludes_empty_vectors() {
    // Indexed recall must exclude the same NULL sparse vectors as a forward scan and rebuild.
    let path = temp_dir("flushed_empty_sparse");
    let schema = CollectionSchema::new("test").with_field(
        FieldSchema::new("sparse", DataType::SparseFp32)
            .with_dimension(3)
            .with_index(IndexParams::FlatSparse(FlatIndexParams::new(
                MetricType::InnerProduct,
            ))),
    );
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let statuses = col
        .insert(vec![
            Doc::new("empty").set(
                "sparse",
                Value::SparseF32 {
                    indices: Vec::new(),
                    values: Vec::new(),
                },
            ),
            Doc::new("present").set(
                "sparse",
                Value::SparseF32 {
                    indices: vec![0],
                    values: vec![1.0],
                },
            ),
        ])
        .unwrap();
    assert!(statuses.iter().all(|status| status.ok()));
    let mut query = VectorQuery::new("sparse", Vec::new(), 10);
    query.sparse_indices = vec![0];
    query.sparse_values = vec![1.0];
    col.flush().unwrap();
    drop(col);
    let reopened = Collection::open(&path, CollectionOptions::default()).unwrap();
    let indexed = reopened.query(query.clone()).unwrap();
    let linear = reopened
        .query(query.clone().with_params(QueryParams {
            is_linear: Some(true),
            ..Default::default()
        }))
        .unwrap();
    let fetched = reopened.fetch(vec!["empty".to_string()]).unwrap();
    reopened
        .create_index(
            "sparse",
            IndexParams::FlatSparse(FlatIndexParams::new(MetricType::InnerProduct)),
            CreateIndexOptions {
                rebuild: true,
                ..Default::default()
            },
        )
        .unwrap();
    let rebuilt = reopened.query(query).unwrap();
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
    assert_eq!(
        linear.iter().map(|doc| doc.pk.as_str()).collect::<Vec<_>>(),
        vec!["present"]
    );
    assert_eq!(
        rebuilt
            .iter()
            .map(|doc| doc.pk.as_str())
            .collect::<Vec<_>>(),
        vec!["present"]
    );
    assert!(fetched["empty"].get("sparse").is_none());
    assert_eq!(
        indexed
            .iter()
            .map(|doc| doc.pk.as_str())
            .collect::<Vec<_>>(),
        vec!["present"]
    );
}

#[test]
fn unflushed_sparse_index_excludes_empty_vectors() {
    // The writing segment and its WAL replay must agree with the flushed index on empty vectors.
    let path = temp_dir("unflushed_empty_sparse");
    let schema = CollectionSchema::new("test").with_field(
        FieldSchema::new("sparse", DataType::SparseFp32)
            .with_dimension(3)
            .with_index(IndexParams::FlatSparse(FlatIndexParams::new(
                MetricType::InnerProduct,
            ))),
    );
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    col.insert(vec![
        Doc::new("empty").set(
            "sparse",
            Value::SparseF32 {
                indices: Vec::new(),
                values: Vec::new(),
            },
        ),
        Doc::new("present").set(
            "sparse",
            Value::SparseF32 {
                indices: vec![0],
                values: vec![1.0],
            },
        ),
    ])
    .unwrap();
    let mut query = VectorQuery::new("sparse", Vec::new(), 10);
    query.sparse_indices = vec![0];
    query.sparse_values = vec![1.0];
    let unflushed = col.query(query.clone()).unwrap();
    drop(col);
    let reopened = Collection::open(&path, CollectionOptions::default()).unwrap();
    let replayed = reopened.query(query).unwrap();
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
    for docs in [unflushed, replayed] {
        assert_eq!(
            docs.iter().map(|doc| doc.pk.as_str()).collect::<Vec<_>>(),
            vec!["present"]
        );
    }
}

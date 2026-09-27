mod common;
use common::*;

#[test]
fn flushed_sparse_index_returns_a_null_vector_until_rebuilt() {
    // Indexed recall must exclude the same NULL sparse vectors as a forward scan and rebuild.
    let path = temp_dir("round_5_empty_sparse");
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
    let path = temp_dir("round_5_unflushed_empty_sparse");
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

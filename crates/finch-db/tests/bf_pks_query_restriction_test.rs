mod common;
use common::*;

#[test]
fn test_query_bf_pks_restricts_results() {
    let path = temp_dir("bf_pks");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![
        Doc::new("1").set("emb", vec![1.0f32, 0.0, 0.0, 0.0]),
        Doc::new("2").set("emb", vec![0.9f32, 0.0, 0.0, 0.0]),
        Doc::new("3").set("emb", vec![0.8f32, 0.0, 0.0, 0.0]),
    ])
    .unwrap();

    // Without bf_pks, pk "1" should be the nearest neighbor.
    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 3);
    let results = col.query(q).unwrap();
    assert_eq!(results[0].pk, "1");

    // With bf_pks, "1" must not appear even though it is the nearest.
    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 3).with_params(QueryParams {
        bf_pks: Some(vec!["2".to_string(), "3".to_string(), "999".to_string()]),
        ..Default::default()
    });
    let results = col.query(q).unwrap();
    assert!(!results.is_empty());
    assert!(results.iter().all(|d| d.pk == "2" || d.pk == "3"));
    assert_ne!(results[0].pk, "1");

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_query_validation_rejects_bad_inputs() {
    let path = temp_dir("query_validate");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("sparse_emb", DataType::SparseFp32).with_dimension(1000));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    // topk too large
    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 2000);
    assert!(col.query(q).is_err());

    // output_fields too large
    let many_fields: Vec<String> = (0..2000).map(|i| format!("f{}", i)).collect();
    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 1).with_output_fields(many_fields);
    assert!(col.query(q).is_err());

    // dense dimension mismatch
    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0], 1);
    assert!(col.query(q).is_err());

    // sparse indices too large
    let mut q = VectorQuery::new("sparse_emb", Vec::new(), 1);
    q.sparse_indices = (0..5000u32).collect();
    q.sparse_values = vec![1.0f32; 5000];
    assert!(col.query(q).is_err());

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_query_empty_query_vector_is_treated_as_filter_only() {
    let path = temp_dir("empty_query_vector_filter_only");
    let schema = basic_schema(4);
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![
        make_doc("d0", vec![1.0, 0.0, 0.0, 0.0], "d0"),
        make_doc("d1", vec![0.0, 1.0, 0.0, 0.0], "d1"),
        make_doc("d2", vec![0.0, 0.0, 1.0, 0.0], "d2"),
    ])
    .unwrap();
    col.flush().unwrap();

    // Field name is set, but query_vector is empty -> treat as filter-only.
    let mut q = VectorQuery::new("emb", Vec::new(), 10).with_filter("label = 'd1'");
    q.output_fields = Some(vec!["label".to_string()]);
    let res = col.query(q).unwrap();
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].pk, "d1");

    // If a vector payload exists but field_name is empty, reject it.
    let err = col
        .query(VectorQuery::new("", vec![1.0f32, 0.0, 0.0, 0.0], 10))
        .unwrap_err();
    assert!(err.message.contains("vector_field"), "err={err:?}");

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

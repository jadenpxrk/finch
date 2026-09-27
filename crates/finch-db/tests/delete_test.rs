mod common;
use common::*;

// ── Delete ────────────────────────────────────────────────────────────────────

#[test]
fn test_delete_by_pk() {
    let path = temp_dir("delete");
    let schema = basic_schema(4);
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    let statuses = col
        .insert(vec![
            make_doc("d1", vec![1.0, 0.0, 0.0, 0.0], "a"),
            make_doc("d2", vec![0.0, 1.0, 0.0, 0.0], "b"),
        ])
        .unwrap();
    assert!(
        statuses.iter().all(|s| s.is_ok()),
        "all inserts should succeed"
    );

    let del = col.delete(vec!["d1".to_string()]).unwrap();
    assert!(del[0].is_ok());

    let stats = col.stats().unwrap();
    assert_eq!(stats.doc_count, 1);

    // d1 should no longer appear in query results
    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 5);
    let results = col.query(q).unwrap();
    assert!(
        results.iter().all(|d| d.pk != "d1"),
        "deleted doc should not appear in results"
    );

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_delete_nonexistent() {
    let path = temp_dir("delete_ne");
    let schema = basic_schema(4);
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    let results = col.delete(vec!["ghost".to_string()]).unwrap();
    assert!(results[0].is_err());

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_delete_by_filter() {
    let path = temp_dir("delete_filter");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("score", DataType::Float32));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![
        Doc::new("d1")
            .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
            .set("score", 10.0f32),
        Doc::new("d2")
            .set("emb", vec![0.0f32, 1.0, 0.0, 0.0])
            .set("score", 2.0f32),
        Doc::new("d3")
            .set("emb", vec![0.0f32, 0.0, 1.0, 0.0])
            .set("score", 8.0f32),
    ])
    .unwrap();

    let status = col.delete_by_filter("score < 5.0").unwrap();
    assert!(status.is_ok());

    let stats = col.stats().unwrap();
    assert_eq!(stats.doc_count, 2); // d2 deleted

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

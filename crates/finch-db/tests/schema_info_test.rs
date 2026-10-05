mod common;
use common::*;

#[test]
fn test_stats_with_deletes() {
    let path = temp_dir("stats_del");
    let schema = basic_schema(4);
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![
        make_doc("d1", vec![1.0, 0.0, 0.0, 0.0], "a"),
        make_doc("d2", vec![0.0, 1.0, 0.0, 0.0], "b"),
        make_doc("d3", vec![0.0, 0.0, 1.0, 0.0], "c"),
    ])
    .unwrap();

    assert_eq!(col.stats().unwrap().doc_count, 3);

    col.delete(vec!["d1".to_string(), "d3".to_string()])
        .unwrap();
    assert_eq!(col.stats().unwrap().doc_count, 1);

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

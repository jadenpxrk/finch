mod common;
use common::*;

#[test]
fn test_fetch_by_pk() {
    let path = temp_dir("fetch");
    let schema = basic_schema(4);
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![make_doc("pk1", vec![1.0, 0.0, 0.0, 0.0], "alice")])
        .unwrap();

    let fetched = col
        .fetch(vec!["pk1".to_string(), "missing".to_string()])
        .unwrap();
    assert!(fetched.contains_key("pk1"));
    assert!(!fetched.contains_key("missing"));
    assert_eq!(fetched["pk1"].pk, "pk1");

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

mod common;
use common::*;

#[test]
fn indexed_like_deletes_rows_whose_prefix_and_suffix_overlap() {
    // LIKE literals must consume distinct characters, or filtered deletion loses nonmatching rows.
    let path = temp_dir("round_2_like_overlap");
    let schema =
        CollectionSchema::new("test")
            .with_field(FieldSchema::new("plain", DataType::String))
            .with_field(FieldSchema::new("indexed", DataType::String).with_index(
                IndexParams::Invert(InvertIndexParams {
                    enable_extended_wildcard: true,
                    ..Default::default()
                }),
            ));
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let statuses = col
        .insert(vec![
            Doc::new("overlap")
                .set("plain", "aba")
                .set("indexed", "aba"),
            Doc::new("match")
                .set("plain", "abba")
                .set("indexed", "abba"),
        ])
        .unwrap();
    assert!(statuses.iter().all(|status| status.ok()));
    col.flush().unwrap();
    drop(col);
    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    let plain = col
        .query(VectorQuery::new("", vec![], 10).with_filter("plain LIKE 'ab%ba'"))
        .unwrap();
    let indexed = col
        .query(VectorQuery::new("", vec![], 10).with_filter("indexed LIKE 'ab%ba'"))
        .unwrap();
    assert!(col.delete_by_filter("indexed LIKE 'ab%ba'").unwrap().ok());
    drop(col);
    let reopened = Collection::open(&path, CollectionOptions::default()).unwrap();
    let survivors = reopened
        .fetch(vec!["overlap".to_string(), "match".to_string()])
        .unwrap();
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
    assert_eq!(plain.len(), 1);
    assert_eq!(plain[0].pk, "match");
    assert_eq!(
        (
            indexed.len(),
            survivors.contains_key("overlap"),
            survivors.contains_key("match")
        ),
        (1, true, false),
        "indexed LIKE matched and deleted the nonmatching overlap row"
    );
}

#[test]
fn sql_aliases_that_swap_column_names_lose_both_values() {
    // Projection aliases must read original values and preserve every selected output name.
    let path = temp_dir("round_2_alias_swap");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("a", DataType::Int64))
        .with_field(FieldSchema::new("b", DataType::Int64));
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let statuses = col
        .insert(vec![Doc::new("row").set("a", 1i64).set("b", 2i64)])
        .unwrap();
    assert!(statuses.iter().all(|status| status.ok()));
    col.flush().unwrap();
    drop(col);
    let reopened = Collection::open(&path, CollectionOptions::default()).unwrap();
    let original = reopened.fetch(vec!["row".to_string()]).unwrap();
    let projected = reopened
        .query_sql("SELECT a AS b, b AS a FROM test LIMIT 1")
        .unwrap();
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
    assert_eq!(original["row"].fields.get("a"), Some(&Value::I64(1)));
    assert_eq!(original["row"].fields.get("b"), Some(&Value::I64(2)));
    assert_eq!(projected.len(), 1);
    assert_eq!(
        (projected[0].fields.get("a"), projected[0].fields.get("b")),
        (Some(&Value::I64(2)), Some(&Value::I64(1)))
    );
}

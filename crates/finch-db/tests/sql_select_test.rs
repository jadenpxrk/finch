mod common;
use common::*;

#[test]
fn sql_aliases_that_swap_column_names_keep_both_values() {
    // Projection aliases must read original values and preserve every selected output name.
    let path = temp_dir("sql_alias_swap");
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

#[test]
fn sql_limit_zero_returns_no_rows() {
    // An explicit zero limit must not be replaced by the default result limit.
    let path = temp_dir("sql_limit_zero");
    let schema =
        CollectionSchema::new("test").with_field(FieldSchema::new("value", DataType::Int64));
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let statuses = col
        .insert(vec![Doc::new("row").set("value", 1i64)])
        .unwrap();
    assert!(statuses.iter().all(|status| status.ok()));
    col.flush().unwrap();
    drop(col);
    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    let rows = col.query_sql("SELECT value FROM test LIMIT 0").unwrap();
    drop(col);
    std::fs::remove_dir_all(path).unwrap();
    assert!(rows.is_empty(), "LIMIT 0 returned {} rows", rows.len());
}

#[test]
fn sql_order_by_sorts_every_match_before_the_limit() {
    // A scalar ORDER BY must sort all matching rows before applying LIMIT.
    let path = temp_dir("sql_order_by_limit");
    let schema =
        CollectionSchema::new("test").with_field(FieldSchema::new("value", DataType::Int64));
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let statuses = col
        .insert(vec![
            Doc::new("first").set("value", 10i64),
            Doc::new("best").set("value", 20i64),
        ])
        .unwrap();
    assert!(statuses.iter().all(|status| status.ok()));
    col.flush().unwrap();
    drop(col);
    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    let all = col
        .query_sql("SELECT value FROM test WHERE value > 0 ORDER BY value DESC LIMIT 2")
        .unwrap();
    let limited = col
        .query_sql("SELECT value FROM test WHERE value > 0 ORDER BY value DESC LIMIT 1")
        .unwrap();
    drop(col);
    std::fs::remove_dir_all(path).unwrap();
    assert_eq!(all[0].get("value"), Some(&Value::I64(20)));
    assert_eq!(limited.len(), 1);
    assert_eq!(limited[0].get("value"), all[0].get("value"));
}

#[test]
fn sql_scalar_order_by_rejects_a_limit_above_the_query_cap() {
    // The sort-then-limit path must enforce the same top-k cap as every other query path.
    let path = temp_dir("sql_order_by_limit_cap");
    let schema =
        CollectionSchema::new("test").with_field(FieldSchema::new("value", DataType::Int64));
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    col.insert(vec![Doc::new("row").set("value", 1i64)])
        .unwrap();
    let capped = col.query_sql("SELECT value FROM test ORDER BY value DESC LIMIT 1024");
    let over = col.query_sql("SELECT value FROM test ORDER BY value DESC LIMIT 1025");
    drop(col);
    std::fs::remove_dir_all(path).unwrap();
    assert_eq!(capped.unwrap().len(), 1);
    assert!(over.is_err());
}

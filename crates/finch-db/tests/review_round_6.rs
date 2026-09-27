mod common;
use common::*;

#[test]
fn sql_limit_zero_returns_rows_instead_of_an_empty_result() {
    // An explicit zero limit must not be replaced by the default result limit.
    let path = temp_dir("round_6_limit_zero");
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

mod common;
use common::*;

#[test]
fn ddl_integer_literal_is_rounded_before_backfill() {
    // Integer literals must retain their exact value when backfilling an integer column.
    let path = temp_dir("round_7_ddl_integer_literal");
    let schema =
        CollectionSchema::new("test").with_field(FieldSchema::new("value", DataType::Int64));
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let statuses = col
        .insert(vec![Doc::new("row").set("value", 1i64)])
        .unwrap();
    assert!(statuses.iter().all(|status| status.ok()));
    col.add_column_with_expression(
        FieldSchema::new("external_id", DataType::Int64),
        Some("9007199254740993"),
        AddColumnOptions::default(),
    )
    .unwrap();
    drop(col);
    let reopened = Collection::open(&path, CollectionOptions::default()).unwrap();
    let docs = reopened.fetch(vec!["row".to_string()]).unwrap();
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
    assert_eq!(
        docs["row"].get("external_id"),
        Some(&Value::I64(9_007_199_254_740_993))
    );
}

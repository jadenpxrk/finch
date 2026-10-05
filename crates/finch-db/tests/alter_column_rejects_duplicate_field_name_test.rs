mod common;
use common::*;

#[test]
fn test_alter_column_rejects_duplicate_name() {
    let path = temp_dir("alter_dup");
    let schema = basic_schema(4);
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.add_column(
        FieldSchema::new("x", DataType::Int32).nullable(),
        AddColumnOptions::default(),
    )
    .unwrap();

    // Renaming "x" to "label" should fail since "label" already exists
    let result = col.alter_column("x", Some("label"), None, AlterColumnOptions::default());
    assert!(
        result.is_err(),
        "renaming to an existing field name should fail"
    );

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

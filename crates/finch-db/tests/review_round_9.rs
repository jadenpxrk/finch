mod common;
use common::*;

#[test]
fn ddl_division_by_one_rounds_an_existing_int64_value() {
    // An identity expression must preserve an existing integer during DDL backfill.
    let path = temp_dir("round_9_ddl_integer_division");
    let schema =
        CollectionSchema::new("test").with_field(FieldSchema::new("value", DataType::Int64));
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let value = 9_007_199_254_740_993i64;
    let statuses = col
        .insert(vec![Doc::new("row").set("value", value)])
        .unwrap();
    assert!(statuses.iter().all(|status| status.ok()));
    col.add_column_with_expression(
        FieldSchema::new("copy", DataType::Int64),
        Some("value / 1"),
        AddColumnOptions::default(),
    )
    .unwrap();
    drop(col);
    let reopened = Collection::open(&path, CollectionOptions::default()).unwrap();
    let docs = reopened.fetch(vec!["row".to_string()]).unwrap();
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
    assert_eq!(docs["row"].get("value"), Some(&Value::I64(value)));
    assert_eq!(docs["row"].get("copy"), Some(&Value::I64(value)));
}

#[test]
fn creating_vector_index_on_empty_scalar_collection_commits_invalid_schema() {
    // Invalid index kinds must be rejected even when there are no segments to build.
    let path = temp_dir("round_9_empty_scalar_index");
    let schema =
        CollectionSchema::new("test").with_field(FieldSchema::new("value", DataType::Int64));
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let result = col.create_index(
        "value",
        IndexParams::Flat(FlatIndexParams {
            metric: MetricType::L2,
            quantize: Default::default(),
            column_major: false,
        }),
        CreateIndexOptions::default(),
    );
    let validation = col.schema_info().validate();
    drop(col);
    let reopened_validation = Collection::open(&path, CollectionOptions::default())
        .map(|reopened| reopened.schema_info().validate());
    std::fs::remove_dir_all(path).unwrap();
    assert!(result.is_err(), "index creation returned {result:?}; schema validation: {validation:?}; reopened schema validation: {reopened_validation:?}");
    assert!(validation.is_ok());
    assert!(matches!(reopened_validation, Ok(Ok(()))));
}

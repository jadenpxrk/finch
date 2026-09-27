mod common;
use common::*;
use finch_types::Status;

#[test]
fn ddl_identity_backfill_corrupts_large_integer_values() {
    // An identity expression must preserve an integer exactly across DDL and reopen.
    let path = temp_dir("round_1_ddl_integer_precision");
    let schema = basic_schema(2).with_field(FieldSchema::new("original", DataType::Int64));
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let original = 9_007_199_254_740_993i64;
    let statuses = col
        .insert(vec![
            make_doc("row", vec![1.0, 0.0], "value").set("original", original)
        ])
        .unwrap();
    assert!(statuses.iter().all(Status::is_ok));
    col.add_column_with_expression(
        FieldSchema::new("copied", DataType::Int64),
        Some("original"),
        AddColumnOptions::default(),
    )
    .unwrap();
    drop(col);

    let reopened = Collection::open(&path, CollectionOptions::default()).unwrap();
    let docs = reopened.fetch(vec!["row".to_string()]).unwrap();
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
    assert_eq!(docs["row"].fields["original"], Value::I64(original));
    assert_eq!(docs["row"].fields["copied"], Value::I64(original));
}

#[test]
fn group_by_splits_equal_positive_and_negative_zero() {
    // Float values that compare equal must occupy one group, including signed zero.
    let path = temp_dir("round_1_group_signed_zero");
    let schema = basic_schema(2).with_field(FieldSchema::new("key", DataType::Float64));
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let statuses = col
        .insert(vec![
            make_doc("positive", vec![1.0, 0.0], "value").set("key", 0.0f64),
            make_doc("negative", vec![1.0, 0.1], "value").set("key", -0.0f64),
        ])
        .unwrap();
    assert!(statuses.iter().all(Status::is_ok));
    let matching = col
        .query(VectorQuery::new("emb", vec![1.0, 0.0], 10).with_filter("key = 0"))
        .unwrap();
    let query = GroupByVectorQuery {
        base: VectorQuery::new("emb", vec![1.0, 0.0], 10),
        group_by_field: "key".to_string(),
        group_count: 2,
        group_topk: 2,
    };
    let writing_groups = col.group_by_query(query.clone()).unwrap();
    col.flush().unwrap();
    drop(col);
    let reopened = Collection::open(&path, CollectionOptions::default()).unwrap();
    let persisted_groups = reopened.group_by_query(query).unwrap();
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
    assert_eq!(matching.len(), 2);
    assert_eq!(
        (writing_groups.len(), persisted_groups.len()),
        (1, 1),
        "both zero-valued documents belong to the same group"
    );
    assert_eq!(persisted_groups[0].docs.len(), 2);
}

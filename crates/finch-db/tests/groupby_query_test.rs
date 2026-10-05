mod common;
use common::*;

#[test]
fn test_group_by_query() {
    let path = temp_dir("group_by");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("category", DataType::String));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![
        Doc::new("d1")
            .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
            .set("category", "A"),
        Doc::new("d2")
            .set("emb", vec![0.9f32, 0.1, 0.0, 0.0])
            .set("category", "A"),
        Doc::new("d3")
            .set("emb", vec![0.0f32, 1.0, 0.0, 0.0])
            .set("category", "B"),
        Doc::new("d4")
            .set("emb", vec![0.0f32, 0.9, 0.1, 0.0])
            .set("category", "B"),
        // NULL group-by keys are ignored (no NULL group).
        Doc::new("d5").set("emb", vec![0.8f32, 0.2, 0.0, 0.0]),
    ])
    .unwrap();

    let base = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10);
    let group_query = GroupByVectorQuery {
        base,
        group_by_field: "category".to_string(),
        // group_topk = groups, group_count = docs per group
        group_count: 2,
        group_topk: 1,
    };

    let results = col.group_by_query(group_query).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].group_value, Value::String("A".to_string()));
    assert_eq!(results[0].docs.len(), 2);
    assert_eq!(results[0].docs[0].pk, "d1");
    assert_eq!(results[0].docs[1].pk, "d2");

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_group_by_system_column_does_not_leak_doc_id_by_default() {
    let path = temp_dir("group_by_system_uid");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("category", DataType::String));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    col.insert(vec![
        Doc::new("d1")
            .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
            .set("category", "A"),
        Doc::new("d2")
            .set("emb", vec![0.9f32, 0.1, 0.0, 0.0])
            .set("category", "A"),
        Doc::new("d3")
            .set("emb", vec![0.0f32, 1.0, 0.0, 0.0])
            .set("category", "B"),
    ])
    .unwrap();

    let mut base = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10);
    base.include_doc_id = false;

    let group_query = GroupByVectorQuery {
        base,
        group_by_field: SYS_GLOBAL_DOC_ID.to_string(),
        group_count: 1,
        group_topk: 2,
    };
    let results = col.group_by_query(group_query).unwrap();
    assert_eq!(results.len(), 2);
    assert!(matches!(results[0].group_value, Value::U64(_)));
    assert_eq!(results[0].docs.len(), 1);
    assert_eq!(results[0].docs[0].doc_id, 0);

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_group_by_validation() {
    let path = temp_dir("group_by_validation");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("category", DataType::String))
        .with_field(FieldSchema::new("tags", DataType::ArrayString));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    col.insert(vec![Doc::new("d1")
        .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
        .set("category", "A")
        .set("tags", vec!["x".to_string()])])
        .unwrap();

    // Missing vector query.
    let group_query = GroupByVectorQuery {
        base: VectorQuery::new("", Vec::new(), 10).with_filter("category = 'A'"),
        group_by_field: "category".to_string(),
        group_count: 1,
        group_topk: 1,
    };
    let err = col.group_by_query(group_query).unwrap_err();
    assert!(err.message.contains("group by should has vector query"));

    // Missing vector payload (field_name present but query_vector empty) is also rejected.
    let group_query = GroupByVectorQuery {
        base: VectorQuery::new("emb", Vec::new(), 10).with_filter("category = 'A'"),
        group_by_field: "category".to_string(),
        group_count: 1,
        group_topk: 1,
    };
    let err = col.group_by_query(group_query).unwrap_err();
    assert!(err.message.contains("group by should has vector query"));

    // Missing group-by field.
    let group_query = GroupByVectorQuery {
        base: VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10),
        group_by_field: "missing".to_string(),
        group_count: 1,
        group_topk: 1,
    };
    let err = col.group_by_query(group_query).unwrap_err();
    assert!(err.message.contains("not defined in schema"));

    // Array group-by field is rejected.
    let group_query = GroupByVectorQuery {
        base: VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10),
        group_by_field: "tags".to_string(),
        group_count: 1,
        group_topk: 1,
    };
    let err = col.group_by_query(group_query).unwrap_err();
    assert!(err
        .message
        .contains("group by fields should not be array data type"));

    // A vector field cannot be a group-by field, so validation reports it as "not defined in schema".
    let group_query = GroupByVectorQuery {
        base: VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10),
        group_by_field: "emb".to_string(),
        group_count: 1,
        group_topk: 1,
    };
    let err = col.group_by_query(group_query).unwrap_err();
    assert!(err.message.contains("not defined in schema"));

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn group_by_puts_positive_and_negative_zero_in_one_group() {
    // Float values that compare equal must occupy one group, including signed zero.
    let path = temp_dir("group_by_signed_zero");
    let schema = basic_schema(2).with_field(FieldSchema::new("key", DataType::Float64));
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let statuses = col
        .insert(vec![
            make_doc("positive", vec![1.0, 0.0], "value").set("key", 0.0f64),
            make_doc("negative", vec![1.0, 0.1], "value").set("key", -0.0f64),
        ])
        .unwrap();
    assert!(statuses.iter().all(|status| status.ok()));
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

use super::*;

#[test]
fn test_query_with_contain_any_and_contain_all_filters() {
    let path = temp_dir("filter_contain");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("tags", DataType::ArrayString));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![
        Doc::new("a").set("emb", vec![1.0f32, 0.0, 0.0, 0.0]).set(
            "tags",
            Value::ArrayString(vec!["red".into(), "small".into()]),
        ),
        Doc::new("b").set("emb", vec![0.9f32, 0.1, 0.0, 0.0]).set(
            "tags",
            Value::ArrayString(vec!["blue".into(), "small".into()]),
        ),
        Doc::new("c")
            .set("emb", vec![0.8f32, 0.2, 0.0, 0.0])
            .set("tags", Value::ArrayString(vec!["green".into()])),
    ])
    .unwrap();

    fn assert_contain_filters(col: &Collection) {
        let q_any = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10)
            .with_filter("tags contain_any ('red', 'blue')");
        let any_results = col.query(q_any).unwrap();
        let any_pks: std::collections::HashSet<_> =
            any_results.iter().map(|d| d.pk.as_str()).collect();
        assert_eq!(any_pks.len(), 2);
        assert!(any_pks.contains("a"));
        assert!(any_pks.contains("b"));

        let q_all = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10)
            .with_filter("tags contain_all ('blue', 'small')");
        let all_results = col.query(q_all).unwrap();
        assert_eq!(all_results.len(), 1);
        assert_eq!(all_results[0].pk, "b");

        let q_not_any = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10)
            .with_filter("tags not contain_any ('red')");
        let not_any_results = col.query(q_not_any).unwrap();
        let not_any_pks: std::collections::HashSet<_> =
            not_any_results.iter().map(|d| d.pk.as_str()).collect();
        assert_eq!(not_any_pks.len(), 2);
        assert!(not_any_pks.contains("b"));
        assert!(not_any_pks.contains("c"));
    }

    // Validate behavior in both writing segment and persisted segments.
    assert_contain_filters(&col);
    col.flush().unwrap();
    assert_contain_filters(&col);

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_array_invert_index_supports_contain_and_array_length_filters() {
    let path = temp_dir("filter_array_invert");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(
            FieldSchema::new("tags", DataType::ArrayString)
                .with_index(IndexParams::Invert(InvertIndexParams::default())),
        );

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![
        Doc::new("a").set("emb", vec![1.0f32, 0.0, 0.0, 0.0]).set(
            "tags",
            Value::ArrayString(vec!["red".into(), "small".into()]),
        ),
        Doc::new("b").set("emb", vec![0.9f32, 0.1, 0.0, 0.0]).set(
            "tags",
            Value::ArrayString(vec!["blue".into(), "small".into()]),
        ),
        Doc::new("c")
            .set("emb", vec![0.8f32, 0.2, 0.0, 0.0])
            .set("tags", Value::ArrayString(vec!["green".into()])),
    ])
    .unwrap();

    fn assert_filters(col: &Collection) {
        let q_any = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10)
            .with_filter("tags contain_any ('red', 'blue')");
        let any_results = col.query(q_any).unwrap();
        let any_pks: std::collections::HashSet<_> =
            any_results.iter().map(|d| d.pk.as_str()).collect();
        assert_eq!(any_pks.len(), 2);
        assert!(any_pks.contains("a"));
        assert!(any_pks.contains("b"));

        let q_len = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10)
            .with_filter("array_length(tags) = 2");
        let len_results = col.query(q_len).unwrap();
        let len_pks: std::collections::HashSet<_> =
            len_results.iter().map(|d| d.pk.as_str()).collect();
        assert_eq!(len_pks.len(), 2);
        assert!(len_pks.contains("a"));
        assert!(len_pks.contains("b"));

        let q_len_gt = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10)
            .with_filter("array_length(tags) > 1");
        let gt_results = col.query(q_len_gt).unwrap();
        assert_eq!(gt_results.len(), 2);
    }

    assert_filters(&col);
    col.flush().unwrap();
    assert_filters(&col);

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_query_with_contain_filters_on_uint_array_accepts_i64_input_and_persists() {
    let path = temp_dir("filter_contain_uint_array");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("nums", DataType::ArrayUint32));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![
        Doc::new("a")
            .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
            // Python/JS bindings naturally encode integer lists as i64; accept and persist as u32.
            .set("nums", Value::ArrayI64(vec![1, 2])),
        Doc::new("b")
            .set("emb", vec![0.9f32, 0.1, 0.0, 0.0])
            .set("nums", Value::ArrayI64(vec![3, 4])),
    ])
    .unwrap();

    fn assert_uint_array_filters(col: &Collection) {
        let q_any = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10)
            .with_filter("nums contain_any (2)");
        let any_results = col.query(q_any).unwrap();
        assert_eq!(any_results.len(), 1);
        assert_eq!(any_results[0].pk, "a");

        let q_all = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10)
            .with_filter("nums contain_all (3, 4)");
        let all_results = col.query(q_all).unwrap();
        assert_eq!(all_results.len(), 1);
        assert_eq!(all_results[0].pk, "b");
    }

    assert_uint_array_filters(&col);
    col.flush().unwrap();
    assert_uint_array_filters(&col);

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_query_contain_filter_list_size_bound_and_array_op_restrictions() {
    let path = temp_dir("filter_contain_bounds");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("tags", DataType::ArrayString));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![Doc::new("a")
        .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
        .set(
            "tags",
            Value::ArrayString(vec!["red".into(), "small".into()]),
        )])
    .unwrap();

    // contain list length must be <= 32.
    let mut many = Vec::new();
    for i in 0..33 {
        many.push(format!("'t{}'", i));
    }
    let filter = format!("tags contain_any ({})", many.join(", "));
    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter(&filter);
    assert!(col.query(q).is_err());

    // array fields only support contain_* ops.
    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("tags = 'red'");
    assert!(col.query(q).is_err());

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_query_array_length_validation_error_messages() {
    // array_length validation cases:
    // - ArrayLengthNonExistField
    // - ArrayLengthOnNonArrayField
    // - ArrayLengthInvalidArgument
    // - ArrayLengthInvalidOp
    let path = temp_dir("filter_array_length_validation");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("name", DataType::String))
        .with_field(FieldSchema::new("tags", DataType::ArrayString));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![Doc::new("a")
        .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
        .set("name", "alice")
        .set(
            "tags",
            Value::ArrayString(vec!["red".into(), "small".into()]),
        )])
    .unwrap();

    let err = col
        .query(
            VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10)
                .with_filter("array_length(not_exist_field) > 1"),
        )
        .unwrap_err();
    assert!(err
        .message()
        .contains("array_length argument not found in schema"));

    let err = col
        .query(
            VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10)
                .with_filter("array_length(name) > 1"),
        )
        .unwrap_err();
    assert!(err.message().contains("array_length only support array"));

    let err = col
        .query(
            VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10)
                .with_filter("array_length(tags) > '1'"),
        )
        .unwrap_err();
    assert!(err
        .message()
        .contains("array_length right side only support integer"));

    // array_length only accepts integer literals (not integer-valued floats).
    let err = col
        .query(
            VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10)
                .with_filter("array_length(tags) = 1.0"),
        )
        .unwrap_err();
    assert!(err
        .message()
        .contains("array_length right side only support integer"));

    // array_length compares are u32-like; negatives are rejected.
    let err = col
        .query(
            VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10)
                .with_filter("array_length(tags) = -1"),
        )
        .unwrap_err();
    assert!(err
        .message()
        .contains("array_length right side only support integer"));

    // array_length RHS must be an integer literal token (functions are rejected).
    let err = col
        .query(
            VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10)
                .with_filter("array_length(tags) > array_length(tags)"),
        )
        .unwrap_err();
    assert!(err
        .message()
        .contains("array_length right side only support integer"));

    let err = col
        .query(
            VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10)
                .with_filter("array_length(tags) like '%'"),
        )
        .unwrap_err();
    assert!(err.message().contains("syntax error"));

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

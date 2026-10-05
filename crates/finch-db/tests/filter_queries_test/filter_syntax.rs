use super::*;

#[test]
fn test_query_with_eq_filter() {
    let path = temp_dir("filter_eq");
    let schema = basic_schema(4);
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![
        make_doc("d1", vec![1.0, 0.0, 0.0, 0.0], "cat"),
        make_doc("d2", vec![0.9, 0.1, 0.0, 0.0], "dog"),
        make_doc("d3", vec![0.8, 0.2, 0.0, 0.0], "cat"),
    ])
    .unwrap();

    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("label = 'cat'");
    let results = col.query(q).unwrap();

    assert!(!results.is_empty());
    for doc in &results {
        assert_eq!(doc.get_str("label").unwrap(), "cat");
    }
    // Should not include "dog"
    assert!(results.iter().all(|d| d.pk != "d2"));

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_query_with_gt_filter() {
    let path = temp_dir("filter_gt");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("score", DataType::Float32));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![
        Doc::new("a")
            .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
            .set("score", 10.0f32),
        Doc::new("b")
            .set("emb", vec![0.9f32, 0.1, 0.0, 0.0])
            .set("score", 5.0f32),
        Doc::new("c")
            .set("emb", vec![0.8f32, 0.2, 0.0, 0.0])
            .set("score", 2.0f32),
    ])
    .unwrap();

    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("score > 4.0");
    let results = col.query(q).unwrap();

    assert!(results.len() <= 2);
    for doc in &results {
        let score = doc.get_f32("score").unwrap().unwrap();
        assert!(score > 4.0, "score {} should be > 4.0", score);
    }

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_query_with_and_filter() {
    let path = temp_dir("filter_and");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("score", DataType::Float32))
        .with_field(FieldSchema::new("label", DataType::String));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![
        Doc::new("a")
            .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
            .set("score", 10.0f32)
            .set("label", "good"),
        Doc::new("b")
            .set("emb", vec![0.9f32, 0.1, 0.0, 0.0])
            .set("score", 10.0f32)
            .set("label", "bad"),
        Doc::new("c")
            .set("emb", vec![0.8f32, 0.2, 0.0, 0.0])
            .set("score", 2.0f32)
            .set("label", "good"),
    ])
    .unwrap();

    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10)
        .with_filter("score > 5.0 AND label = 'good'");
    let results = col.query(q).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].pk, "a");

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_query_with_in_and_not_in_filter() {
    let path = temp_dir("filter_in");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("rank", DataType::Int32));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![
        Doc::new("a")
            .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
            .set("rank", 1i32),
        Doc::new("b")
            .set("emb", vec![0.9f32, 0.1, 0.0, 0.0])
            .set("rank", 2i32),
        Doc::new("c")
            .set("emb", vec![0.8f32, 0.2, 0.0, 0.0])
            .set("rank", 3i32),
        Doc::new("d")
            .set("emb", vec![0.7f32, 0.3, 0.0, 0.0])
            .set("rank", 4i32),
    ])
    .unwrap();

    let q_in = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("rank IN (1, 3)");
    let in_results = col.query(q_in).unwrap();
    assert_eq!(in_results.len(), 2);
    for doc in &in_results {
        let rank = doc.get_f32("rank").unwrap().unwrap() as i32;
        assert!(rank == 1 || rank == 3, "rank {} must match IN list", rank);
    }

    let q_not_in =
        VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("rank NOT IN (1, 3)");
    let not_in_results = col.query(q_not_in).unwrap();
    assert_eq!(not_in_results.len(), 2);
    for doc in &not_in_results {
        let rank = doc.get_f32("rank").unwrap().unwrap() as i32;
        assert!(
            rank == 2 || rank == 4,
            "rank {} must be excluded from IN list",
            rank
        );
    }

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_query_invalid_filter_unknown_field_errors() {
    let path = temp_dir("filter_unknown_field");
    let schema = basic_schema(4);
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![make_doc("d1", vec![1.0, 0.0, 0.0, 0.0], "x")])
        .unwrap();

    let q =
        VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("nonexistent_field = 5");
    let err = col.query(q).unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument);
    assert!(
        err.message.contains("unknown field"),
        "expected unknown-field error, got: {}",
        err.message
    );

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_query_invalid_filter_parse_errors_are_wrapped() {
    let path = temp_dir("filter_parse_error_wrapped");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("age", DataType::Int32));
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![Doc::new("d1")
        .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
        .set("age", 1i32)])
        .unwrap();

    // invalid filter strings fail parsing with `Invalid filter: ...`.
    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("age =");
    let err = col.query(q).unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument);
    assert!(
        err.message.contains("Invalid filter:"),
        "got: {}",
        err.message
    );
    assert!(err.message.contains("syntax error"), "got: {}", err.message);

    // delete_by_filter uses the same wrapper.
    let err = col.delete_by_filter("age =").unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument);
    assert!(
        err.message.contains("Invalid filter:"),
        "got: {}",
        err.message
    );
    assert!(err.message.contains("syntax error"), "got: {}", err.message);

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_query_invalid_filter_sql_standard_quote_doubling_is_rejected() {
    let path = temp_dir("filter_quote_doubling_rejected");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(1))
        .with_field(FieldSchema::new("name", DataType::String));
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![Doc::new("d1")
        .set("emb", vec![1.0f32])
        .set("name", "it's")])
        .unwrap();

    // 'it''s' is not a valid string literal; only backslash escaping is.
    let q = VectorQuery::new("emb", vec![1.0], 10).with_filter("name = 'it''s'");
    let err = col.query(q).unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument);
    assert!(
        err.message.contains("Invalid filter:"),
        "got: {}",
        err.message
    );
    assert!(err.message.contains("syntax error"), "got: {}", err.message);

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_query_filter_comments_are_ignored() {
    let path = temp_dir("filter_comments_ignored");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(1))
        .with_field(FieldSchema::new("age", DataType::Int32))
        .with_field(FieldSchema::new("name", DataType::String));
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![
        Doc::new("d1")
            .set("emb", vec![1.0f32])
            .set("age", 1i32)
            .set("name", "x"),
        Doc::new("d2")
            .set("emb", vec![1.0f32])
            .set("age", 2i32)
            .set("name", "y"),
    ])
    .unwrap();

    // Comments are ignored; unsupported tokens inside comments must not
    // trigger filter parse failures.
    let q = VectorQuery::new("emb", vec![1.0], 10)
        .with_filter("age = 1 -- comment includes == and `backticks`\nAND name = 'x'");
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].pk, "d1");

    let q = VectorQuery::new("emb", vec![1.0], 10)
        .with_filter("age = 2 /* comment includes == and `backticks` */ AND name = 'y'");
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].pk, "d2");

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_query_invalid_filter_type_mismatch_errors() {
    let path = temp_dir("filter_type_mismatch");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("score", DataType::Float32));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![Doc::new("d1")
        .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
        .set("score", 10.0f32)])
        .unwrap();

    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("score = 'high'");
    let err = col.query(q).unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument);
    assert!(
        err.message.contains("type mismatch"),
        "expected type-mismatch error, got: {}",
        err.message
    );

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_query_invalid_filter_integer_fields_reject_float_literals() {
    let path = temp_dir("filter_int_reject_float");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("age", DataType::Uint32))
        .with_field(FieldSchema::new("nums", DataType::ArrayInt32));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![Doc::new("d1")
        .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
        .set("age", 1u32)
        .set("nums", vec![1i32, 2, 3])])
        .unwrap();

    // Float literals are rejected for integer fields.
    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("age = 1.1");
    let err = col.query(q).unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument);
    assert!(
        err.message.contains("type mismatch"),
        "expected type-mismatch error, got: {}",
        err.message
    );

    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("age IN (1.1, 2, 3)");
    let err = col.query(q).unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument);
    assert!(
        err.message.contains("type mismatch"),
        "expected type-mismatch error, got: {}",
        err.message
    );

    let q =
        VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("nums contain_any (1.1)");
    let err = col.query(q).unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument);
    assert!(
        err.message.contains("incompatible") || err.message.contains("type mismatch"),
        "expected type-mismatch error, got: {}",
        err.message
    );

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_query_invalid_filter_bool_op_errors() {
    let path = temp_dir("filter_bool_op");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("flag", DataType::Bool));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![Doc::new("d1")
        .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
        .set("flag", true)])
        .unwrap();

    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("flag > true");
    let err = col.query(q).unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument);
    assert!(
        err.message.contains("bool type") || err.message.contains("bool"),
        "expected bool-op error, got: {}",
        err.message
    );

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_query_bool_in_list_is_supported() {
    let path = temp_dir("filter_bool_in");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("flag", DataType::Bool));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![
        Doc::new("t")
            .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
            .set("flag", true),
        Doc::new("f")
            .set("emb", vec![0.9f32, 0.1, 0.0, 0.0])
            .set("flag", false),
        Doc::new("n").set("emb", vec![0.8f32, 0.2, 0.0, 0.0]), // missing flag => NULL
    ])
    .unwrap();

    // `x IN (true,false)` behaves like `x IS NOT NULL`.
    let q =
        VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("flag IN (true, false)");
    let results = col.query(q).unwrap();
    let pks: std::collections::HashSet<_> = results.into_iter().map(|d| d.pk.clone()).collect();
    assert!(pks.contains("t"));
    assert!(pks.contains("f"));
    assert!(!pks.contains("n"));

    // NULL does not match NOT IN.
    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10)
        .with_filter("flag NOT IN (true, false)");
    let results = col.query(q).unwrap();
    assert!(results.is_empty());

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_query_filter_unsupported_sql_constructs_are_syntax_errors() {
    let path = temp_dir("filter_unsupported_sql_constructs");
    let schema = CollectionSchema::new("test")
        .with_field(
            FieldSchema::new("dummy_vec", DataType::VectorFp32)
                .nullable()
                .with_dimension(1),
        )
        .with_field(FieldSchema::new("name", DataType::String).nullable())
        .with_field(FieldSchema::new("age", DataType::Int32).nullable());

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    // BETWEEN is not part of the filter grammar.
    let q = VectorQuery::new("", vec![], 10)
        .with_filter("age BETWEEN 1 AND 2")
        .with_output_fields(vec![]);
    let err = col.query(q).unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument);
    assert!(err.message.contains("syntax error"));

    // string concatenation operator `||` is not part of the filter grammar.
    let q = VectorQuery::new("", vec![], 10)
        .with_filter("name = 'a' || 'b'")
        .with_output_fields(vec![]);
    let err = col.query(q).unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument);
    assert!(err.message.contains("syntax error"));

    // compound identifiers (with dot) are rejected.
    let q = VectorQuery::new("", vec![], 10)
        .with_filter("t.name = 'x'")
        .with_output_fields(vec![]);
    let err = col.query(q).unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument);
    assert!(err.message.contains("syntax error"));

    // identifier quoting is not supported.
    let q = VectorQuery::new("", vec![], 10)
        .with_filter("`name` = 'x'")
        .with_output_fields(vec![]);
    let err = col.query(q).unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument);
    assert!(err.message.contains("syntax error"));

    // `==` is not a valid equality operator.
    let q = VectorQuery::new("", vec![], 10)
        .with_filter("age == 1")
        .with_output_fields(vec![]);
    let err = col.query(q).unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument);
    assert!(err.message.contains("syntax error"));

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_filter_numeric_literals_are_normalized_no_f64_rounding() {
    let path = temp_dir("filter_normalize_u64");

    let schema = CollectionSchema::new("test")
        .with_field(
            FieldSchema::new("dummy_vec", DataType::VectorFp32)
                .nullable()
                .with_dimension(1),
        )
        .with_field(FieldSchema::new("x", DataType::Uint64).nullable());
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    // Values around 2^53 to catch precision loss when comparing via f64.
    let a = 9_007_199_254_740_992u64; // 2^53
    let b = 9_007_199_254_740_993u64; // 2^53 + 1 (not exactly representable in f64)
    let c = 9_007_199_254_740_994u64;

    let statuses = col
        .insert(vec![
            Doc::new("a").set("x", a),
            Doc::new("b").set("x", b),
            Doc::new("c").set("x", c),
        ])
        .unwrap();
    assert!(statuses.iter().all(|s| s.is_ok()));

    fn pks(col: &Collection, filter: &str) -> Vec<String> {
        let q = VectorQuery::new("", vec![], 10)
            .with_filter(filter)
            .with_output_fields(vec![]);
        let mut out: Vec<String> = col.query(q).unwrap().iter().map(|d| d.pk.clone()).collect();
        out.sort();
        out
    }

    assert_eq!(pks(&col, "x = 9007199254740993"), vec!["b".to_string()]);

    col.flush().unwrap();
    assert_eq!(pks(&col, "x = 9007199254740993"), vec!["b".to_string()]);

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_filter_only_query_double_quoted_strings_work() {
    let path = temp_dir("filter_only_double_quote");
    let schema = CollectionSchema::new("test")
        .with_field(
            FieldSchema::new("dummy_vec", DataType::VectorFp32)
                .nullable()
                .with_dimension(1),
        )
        .with_field(
            FieldSchema::new("name", DataType::String)
                .nullable()
                .not_null(),
        );

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![
        Doc::new("pk_0").set("name", "user_0"),
        Doc::new("pk_1").set("name", "user_1"),
        Doc::new("pk_2").set("name", "user_2"),
        Doc::new("pk_3").set("name", "user_3"),
    ])
    .unwrap();
    col.flush().unwrap();

    let q = VectorQuery::new("", vec![], 10)
        .with_filter(r#"name IN ("user_1", "user_3")"#)
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    let got: std::collections::HashSet<_> = docs.iter().map(|d| d.pk.as_str()).collect();
    assert_eq!(got.len(), 2);
    assert!(got.contains("pk_1"));
    assert!(got.contains("pk_3"));

    let q = VectorQuery::new("", vec![], 10)
        .with_filter(r#"name = "user_2""#)
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].pk, "pk_2");

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_query_relation_expr_shape_validation() {
    let path = temp_dir("filter_relation_expr_shape_validation");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("rank", DataType::Int32))
        .with_field(FieldSchema::new("name", DataType::String));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![Doc::new("a")
        .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
        .set("rank", 1i32)
        .set("name", "alice")])
        .unwrap();

    // relation expr requires identifier on LHS (no `value = field`).
    let err = col
        .query(VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("1 = rank"))
        .unwrap_err();
    assert!(err.message().contains("syntax error"));

    // RHS cannot be another field identifier (value_expr is CONST or FUNC).
    let err = col
        .query(VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("rank = name"))
        .unwrap_err();
    assert!(err.message().contains("syntax error"));

    // arithmetic operators are not supported in the filter grammar.
    let err = col
        .query(VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("rank = rank + 1"))
        .unwrap_err();
    assert!(err.message().contains("syntax error"));

    // `+` is not part of numeric tokens in the filter grammar.
    let err = col
        .query(VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("rank = +1"))
        .unwrap_err();
    assert!(err.message().contains("syntax error"));

    // `- 1` is a syntax error (negative numbers must be a single token: `-1`).
    let err = col
        .query(VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("rank = - 1"))
        .unwrap_err();
    assert!(err.message().contains("syntax error"));

    // parentheses are not allowed around values/identifiers in relation exprs.
    let err = col
        .query(VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("rank = (1)"))
        .unwrap_err();
    assert!(err.message().contains("syntax error"));

    let err = col
        .query(VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("(rank) = 1"))
        .unwrap_err();
    assert!(err.message().contains("syntax error"));

    // `-1` is accepted as an integer literal token (even if it doesn't match any rows).
    let results = col
        .query(VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("rank = -1"))
        .unwrap();
    assert!(results.is_empty());

    // only array_length(...) is supported as a function in relation exprs.
    let err = col
        .query(VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("abs(rank) = 1"))
        .unwrap_err();
    assert!(err.message().contains("Function is not supported. abs"));

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_query_filter_scientific_notation_float_literals_are_supported() {
    let path = temp_dir("filter_scientific_notation_floats");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("score", DataType::Float64));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![Doc::new("a")
        .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
        .set("score", 1000.0f64)])
        .unwrap();

    for filter in ["score = 1e3", "score = 1e+3", "score = 1E3", "score = 1E+3"] {
        let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter(filter);
        let out = col.query(q).unwrap();
        assert_eq!(out.len(), 1, "filter={filter}");
        assert_eq!(out[0].pk, "a");
    }

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_query_filter_null_literal_and_empty_in_list_are_syntax_errors() {
    let path = temp_dir("filter_null_and_empty_in");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("rank", DataType::Int32));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![Doc::new("a")
        .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
        .set("rank", 1i32)])
        .unwrap();

    // NULL is not a valid constant in relation expressions.
    let err = col
        .query(VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("rank = NULL"))
        .unwrap_err();
    assert!(err.message().contains("syntax error"));

    let err = col
        .query(VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("rank IN (NULL)"))
        .unwrap_err();
    assert!(err.message().contains("syntax error"));

    // `IN ()` is a syntax error.
    let err = col
        .query(VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("rank IN ()"))
        .unwrap_err();
    assert!(err.message().contains("syntax error"));

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_query_with_like_wildcards() {
    let path = temp_dir("filter_like");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("name", DataType::String));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![
        Doc::new("a")
            .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
            .set("name", "alice"),
        Doc::new("b")
            .set("emb", vec![0.9f32, 0.1, 0.0, 0.0])
            .set("name", "alix"),
        Doc::new("c")
            .set("emb", vec![0.8f32, 0.2, 0.0, 0.0])
            .set("name", "ali1"),
        Doc::new("d")
            .set("emb", vec![0.7f32, 0.3, 0.0, 0.0])
            .set("name", "bob"),
    ])
    .unwrap();

    let q_prefix =
        VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("name LIKE 'ali%'");
    let prefix_results = col.query(q_prefix).unwrap();
    assert_eq!(prefix_results.len(), 3);

    let q_suffix =
        VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("name LIKE '%ix'");
    let suffix_results = col.query(q_suffix).unwrap();
    assert_eq!(suffix_results.len(), 1);
    assert_eq!(suffix_results[0].pk, "b");

    let q_single_char =
        VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("name LIKE 'ali_'");
    let single_char_results = col.query(q_single_char).unwrap();
    let single_char_pks: std::collections::HashSet<_> =
        single_char_results.iter().map(|d| d.pk.as_str()).collect();
    assert_eq!(single_char_pks.len(), 2);
    assert!(single_char_pks.contains("b"));
    assert!(single_char_pks.contains("c"));

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_contain_operator_text_inside_string_literals_is_not_rewritten() {
    let path = temp_dir("contain_text_in_literal");
    let col =
        Collection::create_and_open(&path, basic_schema(4), CollectionOptions::default()).unwrap();
    let label = "tags CONTAIN_ANY (x)";
    col.insert(vec![
        make_doc("literal", vec![1.0, 0.0, 0.0, 0.0], label),
        make_doc("other", vec![0.0, 1.0, 0.0, 0.0], "other"),
    ])
    .unwrap();
    let filter = format!("label = '{label}'");

    let hits = col
        .query(VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter(filter.clone()))
        .unwrap();
    assert_eq!(
        hits.iter().map(|d| d.pk.as_str()).collect::<Vec<_>>(),
        vec!["literal"]
    );

    col.delete_by_filter(&filter).unwrap();
    let remaining = col
        .fetch(vec!["literal".to_string(), "other".to_string()])
        .unwrap();
    drop(col);
    std::fs::remove_dir_all(&path).ok();
    assert!(!remaining.contains_key("literal"));
    assert!(remaining.contains_key("other"));
}

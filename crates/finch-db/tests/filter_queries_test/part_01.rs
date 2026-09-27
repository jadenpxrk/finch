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
fn test_query_radius_keeps_only_results_within_distance() {
    let path = temp_dir("query_radius");
    let schema = CollectionSchema::new("test").with_field(
        FieldSchema::new("emb", DataType::VectorFp32)
            .with_dimension(1)
            .with_index(IndexParams::Flat(FlatIndexParams {
                metric: MetricType::L2,
                quantize: QuantizeType::Undefined,
                column_major: false,
            })),
    );

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    col.insert(vec![
        Doc::new("a").set("emb", vec![0.0f32]),
        Doc::new("b").set("emb", vec![0.4f32]), // dist=0.16 from query [0]
        Doc::new("c").set("emb", vec![0.6f32]), // dist=0.36
        Doc::new("d").set("emb", vec![1.0f32]), // dist=1.0
    ])
    .unwrap();

    let qp = QueryParams {
        radius: Some(0.25), // include only dist <= 0.25 => {a,b}
        ..QueryParams::default()
    };

    let q = VectorQuery::new("emb", vec![0.0f32], 10).with_params(qp);
    let mut pks: Vec<String> = col.query(q).unwrap().iter().map(|d| d.pk.clone()).collect();
    pks.sort();
    assert_eq!(pks, vec!["a".to_string(), "b".to_string()]);

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

    // reference lexer parity: 'it''s' is not a valid string literal (only backslash escaping).
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

    // reference lexer parity: comments are ignored; unsupported tokens inside comments must not
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

    // reference rejects float literals for integer fields.
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

    // SQL-ish parity: `x IN (true,false)` behaves like `x IS NOT NULL`.
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
fn test_query_binary_and_array_binary_filters_accept_string_literals() {
    let path = temp_dir("filter_binary_literals");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("blob", DataType::Binary))
        .with_field(FieldSchema::new("blobs", DataType::ArrayBinary));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![
        Doc::new("a")
            .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
            .set("blob", Value::Bytes(b"abc".to_vec()))
            .set(
                "blobs",
                Value::ArrayBinary(vec![b"red".to_vec(), b"blue".to_vec()]),
            ),
        Doc::new("b")
            .set("emb", vec![0.9f32, 0.1, 0.0, 0.0])
            .set("blob", Value::Bytes(b"xyz".to_vec()))
            .set("blobs", Value::ArrayBinary(vec![b"green".to_vec()])),
    ])
    .unwrap();

    // BINARY/ARRAY_BINARY use string literals in filters (treated as raw bytes).
    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("blob = 'abc'");
    let r = col.query(q).unwrap();
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].pk, "a");

    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10)
        .with_filter("blobs contain_any ('blue')");
    let r = col.query(q).unwrap();
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].pk, "a");

    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10)
        .with_filter("blobs contain_all ('red', 'blue')");
    let r = col.query(q).unwrap();
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].pk, "a");

    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10)
        .with_filter("blobs contain_any ('missing')");
    let r = col.query(q).unwrap();
    assert!(r.is_empty());

    // `IN (...)` list-values are only supported for string/numeric/bool,
    // not for BINARY.
    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("blob IN ('abc')");
    assert!(col.query(q).is_err());

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_binary_fields_accept_string_values_on_write_and_round_trip() {
    let path = temp_dir("binary_write_round_trip");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("blob", DataType::Binary))
        .with_field(FieldSchema::new("blobs", DataType::ArrayBinary));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    // BINARY/ARRAY_BINARY setters accept string payloads.
    col.insert(vec![Doc::new("a")
        .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
        .set("blob", "abc")
        .set(
            "blobs",
            Value::ArrayString(vec!["red".into(), "blue".into()]),
        )])
    .unwrap();

    fn assert_round_trip(col: &Collection) {
        let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10)
            .with_output_fields(vec!["blob".into(), "blobs".into()]);
        let r = col.query(q).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].pk, "a");
        match r[0].fields.get("blob") {
            Some(Value::Bytes(b)) => assert_eq!(b.as_slice(), b"abc"),
            other => panic!("expected blob bytes, got: {:?}", other),
        }
        match r[0].fields.get("blobs") {
            Some(Value::ArrayBinary(items)) => {
                let as_slices: Vec<&[u8]> = items.iter().map(|v| v.as_slice()).collect();
                assert_eq!(as_slices, vec![b"red".as_slice(), b"blue".as_slice()]);
            }
            other => panic!("expected blobs array_binary, got: {:?}", other),
        }
    }

    assert_round_trip(&col);
    col.flush().unwrap();
    assert_round_trip(&col);

    drop(col);
    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    assert_round_trip(&col);

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

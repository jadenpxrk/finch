use super::*;

#[test]
fn test_sql_accepts_keyword_field_names_and_scalar_aliases() {
    let path = temp_dir("sql_keyword_ident_alias");
    let schema = CollectionSchema::new("test")
        .with_field(
            FieldSchema::new("emb", DataType::VectorFp32)
                .nullable()
                .with_dimension(4),
        )
        .with_field(FieldSchema::new("or", DataType::Int64).nullable())
        .with_field(FieldSchema::new("label", DataType::String).nullable());

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let statuses = col
        .insert(vec![
            Doc::new("d0")
                .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
                .set("or", 1i64)
                .set("label", "d0"),
            Doc::new("d1")
                .set("emb", vec![0.0f32, 1.0, 0.0, 0.0])
                .set("or", 2i64)
                .set("label", "d1"),
        ])
        .unwrap();
    assert!(
        statuses.iter().all(|s| s.is_ok()),
        "expected insert to succeed, got statuses: {statuses:?}"
    );
    col.flush().unwrap();

    let fetched = col.fetch(vec!["d1".to_string()]).unwrap();
    let d1 = fetched.get("d1").expect("d1 should exist");
    assert!(
        matches!(d1.fields.get("or"), Some(Value::I64(2))),
        "d1={d1:?}"
    );

    // Sanity-check persisted-segment filtering for keyword field names.
    let direct = col
        .query(
            VectorQuery::new("", Vec::<f32>::new(), 10)
                .with_filter("or = 2")
                .with_output_fields(vec!["or".to_string()]),
        )
        .unwrap();
    assert_eq!(direct.len(), 1);
    assert!(matches!(direct[0].fields.get("or"), Some(Value::I64(2))));

    // Keyword identifier ("or") should be usable as a field name.
    let res = col
        .query_sql("SELECT or FROM test WHERE or = 2 LIMIT 10")
        .unwrap();
    assert_eq!(res.len(), 1);
    assert!(matches!(res[0].fields.get("or"), Some(Value::I64(2))));

    // Scalar alias should rename the output field when not selecting '*'.
    let res = col
        .query_sql("SELECT label AS l FROM test WHERE or = 1 LIMIT 1")
        .unwrap();
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].fields.get("l").and_then(|v| v.as_str()), Some("d0"));
    assert!(!res[0].fields.contains_key("label"));

    // Asterisk + alias should keep the original column and add the alias.
    let res = col
        .query_sql("SELECT *, label AS l FROM test WHERE or = 1 LIMIT 1")
        .unwrap();
    assert_eq!(res.len(), 1);
    assert_eq!(
        res[0].fields.get("label").and_then(|v| v.as_str()),
        Some("d0")
    );
    assert_eq!(res[0].fields.get("l").and_then(|v| v.as_str()), Some("d0"));

    // Vector alias should rename the vector output field.
    let res = col
        .query_sql("SELECT emb AS e FROM test WHERE _finch_uid_ = 'd0' LIMIT 1")
        .unwrap();
    assert_eq!(res.len(), 1);
    assert!(!res[0].fields.contains_key("emb"));
    assert!(matches!(res[0].fields.get("e"), Some(Value::VecF32(v)) if v.len() == 4));

    // System column alias.
    let res = col
        .query_sql("SELECT _finch_uid_ AS u FROM test WHERE _finch_uid_ = 'd1' LIMIT 1")
        .unwrap();
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].fields.get("u").and_then(|v| v.as_str()), Some("d1"));
    assert!(!res[0].fields.contains_key(SYS_USER_ID));

    // Table name mismatch should be rejected.
    let err = col
        .query_sql("SELECT * FROM other WHERE or = 1 LIMIT 1")
        .unwrap_err();
    assert!(err.code == StatusCode::InvalidArgument);
    assert!(err.message.contains("table not found"), "err={err:?}");

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_sql_and_filter_accept_leading_digit_and_dash_identifiers() {
    // REGULAR_ID allows identifiers starting with digits and containing '-'.
    // Field names such as `1collection` and `1-dash_score_field` must work in filters.
    let path = temp_dir("sql_leading_digit_dash_ident");
    let schema = CollectionSchema::new("1collection")
        .with_field(FieldSchema::new("name", DataType::Uint32).nullable())
        .with_field(FieldSchema::new("1-dash_score_field", DataType::String).nullable())
        .with_field(
            FieldSchema::new("emb", DataType::VectorFp32)
                .nullable()
                .with_dimension(2),
        );

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let statuses = col
        .insert(vec![
            Doc::new("d0")
                .set("name", 1u32)
                .set("1-dash_score_field", "nope")
                .set("emb", vec![0.0f32, 0.0]),
            Doc::new("d1")
                .set("name", 4u32)
                .set("1-dash_score_field", "test")
                .set("emb", vec![0.0f32, 0.0]),
            Doc::new("d2")
                .set("name", 10u32)
                .set("1-dash_score_field", "nope")
                .set("emb", vec![0.0f32, 0.0]),
        ])
        .unwrap();
    assert!(statuses.iter().all(|s| s.is_ok()));
    col.flush().unwrap();

    // Filter parsing: ensure the dash identifier is treated as a field, not arithmetic.
    let docs = col
        .query(
            VectorQuery::new("", Vec::<f32>::new(), 10)
                .with_filter("name < 3 OR name = 4 OR 1-dash_score_field = 'test'"),
        )
        .unwrap();
    assert_eq!(docs.len(), 2);
    assert_eq!(docs[0].pk, "d0");
    assert_eq!(docs[1].pk, "d1");

    // SQL parsing: unquoted leading-digit/dash field name in SELECT/WHERE.
    let res = col
        .query_sql("SELECT 1-dash_score_field FROM 1collection WHERE 1-dash_score_field = \"test\" LIMIT 10")
        .unwrap();
    assert_eq!(res.len(), 1);
    assert_eq!(
        res[0]
            .fields
            .get("1-dash_score_field")
            .and_then(|v| v.as_str()),
        Some("test")
    );

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_sql_order_by_sorts_and_does_not_leak_hidden_fetch_fields() {
    let path = temp_dir("sql_order_by");
    let schema = CollectionSchema::new("test")
        .with_field(
            FieldSchema::new("emb", DataType::VectorFp32)
                .nullable()
                .with_dimension(4),
        )
        .with_field(FieldSchema::new("n", DataType::Int64).nullable())
        .with_field(FieldSchema::new("label", DataType::String).nullable());

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let statuses = col
        .insert(vec![
            Doc::new("d0")
                .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
                .set("n", 2i64)
                .set("label", "b"),
            Doc::new("d1")
                .set("emb", vec![0.0f32, 1.0, 0.0, 0.0])
                .set("n", 1i64)
                .set("label", "a"),
            // Missing `n` should sort as NULL (last).
            Doc::new("d2")
                .set("emb", vec![0.0f32, 0.0, 1.0, 0.0])
                .set("label", "c"),
        ])
        .unwrap();
    assert!(statuses.iter().all(|s| s.is_ok()));
    col.flush().unwrap();

    let res = col
        .query_sql("SELECT label FROM test ORDER BY n ASC LIMIT 10")
        .unwrap();
    assert_eq!(
        res.iter().map(|d| d.pk.as_str()).collect::<Vec<_>>(),
        vec!["d1", "d0", "d2"]
    );
    // Hidden ORDER BY field should not appear in output when not selected.
    for d in &res {
        assert!(!d.fields.contains_key("n"), "unexpected n leaked: {d:?}");
    }

    let res = col
        .query_sql("SELECT label FROM test ORDER BY n DESC LIMIT 10")
        .unwrap();
    assert_eq!(
        res.iter().map(|d| d.pk.as_str()).collect::<Vec<_>>(),
        vec!["d0", "d1", "d2"]
    );

    let res = col
        .query_sql("SELECT _finch_uid_ FROM test ORDER BY _finch_g_doc_id_ DESC LIMIT 10")
        .unwrap();
    assert_eq!(
        res.iter().map(|d| d.pk.as_str()).collect::<Vec<_>>(),
        vec!["d2", "d1", "d0"]
    );

    // ORDER BY vector fields should be rejected.
    let err = col
        .query_sql("SELECT * FROM test ORDER BY emb ASC LIMIT 10")
        .unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument, "err={err:?}");

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_sql_score_virtual_column_and_limit_signed_int() {
    let path = temp_dir("sql_score_and_limit");
    let schema = CollectionSchema::new("test")
        .with_field(
            FieldSchema::new("emb", DataType::VectorFp32)
                .with_dimension(4)
                .with_index(IndexParams::Flat(FlatIndexParams::new(MetricType::L2))),
        )
        .with_field(FieldSchema::new("label", DataType::String));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let statuses = col
        .insert(vec![
            Doc::new("d0")
                .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
                .set("label", "d0"),
            Doc::new("d1")
                .set("emb", vec![0.0f32, 1.0, 0.0, 0.0])
                .set("label", "d1"),
            Doc::new("d2")
                .set("emb", vec![10.0f32, 0.0, 0.0, 0.0])
                .set("label", "d2"),
        ])
        .unwrap();
    assert!(statuses.iter().all(|s| s.is_ok()));
    col.flush().unwrap();

    // An explicit LIMIT 0 returns nothing, a negative LIMIT is rejected, and no LIMIT uses the default.
    let res = col.query_sql("SELECT label FROM test LIMIT 0").unwrap();
    assert!(res.is_empty());
    assert!(col.query_sql("SELECT label FROM test LIMIT -1").is_err());
    let res = col.query_sql("SELECT label FROM test").unwrap();
    assert_eq!(res.len(), 3);

    // `_finch_score` is a virtual output column (backed by Doc.score).
    let res = col
        .query_sql("SELECT _finch_uid_, _finch_score AS s FROM test WHERE emb = [1,0,0,0] ORDER BY _finch_score ASC LIMIT 3")
        .unwrap();
    assert_eq!(res.len(), 3);
    assert_eq!(res[0].pk, "d0"); // exact match should rank first
    let scores: Vec<f32> = res
        .iter()
        .map(|d| match d.fields.get("s") {
            Some(Value::F32(x)) => *x,
            other => panic!("expected s to be F32, got {other:?}"),
        })
        .collect();
    assert!(
        scores[0] <= scores[1] && scores[1] <= scores[2],
        "scores={scores:?}"
    );

    let res = col
        .query_sql(
            "SELECT _finch_uid_ FROM test WHERE emb = [1,0,0,0] ORDER BY _finch_score DESC LIMIT 3",
        )
        .unwrap();
    assert_eq!(res.len(), 3);
    assert_eq!(res[0].pk, "d2"); // farthest should rank first with DESC

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_sql_matrix_vector_literal_is_flattened_into_a_single_query_vector() {
    let path = temp_dir("sql_matrix_vec_flatten");
    let schema = basic_schema(8);

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    col.insert(vec![
        make_doc("d0", vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], "d0"),
        make_doc("d1", vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], "d1"),
    ])
    .unwrap();
    col.flush().unwrap();

    // Single-row matrix is accepted (equivalent to a plain vector literal).
    let res = col
        .query_sql("SELECT _finch_uid_ FROM test WHERE emb = [[1,0,0,0,0,1,0,0]] LIMIT 1")
        .unwrap();
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].pk, "d0");

    // Multi-row matrix literals are flattened by concatenation.
    let res = col
        .query_sql("SELECT _finch_uid_ FROM test WHERE emb = [[1,0,0,0],[0,1,0,0]] LIMIT 1")
        .unwrap();
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].pk, "d0");

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_sql_vector_condition_validation_and_vector_literal_lexing() {
    let path = temp_dir("sql_vec_validation");
    let schema = basic_schema(4);

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    col.insert(vec![
        make_doc("d0", vec![1.0, 0.0, 0.0, 0.0], "d0"),
        make_doc("d1", vec![0.0, 1.0, 0.0, 0.0], "d1"),
    ])
    .unwrap();
    col.flush().unwrap();

    // Vector fields only support '=' in the SQL WHERE clause.
    let err = col
        .query_sql("SELECT _finch_uid_ FROM test WHERE emb != [1,0,0,0] LIMIT 1")
        .unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument, "err={err:?}");
    assert!(
        err.message.contains("vector field only support EQ"),
        "err={err:?}"
    );

    // Vector fields must compare against a VECTOR/MATRIX literal (not a scalar).
    let err = col
        .query_sql("SELECT _finch_uid_ FROM test WHERE emb = 1 LIMIT 1")
        .unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument, "err={err:?}");
    assert!(
        err.message.contains("invalid vector value node"),
        "err={err:?}"
    );

    // Non-vector fields cannot be compared to vector literals.
    let err = col
        .query_sql("SELECT _finch_uid_ FROM test WHERE label = [1,0,0,0] LIMIT 1")
        .unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument, "err={err:?}");
    assert!(
        err.message.contains("field type and value type not match"),
        "err={err:?}"
    );

    // Schema-free fields cannot be compared against vectors.
    let err = col
        .query_sql("SELECT _finch_uid_ FROM test WHERE unknown = [1,0,0,0] LIMIT 1")
        .unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument, "err={err:?}");
    assert!(
        err.message
            .contains("vector vector not supported for schema free field"),
        "err={err:?}"
    );

    // The VECTOR token does not allow `1.` (digits required after '.').
    let err = col
        .query_sql("SELECT _finch_uid_ FROM test WHERE emb = [1.,0,0,0] LIMIT 1")
        .unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument, "err={err:?}");
    assert!(err.message.contains("vector format error"), "err={err:?}");

    // The VECTOR token allows leading-dot floats like `.1`.
    let res = col
        .query_sql("SELECT _finch_uid_ FROM test WHERE emb = [.1,0,0,0] LIMIT 1")
        .unwrap();
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].pk, "d0");

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_sql_function_call_is_rejected_in_like_in_is_null_and_contain_positions() {
    let path = temp_dir("sql_func_left_restrictions");
    let schema = basic_schema(4);

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    col.insert(vec![
        make_doc("d0", vec![1.0, 0.0, 0.0, 0.0], "d0"),
        make_doc("d1", vec![0.0, 1.0, 0.0, 0.0], "d1"),
    ])
    .unwrap();
    col.flush().unwrap();

    // The grammar only allows function_call rel_oper value_expr.
    let err = col
        .query_sql("SELECT _finch_uid_ FROM test WHERE array_length(label) IS NULL LIMIT 1")
        .unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument, "err={err:?}");
    assert!(err.message.contains("syntax error"), "err={err:?}");

    let err = col
        .query_sql("SELECT _finch_uid_ FROM test WHERE array_length(label) LIKE 'a%' LIMIT 1")
        .unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument, "err={err:?}");
    assert!(err.message.contains("syntax error"), "err={err:?}");

    let err = col
        .query_sql("SELECT _finch_uid_ FROM test WHERE array_length(label) IN (1, 2) LIMIT 1")
        .unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument, "err={err:?}");
    assert!(err.message.contains("syntax error"), "err={err:?}");

    let err = col
        .query_sql(
            "SELECT _finch_uid_ FROM test WHERE array_length(label) CONTAIN_ANY ('a') LIMIT 1",
        )
        .unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument, "err={err:?}");
    assert!(err.message.contains("syntax error"), "err={err:?}");

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_invert_prefilter_normalizes_numeric_literals() {
    let path = temp_dir("invert_numeric_norm");

    // Add a numeric scalar field with an inverted index, plus a vector index so
    // persisted-segment search takes the index+prefilter path.
    let schema = CollectionSchema::new("test")
        .with_field(
            FieldSchema::new("emb", DataType::VectorFp32)
                .with_dimension(4)
                .with_index(IndexParams::Hnsw(HnswIndexParams::new(MetricType::L2))),
        )
        .with_field(
            FieldSchema::new("age", DataType::Int32)
                .with_index(IndexParams::Invert(InvertIndexParams::default())),
        );

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    let statuses = col
        .insert(vec![
            Doc::new("d1")
                .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
                .set("age", 10i32),
            Doc::new("d2")
                .set("emb", vec![0.0f32, 1.0, 0.0, 0.0])
                .set("age", 30i32),
            Doc::new("d3")
                .set("emb", vec![0.0f32, 0.0, 1.0, 0.0])
                .set("age", 50i32),
        ])
        .unwrap();
    for s in &statuses {
        assert!(s.is_ok(), "insert status should be ok, got: {}", s);
    }

    col.flush().unwrap();

    // SQL parser produces integer literals as I64; prefilter must normalize to
    // the schema's Int32 so invert-index equality lookup works.
    let q = VectorQuery::new("emb", vec![0.0, 1.0, 0.0, 0.0], 10).with_filter("age = 30");
    let results = col.query(q).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].pk, "d2");

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

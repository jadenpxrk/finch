use super::*;

#[test]
fn test_sql_score_virtual_column_and_limit_signed_int_parity() {
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

    // LIMIT accepts signed integers; <=0 behaves like "unset" and uses the default topN.
    let res = col.query_sql("SELECT label FROM test LIMIT 0").unwrap();
    assert_eq!(res.len(), 3);
    let res = col.query_sql("SELECT label FROM test LIMIT -1").unwrap();
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
fn test_sql_vector_condition_validation_and_vector_literal_lexing_match_reference() {
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

    // Schema-free fields cannot be compared against vectors (reference semantic error).
    let err = col
        .query_sql("SELECT _finch_uid_ FROM test WHERE unknown = [1,0,0,0] LIMIT 1")
        .unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument, "err={err:?}");
    assert!(
        err.message
            .contains("vector vector not supported for schema free field"),
        "err={err:?}"
    );

    // reference VECTOR token does not allow `1.` (digits required after '.').
    let err = col
        .query_sql("SELECT _finch_uid_ FROM test WHERE emb = [1.,0,0,0] LIMIT 1")
        .unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument, "err={err:?}");
    assert!(err.message.contains("vector format error"), "err={err:?}");

    // reference VECTOR token allows leading-dot floats like `.1`.
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

    // reference grammar only allows function_call rel_oper value_expr.
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
fn test_insert_vector_fp16_and_int8_are_accepted_and_searchable() {
    let path = temp_dir("insert_vec_fp16_int8");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb16", DataType::VectorFp16).with_dimension(4))
        .with_field(FieldSchema::new("emb8", DataType::VectorInt8).with_dimension(4));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![
        Doc::new("a")
            .set(
                "emb16",
                vec![
                    f16::from_f32(1.0),
                    f16::from_f32(0.0),
                    f16::from_f32(0.0),
                    f16::from_f32(0.0),
                ],
            )
            .set("emb8", vec![1i8, 0, 0, 0]),
        Doc::new("b")
            .set(
                "emb16",
                vec![
                    f16::from_f32(0.0),
                    f16::from_f32(1.0),
                    f16::from_f32(0.0),
                    f16::from_f32(0.0),
                ],
            )
            .set("emb8", vec![0i8, 1, 0, 0]),
    ])
    .unwrap();

    // Query against both fields (writing segment).
    let q16 = VectorQuery::new("emb16", vec![1.0f32, 0.0, 0.0, 0.0], 2);
    let r16 = col.query(q16).unwrap();
    assert_eq!(r16[0].pk, "a");

    let q8 = VectorQuery::new("emb8", vec![1.0f32, 0.0, 0.0, 0.0], 2);
    let r8 = col.query(q8).unwrap();
    assert_eq!(r8[0].pk, "a");

    // Persist and query again.
    col.flush().unwrap();
    let q16 = VectorQuery::new("emb16", vec![1.0f32, 0.0, 0.0, 0.0], 2);
    let r16 = col.query(q16).unwrap();
    assert_eq!(r16[0].pk, "a");

    let q8 = VectorQuery::new("emb8", vec![1.0f32, 0.0, 0.0, 0.0], 2);
    let r8 = col.query(q8).unwrap();
    assert_eq!(r8[0].pk, "a");

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_fp16_fields_round_f32_payloads_on_write() {
    let path = temp_dir("fp16_rounding_on_write");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb16", DataType::VectorFp16).with_dimension(4))
        .with_field(FieldSchema::new("sparse16", DataType::SparseFp16).with_dimension(8));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    let emb_in = vec![0.1f32, 0.2, 0.3, 0.4];
    let emb_expected: Vec<f32> = emb_in.iter().map(|&x| f16::from_f32(x).to_f32()).collect();

    let sparse_indices = vec![1u32, 3u32, 7u32];
    let sparse_values_in = vec![0.3333f32, 0.12345, -0.9876];
    let sparse_values_expected: Vec<f32> = sparse_values_in
        .iter()
        .map(|&x| f16::from_f32(x).to_f32())
        .collect();

    col.insert(vec![Doc::new("d1").set("emb16", emb_in.clone()).set(
        "sparse16",
        Value::SparseF32 {
            indices: sparse_indices.clone(),
            values: sparse_values_in.clone(),
        },
    )])
    .unwrap();

    // Validate in-memory write normalization via fetch().
    let fetched = col.fetch(vec!["d1".to_string()]).unwrap();
    let d1 = fetched.get("d1").unwrap();
    match d1.fields.get("emb16") {
        Some(Value::VecF32(v)) => assert_eq!(v, &emb_expected),
        other => panic!("expected emb16 VecF32, got {other:?}"),
    }
    match d1.fields.get("sparse16") {
        Some(Value::SparseF32 { indices, values }) => {
            assert_eq!(indices, &sparse_indices);
            assert_eq!(values, &sparse_values_expected);
        }
        other => panic!("expected sparse16 SparseF32, got {other:?}"),
    }

    // Validate persisted round-trip.
    col.flush().unwrap();
    drop(col);

    let reopened = Collection::open(&path, CollectionOptions::default()).unwrap();
    let fetched = reopened.fetch(vec!["d1".to_string()]).unwrap();
    let d1 = fetched.get("d1").unwrap();
    match d1.fields.get("emb16") {
        Some(Value::VecF32(v)) => assert_eq!(v, &emb_expected),
        other => panic!("expected emb16 VecF32, got {other:?}"),
    }
    match d1.fields.get("sparse16") {
        Some(Value::SparseF32 { indices, values }) => {
            assert_eq!(indices, &sparse_indices);
            assert_eq!(values, &sparse_values_expected);
        }
        other => panic!("expected sparse16 SparseF32, got {other:?}"),
    }

    drop(reopened);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_integer_vector_fields_normalize_f32_payloads_on_write() {
    let path = temp_dir("int_vec_normalize_on_write");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("i8v", DataType::VectorInt8).with_dimension(4));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    let statuses = col
        .insert(vec![
            Doc::new("d1").set("i8v", vec![1.9f32, -1.9, 9999.0, f32::NAN])
        ])
        .unwrap();
    assert!(statuses.iter().all(|s| s.is_ok()), "statuses={statuses:?}");

    fn assert_doc(doc: &Doc) {
        match doc.fields.get("i8v") {
            Some(Value::VecF32(v)) => assert_eq!(v, &vec![1.0, -1.0, i8::MAX as f32, 0.0]),
            other => panic!("expected i8v VecF32, got {other:?}"),
        }
    }

    let fetched = col.fetch(vec!["d1".to_string()]).unwrap();
    assert_doc(fetched.get("d1").unwrap());

    col.flush().unwrap();
    drop(col);

    let reopened = Collection::open(&path, CollectionOptions::default()).unwrap();
    let fetched = reopened.fetch(vec!["d1".to_string()]).unwrap();
    assert_doc(fetched.get("d1").unwrap());

    drop(reopened);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_schema_rejects_unsupported_dense_vector_dtypes() {
    let path = temp_dir("insert_vec_fp64_int16_int4");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb64", DataType::VectorFp64).with_dimension(4))
        .with_field(FieldSchema::new("emb16", DataType::VectorInt16).with_dimension(4))
        .with_field(FieldSchema::new("emb4", DataType::VectorInt4).with_dimension(4));

    let err = match Collection::create_and_open(&path, schema, CollectionOptions::default()) {
        Ok(_) => panic!("expected schema validation error"),
        Err(err) => err,
    };
    assert!(
        err.message.contains("dense_vector's data type"),
        "err={err}"
    );
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_schema_rejects_binary_dense_vectors() {
    let path = temp_dir("insert_vec_binary");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("b32", DataType::VectorBinary32).with_dimension(2))
        .with_field(FieldSchema::new("b64", DataType::VectorBinary64).with_dimension(1));

    let err = match Collection::create_and_open(&path, schema, CollectionOptions::default()) {
        Ok(_) => panic!("expected schema validation error"),
        Err(err) => err,
    };
    assert!(
        err.message.contains("dense_vector's data type"),
        "err={err}"
    );
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

#[test]
fn test_doc_validation_rejects_bad_pk_and_missing_required_fields() {
    let path = temp_dir("doc_validate");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("label", DataType::String).not_null());
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    // Bad PK (illegal char '/')
    let statuses = col
        .insert(vec![Doc::new("bad/pk")
            .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
            .set("label", "x")])
        .unwrap();
    assert!(statuses[0].is_err());

    // Missing required (not-null) field
    let statuses = col
        .insert(vec![Doc::new("ok").set("emb", vec![1.0f32, 0.0, 0.0, 0.0])])
        .unwrap();
    assert!(statuses[0].is_err());

    // Unknown field
    let statuses = col
        .insert(vec![Doc::new("ok2")
            .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
            .set("label", "x")
            .set("unknown", 1i64)])
        .unwrap();
    assert!(statuses[0].is_err());

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_insert_duplicate_pk() {
    let path = temp_dir("dup_pk");
    let schema = basic_schema(4);
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![make_doc("d1", vec![1.0, 0.0, 0.0, 0.0], "a")])
        .unwrap();
    let results = col
        .insert(vec![make_doc("d1", vec![0.0, 1.0, 0.0, 0.0], "b")])
        .unwrap();
    assert!(results[0].is_err(), "duplicate pk should fail");

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_write_batch_size_limit_is_enforced() {
    let path = temp_dir("write_batch_limit");
    let schema = basic_schema(4);
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    let docs: Vec<Doc> = (0..1025)
        .map(|i| {
            Doc::new(format!("d{}", i))
                .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
                .set("label", "x")
        })
        .collect();
    assert!(col.insert(docs).is_err());

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_query_output_semantics_match_compat_suite() {
    let path = temp_dir("query_output_semantics");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("label", DataType::String))
        .with_field(FieldSchema::new("category", DataType::String));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    col.insert(vec![
        Doc::new("d1")
            .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
            .set("label", "alpha")
            .set("category", "science"),
        Doc::new("d2")
            .set("emb", vec![0.0f32, 1.0, 0.0, 0.0])
            .set("label", "beta")
            .set("category", "math"),
    ])
    .unwrap();

    // Default behavior: all scalar fields returned, vectors/doc_id excluded.
    let q_default = VectorQuery::new("emb", vec![0.0, 1.0, 0.0, 0.0], 1);
    let default_results = col.query(q_default).unwrap();
    assert_eq!(default_results.len(), 1);
    let default_doc = &default_results[0];
    assert_eq!(default_doc.doc_id, 0);
    assert!(default_doc.fields.contains_key("label"));
    assert!(default_doc.fields.contains_key("category"));
    assert!(!default_doc.fields.contains_key("emb"));

    // output_fields may include "*" (asterisk) to mean "all scalar fields".
    let q_star = VectorQuery::new("emb", vec![0.0, 1.0, 0.0, 0.0], 1)
        .with_output_fields(vec!["*".to_string()]);
    let star_results = col.query(q_star).unwrap();
    assert_eq!(star_results.len(), 1);
    let star_doc = &star_results[0];
    assert!(star_doc.fields.contains_key("label"));
    assert!(star_doc.fields.contains_key("category"));
    assert!(!star_doc.fields.contains_key("emb"));

    // Mixed "*" and explicit fields still acts like "*".
    let q_star_mixed = VectorQuery::new("emb", vec![0.0, 1.0, 0.0, 0.0], 1)
        .with_output_fields(vec!["label".to_string(), "*".to_string()]);
    let star_mixed_results = col.query(q_star_mixed).unwrap();
    assert_eq!(star_mixed_results.len(), 1);
    let star_mixed_doc = &star_mixed_results[0];
    assert!(star_mixed_doc.fields.contains_key("label"));
    assert!(star_mixed_doc.fields.contains_key("category"));
    assert!(!star_mixed_doc.fields.contains_key("emb"));

    // Explicit include flags should expose vector and internal doc_id.
    let mut q_with_includes = VectorQuery::new("emb", vec![0.0, 1.0, 0.0, 0.0], 1);
    q_with_includes.include_vector = true;
    q_with_includes.include_doc_id = true;
    let include_results = col.query(q_with_includes).unwrap();
    assert_eq!(include_results.len(), 1);
    let include_doc = &include_results[0];
    assert_eq!(include_doc.pk, "d2");
    let fetched = col.fetch(vec!["d2".to_string()]).unwrap();
    assert_eq!(include_doc.doc_id, fetched.get("d2").unwrap().doc_id);
    assert!(include_doc.fields.contains_key("emb"));

    // Compatibility tri-state semantics:
    // Some([]) => return no fields.
    let q_no_fields =
        VectorQuery::new("emb", vec![0.0, 1.0, 0.0, 0.0], 1).with_output_fields(vec![]);
    let no_field_results = col.query(q_no_fields).unwrap();
    assert_eq!(no_field_results.len(), 1);
    assert!(no_field_results[0].fields.is_empty());

    // include_vector is independent of output_fields. When
    // output_fields=[] (select nothing) and include_vector=true, reference still
    // returns vector fields.
    let mut q_vec_only =
        VectorQuery::new("emb", vec![0.0, 1.0, 0.0, 0.0], 1).with_output_fields(vec![]);
    q_vec_only.include_vector = true;
    let vec_only_results = col.query(q_vec_only).unwrap();
    assert_eq!(vec_only_results.len(), 1);
    assert!(vec_only_results[0].fields.contains_key("emb"));
    assert!(!vec_only_results[0].fields.contains_key("label"));
    assert!(!vec_only_results[0].fields.contains_key("category"));

    // Some([...]) => return selected fields only.
    let q_selected = VectorQuery::new("emb", vec![0.0, 1.0, 0.0, 0.0], 1)
        .with_output_fields(vec!["label".to_string()]);
    let selected_results = col.query(q_selected).unwrap();
    assert_eq!(selected_results.len(), 1);
    let selected_doc = &selected_results[0];
    assert_eq!(selected_doc.fields.len(), 1);
    assert!(selected_doc.fields.contains_key("label"));

    // Filter-only query: "*" should behave like default-all.
    let q_filter_star = VectorQuery::new("", Vec::new(), 10)
        .with_filter("label = 'alpha'")
        .with_output_fields(vec!["*".to_string()]);
    let filter_star_results = col.query(q_filter_star).unwrap();
    assert_eq!(filter_star_results.len(), 1);
    assert_eq!(filter_star_results[0].pk, "d1");
    assert!(filter_star_results[0].fields.contains_key("label"));
    assert!(filter_star_results[0].fields.contains_key("category"));

    // selecting a non-existent output field is an error.
    let q_bad_out = VectorQuery::new("emb", vec![0.0, 1.0, 0.0, 0.0], 1)
        .with_output_fields(vec!["label".to_string(), "not_exist_field".to_string()]);
    let err = col.query(q_bad_out).unwrap_err();
    assert!(err.message.contains("not defined in schema"));

    // Even when "*" is present, explicit non-existent fields still error.
    let q_bad_out_star = VectorQuery::new("emb", vec![0.0, 1.0, 0.0, 0.0], 1)
        .with_output_fields(vec!["*".to_string(), "not_exist_field".to_string()]);
    let err = col.query(q_bad_out_star).unwrap_err();
    assert!(err.message.contains("not defined in schema"));

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

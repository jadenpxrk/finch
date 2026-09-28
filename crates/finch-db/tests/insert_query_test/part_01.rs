use super::*;

#[test]
fn test_insert_and_query_writing_segment() {
    let path = temp_dir("insert_query");
    let schema = basic_schema(4);
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    let docs = vec![
        make_doc("d1", vec![1.0, 0.0, 0.0, 0.0], "alpha"),
        make_doc("d2", vec![0.0, 1.0, 0.0, 0.0], "beta"),
        make_doc("d3", vec![0.0, 0.0, 1.0, 0.0], "gamma"),
    ];
    let statuses = col.insert(docs).unwrap();
    assert!(
        statuses.iter().all(|s| s.is_ok()),
        "all inserts should succeed"
    );

    let stats = col.stats().unwrap();
    assert_eq!(stats.doc_count, 3);

    // Nearest to [1,0,0,0] should be d1
    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 3);
    let results = col.query(q).unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].pk, "d1");

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_system_columns_work_in_filters_and_sql_queries() {
    let path = temp_dir("system_cols_sql");
    let schema = basic_schema(4);

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    // Force two persisted segments: [0,1] then [2,3].
    col.insert(vec![
        make_doc("d0", vec![1.0, 0.0, 0.0, 0.0], "d0"),
        make_doc("d1", vec![0.0, 1.0, 0.0, 0.0], "d1"),
    ])
    .unwrap();
    col.flush().unwrap();
    col.insert(vec![
        make_doc("d2", vec![0.0, 0.0, 1.0, 0.0], "d2"),
        make_doc("d3", vec![0.0, 0.0, 0.0, 1.0], "d3"),
    ])
    .unwrap();
    col.flush().unwrap();

    // Filter on _finch_uid_ (primary key).
    let q = VectorQuery {
        topk: 10,
        field_name: String::new(),
        id: None,
        query_vector: Vec::new(),
        query_vector_u32: Vec::new(),
        query_vector_u64: Vec::new(),
        sparse_indices: Vec::new(),
        sparse_values: Vec::new(),
        filter: Some(format!("{SYS_USER_ID} = 'd2'")),
        include_vector: false,
        include_doc_id: true,
        output_fields: Some(Vec::new()),
        query_params: QueryParams::default(),
    };
    let res = col.query(q).unwrap();
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].pk, "d2");
    assert_eq!(res[0].doc_id, 2);

    // Filter on _finch_g_doc_id_.
    let q = VectorQuery {
        topk: 10,
        field_name: String::new(),
        id: None,
        query_vector: Vec::new(),
        query_vector_u32: Vec::new(),
        query_vector_u64: Vec::new(),
        sparse_indices: Vec::new(),
        sparse_values: Vec::new(),
        filter: Some(format!("{SYS_GLOBAL_DOC_ID} >= 3")),
        include_vector: false,
        include_doc_id: true,
        output_fields: Some(Vec::new()),
        query_params: QueryParams::default(),
    };
    let res = col.query(q).unwrap();
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].pk, "d3");
    assert_eq!(res[0].doc_id, 3);

    // Filter on _finch_row_id_ should match row 0 in each segment => doc_id 0 and 2.
    let q = VectorQuery {
        topk: 10,
        field_name: String::new(),
        id: None,
        query_vector: Vec::new(),
        query_vector_u32: Vec::new(),
        query_vector_u64: Vec::new(),
        sparse_indices: Vec::new(),
        sparse_values: Vec::new(),
        filter: Some(format!("{SYS_LOCAL_ROW_ID} = 0")),
        include_vector: false,
        include_doc_id: true,
        output_fields: Some(Vec::new()),
        query_params: QueryParams::default(),
    };
    let res = col.query(q).unwrap();
    let ids: Vec<u64> = res.iter().map(|d| d.doc_id).collect();
    assert_eq!(ids, vec![0, 2]);

    // SQL surface: project system columns into fields, and support vector condition.
    let res = col.query_sql(&format!(
        "SELECT {SYS_USER_ID}, {SYS_LOCAL_ROW_ID}, {SYS_GLOBAL_DOC_ID} FROM test WHERE {SYS_LOCAL_ROW_ID} = 0 LIMIT 10"
    )).unwrap();
    assert_eq!(res.len(), 2);
    for d in &res {
        assert_eq!(
            d.fields.get(SYS_USER_ID).and_then(|v| v.as_str()),
            Some(d.pk.as_str())
        );
        assert!(matches!(
            d.fields.get(SYS_LOCAL_ROW_ID),
            Some(Value::U64(0))
        ));
        assert!(matches!(d.fields.get(SYS_GLOBAL_DOC_ID), Some(Value::U64(x)) if *x == d.doc_id));
    }

    let res = col
        .query_sql("SELECT _finch_uid_ FROM test WHERE emb = [1,0,0,0] LIMIT 1")
        .unwrap();
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].pk, "d0");
    assert_eq!(
        res[0].fields.get(SYS_USER_ID).and_then(|v| v.as_str()),
        Some("d0")
    );

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_row_id_semantics_match_reference_after_optimize_rebuild_compaction() {
    // reference `_finch_row_id_` is a segment-local row index (0..doc_count-1).
    // When optimize runs in "rebuild" mode (drop deletes), global doc_ids can
    // become sparse within the output segment; `_finch_row_id_` must remain
    // dense and reflect the row order within the rebuilt segment.
    let path = temp_dir("row_id_rebuild_compaction");
    let schema = basic_schema(4);
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    let statuses = col
        .insert(vec![
            make_doc("d0", vec![1.0, 0.0, 0.0, 0.0], "d0"),
            make_doc("d1", vec![0.0, 1.0, 0.0, 0.0], "d1"),
            make_doc("d2", vec![0.0, 0.0, 1.0, 0.0], "d2"),
            make_doc("d3", vec![0.0, 0.0, 0.0, 1.0], "d3"),
            make_doc("d4", vec![1.0, 1.0, 0.0, 0.0], "d4"),
        ])
        .unwrap();
    assert!(statuses.iter().all(|s| s.is_ok()));
    col.flush().unwrap();

    // Delete 2/5 docs => delete ratio 40% (> 30%), so optimize should rebuild.
    let dels = col
        .delete(vec!["d1".to_string(), "d3".to_string()])
        .unwrap();
    assert!(dels.iter().all(|s| s.is_ok()));

    col.optimize(OptimizeOptions::default()).unwrap();

    // Filter-only scan: ensure `_finch_row_id_` is dense and matches doc-id order rank.
    let q = VectorQuery {
        topk: 10,
        field_name: String::new(),
        id: None,
        query_vector: Vec::new(),
        query_vector_u32: Vec::new(),
        query_vector_u64: Vec::new(),
        sparse_indices: Vec::new(),
        sparse_values: Vec::new(),
        filter: None,
        include_vector: false,
        include_doc_id: true,
        output_fields: Some(vec![
            SYS_GLOBAL_DOC_ID.to_string(),
            SYS_LOCAL_ROW_ID.to_string(),
        ]),
        query_params: QueryParams::default(),
    };
    let res = col.query(q).unwrap();
    let mut pairs: Vec<(u64, u64)> = res
        .iter()
        .map(|d| {
            let row_id = match d.fields.get(SYS_LOCAL_ROW_ID) {
                Some(Value::U64(v)) => *v,
                other => panic!("expected _finch_row_id_ to be U64, got {other:?}"),
            };
            (d.doc_id, row_id)
        })
        .collect();
    pairs.sort_by_key(|(doc_id, _)| *doc_id);
    assert_eq!(pairs, vec![(0, 0), (2, 1), (4, 2)]);

    // Filter by row_id should use the rebuilt segment-local row index.
    let q = VectorQuery {
        topk: 10,
        field_name: String::new(),
        id: None,
        query_vector: Vec::new(),
        query_vector_u32: Vec::new(),
        query_vector_u64: Vec::new(),
        sparse_indices: Vec::new(),
        sparse_values: Vec::new(),
        filter: Some(format!("{SYS_LOCAL_ROW_ID} = 1")),
        include_vector: false,
        include_doc_id: true,
        output_fields: Some(vec![
            SYS_GLOBAL_DOC_ID.to_string(),
            SYS_LOCAL_ROW_ID.to_string(),
        ]),
        query_params: QueryParams::default(),
    };
    let res = col.query(q).unwrap();
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].doc_id, 2);
    assert!(matches!(
        res[0].fields.get(SYS_LOCAL_ROW_ID),
        Some(Value::U64(1))
    ));

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_optimize_rebuild_creates_new_segment_and_removes_old_segment_dirs() {
    use finch_db::version::VersionManager;

    let path = temp_dir("optimize_rebuild_segment_ids");
    let schema = basic_schema(4);
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    let st = col
        .insert(vec![
            make_doc("d0", vec![1.0, 0.0, 0.0, 0.0], "d0"),
            make_doc("d1", vec![0.0, 1.0, 0.0, 0.0], "d1"),
            make_doc("d2", vec![0.0, 0.0, 1.0, 0.0], "d2"),
            make_doc("d3", vec![0.0, 0.0, 0.0, 1.0], "d3"),
            make_doc("d4", vec![1.0, 1.0, 0.0, 0.0], "d4"),
        ])
        .unwrap();
    assert!(st.iter().all(|s| s.is_ok()));
    col.flush().unwrap();

    assert!(
        path.join("seg_0").exists(),
        "expected persisted seg_0 after flush"
    );

    // Trigger rebuild mode by exceeding the delete ratio threshold.
    let dels = col
        .delete(vec!["d1".to_string(), "d3".to_string()])
        .unwrap();
    assert!(dels.iter().all(|s| s.is_ok()));
    col.optimize(OptimizeOptions::default()).unwrap();
    drop(col);

    // rebuild compaction writes new segment ids and removes old persisted segment dirs.
    assert!(
        !path.join("seg_0").exists(),
        "expected old seg_0 to be removed"
    );

    let vm = VersionManager::load(&path).unwrap();
    let v = vm.current();
    assert_eq!(
        v.persisted_segments.len(),
        1,
        "expected single rebuilt output segment"
    );
    let out_id = v.persisted_segments[0].segment_id;
    assert!(
        out_id >= 2,
        "expected rebuilt segment id to be >= 2, got {out_id}"
    );
    assert!(path.join(format!("seg_{out_id}")).exists());

    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_row_id_semantics_preserve_holes_when_optimize_does_not_rebuild() {
    // When delete ratio is below the rebuild threshold, optimize should not drop deleted docs.
    // `_finch_row_id_` should remain the segment-local row index and therefore preserve holes for
    // deleted rows (reference behavior).
    let path = temp_dir("row_id_no_rebuild_holes");
    let schema = basic_schema(4);
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    let statuses = col
        .insert(vec![
            make_doc("d0", vec![1.0, 0.0, 0.0, 0.0], "d0"),
            make_doc("d1", vec![0.0, 1.0, 0.0, 0.0], "d1"),
            make_doc("d2", vec![0.0, 0.0, 1.0, 0.0], "d2"),
            make_doc("d3", vec![0.0, 0.0, 0.0, 1.0], "d3"),
            make_doc("d4", vec![1.0, 1.0, 0.0, 0.0], "d4"),
        ])
        .unwrap();
    assert!(statuses.iter().all(|s| s.is_ok()));
    col.flush().unwrap();

    // Delete 1/5 docs => delete ratio 20% (<= 30%), so optimize should not rebuild.
    let dels = col.delete(vec!["d1".to_string()]).unwrap();
    assert!(dels.iter().all(|s| s.is_ok()));

    col.optimize(OptimizeOptions::default()).unwrap();

    let q = VectorQuery {
        topk: 10,
        field_name: String::new(),
        id: None,
        query_vector: Vec::new(),
        query_vector_u32: Vec::new(),
        query_vector_u64: Vec::new(),
        sparse_indices: Vec::new(),
        sparse_values: Vec::new(),
        filter: None,
        include_vector: false,
        include_doc_id: true,
        output_fields: Some(vec![
            SYS_GLOBAL_DOC_ID.to_string(),
            SYS_LOCAL_ROW_ID.to_string(),
        ]),
        query_params: QueryParams::default(),
    };
    let res = col.query(q).unwrap();
    let mut pairs: Vec<(u64, u64)> = res
        .iter()
        .map(|d| {
            let row_id = match d.fields.get(SYS_LOCAL_ROW_ID) {
                Some(Value::U64(v)) => *v,
                other => panic!("expected _finch_row_id_ to be U64, got {other:?}"),
            };
            (d.doc_id, row_id)
        })
        .collect();
    pairs.sort_by_key(|(doc_id, _)| *doc_id);
    assert_eq!(pairs, vec![(0, 0), (2, 2), (3, 3), (4, 4)]);

    // Deleted row_id should return no docs.
    let q = VectorQuery {
        topk: 10,
        field_name: String::new(),
        id: None,
        query_vector: Vec::new(),
        query_vector_u32: Vec::new(),
        query_vector_u64: Vec::new(),
        sparse_indices: Vec::new(),
        sparse_values: Vec::new(),
        filter: Some(format!("{SYS_LOCAL_ROW_ID} = 1")),
        include_vector: false,
        include_doc_id: true,
        output_fields: Some(vec![SYS_LOCAL_ROW_ID.to_string()]),
        query_params: QueryParams::default(),
    };
    let res = col.query(q).unwrap();
    assert!(res.is_empty());

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_vectorquery_output_fields_support_system_columns() {
    let path = temp_dir("system_cols_output_fields");
    let schema = basic_schema(4);

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    col.insert(vec![
        make_doc("d0", vec![1.0, 0.0, 0.0, 0.0], "d0"),
        make_doc("d1", vec![0.0, 1.0, 0.0, 0.0], "d1"),
    ])
    .unwrap();
    col.flush().unwrap();

    let q = VectorQuery {
        topk: 10,
        field_name: String::new(),
        id: None,
        query_vector: Vec::new(),
        query_vector_u32: Vec::new(),
        query_vector_u64: Vec::new(),
        sparse_indices: Vec::new(),
        sparse_values: Vec::new(),
        filter: Some(format!("{SYS_GLOBAL_DOC_ID} >= 0")),
        include_vector: false,
        include_doc_id: false, // should be upgraded internally due to system columns
        output_fields: Some(vec![
            SYS_USER_ID.to_string(),
            SYS_LOCAL_ROW_ID.to_string(),
            SYS_GLOBAL_DOC_ID.to_string(),
            SYS_SCORE.to_string(),
        ]),
        query_params: QueryParams::default(),
    };
    let res = col.query(q).unwrap();
    assert_eq!(res.len(), 2);
    for d in &res {
        assert_eq!(
            d.fields.get(SYS_USER_ID).and_then(|v| v.as_str()),
            Some(d.pk.as_str())
        );
        assert!(matches!(d.fields.get(SYS_GLOBAL_DOC_ID), Some(Value::U64(x)) if *x == d.doc_id));
        // With a single persisted segment, row_id == doc_id.
        assert!(matches!(d.fields.get(SYS_LOCAL_ROW_ID), Some(Value::U64(x)) if *x == d.doc_id));
        assert!(matches!(d.fields.get(SYS_SCORE), Some(Value::F32(x)) if *x == 0.0));
    }

    let mut q = VectorQuery::new("emb", vec![1.0f32, 0.0, 0.0, 0.0], 2);
    q.output_fields = Some(vec![SYS_SCORE.to_string()]);
    let res = col.query(q).unwrap();
    assert_eq!(res.len(), 2);
    for d in &res {
        let Some(Value::F32(s)) = d.fields.get(SYS_SCORE) else {
            panic!("expected {SYS_SCORE} in fields: {d:?}");
        };
        assert!(
            (*s - d.score).abs() <= 1e-6,
            "score mismatch: field={s} member={} doc={d:?}",
            d.score
        );
    }

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

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
    // This test mirrors `reference/tests/db/sqlengine/query_info_test.cc` which uses
    // `1collection` and `1-dash_score_field` in filters.
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

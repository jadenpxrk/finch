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
fn test_row_id_semantics_after_optimize_rebuild_compaction() {
    // `_finch_row_id_` is a segment-local row index (0..doc_count-1).
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
    // deleted rows.
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
fn test_query_output_fields_semantics() {
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

    // `output_fields` is tri-state:
    // Some([]) => return no fields.
    let q_no_fields =
        VectorQuery::new("emb", vec![0.0, 1.0, 0.0, 0.0], 1).with_output_fields(vec![]);
    let no_field_results = col.query(q_no_fields).unwrap();
    assert_eq!(no_field_results.len(), 1);
    assert!(no_field_results[0].fields.is_empty());

    // include_vector is independent of output_fields. When
    // output_fields=[] (select nothing) and include_vector=true still
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

use super::*;

#[test]
fn test_optimize_rebuild_threshold_is_strictly_greater_than_0_3() {
    use finch_db::version::VersionManager;

    // Case 1: exactly 30% deleted -> should NOT rebuild (no physical drop).
    {
        let path = temp_dir("optimize_rebuild_threshold_boundary_no_rebuild");
        let schema = CollectionSchema::new("test")
            .with_field(FieldSchema::new("id", DataType::Int64).not_null())
            .with_field(
                FieldSchema::new("emb", DataType::VectorFp32)
                    .with_dimension(4)
                    .with_index(IndexParams::Flat(FlatIndexParams::new(MetricType::L2))),
            );
        let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

        let docs: Vec<Doc> = (0..10u64)
            .map(|i| {
                Doc::new(format!("d{i}"))
                    .set("id", i as i64)
                    .set("emb", vec![i as f32; 4])
            })
            .collect();
        col.insert(docs).unwrap();
        col.flush().unwrap();

        let statuses = col
            .delete(vec!["d0".to_string(), "d1".to_string(), "d2".to_string()])
            .unwrap();
        assert!(statuses.iter().all(|s| s.is_ok()));

        col.optimize(OptimizeOptions::default()).unwrap();

        let vm = VersionManager::load(&path).unwrap();
        let v = vm.current();
        let persisted_count: u64 = v.persisted_segments.iter().map(|s| s.doc_count).sum();
        assert_eq!(
            persisted_count, 10,
            "at exactly 0.3 delete ratio, optimize must not rebuild and should retain tombstoned docs in forward stores"
        );
        assert_eq!(v.persisted_segments.len(), 1);
        assert_eq!(v.persisted_segments[0].segment_id, 0);

        let stats = col.stats().unwrap();
        assert_eq!(
            stats.doc_count, 7,
            "stats should still reflect live docs only"
        );

        drop(col);
        std::fs::remove_dir_all(&path).ok();
    }

    // Case 2: 40% deleted -> must rebuild (physical drop).
    {
        let path = temp_dir("optimize_rebuild_threshold_boundary_rebuild");
        let schema = CollectionSchema::new("test")
            .with_field(FieldSchema::new("id", DataType::Int64).not_null())
            .with_field(
                FieldSchema::new("emb", DataType::VectorFp32)
                    .with_dimension(4)
                    .with_index(IndexParams::Flat(FlatIndexParams::new(MetricType::L2))),
            );
        let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

        let docs: Vec<Doc> = (0..10u64)
            .map(|i| {
                Doc::new(format!("d{i}"))
                    .set("id", i as i64)
                    .set("emb", vec![i as f32; 4])
            })
            .collect();
        col.insert(docs).unwrap();
        col.flush().unwrap();

        let statuses = col
            .delete(vec![
                "d0".to_string(),
                "d1".to_string(),
                "d2".to_string(),
                "d3".to_string(),
            ])
            .unwrap();
        assert!(statuses.iter().all(|s| s.is_ok()));

        col.optimize(OptimizeOptions::default()).unwrap();

        let vm = VersionManager::load(&path).unwrap();
        let v = vm.current();
        let persisted_count: u64 = v.persisted_segments.iter().map(|s| s.doc_count).sum();
        assert_eq!(
            persisted_count, 6,
            "when delete ratio is > 0.3, optimize must rebuild and physically drop deleted docs"
        );
        assert_eq!(v.persisted_segments.len(), 1);
        assert_ne!(
            v.persisted_segments[0].segment_id, 0,
            "rebuild should create a new segment id"
        );
        assert!(
            !path.join("seg_0").exists(),
            "old segment dir should be removed after rebuild"
        );

        let stats = col.stats().unwrap();
        assert_eq!(stats.doc_count, 6);

        drop(col);
        std::fs::remove_dir_all(&path).ok();
    }
}

#[test]
fn test_optimize_removes_orphan_output_segment_dir_before_writing() {
    use finch_db::version::VersionManager;

    let path = temp_dir("optimize_orphan_output_dir");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("id", DataType::Int64).not_null());

    let opts = CollectionOptions {
        max_buffer_size: 1024, // force multiple persisted segments
        ..CollectionOptions::default()
    };
    let col = Collection::create_and_open(&path, schema, opts).unwrap();

    let docs: Vec<Doc> = (0..200u64)
        .map(|i| {
            Doc::new(format!("d{i}"))
                .set("id", i as i64)
                .set("emb", vec![i as f32; 4])
        })
        .collect();
    for chunk in docs.chunks(128) {
        col.insert(chunk.to_vec()).unwrap();
    }
    col.flush().unwrap();

    let vm = VersionManager::load(&path).unwrap();
    let v = vm.current();
    let active_writing = v.writing_segment_id.unwrap_or(0);
    let base_out = v.next_segment_id.max(active_writing.saturating_add(1));

    // Simulate a previous crashed optimize that left behind an orphan output directory.
    let orphan = path.join(format!("seg_{}", base_out));
    std::fs::create_dir_all(&orphan).unwrap();
    std::fs::write(orphan.join("ORPHAN"), b"stale").unwrap();
    assert!(orphan.join("ORPHAN").exists());

    col.optimize(OptimizeOptions {
        target_segment_size: Some(u64::MAX),
        ..Default::default()
    })
    .unwrap();

    // Optimize should have removed the orphan and written a valid segment directory in its place.
    assert!(path.join(format!("seg_{}", base_out)).exists());
    assert!(!path.join(format!("seg_{}/ORPHAN", base_out)).exists());

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

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
fn test_query_array_length_validation_error_messages_match_reference() {
    // Mirrors reference `QueryInfoTest` array_length analyzer cases:
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

#[test]
fn test_query_relation_expr_shape_validation_matches_reference() {
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

    // reference grammar parity: NULL is not a valid constant in relation expressions.
    let err = col
        .query(VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("rank = NULL"))
        .unwrap_err();
    assert!(err.message().contains("syntax error"));

    let err = col
        .query(VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("rank IN (NULL)"))
        .unwrap_err();
    assert!(err.message().contains("syntax error"));

    // reference grammar parity: `IN ()` is a syntax error.
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

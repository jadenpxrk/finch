mod common;
use common::*;

// ── Schema validation ─────────────────────────────────────────────────────────

#[test]
fn test_schema_validation_empty_name() {
    let path = temp_dir("schema_val");
    let schema = CollectionSchema::new("").with_field(
        FieldSchema::new("emb", DataType::VectorFp32)
            .nullable()
            .with_dimension(4),
    );
    assert!(Collection::create_and_open(&path, schema, CollectionOptions::default()).is_err());
}

#[test]
fn test_schema_validation_vector_no_dim() {
    let path = temp_dir("schema_dim");
    // Vector field without dimension should fail validation
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).nullable()); // no dimension
    assert!(Collection::create_and_open(&path, schema, CollectionOptions::default()).is_err());
}

#[test]
fn test_query_contain_empty_list_semantics_match_reference() {
    let path = temp_dir("contain_empty_list_semantics");

    let schema = CollectionSchema::new("test")
        .with_field(
            FieldSchema::new("emb", DataType::VectorFp32)
                .nullable()
                .with_dimension(4),
        )
        .with_field(FieldSchema::new("tags", DataType::ArrayString).nullable());

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let statuses = col
        .insert(vec![
            // Missing tags => NULL
            Doc::new("d1").set("emb", vec![1.0f32, 0.0, 0.0, 0.0]),
            // Explicit null tags => NULL
            Doc::new("d2")
                .set("emb", vec![0.0f32, 1.0, 0.0, 0.0])
                .set("tags", Value::Null),
            // Non-null tags
            Doc::new("d3")
                .set("emb", vec![0.0f32, 0.0, 1.0, 0.0])
                .set("tags", Value::ArrayString(vec!["a".to_string()])),
            // Empty-but-non-null tags
            Doc::new("d4")
                .set("emb", vec![0.0f32, 0.0, 0.0, 1.0])
                .set("tags", Value::ArrayString(Vec::<String>::new())),
        ])
        .unwrap();
    assert!(statuses.iter().all(|s| s.is_ok()));

    fn pks(col: &Collection, filter: &str) -> Vec<String> {
        let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter(filter);
        let mut out: Vec<String> = col.query(q).unwrap().iter().map(|d| d.pk.clone()).collect();
        out.sort();
        out
    }

    // `contain_any()` => false.
    assert!(pks(&col, "tags contain_any ()").is_empty());
    // `contain_all()` => `is not null`.
    assert_eq!(
        pks(&col, "tags contain_all ()"),
        vec!["d3".to_string(), "d4".to_string()]
    );
    // `not contain_all()` => false.
    assert!(pks(&col, "tags NOT CONTAIN_ALL ()").is_empty());
    // `not contain_any()` => `is not null`.
    assert_eq!(
        pks(&col, "tags NOT CONTAIN_ANY ()"),
        vec!["d3".to_string(), "d4".to_string()]
    );

    // Persist and re-check.
    col.flush().unwrap();
    assert!(pks(&col, "tags contain_any ()").is_empty());
    assert_eq!(
        pks(&col, "tags contain_all ()"),
        vec!["d3".to_string(), "d4".to_string()]
    );
    assert!(pks(&col, "tags NOT CONTAIN_ALL ()").is_empty());
    assert_eq!(
        pks(&col, "tags NOT CONTAIN_ANY ()"),
        vec!["d3".to_string(), "d4".to_string()]
    );

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_invert_index_tracks_null_and_nonnull_for_is_null_filters() {
    let path = temp_dir("invert_null_nonnull_markers");

    let schema = CollectionSchema::new("test")
        .with_field(
            FieldSchema::new("emb", DataType::VectorFp32)
                .nullable()
                .with_dimension(4)
                .with_index(IndexParams::Hnsw(HnswIndexParams::new(MetricType::L2))),
        )
        .with_field(
            FieldSchema::new("tag", DataType::String)
                .nullable()
                .with_index(IndexParams::Invert(InvertIndexParams::default())),
        );

    {
        let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
        col.insert(vec![
            // Missing tag => IS NULL
            Doc::new("d1").set("emb", vec![1.0f32, 0.0, 0.0, 0.0]),
            // Explicit null => IS NULL
            Doc::new("d2")
                .set("emb", vec![0.0f32, 1.0, 0.0, 0.0])
                .set("tag", Value::Null),
            // Non-null => IS NOT NULL
            Doc::new("d3")
                .set("emb", vec![0.0f32, 0.0, 1.0, 0.0])
                .set("tag", "x"),
        ])
        .unwrap();
        col.flush().unwrap();
    }

    let ro_opts = CollectionOptions {
        read_only: true,
        enable_mmap: false,
        index_storage: None,
        forward_storage: None,
        max_buffer_size: CollectionOptions::DEFAULT_MAX_BUFFER_SIZE,
        forward_file_format: None,
    };

    let (d1_id, d2_id, d3_id) = {
        let col = Collection::open(&path, ro_opts).unwrap();
        let fetched = col
            .fetch(vec!["d1".to_string(), "d2".to_string(), "d3".to_string()])
            .unwrap();
        (
            fetched.get("d1").unwrap().doc_id,
            fetched.get("d2").unwrap().doc_id,
            fetched.get("d3").unwrap().doc_id,
        )
    };

    // Find the segment's invert index that contains our entries (post-flush,
    // the new writing segment may exist but should be empty).
    let mut idx: Option<InvertIndex> = None;
    for entry in std::fs::read_dir(&path).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name();
        if !name.to_string_lossy().starts_with("seg_") {
            continue;
        }
        let inv_path = entry.path().join("tag_invert");
        if !entry.path().join("tag_invert.keys").exists() {
            continue;
        }
        let ro = InvertIndex::open_read_only(
            &inv_path,
            "tag".to_string(),
            DataType::String,
            InvertIndexParams::default(),
        )
        .unwrap();
        if !ro.lookup_is_null().unwrap().is_empty() || !ro.lookup_is_not_null().unwrap().is_empty()
        {
            idx = Some(ro);
            break;
        }
    }
    let idx = idx.expect("expected to find a tag_invert RocksDB with entries");

    let nulls = idx.lookup_is_null().unwrap();
    assert!(nulls.contains(d1_id));
    assert!(nulls.contains(d2_id));
    assert!(!nulls.contains(d3_id));

    let nonnulls = idx.lookup_is_not_null().unwrap();
    assert!(!nonnulls.contains(d1_id));
    assert!(!nonnulls.contains(d2_id));
    assert!(nonnulls.contains(d3_id));

    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_query_eq_or_rewrites_to_in_and_ne_or_rewrites_to_not_in() {
    let path = temp_dir("rewrite_eq_or_ne_or");

    let schema = CollectionSchema::new("test")
        .with_field(
            FieldSchema::new("emb", DataType::VectorFp32)
                .nullable()
                .with_dimension(4),
        )
        .with_field(FieldSchema::new("age", DataType::Int64).nullable());

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let statuses = col
        .insert(vec![
            Doc::new("d10")
                .set("emb", vec![10.0f32, 0.0, 0.0, 0.0])
                .set("age", 10_i64),
            Doc::new("d20")
                .set("emb", vec![20.0f32, 0.0, 0.0, 0.0])
                .set("age", 20_i64),
            Doc::new("d30")
                .set("emb", vec![30.0f32, 0.0, 0.0, 0.0])
                .set("age", 30_i64),
        ])
        .unwrap();
    assert!(statuses.iter().all(|s| s.is_ok()));

    fn pks(col: &Collection, filter: &str) -> Vec<String> {
        let q = VectorQuery::new("emb", vec![0.0, 0.0, 0.0, 0.0], 10).with_filter(filter);
        let mut out: Vec<String> = col.query(q).unwrap().iter().map(|d| d.pk.clone()).collect();
        out.sort();
        out
    }

    let all_ages = vec!["d10".to_string(), "d20".to_string(), "d30".to_string()];

    // `a = 10 OR a = 20` becomes `a IN (10,20)` (same results).
    assert_eq!(
        pks(&col, "age = 10 OR age = 20"),
        vec!["d10".to_string(), "d20".to_string()]
    );

    // `a != 10 OR a != 20` holds for every non-null `a`; it is not `a NOT IN (10,20)`.
    assert_eq!(pks(&col, "age != 10 OR age != 20"), all_ages);

    col.flush().unwrap();
    assert_eq!(
        pks(&col, "age = 10 OR age = 20"),
        vec!["d10".to_string(), "d20".to_string()]
    );
    assert_eq!(pks(&col, "age != 10 OR age != 20"), all_ages);

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_create_invert_index_builds_null_markers_for_is_null_pushdown() {
    let path = temp_dir("create_invert_null_markers");

    let schema = CollectionSchema::new("test")
        .with_field(
            FieldSchema::new("emb", DataType::VectorFp32)
                .nullable()
                .with_dimension(4),
        )
        .with_field(FieldSchema::new("tag", DataType::String).nullable());

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let statuses = col
        .insert(vec![
            // Missing tag => NULL
            Doc::new("d1").set("emb", vec![1.0f32, 0.0, 0.0, 0.0]),
            // Explicit null => NULL
            Doc::new("d2")
                .set("emb", vec![0.0f32, 1.0, 0.0, 0.0])
                .set("tag", Value::Null),
            // Non-null => IS NOT NULL
            Doc::new("d3")
                .set("emb", vec![0.0f32, 0.0, 1.0, 0.0])
                .set("tag", "x"),
        ])
        .unwrap();
    assert!(statuses.iter().all(|s| s.is_ok()));
    col.flush().unwrap();

    // Build invert index after the fact (DDL). This must include NULL/nonnull
    // markers so `IS NULL` pushdown stays correct.
    col.create_index(
        "tag",
        IndexParams::Invert(InvertIndexParams::default()),
        CreateIndexOptions::default(),
    )
    .unwrap();

    fn pks(col: &Collection, filter: &str) -> Vec<String> {
        let q = VectorQuery::new("emb", vec![0.0, 0.0, 0.0, 0.0], 10).with_filter(filter);
        let mut out: Vec<String> = col.query(q).unwrap().iter().map(|d| d.pk.clone()).collect();
        out.sort();
        out
    }

    assert_eq!(
        pks(&col, "tag IS NULL"),
        vec!["d1".to_string(), "d2".to_string()]
    );
    assert_eq!(pks(&col, "tag IS NOT NULL"), vec!["d3".to_string()]);

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_schema_default_max_docs_matches_compat_default() {
    let schema = CollectionSchema::new("test");
    assert_eq!(schema.max_doc_count_per_segment, MAX_DOC_COUNT_PER_SEGMENT);
}

#[test]
fn test_schema_validation_rejects_too_small_segment_limit() {
    let path = temp_dir("schema_small_seg");
    let schema = CollectionSchema::new("test")
        .with_field(
            FieldSchema::new("emb", DataType::VectorFp32)
                .nullable()
                .with_dimension(4),
        )
        .with_max_docs_per_segment(MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD - 1);
    assert!(Collection::create_and_open(&path, schema, CollectionOptions::default()).is_err());
}

#[test]
fn test_sql_filter_allows_keyword_field_names() {
    let path = temp_dir("keyword_fields_filter");

    let schema = CollectionSchema::new("test")
        .with_field(
            FieldSchema::new("dummy_vec", DataType::VectorFp32)
                .nullable()
                .with_dimension(1),
        )
        .with_field(FieldSchema::new("and", DataType::Int64).nullable())
        .with_field(FieldSchema::new("or", DataType::String).nullable())
        .with_field(FieldSchema::new("select", DataType::Int64).nullable());

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let statuses = col
        .insert(vec![
            Doc::new("d1")
                .set("and", 1_i64)
                .set("or", "x")
                .set("select", 9_i64),
            Doc::new("d2")
                .set("and", 2_i64)
                .set("or", "y")
                .set("select", 10_i64),
        ])
        .unwrap();
    assert!(statuses.iter().all(|s| s.is_ok()));

    // Filter-only query: empty field_name means "no vector query".
    fn pks(col: &Collection, filter: &str) -> Vec<String> {
        let q = VectorQuery::new("", Vec::<f32>::new(), 10).with_filter(filter);
        let mut out: Vec<String> = col.query(q).unwrap().iter().map(|d| d.pk.clone()).collect();
        out.sort();
        out
    }

    assert_eq!(pks(&col, "and = 1"), vec!["d1".to_string()]);
    assert_eq!(pks(&col, "or = 'y'"), vec!["d2".to_string()]);
    assert_eq!(pks(&col, "select != 10"), vec!["d1".to_string()]);

    // Ensure keyword operators still work: `OR` here is the boolean operator.
    assert_eq!(
        pks(&col, "or = 'x' OR and = 2"),
        vec!["d1".to_string(), "d2".to_string()]
    );

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_small_scalar_fields_reject_mismatched_value_variants() {
    let path = temp_dir("small_scalar_variants");
    let mut schema = CollectionSchema::new("test").with_field(
        FieldSchema::new("emb", DataType::VectorFp32)
            .nullable()
            .with_dimension(2),
    );
    let cases = [
        ("i8", DataType::Int8, Value::I8(7), Value::I64(7)),
        ("i16", DataType::Int16, Value::I16(7), Value::I64(7)),
        ("u8", DataType::Uint8, Value::U8(7), Value::I64(7)),
        ("u16", DataType::Uint16, Value::U16(7), Value::I64(7)),
        (
            "f16",
            DataType::Float16,
            Value::F16(f16::from_f32(1.5)),
            Value::F64(1.5),
        ),
    ];
    for (name, data_type, _, _) in &cases {
        schema = schema.with_field(FieldSchema::new(*name, *data_type).nullable());
    }
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let doc = |pk: &str, name: &str, value: &Value| {
        Doc::new(pk)
            .set("emb", vec![1.0f32, 0.0])
            .set(name, value.clone())
    };
    for (name, _, matching, mismatched) in &cases {
        let statuses = col
            .insert(vec![
                doc(&format!("ok_{name}"), name, matching),
                doc(&format!("bad_{name}"), name, mismatched),
            ])
            .unwrap();
        assert!(statuses[0].is_ok(), "{name}: {}", statuses[0].message);
        assert!(
            statuses[1].message.contains("type mismatch"),
            "{name}: {}",
            statuses[1].message
        );
    }
    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

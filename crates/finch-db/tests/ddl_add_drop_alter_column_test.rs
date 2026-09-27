mod common;
use common::*;

// ── DDL: add / drop / alter column ───────────────────────────────────────────

#[test]
fn test_add_column() {
    let path = temp_dir("add_col");
    let schema = basic_schema(4);
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![
        make_doc("d1", vec![1.0, 0.0, 0.0, 0.0], "a"),
        make_doc("d2", vec![0.0, 1.0, 0.0, 0.0], "b"),
    ])
    .unwrap();

    // add_column supports basic numeric scalar types (nullable by default).
    col.add_column(
        FieldSchema::new("num", DataType::Int32),
        AddColumnOptions::default(),
    )
    .unwrap();

    let info = col.schema_info();
    assert!(
        info.has_field("num"),
        "num field should exist after add_column"
    );

    // Backfilled column is NULL for existing docs.
    let fetched = col.fetch(vec!["d1".to_string(), "d2".to_string()]).unwrap();
    assert!(fetched["d1"].is_null("num"));
    assert!(fetched["d2"].is_null("num"));

    // Adding the same field again should fail
    let err = col.add_column(
        FieldSchema::new("num", DataType::Int32),
        AddColumnOptions::default(),
    );
    assert!(err.is_err());

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_add_column_with_expression_backfills_values_and_persists() {
    let path = temp_dir("add_col_expr");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("a", DataType::Int32))
        .with_field(FieldSchema::new("b", DataType::Int32));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    let statuses = col
        .insert(vec![
            Doc::new("d1")
                .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
                .set("a", 2i32)
                .set("b", 3i32),
            Doc::new("d2")
                .set("emb", vec![0.0f32, 1.0, 0.0, 0.0])
                .set("a", 10i32)
                .set("b", -4i32),
        ])
        .unwrap();
    assert!(
        statuses.iter().all(|s| s.is_ok()),
        "all inserts should succeed (got: {:?})",
        statuses
    );

    col.add_column_with_expression(
        FieldSchema::new("c", DataType::Int32),
        Some("a + b * 2"),
        AddColumnOptions::default(),
    )
    .unwrap();

    let fetched = col.fetch(vec!["d1".to_string(), "d2".to_string()]).unwrap();
    assert_eq!(fetched["d1"].get("c"), Some(&Value::I32(8)));
    assert_eq!(fetched["d2"].get("c"), Some(&Value::I32(2)));

    drop(col);
    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    let fetched = col.fetch(vec!["d1".to_string(), "d2".to_string()]).unwrap();
    assert_eq!(fetched["d1"].get("c"), Some(&Value::I32(8)));
    assert_eq!(fetched["d2"].get("c"), Some(&Value::I32(2)));

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_add_column_not_null_requires_expression() {
    let path = temp_dir("add_col_not_null_requires_expr");
    let schema = basic_schema(4);
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    let result = col.add_column_with_expression(
        FieldSchema::new("x", DataType::Int32).not_null(),
        None,
        AddColumnOptions::default(),
    );
    assert!(
        result.is_err(),
        "non-null add_column must provide expression"
    );

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_drop_column() {
    let path = temp_dir("drop_col");
    let schema = basic_schema(4);
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.add_column(
        FieldSchema::new("x", DataType::Int32),
        AddColumnOptions::default(),
    )
    .unwrap();
    col.drop_column("x").unwrap();
    let info = col.schema_info();
    assert!(
        !info.has_field("x"),
        "x field should be gone after drop_column"
    );

    // Dropping non-existent field should fail
    assert!(col.drop_column("nonexistent").is_err());

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_alter_column_rename() {
    let path = temp_dir("alter_col");
    let schema = basic_schema(4);
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.add_column(
        FieldSchema::new("x", DataType::Int32),
        AddColumnOptions::default(),
    )
    .unwrap();
    col.alter_column("x", Some("category"), None, AlterColumnOptions::default())
        .unwrap();
    let info = col.schema_info();
    assert!(!info.has_field("x"), "old name should be gone");
    assert!(info.has_field("category"), "new name should exist");

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_add_column_manifest_failure_keeps_old_schema_and_segments() {
    let path = temp_dir("add_col_manifest_failure");
    let schema = basic_schema(4);
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![make_doc("d1", vec![1.0, 0.0, 0.0, 0.0], "a")])
        .unwrap();
    col.flush().unwrap();
    let before_seg_dirs: std::collections::BTreeSet<_> = std::fs::read_dir(&path)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|name| name.starts_with("seg_"))
        .collect();

    std::fs::create_dir(path.join("manifest.tmp")).unwrap();
    let result = col.add_column(
        FieldSchema::new("x", DataType::Int32),
        AddColumnOptions::default(),
    );
    assert!(result.is_err(), "manifest commit failure must fail DDL");
    std::fs::remove_dir_all(path.join("manifest.tmp")).unwrap();
    drop(col);

    let after_seg_dirs: std::collections::BTreeSet<_> = std::fs::read_dir(&path)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|name| name.starts_with("seg_"))
        .collect();
    assert_eq!(after_seg_dirs, before_seg_dirs);

    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    assert!(!col.schema_info().has_field("x"));
    let fetched = col.fetch(vec!["d1".to_string()]).unwrap();
    assert!(fetched.contains_key("d1"));

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_alter_column_manifest_failure_keeps_old_schema_and_segments() {
    let path = temp_dir("alter_col_manifest_failure");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("score", DataType::Int32));
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![Doc::new("d1")
        .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
        .set("score", 7i32)])
        .unwrap();
    col.flush().unwrap();
    let before_seg_dirs: std::collections::BTreeSet<_> = std::fs::read_dir(&path)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|name| name.starts_with("seg_"))
        .collect();

    std::fs::create_dir(path.join("manifest.tmp")).unwrap();
    let result = col.alter_column("score", Some("rank"), None, AlterColumnOptions::default());
    assert!(result.is_err(), "manifest commit failure must fail DDL");
    std::fs::remove_dir_all(path.join("manifest.tmp")).unwrap();
    drop(col);

    let after_seg_dirs: std::collections::BTreeSet<_> = std::fs::read_dir(&path)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|name| name.starts_with("seg_"))
        .collect();
    assert_eq!(after_seg_dirs, before_seg_dirs);

    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    let info = col.schema_info();
    assert!(info.has_field("score"));
    assert!(!info.has_field("rank"));

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_add_and_alter_invert_indexed_numeric_columns_reopen() {
    let path = temp_dir("ddl_invert_reopen");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(
            FieldSchema::new("score", DataType::Int64)
                .with_index(IndexParams::Invert(InvertIndexParams::default())),
        );
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![
        Doc::new("d1")
            .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
            .set("score", 1i64),
        Doc::new("d2")
            .set("emb", vec![0.0f32, 1.0, 0.0, 0.0])
            .set("score", 2i64),
    ])
    .unwrap();
    col.flush().unwrap();

    col.add_column_with_expression(
        FieldSchema::new("bonus", DataType::Int64)
            .with_index(IndexParams::Invert(InvertIndexParams::default())),
        Some("score + 10"),
        AddColumnOptions::default(),
    )
    .unwrap();
    col.alter_column("score", Some("rank"), None, AlterColumnOptions::default())
        .unwrap();
    drop(col);

    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    let q = VectorQuery::new("", Vec::<f32>::new(), 10).with_filter("bonus = 12 AND rank = 2");
    let results = col.query(q).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].pk, "d2");

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_alter_column_rename_rebuilds_empty_vector_indexes() {
    let dense = |params: Option<IndexParams>| {
        let field = FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4);
        let query = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 5);
        (params.map_or(field.clone(), |p| field.with_index(p)), query)
    };
    let sparse = |params: IndexParams| {
        let mut query = VectorQuery::new("emb", Vec::new(), 5);
        query.sparse_indices = vec![0];
        query.sparse_values = vec![1.0];
        (
            FieldSchema::new("emb", DataType::SparseFp32).with_index(params),
            query,
        )
    };
    let cases = [
        dense(None),
        dense(Some(IndexParams::Flat(FlatIndexParams::new(
            MetricType::L2,
        )))),
        dense(Some(IndexParams::Hnsw(HnswIndexParams::new(
            MetricType::L2,
        )))),
        dense(Some(IndexParams::Ivf(IvfIndexParams::new(MetricType::L2)))),
        sparse(IndexParams::FlatSparse(FlatIndexParams::new(
            MetricType::InnerProduct,
        ))),
        sparse(IndexParams::HnswSparse(HnswIndexParams::new(
            MetricType::InnerProduct,
        ))),
    ];
    for (vector_field, query) in cases {
        let path = temp_dir("alter_empty_vec");
        let schema = CollectionSchema::new("test")
            .with_field(vector_field.clone())
            .with_field(FieldSchema::new("weight", DataType::Float64));
        let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
        col.insert(vec![Doc::new("1").set("weight", 80.5f64)])
            .unwrap();
        col.flush().unwrap();

        col.alter_column("weight", Some("mass"), None, AlterColumnOptions::default())
            .unwrap_or_else(|e| panic!("{:?}: {e:?}", vector_field.index_params));

        let fetched = col.fetch(vec!["1".to_string()]).unwrap();
        assert_eq!(fetched["1"].fields["mass"].as_f64(), Some(80.5));
        assert!(col.query(query).unwrap().is_empty());

        drop(col);
        std::fs::remove_dir_all(&path).ok();
    }
}

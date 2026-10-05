mod common;
use common::*;

#[test]
fn test_create_flat_index_and_search() {
    let path = temp_dir("flat_index");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("label", DataType::String));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![
        make_doc("d1", vec![1.0, 0.0, 0.0, 0.0], "a"),
        make_doc("d2", vec![0.0, 1.0, 0.0, 0.0], "b"),
        make_doc("d3", vec![0.0, 0.0, 1.0, 0.0], "c"),
    ])
    .unwrap();

    col.create_index(
        "emb",
        IndexParams::Flat(FlatIndexParams {
            metric: MetricType::L2,
            quantize: Default::default(),
            column_major: false,
        }),
        CreateIndexOptions::default(),
    )
    .unwrap();

    // After create_index, the writing segment is flushed and index is built
    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 2);
    let results = col.query(q).unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].pk, "d1");

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_create_index_failure_does_not_commit_schema_metadata() {
    let path = temp_dir("index_failure_no_metadata");
    let schema = basic_schema(4);
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    col.insert(vec![make_doc("d1", vec![1.0, 0.0, 0.0, 0.0], "a")])
        .unwrap();
    col.flush().unwrap();

    let result = col.create_index(
        "label",
        IndexParams::Flat(FlatIndexParams {
            metric: MetricType::L2,
            quantize: Default::default(),
            column_major: false,
        }),
        CreateIndexOptions::default(),
    );
    assert!(result.is_err());
    assert!(col
        .schema_info()
        .get_field("label")
        .and_then(|f| f.index_params.as_ref())
        .is_none());
    drop(col);

    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    assert!(col
        .schema_info()
        .get_field("label")
        .and_then(|f| f.index_params.as_ref())
        .is_none());

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_optimize_index_build_failure_does_not_commit_segment_metadata() {
    let path = temp_dir("optimize_index_failure_no_metadata");
    let schema = CollectionSchema::new("test").with_field(
        FieldSchema::new("emb", DataType::VectorFp32)
            .with_dimension(4)
            .with_index(IndexParams::Hnsw(HnswIndexParams::new(MetricType::L2))),
    );
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let statuses = col
        .insert(vec![Doc::new("d1").set("emb", vec![1.0f32, 0.0, 0.0, 0.0])])
        .unwrap();
    assert!(statuses.iter().all(|s| s.is_ok()), "statuses={statuses:?}");
    col.flush().unwrap();

    std::fs::write(
        path.join("seg_0").join("idx_emb"),
        b"not an index directory",
    )
    .unwrap();
    assert!(col.optimize(OptimizeOptions::default()).is_err());
    drop(col);

    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    let stats = col.stats().unwrap();
    assert_eq!(stats.doc_count, 1);

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn vector_index_on_a_scalar_field_is_rejected_in_an_empty_collection() {
    // Invalid index kinds must be rejected even when there are no segments to build.
    let path = temp_dir("vector_index_on_scalar_field");
    let schema =
        CollectionSchema::new("test").with_field(FieldSchema::new("value", DataType::Int64));
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let result = col.create_index(
        "value",
        IndexParams::Flat(FlatIndexParams {
            metric: MetricType::L2,
            quantize: Default::default(),
            column_major: false,
        }),
        CreateIndexOptions::default(),
    );
    let validation = col.schema_info().validate();
    drop(col);
    let reopened_validation = Collection::open(&path, CollectionOptions::default())
        .map(|reopened| reopened.schema_info().validate());
    std::fs::remove_dir_all(path).unwrap();
    assert!(result.is_err(), "index creation returned {result:?}; schema validation: {validation:?}; reopened schema validation: {reopened_validation:?}");
    assert!(validation.is_ok());
    assert!(matches!(reopened_validation, Ok(Ok(()))));
}

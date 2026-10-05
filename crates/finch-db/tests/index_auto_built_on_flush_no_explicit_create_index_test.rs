mod common;
use common::*;

#[test]
fn test_index_auto_built_on_flush() {
    let path = temp_dir("auto_idx");
    // Schema has no explicit index params: assign_default_index_params adds Flat
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![
        Doc::new("d1").set("emb", vec![1.0f32, 0.0, 0.0, 0.0]),
        Doc::new("d2").set("emb", vec![0.0f32, 1.0, 0.0, 0.0]),
    ])
    .unwrap();

    // Flush: should auto-build Flat index from default schema params
    col.flush().unwrap();

    // Schema should record the default index params
    let info = col.schema_info();
    assert!(
        info.get_field("emb")
            .map(|f| f.index_params.is_some())
            .unwrap_or(false),
        "default index params should be present after create"
    );

    // Query should work correctly from the persisted segment
    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 2);
    let results = col.query(q).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].pk, "d1");

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

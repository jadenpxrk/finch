mod common;
use common::*;

// ── Default index params assigned at create time ──────────────────────────────

#[test]
fn test_default_index_params_assigned() {
    let path = temp_dir("default_params");
    // No explicit index params in schema
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("label", DataType::String));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let info = col.schema_info();

    // Vector field should have default Flat params assigned
    let emb = info.get_field("emb").expect("emb field missing");
    assert!(
        emb.index_params.is_some(),
        "vector field should get default index params"
    );
    match emb.index_params.as_ref().unwrap() {
        IndexParams::Flat(p) => {
            assert_eq!(
                p.metric,
                MetricType::InnerProduct,
                "default dense Flat metric should be InnerProduct"
            );
        }
        _ => panic!("default index for dense vector should be Flat"),
    }

    // Scalar field should have no index params
    let label = info.get_field("label").expect("label field missing");
    assert!(
        label.index_params.is_none(),
        "scalar field should not get default index params"
    );

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

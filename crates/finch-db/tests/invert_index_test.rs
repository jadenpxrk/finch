mod common;
use common::*;

// ── Invert index ──────────────────────────────────────────────────────────────

#[test]
fn test_invert_index_filter() {
    let path = temp_dir("invert");
    // Invert index is declared in schema (built inline during insert), not via create_index
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(
            FieldSchema::new("category", DataType::String)
                .with_index(IndexParams::Invert(InvertIndexParams::default())),
        );

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![
        Doc::new("d1")
            .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
            .set("category", "science"),
        Doc::new("d2")
            .set("emb", vec![0.9f32, 0.1, 0.0, 0.0])
            .set("category", "art"),
        Doc::new("d3")
            .set("emb", vec![0.8f32, 0.2, 0.0, 0.0])
            .set("category", "science"),
    ])
    .unwrap();

    // Filter-based query should work correctly with invert index
    let q =
        VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("category = 'science'");
    let results = col.query(q).unwrap();

    assert!(!results.is_empty());
    for doc in &results {
        assert_eq!(doc.get_str("category").unwrap(), "science");
    }
    assert!(results.iter().all(|d| d.pk != "d2"));

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

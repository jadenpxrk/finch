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

#[test]
fn test_writing_segment_invert_index_lives_in_memory_and_freezes_on_flush() {
    let path = temp_dir("invert_writing_in_memory");
    let range_and_suffix = InvertIndexParams {
        enable_range_optimization: true,
        enable_extended_wildcard: true,
    };
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(
            FieldSchema::new("category", DataType::String)
                .nullable()
                .with_index(IndexParams::Invert(range_and_suffix.clone())),
        )
        .with_field(
            FieldSchema::new("n", DataType::Int64)
                .with_index(IndexParams::Invert(range_and_suffix)),
        );
    let doc = |pk: &str, category: Option<&str>, n: i64| {
        let d = Doc::new(pk)
            .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
            .set("n", n);
        match category {
            Some(c) => d.set("category", c),
            None => d,
        }
    };
    let filters = [
        "category = 'science'",
        "category LIKE '%nce'",
        "category LIKE 'sc%'",
        "category IS NULL",
        "category IS NOT NULL",
        "n >= 2",
        "n < 3 AND category != 'art'",
    ];
    let answers = |col: &Collection| -> Vec<Vec<String>> {
        filters
            .iter()
            .map(|f| {
                let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter(*f);
                let mut pks: Vec<String> =
                    col.query(q).unwrap().iter().map(|d| d.pk.clone()).collect();
                pks.sort();
                pks
            })
            .collect()
    };
    let category_invert = path.join("seg_0").join("category_invert");

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    col.insert(vec![
        doc("d1", Some("science"), 1),
        doc("d2", Some("art"), 2),
        doc("d3", Some("science"), 3),
        doc("d4", None, 4),
    ])
    .unwrap();
    // The upsert removes d2's old postings.
    col.upsert(vec![doc("d2", Some("finance"), 2)]).unwrap();
    let expected = answers(&col);
    assert_eq!(expected[0], vec!["d1", "d3"]);
    assert_eq!(expected[1], vec!["d1", "d2", "d3"]);
    assert_eq!(expected[3], vec!["d4"]);
    assert!(
        !category_invert.exists(),
        "writing segment kept its index on disk"
    );
    drop(col);

    // WAL replay rebuilds the in-memory index.
    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    assert_eq!(answers(&col), expected);
    col.flush().unwrap();
    assert!(path.join("seg_0").join("category_invert.keys").is_file());
    assert!(!category_invert.exists());
    assert_eq!(answers(&col), expected);
    drop(col);

    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    assert_eq!(answers(&col), expected);
    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

mod common;
use common::*;

// ── Persistence / reopen ──────────────────────────────────────────────────────

#[test]
fn test_reopen_collection() {
    let path = temp_dir("reopen");
    let schema = basic_schema(4);

    // Create and populate
    {
        let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
        col.insert(vec![make_doc("d1", vec![1.0, 0.0, 0.0, 0.0], "alpha")])
            .unwrap();
        col.flush().unwrap();
        // col dropped here, lock released
    }

    // Reopen and verify
    {
        let col = Collection::open(&path, CollectionOptions::default()).unwrap();
        let stats = col.stats().unwrap();
        assert_eq!(stats.doc_count, 1);

        let fetched = col.fetch(vec!["d1".to_string()]).unwrap();
        assert!(fetched.contains_key("d1"));

        let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 1);
        let results = col.query(q).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].pk, "d1");
    }

    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_reopen_preserves_schema() {
    let path = temp_dir("reopen_schema");
    let schema = CollectionSchema::new("my_col")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(8))
        .with_field(FieldSchema::new("title", DataType::String))
        .with_field(FieldSchema::new("score", DataType::Float32));

    {
        let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
        drop(col);
    }

    {
        let col = Collection::open(&path, CollectionOptions::default()).unwrap();
        let info = col.schema_info();
        assert_eq!(info.name, "my_col");
        assert!(info.has_field("emb"));
        assert!(info.has_field("title"));
        assert!(info.has_field("score"));
    }

    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_reopen_preserves_invert_extended_wildcard_param() {
    let path = temp_dir("reopen_invert_ext_wc");
    let inv = InvertIndexParams {
        enable_range_optimization: false,
        enable_extended_wildcard: true,
    };

    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(
            FieldSchema::new("label", DataType::String)
                .not_null()
                .with_index(IndexParams::Invert(inv.clone())),
        );

    {
        let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
        col.flush().unwrap();
        drop(col);
    }

    {
        let col = Collection::open(&path, CollectionOptions::default()).unwrap();
        let info = col.schema_info();
        let params = info
            .get_field("label")
            .and_then(|f| f.index_params.clone())
            .expect("label should have an index");
        let IndexParams::Invert(loaded) = params else {
            panic!("expected invert index params");
        };
        assert_eq!(
            loaded.enable_range_optimization,
            inv.enable_range_optimization
        );
        assert_eq!(
            loaded.enable_extended_wildcard,
            inv.enable_extended_wildcard
        );
    }

    std::fs::remove_dir_all(&path).ok();
}

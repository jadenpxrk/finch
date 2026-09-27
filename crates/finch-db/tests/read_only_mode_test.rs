mod common;
use common::*;

// ── Read-only mode ────────────────────────────────────────────────────────────

#[test]
fn test_read_only_rejects_mutations() {
    let path = temp_dir("readonly");
    let schema = basic_schema(4);

    {
        let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
        col.insert(vec![make_doc("d1", vec![1.0, 0.0, 0.0, 0.0], "a")])
            .unwrap();
        col.flush().unwrap();
    }

    let opts = CollectionOptions {
        read_only: true,
        enable_mmap: false,
        index_storage: None,
        forward_storage: None,
        max_buffer_size: CollectionOptions::DEFAULT_MAX_BUFFER_SIZE,
        forward_file_format: None,
    };
    let col = Collection::open(&path, opts).unwrap();

    // All mutating operations should fail
    assert!(col
        .insert(vec![make_doc("d2", vec![0.0, 1.0, 0.0, 0.0], "b")])
        .is_err());
    assert!(col.delete(vec!["d1".to_string()]).is_err());
    assert!(col.flush().is_err());

    // Reads should work fine
    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 1);
    let results = col.query(q).unwrap();
    assert_eq!(results[0].pk, "d1");

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_read_only_allows_multiple_open() {
    let path = temp_dir("readonly_multi_open");
    let schema = basic_schema(4);

    {
        let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
        col.insert(vec![make_doc("d1", vec![1.0, 0.0, 0.0, 0.0], "a")])
            .unwrap();
        col.flush().unwrap();
    }

    let opts = CollectionOptions {
        read_only: true,
        enable_mmap: false,
        index_storage: None,
        forward_storage: None,
        max_buffer_size: CollectionOptions::DEFAULT_MAX_BUFFER_SIZE,
        forward_file_format: None,
    };
    let col1 = Collection::open(&path, opts.clone()).unwrap();
    let col2 = Collection::open(&path, opts).unwrap();

    let s1 = col1.stats().unwrap();
    let s2 = col2.stats().unwrap();
    assert_eq!(s1.doc_count, 1);
    assert_eq!(s2.doc_count, 1);

    drop(col2);
    drop(col1);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_read_only_allows_multiple_open_with_invert_index() {
    let path = temp_dir("readonly_multi_open_invert");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(
            FieldSchema::new("id", DataType::Int64)
                .with_index(IndexParams::Invert(InvertIndexParams::default())),
        );

    {
        let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
        col.insert(vec![
            Doc::new("d1")
                .set("id", 1_i64)
                .set("emb", vec![1.0f32, 0.0, 0.0, 0.0]),
            Doc::new("d2")
                .set("id", 2_i64)
                .set("emb", vec![0.0f32, 1.0, 0.0, 0.0]),
        ])
        .unwrap();
        col.flush().unwrap();
    }

    let opts = CollectionOptions {
        read_only: true,
        enable_mmap: false,
        index_storage: None,
        forward_storage: None,
        max_buffer_size: CollectionOptions::DEFAULT_MAX_BUFFER_SIZE,
        forward_file_format: None,
    };

    let col1 = Collection::open(&path, opts.clone()).unwrap();
    let col2 = Collection::open(&path, opts).unwrap();

    let s1 = col1.stats().unwrap();
    let s2 = col2.stats().unwrap();
    assert_eq!(s1.doc_count, 2);
    assert_eq!(s2.doc_count, 2);

    drop(col2);
    drop(col1);
    std::fs::remove_dir_all(&path).ok();
}

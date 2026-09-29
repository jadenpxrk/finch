mod common;
use common::*;

// ── Flush and persisted segment search ────────────────────────────────────────

#[test]
fn test_flush_creates_persisted_segment() {
    let path = temp_dir("flush");
    let schema = basic_schema(4);
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![make_doc("d1", vec![1.0, 0.0, 0.0, 0.0], "alpha")])
        .unwrap();

    let stats_before = col.stats().unwrap();
    assert_eq!(stats_before.segment_count, 1); // Only writing segment

    col.flush().unwrap();

    let stats_after = col.stats().unwrap();
    assert_eq!(stats_after.segment_count, 2); // Writing + persisted

    // Doc should still be queryable from persisted segment
    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 1);
    let results = col.query(q).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].pk, "d1");

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_flush_next_segment_failure_does_not_advance_manifest() {
    let path = temp_dir("flush_next_segment_failure");
    let schema = CollectionSchema::new("test")
        .with_field(
            FieldSchema::new("id", DataType::Int64).with_index(IndexParams::Invert(
                InvertIndexParams {
                    enable_range_optimization: true,
                    enable_extended_wildcard: false,
                },
            )),
        )
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4));
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    let statuses = col
        .insert(vec![Doc::new("d1")
            .set("id", 1i64)
            .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])])
        .unwrap();
    assert!(statuses.iter().all(|s| s.is_ok()), "statuses={statuses:?}");

    // A directory where the next writing segment's WAL goes makes creating it fail.
    std::fs::create_dir(path.join("wal_1.log")).unwrap();
    assert!(col.flush().is_err());
    drop(col);
    std::fs::remove_dir(path.join("wal_1.log")).unwrap();

    let read_only = CollectionOptions {
        read_only: true,
        ..CollectionOptions::default()
    };
    let err = match Collection::open(&path, read_only) {
        Ok(_) => panic!("read-only open should require WAL recovery"),
        Err(err) => err,
    };
    assert!(err.message.contains("pending WAL recovery"), "err={err:?}");

    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    let fetched = col.fetch(vec!["d1".to_string()]).unwrap();
    assert!(fetched.contains_key("d1"));
    col.flush().unwrap();
    drop(col);

    let read_only = CollectionOptions {
        read_only: true,
        ..CollectionOptions::default()
    };
    Collection::open(&path, read_only).unwrap();
    std::fs::remove_dir_all(&path).ok();
}

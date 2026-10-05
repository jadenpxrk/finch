mod common;
use common::*;

#[test]
fn test_wal_delete_replay_syncs_version() {
    let path = temp_dir("wal_del_replay");
    let schema = basic_schema(4);

    // Insert, flush to persisted, then delete without flushing: delete lives only in WAL
    {
        let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
        col.insert(vec![
            make_doc("d1", vec![1.0, 0.0, 0.0, 0.0], "a"),
            make_doc("d2", vec![0.0, 1.0, 0.0, 0.0], "b"),
            make_doc("d3", vec![0.0, 0.0, 1.0, 0.0], "c"),
        ])
        .unwrap();
        col.flush().unwrap();
        col.delete(vec!["d2".to_string()]).unwrap();
        // Drop without flushing: d2's delete is in the WAL
    }

    // Reopen: WAL replay must restore the delete
    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    assert_eq!(
        col.stats().unwrap().doc_count,
        2,
        "deleted doc should not count"
    );

    let q = VectorQuery::new("emb", vec![0.0, 1.0, 0.0, 0.0], 3);
    let results = col.query(q).unwrap();
    assert!(
        results.iter().all(|d| d.pk != "d2"),
        "d2 must not appear after WAL delete replay"
    );

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_wal_delete_replay_fails_open_when_manifest_flush_fails() {
    let path = temp_dir("wal_del_replay_manifest_fail");
    let schema = basic_schema(4);
    {
        let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
        col.insert(vec![
            make_doc("d1", vec![1.0, 0.0, 0.0, 0.0], "a"),
            make_doc("d2", vec![0.0, 1.0, 0.0, 0.0], "b"),
        ])
        .unwrap();
        col.flush().unwrap();
        col.delete(vec!["d2".to_string()]).unwrap();
    }

    // A directory where the manifest temp file goes makes the manifest flush fail.
    let blocker = path.join("manifest.tmp");
    std::fs::create_dir(&blocker).unwrap();
    assert!(Collection::open(&path, CollectionOptions::default()).is_err());

    std::fs::remove_dir(&blocker).unwrap();
    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    assert_eq!(col.stats().unwrap().doc_count, 1);
    let q = VectorQuery::new("emb", vec![0.0, 1.0, 0.0, 0.0], 2);
    assert!(col.query(q).unwrap().iter().all(|d| d.pk != "d2"));

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

mod common;
use common::*;

// ── WAL crash recovery ────────────────────────────────────────────────────────

#[test]
fn test_wal_crash_recovery() {
    let path = temp_dir("wal_recovery");
    let schema = basic_schema(4);

    // Insert without flushing: data lives only in the WAL
    {
        let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
        col.insert(vec![
            make_doc("d1", vec![1.0, 0.0, 0.0, 0.0], "a"),
            make_doc("d2", vec![0.0, 1.0, 0.0, 0.0], "b"),
        ])
        .unwrap();
        // Drop without flush: simulates a crash
    }

    // Reopen: WAL should be replayed
    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    assert_eq!(
        col.stats().unwrap().doc_count,
        2,
        "WAL replay should restore all docs"
    );

    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 2);
    let results = col.query(q).unwrap();
    assert_eq!(
        results.len(),
        2,
        "both WAL-recovered docs should be queryable"
    );

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_legacy_wal_log_is_migrated_and_replayed() {
    let path = temp_dir("legacy_wal_migrate");
    let schema = basic_schema(4);

    {
        let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
        col.insert(vec![make_doc("d1", vec![1.0, 0.0, 0.0, 0.0], "a")])
            .unwrap();
        // Drop without flush: simulates a crash.
    }

    // Simulate older Finch which used a single `wal.log`.
    let wal0 = path.join("wal_0.log");
    let legacy = path.join("wal.log");
    std::fs::rename(&wal0, &legacy).unwrap();

    // Reopen: should migrate wal.log -> wal_0.log and replay it.
    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    assert_eq!(col.stats().unwrap().doc_count, 1);
    assert!(col
        .fetch(vec!["d1".to_string()])
        .unwrap()
        .contains_key("d1"));

    // Ensure the migrated filename exists (migration happened).
    assert!(path.join("wal_0.log").exists());

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_wal_versioning_ignores_old_wal_files() {
    use serde::Serialize;

    #[derive(Serialize)]
    enum TestWalOp {
        Insert,
    }

    #[derive(Serialize)]
    struct TestWalEntry {
        op: TestWalOp,
        doc_id: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        prev_doc_id: Option<u64>,
        pk: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        doc: Option<Doc>,
    }

    let path = temp_dir("wal_versioned_ignore");
    let schema = basic_schema(4);

    {
        let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
        col.insert(vec![make_doc("d1", vec![1.0, 0.0, 0.0, 0.0], "a")])
            .unwrap();
        col.flush().unwrap();
    }

    // After flush, the active writing-segment id advances and Finch will replay
    // only that WAL file. A stale `wal_0.log` should not be replayed.
    let stale = TestWalEntry {
        op: TestWalOp::Insert,
        doc_id: 1,
        prev_doc_id: None,
        pk: "d2".to_string(),
        doc: Some(make_doc("d2", vec![0.0, 1.0, 0.0, 0.0], "b")),
    };
    let stale_json = serde_json::to_string(&stale).unwrap();
    std::fs::write(path.join("wal_0.log"), format!("{}\n", stale_json)).unwrap();

    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    assert_eq!(col.stats().unwrap().doc_count, 1);
    assert!(!col
        .fetch(vec!["d2".to_string()])
        .unwrap()
        .contains_key("d2"));

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_wal_replay_rebuilds_id_map_if_missing() {
    let path = temp_dir("wal_rebuild_idmap");
    let schema = basic_schema(4);

    // Insert without flushing: data lives only in the WAL.
    {
        let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
        col.insert(vec![make_doc("d1", vec![1.0, 0.0, 0.0, 0.0], "a")])
            .unwrap();
        // Drop without flush: simulates a crash.
    }

    // Simulate missing/corrupted id_map state on disk.
    let _ = std::fs::remove_dir_all(path.join("id_map"));

    // Reopen: WAL replay should restore both docs and PK->doc_id mapping.
    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    assert_eq!(col.stats().unwrap().doc_count, 1);

    let fetched = col.fetch(vec!["d1".to_string()]).unwrap();
    assert!(
        fetched.contains_key("d1"),
        "fetch must succeed after WAL replay rebuilds id_map"
    );

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_wal_replay_applies_update_tombstone_and_pk_remap() {
    let path = temp_dir("wal_update_tombstone");
    let schema = basic_schema(4);

    // Insert then update without flushing: both ops live only in the WAL.
    {
        let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
        col.insert(vec![make_doc("d1", vec![1.0, 0.0, 0.0, 0.0], "old")])
            .unwrap();
        col.update(vec![Doc::new("d1").set("label", "new")])
            .unwrap();
        // Drop without flush: simulates a crash.
    }

    // Reopen: WAL should replay insert + update, remap PK to the new doc_id,
    // and tombstone the old doc_id so it never surfaces in queries.
    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    assert_eq!(col.stats().unwrap().doc_count, 1);

    let fetched = col.fetch(vec!["d1".to_string()]).unwrap();
    assert_eq!(
        fetched.get("d1").and_then(|d| d.get_str("label")),
        Some("new")
    );

    let q_old = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 5).with_filter("label = 'old'");
    let got_old = col.query(q_old).unwrap();
    assert!(got_old.is_empty());

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_wal_with_corrupt_header_fails_open_and_is_kept() {
    let path = temp_dir("wal_corrupt_header");
    {
        let col = Collection::create_and_open(&path, basic_schema(4), CollectionOptions::default())
            .unwrap();
        col.insert(vec![make_doc("d1", vec![1.0, 0.0, 0.0, 0.0], "a")])
            .unwrap();
    }
    let wal0 = path.join("wal_0.log");
    let mut bytes = std::fs::read(&wal0).unwrap();
    assert!(bytes.len() > 64, "the WAL must hold a header and a record");
    bytes[0] = 7;
    std::fs::write(&wal0, &bytes).unwrap();

    assert!(Collection::open(&path, CollectionOptions::default()).is_err());
    assert_eq!(std::fs::read(&wal0).unwrap(), bytes);

    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_wal_shorter_than_header_fails_open_and_is_kept() {
    let path = temp_dir("wal_short");
    drop(
        Collection::create_and_open(&path, basic_schema(4), CollectionOptions::default()).unwrap(),
    );
    let wal0 = path.join("wal_0.log");
    std::fs::write(&wal0, [7u8; 10]).unwrap();

    let Err(err) = Collection::open(&path, CollectionOptions::default()) else {
        panic!("a WAL shorter than its header must fail the open");
    };
    assert!(err.message.contains("wal_0.log"), "{}", err.message);
    assert!(err.message.contains("10 bytes"), "{}", err.message);
    assert_eq!(std::fs::read(&wal0).unwrap(), [7u8; 10]);

    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_empty_wal_is_reused_as_fresh_log() {
    let path = temp_dir("wal_empty");
    drop(
        Collection::create_and_open(&path, basic_schema(4), CollectionOptions::default()).unwrap(),
    );
    let wal0 = path.join("wal_0.log");
    std::fs::write(&wal0, []).unwrap();

    {
        let col = Collection::open(&path, CollectionOptions::default()).unwrap();
        col.insert(vec![make_doc("d1", vec![1.0, 0.0, 0.0, 0.0], "a")])
            .unwrap();
    }

    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    assert_eq!(col.stats().unwrap().doc_count, 1);

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

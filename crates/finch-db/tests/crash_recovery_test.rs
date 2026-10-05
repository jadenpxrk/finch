//! Crash/restart durability tests (WAL replay + delete replay).
//!
//! These tests spawn the current integration-test binary as a child process
//! to simulate an abrupt process termination without running Rust destructors.

use std::process::Command;

use finch_db::Collection;
use finch_types::{
    CollectionOptions, CollectionSchema, DataType, Doc, FieldSchema, IndexParams,
    InvertIndexParams, VectorQuery,
};

const ENV_CRASH_PATH: &str = "FINCH_CRASH_PATH";

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("finch_crash_test_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn crash_schema() -> CollectionSchema {
    CollectionSchema::new("crash")
        .with_field(
            FieldSchema::new("id", DataType::Int64).with_index(IndexParams::Invert(
                InvertIndexParams {
                    enable_range_optimization: true,
                    enable_extended_wildcard: false,
                },
            )),
        )
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
}

#[test]
fn test_wal_recovery_after_crash_subprocess() {
    let path = temp_dir("wal_replay");

    let exe = std::env::current_exe().expect("current_exe");
    let status = Command::new(exe)
        .arg("--ignored")
        .arg("--exact")
        .arg("crash_helper_write_wal_and_exit")
        .env(ENV_CRASH_PATH, path.to_string_lossy().to_string())
        .status()
        .expect("spawn helper");
    assert!(status.success(), "helper exit status: {status}");

    let col = Collection::open(&path, CollectionOptions::default()).expect("open after crash");
    let stats = col.stats().expect("stats");
    assert_eq!(stats.doc_count, 2, "expected delete replay to apply");

    // Deleted doc should not appear in results.
    let q = VectorQuery::new("emb", vec![1.0f32, 0.0, 0.0, 0.0], 10);
    let results = col.query(q).expect("query");
    let pks: Vec<String> = results.iter().map(|d| d.pk.clone()).collect();
    assert!(pks.contains(&"a".to_string()));
    assert!(pks.contains(&"b".to_string()));
    assert!(!pks.contains(&"c".to_string()));
}

/// Helper test run in a child process.
///
/// Marked ignored so it never runs as part of the normal test suite.
#[test]
#[ignore]
fn crash_helper_write_wal_and_exit() {
    let path = std::env::var(ENV_CRASH_PATH).expect("FINCH_CRASH_PATH must be set");
    let path = std::path::PathBuf::from(path);

    // Ensure WAL is flushed/fsynced for durability in this test.
    let cfg = finch_types::GlobalConfigData {
        wal_flush_every_docs: 1,
        wal_fsync_every_docs: 1,
        query_thread_count: 1,
        optimize_thread_count: 1,
        ..finch_types::GlobalConfigData::default()
    };
    finch_db::initialize_global_config(cfg).expect("init global config");

    let schema = crash_schema();
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default())
        .expect("create_and_open");

    let docs = vec![
        Doc::new("a")
            .set("id", 0i64)
            .set("emb", vec![1.0f32, 0.0, 0.0, 0.0]),
        Doc::new("b")
            .set("id", 1i64)
            .set("emb", vec![0.0f32, 1.0, 0.0, 0.0]),
        Doc::new("c")
            .set("id", 2i64)
            .set("emb", vec![0.0f32, 0.0, 1.0, 0.0]),
    ];
    let statuses = col.insert(docs).expect("insert");
    assert!(
        statuses.iter().all(|s| s.is_ok()),
        "insert statuses: {statuses:?}"
    );

    // Delete one doc and ensure the delete is also WAL-recorded.
    col.delete(vec!["c".to_string()]).expect("delete");

    // Exit immediately without dropping the collection (simulates a crash).
    std::process::exit(0);
}

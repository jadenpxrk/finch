mod common;
use common::*;

use std::path::Path;

// A read-only collection directory makes every new file there fail: segment dumps and delete
// bitmap snapshots. Open files such as the WAL keep working.
fn set_dir_readonly(path: &Path, readonly: bool) -> bool {
    let mut perms = std::fs::metadata(path).unwrap().permissions();
    perms.set_readonly(readonly);
    std::fs::set_permissions(path, perms).unwrap();
    let probe = path.join("probe");
    let writable = std::fs::write(&probe, b"").is_ok();
    let _ = std::fs::remove_file(&probe);
    // Root ignores the permission bits, so the failure cannot be forced.
    writable != readonly
}

fn doc(pk: &str) -> Doc {
    make_doc(pk, vec![1.0, 0.0, 0.0, 0.0], pk)
}

fn fetched(col: &Collection, pks: &[&str]) -> Vec<String> {
    let found = col
        .fetch(pks.iter().map(|pk| pk.to_string()).collect())
        .unwrap();
    let mut found: Vec<String> = found.into_keys().collect();
    found.sort();
    found
}

#[test]
fn test_rotation_failure_keeps_statuses_of_written_docs() {
    let path = temp_dir("batch_status_rotation");
    let opts = CollectionOptions {
        max_buffer_size: 1,
        ..CollectionOptions::default()
    };
    let col = Collection::create_and_open(&path, basic_schema(4), opts.clone()).unwrap();
    if !set_dir_readonly(&path, true) {
        return;
    }

    let results = col.insert(vec![doc("a"), doc("b"), doc("c")]);
    set_dir_readonly(&path, false);
    let results = results.unwrap();

    assert!(results[0].is_ok());
    assert!(!results[1].is_ok());
    assert!(!results[2].is_ok());
    assert_eq!(fetched(&col, &["a", "b", "c"]), vec!["a"]);

    drop(col);
    let col = Collection::open(&path, opts).unwrap();
    assert_eq!(fetched(&col, &["a", "b", "c"]), vec!["a"]);

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_failed_delete_snapshot_reports_every_committed_delete() {
    let path = temp_dir("batch_status_delete");
    let col =
        Collection::create_and_open(&path, basic_schema(4), CollectionOptions::default()).unwrap();
    col.insert(vec![doc("a"), doc("b"), doc("c"), doc("d")])
        .unwrap();
    if !set_dir_readonly(&path, true) {
        return;
    }

    let by_pk = col.delete(vec!["a".to_string(), "zz".to_string()]);
    let by_filter = col.delete_by_filter("label = 'b'");
    set_dir_readonly(&path, false);

    let by_pk = by_pk.unwrap();
    assert!(by_pk[0].is_ok());
    assert_eq!(by_pk[1].code, StatusCode::NotFound);
    assert!(by_filter.unwrap().is_ok());
    assert_eq!(fetched(&col, &["a", "b", "c", "d"]), vec!["c", "d"]);

    drop(col);
    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    assert_eq!(fetched(&col, &["a", "b", "c", "d"]), vec!["c", "d"]);

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

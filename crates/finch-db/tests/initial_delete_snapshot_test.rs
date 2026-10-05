mod common;
use common::*;

#[test]
fn test_create_writes_initial_delete_0_bitmap() {
    let path = temp_dir("delete0_init");
    let schema = basic_schema(4);
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    assert!(
        path.join("delete_0.bitmap").exists(),
        "create_and_open should persist initial delete_0.bitmap"
    );
    assert!(
        !path.join("delete_1.bitmap").exists(),
        "create_and_open should not advance delete suffix to 1"
    );

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_delete_bitmap_rotation_removes_old_snapshot() {
    let path = temp_dir("delete_bitmap_rotation");
    let schema = basic_schema(4);
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![Doc::new("d1").set("emb", vec![1.0f32, 0.0, 0.0, 0.0])])
        .unwrap();

    assert!(path.join("delete_0.bitmap").exists());
    assert!(!path.join("delete_1.bitmap").exists());

    let res = col.delete(vec!["d1".to_string()]).unwrap();
    assert!(res[0].is_ok());

    assert!(
        path.join("delete_1.bitmap").exists(),
        "expected delete_1.bitmap after delete()"
    );
    assert!(
        !path.join("delete_0.bitmap").exists(),
        "expected delete_0.bitmap to be removed after delete snapshot advances"
    );

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_id_map_snapshot_rotation_removes_old_snapshots() {
    let path = temp_dir("id_map_rotation");
    let schema = basic_schema(4);
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    // 1st flush: create id_map_1 snapshot (suffix advances from 0 -> 1).
    col.insert(vec![
        Doc::new("d1").set("emb", vec![1.0f32, 0.0, 0.0, 0.0]),
        Doc::new("d2").set("emb", vec![0.0f32, 1.0, 0.0, 0.0]),
    ])
    .unwrap();
    col.flush().unwrap();

    assert!(path.join("id_map").exists(), "base id_map should exist");
    assert!(
        path.join("id_map_1").exists(),
        "expected id_map_1 snapshot after first flush"
    );

    // 2nd flush: create id_map_2 snapshot and remove id_map_1.
    col.insert(vec![
        Doc::new("d3").set("emb", vec![0.0f32, 0.0, 1.0, 0.0]),
        Doc::new("d4").set("emb", vec![0.0f32, 0.0, 0.0, 1.0]),
    ])
    .unwrap();
    col.flush().unwrap();

    assert!(
        path.join("id_map_2").exists(),
        "expected id_map_2 snapshot after second flush"
    );
    assert!(
        !path.join("id_map_1").exists(),
        "expected id_map_1 to be removed after snapshot rotation"
    );

    // Ensure reopen can still resolve PKs.
    drop(col);
    let reopen = Collection::open(&path, CollectionOptions::default()).unwrap();
    let fetched = reopen
        .fetch(vec!["d1".to_string(), "d4".to_string()])
        .unwrap();
    assert!(fetched.contains_key("d1"));
    assert!(fetched.contains_key("d4"));

    drop(reopen);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_id_map_snapshot_rotation_does_not_remove_active_checkpoint_on_reopen() {
    let path = temp_dir("id_map_rotation_reopen");
    let schema = basic_schema(4);
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![Doc::new("d1").set("emb", vec![1.0f32, 0.0, 0.0, 0.0])])
        .unwrap();
    col.flush().unwrap();
    assert!(
        path.join("id_map_1").exists(),
        "expected id_map_1 after flush"
    );
    drop(col);

    // Reopen: the live id_map is now `id_map_1` (manifest suffix=1). Snapshot rotation must
    // not delete the active fjall directory.
    let reopen = Collection::open(&path, CollectionOptions::default()).unwrap();
    reopen
        .insert(vec![Doc::new("d2").set("emb", vec![0.0f32, 1.0, 0.0, 0.0])])
        .unwrap();
    reopen.flush().unwrap();

    assert!(
        path.join("id_map_1").exists(),
        "active id_map_1 should still exist after flush"
    );
    assert!(
        path.join("id_map_2").exists(),
        "expected id_map_2 checkpoint after flush"
    );

    drop(reopen);
    std::fs::remove_dir_all(&path).ok();
}

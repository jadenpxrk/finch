mod common;
use common::*;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn indexed_schema() -> CollectionSchema {
    CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(
            FieldSchema::new("label", DataType::String)
                .with_index(IndexParams::Invert(InvertIndexParams::default())),
        )
}

fn doc(pk: &str) -> Doc {
    make_doc(pk, vec![1.0, 0.0, 0.0, 0.0], "a")
}

fn read_only() -> CollectionOptions {
    CollectionOptions {
        read_only: true,
        ..CollectionOptions::default()
    }
}

fn all_dirs(root: &Path) -> BTreeSet<PathBuf> {
    let mut out = BTreeSet::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let p = entry.unwrap().path();
            if p.is_dir() {
                out.insert(p.clone());
                stack.push(p);
            }
        }
    }
    out
}

fn copy_dirs(root: &Path) -> Vec<PathBuf> {
    all_dirs(root)
        .into_iter()
        .filter(|p| {
            let name = p.file_name().unwrap().to_string_lossy();
            name.starts_with("id_map_ro_") || name.starts_with("invert_ro_")
        })
        .collect()
}

// Writes a fjall database holding `entries` into `dir`, standing in for a crashed flush's output.
fn write_stray_id_map_dir(dir: &Path, entries: &[(&str, u64)]) {
    let _ = std::fs::remove_dir_all(dir);
    let db = fjall::Database::builder(dir).open().unwrap();
    let items = db
        .keyspace("id_map", fjall::KeyspaceCreateOptions::default)
        .unwrap();
    for (pk, doc_id) in entries {
        items.insert(pk.as_bytes(), doc_id.to_le_bytes()).unwrap();
    }
    db.persist(fjall::PersistMode::SyncAll).unwrap();
}

#[test]
fn flush_during_concurrent_writes_then_reopen_sees_every_key() {
    let path = temp_dir("idmap_ckpt_concurrent");
    let col =
        Collection::create_and_open(&path, indexed_schema(), CollectionOptions::default()).unwrap();
    let writers = 4;
    let per_writer = 1500;

    std::thread::scope(|s| {
        for w in 0..writers {
            let col = &col;
            s.spawn(move || {
                for batch in 0..per_writer / 50 {
                    let docs = (0..50)
                        .map(|i| doc(&format!("w{w}_{}", batch * 50 + i)))
                        .collect();
                    for status in col.insert(docs).unwrap() {
                        assert!(status.is_ok(), "{status:?}");
                    }
                }
            });
        }
        let col = &col;
        s.spawn(move || {
            for _ in 0..20 {
                col.flush().unwrap();
            }
        });
    });

    let pks: Vec<String> = (0..writers)
        .flat_map(|w| (0..per_writer).map(move |i| format!("w{w}_{i}")))
        .collect();
    drop(col);

    let reopened = Collection::open(&path, CollectionOptions::default()).unwrap();
    let fetched = reopened.fetch(pks.clone()).unwrap();
    assert_eq!(fetched.len(), pks.len(), "read-write reopen lost keys");
    reopened.flush().unwrap();
    drop(reopened);

    let ro = Collection::open(&path, read_only()).unwrap();
    let fetched = ro.fetch(pks.clone()).unwrap();
    assert_eq!(fetched.len(), pks.len(), "read-only reopen lost keys");

    drop(ro);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn read_only_open_leaves_no_new_directories() {
    let path = temp_dir("idmap_ckpt_ro_dirs");
    let col =
        Collection::create_and_open(&path, indexed_schema(), CollectionOptions::default()).unwrap();
    col.insert(vec![doc("d1"), doc("d2")]).unwrap();
    col.flush().unwrap();
    drop(col);

    let before = all_dirs(&path);
    let ro = Collection::open(&path, read_only()).unwrap();
    assert!(ro.fetch(vec!["d1".to_string()]).unwrap().contains_key("d1"));
    assert_eq!(
        all_dirs(&path),
        before,
        "read-only open created directories"
    );
    drop(ro);

    // A read-write open loads persisted invert indexes without copying them either.
    let rw = Collection::open(&path, CollectionOptions::default()).unwrap();
    let hits = rw
        .query_sql("SELECT * FROM test WHERE label = 'a'")
        .unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(copy_dirs(&path), Vec::<PathBuf>::new());

    drop(rw);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn killed_process_leftover_directories_are_removed_on_open() {
    let path = temp_dir("idmap_ckpt_leftover");
    let col =
        Collection::create_and_open(&path, indexed_schema(), CollectionOptions::default()).unwrap();
    col.insert(vec![doc("d1")]).unwrap();
    col.flush().unwrap();
    drop(col);

    let leftovers = [
        path.join("id_map_ro_4000000_0"),
        path.join("seg_0").join("invert_ro_4000000_3"),
    ];
    for dir in &leftovers {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("junk"), b"x").unwrap();
    }

    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    for dir in &leftovers {
        assert!(!dir.exists(), "{dir:?} survived open");
    }
    assert!(col
        .fetch(vec!["d1".to_string()])
        .unwrap()
        .contains_key("d1"));

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn flush_replaces_a_checkpoint_left_by_a_crashed_flush() {
    let path = temp_dir("idmap_ckpt_stale_dest");
    let col =
        Collection::create_and_open(&path, indexed_schema(), CollectionOptions::default()).unwrap();
    col.insert(vec![doc("d1")]).unwrap();
    let d1 = col.fetch(vec!["d1".into()]).unwrap()["d1"].doc_id;

    // A flush that died before its manifest commit left `id_map_1` behind.
    write_stray_id_map_dir(&path.join("id_map_1"), &[("ghost", d1)]);
    col.flush().unwrap();
    drop(col);

    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    let fetched = col.fetch(vec!["d1".into(), "ghost".into()]).unwrap();
    assert!(fetched.contains_key("d1"));
    assert!(!fetched.contains_key("ghost"), "stale checkpoint leaked");

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn reopen_after_flushing_an_empty_id_map() {
    let path = temp_dir("idmap_ckpt_empty");
    let col =
        Collection::create_and_open(&path, indexed_schema(), CollectionOptions::default()).unwrap();
    col.insert(vec![doc("d1")]).unwrap();
    col.delete(vec!["d1".to_string()]).unwrap();
    col.flush().unwrap();
    drop(col);

    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    assert!(col.fetch(vec!["d1".into()]).unwrap().is_empty());
    col.insert(vec![doc("d1")]).unwrap();
    assert!(col.fetch(vec!["d1".into()]).unwrap().contains_key("d1"));

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

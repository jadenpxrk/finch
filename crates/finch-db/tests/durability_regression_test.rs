use finch_db::Collection;
use finch_types::{CollectionOptions, CollectionSchema, DataType, Doc, FieldSchema, VectorQuery};
use std::{
    io::Write,
    path::PathBuf,
    sync::{Arc, Barrier},
};

fn path(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("finch_review_{}_{}", name, std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    path
}

fn schema() -> CollectionSchema {
    CollectionSchema::new("review").with_field(
        FieldSchema::new("emb", DataType::VectorFp32)
            .nullable()
            .with_dimension(4),
    )
}

fn doc(pk: &str) -> Doc {
    Doc::new(pk).set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
}

#[test]
fn writes_after_recovering_a_torn_wal_survive_reopen() {
    let path = path("torn_wal");
    let col = Collection::create_and_open(&path, schema(), CollectionOptions::default()).unwrap();
    assert!(col.insert(vec![doc("before")]).unwrap()[0].is_ok());
    drop(col);
    let mut wal = std::fs::OpenOptions::new()
        .append(true)
        .open(path.join("wal_0.log"))
        .unwrap();
    wal.write_all(&[0xff, 0xff]).unwrap();
    wal.sync_all().unwrap();
    drop(wal);
    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    assert!(col
        .fetch(vec!["before".into()])
        .unwrap()
        .contains_key("before"));
    assert!(col.insert(vec![doc("after")]).unwrap()[0].is_ok());
    assert!(col
        .fetch(vec!["after".into()])
        .unwrap()
        .contains_key("after"));
    drop(col);
    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    let found = col
        .fetch(vec!["after".into()])
        .unwrap()
        .contains_key("after");
    drop(col);
    std::fs::remove_dir_all(path).unwrap();
    assert!(
        found,
        "successful write after WAL recovery vanished on second reopen"
    );
}

#[test]
fn writes_after_failed_wal_rotation_survive_reopen() {
    let path = path("wal_rotation");
    let col = Collection::create_and_open(&path, schema(), CollectionOptions::default()).unwrap();
    assert!(col.insert(vec![doc("before")]).unwrap()[0].is_ok());
    std::fs::create_dir(path.join("wal_1.log")).unwrap();
    assert!(col.flush().is_err());
    std::fs::remove_dir(path.join("wal_1.log")).unwrap();
    assert!(col.insert(vec![doc("after")]).unwrap()[0].is_ok());
    assert!(col
        .fetch(vec!["after".into()])
        .unwrap()
        .contains_key("after"));
    drop(col);
    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    let found = col
        .fetch(vec!["after".into()])
        .unwrap()
        .contains_key("after");
    drop(col);
    std::fs::remove_dir_all(path).unwrap();
    assert!(
        found,
        "successful write after failed WAL rotation vanished on reopen"
    );
}

#[test]
fn concurrent_flush_preserves_query_results() {
    let path = path("concurrent_flush");
    let col = Collection::create_and_open(&path, schema(), CollectionOptions::default()).unwrap();
    let docs = (0..128).map(|i| doc(&format!("pk{i}"))).collect();
    assert!(col
        .insert(docs)
        .unwrap()
        .iter()
        .all(|status| status.is_ok()));
    let barrier = Arc::new(Barrier::new(9));
    let threads: Vec<_> = (0..8)
        .map(|_| {
            let col = col.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                (0..100)
                    .map(|_| {
                        col.query(VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 128))
                            .unwrap()
                            .len()
                    })
                    .min()
                    .unwrap()
            })
        })
        .collect();
    barrier.wait();
    col.flush().unwrap();
    let min_count = threads
        .into_iter()
        .map(|thread| thread.join().unwrap())
        .min()
        .unwrap();
    drop(col);
    std::fs::remove_dir_all(path).unwrap();
    assert_eq!(
        min_count, 128,
        "flush temporarily hid committed records from a concurrent reader"
    );
}

#[test]
fn new_records_remain_visible_after_compaction_and_reopen() {
    let path = path("compacted_ids");
    let col = Collection::create_and_open(&path, schema(), CollectionOptions::default()).unwrap();
    assert!(col.insert(vec![doc("deleted")]).unwrap()[0].is_ok());
    col.flush().unwrap();
    assert!(col.delete(vec!["deleted".into()]).unwrap()[0].is_ok());
    col.optimize(finch_types::OptimizeOptions::default())
        .unwrap();
    drop(col);
    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    assert!(col.insert(vec![doc("new")]).unwrap()[0].is_ok());
    let found = col
        .query(VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10))
        .unwrap()
        .iter()
        .any(|doc| doc.pk == "new");
    drop(col);
    std::fs::remove_dir_all(path).unwrap();
    assert!(
        found,
        "new document inherited a compacted-away document's tombstone"
    );
}

#[test]
fn nonfinite_input_cannot_destroy_later_wal_records() {
    let path = path("nan_wal");
    let col = Collection::create_and_open(&path, schema(), CollectionOptions::default()).unwrap();
    let bad = Doc::new("nonfinite").set("emb", vec![f32::NAN, 0.0, 0.0, 0.0]);
    let statuses = col.insert(vec![bad, doc("valid")]).unwrap();
    assert!(statuses[1].is_ok());
    drop(col);
    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    let found = col
        .fetch(vec!["valid".into()])
        .unwrap()
        .contains_key("valid");
    drop(col);
    std::fs::remove_dir_all(path).unwrap();
    assert!(
        found,
        "accepted nonfinite value made a later valid WAL write unrecoverable"
    );
}

#[test]
fn nonfinite_and_out_of_range_floats_are_rejected() {
    let path = path("float_range");
    let schema = CollectionSchema::new("review")
        .with_field(
            FieldSchema::new("emb", DataType::VectorFp32)
                .nullable()
                .with_dimension(2),
        )
        .with_field(
            FieldSchema::new("half", DataType::VectorFp16)
                .nullable()
                .with_dimension(2),
        )
        .with_field(FieldSchema::new("score", DataType::Float64).nullable());
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let base = |pk: &str| Doc::new(pk).set("emb", vec![1.0f32, 0.0]);
    let statuses = col
        .insert(vec![
            Doc::new("inf").set("emb", vec![f32::INFINITY, 0.0]),
            base("half_overflow").set("half", vec![70000.0f32, 0.0]),
            base("nan_scalar").set("score", f64::NAN),
            base("ok")
                .set("half", vec![1.5f32, 0.0])
                .set("score", 1.0f64),
        ])
        .unwrap();
    drop(col);
    std::fs::remove_dir_all(path).unwrap();
    for status in &statuses[..3] {
        assert!(
            status.message.contains("NaN or infinite"),
            "{}",
            status.message
        );
    }
    assert!(statuses[3].is_ok(), "{}", statuses[3].message);
}

#[test]
fn summed_duplicate_sparse_indices_cannot_destroy_later_wal_records() {
    let path = path("sparse_dup_sum");
    let schema = CollectionSchema::new("review").with_field(
        FieldSchema::new("sparse", DataType::SparseFp32)
            .nullable()
            .with_dimension(16),
    );
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let sparse = |values: Vec<f32>| finch_types::Value::SparseF32 {
        indices: vec![1, 1],
        values,
    };
    let statuses = col
        .insert(vec![
            Doc::new("overflow").set("sparse", sparse(vec![3e38, 3e38])),
            Doc::new("valid").set("sparse", sparse(vec![1.0, 2.0])),
        ])
        .unwrap();
    drop(col);
    let found = Collection::open(&path, CollectionOptions::default()).map(|col| {
        col.fetch(vec!["valid".into()])
            .unwrap()
            .contains_key("valid")
    });
    std::fs::remove_dir_all(path).unwrap();
    assert!(
        statuses[0].message.contains("sparse"),
        "{}",
        statuses[0].message
    );
    assert!(statuses[1].is_ok(), "{}", statuses[1].message);
    assert!(
        matches!(found, Ok(true)),
        "summed duplicate sparse values made a later valid WAL write unrecoverable"
    );
}

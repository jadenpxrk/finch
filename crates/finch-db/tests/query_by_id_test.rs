use finch_db::Collection;
use finch_types::{CollectionOptions, CollectionSchema, DataType, Doc, FieldSchema, VectorQuery};
use std::sync::atomic::{AtomicU64, Ordering};

fn temp_dir(name: &str) -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir =
        std::env::temp_dir().join(format!("finch_test_{}_{}_{}", name, std::process::id(), n));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

#[test]
fn test_query_by_id_uses_existing_doc_vector() {
    let path = temp_dir("query_by_id");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("label", DataType::String));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![
        Doc::new("d0")
            .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
            .set("label", "d0"),
        Doc::new("d1")
            .set("emb", vec![0.0f32, 1.0, 0.0, 0.0])
            .set("label", "d1"),
        Doc::new("d2")
            .set("emb", vec![0.0f32, 0.0, 1.0, 0.0])
            .set("label", "d2"),
    ])
    .unwrap();
    col.flush().unwrap();

    // Baseline: explicit query vector.
    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 3);
    let res = col.query(q).unwrap();
    assert_eq!(res[0].pk, "d0");

    // Query-by-id: should behave equivalently.
    let mut q = VectorQuery::new("emb", Vec::new(), 3);
    q.id = Some("d0".to_string());
    let res = col.query(q).unwrap();
    assert_eq!(res[0].pk, "d0");

    // Reject providing both id and vector payload.
    let mut q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 3);
    q.id = Some("d0".to_string());
    assert!(col.query(q).is_err());
}

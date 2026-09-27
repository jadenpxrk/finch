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
fn test_query_profiled_returns_timings_and_docs() {
    let path = temp_dir("query_profiled");
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
    ])
    .unwrap();
    col.flush().unwrap();

    let (docs, prof) = col
        .query_profiled(VectorQuery::new("emb", vec![1.0f32, 0.0, 0.0, 0.0], 2))
        .unwrap();
    assert_eq!(docs.len(), 2);
    assert_eq!(docs[0].pk, "d0");
    assert!(prof.segments_searched >= 1);
    assert!(prof.candidate_count >= docs.len());

    // Filter-only path should populate filter_only timing.
    let mut q = VectorQuery::new("emb", Vec::new(), 10);
    q.output_fields = Some(vec!["label".to_string()]);
    let (_docs, prof) = col.query_profiled(q).unwrap();
    assert!(prof.filter_only >= std::time::Duration::from_secs(0));
}

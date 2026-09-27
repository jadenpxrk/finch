mod common;
use common::*;

// ── Upsert / Update ───────────────────────────────────────────────────────────

#[test]
fn test_upsert() {
    let path = temp_dir("upsert");
    let schema = basic_schema(4);
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    // Insert then upsert (should replace)
    col.insert(vec![make_doc("d1", vec![1.0, 0.0, 0.0, 0.0], "original")])
        .unwrap();
    col.upsert(vec![make_doc("d1", vec![0.0, 1.0, 0.0, 0.0], "updated")])
        .unwrap();

    let stats = col.stats().unwrap();
    assert_eq!(stats.doc_count, 1); // Still 1 after upsert

    let fetched = col.fetch(vec!["d1".to_string()]).unwrap();
    assert!(fetched.contains_key("d1"));

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_update_existing() {
    let path = temp_dir("update");
    let schema = basic_schema(4);
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![make_doc("d1", vec![1.0, 0.0, 0.0, 0.0], "old")])
        .unwrap();
    let upd = col
        .update(vec![make_doc("d1", vec![0.5, 0.5, 0.0, 0.0], "new")])
        .unwrap();
    assert!(upd[0].is_ok());

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_update_merges_patch_and_tombstones_old_version() {
    let path = temp_dir("update_merge_tombstone");

    // Force persisted-segment indexed search for the old version.
    let schema = CollectionSchema::new("test")
        .with_field(
            FieldSchema::new("emb", DataType::VectorFp32)
                .with_dimension(4)
                .with_index(IndexParams::Hnsw(HnswIndexParams::new(MetricType::L2))),
        )
        .with_field(
            FieldSchema::new("label", DataType::String)
                .with_index(IndexParams::Invert(InvertIndexParams::default())),
        );

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![make_doc("d1", vec![1.0, 0.0, 0.0, 0.0], "old")])
        .unwrap();
    col.flush().unwrap(); // old version now persisted + indexed

    // Patch update should merge, keeping emb, and should tombstone the old doc_id.
    let statuses = col
        .update(vec![Doc::new("d1").set("label", "new")])
        .unwrap();
    assert!(statuses[0].is_ok());

    // Query for the new label must return the updated doc (writing segment brute-force).
    let mut q_new =
        VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 5).with_filter("label = 'new'");
    q_new.include_vector = true;
    let got_new = col.query(q_new).unwrap();
    assert_eq!(got_new.len(), 1);
    assert_eq!(got_new[0].pk, "d1");
    assert_eq!(got_new[0].get_str("label"), Some("new"));
    assert!(
        got_new[0].get_vec_f32("emb").is_some(),
        "update must keep original vector via merge"
    );

    // Query for the old label must not return the tombstoned persisted version.
    let q_old = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 5).with_filter("label = 'old'");
    let got_old = col.query(q_old).unwrap();
    assert!(got_old.is_empty(), "tombstoned old version must not appear");

    let stats = col.stats().unwrap();
    assert_eq!(stats.doc_count, 1);

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_update_nonexistent() {
    let path = temp_dir("update_ne");
    let schema = basic_schema(4);
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    let results = col
        .update(vec![make_doc("ghost", vec![1.0, 0.0, 0.0, 0.0], "x")])
        .unwrap();
    assert!(results[0].is_err());

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_retried_flush_after_manifest_failure_keeps_upsert_tombstone() {
    let path = temp_dir("retry_flush_tombstone");
    let col =
        Collection::create_and_open(&path, basic_schema(4), CollectionOptions::default()).unwrap();
    assert!(col
        .insert(vec![make_doc("p", vec![1.0, 0.0, 0.0, 0.0], "old")])
        .unwrap()[0]
        .is_ok());
    col.flush().unwrap();
    assert!(col
        .upsert(vec![make_doc("p", vec![0.0, 1.0, 0.0, 0.0], "new")])
        .unwrap()[0]
        .is_ok());

    std::fs::create_dir(path.join("manifest.tmp")).unwrap();
    assert!(col.flush().is_err(), "manifest failure must fail the flush");
    std::fs::remove_dir_all(path.join("manifest.tmp")).unwrap();
    col.flush().unwrap();
    drop(col);

    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    let hits = col
        .query(VectorQuery::new("emb", vec![1.0, 1.0, 0.0, 0.0], 10))
        .unwrap();
    drop(col);
    std::fs::remove_dir_all(&path).ok();
    let labels: Vec<_> = hits
        .iter()
        .map(|d| (d.pk.clone(), d.get_str("label").map(str::to_string)))
        .collect();
    assert_eq!(labels, vec![("p".to_string(), Some("new".to_string()))]);
}

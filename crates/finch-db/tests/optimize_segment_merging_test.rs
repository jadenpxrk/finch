mod common;
use common::*;

#[test]
fn test_optimize_merges_segments() {
    let path = temp_dir("optimize");
    let doc_count = (MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD + 50) as usize;
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_max_docs_per_segment(MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD);

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    // Insert enough docs to create multiple segments.
    for i in 0..doc_count {
        let pk = format!("d{}", i);
        let vec = vec![i as f32, 0.0, 0.0, 0.0];
        col.insert(vec![Doc::new(&pk).set("emb", vec)]).unwrap();
    }

    col.flush().unwrap();
    let stats_before = col.stats().unwrap();
    let segs_before = stats_before.segment_count;

    col.optimize(OptimizeOptions::default()).unwrap();

    let stats_after = col.stats().unwrap();
    // After merging, there should be fewer segments
    assert!(
        stats_after.segment_count <= segs_before,
        "optimize should not increase segment count"
    );
    // All docs should still be queryable
    assert_eq!(stats_after.doc_count, doc_count as u64);

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_optimize_with_invert_index_does_not_reuse_writing_segment_id() {
    let path = temp_dir("optimize_invert_lock");
    let doc_count = (MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD + 100) as usize;
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(
            FieldSchema::new("id", DataType::Int64)
                .with_index(IndexParams::Invert(InvertIndexParams::default())),
        )
        .with_max_docs_per_segment(MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD);

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    let docs: Vec<Doc> = (0..doc_count)
        .map(|i| {
            let mut v = vec![0.0f32; 4];
            v[i % 4] = 1.0;
            Doc::new(format!("d{}", i))
                .set("emb", v)
                .set("id", i as i64)
        })
        .collect();
    for chunk in docs.chunks(1024) {
        col.insert(chunk.to_vec()).unwrap();
    }

    // Regression check: optimize should not conflict with the active writing segment's
    // invert-index fjall lock.
    col.optimize(OptimizeOptions::default()).unwrap();

    let stats = col.stats().unwrap();
    assert_eq!(stats.doc_count, doc_count as u64);

    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 3);
    let results = col.query(q).unwrap();
    assert_eq!(results.len(), 3);

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn repeated_compaction_keeps_documents_when_segment_ids_reorder_rows() {
    // Compaction must preserve doc-id order even when newer writes use a reserved lower segment id.
    let path = temp_dir("compaction_segment_id_order");
    let schema = CollectionSchema::new("test").with_field(FieldSchema::new("n", DataType::Int64));
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    for (pk, n) in [("a", 0i64), ("b", 1)] {
        assert!(col.insert(vec![Doc::new(pk).set("n", n)]).unwrap()[0].ok());
        col.flush().unwrap();
    }
    col.optimize(OptimizeOptions::default()).unwrap();
    for (pk, n) in [("c", 2i64), ("d", 3)] {
        assert!(col.insert(vec![Doc::new(pk).set("n", n)]).unwrap()[0].ok());
        col.flush().unwrap();
    }
    let pks = ["a", "b", "c", "d"].map(str::to_string).to_vec();
    let before = col.fetch(pks.clone()).unwrap();
    col.optimize(OptimizeOptions::default()).unwrap();
    drop(col);
    let reopened = Collection::open(&path, CollectionOptions::default()).unwrap();
    let after = reopened.fetch(pks).unwrap();
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
    let mut before_keys = before.keys().cloned().collect::<Vec<_>>();
    let mut after_keys = after.keys().cloned().collect::<Vec<_>>();
    before_keys.sort();
    after_keys.sort();
    assert_eq!(before_keys, vec!["a", "b", "c", "d"]);
    assert_eq!(after_keys, before_keys);
}

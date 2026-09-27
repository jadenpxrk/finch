mod common;
use common::*;

// ── Query correctness after optimize + index ──────────────────────────────────

#[test]
fn test_query_after_optimize_with_index() {
    let path = temp_dir("opt_index");
    let doc_count = (MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD + 200) as usize;
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_max_docs_per_segment(MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD);

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    let docs: Vec<Doc> = (0..doc_count)
        .map(|i| {
            let mut v = vec![0.0f32; 4];
            v[i % 4] = 1.0;
            Doc::new(format!("d{}", i)).set("emb", v)
        })
        .collect();
    for chunk in docs.chunks(1024) {
        col.insert(chunk.to_vec()).unwrap();
    }

    col.create_index(
        "emb",
        IndexParams::Flat(FlatIndexParams {
            metric: MetricType::L2,
            quantize: Default::default(),
            column_major: false,
        }),
        CreateIndexOptions::default(),
    )
    .unwrap();

    col.optimize(OptimizeOptions::default()).unwrap();

    // All docs should still be present and correct after merge
    assert_eq!(col.stats().unwrap().doc_count, doc_count as u64);

    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 3);
    let results = col.query(q).unwrap();
    assert_eq!(results.len(), 3);
    let top_idx: usize = results[0].pk[1..].parse().unwrap();
    assert_eq!(
        top_idx % 4,
        0,
        "top result should be in the [1,0,0,0] cluster"
    );

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

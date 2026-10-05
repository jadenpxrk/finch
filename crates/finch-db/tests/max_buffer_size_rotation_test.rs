mod common;
use common::*;

#[test]
fn test_max_buffer_size_triggers_rotation() {
    let path = temp_dir("max_buffer_rotate");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_max_docs_per_segment(1_000_000);

    let opts = CollectionOptions {
        read_only: false,
        enable_mmap: false,
        index_storage: None,
        forward_storage: None,
        max_buffer_size: 1,
        forward_file_format: None,
    };
    let col = Collection::create_and_open(&path, schema, opts).unwrap();

    col.insert(vec![Doc::new("d1").set("emb", vec![1.0f32, 0.0, 0.0, 0.0])])
        .unwrap();

    let stats = col.stats().unwrap();
    assert!(
        stats.segment_count >= 2,
        "small max_buffer_size should force writing segment rotation"
    );

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_optimize_target_segment_size_and_concurrency() {
    let path = temp_dir("opt_target_and_concurrency");
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

    // Very small target forces optimize to emit multiple output segments.
    col.optimize(OptimizeOptions {
        target_segment_size: Some(1),
        concurrency: Some(2),
        ..Default::default()
    })
    .unwrap();

    let stats = col.stats().unwrap();
    assert_eq!(stats.doc_count, doc_count as u64);
    assert!(
        stats.segment_count > 2,
        "target_segment_size should split optimized output into multiple segments"
    );

    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 3);
    let results = col.query(q).unwrap();
    assert_eq!(results.len(), 3);

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_optimize_concurrency_zero_does_not_assume_contiguous_doc_ids() {
    let path = temp_dir("opt_concurrency_zero");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_max_docs_per_segment(MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD);

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    // Insert a small set, flush to create persisted segments.
    let doc_count = (MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD + 100) as usize;
    let docs: Vec<Doc> = (0..doc_count)
        .map(|i| {
            let mut v = vec![0.0f32; 4];
            v[i % 4] = 1.0;
            Doc::new(format!("d{i}")).set("emb", v)
        })
        .collect();
    for chunk in docs.chunks(1024) {
        col.insert(chunk.to_vec()).unwrap();
    }
    col.flush().unwrap();

    // Create holes by deleting a subset before optimize.
    col.delete(vec!["d3".to_string()]).unwrap();
    col.delete(vec!["d7".to_string()]).unwrap();
    col.delete(vec!["d12".to_string()]).unwrap();
    col.delete(vec![format!(
        "d{}",
        MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD + 5
    )])
    .unwrap();

    // Force the non-parallel optimize path.
    col.optimize(OptimizeOptions {
        concurrency: Some(0),
        ..Default::default()
    })
    .unwrap();

    let stats = col.stats().unwrap();
    assert_eq!(stats.doc_count, (doc_count - 4) as u64);

    // Query should still work after optimize.
    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 100);
    let results = col.query(q).unwrap();
    let mut pks: Vec<String> = results.iter().map(|d| d.pk.clone()).collect();
    pks.sort();
    assert!(!pks.contains(&"d3".to_string()));
    assert!(!pks.contains(&"d7".to_string()));
    assert!(!pks.contains(&"d12".to_string()));
    assert!(!pks.contains(&format!("d{}", MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD + 5)));
    assert_eq!(pks.len(), (doc_count - 4).min(100));

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_manifest_and_segment_metadata_after_flush_and_optimize() {
    use finch_db::version::VersionManager;

    let path = temp_dir("manifest_segment_meta");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_max_docs_per_segment(MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD);

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    let doc_count = (MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD as usize * 2) + 500;
    let docs: Vec<Doc> = (0..doc_count)
        .map(|i| {
            let mut v = vec![0.0f32; 4];
            v[i % 4] = 1.0;
            Doc::new(format!("d{i}")).set("emb", v)
        })
        .collect();
    for chunk in docs.chunks(1024) {
        col.insert(chunk.to_vec()).unwrap();
    }

    col.flush().unwrap();

    let vm = VersionManager::load(&path).unwrap();
    let v = vm.current();
    assert_eq!(
        v.persisted_segments.len(),
        3,
        "expected 2 full + 1 partial segment"
    );
    let mut seg_ids: Vec<u32> = v.persisted_segments.iter().map(|s| s.segment_id).collect();
    seg_ids.sort_unstable();
    assert_eq!(seg_ids, vec![0, 1, 2]);

    let mut segs = v.persisted_segments.clone();
    segs.sort_unstable_by_key(|s| s.segment_id);
    assert_eq!(segs[0].min_doc_id, 0);
    assert_eq!(
        segs[0].max_doc_id,
        MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD - 1
    );
    assert_eq!(segs[0].doc_count, MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD);
    assert_eq!(segs[1].min_doc_id, MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD);
    assert_eq!(
        segs[1].max_doc_id,
        MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD * 2 - 1
    );
    assert_eq!(segs[1].doc_count, MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD);
    assert_eq!(
        segs[2].min_doc_id,
        MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD * 2
    );
    assert_eq!(segs[2].max_doc_id, (doc_count - 1) as u64);
    assert_eq!(segs[2].doc_count, 500);

    // Manifest files are rolling: only the newest `manifest.N` should remain.
    let manifest_files: Vec<_> = std::fs::read_dir(&path)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with("manifest."))
        .collect();
    assert_eq!(manifest_files.len(), 1);

    // Optimize merges into a new segment id and removes old segment dirs.
    col.optimize(OptimizeOptions::default()).unwrap();

    let vm = VersionManager::load(&path).unwrap();
    let v = vm.current();
    // when each segment is already at the schema max-docs limit, optimize should not
    // merge segments (it would exceed the cap). With no index declared in the schema, optimize is a no-op.
    assert_eq!(v.persisted_segments.len(), 3);
    let mut segs = v.persisted_segments.clone();
    segs.sort_unstable_by_key(|s| s.segment_id);
    assert_eq!(
        segs.iter().map(|s| s.segment_id).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(segs[0].min_doc_id, 0);
    assert_eq!(
        segs[0].max_doc_id,
        MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD - 1
    );
    assert_eq!(segs[0].doc_count, MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD);
    assert_eq!(segs[1].min_doc_id, MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD);
    assert_eq!(
        segs[1].max_doc_id,
        MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD * 2 - 1
    );
    assert_eq!(segs[1].doc_count, MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD);
    assert_eq!(
        segs[2].min_doc_id,
        MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD * 2
    );
    assert_eq!(segs[2].max_doc_id, (doc_count - 1) as u64);
    assert_eq!(segs[2].doc_count, 500);

    assert!(path.join("seg_0").exists());
    assert!(path.join("seg_1").exists());
    assert!(path.join("seg_2").exists());

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_create_index_with_concurrency_option() {
    let path = temp_dir("create_index_concurrency");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_max_docs_per_segment(MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD);

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    col.insert(vec![
        Doc::new("d1").set("emb", vec![1.0f32, 0.0, 0.0, 0.0]),
        Doc::new("d2").set("emb", vec![0.0f32, 1.0, 0.0, 0.0]),
        Doc::new("d3").set("emb", vec![0.0f32, 0.0, 1.0, 0.0]),
        Doc::new("d4").set("emb", vec![0.0f32, 0.0, 0.0, 1.0]),
    ])
    .unwrap();

    col.create_index(
        "emb",
        IndexParams::Flat(FlatIndexParams {
            metric: MetricType::InnerProduct,
            quantize: QuantizeType::Undefined,
            column_major: false,
        }),
        CreateIndexOptions {
            rebuild: true,
            concurrency: Some(2),
        },
    )
    .unwrap();

    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 2);
    let results = col.query(q).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].pk, "d1");

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

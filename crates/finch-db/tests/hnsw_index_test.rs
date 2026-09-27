mod common;
use common::*;

// ── HNSW index ───────────────────────────────────────────────────────────────

#[test]
fn test_hnsw_index_search() {
    let path = temp_dir("hnsw");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(8));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    // Insert 20 docs to give HNSW something to work with
    let docs: Vec<Doc> = (0..20)
        .map(|i| {
            let mut v = vec![0.0f32; 8];
            v[i % 8] = 1.0;
            Doc::new(format!("d{}", i)).set("emb", v)
        })
        .collect();
    col.insert(docs).unwrap();

    col.create_index(
        "emb",
        IndexParams::Hnsw(HnswIndexParams {
            m: 8,
            ef_construction: 100,
            scaling_factor: 50,
            metric: MetricType::L2,
            quantize: Default::default(),
            build_concurrency: None,
        }),
        CreateIndexOptions::default(),
    )
    .unwrap();

    let q = VectorQuery::new(
        "emb",
        {
            let mut v = vec![0.0f32; 8];
            v[0] = 1.0;
            v
        },
        3,
    )
    .with_params(QueryParams {
        ef: Some(50),
        n_probe: None,
        ..Default::default()
    });

    let results = col.query(q).unwrap();
    assert!(!results.is_empty());
    // Top result should be any of d0, d8, d16: all have vector [1,0,0,0,0,0,0,0]
    let top_idx: usize = results[0].pk[1..].parse().unwrap();
    assert_eq!(
        top_idx % 8,
        0,
        "Expected a doc from the [1,0,...] cluster as top result, got {}",
        results[0].pk
    );

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_hnsw_rebuild_with_failed_manifest_commit_keeps_old_index() {
    let path = temp_dir("hnsw_rebuild_manifest_failure");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(8));
    let hnsw = |m| IndexParams::Hnsw(HnswIndexParams::new(MetricType::L2).with_m(m));
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let docs: Vec<Doc> = (0..20)
        .map(|i| {
            let mut v = vec![0.0f32; 8];
            v[i % 8] = 1.0;
            Doc::new(format!("d{}", i)).set("emb", v)
        })
        .collect();
    col.insert(docs).unwrap();
    let rebuild = CreateIndexOptions {
        rebuild: true,
        ..Default::default()
    };
    col.create_index("emb", hnsw(16), rebuild.clone()).unwrap();

    std::fs::create_dir(path.join("manifest.tmp")).unwrap();
    let failed = col.create_index("emb", hnsw(32), rebuild.clone());
    assert!(failed.is_err(), "manifest commit failure must fail rebuild");
    std::fs::remove_dir_all(path.join("manifest.tmp")).unwrap();
    drop(col);

    let reopen_m = || {
        let col = Collection::open(&path, CollectionOptions::default()).unwrap();
        let hits = col
            .query(VectorQuery::new(
                "emb",
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                3,
            ))
            .unwrap();
        assert_eq!(hits.len(), 3);
        match col
            .schema_info()
            .get_field("emb")
            .unwrap()
            .index_params
            .clone()
        {
            Some(IndexParams::Hnsw(p)) => (col, p.m),
            other => panic!("{other:?}"),
        }
    };
    let (col, m) = reopen_m();
    assert_eq!(m, 16);

    col.create_index("emb", hnsw(32), rebuild).unwrap();
    drop(col);
    let (col, m) = reopen_m();
    drop(col);
    std::fs::remove_dir_all(&path).ok();
    assert_eq!(m, 32);
}

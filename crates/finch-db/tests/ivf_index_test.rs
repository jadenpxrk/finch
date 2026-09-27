mod common;
use common::*;

// ── IVF index ─────────────────────────────────────────────────────────────────

#[test]
fn test_ivf_index_search() {
    let path = temp_dir("ivf");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    // IVF needs at least n_list docs; use n_list=4 with 20 docs
    let docs: Vec<Doc> = (0..20)
        .map(|i| {
            let mut v = vec![0.0f32; 4];
            v[i % 4] = 1.0;
            Doc::new(format!("d{}", i)).set("emb", v)
        })
        .collect();
    col.insert(docs).unwrap();

    col.create_index(
        "emb",
        IndexParams::Ivf(IvfIndexParams {
            n_list: 4,
            n_iters: 5,
            use_soar: false,
            l1_index: None,
            metric: MetricType::L2,
            quantize: Default::default(),
        }),
        CreateIndexOptions::default(),
    )
    .unwrap();

    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 3).with_params(QueryParams {
        ef: None,
        n_probe: Some(4),
        ..Default::default()
    });
    let results = col.query(q).unwrap();
    assert!(!results.is_empty());
    // d0 and d4/d8/d12/d16 all have [1,0,0,0]: top result must be one of them
    let top_idx: usize = results[0].pk[1..].parse().unwrap();
    assert_eq!(
        top_idx % 4,
        0,
        "top result should be in the [1,0,0,0] cluster"
    );

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_ivf_index_auto_n_list_zero_does_not_panic() {
    let path = temp_dir("ivf_auto_nlist");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    let docs: Vec<Doc> = (0..12)
        .map(|i| {
            let mut v = vec![0.0f32; 4];
            v[i % 4] = 1.0;
            Doc::new(format!("d{}", i)).set("emb", v)
        })
        .collect();
    col.insert(docs).unwrap();

    col.create_index(
        "emb",
        IndexParams::Ivf(IvfIndexParams {
            n_list: 0, // auto
            n_iters: 2,
            use_soar: false,
            l1_index: None,
            metric: MetricType::L2,
            quantize: Default::default(),
        }),
        CreateIndexOptions::default(),
    )
    .unwrap();

    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 3).with_params(QueryParams {
        n_probe: Some(4),
        ..Default::default()
    });
    let results = col.query(q).unwrap();
    assert!(!results.is_empty());

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

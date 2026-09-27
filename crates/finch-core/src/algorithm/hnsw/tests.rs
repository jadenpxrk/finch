use super::super::flat::MemoryStorage;
use super::*;
use finch_types::{MetricType, QuantizeType};

#[test]
fn test_select_neighbors_heuristic_diversifies() {
    // Candidate distances are to the (implicit) query, sorted ascending.
    // 2 is very close to 1, so it should be rejected in favor of 3.
    let candidates = vec![(1.0, 1u32), (1.1, 2u32), (1.0, 3u32)];
    let selected = HnswBuilder::select_neighbors_heuristic(&candidates, 2, |a, b| {
        let (x, y) = if a < b { (a, b) } else { (b, a) };
        match (x, y) {
            (1, 2) => 0.05,
            _ => 10.0,
        }
    });
    assert_eq!(selected, vec![1, 3]);
}

#[test]
fn test_hnsw_build_search() {
    let params = HnswIndexParams::new(MetricType::L2);
    let dim = 4;
    let mut builder = HnswBuilder::new(dim, params.clone());

    builder.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
    builder.add(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();
    builder.add(3, &[0.0, 0.0, 1.0, 0.0]).unwrap();
    builder.add(4, &[0.0, 0.0, 0.0, 1.0]).unwrap();
    builder.add(5, &[0.5, 0.5, 0.0, 0.0]).unwrap();

    let mut storage = MemoryStorage::new();
    builder.dump(&mut storage).unwrap();

    let searcher = HnswSearcher::load(&storage, &params).unwrap();
    let results = searcher.search(&[1.0, 0.0, 0.0, 0.0], 2, 10, None).unwrap();

    assert!(!results.is_empty());
    assert_eq!(results[0].0, 1);
}

#[test]
fn test_hnsw_build_search_quantized_fp16() {
    let params = HnswIndexParams::new(MetricType::L2).with_quantize(QuantizeType::Fp16);
    let dim = 4;
    let mut builder = HnswBuilder::new(dim, params.clone());

    builder.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
    builder.add(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();
    builder.add(3, &[0.0, 0.0, 1.0, 0.0]).unwrap();
    builder.add(4, &[0.0, 0.0, 0.0, 1.0]).unwrap();
    builder.add(5, &[0.5, 0.5, 0.0, 0.0]).unwrap();

    let mut storage = MemoryStorage::new();
    builder.dump(&mut storage).unwrap();

    let searcher = HnswSearcher::load(&storage, &params).unwrap();
    let results = searcher.search(&[1.0, 0.0, 0.0, 0.0], 2, 10, None).unwrap();

    assert!(!results.is_empty());
    assert_eq!(results[0].0, 1);
}

#[test]
fn test_upper_neighbors_aligned_with_node_ids() {
    let mut params = HnswIndexParams::new(MetricType::L2);
    // Increase chance of upper levels to exercise level-1+ bookkeeping.
    params.scaling_factor = 2;

    let dim = 8;
    let mut builder = HnswBuilder::new(dim, params);

    for i in 0..256u64 {
        let mut v = vec![0.0f32; dim];
        v[(i as usize) % dim] = 1.0;
        builder.add(i, &v).unwrap();
    }

    let node_count = builder.keys.len();
    let levels = &builder.levels;
    let upper = &builder.upper_neighbors;

    for level_nodes in upper.iter() {
        assert_eq!(
            level_nodes.offsets.len(),
            node_count,
            "each upper level must have one slot per node id",
        );
    }

    for (node_id, &node_level) in levels.iter().enumerate().take(node_count) {
        let node_level = node_level as usize;
        for (lv_idx, level_nodes) in upper.iter().enumerate() {
            if lv_idx < node_level {
                assert_ne!(
                    level_nodes.offsets[node_id],
                    u32::MAX,
                    "node {} should have allocated slot at upper level {}",
                    node_id,
                    lv_idx + 1,
                );
                let off = level_nodes.offsets[node_id] as usize;
                assert!(
                    off + builder.params.m <= level_nodes.neighbors.len(),
                    "node {} upper level {} offset out of range",
                    node_id,
                    lv_idx + 1
                );
            } else {
                assert!(
                    level_nodes.offsets[node_id] == u32::MAX,
                    "node {} should not have neighbors above its level (upper level {})",
                    node_id,
                    lv_idx + 1,
                );
            }
        }
    }
}

#[test]
fn test_build_parallel_basic() {
    let params = HnswIndexParams::new(MetricType::L2);
    let dim = 4;
    let keys: Vec<u64> = vec![1, 2, 3, 4, 5];
    let vectors: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5,
        0.0, 0.0,
    ];

    let builder = HnswBuilder::build_parallel(&keys, &vectors, dim, params.clone(), 2).unwrap();

    let mut storage = MemoryStorage::new();
    builder.dump(&mut storage).unwrap();

    let searcher = HnswSearcher::load(&storage, &params).unwrap();
    let results = searcher.search(&[1.0, 0.0, 0.0, 0.0], 2, 10, None).unwrap();

    assert!(!results.is_empty());
    assert_eq!(results[0].0, 1);
}

#[test]
fn test_build_parallel_self_recall() {
    // Build 1K vectors and verify 100% recall@1 (each vector finds itself).
    let dim: usize = 128;
    let n: usize = 1000;
    let params = HnswIndexParams::new(MetricType::L2);
    let keys: Vec<u64> = (0..n as u64).collect();
    let mut rng = rand::thread_rng();
    let vectors: Vec<f32> = (0..n * dim).map(|_| rng.gen::<f32>()).collect();

    let builder = HnswBuilder::build_parallel(&keys, &vectors, dim, params.clone(), 4).unwrap();

    let mut storage = MemoryStorage::new();
    builder.dump(&mut storage).unwrap();

    let searcher = HnswSearcher::load(&storage, &params).unwrap();

    let mut miss = 0;
    for i in 0..n {
        let query = &vectors[i * dim..(i + 1) * dim];
        let results = searcher.search(query, 1, 100, None).unwrap();
        if results.is_empty() || results[0].0 != i as u64 {
            miss += 1;
        }
    }
    assert_eq!(
        miss, 0,
        "parallel build should have 100% recall@1 for self-lookup"
    );
}

#[test]
fn test_build_parallel_edge_cases() {
    let params = HnswIndexParams::new(MetricType::L2);
    let dim = 4;

    // N=0
    let builder = HnswBuilder::build_parallel(&[], &[], dim, params.clone(), 4).unwrap();
    assert_eq!(builder.keys.len(), 0);

    // N=1
    let builder =
        HnswBuilder::build_parallel(&[1], &[1.0, 0.0, 0.0, 0.0], dim, params.clone(), 4).unwrap();
    assert_eq!(builder.keys.len(), 1);

    // N=2
    let builder = HnswBuilder::build_parallel(
        &[1, 2],
        &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        dim,
        params.clone(),
        4,
    )
    .unwrap();
    assert_eq!(builder.keys.len(), 2);

    // concurrency > N
    let builder = HnswBuilder::build_parallel(
        &[1, 2, 3],
        &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        dim,
        params.clone(),
        100,
    )
    .unwrap();
    assert_eq!(builder.keys.len(), 3);
}

#[test]
fn test_build_parallel_sequential_fallback() {
    // concurrency=1 should use sequential fallback and produce valid results.
    let params = HnswIndexParams::new(MetricType::L2);
    let dim = 4;
    let keys: Vec<u64> = vec![1, 2, 3, 4, 5];
    let vectors: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5,
        0.0, 0.0,
    ];

    let builder = HnswBuilder::build_parallel(&keys, &vectors, dim, params.clone(), 1).unwrap();

    let mut storage = MemoryStorage::new();
    builder.dump(&mut storage).unwrap();

    let searcher = HnswSearcher::load(&storage, &params).unwrap();
    let results = searcher.search(&[1.0, 0.0, 0.0, 0.0], 2, 10, None).unwrap();

    assert!(!results.is_empty());
    assert_eq!(results[0].0, 1);
}

#[test]
fn test_build_parallel_cosine() {
    let params = HnswIndexParams::new(MetricType::Cosine);
    let dim = 4;
    let keys: Vec<u64> = vec![1, 2, 3, 4, 5];
    let vectors: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5,
        0.0, 0.0,
    ];

    let builder = HnswBuilder::build_parallel(&keys, &vectors, dim, params.clone(), 2).unwrap();

    let mut storage = MemoryStorage::new();
    builder.dump(&mut storage).unwrap();

    let searcher = HnswSearcher::load(&storage, &params).unwrap();
    let results = searcher.search(&[1.0, 0.0, 0.0, 0.0], 2, 10, None).unwrap();

    assert!(!results.is_empty());
    assert_eq!(results[0].0, 1);
}

#[test]
fn test_build_parallel_inner_product() {
    let params = HnswIndexParams::new(MetricType::InnerProduct);
    let dim = 4;
    let keys: Vec<u64> = vec![1, 2, 3, 4, 5];
    let vectors: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5,
        0.0, 0.0,
    ];

    let builder = HnswBuilder::build_parallel(&keys, &vectors, dim, params.clone(), 2).unwrap();

    let mut storage = MemoryStorage::new();
    builder.dump(&mut storage).unwrap();

    let searcher = HnswSearcher::load(&storage, &params).unwrap();
    let results = searcher.search(&[1.0, 0.0, 0.0, 0.0], 2, 10, None).unwrap();

    assert!(!results.is_empty());
    assert_eq!(results[0].0, 1);
}

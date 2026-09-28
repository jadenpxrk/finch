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
    let results = searcher
        .search(&[1.0, 0.0, 0.0, 0.0], 2, HnswSearchParams::new(10), None)
        .unwrap();

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
    let results = searcher
        .search(&[1.0, 0.0, 0.0, 0.0], 2, HnswSearchParams::new(10), None)
        .unwrap();

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
    let results = searcher
        .search(&[1.0, 0.0, 0.0, 0.0], 2, HnswSearchParams::new(10), None)
        .unwrap();

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
        let results = searcher
            .search(query, 1, HnswSearchParams::new(100), None)
            .unwrap();
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
    let results = searcher
        .search(&[1.0, 0.0, 0.0, 0.0], 2, HnswSearchParams::new(10), None)
        .unwrap();

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
    let results = searcher
        .search(&[1.0, 0.0, 0.0, 0.0], 2, HnswSearchParams::new(10), None)
        .unwrap();

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
    let results = searcher
        .search(&[1.0, 0.0, 0.0, 0.0], 2, HnswSearchParams::new(10), None)
        .unwrap();

    assert!(!results.is_empty());
    assert_eq!(results[0].0, 1);
}

fn build_random_index(params: &HnswIndexParams, n: usize, dim: usize) -> (MemoryStorage, Vec<f32>) {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let keys: Vec<u64> = (0..n as u64).collect();
    let vectors: Vec<f32> = (0..n * dim).map(|_| rng.gen::<f32>()).collect();
    let builder = HnswBuilder::build_parallel(&keys, &vectors, dim, params.clone(), 1).unwrap();
    let mut storage = MemoryStorage::new();
    builder.dump(&mut storage).unwrap();
    (storage, vectors)
}

/// Allows keys whose remainder mod 5 is 3 or 4, so 60 percent of keys are rejected.
struct SixtyPercentDeleted;
impl DocFilter for SixtyPercentDeleted {
    fn is_valid(&self, doc_id: u64) -> ZResult<bool> {
        Ok(doc_id % 5 >= 3)
    }
}

#[test]
fn filtered_search_fills_topk_when_most_documents_are_deleted() {
    let (n, dim, topk) = (400, 8, 10);
    let params = HnswIndexParams::new(MetricType::L2)
        .with_m(16)
        .with_ef_construction(200);
    let (storage, vectors) = build_random_index(&params, n, dim);
    let searcher = HnswSearcher::load(&storage, &params).unwrap();

    for q in 0..5 {
        let query = &vectors[q * dim..(q + 1) * dim];
        let hits = searcher
            .search(
                query,
                topk,
                HnswSearchParams::new(topk),
                Some(&SixtyPercentDeleted),
            )
            .unwrap();

        let mut exact: Vec<(f32, u64)> = (0..n as u64)
            .filter(|&key| key % 5 >= 3)
            .map(|key| {
                let row = &vectors[key as usize * dim..(key as usize + 1) * dim];
                (simd::l2_f32(query, row), key)
            })
            .collect();
        exact.sort_by(|a, b| a.0.total_cmp(&b.0));
        let exact_keys: Vec<u64> = exact.iter().take(topk).map(|&(_, key)| key).collect();
        let hit_keys: Vec<u64> = hits.iter().map(|&(key, _)| key).collect();
        assert_eq!(hit_keys, exact_keys, "query {q}");
    }
}

#[test]
fn load_rejects_metric_other_than_the_one_built_with() {
    let cosine = HnswIndexParams::new(MetricType::Cosine);
    let (storage, _) = build_random_index(&cosine, 20, 4);
    let Err(err) = HnswSearcher::load(&storage, &HnswIndexParams::new(MetricType::L2)) else {
        panic!("a Cosine index opened with L2 params must not load");
    };
    assert!(err.message.contains("metric mismatch"), "{}", err.message);
}

#[test]
fn load_opens_index_written_without_header() {
    let cosine = HnswIndexParams::new(MetricType::Cosine);
    let (storage, vectors) = build_random_index(&cosine, 20, 4);
    let mut legacy = MemoryStorage::new();
    for name in [
        SEG_KEYS,
        SEG_VECTORS,
        SEG_L0_NEIGHBORS,
        SEG_UPPER_NEIGHBORS,
        SEG_UPPER_INDEX,
        SEG_META,
        SEG_LEVELS,
    ] {
        let bytes = storage.read_segment(name).unwrap();
        legacy.write_segment(name, bytes.as_slice()).unwrap();
    }
    let searcher = HnswSearcher::load(&legacy, &cosine).unwrap();
    let hits = searcher
        .search(&vectors[..4], 1, HnswSearchParams::new(10), None)
        .unwrap();
    assert_eq!(hits[0].0, 0);
}

#[test]
fn load_rejects_out_of_range_entry_point() {
    let params = HnswIndexParams::new(MetricType::L2);
    let (mut storage, _) = build_random_index(&params, 20, 4);
    let mut meta = storage.read_segment(SEG_META).unwrap().as_slice().to_vec();
    // Entry point follows the u64 count and u64 dim.
    meta[16..20].copy_from_slice(&1000u32.to_le_bytes());
    storage.write_segment(SEG_META, &meta).unwrap();
    let Err(err) = HnswSearcher::load(&storage, &params) else {
        panic!("an out-of-range entry point must not load");
    };
    assert!(err.message.contains("entry point"), "{}", err.message);
}

#[test]
fn load_rejects_out_of_range_upper_neighbor() {
    let params = HnswIndexParams::new(MetricType::L2).with_m(2);
    let (mut storage, _) = build_random_index(&params, 200, 4);
    let index = storage
        .read_segment(SEG_UPPER_INDEX)
        .unwrap()
        .as_slice()
        .to_vec();
    // Level 1 slot offsets start after `num_levels`, `node_count`, and `fixed_len`.
    let slot = index[12..]
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .find(|&off| off != u32::MAX)
        .expect("m = 2 puts some nodes on level 1") as usize;
    let mut upper = storage
        .read_segment(SEG_UPPER_NEIGHBORS)
        .unwrap()
        .as_slice()
        .to_vec();
    upper[slot..slot + 4].copy_from_slice(&5000u32.to_le_bytes());
    storage.write_segment(SEG_UPPER_NEIGHBORS, &upper).unwrap();
    let Err(err) = HnswSearcher::load(&storage, &params) else {
        panic!("an out-of-range upper-level link must not load");
    };
    assert!(err.message.contains("upper neighbor id"), "{}", err.message);
}

#[test]
fn header_records_build_tuning() {
    let mut params = HnswIndexParams::new(MetricType::InnerProduct);
    params.build_tuning.heuristic_dim = Some(3);
    params.build_tuning.qdrant_backlink = true;
    params.build_tuning.l0_refine_candidate_cap = Some(64);
    params.build_tuning.prune_alpha = 1.25;
    let (storage, _) = build_random_index(&params, 20, 4);
    let header = storage
        .read_segment(SEG_HEADER)
        .unwrap()
        .as_slice()
        .to_vec();
    let words: Vec<u32> = header[8..]
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect();
    assert_eq!(&header[..8], b"FINCHHNS");
    // version, metric, heuristic dim, flags (keep pruned, repair, qdrant backlink), refine cap, alpha
    assert_eq!(
        words,
        vec![
            1,
            MetricType::InnerProduct as u32,
            3,
            0b1110,
            64,
            1.25f32.to_bits()
        ]
    );
    HnswSearcher::load(&storage, &params).unwrap();
}

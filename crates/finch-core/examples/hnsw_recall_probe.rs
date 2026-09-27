use std::time::Instant;

use finch_core::{HnswBuilder, HnswSearcher, MemoryStorage};
use finch_types::{HnswIndexParams, MetricType};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn distance(metric: MetricType, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        MetricType::Cosine => {
            let mut dot = 0.0f32;
            let mut a2 = 0.0f32;
            let mut b2 = 0.0f32;
            for (&x, &y) in a.iter().zip(b) {
                dot += x * y;
                a2 += x * x;
                b2 += y * y;
            }
            let denom = (a2.sqrt() * b2.sqrt()).max(1e-12);
            1.0 - dot / denom
        }
        _ => a
            .iter()
            .zip(b)
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum(),
    }
}

fn exact_topk(
    vectors: &[f32],
    dim: usize,
    query: &[f32],
    topk: usize,
    metric: MetricType,
) -> Vec<usize> {
    let mut scored: Vec<(f32, usize)> = vectors
        .chunks_exact(dim)
        .enumerate()
        .map(|(idx, v)| (distance(metric, v, query), idx))
        .collect();
    scored.sort_unstable_by(|a, b| {
        let by_dist = a.0.total_cmp(&b.0);
        if by_dist == std::cmp::Ordering::Equal {
            a.1.cmp(&b.1)
        } else {
            by_dist
        }
    });
    scored.into_iter().take(topk).map(|(_, idx)| idx).collect()
}

fn recall_at(results: &[(u64, f32)], truth: &[usize]) -> f32 {
    let mut hits = 0usize;
    for &(key, _) in results {
        if truth.contains(&(key as usize)) {
            hits += 1;
        }
    }
    hits as f32 / truth.len() as f32
}

fn build_searcher(
    keys: &[u64],
    vectors: &[f32],
    dim: usize,
    params: &HnswIndexParams,
    concurrency: usize,
) -> HnswSearcher {
    let t0 = Instant::now();
    let builder =
        HnswBuilder::build_parallel(keys, vectors, dim, params.clone(), concurrency).unwrap();
    let build = t0.elapsed();

    let mut storage = MemoryStorage::new();
    builder.dump(&mut storage).unwrap();
    let searcher = HnswSearcher::load(&storage, params).unwrap();
    eprintln!(
        "build concurrency={concurrency} took {:.3}s",
        build.as_secs_f64()
    );
    searcher
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n = args
        .get(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(20_000usize);
    let dim = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(64usize);
    let query_count = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(200usize);
    let topk = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(100usize);
    let parallelism = args.get(5).and_then(|v| v.parse().ok()).unwrap_or(8usize);
    let centers = args.get(6).and_then(|v| v.parse().ok()).unwrap_or(256usize);
    let noise = args.get(7).and_then(|v| v.parse().ok()).unwrap_or(0.08f32);
    let metric = match args.get(8).map(String::as_str) {
        Some("cosine") | Some("COSINE") => MetricType::Cosine,
        _ => MetricType::L2,
    };

    let mut rng = StdRng::seed_from_u64(0x5eed_1234);
    let mut vectors = vec![0.0f32; n * dim];
    let mut queries = vec![0.0f32; query_count * dim];

    if centers == 0 {
        for v in &mut vectors {
            *v = rng.gen::<f32>() * 2.0 - 1.0;
        }
        for v in &mut queries {
            *v = rng.gen::<f32>() * 2.0 - 1.0;
        }
    } else {
        let mut center_vecs = vec![0.0f32; centers * dim];
        for v in &mut center_vecs {
            *v = rng.gen::<f32>() * 2.0 - 1.0;
        }

        for idx in 0..n {
            let c = idx % centers;
            let center = &center_vecs[c * dim..(c + 1) * dim];
            let dst = &mut vectors[idx * dim..(idx + 1) * dim];
            for d in 0..dim {
                let jitter = (rng.gen::<f32>() - 0.5) * noise;
                dst[d] = center[d] + jitter;
            }
        }

        for idx in 0..query_count {
            let c = idx % centers;
            let center = &center_vecs[c * dim..(c + 1) * dim];
            let dst = &mut queries[idx * dim..(idx + 1) * dim];
            for d in 0..dim {
                let jitter = (rng.gen::<f32>() - 0.5) * noise;
                dst[d] = center[d] + jitter;
            }
        }
    }

    let keys: Vec<u64> = (0..n as u64).collect();
    let params = HnswIndexParams::new(metric)
        .with_m(16)
        .with_ef_construction(128);

    let searchers = [
        ("seq", build_searcher(&keys, &vectors, dim, &params, 1)),
        (
            "par",
            build_searcher(&keys, &vectors, dim, &params, parallelism),
        ),
    ];

    for (name, searcher) in searchers {
        let t0 = Instant::now();
        let mut recall_sum = 0.0f32;
        for query in queries.chunks_exact(dim) {
            let truth = exact_topk(&vectors, dim, query, topk, metric);
            let results = searcher.search(query, topk, 128, None).unwrap();
            recall_sum += recall_at(&results, &truth);
        }
        let elapsed = t0.elapsed().as_secs_f64();
        println!(
            "{name}: n={n} dim={dim} q={query_count} topk={topk} centers={centers} noise={noise} metric={metric:?} recall={:.4} qps={:.1}",
            recall_sum / query_count as f32,
            query_count as f64 / elapsed
        );
    }
}

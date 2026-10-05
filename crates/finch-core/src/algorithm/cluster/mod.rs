//! K-means clustering for IVF index training

use crate::metric::{make_metric, Metric};
use finch_types::{IvfIndexParams, Status, ZResult};
use rand::Rng;
use rayon::prelude::*;

/// K-means clustering result
pub struct KmeansCluster {
    /// Flat centroid storage: n_list × dim contiguous f32s
    pub centroids: Vec<f32>,
    pub dim: usize,
    pub n_list: usize,
}

/// `n` row-major training vectors of `dim` f32s, clustered into `n_list` centroids.
#[derive(Clone, Copy)]
struct KmeansShape {
    n: usize,
    dim: usize,
    n_list: usize,
}

impl KmeansCluster {
    /// Access centroid `i` as a slice
    pub fn centroid(&self, i: usize) -> &[f32] {
        &self.centroids[i * self.dim..(i + 1) * self.dim]
    }

    /// Train k-means with k-means++ initialization and parallel assignment.
    ///
    /// `vectors` is a flat row-major buffer: `n` vectors of `dim` f32s.
    pub fn train(vectors: &[f32], n: usize, dim: usize, config: &IvfIndexParams) -> ZResult<Self> {
        if n == 0 {
            return Err(Status::invalid_argument("no vectors to cluster"));
        }
        debug_assert_eq!(vectors.len(), n * dim);

        let requested = if config.n_list == 0 {
            let auto = (n as f64).sqrt().round() as usize;
            auto.clamp(1, 1024)
        } else {
            config.n_list
        };
        let shape = KmeansShape {
            n,
            dim,
            n_list: requested.min(n).max(1),
        };

        let mut rng = rand::thread_rng();
        let metric = make_metric(config.metric);
        let mut centroids = kmeans_pp_init(vectors, shape, metric.as_ref(), &mut rng);

        let mut assignments = vec![0usize; n];
        for _iter in 0..config.n_iters {
            assign_nearest(
                vectors,
                &centroids,
                shape,
                metric.as_ref(),
                &mut assignments,
            );
            update_centroids(vectors, &assignments, shape, &mut centroids, &mut rng);
        }

        Ok(KmeansCluster {
            centroids,
            dim,
            n_list: shape.n_list,
        })
    }

    /// Find nearest centroid to a query vector
    pub fn find_nearest(&self, query: &[f32], metric: &dyn Metric) -> usize {
        let mut best = (0, f32::INFINITY);
        for i in 0..self.n_list {
            let c = self.centroid(i);
            let d = metric.distance(query, c);
            if d < best.1 {
                best = (i, d);
            }
        }
        best.0
    }

    /// Find n_probe nearest centroids using partial sort (O(n) instead of O(n log n))
    pub fn find_nearest_n(&self, query: &[f32], n_probe: usize, metric: &dyn Metric) -> Vec<usize> {
        let mut dists: Vec<(usize, f32)> = (0..self.n_list)
            .map(|i| (i, metric.distance(query, self.centroid(i))))
            .collect();

        let k = n_probe.min(dists.len());
        if k == 0 {
            return Vec::new();
        }
        if k >= dists.len() {
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            return dists.iter().map(|&(i, _)| i).collect();
        }

        // Partial sort: O(n) to find top-k, then O(k log k) to sort those
        let nth = k - 1;
        dists.select_nth_unstable_by(nth, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        dists.truncate(k);
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.iter().map(|&(i, _)| i).collect()
    }
}

/// k-means++ seeding: next centroids are sampled in proportion to distance from those chosen.
fn kmeans_pp_init<R: Rng + ?Sized>(
    vectors: &[f32],
    shape: KmeansShape,
    metric: &dyn Metric,
    rng: &mut R,
) -> Vec<f32> {
    let KmeansShape { n, dim, n_list } = shape;
    let mut centroids: Vec<f32> = Vec::with_capacity(n_list * dim);

    let first = rng.gen_range(0..n);
    centroids.extend_from_slice(&vectors[first * dim..(first + 1) * dim]);

    let mut best_dists: Vec<f32> = (0..n)
        .map(|i| metric.distance(&vectors[i * dim..(i + 1) * dim], &centroids[0..dim]))
        .collect();

    while centroids.len() / dim < n_list {
        let next_idx = sample_by_distance(&best_dists, rng);
        centroids.extend_from_slice(&vectors[next_idx * dim..(next_idx + 1) * dim]);
        let n_centroids = centroids.len() / dim;
        let last_c = &centroids[(n_centroids - 1) * dim..n_centroids * dim];
        for i in 0..n {
            let d = metric.distance(&vectors[i * dim..(i + 1) * dim], last_c);
            if d.is_finite() && d < best_dists[i] {
                best_dists[i] = d;
            }
        }
    }
    centroids
}

/// Index drawn with probability proportional to `best_dists`; uniform when all are zero.
fn sample_by_distance<R: Rng + ?Sized>(best_dists: &[f32], rng: &mut R) -> usize {
    let mut total: f64 = 0.0;
    for &d in best_dists {
        if d.is_finite() && d > 0.0 {
            total += d as f64;
        }
    }

    if total <= 0.0 || !total.is_finite() {
        return rng.gen_range(0..best_dists.len());
    }
    let mut r = rng.gen::<f64>() * total;
    let mut chosen = 0usize;
    for (i, &d) in best_dists.iter().enumerate() {
        if !d.is_finite() || d <= 0.0 {
            continue;
        }
        r -= d as f64;
        if r <= 0.0 {
            chosen = i;
            break;
        }
    }
    chosen
}

/// Assignment step (parallel): nearest centroid per vector.
fn assign_nearest(
    vectors: &[f32],
    centroids: &[f32],
    shape: KmeansShape,
    metric: &dyn Metric,
    assignments: &mut [usize],
) {
    let KmeansShape { dim, n_list, .. } = shape;
    assignments
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, asgn)| {
            let v = &vectors[i * dim..(i + 1) * dim];
            let mut best = (0, f32::INFINITY);
            for c_idx in 0..n_list {
                let c = &centroids[c_idx * dim..(c_idx + 1) * dim];
                let d = metric.distance(v, c);
                if d < best.1 {
                    best = (c_idx, d);
                }
            }
            *asgn = best.0;
        });
}

/// Update step: mean of each cluster (flat f64 accumulator); empty clusters restart at a random vector.
fn update_centroids<R: Rng + ?Sized>(
    vectors: &[f32],
    assignments: &[usize],
    shape: KmeansShape,
    centroids: &mut [f32],
    rng: &mut R,
) {
    let KmeansShape { n, dim, n_list } = shape;
    let mut sums = vec![0.0f64; n_list * dim];
    let mut counts = vec![0usize; n_list];

    for (i, &asgn) in assignments.iter().enumerate() {
        counts[asgn] += 1;
        let sum_row = &mut sums[asgn * dim..(asgn + 1) * dim];
        let vec_row = &vectors[i * dim..(i + 1) * dim];
        for (d, &x) in sum_row.iter_mut().zip(vec_row.iter()) {
            *d += x as f64;
        }
    }

    for c_idx in 0..n_list {
        let dst = &mut centroids[c_idx * dim..(c_idx + 1) * dim];
        if counts[c_idx] == 0 {
            let rand_idx = rng.gen_range(0..n);
            dst.copy_from_slice(&vectors[rand_idx * dim..(rand_idx + 1) * dim]);
        } else {
            let cnt = counts[c_idx] as f64;
            let src = &sums[c_idx * dim..(c_idx + 1) * dim];
            for (d, &s) in dst.iter_mut().zip(src.iter()) {
                *d = (s / cnt) as f32;
            }
        }
    }
}

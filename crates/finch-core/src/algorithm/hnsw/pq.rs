//! Product Quantization (PQ) for HNSW two-phase search.
//!
//! Each vector of dimension `dim` is split into `n_subspaces` sub-vectors of
//! `sub_dim = dim / n_subspaces` dimensions.  Each sub-vector is quantized to
//! the index of its nearest centroid (8-bit code, 256 centroids per subspace).
//!
//! At search time, Asymmetric Distance Computation (ADC) precomputes a distance
//! lookup table for the query, then sums table entries per code: O(n_subspaces)
//! additions per candidate instead of O(dim) multiply-adds.

use rand::Rng;

/// Product quantizer: trained centroids + encoding/distance routines.
pub struct ProductQuantizer {
    /// Number of subspaces (e.g. 48 for dim=1536, giving sub_dim=32).
    pub n_subspaces: usize,
    /// Dimensionality of each subspace (dim / n_subspaces).
    pub sub_dim: usize,
    /// Number of centroids per subspace (always 256 for 8-bit codes).
    pub n_centroids: usize,
    /// Flat centroid storage: n_subspaces × n_centroids × sub_dim.
    pub centroids: Vec<f32>,
}

/// Stored PQ data loaded from disk.
pub struct PqData {
    pub pq: ProductQuantizer,
    /// Flat PQ codes: n_vectors × n_subspaces bytes.
    pub codes: Vec<u8>,
    /// Number of vectors.
    pub n_vectors: usize,
}

impl PqData {
    /// Get the PQ codes for vector `id`.
    #[inline]
    pub fn get_codes(&self, id: u32) -> &[u8] {
        let off = id as usize * self.pq.n_subspaces;
        &self.codes[off..off + self.pq.n_subspaces]
    }
}

impl ProductQuantizer {
    /// Choose a good number of subspaces for the given dimension.
    /// Targets ~32 dims per subspace, falling back to fewer if dim is small.
    pub fn default_n_subspaces(dim: usize) -> usize {
        if dim <= 8 {
            return 1;
        }
        // Try to divide evenly into subspaces of ~32 dims.
        let target = 32usize;
        let n = (dim + target - 1) / target;
        // Must divide evenly.
        let mut best = 1;
        for candidate in (1..=n.max(1)).rev() {
            if dim % candidate == 0 {
                best = candidate;
                break;
            }
        }
        // Fallback: if nothing divides evenly (unlikely), just use dim.
        if best == 1 && dim > 256 {
            // Try common factors.
            for &c in &[48, 64, 32, 16, 8, 4, 2] {
                if dim % c == 0 {
                    return c;
                }
            }
        }
        best
    }

    /// Train PQ centroids using simple k-means.
    ///
    /// `vectors`: flat row-major, `n` vectors of `dim` f32s.
    /// `n_subspaces`: number of subspaces (dim must be divisible by this).
    /// `n_iters`: k-means iterations (default 20-25 is fine).
    /// `sample_limit`: if n > sample_limit, sample this many vectors for training.
    pub fn train(
        vectors: &[f32],
        n: usize,
        dim: usize,
        n_subspaces: usize,
        n_iters: usize,
        sample_limit: usize,
    ) -> Self {
        assert!(dim > 0 && n_subspaces > 0 && dim % n_subspaces == 0);
        let sub_dim = dim / n_subspaces;
        let n_centroids = 256usize;

        // Sample vectors if too many.
        let (train_vecs, train_n) = if n > sample_limit && sample_limit > 0 {
            let mut rng = rand::thread_rng();
            let mut indices: Vec<usize> = (0..n).collect();
            // Partial Fisher-Yates for sample_limit elements.
            let limit = sample_limit.min(n);
            for i in 0..limit {
                let j = rng.gen_range(i..n);
                indices.swap(i, j);
            }
            indices.truncate(limit);
            let mut sampled = Vec::with_capacity(limit * dim);
            for &idx in &indices {
                sampled.extend_from_slice(&vectors[idx * dim..(idx + 1) * dim]);
            }
            (sampled, limit)
        } else {
            (vectors.to_vec(), n)
        };

        // Train each subspace independently.
        let mut centroids = Vec::with_capacity(n_subspaces * n_centroids * sub_dim);

        for s in 0..n_subspaces {
            let subspace = Subspace {
                dim,
                offset: s * sub_dim,
                sub_dim,
            };
            let kmeans = SubspaceKmeans {
                vectors: &train_vecs,
                n: train_n,
                subspace,
                k: n_centroids.min(train_n),
            };
            centroids.extend_from_slice(&kmeans.train(n_centroids, n_iters));
        }

        ProductQuantizer {
            n_subspaces,
            sub_dim,
            n_centroids,
            centroids,
        }
    }

    /// Get centroid `c` for subspace `s`.
    #[inline]
    pub fn centroid(&self, s: usize, c: usize) -> &[f32] {
        let off = (s * self.n_centroids + c) * self.sub_dim;
        &self.centroids[off..off + self.sub_dim]
    }

    /// Encode a single vector into PQ codes.
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let mut codes = Vec::with_capacity(self.n_subspaces);
        for s in 0..self.n_subspaces {
            let sub_vec = &vector[s * self.sub_dim..(s + 1) * self.sub_dim];
            let mut best_c = 0u8;
            let mut best_dist = f32::INFINITY;
            for c in 0..self.n_centroids {
                let cent = self.centroid(s, c);
                let d = l2_sub(sub_vec, cent);
                if d < best_dist {
                    best_dist = d;
                    best_c = c as u8;
                }
            }
            codes.push(best_c);
        }
        codes
    }

    /// Encode a single vector, appending codes to `buf`.
    pub fn encode_append(&self, vector: &[f32], buf: &mut Vec<u8>) {
        for s in 0..self.n_subspaces {
            let sub_vec = &vector[s * self.sub_dim..(s + 1) * self.sub_dim];
            let mut best_c = 0u8;
            let mut best_dist = f32::INFINITY;
            for c in 0..self.n_centroids {
                let cent = self.centroid(s, c);
                let d = l2_sub(sub_vec, cent);
                if d < best_dist {
                    best_dist = d;
                    best_c = c as u8;
                }
            }
            buf.push(best_c);
        }
    }

    /// Precompute the ADC distance lookup table for a given query.
    ///
    /// Returns a flat table of n_subspaces × n_centroids f32 values.
    /// table[s * 256 + c] = L2(query_sub[s], centroid[s][c]).
    pub fn compute_distance_table(&self, query: &[f32]) -> Vec<f32> {
        let mut table = Vec::with_capacity(self.n_subspaces * self.n_centroids);
        for s in 0..self.n_subspaces {
            let q_sub = &query[s * self.sub_dim..(s + 1) * self.sub_dim];
            for c in 0..self.n_centroids {
                let cent = self.centroid(s, c);
                table.push(l2_sub(q_sub, cent));
            }
        }
        table
    }

    /// Precompute an IP-based ADC distance lookup table.
    ///
    /// table[s * 256 + c] = IP(query_sub[s], centroid[s][c]).
    pub fn compute_ip_distance_table(&self, query: &[f32]) -> Vec<f32> {
        let mut table = Vec::with_capacity(self.n_subspaces * self.n_centroids);
        for s in 0..self.n_subspaces {
            let q_sub = &query[s * self.sub_dim..(s + 1) * self.sub_dim];
            for c in 0..self.n_centroids {
                let cent = self.centroid(s, c);
                table.push(ip_sub(q_sub, cent));
            }
        }
        table
    }

    /// Serialize PQ centroids to bytes (for storage).
    pub fn serialize_centroids(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(self.n_subspaces as u32).to_le_bytes());
        buf.extend_from_slice(&(self.sub_dim as u32).to_le_bytes());
        buf.extend_from_slice(&(self.n_centroids as u32).to_le_bytes());
        for &f in &self.centroids {
            buf.extend_from_slice(&f.to_le_bytes());
        }
        buf
    }

    /// Deserialize PQ centroids from bytes.
    pub fn deserialize_centroids(data: &[u8]) -> Option<Self> {
        if data.len() < 12 {
            return None;
        }
        let n_subspaces = u32::from_le_bytes(data[0..4].try_into().ok()?) as usize;
        let sub_dim = u32::from_le_bytes(data[4..8].try_into().ok()?) as usize;
        let n_centroids = u32::from_le_bytes(data[8..12].try_into().ok()?) as usize;
        let expected = 12 + n_subspaces * n_centroids * sub_dim * 4;
        if data.len() < expected || n_centroids == 0 || sub_dim == 0 || n_subspaces == 0 {
            return None;
        }
        let mut centroids = Vec::with_capacity(n_subspaces * n_centroids * sub_dim);
        let float_data = &data[12..];
        for i in 0..n_subspaces * n_centroids * sub_dim {
            let off = i * 4;
            centroids.push(f32::from_le_bytes(
                float_data[off..off + 4].try_into().ok()?,
            ));
        }
        Some(ProductQuantizer {
            n_subspaces,
            sub_dim,
            n_centroids,
            centroids,
        })
    }
}

/// Compute distance from precomputed ADC table.
///
/// For L2: sum of table[s * 256 + codes[s]] gives approximate L2 distance.
/// For IP (cosine-normalized): sum gives approximate IP; caller negates/transforms.
#[inline]
pub fn distance_from_table(table: &[f32], codes: &[u8], n_subspaces: usize) -> f32 {
    assert!(codes.len() >= n_subspaces && table.len() >= n_subspaces * 256);
    let mut sum = 0.0f32;
    let mut i = 0;
    let end4 = n_subspaces & !3;
    // Unroll 4x to hide dependency chain.
    while i < end4 {
        // SAFETY: i + 3 < n_subspaces and a code is < 256, so both lookups are within the assert above.
        sum += unsafe {
            *table.get_unchecked(i * 256 + *codes.get_unchecked(i) as usize)
                + *table.get_unchecked((i + 1) * 256 + *codes.get_unchecked(i + 1) as usize)
                + *table.get_unchecked((i + 2) * 256 + *codes.get_unchecked(i + 2) as usize)
                + *table.get_unchecked((i + 3) * 256 + *codes.get_unchecked(i + 3) as usize)
        };
        i += 4;
    }
    while i < n_subspaces {
        // SAFETY: i < n_subspaces and a code is < 256, so both lookups are within the assert above.
        sum += unsafe { *table.get_unchecked(i * 256 + *codes.get_unchecked(i) as usize) };
        i += 1;
    }
    sum
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// L2 squared distance on sub-vectors (small dim, no SIMD needed).
#[inline]
fn l2_sub(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

/// Inner product on sub-vectors.
#[inline]
fn ip_sub(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// Position of one subspace inside `dim`-wide rows.
struct Subspace {
    dim: usize,
    offset: usize,
    sub_dim: usize,
}

/// K-means over one subspace of `n` row-major training vectors, with `k` centroids.
struct SubspaceKmeans<'a> {
    vectors: &'a [f32],
    n: usize,
    subspace: Subspace,
    k: usize,
}

impl SubspaceKmeans<'_> {
    fn row(&self, i: usize) -> &[f32] {
        let start = i * self.subspace.dim + self.subspace.offset;
        &self.vectors[start..start + self.subspace.sub_dim]
    }

    /// Trains `k` centroids, zero-padded to `n_centroids` (padding is never assigned).
    fn train(&self, n_centroids: usize, n_iters: usize) -> Vec<f32> {
        let sub_dim = self.subspace.sub_dim;
        let mut rng = rand::thread_rng();
        let mut centroids = self.random_init(&mut rng);
        let mut assignments = vec![0u8; self.n];
        for _iter in 0..n_iters {
            self.assign(&centroids, &mut assignments);
            self.update(&assignments, &mut centroids, &mut rng);
        }
        if self.k < n_centroids {
            centroids.resize(n_centroids * sub_dim, 0.0);
        }
        centroids
    }

    /// Random initialization: `k` distinct sub-vectors as initial centroids.
    fn random_init<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<f32> {
        let mut centroids = Vec::with_capacity(self.k * self.subspace.sub_dim);
        let mut chosen = std::collections::HashSet::new();
        while chosen.len() < self.k {
            let idx = rng.gen_range(0..self.n);
            if chosen.insert(idx) {
                centroids.extend_from_slice(self.row(idx));
            }
        }
        centroids
    }

    fn assign(&self, centroids: &[f32], assignments: &mut [u8]) {
        let sub_dim = self.subspace.sub_dim;
        for (i, slot) in assignments.iter_mut().enumerate() {
            let v = self.row(i);
            let mut best_c = 0u8;
            let mut best_dist = f32::INFINITY;
            for c in 0..self.k {
                let d = l2_sub(v, &centroids[c * sub_dim..(c + 1) * sub_dim]);
                if d < best_dist {
                    best_dist = d;
                    best_c = c as u8;
                }
            }
            *slot = best_c;
        }
    }

    /// Mean of each cluster; dead centroids restart at a random sub-vector.
    fn update<R: Rng + ?Sized>(&self, assignments: &[u8], centroids: &mut [f32], rng: &mut R) {
        let sub_dim = self.subspace.sub_dim;
        let mut sums = vec![0.0f64; self.k * sub_dim];
        let mut counts = vec![0usize; self.k];
        for (i, &c) in assignments.iter().enumerate() {
            let c = c as usize;
            counts[c] += 1;
            let sum_row = &mut sums[c * sub_dim..(c + 1) * sub_dim];
            for (s, &x) in sum_row.iter_mut().zip(self.row(i).iter()) {
                *s += x as f64;
            }
        }

        for c in 0..self.k {
            let dst = &mut centroids[c * sub_dim..(c + 1) * sub_dim];
            if counts[c] == 0 {
                dst.copy_from_slice(self.row(rng.gen_range(0..self.n)));
            } else {
                let cnt = counts[c] as f64;
                let src = &sums[c * sub_dim..(c + 1) * sub_dim];
                for (d, &s) in dst.iter_mut().zip(src.iter()) {
                    *d = (s / cnt) as f32;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pq_train_encode_distance() {
        let dim = 8;
        let n_subspaces = 2;
        let n = 100;
        let mut rng = rand::thread_rng();
        let vectors: Vec<f32> = (0..n * dim).map(|_| rng.gen::<f32>()).collect();

        let pq = ProductQuantizer::train(&vectors, n, dim, n_subspaces, 10, 200);
        assert_eq!(pq.n_subspaces, 2);
        assert_eq!(pq.sub_dim, 4);
        assert_eq!(pq.n_centroids, 256);
        assert_eq!(pq.centroids.len(), 2 * 256 * 4);

        // Encode first vector and check codes are valid.
        let codes = pq.encode(&vectors[0..dim]);
        assert_eq!(codes.len(), 2);

        // Distance table + lookup should give reasonable result.
        let table = pq.compute_distance_table(&vectors[0..dim]);
        assert_eq!(table.len(), 2 * 256);
        let d = distance_from_table(&table, &codes, n_subspaces);
        // Distance to self via PQ should be small (not necessarily zero due to quantization).
        assert!(d < 1.0, "self-distance via PQ = {d}");
    }

    #[test]
    fn test_pq_serialize_deserialize() {
        let pq = ProductQuantizer {
            n_subspaces: 2,
            sub_dim: 4,
            n_centroids: 256,
            centroids: vec![0.5f32; 2 * 256 * 4],
        };
        let buf = pq.serialize_centroids();
        let pq2 = ProductQuantizer::deserialize_centroids(&buf).unwrap();
        assert_eq!(pq2.n_subspaces, 2);
        assert_eq!(pq2.sub_dim, 4);
        assert_eq!(pq2.n_centroids, 256);
        assert_eq!(pq2.centroids.len(), pq.centroids.len());
    }

    #[test]
    fn test_distance_from_table() {
        let n_sub = 4;
        // Create a simple table where table[s*256+c] = s*256+c as f32.
        let table: Vec<f32> = (0..n_sub * 256).map(|i| i as f32).collect();
        let codes = vec![0u8, 1, 2, 3];
        let d = distance_from_table(&table, &codes, n_sub);
        // Expected: 0*256+0 + 1*256+1 + 2*256+2 + 3*256+3 = 0 + 257 + 514 + 771 = 1542
        assert_eq!(d, 1542.0);
    }

    #[test]
    fn test_default_n_subspaces() {
        assert_eq!(ProductQuantizer::default_n_subspaces(1536), 48);
        assert_eq!(ProductQuantizer::default_n_subspaces(768), 24);
        assert_eq!(ProductQuantizer::default_n_subspaces(128), 4);
        assert!(ProductQuantizer::default_n_subspaces(4) >= 1);
    }
}

//! Pure scalar (no SIMD) fallback implementations

/// Scalar L2 squared distance
pub fn l2_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Scalar inner product
pub fn ip_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Scalar MIPS-L2 distance (localized spherical injection, e2=0.0).
///
/// Computes "Mips SphericalInjection Squared Euclidean Distance":
/// `2 - 2 * ip(a,b) / max(||a||^2, ||b||^2)`.
pub fn mips_l2_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut ip = 0.0f32;
    let mut u2 = 0.0f32;
    let mut v2 = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        ip += x * y;
        u2 += x * x;
        v2 += y * y;
    }
    let denom = u2.max(v2);
    2.0 - 2.0 * ip / denom
}

/// Scalar cosine distance (1 - cosine_similarity)
pub fn cosine_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-10 {
        // if either vector has zero magnitude, treat cosine distance as 0.0
        // rather than max distance.
        0.0
    } else {
        1.0 - (dot / denom)
    }
}

/// Scalar Hamming distance (popcount of XOR)
pub fn hamming_u8_scalar(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Scalar batch L2 (row-major matrix)
pub fn l2_batch_f32_scalar(matrix: &[f32], query: &[f32], m: usize, dim: usize, out: &mut [f32]) {
    for i in 0..m {
        let row = &matrix[i * dim..(i + 1) * dim];
        out[i] = l2_f32_scalar(row, query);
    }
}

/// Scalar batch inner product (row-major matrix)
pub fn ip_batch_f32_scalar(matrix: &[f32], query: &[f32], m: usize, dim: usize, out: &mut [f32]) {
    for i in 0..m {
        let row = &matrix[i * dim..(i + 1) * dim];
        out[i] = ip_f32_scalar(row, query);
    }
}

/// Scalar batch cosine distance (row-major matrix).
///
/// Precomputes `||query||²` once and reuses across all rows.
pub fn cosine_batch_f32_scalar(
    matrix: &[f32],
    query: &[f32],
    m: usize,
    dim: usize,
    out: &mut [f32],
) {
    let mut norm_b = 0.0f32;
    for &q in query.iter().take(dim) {
        norm_b += q * q;
    }
    for i in 0..m {
        let row = &matrix[i * dim..(i + 1) * dim];
        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        for (x, y) in row.iter().zip(query.iter()) {
            dot += x * y;
            norm_a += x * x;
        }
        let denom = (norm_a * norm_b).sqrt();
        out[i] = if denom < 1e-10 {
            0.0
        } else {
            1.0 - (dot / denom)
        };
    }
}

/// Scalar batch MIPS-L2 (row-major matrix).
pub fn mips_l2_batch_f32_scalar(
    matrix: &[f32],
    query: &[f32],
    m: usize,
    dim: usize,
    out: &mut [f32],
) {
    let mut v2 = 0.0f32;
    for &q in query.iter().take(dim) {
        v2 += q * q;
    }
    for i in 0..m {
        let row = &matrix[i * dim..(i + 1) * dim];
        let mut ip = 0.0f32;
        let mut u2 = 0.0f32;
        for (x, y) in row.iter().zip(query.iter()) {
            ip += x * y;
            u2 += x * x;
        }
        let denom = u2.max(v2);
        out[i] = 2.0 - 2.0 * ip / denom;
    }
}

// Scatter-pointer batch: vectors at arbitrary (non-contiguous) addresses

/// Scalar scatter-pointer L2 batch.
///
/// # Safety
/// Each `ptrs[i]` must point to at least `dim` readable f32s.
// SAFETY: caller guarantees each `ptrs[i]` has `dim` readable f32s and `query.len() >= dim`.
pub unsafe fn l2_scatter_f32_scalar(
    ptrs: &[*const f32],
    query: &[f32],
    dim: usize,
    out: &mut [f32],
) {
    for (i, &p) in ptrs.iter().enumerate() {
        let vec = std::slice::from_raw_parts(p, dim);
        out[i] = l2_f32_scalar(vec, query);
    }
}

/// Scalar scatter-pointer inner product batch.
///
/// # Safety
/// Each `ptrs[i]` must point to at least `dim` readable f32s.
// SAFETY: caller guarantees each `ptrs[i]` has `dim` readable f32s and `query.len() >= dim`.
pub unsafe fn ip_scatter_f32_scalar(
    ptrs: &[*const f32],
    query: &[f32],
    dim: usize,
    out: &mut [f32],
) {
    for (i, &p) in ptrs.iter().enumerate() {
        let vec = std::slice::from_raw_parts(p, dim);
        out[i] = ip_f32_scalar(vec, query);
    }
}

//! SIMD-accelerated distance functions with runtime dispatch.
//!
//! Function pointers are resolved once at first use via `OnceLock`, eliminating
//! per-call `is_x86_feature_detected!` overhead in tight loops (HNSW beam search,
//! heuristic neighbor selection).

mod scalar;

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod tail;

#[cfg(target_arch = "x86_64")]
mod x86;

#[cfg(target_arch = "aarch64")]
mod aarch64;

pub use scalar::*;

#[cfg(target_arch = "x86_64")]
use std::sync::OnceLock;

#[cfg(target_arch = "x86_64")]
type DistFn = fn(&[f32], &[f32]) -> f32;
#[cfg(target_arch = "x86_64")]
// SAFETY: same contract as `l2_scatter_f32`.
type ScatterFn = unsafe fn(&[*const f32], &[f32], usize, &mut [f32]);

// Resolver: pick the best implementation once, cache forever.

#[cfg(target_arch = "x86_64")]
fn resolve_l2() -> DistFn {
    // Wrappers that are plain `fn` (not `unsafe fn`) so they match `DistFn`.
    fn avx512(a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: picked only after `avx512f` detection; the public wrapper equalizes lengths.
        unsafe { x86::avx512::l2_f32(a, b) }
    }
    fn avx2(a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: picked only after `avx2` detection; the public wrapper equalizes lengths.
        unsafe { x86::avx2::l2_f32(a, b) }
    }
    fn sse4(a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: picked only after `sse4.1` detection; the public wrapper equalizes lengths.
        unsafe { x86::sse4::l2_f32(a, b) }
    }

    if is_x86_feature_detected!("avx512f") {
        return avx512;
    }
    if is_x86_feature_detected!("avx2") {
        return avx2;
    }
    if is_x86_feature_detected!("sse4.1") {
        return sse4;
    }
    scalar::l2_f32_scalar
}

#[cfg(target_arch = "x86_64")]
fn resolve_ip() -> DistFn {
    fn avx512(a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: picked only after `avx512f` detection; the public wrapper equalizes lengths.
        unsafe { x86::avx512::ip_f32(a, b) }
    }
    fn avx2(a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: picked only after `avx2` detection; the public wrapper equalizes lengths.
        unsafe { x86::avx2::ip_f32(a, b) }
    }
    fn sse4(a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: picked only after `sse4.1` detection; the public wrapper equalizes lengths.
        unsafe { x86::sse4::ip_f32(a, b) }
    }

    if is_x86_feature_detected!("avx512f") {
        return avx512;
    }
    if is_x86_feature_detected!("avx2") {
        return avx2;
    }
    if is_x86_feature_detected!("sse4.1") {
        return sse4;
    }
    scalar::ip_f32_scalar
}

#[cfg(target_arch = "x86_64")]
fn resolve_cosine() -> DistFn {
    fn avx512(a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: picked only after `avx512f` detection; the public wrapper equalizes lengths.
        unsafe { x86::avx512::cosine_f32(a, b) }
    }
    fn avx2(a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: picked only after `avx2` detection; the public wrapper equalizes lengths.
        unsafe { x86::avx2::cosine_f32(a, b) }
    }
    fn sse4(a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: picked only after `sse4.1` detection; the public wrapper equalizes lengths.
        unsafe { x86::sse4::cosine_f32(a, b) }
    }

    if is_x86_feature_detected!("avx512f") {
        return avx512;
    }
    if is_x86_feature_detected!("avx2") {
        return avx2;
    }
    if is_x86_feature_detected!("sse4.1") {
        return sse4;
    }
    scalar::cosine_f32_scalar
}

#[cfg(target_arch = "x86_64")]
fn resolve_l2_scatter() -> ScatterFn {
    if is_x86_feature_detected!("avx512f") {
        return x86::avx512::l2_scatter_f32;
    }
    if is_x86_feature_detected!("avx2") {
        return x86::avx2::l2_scatter_f32;
    }
    scalar::l2_scatter_f32_scalar
}

#[cfg(target_arch = "x86_64")]
fn resolve_ip_scatter() -> ScatterFn {
    if is_x86_feature_detected!("avx512f") {
        return x86::avx512::ip_scatter_f32;
    }
    if is_x86_feature_detected!("avx2") {
        return x86::avx2::ip_scatter_f32;
    }
    scalar::ip_scatter_f32_scalar
}

#[cfg(target_arch = "x86_64")]
static L2_FN: OnceLock<DistFn> = OnceLock::new();
#[cfg(target_arch = "x86_64")]
static IP_FN: OnceLock<DistFn> = OnceLock::new();
#[cfg(target_arch = "x86_64")]
static COSINE_FN: OnceLock<DistFn> = OnceLock::new();
#[cfg(target_arch = "x86_64")]
static L2_SCATTER_FN: OnceLock<ScatterFn> = OnceLock::new();
#[cfg(target_arch = "x86_64")]
static IP_SCATTER_FN: OnceLock<ScatterFn> = OnceLock::new();
#[cfg(target_arch = "x86_64")]
static HAS_AVX2: OnceLock<bool> = OnceLock::new();

#[cfg(target_arch = "x86_64")]
fn has_avx2() -> bool {
    *HAS_AVX2.get_or_init(|| is_x86_feature_detected!("avx2"))
}

// The SIMD kernels read `b` at every index of `a`; the scalar ones zip.
#[inline]
fn common_prefix<'a>(a: &'a [f32], b: &'a [f32]) -> (&'a [f32], &'a [f32]) {
    let n = a.len().min(b.len());
    (&a[..n], &b[..n])
}

/// L2 (squared Euclidean) distance between two f32 vectors
#[inline]
pub fn l2_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let (a, b) = common_prefix(a, b);
    #[cfg(target_arch = "x86_64")]
    {
        let f = L2_FN.get_or_init(resolve_l2);
        f(a, b)
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64; `common_prefix` made the lengths equal.
        unsafe { aarch64::neon::l2_f32(a, b) }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    scalar::l2_f32_scalar(a, b)
}

/// Inner product (dot product) between two f32 vectors
#[inline]
pub fn ip_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let (a, b) = common_prefix(a, b);
    #[cfg(target_arch = "x86_64")]
    {
        let f = IP_FN.get_or_init(resolve_ip);
        f(a, b)
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64; `common_prefix` made the lengths equal.
        unsafe { aarch64::neon::ip_f32(a, b) }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    scalar::ip_f32_scalar(a, b)
}

/// MIPS-L2 distance (localized spherical injection, e2=0.0).
pub fn mips_l2_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    scalar::mips_l2_f32_scalar(a, b)
}

/// Cosine distance (1 - cosine_similarity) between two f32 vectors
#[inline]
pub fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let (a, b) = common_prefix(a, b);
    #[cfg(target_arch = "x86_64")]
    {
        let f = COSINE_FN.get_or_init(resolve_cosine);
        f(a, b)
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64; `common_prefix` made the lengths equal.
        unsafe { aarch64::neon::cosine_f32(a, b) }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    scalar::cosine_f32_scalar(a, b)
}

/// Hamming distance between two byte slices (bit difference count)
pub fn hamming_u8(a: &[u8], b: &[u8]) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    scalar::hamming_u8_scalar(a, b)
}

/// Batch L2 distances: `matrix` is `m` row vectors of `dim` f32s, `query` is one vector.
pub fn l2_batch_f32(matrix: &[f32], query: &[f32], m: usize, dim: usize, out: &mut [f32]) {
    debug_assert_eq!(matrix.len(), m * dim);
    debug_assert_eq!(query.len(), dim);
    debug_assert_eq!(out.len(), m);
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            assert!(matrix.len() >= m * dim && query.len() >= dim && out.len() >= m);
            // SAFETY: `has_avx2` detected avx2; slice bounds asserted above.
            unsafe { x86::avx2::l2_batch_f32(matrix, query, m, dim, out) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        assert!(matrix.len() >= m * dim && query.len() >= dim && out.len() >= m);
        // SAFETY: NEON is baseline on aarch64; slice bounds asserted above.
        unsafe { aarch64::neon::l2_batch_f32(matrix, query, m, dim, out) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    scalar::l2_batch_f32_scalar(matrix, query, m, dim, out);
}

/// Batch inner product distances.
pub fn ip_batch_f32(matrix: &[f32], query: &[f32], m: usize, dim: usize, out: &mut [f32]) {
    debug_assert_eq!(matrix.len(), m * dim);
    debug_assert_eq!(query.len(), dim);
    debug_assert_eq!(out.len(), m);
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            assert!(matrix.len() >= m * dim && query.len() >= dim && out.len() >= m);
            // SAFETY: `has_avx2` detected avx2; slice bounds asserted above.
            unsafe { x86::avx2::ip_batch_f32(matrix, query, m, dim, out) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        assert!(matrix.len() >= m * dim && query.len() >= dim && out.len() >= m);
        // SAFETY: NEON is baseline on aarch64; slice bounds asserted above.
        unsafe { aarch64::neon::ip_batch_f32(matrix, query, m, dim, out) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    scalar::ip_batch_f32_scalar(matrix, query, m, dim, out);
}

/// Batch cosine distances.
///
/// Precomputes `||query||²` once instead of per-row, saving ~1/3 of FMA work.
pub fn cosine_batch_f32(matrix: &[f32], query: &[f32], m: usize, dim: usize, out: &mut [f32]) {
    debug_assert_eq!(matrix.len(), m * dim);
    debug_assert_eq!(query.len(), dim);
    debug_assert_eq!(out.len(), m);
    scalar::cosine_batch_f32_scalar(matrix, query, m, dim, out);
}

/// Batch MIPS-L2 distances.
pub fn mips_l2_batch_f32(matrix: &[f32], query: &[f32], m: usize, dim: usize, out: &mut [f32]) {
    debug_assert_eq!(matrix.len(), m * dim);
    debug_assert_eq!(query.len(), dim);
    debug_assert_eq!(out.len(), m);
    scalar::mips_l2_batch_f32_scalar(matrix, query, m, dim, out);
}

// Scatter-pointer batch: vectors at arbitrary (non-contiguous) addresses

/// Scatter-pointer L2 batch: computes L2 squared distance from `query` to
/// each vector pointed to by `ptrs[i]`. Results written to `out[i]`.
///
/// # Safety
/// Each `ptrs[i]` must point to at least `dim` readable f32s.
#[inline]
// SAFETY: caller guarantees each `ptrs[i]` has `dim` readable f32s and `query.len() >= dim`.
pub unsafe fn l2_scatter_f32(ptrs: &[*const f32], query: &[f32], dim: usize, out: &mut [f32]) {
    debug_assert!(query.len() >= dim);
    debug_assert!(out.len() >= ptrs.len());
    #[cfg(target_arch = "x86_64")]
    {
        let f = L2_SCATTER_FN.get_or_init(resolve_l2_scatter);
        f(ptrs, query, dim, out)
    }
    #[cfg(target_arch = "aarch64")]
    {
        aarch64::neon::l2_scatter_f32(ptrs, query, dim, out)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    scalar::l2_scatter_f32_scalar(ptrs, query, dim, out);
}

/// Scatter-pointer IP batch: computes inner product from `query` to
/// each vector pointed to by `ptrs[i]`. Results written to `out[i]`.
///
/// # Safety
/// Each `ptrs[i]` must point to at least `dim` readable f32s.
#[inline]
// SAFETY: caller guarantees each `ptrs[i]` has `dim` readable f32s and `query.len() >= dim`.
pub unsafe fn ip_scatter_f32(ptrs: &[*const f32], query: &[f32], dim: usize, out: &mut [f32]) {
    debug_assert!(query.len() >= dim);
    debug_assert!(out.len() >= ptrs.len());
    #[cfg(target_arch = "x86_64")]
    {
        let f = IP_SCATTER_FN.get_or_init(resolve_ip_scatter);
        f(ptrs, query, dim, out)
    }
    #[cfg(target_arch = "aarch64")]
    {
        aarch64::neon::ip_scatter_f32(ptrs, query, dim, out)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    scalar::ip_scatter_f32_scalar(ptrs, query, dim, out);
}

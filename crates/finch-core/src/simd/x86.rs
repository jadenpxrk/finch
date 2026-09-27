//! x86_64 SIMD implementations (AVX-512, AVX2, and SSE4)

pub mod avx512 {
    use crate::simd::tail::{
        cosine_finish, ip_tail, l2_tail, scatter_group, scatter_group_finish, CosineSums,
        SCATTER_PREFETCH_FLOATS as PF_DIST,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// AVX-512 L2 squared distance for f32 vectors.
    ///
    /// Processes 64 floats per iteration with 4 independent accumulators,
    /// leveraging 32 ZMM registers to eliminate spills.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    // SAFETY: caller checks `avx512f` and guarantees `b.len() >= a.len()`.
    pub unsafe fn l2_f32(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        let mut sum0 = _mm512_setzero_ps();
        let mut sum1 = _mm512_setzero_ps();
        let mut sum2 = _mm512_setzero_ps();
        let mut sum3 = _mm512_setzero_ps();

        // 4x unrolled: 64 floats per iteration across 4 independent accumulators.
        let chunks4 = n / 64;
        for i in 0..chunks4 {
            let base = i * 64;
            let d0 = _mm512_sub_ps(
                _mm512_loadu_ps(a_ptr.add(base)),
                _mm512_loadu_ps(b_ptr.add(base)),
            );
            let d1 = _mm512_sub_ps(
                _mm512_loadu_ps(a_ptr.add(base + 16)),
                _mm512_loadu_ps(b_ptr.add(base + 16)),
            );
            let d2 = _mm512_sub_ps(
                _mm512_loadu_ps(a_ptr.add(base + 32)),
                _mm512_loadu_ps(b_ptr.add(base + 32)),
            );
            let d3 = _mm512_sub_ps(
                _mm512_loadu_ps(a_ptr.add(base + 48)),
                _mm512_loadu_ps(b_ptr.add(base + 48)),
            );
            sum0 = _mm512_fmadd_ps(d0, d0, sum0);
            sum1 = _mm512_fmadd_ps(d1, d1, sum1);
            sum2 = _mm512_fmadd_ps(d2, d2, sum2);
            sum3 = _mm512_fmadd_ps(d3, d3, sum3);
        }

        let mut acc = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
        // Handle remaining 16-wide chunks.
        let chunks = n / 16;
        for i in (chunks4 * 4)..chunks {
            let d = _mm512_sub_ps(
                _mm512_loadu_ps(a_ptr.add(i * 16)),
                _mm512_loadu_ps(b_ptr.add(i * 16)),
            );
            acc = _mm512_fmadd_ps(d, d, acc);
        }

        l2_tail(a, b, chunks * 16, _mm512_reduce_add_ps(acc))
    }

    /// AVX-512 inner product for f32 vectors (4-accumulator unrolled).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    // SAFETY: caller checks `avx512f` and guarantees `b.len() >= a.len()`.
    pub unsafe fn ip_f32(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        let mut sum0 = _mm512_setzero_ps();
        let mut sum1 = _mm512_setzero_ps();
        let mut sum2 = _mm512_setzero_ps();
        let mut sum3 = _mm512_setzero_ps();

        let chunks4 = n / 64;
        for i in 0..chunks4 {
            let base = i * 64;
            sum0 = _mm512_fmadd_ps(
                _mm512_loadu_ps(a_ptr.add(base)),
                _mm512_loadu_ps(b_ptr.add(base)),
                sum0,
            );
            sum1 = _mm512_fmadd_ps(
                _mm512_loadu_ps(a_ptr.add(base + 16)),
                _mm512_loadu_ps(b_ptr.add(base + 16)),
                sum1,
            );
            sum2 = _mm512_fmadd_ps(
                _mm512_loadu_ps(a_ptr.add(base + 32)),
                _mm512_loadu_ps(b_ptr.add(base + 32)),
                sum2,
            );
            sum3 = _mm512_fmadd_ps(
                _mm512_loadu_ps(a_ptr.add(base + 48)),
                _mm512_loadu_ps(b_ptr.add(base + 48)),
                sum3,
            );
        }

        let mut acc = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
        let chunks = n / 16;
        for i in (chunks4 * 4)..chunks {
            acc = _mm512_fmadd_ps(
                _mm512_loadu_ps(a_ptr.add(i * 16)),
                _mm512_loadu_ps(b_ptr.add(i * 16)),
                acc,
            );
        }

        ip_tail(a, b, chunks * 16, _mm512_reduce_add_ps(acc))
    }

    /// AVX-512 cosine distance
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    // SAFETY: caller checks `avx512f` and guarantees `b.len() >= a.len()`.
    pub unsafe fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        let mut vdot0 = _mm512_setzero_ps();
        let mut vdot1 = _mm512_setzero_ps();
        let mut vnorm_a0 = _mm512_setzero_ps();
        let mut vnorm_a1 = _mm512_setzero_ps();
        let mut vnorm_b0 = _mm512_setzero_ps();
        let mut vnorm_b1 = _mm512_setzero_ps();

        let chunks2 = n / 32;
        for i in 0..chunks2 {
            let base = i * 32;
            let va0 = _mm512_loadu_ps(a_ptr.add(base));
            let vb0 = _mm512_loadu_ps(b_ptr.add(base));
            let va1 = _mm512_loadu_ps(a_ptr.add(base + 16));
            let vb1 = _mm512_loadu_ps(b_ptr.add(base + 16));
            vdot0 = _mm512_fmadd_ps(va0, vb0, vdot0);
            vdot1 = _mm512_fmadd_ps(va1, vb1, vdot1);
            vnorm_a0 = _mm512_fmadd_ps(va0, va0, vnorm_a0);
            vnorm_a1 = _mm512_fmadd_ps(va1, va1, vnorm_a1);
            vnorm_b0 = _mm512_fmadd_ps(vb0, vb0, vnorm_b0);
            vnorm_b1 = _mm512_fmadd_ps(vb1, vb1, vnorm_b1);
        }

        let mut vdot = _mm512_add_ps(vdot0, vdot1);
        let mut vnorm_a = _mm512_add_ps(vnorm_a0, vnorm_a1);
        let mut vnorm_b = _mm512_add_ps(vnorm_b0, vnorm_b1);
        let chunks = n / 16;
        for i in (chunks2 * 2)..chunks {
            let va = _mm512_loadu_ps(a_ptr.add(i * 16));
            let vb = _mm512_loadu_ps(b_ptr.add(i * 16));
            vdot = _mm512_fmadd_ps(va, vb, vdot);
            vnorm_a = _mm512_fmadd_ps(va, va, vnorm_a);
            vnorm_b = _mm512_fmadd_ps(vb, vb, vnorm_b);
        }

        let sums = CosineSums {
            dot: _mm512_reduce_add_ps(vdot),
            norm_a: _mm512_reduce_add_ps(vnorm_a),
            norm_b: _mm512_reduce_add_ps(vnorm_b),
        };
        cosine_finish(a, b, chunks * 16, sums)
    }

    /// AVX-512 scatter L2 batch
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    // SAFETY: caller checks `avx512f` and guarantees each `ptrs[i]` has `dim` readable f32s and `query.len() >= dim`.
    pub unsafe fn l2_scatter_f32(ptrs: &[*const f32], query: &[f32], dim: usize, out: &mut [f32]) {
        let n = ptrs.len();
        let chunks = dim / 16;
        let q_ptr = query.as_ptr();

        let quads = n / 4;
        for g in 0..quads {
            let [p0, p1, p2, p3] = scatter_group!(ptrs, g);
            if g + 1 < quads {
                _mm_prefetch::<_MM_HINT_T0>(ptrs[(g + 1) * 4] as *const i8);
                _mm_prefetch::<_MM_HINT_T0>(ptrs[(g + 1) * 4 + 1] as *const i8);
                _mm_prefetch::<_MM_HINT_T0>(ptrs[(g + 1) * 4 + 2] as *const i8);
                _mm_prefetch::<_MM_HINT_T0>(ptrs[(g + 1) * 4 + 3] as *const i8);
            }
            let mut s0 = _mm512_setzero_ps();
            let mut s1 = _mm512_setzero_ps();
            let mut s2 = _mm512_setzero_ps();
            let mut s3 = _mm512_setzero_ps();
            for c in 0..chunks {
                let off = c * 16;
                let pf = off + PF_DIST;
                if c & 1 == 0 && pf < dim {
                    _mm_prefetch::<_MM_HINT_T0>(p0.add(pf) as *const i8);
                    _mm_prefetch::<_MM_HINT_T0>(p1.add(pf) as *const i8);
                    _mm_prefetch::<_MM_HINT_T0>(p2.add(pf) as *const i8);
                    _mm_prefetch::<_MM_HINT_T0>(p3.add(pf) as *const i8);
                }
                let q = _mm512_loadu_ps(q_ptr.add(off));
                let d0 = _mm512_sub_ps(_mm512_loadu_ps(p0.add(off)), q);
                let d1 = _mm512_sub_ps(_mm512_loadu_ps(p1.add(off)), q);
                let d2 = _mm512_sub_ps(_mm512_loadu_ps(p2.add(off)), q);
                let d3 = _mm512_sub_ps(_mm512_loadu_ps(p3.add(off)), q);
                s0 = _mm512_fmadd_ps(d0, d0, s0);
                s1 = _mm512_fmadd_ps(d1, d1, s1);
                s2 = _mm512_fmadd_ps(d2, d2, s2);
                s3 = _mm512_fmadd_ps(d3, d3, s3);
            }
            let mut r0 = _mm512_reduce_add_ps(s0);
            let mut r1 = _mm512_reduce_add_ps(s1);
            let mut r2 = _mm512_reduce_add_ps(s2);
            let mut r3 = _mm512_reduce_add_ps(s3);
            scatter_group_finish!(
                l2,
                q_ptr,
                [p0, p1, p2, p3],
                [r0, r1, r2, r3],
                chunks * 16,
                dim,
                out[g]
            );
        }
        for i in (quads * 4)..n {
            let vec = std::slice::from_raw_parts(ptrs[i], dim);
            out[i] = l2_f32(vec, query);
        }
    }

    /// AVX-512 scatter IP batch
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    // SAFETY: caller checks `avx512f` and guarantees each `ptrs[i]` has `dim` readable f32s and `query.len() >= dim`.
    pub unsafe fn ip_scatter_f32(ptrs: &[*const f32], query: &[f32], dim: usize, out: &mut [f32]) {
        let n = ptrs.len();
        let chunks = dim / 16;
        let q_ptr = query.as_ptr();

        let quads = n / 4;
        for g in 0..quads {
            let [p0, p1, p2, p3] = scatter_group!(ptrs, g);
            if g + 1 < quads {
                _mm_prefetch::<_MM_HINT_T0>(ptrs[(g + 1) * 4] as *const i8);
                _mm_prefetch::<_MM_HINT_T0>(ptrs[(g + 1) * 4 + 1] as *const i8);
                _mm_prefetch::<_MM_HINT_T0>(ptrs[(g + 1) * 4 + 2] as *const i8);
                _mm_prefetch::<_MM_HINT_T0>(ptrs[(g + 1) * 4 + 3] as *const i8);
            }
            let mut s0 = _mm512_setzero_ps();
            let mut s1 = _mm512_setzero_ps();
            let mut s2 = _mm512_setzero_ps();
            let mut s3 = _mm512_setzero_ps();
            for c in 0..chunks {
                let off = c * 16;
                let pf = off + PF_DIST;
                if c & 1 == 0 && pf < dim {
                    _mm_prefetch::<_MM_HINT_T0>(p0.add(pf) as *const i8);
                    _mm_prefetch::<_MM_HINT_T0>(p1.add(pf) as *const i8);
                    _mm_prefetch::<_MM_HINT_T0>(p2.add(pf) as *const i8);
                    _mm_prefetch::<_MM_HINT_T0>(p3.add(pf) as *const i8);
                }
                let q = _mm512_loadu_ps(q_ptr.add(off));
                s0 = _mm512_fmadd_ps(_mm512_loadu_ps(p0.add(off)), q, s0);
                s1 = _mm512_fmadd_ps(_mm512_loadu_ps(p1.add(off)), q, s1);
                s2 = _mm512_fmadd_ps(_mm512_loadu_ps(p2.add(off)), q, s2);
                s3 = _mm512_fmadd_ps(_mm512_loadu_ps(p3.add(off)), q, s3);
            }
            let mut r0 = _mm512_reduce_add_ps(s0);
            let mut r1 = _mm512_reduce_add_ps(s1);
            let mut r2 = _mm512_reduce_add_ps(s2);
            let mut r3 = _mm512_reduce_add_ps(s3);
            scatter_group_finish!(
                ip,
                q_ptr,
                [p0, p1, p2, p3],
                [r0, r1, r2, r3],
                chunks * 16,
                dim,
                out[g]
            );
        }
        for i in (quads * 4)..n {
            let vec = std::slice::from_raw_parts(ptrs[i], dim);
            out[i] = ip_f32(vec, query);
        }
    }
}

pub mod avx2 {
    use crate::simd::tail::{
        cosine_finish, ip_tail, l2_tail, scatter_group, scatter_group_finish, CosineSums,
        SCATTER_PREFETCH_FLOATS as PF_DIST,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// AVX2 L2 squared distance for f32 vectors.
    ///
    /// Uses 4 independent accumulators (processing 32 floats per iteration) to
    /// hide FMA latency. Modern x86 CPUs (Haswell+) have 2 FMA units with
    /// 4-cycle latency; a single accumulator achieves ~25% of throughput.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    // SAFETY: caller checks `avx2` and guarantees `b.len() >= a.len()`.
    pub unsafe fn l2_f32(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();

        // 4x unrolled: 32 floats per iteration across 4 independent accumulators.
        let chunks4 = n / 32;
        for i in 0..chunks4 {
            let base = i * 32;
            let d0 = _mm256_sub_ps(
                _mm256_loadu_ps(a_ptr.add(base)),
                _mm256_loadu_ps(b_ptr.add(base)),
            );
            let d1 = _mm256_sub_ps(
                _mm256_loadu_ps(a_ptr.add(base + 8)),
                _mm256_loadu_ps(b_ptr.add(base + 8)),
            );
            let d2 = _mm256_sub_ps(
                _mm256_loadu_ps(a_ptr.add(base + 16)),
                _mm256_loadu_ps(b_ptr.add(base + 16)),
            );
            let d3 = _mm256_sub_ps(
                _mm256_loadu_ps(a_ptr.add(base + 24)),
                _mm256_loadu_ps(b_ptr.add(base + 24)),
            );
            sum0 = _mm256_fmadd_ps(d0, d0, sum0);
            sum1 = _mm256_fmadd_ps(d1, d1, sum1);
            sum2 = _mm256_fmadd_ps(d2, d2, sum2);
            sum3 = _mm256_fmadd_ps(d3, d3, sum3);
        }

        // Reduce 4 accumulators, then handle any remaining 8-wide chunks.
        let mut acc = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
        let chunks = n / 8;
        for i in (chunks4 * 4)..chunks {
            let d = _mm256_sub_ps(
                _mm256_loadu_ps(a_ptr.add(i * 8)),
                _mm256_loadu_ps(b_ptr.add(i * 8)),
            );
            acc = _mm256_fmadd_ps(d, d, acc);
        }

        l2_tail(a, b, chunks * 8, hsum256_ps(acc))
    }

    /// AVX2 inner product for f32 vectors (4-accumulator unrolled).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    // SAFETY: caller checks `avx2` and guarantees `b.len() >= a.len()`.
    pub unsafe fn ip_f32(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();

        let chunks4 = n / 32;
        for i in 0..chunks4 {
            let base = i * 32;
            sum0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a_ptr.add(base)),
                _mm256_loadu_ps(b_ptr.add(base)),
                sum0,
            );
            sum1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a_ptr.add(base + 8)),
                _mm256_loadu_ps(b_ptr.add(base + 8)),
                sum1,
            );
            sum2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a_ptr.add(base + 16)),
                _mm256_loadu_ps(b_ptr.add(base + 16)),
                sum2,
            );
            sum3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a_ptr.add(base + 24)),
                _mm256_loadu_ps(b_ptr.add(base + 24)),
                sum3,
            );
        }

        let mut acc = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
        let chunks = n / 8;
        for i in (chunks4 * 4)..chunks {
            acc = _mm256_fmadd_ps(
                _mm256_loadu_ps(a_ptr.add(i * 8)),
                _mm256_loadu_ps(b_ptr.add(i * 8)),
                acc,
            );
        }

        ip_tail(a, b, chunks * 8, hsum256_ps(acc))
    }

    /// AVX2 cosine distance
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    // SAFETY: caller checks `avx2` and guarantees `b.len() >= a.len()`.
    pub unsafe fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        let mut vdot0 = _mm256_setzero_ps();
        let mut vdot1 = _mm256_setzero_ps();
        let mut vnorm_a0 = _mm256_setzero_ps();
        let mut vnorm_a1 = _mm256_setzero_ps();
        let mut vnorm_b0 = _mm256_setzero_ps();
        let mut vnorm_b1 = _mm256_setzero_ps();

        // 2x unrolled (3 independent FMA chains already break latency deps;
        // unrolling by 2 additionally hides memory latency).
        let chunks2 = n / 16;
        for i in 0..chunks2 {
            let base = i * 16;
            let va0 = _mm256_loadu_ps(a_ptr.add(base));
            let vb0 = _mm256_loadu_ps(b_ptr.add(base));
            let va1 = _mm256_loadu_ps(a_ptr.add(base + 8));
            let vb1 = _mm256_loadu_ps(b_ptr.add(base + 8));
            vdot0 = _mm256_fmadd_ps(va0, vb0, vdot0);
            vdot1 = _mm256_fmadd_ps(va1, vb1, vdot1);
            vnorm_a0 = _mm256_fmadd_ps(va0, va0, vnorm_a0);
            vnorm_a1 = _mm256_fmadd_ps(va1, va1, vnorm_a1);
            vnorm_b0 = _mm256_fmadd_ps(vb0, vb0, vnorm_b0);
            vnorm_b1 = _mm256_fmadd_ps(vb1, vb1, vnorm_b1);
        }

        let mut vdot = _mm256_add_ps(vdot0, vdot1);
        let mut vnorm_a = _mm256_add_ps(vnorm_a0, vnorm_a1);
        let mut vnorm_b = _mm256_add_ps(vnorm_b0, vnorm_b1);
        let chunks = n / 8;
        for i in (chunks2 * 2)..chunks {
            let va = _mm256_loadu_ps(a_ptr.add(i * 8));
            let vb = _mm256_loadu_ps(b_ptr.add(i * 8));
            vdot = _mm256_fmadd_ps(va, vb, vdot);
            vnorm_a = _mm256_fmadd_ps(va, va, vnorm_a);
            vnorm_b = _mm256_fmadd_ps(vb, vb, vnorm_b);
        }

        let sums = CosineSums {
            dot: hsum256_ps(vdot),
            norm_a: hsum256_ps(vnorm_a),
            norm_b: hsum256_ps(vnorm_b),
        };
        cosine_finish(a, b, chunks * 8, sums)
    }

    /// AVX2 batch L2 with query vector hoisting.
    ///
    /// Loads each query chunk once into registers and reuses across all rows,
    /// halving memory traffic for the query.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    // SAFETY: caller checks `avx2` and guarantees `matrix` has `m * dim` f32s, `query` `dim`, `out` `m`.
    pub unsafe fn l2_batch_f32(
        matrix: &[f32],
        query: &[f32],
        m: usize,
        dim: usize,
        out: &mut [f32],
    ) {
        let chunks = dim / 8;
        let q_ptr = query.as_ptr();
        let m_ptr = matrix.as_ptr();

        // Process 2 rows at a time to interleave memory latency
        let pairs = m / 2;
        for p in 0..pairs {
            let r0 = m_ptr.add(p * 2 * dim);
            let r1 = m_ptr.add((p * 2 + 1) * dim);
            let mut s0 = _mm256_setzero_ps();
            let mut s1 = _mm256_setzero_ps();
            for c in 0..chunks {
                let q = _mm256_loadu_ps(q_ptr.add(c * 8));
                let d0 = _mm256_sub_ps(_mm256_loadu_ps(r0.add(c * 8)), q);
                let d1 = _mm256_sub_ps(_mm256_loadu_ps(r1.add(c * 8)), q);
                s0 = _mm256_fmadd_ps(d0, d0, s0);
                s1 = _mm256_fmadd_ps(d1, d1, s1);
            }
            let mut res0 = hsum256_ps(s0);
            let mut res1 = hsum256_ps(s1);
            for (j, &q) in query.iter().enumerate().take(dim).skip(chunks * 8) {
                let d0 = *m_ptr.add(p * 2 * dim + j) - q;
                let d1 = *m_ptr.add((p * 2 + 1) * dim + j) - q;
                res0 += d0 * d0;
                res1 += d1 * d1;
            }
            out[p * 2] = res0;
            out[p * 2 + 1] = res1;
        }
        // Handle odd remainder row
        if !m.is_multiple_of(2) {
            let i = m - 1;
            out[i] = l2_f32(&matrix[i * dim..(i + 1) * dim], query);
        }
    }

    /// AVX2 batch inner product with query vector hoisting.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    // SAFETY: caller checks `avx2` and guarantees `matrix` has `m * dim` f32s, `query` `dim`, `out` `m`.
    pub unsafe fn ip_batch_f32(
        matrix: &[f32],
        query: &[f32],
        m: usize,
        dim: usize,
        out: &mut [f32],
    ) {
        let chunks = dim / 8;
        let q_ptr = query.as_ptr();
        let m_ptr = matrix.as_ptr();

        let pairs = m / 2;
        for p in 0..pairs {
            let r0 = m_ptr.add(p * 2 * dim);
            let r1 = m_ptr.add((p * 2 + 1) * dim);
            let mut s0 = _mm256_setzero_ps();
            let mut s1 = _mm256_setzero_ps();
            for c in 0..chunks {
                let q = _mm256_loadu_ps(q_ptr.add(c * 8));
                s0 = _mm256_fmadd_ps(_mm256_loadu_ps(r0.add(c * 8)), q, s0);
                s1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1.add(c * 8)), q, s1);
            }
            let mut res0 = hsum256_ps(s0);
            let mut res1 = hsum256_ps(s1);
            for (j, &q) in query.iter().enumerate().take(dim).skip(chunks * 8) {
                res0 += *m_ptr.add(p * 2 * dim + j) * q;
                res1 += *m_ptr.add((p * 2 + 1) * dim + j) * q;
            }
            out[p * 2] = res0;
            out[p * 2 + 1] = res1;
        }
        if !m.is_multiple_of(2) {
            let i = m - 1;
            out[i] = ip_f32(&matrix[i * dim..(i + 1) * dim], query);
        }
    }

    /// Horizontal sum of 8 f32 lanes in a __m256 register
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    // SAFETY: caller checks `avx2`.
    unsafe fn hsum256_ps(v: __m256) -> f32 {
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let sum128 = _mm_add_ps(hi, lo);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(shuf, sums);
        let final_sum = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(final_sum)
    }

    // -----------------------------------------------------------------------
    // Scatter-pointer batch: vectors at arbitrary (non-contiguous) addresses.
    //
    // Processes 4 vectors at a time, sharing the query vector in registers
    // across all 4 distance computations.  This gives 4 independent FMA
    // chains, keeping both FMA units busy and hiding memory latency from
    // the scattered vector loads.
    // -----------------------------------------------------------------------

    /// AVX2 scatter L2 batch: computes L2 distance from `query` to each
    /// vector pointed to by `ptrs`.
    ///
    /// # Safety
    /// Each `ptrs[i]` must point to at least `dim` readable f32s.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    // SAFETY: caller checks `avx2` and guarantees each `ptrs[i]` has `dim` readable f32s and `query.len() >= dim`.
    pub unsafe fn l2_scatter_f32(ptrs: &[*const f32], query: &[f32], dim: usize, out: &mut [f32]) {
        let n = ptrs.len();
        let chunks = dim / 8;
        let q_ptr = query.as_ptr();

        // Process groups of 4 vectors at a time.
        let quads = n / 4;
        for g in 0..quads {
            let [p0, p1, p2, p3] = scatter_group!(ptrs, g);
            // Prefetch first cache lines of NEXT group's vectors.
            if g + 1 < quads {
                _mm_prefetch::<_MM_HINT_T0>(ptrs[(g + 1) * 4] as *const i8);
                _mm_prefetch::<_MM_HINT_T0>(ptrs[(g + 1) * 4 + 1] as *const i8);
                _mm_prefetch::<_MM_HINT_T0>(ptrs[(g + 1) * 4 + 2] as *const i8);
                _mm_prefetch::<_MM_HINT_T0>(ptrs[(g + 1) * 4 + 3] as *const i8);
            }
            let mut s0 = _mm256_setzero_ps();
            let mut s1 = _mm256_setzero_ps();
            let mut s2 = _mm256_setzero_ps();
            let mut s3 = _mm256_setzero_ps();
            for c in 0..chunks {
                let off = c * 8;
                // Prefetch ahead within each vector on cache-line boundaries.
                let pf = off + PF_DIST;
                if c & 1 == 0 && pf < dim {
                    _mm_prefetch::<_MM_HINT_T0>(p0.add(pf) as *const i8);
                    _mm_prefetch::<_MM_HINT_T0>(p1.add(pf) as *const i8);
                    _mm_prefetch::<_MM_HINT_T0>(p2.add(pf) as *const i8);
                    _mm_prefetch::<_MM_HINT_T0>(p3.add(pf) as *const i8);
                }
                let q = _mm256_loadu_ps(q_ptr.add(off));
                let d0 = _mm256_sub_ps(_mm256_loadu_ps(p0.add(off)), q);
                let d1 = _mm256_sub_ps(_mm256_loadu_ps(p1.add(off)), q);
                let d2 = _mm256_sub_ps(_mm256_loadu_ps(p2.add(off)), q);
                let d3 = _mm256_sub_ps(_mm256_loadu_ps(p3.add(off)), q);
                s0 = _mm256_fmadd_ps(d0, d0, s0);
                s1 = _mm256_fmadd_ps(d1, d1, s1);
                s2 = _mm256_fmadd_ps(d2, d2, s2);
                s3 = _mm256_fmadd_ps(d3, d3, s3);
            }
            let mut r0 = hsum256_ps(s0);
            let mut r1 = hsum256_ps(s1);
            let mut r2 = hsum256_ps(s2);
            let mut r3 = hsum256_ps(s3);
            scatter_group_finish!(
                l2,
                q_ptr,
                [p0, p1, p2, p3],
                [r0, r1, r2, r3],
                chunks * 8,
                dim,
                out[g]
            );
        }
        // Handle remaining vectors one at a time.
        for i in (quads * 4)..n {
            let vec = std::slice::from_raw_parts(ptrs[i], dim);
            out[i] = l2_f32(vec, query);
        }
    }

    /// AVX2 scatter IP batch: computes inner product from `query` to each
    /// vector pointed to by `ptrs`.
    ///
    /// # Safety
    /// Each `ptrs[i]` must point to at least `dim` readable f32s.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    // SAFETY: caller checks `avx2` and guarantees each `ptrs[i]` has `dim` readable f32s and `query.len() >= dim`.
    pub unsafe fn ip_scatter_f32(ptrs: &[*const f32], query: &[f32], dim: usize, out: &mut [f32]) {
        let n = ptrs.len();
        let chunks = dim / 8;
        let q_ptr = query.as_ptr();

        let quads = n / 4;
        for g in 0..quads {
            let [p0, p1, p2, p3] = scatter_group!(ptrs, g);
            if g + 1 < quads {
                _mm_prefetch::<_MM_HINT_T0>(ptrs[(g + 1) * 4] as *const i8);
                _mm_prefetch::<_MM_HINT_T0>(ptrs[(g + 1) * 4 + 1] as *const i8);
                _mm_prefetch::<_MM_HINT_T0>(ptrs[(g + 1) * 4 + 2] as *const i8);
                _mm_prefetch::<_MM_HINT_T0>(ptrs[(g + 1) * 4 + 3] as *const i8);
            }
            let mut s0 = _mm256_setzero_ps();
            let mut s1 = _mm256_setzero_ps();
            let mut s2 = _mm256_setzero_ps();
            let mut s3 = _mm256_setzero_ps();
            for c in 0..chunks {
                let off = c * 8;
                let pf = off + PF_DIST;
                if c & 1 == 0 && pf < dim {
                    _mm_prefetch::<_MM_HINT_T0>(p0.add(pf) as *const i8);
                    _mm_prefetch::<_MM_HINT_T0>(p1.add(pf) as *const i8);
                    _mm_prefetch::<_MM_HINT_T0>(p2.add(pf) as *const i8);
                    _mm_prefetch::<_MM_HINT_T0>(p3.add(pf) as *const i8);
                }
                let q = _mm256_loadu_ps(q_ptr.add(off));
                s0 = _mm256_fmadd_ps(_mm256_loadu_ps(p0.add(off)), q, s0);
                s1 = _mm256_fmadd_ps(_mm256_loadu_ps(p1.add(off)), q, s1);
                s2 = _mm256_fmadd_ps(_mm256_loadu_ps(p2.add(off)), q, s2);
                s3 = _mm256_fmadd_ps(_mm256_loadu_ps(p3.add(off)), q, s3);
            }
            let mut r0 = hsum256_ps(s0);
            let mut r1 = hsum256_ps(s1);
            let mut r2 = hsum256_ps(s2);
            let mut r3 = hsum256_ps(s3);
            scatter_group_finish!(
                ip,
                q_ptr,
                [p0, p1, p2, p3],
                [r0, r1, r2, r3],
                chunks * 8,
                dim,
                out[g]
            );
        }
        for i in (quads * 4)..n {
            let vec = std::slice::from_raw_parts(ptrs[i], dim);
            out[i] = ip_f32(vec, query);
        }
    }
}

pub mod sse4 {
    use crate::simd::tail::{cosine_finish, ip_tail, l2_tail, CosineSums};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// SSE4 L2 squared distance
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse4.1")]
    // SAFETY: caller checks `sse4.1` and guarantees `b.len() >= a.len()`.
    pub unsafe fn l2_f32(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let mut sum = _mm_setzero_ps();
        let chunks = n / 4;
        for i in 0..chunks {
            let va = _mm_loadu_ps(a.as_ptr().add(i * 4));
            let vb = _mm_loadu_ps(b.as_ptr().add(i * 4));
            let diff = _mm_sub_ps(va, vb);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }
        l2_tail(a, b, chunks * 4, hsum128_ps(sum))
    }

    /// SSE4 inner product
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse4.1")]
    // SAFETY: caller checks `sse4.1` and guarantees `b.len() >= a.len()`.
    pub unsafe fn ip_f32(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let mut sum = _mm_setzero_ps();
        let chunks = n / 4;
        for i in 0..chunks {
            let va = _mm_loadu_ps(a.as_ptr().add(i * 4));
            let vb = _mm_loadu_ps(b.as_ptr().add(i * 4));
            sum = _mm_add_ps(sum, _mm_mul_ps(va, vb));
        }
        ip_tail(a, b, chunks * 4, hsum128_ps(sum))
    }

    /// SSE4 cosine distance
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse4.1")]
    // SAFETY: caller checks `sse4.1` and guarantees `b.len() >= a.len()`.
    pub unsafe fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let mut vdot = _mm_setzero_ps();
        let mut vnorm_a = _mm_setzero_ps();
        let mut vnorm_b = _mm_setzero_ps();
        let chunks = n / 4;
        for i in 0..chunks {
            let va = _mm_loadu_ps(a.as_ptr().add(i * 4));
            let vb = _mm_loadu_ps(b.as_ptr().add(i * 4));
            vdot = _mm_add_ps(vdot, _mm_mul_ps(va, vb));
            vnorm_a = _mm_add_ps(vnorm_a, _mm_mul_ps(va, va));
            vnorm_b = _mm_add_ps(vnorm_b, _mm_mul_ps(vb, vb));
        }
        let sums = CosineSums {
            dot: hsum128_ps(vdot),
            norm_a: hsum128_ps(vnorm_a),
            norm_b: hsum128_ps(vnorm_b),
        };
        cosine_finish(a, b, chunks * 4, sums)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse4.1")]
    // SAFETY: caller checks `sse4.1`.
    unsafe fn hsum128_ps(v: __m128) -> f32 {
        let shuf = _mm_movehdup_ps(v);
        let sums = _mm_add_ps(v, shuf);
        let shuf2 = _mm_movehl_ps(shuf, sums);
        let final_sum = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(final_sum)
    }
}

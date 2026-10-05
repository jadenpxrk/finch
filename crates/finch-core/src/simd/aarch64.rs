//! AArch64 NEON SIMD implementations

pub mod neon {
    use crate::simd::tail::{
        cosine_finish, ip_tail, l2_tail, scatter_group, scatter_group_finish, CosineSums,
        SCATTER_PREFETCH_FLOATS as PF_DIST,
    };
    #[cfg(target_arch = "aarch64")]
    use std::arch::aarch64::*;

    /// L2 squared distance with 4-accumulator unrolling.
    ///
    /// A single accumulator creates a 4-cycle FMA latency dependency chain.
    /// M2 has 2 FMA execution units per performance core; 4 independent accumulators
    /// keep both units fed and approach theoretical throughput (~8 f32/cycle).
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    // SAFETY: caller guarantees `b.len() >= a.len()`; NEON is baseline on aarch64.
    pub unsafe fn l2_f32(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);
        let mut sum2 = vdupq_n_f32(0.0);
        let mut sum3 = vdupq_n_f32(0.0);

        // 4x unrolled: 16 floats per iteration across 4 independent accumulators.
        let chunks4 = n / 16;
        for i in 0..chunks4 {
            let base = i * 16;
            let d0 = vsubq_f32(vld1q_f32(a_ptr.add(base)), vld1q_f32(b_ptr.add(base)));
            let d1 = vsubq_f32(
                vld1q_f32(a_ptr.add(base + 4)),
                vld1q_f32(b_ptr.add(base + 4)),
            );
            let d2 = vsubq_f32(
                vld1q_f32(a_ptr.add(base + 8)),
                vld1q_f32(b_ptr.add(base + 8)),
            );
            let d3 = vsubq_f32(
                vld1q_f32(a_ptr.add(base + 12)),
                vld1q_f32(b_ptr.add(base + 12)),
            );
            sum0 = vfmaq_f32(sum0, d0, d0);
            sum1 = vfmaq_f32(sum1, d1, d1);
            sum2 = vfmaq_f32(sum2, d2, d2);
            sum3 = vfmaq_f32(sum3, d3, d3);
        }

        // Reduce 4 accumulators, then handle any remaining 4-wide chunks.
        let mut acc = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
        let chunks = n / 4;
        for i in (chunks4 * 4)..chunks {
            let d = vsubq_f32(vld1q_f32(a_ptr.add(i * 4)), vld1q_f32(b_ptr.add(i * 4)));
            acc = vfmaq_f32(acc, d, d);
        }

        l2_tail(a, b, chunks * 4, vaddvq_f32(acc))
    }

    /// Inner product with 4-accumulator unrolling.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    // SAFETY: caller guarantees `b.len() >= a.len()`; NEON is baseline on aarch64.
    pub unsafe fn ip_f32(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);
        let mut sum2 = vdupq_n_f32(0.0);
        let mut sum3 = vdupq_n_f32(0.0);

        let chunks4 = n / 16;
        for i in 0..chunks4 {
            let base = i * 16;
            sum0 = vfmaq_f32(sum0, vld1q_f32(a_ptr.add(base)), vld1q_f32(b_ptr.add(base)));
            sum1 = vfmaq_f32(
                sum1,
                vld1q_f32(a_ptr.add(base + 4)),
                vld1q_f32(b_ptr.add(base + 4)),
            );
            sum2 = vfmaq_f32(
                sum2,
                vld1q_f32(a_ptr.add(base + 8)),
                vld1q_f32(b_ptr.add(base + 8)),
            );
            sum3 = vfmaq_f32(
                sum3,
                vld1q_f32(a_ptr.add(base + 12)),
                vld1q_f32(b_ptr.add(base + 12)),
            );
        }

        let mut acc = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
        let chunks = n / 4;
        for i in (chunks4 * 4)..chunks {
            acc = vfmaq_f32(
                acc,
                vld1q_f32(a_ptr.add(i * 4)),
                vld1q_f32(b_ptr.add(i * 4)),
            );
        }

        ip_tail(a, b, chunks * 4, vaddvq_f32(acc))
    }

    /// Cosine distance with 2-accumulator unrolling.
    ///
    /// The 3 independent FMA chains (dot, norm_a, norm_b) already break latency
    /// dependencies; unrolling by 2 additionally hides memory latency.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    // SAFETY: caller guarantees `b.len() >= a.len()`; NEON is baseline on aarch64.
    pub unsafe fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        let mut vdot0 = vdupq_n_f32(0.0);
        let mut vdot1 = vdupq_n_f32(0.0);
        let mut vnorm_a0 = vdupq_n_f32(0.0);
        let mut vnorm_a1 = vdupq_n_f32(0.0);
        let mut vnorm_b0 = vdupq_n_f32(0.0);
        let mut vnorm_b1 = vdupq_n_f32(0.0);

        let chunks2 = n / 8;
        for i in 0..chunks2 {
            let base = i * 8;
            let va0 = vld1q_f32(a_ptr.add(base));
            let vb0 = vld1q_f32(b_ptr.add(base));
            let va1 = vld1q_f32(a_ptr.add(base + 4));
            let vb1 = vld1q_f32(b_ptr.add(base + 4));
            vdot0 = vfmaq_f32(vdot0, va0, vb0);
            vdot1 = vfmaq_f32(vdot1, va1, vb1);
            vnorm_a0 = vfmaq_f32(vnorm_a0, va0, va0);
            vnorm_a1 = vfmaq_f32(vnorm_a1, va1, va1);
            vnorm_b0 = vfmaq_f32(vnorm_b0, vb0, vb0);
            vnorm_b1 = vfmaq_f32(vnorm_b1, vb1, vb1);
        }

        let mut vdot = vaddq_f32(vdot0, vdot1);
        let mut vnorm_a = vaddq_f32(vnorm_a0, vnorm_a1);
        let mut vnorm_b = vaddq_f32(vnorm_b0, vnorm_b1);
        let chunks = n / 4;
        for i in (chunks2 * 2)..chunks {
            let va = vld1q_f32(a_ptr.add(i * 4));
            let vb = vld1q_f32(b_ptr.add(i * 4));
            vdot = vfmaq_f32(vdot, va, vb);
            vnorm_a = vfmaq_f32(vnorm_a, va, va);
            vnorm_b = vfmaq_f32(vnorm_b, vb, vb);
        }

        let sums = CosineSums {
            dot: vaddvq_f32(vdot),
            norm_a: vaddvq_f32(vnorm_a),
            norm_b: vaddvq_f32(vnorm_b),
        };
        cosine_finish(a, b, chunks * 4, sums)
    }

    /// Batch L2 distances with query vector hoisting.
    ///
    /// Loads each query chunk once and reuses across row pairs, halving
    /// query memory traffic and interleaving two rows to hide load latency.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    // SAFETY: caller guarantees `matrix` has `m * dim` f32s, `query` `dim`, `out` `m`; NEON is baseline on aarch64.
    pub unsafe fn l2_batch_f32(
        matrix: &[f32],
        query: &[f32],
        m: usize,
        dim: usize,
        out: &mut [f32],
    ) {
        let chunks = dim / 4;
        let q_ptr = query.as_ptr();
        let m_ptr = matrix.as_ptr();

        let pairs = m / 2;
        for p in 0..pairs {
            let r0 = m_ptr.add(p * 2 * dim);
            let r1 = m_ptr.add((p * 2 + 1) * dim);
            let mut s0 = vdupq_n_f32(0.0);
            let mut s1 = vdupq_n_f32(0.0);
            for c in 0..chunks {
                let q = vld1q_f32(q_ptr.add(c * 4));
                let d0 = vsubq_f32(vld1q_f32(r0.add(c * 4)), q);
                let d1 = vsubq_f32(vld1q_f32(r1.add(c * 4)), q);
                s0 = vfmaq_f32(s0, d0, d0);
                s1 = vfmaq_f32(s1, d1, d1);
            }
            let mut res0 = vaddvq_f32(s0);
            let mut res1 = vaddvq_f32(s1);
            for (j, &qv) in query.iter().enumerate().take(dim).skip(chunks * 4) {
                let d0 = *m_ptr.add(p * 2 * dim + j) - qv;
                let d1 = *m_ptr.add((p * 2 + 1) * dim + j) - qv;
                res0 += d0 * d0;
                res1 += d1 * d1;
            }
            out[p * 2] = res0;
            out[p * 2 + 1] = res1;
        }
        if !m.is_multiple_of(2) {
            let i = m - 1;
            out[i] = l2_f32(&matrix[i * dim..(i + 1) * dim], query);
        }
    }

    /// Batch inner product distances with query vector hoisting.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    // SAFETY: caller guarantees `matrix` has `m * dim` f32s, `query` `dim`, `out` `m`; NEON is baseline on aarch64.
    pub unsafe fn ip_batch_f32(
        matrix: &[f32],
        query: &[f32],
        m: usize,
        dim: usize,
        out: &mut [f32],
    ) {
        let chunks = dim / 4;
        let q_ptr = query.as_ptr();
        let m_ptr = matrix.as_ptr();

        let pairs = m / 2;
        for p in 0..pairs {
            let r0 = m_ptr.add(p * 2 * dim);
            let r1 = m_ptr.add((p * 2 + 1) * dim);
            let mut s0 = vdupq_n_f32(0.0);
            let mut s1 = vdupq_n_f32(0.0);
            for c in 0..chunks {
                let q = vld1q_f32(q_ptr.add(c * 4));
                s0 = vfmaq_f32(s0, vld1q_f32(r0.add(c * 4)), q);
                s1 = vfmaq_f32(s1, vld1q_f32(r1.add(c * 4)), q);
            }
            let mut res0 = vaddvq_f32(s0);
            let mut res1 = vaddvq_f32(s1);
            for (j, &qv) in query.iter().enumerate().take(dim).skip(chunks * 4) {
                res0 += *m_ptr.add(p * 2 * dim + j) * qv;
                res1 += *m_ptr.add((p * 2 + 1) * dim + j) * qv;
            }
            out[p * 2] = res0;
            out[p * 2 + 1] = res1;
        }
        if !m.is_multiple_of(2) {
            let i = m - 1;
            out[i] = ip_f32(&matrix[i * dim..(i + 1) * dim], query);
        }
    }

    // Scatter-pointer batch: vectors at arbitrary (non-contiguous) addresses.
    //
    // Processes 4 vectors simultaneously, sharing the query vector in registers
    // across all 4 distance computations (4 independent FMA chains).

    /// NEON scatter L2 batch.
    ///
    /// # Safety
    /// Each `ptrs[i]` must point to at least `dim` readable f32s.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    // SAFETY: caller guarantees each `ptrs[i]` has `dim` readable f32s and `query.len() >= dim`; NEON is baseline on aarch64.
    pub unsafe fn l2_scatter_f32(ptrs: &[*const f32], query: &[f32], dim: usize, out: &mut [f32]) {
        let n = ptrs.len();
        let chunks = dim / 4;
        let q_ptr = query.as_ptr();

        // Process groups of 4 vectors at a time.
        let quads = n / 4;
        for g in 0..quads {
            let [p0, p1, p2, p3] = scatter_group!(ptrs, g);
            // Prefetch first cache lines of next group's vectors.
            if g + 1 < quads {
                std::arch::asm!("prfm pldl1keep, [{p}]", p = in(reg) ptrs[(g+1)*4], options(nostack, preserves_flags));
                std::arch::asm!("prfm pldl1keep, [{p}]", p = in(reg) ptrs[(g+1)*4+1], options(nostack, preserves_flags));
                std::arch::asm!("prfm pldl1keep, [{p}]", p = in(reg) ptrs[(g+1)*4+2], options(nostack, preserves_flags));
                std::arch::asm!("prfm pldl1keep, [{p}]", p = in(reg) ptrs[(g+1)*4+3], options(nostack, preserves_flags));
            }
            let mut s0 = vdupq_n_f32(0.0);
            let mut s1 = vdupq_n_f32(0.0);
            let mut s2 = vdupq_n_f32(0.0);
            let mut s3 = vdupq_n_f32(0.0);
            for c in 0..chunks {
                let off = c * 4;
                // Prefetch ahead every 4 chunks (every cache line boundary).
                let pf = off + PF_DIST;
                if c & 3 == 0 && pf < dim {
                    std::arch::asm!("prfm pldl1keep, [{p}]", p = in(reg) p0.add(pf), options(nostack, preserves_flags));
                    std::arch::asm!("prfm pldl1keep, [{p}]", p = in(reg) p1.add(pf), options(nostack, preserves_flags));
                    std::arch::asm!("prfm pldl1keep, [{p}]", p = in(reg) p2.add(pf), options(nostack, preserves_flags));
                    std::arch::asm!("prfm pldl1keep, [{p}]", p = in(reg) p3.add(pf), options(nostack, preserves_flags));
                }
                let q = vld1q_f32(q_ptr.add(off));
                let d0 = vsubq_f32(vld1q_f32(p0.add(off)), q);
                let d1 = vsubq_f32(vld1q_f32(p1.add(off)), q);
                let d2 = vsubq_f32(vld1q_f32(p2.add(off)), q);
                let d3 = vsubq_f32(vld1q_f32(p3.add(off)), q);
                s0 = vfmaq_f32(s0, d0, d0);
                s1 = vfmaq_f32(s1, d1, d1);
                s2 = vfmaq_f32(s2, d2, d2);
                s3 = vfmaq_f32(s3, d3, d3);
            }
            let mut r0 = vaddvq_f32(s0);
            let mut r1 = vaddvq_f32(s1);
            let mut r2 = vaddvq_f32(s2);
            let mut r3 = vaddvq_f32(s3);
            scatter_group_finish!(
                l2,
                q_ptr,
                [p0, p1, p2, p3],
                [r0, r1, r2, r3],
                chunks * 4,
                dim,
                out[g]
            );
        }
        for i in (quads * 4)..n {
            let vec = std::slice::from_raw_parts(ptrs[i], dim);
            out[i] = l2_f32(vec, query);
        }
    }

    /// NEON scatter IP batch.
    ///
    /// # Safety
    /// Each `ptrs[i]` must point to at least `dim` readable f32s.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    // SAFETY: caller guarantees each `ptrs[i]` has `dim` readable f32s and `query.len() >= dim`; NEON is baseline on aarch64.
    pub unsafe fn ip_scatter_f32(ptrs: &[*const f32], query: &[f32], dim: usize, out: &mut [f32]) {
        let n = ptrs.len();
        let chunks = dim / 4;
        let q_ptr = query.as_ptr();

        let quads = n / 4;
        for g in 0..quads {
            let [p0, p1, p2, p3] = scatter_group!(ptrs, g);
            if g + 1 < quads {
                std::arch::asm!("prfm pldl1keep, [{p}]", p = in(reg) ptrs[(g+1)*4], options(nostack, preserves_flags));
                std::arch::asm!("prfm pldl1keep, [{p}]", p = in(reg) ptrs[(g+1)*4+1], options(nostack, preserves_flags));
                std::arch::asm!("prfm pldl1keep, [{p}]", p = in(reg) ptrs[(g+1)*4+2], options(nostack, preserves_flags));
                std::arch::asm!("prfm pldl1keep, [{p}]", p = in(reg) ptrs[(g+1)*4+3], options(nostack, preserves_flags));
            }
            let mut s0 = vdupq_n_f32(0.0);
            let mut s1 = vdupq_n_f32(0.0);
            let mut s2 = vdupq_n_f32(0.0);
            let mut s3 = vdupq_n_f32(0.0);
            for c in 0..chunks {
                let off = c * 4;
                let pf = off + PF_DIST;
                if c & 3 == 0 && pf < dim {
                    std::arch::asm!("prfm pldl1keep, [{p}]", p = in(reg) p0.add(pf), options(nostack, preserves_flags));
                    std::arch::asm!("prfm pldl1keep, [{p}]", p = in(reg) p1.add(pf), options(nostack, preserves_flags));
                    std::arch::asm!("prfm pldl1keep, [{p}]", p = in(reg) p2.add(pf), options(nostack, preserves_flags));
                    std::arch::asm!("prfm pldl1keep, [{p}]", p = in(reg) p3.add(pf), options(nostack, preserves_flags));
                }
                let q = vld1q_f32(q_ptr.add(off));
                s0 = vfmaq_f32(s0, vld1q_f32(p0.add(off)), q);
                s1 = vfmaq_f32(s1, vld1q_f32(p1.add(off)), q);
                s2 = vfmaq_f32(s2, vld1q_f32(p2.add(off)), q);
                s3 = vfmaq_f32(s3, vld1q_f32(p3.add(off)), q);
            }
            let mut r0 = vaddvq_f32(s0);
            let mut r1 = vaddvq_f32(s1);
            let mut r2 = vaddvq_f32(s2);
            let mut r3 = vaddvq_f32(s3);
            scatter_group_finish!(
                ip,
                q_ptr,
                [p0, p1, p2, p3],
                [r0, r1, r2, r3],
                chunks * 4,
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

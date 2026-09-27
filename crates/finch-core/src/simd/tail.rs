//! Scalar remainders shared by the per-architecture SIMD kernels; never used inside a vector loop.

/// Floats to prefetch ahead within each scatter row (16 cache lines).
pub(super) const SCATTER_PREFETCH_FLOATS: usize = 256;

/// Adds the squared differences of `a[start..]` and `b[start..]` to `acc`.
#[inline(always)]
pub(super) fn l2_tail(a: &[f32], b: &[f32], start: usize, acc: f32) -> f32 {
    let mut result = acc;
    for i in start..a.len() {
        let d = a[i] - b[i];
        result += d * d;
    }
    result
}

/// Adds the products of `a[start..]` and `b[start..]` to `acc`.
#[inline(always)]
pub(super) fn ip_tail(a: &[f32], b: &[f32], start: usize, acc: f32) -> f32 {
    let mut result = acc;
    for i in start..a.len() {
        result += a[i] * b[i];
    }
    result
}

/// Horizontal sums of the vector part of a cosine kernel.
pub(super) struct CosineSums {
    pub(super) dot: f32,
    pub(super) norm_a: f32,
    pub(super) norm_b: f32,
}

/// Adds the scalar remainder from `start` and returns `1 - cos(a, b)`, or 0 for near-zero norms.
#[inline(always)]
pub(super) fn cosine_finish(a: &[f32], b: &[f32], start: usize, sums: CosineSums) -> f32 {
    let CosineSums {
        mut dot,
        mut norm_a,
        mut norm_b,
    } = sums;
    for i in start..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        1.0 - (dot / denom)
    }
}

/// The four row pointers of scatter group `$g`.
macro_rules! scatter_group {
    ($ptrs:ident, $g:ident) => {
        [
            $ptrs[$g * 4],
            $ptrs[$g * 4 + 1],
            $ptrs[$g * 4 + 2],
            $ptrs[$g * 4 + 3],
        ]
    };
}

/// Scalar columns and output store of a four-row scatter group; a macro so the group loop gains no call.
macro_rules! scatter_group_finish {
    (
        l2,
        $q:ident,
        [$p0:ident, $p1:ident, $p2:ident, $p3:ident],
        [$r0:ident, $r1:ident, $r2:ident, $r3:ident],
        $start:expr,
        $dim:ident,
        $out:ident[$g:ident]
    ) => {
        for j in $start..$dim {
            let qv = *$q.add(j);
            let d0 = *$p0.add(j) - qv;
            let d1 = *$p1.add(j) - qv;
            let d2 = *$p2.add(j) - qv;
            let d3 = *$p3.add(j) - qv;
            $r0 += d0 * d0;
            $r1 += d1 * d1;
            $r2 += d2 * d2;
            $r3 += d3 * d3;
        }
        $out[$g * 4] = $r0;
        $out[$g * 4 + 1] = $r1;
        $out[$g * 4 + 2] = $r2;
        $out[$g * 4 + 3] = $r3;
    };
    (
        ip,
        $q:ident,
        [$p0:ident, $p1:ident, $p2:ident, $p3:ident],
        [$r0:ident, $r1:ident, $r2:ident, $r3:ident],
        $start:expr,
        $dim:ident,
        $out:ident[$g:ident]
    ) => {
        for j in $start..$dim {
            let qv = *$q.add(j);
            $r0 += *$p0.add(j) * qv;
            $r1 += *$p1.add(j) * qv;
            $r2 += *$p2.add(j) * qv;
            $r3 += *$p3.add(j) * qv;
        }
        $out[$g * 4] = $r0;
        $out[$g * 4 + 1] = $r1;
        $out[$g * 4 + 2] = $r2;
        $out[$g * 4 + 3] = $r3;
    };
}

pub(super) use scatter_group;
pub(super) use scatter_group_finish;

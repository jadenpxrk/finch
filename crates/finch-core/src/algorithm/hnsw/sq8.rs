//! SQ8 (Scalar Quantization, 8-bit unsigned) encoding and distance computation
//! for two-phase HNSW search.
//!
//! Each vector is encoded as: [min: f32 LE, max: f32 LE, codes: [u8; dim]]
//! where code[i] = round((v[i] - min) / (max - min) * 255.0).
//!
//! Total bytes per vector: 8 + dim.

/// SQ8-encoded vectors stored contiguously.
pub struct Sq8Codes {
    /// Flat buffer: n vectors, each `sq8_stride` bytes.
    pub data: Vec<u8>,
    /// Bytes per encoded vector: 8 + dim.
    pub sq8_stride: usize,
}

impl Sq8Codes {
    /// Access the raw SQ8 data for vector `id`.
    #[inline]
    pub fn get(&self, id: u32) -> &[u8] {
        let off = id as usize * self.sq8_stride;
        &self.data[off..off + self.sq8_stride]
    }
}

/// Encode a single f32 vector to SQ8 and append to `buf`.
pub fn sq8_encode_append(v: &[f32], buf: &mut Vec<u8>) {
    let dim = v.len();
    let mut vmin = f32::INFINITY;
    let mut vmax = f32::NEG_INFINITY;
    for &x in v.iter() {
        if x < vmin {
            vmin = x;
        }
        if x > vmax {
            vmax = x;
        }
    }
    // Handle degenerate case (constant vector).
    if vmax <= vmin {
        vmax = vmin + 1.0;
    }
    buf.extend_from_slice(&vmin.to_le_bytes());
    buf.extend_from_slice(&vmax.to_le_bytes());
    let inv_range = 255.0 / (vmax - vmin);
    buf.reserve(dim);
    for &x in v.iter() {
        let code = ((x - vmin) * inv_range + 0.5) as u32;
        buf.push(code.min(255) as u8);
    }
}

// ---------------------------------------------------------------------------
// Scalar SQ8 distance functions
// ---------------------------------------------------------------------------

/// Compute inner product between a query (f32) and an SQ8-encoded vector.
///
/// Returns the dot product (not negated). Caller decides sign convention.
#[inline]
pub fn sq8_ip_distance(query: &[f32], sq8_data: &[u8], dim: usize) -> f32 {
    debug_assert!(sq8_data.len() >= 8 + dim);
    let (min, max) = crate::quantizer::f32_pair_header(sq8_data);
    let scale = (max - min) / 255.0;
    let codes = &sq8_data[8..8 + dim];

    let mut dot = 0.0f32;
    let mut i = 0;
    // Unroll 4x for scalar pipelining.
    let end4 = dim & !3;
    while i < end4 {
        let v0 = min + (codes[i] as f32) * scale;
        let v1 = min + (codes[i + 1] as f32) * scale;
        let v2 = min + (codes[i + 2] as f32) * scale;
        let v3 = min + (codes[i + 3] as f32) * scale;
        dot += query[i] * v0 + query[i + 1] * v1 + query[i + 2] * v2 + query[i + 3] * v3;
        i += 4;
    }
    while i < dim {
        dot += query[i] * (min + (codes[i] as f32) * scale);
        i += 1;
    }
    dot
}

/// Compute L2 squared distance between a query (f32) and an SQ8-encoded vector.
#[inline]
pub fn sq8_l2_distance(query: &[f32], sq8_data: &[u8], dim: usize) -> f32 {
    debug_assert!(sq8_data.len() >= 8 + dim);
    let (min, max) = crate::quantizer::f32_pair_header(sq8_data);
    let scale = (max - min) / 255.0;
    let codes = &sq8_data[8..8 + dim];

    let mut sum = 0.0f32;
    let mut i = 0;
    let end4 = dim & !3;
    while i < end4 {
        let d0 = query[i] - (min + (codes[i] as f32) * scale);
        let d1 = query[i + 1] - (min + (codes[i + 1] as f32) * scale);
        let d2 = query[i + 2] - (min + (codes[i + 2] as f32) * scale);
        let d3 = query[i + 3] - (min + (codes[i + 3] as f32) * scale);
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        i += 4;
    }
    while i < dim {
        let d = query[i] - (min + (codes[i] as f32) * scale);
        sum += d * d;
        i += 1;
    }
    sum
}

// ---------------------------------------------------------------------------
// NEON-accelerated SQ8 inner product (aarch64 only)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
pub fn sq8_ip_distance_neon(query: &[f32], sq8_data: &[u8], dim: usize) -> f32 {
    use std::arch::aarch64::*;

    debug_assert!(sq8_data.len() >= 8 + dim);
    let (min_val, max_val) = crate::quantizer::f32_pair_header(sq8_data);
    let scale = (max_val - min_val) / 255.0;
    let codes = &sq8_data[8..8 + dim];
    let query = &query[..dim];

    // SAFETY: `codes` and `query` hold `dim` elements and the vector loads stay below `dim & !15`.
    unsafe {
        let v_scale = vdupq_n_f32(scale);
        let v_min = vdupq_n_f32(min_val);
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        let mut i = 0usize;
        let end16 = dim & !15;

        while i < end16 {
            // Load 16 u8 codes.
            let raw = vld1q_u8(codes.as_ptr().add(i));
            // Widen to two u16x8.
            let lo16 = vmovl_u8(vget_low_u8(raw));
            let hi16 = vmovl_u8(vget_high_u8(raw));
            // Widen to four f32x4.
            let f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16)));
            let f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo16)));
            let f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16)));
            let f3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi16)));

            // dequantize: min + code * scale
            let dq0 = vfmaq_f32(v_min, f0, v_scale);
            let dq1 = vfmaq_f32(v_min, f1, v_scale);
            let dq2 = vfmaq_f32(v_min, f2, v_scale);
            let dq3 = vfmaq_f32(v_min, f3, v_scale);

            // Load query and FMA into accumulators.
            let q0 = vld1q_f32(query.as_ptr().add(i));
            let q1 = vld1q_f32(query.as_ptr().add(i + 4));
            let q2 = vld1q_f32(query.as_ptr().add(i + 8));
            let q3 = vld1q_f32(query.as_ptr().add(i + 12));

            acc0 = vfmaq_f32(acc0, q0, dq0);
            acc1 = vfmaq_f32(acc1, q1, dq1);
            acc2 = vfmaq_f32(acc2, q2, dq2);
            acc3 = vfmaq_f32(acc3, q3, dq3);

            i += 16;
        }

        // Reduce 4 accumulators.
        acc0 = vaddq_f32(acc0, acc1);
        acc2 = vaddq_f32(acc2, acc3);
        acc0 = vaddq_f32(acc0, acc2);
        let mut result = vaddvq_f32(acc0);

        // Handle tail.
        while i < dim {
            let v = min_val + (codes[i] as f32) * scale;
            result += query[i] * v;
            i += 1;
        }
        result
    }
}

#[cfg(target_arch = "aarch64")]
pub fn sq8_l2_distance_neon(query: &[f32], sq8_data: &[u8], dim: usize) -> f32 {
    use std::arch::aarch64::*;

    debug_assert!(sq8_data.len() >= 8 + dim);
    let (min_val, max_val) = crate::quantizer::f32_pair_header(sq8_data);
    let scale = (max_val - min_val) / 255.0;
    let codes = &sq8_data[8..8 + dim];
    let query = &query[..dim];

    // SAFETY: `codes` and `query` hold `dim` elements and the vector loads stay below `dim & !15`.
    unsafe {
        let v_scale = vdupq_n_f32(scale);
        let v_min = vdupq_n_f32(min_val);
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        let mut i = 0usize;
        let end16 = dim & !15;

        while i < end16 {
            let raw = vld1q_u8(codes.as_ptr().add(i));
            let lo16 = vmovl_u8(vget_low_u8(raw));
            let hi16 = vmovl_u8(vget_high_u8(raw));
            let f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16)));
            let f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo16)));
            let f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16)));
            let f3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi16)));

            let dq0 = vfmaq_f32(v_min, f0, v_scale);
            let dq1 = vfmaq_f32(v_min, f1, v_scale);
            let dq2 = vfmaq_f32(v_min, f2, v_scale);
            let dq3 = vfmaq_f32(v_min, f3, v_scale);

            let q0 = vld1q_f32(query.as_ptr().add(i));
            let q1 = vld1q_f32(query.as_ptr().add(i + 4));
            let q2 = vld1q_f32(query.as_ptr().add(i + 8));
            let q3 = vld1q_f32(query.as_ptr().add(i + 12));

            let d0 = vsubq_f32(q0, dq0);
            let d1 = vsubq_f32(q1, dq1);
            let d2 = vsubq_f32(q2, dq2);
            let d3 = vsubq_f32(q3, dq3);

            acc0 = vfmaq_f32(acc0, d0, d0);
            acc1 = vfmaq_f32(acc1, d1, d1);
            acc2 = vfmaq_f32(acc2, d2, d2);
            acc3 = vfmaq_f32(acc3, d3, d3);

            i += 16;
        }

        acc0 = vaddq_f32(acc0, acc1);
        acc2 = vaddq_f32(acc2, acc3);
        acc0 = vaddq_f32(acc0, acc2);
        let mut result = vaddvq_f32(acc0);

        while i < dim {
            let v = min_val + (codes[i] as f32) * scale;
            let d = query[i] - v;
            result += d * d;
            i += 1;
        }
        result
    }
}

/// Runtime-dispatched SQ8 IP distance: uses NEON on aarch64, scalar fallback otherwise.
#[inline]
pub fn sq8_ip_dispatch(query: &[f32], sq8_data: &[u8], dim: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        sq8_ip_distance_neon(query, sq8_data, dim)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        sq8_ip_distance(query, sq8_data, dim)
    }
}

/// Runtime-dispatched SQ8 L2 distance: uses NEON on aarch64, scalar fallback otherwise.
#[inline]
pub fn sq8_l2_dispatch(query: &[f32], sq8_data: &[u8], dim: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        sq8_l2_distance_neon(query, sq8_data, dim)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        sq8_l2_distance(query, sq8_data, dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sq8_encode_decode_ip() {
        let v = vec![0.0f32, 0.25, 0.5, 0.75, 1.0];
        let mut buf = Vec::new();
        sq8_encode_append(&v, &mut buf);
        assert_eq!(buf.len(), 8 + 5);

        let query = vec![1.0f32, 1.0, 1.0, 1.0, 1.0];
        let ip = sq8_ip_distance(&query, &buf, 5);
        // True IP = 0 + 0.25 + 0.5 + 0.75 + 1.0 = 2.5
        assert!((ip - 2.5).abs() < 0.02, "ip={ip}");
    }

    #[test]
    fn test_sq8_encode_decode_l2() {
        let v = vec![0.0f32, 0.25, 0.5, 0.75, 1.0];
        let mut buf = Vec::new();
        sq8_encode_append(&v, &mut buf);

        let query = vec![0.0f32, 0.0, 0.0, 0.0, 0.0];
        let l2 = sq8_l2_distance(&query, &buf, 5);
        // True L2 = 0^2 + 0.25^2 + 0.5^2 + 0.75^2 + 1.0^2 = 1.875
        assert!((l2 - 1.875).abs() < 0.02, "l2={l2}");
    }

    #[test]
    fn test_sq8_roundtrip_precision() {
        // Test with a realistic range
        let dim = 32;
        let v: Vec<f32> = (0..dim)
            .map(|i| -1.0 + 2.0 * (i as f32) / (dim as f32 - 1.0))
            .collect();
        let mut buf = Vec::new();
        sq8_encode_append(&v, &mut buf);

        // Decode manually and check max error
        let (min_val, max_val) = crate::quantizer::f32_pair_header(buf);
        let scale = (max_val - min_val) / 255.0;
        let mut max_err = 0.0f32;
        for i in 0..dim {
            let decoded = min_val + (buf[8 + i] as f32) * scale;
            let err = (decoded - v[i]).abs();
            if err > max_err {
                max_err = err;
            }
        }
        // Max quantization error should be <= scale/2 ~ 2/(2*255) ~ 0.004
        assert!(max_err < 0.01, "max_err={max_err}");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_sq8_neon_matches_scalar() {
        let dim = 48;
        let v: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1 - 2.0).collect();
        let query: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.05 + 0.5).collect();
        let mut buf = Vec::new();
        sq8_encode_append(&v, &mut buf);

        let ip_scalar = sq8_ip_distance(&query, &buf, dim);
        let ip_neon = sq8_ip_distance_neon(&query, &buf, dim);
        assert!(
            (ip_scalar - ip_neon).abs() < 0.01,
            "scalar={ip_scalar}, neon={ip_neon}"
        );

        let l2_scalar = sq8_l2_distance(&query, &buf, dim);
        let l2_neon = sq8_l2_distance_neon(&query, &buf, dim);
        assert!(
            (l2_scalar - l2_neon).abs() < 0.01,
            "scalar={l2_scalar}, neon={l2_neon}"
        );
    }
}

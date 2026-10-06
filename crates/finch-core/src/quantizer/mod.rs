//! Vector quantizers for compressed storage

use finch_types::{MetricType, QuantizeType};
use half::f16;

pub(crate) fn quantize_type_from_u32(v: u32) -> QuantizeType {
    match v {
        1 => QuantizeType::Fp16,
        2 => QuantizeType::Int8,
        3 => QuantizeType::Int4,
        _ => QuantizeType::Undefined,
    }
}

/// Returns the bytes of one `dim`-dimensional vector in the `quantize` encoding.
pub fn bytes_per_vector(quantize: QuantizeType, dim: usize) -> usize {
    match quantize {
        QuantizeType::Undefined => dim * 4,
        QuantizeType::Fp16 => dim * 2,
        QuantizeType::Int8 => 8 + dim,
        QuantizeType::Int4 => 8 + dim.div_ceil(2),
    }
}

/// Append a single vector in the on-disk encoding used by `QuantizeType`.
///
/// This is a streaming-friendly alternative to `Converter::convert` that avoids
/// intermediate allocations (important for large dumps).
pub fn quantize_append(quantize: QuantizeType, input: &[f32], out: &mut Vec<u8>) {
    match quantize {
        QuantizeType::Undefined => {
            out.extend(input.iter().flat_map(|x| x.to_le_bytes()));
        }
        QuantizeType::Fp16 => {
            out.reserve(input.len() * 2);
            for &x in input {
                let bits = f16::from_f32(x).to_bits();
                out.extend_from_slice(&bits.to_le_bytes());
            }
        }
        QuantizeType::Int8 => {
            let min = input.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let range = max - min;
            let scale = if range > 1e-10 { 254.0 / range } else { 1.0 };
            let bias = min;

            out.reserve(8 + input.len());
            out.extend_from_slice(&scale.to_le_bytes());
            out.extend_from_slice(&bias.to_le_bytes());
            for &x in input {
                let q = ((x - bias) * scale).round().clamp(-127.0, 127.0) as i8;
                out.push(q as u8);
            }
        }
        QuantizeType::Int4 => {
            let min = input.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let range = max - min;
            let scale = if range > 1e-10 { 14.0 / range } else { 1.0 };
            let bias = min;

            let packed_len = input.len().div_ceil(2);
            out.reserve(8 + packed_len);
            out.extend_from_slice(&scale.to_le_bytes());
            out.extend_from_slice(&bias.to_le_bytes());

            for chunk in input.chunks(2) {
                let q0 = ((chunk[0] - bias) * scale).round().clamp(0.0, 15.0) as u8;
                let q1 = if chunk.len() > 1 {
                    ((chunk[1] - bias) * scale).round().clamp(0.0, 15.0) as u8
                } else {
                    0
                };
                out.push((q0 & 0x0F) | ((q1 & 0x0F) << 4));
            }
        }
    }
}

/// Compute distance between a query vector (f32) and an on-disk encoded vector.
///
/// Returned value matches the semantics of `Metric::distance()`:
/// - L2: smaller is better
/// - InnerProduct: returns negative dot (smaller is better)
/// - Cosine: returns 1 - cosine_similarity (smaller is better)
pub fn distance_to_quantized(
    metric: MetricType,
    query: &[f32],
    data: &[u8],
    dim: usize,
    quantize: QuantizeType,
) -> f32 {
    let q_sq_norm = query_sq_norm(metric, query, dim);
    distance_to_quantized_with_query_sq_norm(metric, query, q_sq_norm, data, dim, quantize)
}

/// `||query[..dim]||²` for metrics whose quantized distance needs it, else 0.
#[inline]
pub(crate) fn query_sq_norm(metric: MetricType, query: &[f32], dim: usize) -> f32 {
    if matches!(metric, MetricType::Cosine | MetricType::MipsL2) {
        query.iter().take(dim).map(|v| v * v).sum::<f32>()
    } else {
        0.0
    }
}

/// Like `distance_to_quantized`, but allows callers to precompute `query_sq_norm`
/// (sum of squares over the first `dim` elements) to avoid O(dim) work per candidate.
pub(crate) fn distance_to_quantized_with_query_sq_norm(
    metric: MetricType,
    query: &[f32],
    query_sq_norm: f32,
    data: &[u8],
    dim: usize,
    quantize: QuantizeType,
) -> f32 {
    match metric {
        MetricType::InnerProduct => {
            let mut dot = 0.0f32;
            accumulate_quantized(dim, quantize, data, |i, x| {
                dot += x * query[i];
            });
            -dot
        }
        MetricType::Cosine => {
            let mut dot = 0.0f32;
            let mut norm_x = 0.0f32;
            let norm_q = query_sq_norm;
            accumulate_quantized(dim, quantize, data, |i, x| {
                dot += x * query[i];
                norm_x += x * x;
            });
            let denom = (norm_x * norm_q).sqrt();
            if denom < 1e-10 {
                0.0
            } else {
                1.0 - (dot / denom)
            }
        }
        MetricType::MipsL2 => {
            // localized spherical injection (e2=0.0)
            //   dist = 2 - 2 * ip(a,b) / max(||a||^2, ||b||^2)
            let mut dot = 0.0f32;
            let mut norm_x = 0.0f32;
            let norm_q = query_sq_norm;
            accumulate_quantized(dim, quantize, data, |i, x| {
                dot += x * query[i];
                norm_x += x * x;
            });
            let denom = norm_x.max(norm_q);
            2.0 - 2.0 * dot / denom
        }
        MetricType::L2 | MetricType::Undefined | MetricType::Hamming => {
            // Hamming doesn't really make sense for these encodings; treat it as L2.
            let mut sum = 0.0f32;
            accumulate_quantized(dim, quantize, data, |i, x| {
                let d = x - query[i];
                sum += d * d;
            });
            sum
        }
    }
}

fn accumulate_quantized<F: FnMut(usize, f32)>(
    dim: usize,
    quantize: QuantizeType,
    data: &[u8],
    mut f: F,
) {
    match quantize {
        QuantizeType::Undefined => {
            // Interpret as little-endian f32 bytes.
            for (i, chunk) in data.as_chunks::<4>().0.iter().take(dim).enumerate() {
                f(i, f32::from_le_bytes(*chunk));
            }
        }
        QuantizeType::Fp16 => {
            for (i, chunk) in data.as_chunks::<2>().0.iter().take(dim).enumerate() {
                let x = f16::from_bits(u16::from_le_bytes(*chunk)).to_f32();
                f(i, x);
            }
        }
        QuantizeType::Int8 => {
            if data.len() < 8 {
                return;
            }
            let (scale, bias) = f32_pair_header(data);
            let base = 8;
            for i in 0..dim {
                let off = base + i;
                if off >= data.len() {
                    break;
                }
                let q = data[off] as i8 as f32;
                let x = q / scale + bias;
                f(i, x);
            }
        }
        QuantizeType::Int4 => {
            if data.len() < 8 {
                return;
            }
            let (scale, bias) = f32_pair_header(data);
            let base = 8;
            for i in 0..dim {
                let off = base + (i / 2);
                if off >= data.len() {
                    break;
                }
                let byte = data[off];
                let nib = if (i & 1) == 0 {
                    byte & 0x0F
                } else {
                    (byte >> 4) & 0x0F
                };
                let x = (nib as f32) / scale + bias;
                f(i, x);
            }
        }
    }
}

// Reads the two little-endian f32 values that open int8, int4, and SQ8 encodings.
#[inline]
pub(crate) fn f32_pair_header(data: &[u8]) -> (f32, f32) {
    let h = &data[..8];
    (
        f32::from_le_bytes([h[0], h[1], h[2], h[3]]),
        f32::from_le_bytes([h[4], h[5], h[6], h[7]]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn test_distance_to_quantized_mips_l2_matches_reference_localized_spherical_injection() {
        let query = vec![1.0f32, 2.0];
        let v = [3.0f32, 4.0];

        // Encode `v` as fp16 bytes (exact for these small integers).
        let mut bytes = Vec::new();
        for x in v.iter() {
            bytes.extend_from_slice(&f16::from_f32(*x).to_bits().to_le_bytes());
        }

        let ip = 1.0f32 * 3.0 + 2.0 * 4.0;
        let u2 = 3.0f32 * 3.0 + 4.0 * 4.0;
        let v2 = 1.0f32 * 1.0 + 2.0 * 2.0;
        let expected = 2.0 - 2.0 * ip / u2.max(v2);

        let q2 = query.iter().map(|x| x * x).sum::<f32>();
        let got = distance_to_quantized_with_query_sq_norm(
            MetricType::MipsL2,
            &query,
            q2,
            &bytes,
            2,
            QuantizeType::Fp16,
        );
        assert!(
            (got - expected).abs() < 1e-6,
            "got={got} expected={expected}"
        );
    }
}

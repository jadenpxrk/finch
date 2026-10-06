//! Distance metric abstraction and implementations

use crate::simd;
use finch_types::MetricType;

/// Trait for computing distances between vectors
pub trait Metric: Send + Sync {
    /// Returns the distance between `a` and `b`.
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;

    /// Batch distances: matrix rows vs query, results in `out`
    fn batch_distance(&self, matrix: &[f32], query: &[f32], m: usize, dim: usize, out: &mut [f32]);

    /// Returns the metric this implementation computes.
    fn metric_type(&self) -> MetricType;

    /// Whether a smaller distance is better (true for L2; false for IP)
    fn lower_is_better(&self) -> bool {
        true
    }
}

pub(crate) struct L2Metric;
pub(crate) struct IpMetric;
pub(crate) struct CosineMetric;
pub(crate) struct HammingMetric;
pub(crate) struct MipsL2Metric;

impl Metric for L2Metric {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        simd::l2_f32(a, b)
    }
    fn batch_distance(&self, matrix: &[f32], query: &[f32], m: usize, dim: usize, out: &mut [f32]) {
        simd::l2_batch_f32(matrix, query, m, dim, out);
    }
    fn metric_type(&self) -> MetricType {
        MetricType::L2
    }
    fn lower_is_better(&self) -> bool {
        true
    }
}

impl Metric for IpMetric {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        // Return negative IP so smaller = better
        -simd::ip_f32(a, b)
    }
    fn batch_distance(&self, matrix: &[f32], query: &[f32], m: usize, dim: usize, out: &mut [f32]) {
        simd::ip_batch_f32(matrix, query, m, dim, out);
        for v in out.iter_mut() {
            *v = -*v;
        }
    }
    fn metric_type(&self) -> MetricType {
        MetricType::InnerProduct
    }
    fn lower_is_better(&self) -> bool {
        true
    }
}

impl Metric for CosineMetric {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        simd::cosine_f32(a, b)
    }
    fn batch_distance(&self, matrix: &[f32], query: &[f32], m: usize, dim: usize, out: &mut [f32]) {
        simd::cosine_batch_f32(matrix, query, m, dim, out);
    }
    fn metric_type(&self) -> MetricType {
        MetricType::Cosine
    }
    fn lower_is_better(&self) -> bool {
        true
    }
}

impl Metric for HammingMetric {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        hamming_f32_bits(a, b) as f32
    }
    fn batch_distance(&self, matrix: &[f32], query: &[f32], m: usize, dim: usize, out: &mut [f32]) {
        for i in 0..m {
            let row = &matrix[i * dim..(i + 1) * dim];
            out[i] = hamming_f32_bits(row, query) as f32;
        }
    }
    fn metric_type(&self) -> MetricType {
        MetricType::Hamming
    }
    fn lower_is_better(&self) -> bool {
        true
    }
}

impl Metric for MipsL2Metric {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        // MetricType::MipsL2 is *not* plain L2.
        // "Mips SphericalInjection Squared Euclidean Distance" with
        // localized spherical injection (e2=0.0):
        //   dist = 2 - 2 * ip(a,b) / max(||a||^2, ||b||^2)
        simd::mips_l2_f32(a, b)
    }
    fn batch_distance(&self, matrix: &[f32], query: &[f32], m: usize, dim: usize, out: &mut [f32]) {
        simd::mips_l2_batch_f32(matrix, query, m, dim, out);
    }
    fn metric_type(&self) -> MetricType {
        MetricType::MipsL2
    }
    fn lower_is_better(&self) -> bool {
        true
    }
}

// Bit difference count over the raw f32 bits.
fn hamming_f32_bits(a: &[f32], b: &[f32]) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x.to_bits() ^ y.to_bits()).count_ones())
        .sum()
}

/// Factory function
pub fn make_metric(metric_type: MetricType) -> Box<dyn Metric> {
    match metric_type {
        MetricType::L2 | MetricType::Undefined => Box::new(L2Metric),
        MetricType::InnerProduct => Box::new(IpMetric),
        MetricType::Cosine => Box::new(CosineMetric),
        MetricType::Hamming => Box::new(HammingMetric),
        MetricType::MipsL2 => Box::new(MipsL2Metric),
    }
}

/// Normalize an f32 vector to unit length in-place
pub(crate) fn normalize_l2(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        let inv_norm = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv_norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mips_l2_distance_matches_reference_localized_spherical_injection_formula() {
        // formula: 2 - 2*ip/max(u2,v2)
        let a = [1.0f32, 2.0, 0.0];
        let b = [3.0f32, 4.0, 0.0];
        let ip: f32 = 1.0f32 * 3.0 + 2.0 * 4.0;
        let u2: f32 = 1.0f32 * 1.0 + 2.0 * 2.0;
        let v2: f32 = 3.0f32 * 3.0 + 4.0 * 4.0;
        let expected = 2.0 - 2.0 * ip / u2.max(v2);

        let metric = MipsL2Metric;
        let got = metric.distance(&a, &b);
        assert!(
            (got - expected).abs() < 1e-6,
            "got={got} expected={expected}"
        );
    }
}

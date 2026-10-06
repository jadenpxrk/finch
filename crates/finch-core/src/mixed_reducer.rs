use std::collections::HashMap;
use std::hash::Hash;

use finch_types::MetricType;

/// Fuses ranked lists by reciprocal rank fusion.
#[derive(Debug, Clone)]
pub struct RrfReducer {
    /// RRF rank constant `k`. The default is 60.
    pub rank_constant: u32,
}

impl Default for RrfReducer {
    fn default() -> Self {
        Self { rank_constant: 60 }
    }
}

impl RrfReducer {
    #[inline]
    fn rrf_score(&self, rank0: usize) -> f32 {
        // rank0 is zero-based, so the + 1 gives the 1 / (k + rank) of one-based ranks.
        1.0 / ((self.rank_constant as f32) + (rank0 as f32) + 1.0)
    }

    /// Fuse multiple ranked lists using Reciprocal Rank Fusion.
    ///
    /// `lists` must be ordered best-to-worst (rank 0 is best).
    pub fn fuse<K: Eq + Hash + Clone>(&self, lists: &[Vec<K>]) -> Vec<(K, f32)> {
        let mut scores: HashMap<K, f32> = HashMap::new();
        // Ties keep the order in which keys first appear, so equal scores sort the same every run.
        let mut first_seen: HashMap<K, u64> = HashMap::new();
        let mut ordinal: u64 = 0;
        for list in lists {
            for (rank0, key) in list.iter().enumerate() {
                let s = self.rrf_score(rank0);
                *scores.entry(key.clone()).or_insert(0.0) += s;
                if !first_seen.contains_key(key) {
                    first_seen.insert(key.clone(), ordinal);
                    ordinal = ordinal.saturating_add(1);
                }
            }
        }

        let mut out: Vec<(K, f32)> = scores.into_iter().collect();
        out.sort_by(|a, b| {
            let ord = b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal);
            if ord != std::cmp::Ordering::Equal {
                return ord;
            }
            let oa = first_seen.get(&a.0).copied().unwrap_or(u64::MAX);
            let ob = first_seen.get(&b.0).copied().unwrap_or(u64::MAX);
            oa.cmp(&ob)
        });
        out
    }
}

/// Fuses ranked lists by their weighted, normalized scores.
#[derive(Debug, Clone)]
pub struct WeightedReducer {
    /// Metric of the scores, which sets how a score normalizes.
    pub metric: MetricType,
    /// Per-list weight keyed by list name (vector field name). Default weight is 1.0.
    pub weights: HashMap<String, f32>,
}

impl WeightedReducer {
    /// Maps a raw distance or similarity into [0, 1], where higher is better.
    pub(crate) fn normalize_score(&self, score: f32) -> f32 {
        match self.metric {
            MetricType::L2 => 1.0 - 2.0 * score.atan() / std::f32::consts::PI,
            MetricType::InnerProduct => 0.5 + score.atan() / std::f32::consts::PI,
            MetricType::Cosine => 1.0 - score / 2.0,
            _ => {
                // Keep behavior strict; callers should pick a supported metric.
                0.0
            }
        }
    }

    /// Fuse multiple scored lists via normalized weighted sum.
    ///
    /// Each list must be ordered best-to-worst; ordering is not used directly,
    /// but typical callers only supply the top-N per list.
    pub fn fuse<K: Eq + Hash + Clone>(&self, lists: &[(String, Vec<(K, f32)>)]) -> Vec<(K, f32)> {
        let mut scores: HashMap<K, f32> = HashMap::new();
        let mut first_seen: HashMap<K, u64> = HashMap::new();
        let mut ordinal: u64 = 0;
        for (name, list) in lists {
            let w = self.weights.get(name).copied().unwrap_or(1.0);
            for (k, s) in list {
                let ns = self.normalize_score(*s);
                *scores.entry(k.clone()).or_insert(0.0) += ns * w;
                if !first_seen.contains_key(k) {
                    first_seen.insert(k.clone(), ordinal);
                    ordinal = ordinal.saturating_add(1);
                }
            }
        }
        let mut out: Vec<(K, f32)> = scores.into_iter().collect();
        out.sort_by(|a, b| {
            let ord = b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal);
            if ord != std::cmp::Ordering::Equal {
                return ord;
            }
            let oa = first_seen.get(&a.0).copied().unwrap_or(u64::MAX);
            let ob = first_seen.get(&b.0).copied().unwrap_or(u64::MAX);
            oa.cmp(&ob)
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_fuse_prefers_consensus() {
        let r = RrfReducer { rank_constant: 60 };
        let a = vec!["d1", "d2", "d3"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let b = vec!["d2", "d1", "d4"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let out = r.fuse::<String>(&[a, b]);
        let keys: Vec<String> = out.into_iter().map(|(k, _)| k).collect();
        assert_eq!(keys[0], "d1".to_string());
        assert_eq!(keys[1], "d2".to_string());
    }

    #[test]
    fn test_weighted_normalization_matches_reference_shapes() {
        let w = WeightedReducer {
            metric: MetricType::L2,
            weights: HashMap::new(),
        };
        // L2: distance 0 should be near 1.
        assert!(w.normalize_score(0.0) > 0.99);
        // Large distance -> near 0.
        assert!(w.normalize_score(1e6) < 0.01);

        let w = WeightedReducer {
            metric: MetricType::InnerProduct,
            weights: HashMap::new(),
        };
        // IP: score 0 -> 0.5
        let s = w.normalize_score(0.0);
        assert!((s - 0.5).abs() < 1e-6);
    }
}

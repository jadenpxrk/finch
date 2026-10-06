use crate::retrieval::SpanSearchHit;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Retrieval quality of one ranked hit list.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RetrievalMetrics {
    /// Fraction of relevant spans in the top `k` hits.
    pub recall_at_k: f32,
    /// Fraction of relevant spans in the top `k` hits. It equals `recall_at_k`.
    pub gold_evidence_hit_rate: f32,
    /// Inverse rank of the first relevant hit. 0.0 means no relevant hit.
    pub reciprocal_rank: f32,
    /// Normalized discounted cumulative gain of the top `k` hits.
    pub ndcg_at_k: f32,
    /// Fraction of required exact tokens that no top hit contains.
    pub exact_token_miss_rate: f32,
    /// Fraction of the top `k` hits that are stale or inadmissible.
    pub stale_false_positive_rate: f32,
}

/// Hybrid retrieval metrics minus the metrics of one baseline.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BaselineDelta {
    /// Recall gain.
    pub recall_delta: f32,
    /// Gold evidence hit rate gain.
    pub gold_evidence_hit_rate_delta: f32,
    /// Reciprocal rank gain.
    pub reciprocal_rank_delta: f32,
    /// nDCG gain.
    pub ndcg_delta: f32,
    /// Change in the exact token miss rate.
    pub exact_token_miss_rate_delta: f32,
    /// Change in the stale false positive rate.
    pub stale_false_positive_rate_delta: f32,
}

/// Metrics of vector, keyword, and hybrid retrieval, and the hybrid gains.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RetrievalBaselineReport {
    /// Metrics of vector retrieval.
    pub vector: RetrievalMetrics,
    /// Metrics of keyword retrieval.
    pub keyword: RetrievalMetrics,
    /// Metrics of hybrid retrieval.
    pub hybrid: RetrievalMetrics,
    /// Hybrid metrics minus vector metrics.
    pub hybrid_vs_vector: BaselineDelta,
    /// Hybrid metrics minus keyword metrics.
    pub hybrid_vs_keyword: BaselineDelta,
}

impl BaselineDelta {
    /// Returns true when the recall and nDCG gains reach the given minimums.
    pub fn meets_gate(self, min_recall_delta: f32, min_ndcg_delta: f32) -> bool {
        self.recall_delta >= min_recall_delta && self.ndcg_delta >= min_ndcg_delta
    }
}

/// The three ranked hit lists a retrieval baseline compares.
pub struct RetrievalBaselineHits<'a> {
    /// Hits from vector retrieval.
    pub vector: &'a [SpanSearchHit],
    /// Hits from keyword retrieval.
    pub keyword: &'a [SpanSearchHit],
    /// Hits from hybrid retrieval.
    pub hybrid: &'a [SpanSearchHit],
}

/// Scores vector, keyword, and hybrid hits and compares hybrid with each baseline.
pub fn evaluate_retrieval_baseline(
    hits: &RetrievalBaselineHits<'_>,
    relevant_span_ids: &[String],
    required_exact_tokens: &[String],
    stale_or_inadmissible_span_ids: &[String],
    k: usize,
) -> RetrievalBaselineReport {
    let vector = evaluate_span_hits_with_admission(
        hits.vector,
        relevant_span_ids,
        required_exact_tokens,
        stale_or_inadmissible_span_ids,
        k,
    );
    let keyword = evaluate_span_hits_with_admission(
        hits.keyword,
        relevant_span_ids,
        required_exact_tokens,
        stale_or_inadmissible_span_ids,
        k,
    );
    let hybrid = evaluate_span_hits_with_admission(
        hits.hybrid,
        relevant_span_ids,
        required_exact_tokens,
        stale_or_inadmissible_span_ids,
        k,
    );
    RetrievalBaselineReport {
        vector,
        keyword,
        hybrid,
        hybrid_vs_vector: compare_to_baseline(hybrid, vector),
        hybrid_vs_keyword: compare_to_baseline(hybrid, keyword),
    }
}

#[cfg(test)]
pub(crate) fn evaluate_span_hits(
    hits: &[SpanSearchHit],
    relevant_span_ids: &[String],
    k: usize,
) -> RetrievalMetrics {
    evaluate_span_hits_with_admission(hits, relevant_span_ids, &[], &[], k)
}

/// Scores the top `k` hits against relevant, required-token, and stale span lists.
pub fn evaluate_span_hits_with_admission(
    hits: &[SpanSearchHit],
    relevant_span_ids: &[String],
    required_exact_tokens: &[String],
    stale_or_inadmissible_span_ids: &[String],
    k: usize,
) -> RetrievalMetrics {
    if k == 0 || relevant_span_ids.is_empty() {
        return RetrievalMetrics {
            recall_at_k: 0.0,
            gold_evidence_hit_rate: 0.0,
            reciprocal_rank: 0.0,
            ndcg_at_k: 0.0,
            exact_token_miss_rate: if required_exact_tokens.is_empty() {
                0.0
            } else {
                1.0
            },
            stale_false_positive_rate: 0.0,
        };
    }

    let relevant = relevant_span_ids.iter().collect::<HashSet<_>>();
    let inadmissible = stale_or_inadmissible_span_ids
        .iter()
        .collect::<HashSet<_>>();
    let mut hits_found = 0usize;
    let mut reciprocal_rank = 0.0f32;
    let mut dcg = 0.0f32;
    let mut stale_hits = 0usize;
    let top_hits = hits.iter().take(k).collect::<Vec<_>>();

    for (rank, hit) in top_hits.iter().enumerate() {
        if relevant.contains(&hit.span.id) {
            hits_found += 1;
            if reciprocal_rank == 0.0 {
                reciprocal_rank = 1.0 / (rank as f32 + 1.0);
            }
            dcg += 1.0 / ((rank + 2) as f32).log2();
        }
        if inadmissible.contains(&hit.span.id) {
            stale_hits += 1;
        }
    }

    let ideal_hits = relevant.len().min(k);
    let idcg = (0..ideal_hits)
        .map(|rank| 1.0 / ((rank + 2) as f32).log2())
        .sum::<f32>();
    let missed_exact_tokens = required_exact_tokens
        .iter()
        .filter(|token| {
            let token = token.to_lowercase();
            !top_hits.iter().any(|hit| {
                hit.span.text.to_lowercase().contains(&token)
                    || hit.span.lexical_text.to_lowercase().contains(&token)
            })
        })
        .count();
    let inspected_hits = top_hits.len();
    let recall_at_k = hits_found as f32 / relevant.len() as f32;

    RetrievalMetrics {
        recall_at_k,
        gold_evidence_hit_rate: recall_at_k,
        reciprocal_rank,
        ndcg_at_k: if idcg > 0.0 { dcg / idcg } else { 0.0 },
        exact_token_miss_rate: if required_exact_tokens.is_empty() {
            0.0
        } else {
            missed_exact_tokens as f32 / required_exact_tokens.len() as f32
        },
        stale_false_positive_rate: if inspected_hits == 0 {
            0.0
        } else {
            stale_hits as f32 / inspected_hits as f32
        },
    }
}

pub(crate) fn compare_to_baseline(
    candidate: RetrievalMetrics,
    baseline: RetrievalMetrics,
) -> BaselineDelta {
    BaselineDelta {
        recall_delta: candidate.recall_at_k - baseline.recall_at_k,
        gold_evidence_hit_rate_delta: candidate.gold_evidence_hit_rate
            - baseline.gold_evidence_hit_rate,
        reciprocal_rank_delta: candidate.reciprocal_rank - baseline.reciprocal_rank,
        ndcg_delta: candidate.ndcg_at_k - baseline.ndcg_at_k,
        exact_token_miss_rate_delta: candidate.exact_token_miss_rate
            - baseline.exact_token_miss_rate,
        stale_false_positive_rate_delta: candidate.stale_false_positive_rate
            - baseline.stale_false_positive_rate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{MemoryScope, MemoryStatus, SourceType, SpanRecord, Visibility};

    fn hit(id: &str) -> SpanSearchHit {
        SpanSearchHit {
            span: SpanRecord {
                id: id.to_string(),
                scope: MemoryScope::new("eval"),
                status: MemoryStatus::Active,
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                source_type: SourceType::Episode,
                source_id: "ep".to_string(),
                byte_start: 0,
                byte_end: 1,
                char_start: 0,
                char_end: 1,
                token_start: None,
                token_end: None,
                span_index: 0,
                text: id.to_string(),
                text_hash: id.to_string(),
                chunker_version: "test".to_string(),
                embedding_model: None,
                embedding_version: None,
                lexical_text: id.to_string(),
                created_at_ms: 0,
                valid_from_ms: None,
                valid_to_ms: None,
                provenance: Vec::new(),
            },
            score: 1.0,
        }
    }

    #[test]
    fn evaluate_span_hits_computes_basic_metrics() {
        let hits = vec![hit("miss"), hit("gold_1"), hit("gold_2")];
        let metrics = evaluate_span_hits(&hits, &["gold_1".to_string(), "gold_2".to_string()], 3);
        assert_eq!(metrics.recall_at_k, 1.0);
        assert_eq!(metrics.gold_evidence_hit_rate, 1.0);
        assert_eq!(metrics.reciprocal_rank, 0.5);
        assert!(metrics.ndcg_at_k > 0.0);
    }

    #[test]
    fn evaluate_span_hits_measures_exact_token_and_stale_failures() {
        let hits = vec![hit("stale"), hit("gold_1")];
        let metrics = evaluate_span_hits_with_admission(
            &hits,
            &["gold_1".to_string()],
            &["gold_1".to_string(), "missing-token".to_string()],
            &["stale".to_string()],
            2,
        );
        assert_eq!(metrics.recall_at_k, 1.0);
        assert_eq!(metrics.exact_token_miss_rate, 0.5);
        assert_eq!(metrics.stale_false_positive_rate, 0.5);
    }

    #[test]
    fn baseline_delta_gate_is_explicit() {
        let candidate = RetrievalMetrics {
            recall_at_k: 0.8,
            gold_evidence_hit_rate: 0.8,
            reciprocal_rank: 1.0,
            ndcg_at_k: 0.9,
            exact_token_miss_rate: 0.0,
            stale_false_positive_rate: 0.0,
        };
        let baseline = RetrievalMetrics {
            recall_at_k: 0.7,
            gold_evidence_hit_rate: 0.7,
            reciprocal_rank: 1.0,
            ndcg_at_k: 0.8,
            exact_token_miss_rate: 0.2,
            stale_false_positive_rate: 0.1,
        };
        assert!(compare_to_baseline(candidate, baseline).meets_gate(0.05, 0.05));
        assert_eq!(
            compare_to_baseline(candidate, baseline).exact_token_miss_rate_delta,
            -0.2
        );
    }

    #[test]
    fn retrieval_metrics_are_json_serializable() {
        let metrics = RetrievalMetrics {
            recall_at_k: 1.0,
            gold_evidence_hit_rate: 1.0,
            reciprocal_rank: 0.5,
            ndcg_at_k: 0.8,
            exact_token_miss_rate: 0.0,
            stale_false_positive_rate: 0.0,
        };
        let json = serde_json::to_string(&metrics).unwrap();
        assert!(json.contains("recall_at_k"));
        assert_eq!(
            serde_json::from_str::<RetrievalMetrics>(&json).unwrap(),
            metrics
        );
    }

    #[test]
    fn retrieval_baseline_report_compares_three_paths() {
        let vector_hits = vec![hit("miss")];
        let keyword_hits = vec![hit("gold")];
        let hybrid_hits = vec![hit("gold")];
        let report = evaluate_retrieval_baseline(
            &RetrievalBaselineHits {
                vector: &vector_hits,
                keyword: &keyword_hits,
                hybrid: &hybrid_hits,
            },
            &["gold".to_string()],
            &["gold".to_string()],
            &[],
            1,
        );

        assert_eq!(report.vector.recall_at_k, 0.0);
        assert_eq!(report.keyword.recall_at_k, 1.0);
        assert_eq!(report.hybrid.recall_at_k, 1.0);
        assert_eq!(report.hybrid_vs_vector.recall_delta, 1.0);
        assert_eq!(report.hybrid_vs_keyword.recall_delta, 0.0);
    }
}

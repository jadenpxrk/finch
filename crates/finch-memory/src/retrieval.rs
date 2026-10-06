use crate::types::{CorrectionOperation, CorrectionRecord, MemoryScope, MemoryStatus, SpanRecord};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
#[cfg(test)]
pub(crate) struct SpanSearchCandidate<'a> {
    pub span: &'a SpanRecord,
    pub embedding: &'a [f32],
}

#[derive(Debug, Clone)]
pub(crate) struct SpanSearchOptions {
    pub k: usize,
    pub scope: Option<MemoryScope>,
    pub at_ms: Option<i64>,
    pub include_historical: bool,
    pub include_inactive: bool,
}

impl Default for SpanSearchOptions {
    fn default() -> Self {
        Self {
            k: 10,
            scope: None,
            at_ms: None,
            include_historical: false,
            include_inactive: false,
        }
    }
}

/// A span that a search returned, with its score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpanSearchHit {
    /// Span that matched.
    pub span: SpanRecord,
    /// Ranking score. A higher score ranks first.
    pub score: f32,
}

pub(crate) fn span_active_at(span: &SpanRecord, at_ms: Option<i64>) -> bool {
    if !matches!(span.status, MemoryStatus::Active) {
        return false;
    }
    let Some(at_ms) = at_ms else {
        return true;
    };
    span.valid_from_ms.is_none_or(|from| from <= at_ms)
        && span.valid_to_ms.is_none_or(|to| at_ms < to)
}

#[cfg(test)]
pub(crate) fn search_spans<'a>(
    query_embedding: &[f32],
    candidates: impl IntoIterator<Item = SpanSearchCandidate<'a>>,
    options: SpanSearchOptions,
) -> Vec<SpanSearchHit> {
    if query_embedding.is_empty() || options.k == 0 {
        return Vec::new();
    }
    let mut hits: Vec<SpanSearchHit> = candidates
        .into_iter()
        .filter(|candidate| {
            (options.include_inactive || matches!(candidate.span.status, MemoryStatus::Active))
                && (options.include_historical || span_active_at(candidate.span, options.at_ms))
                && options
                    .scope
                    .as_ref()
                    .is_none_or(|scope| candidate.span.scope.matches_filter(scope))
        })
        .filter_map(|candidate| {
            cosine_similarity(query_embedding, candidate.embedding).map(|score| SpanSearchHit {
                span: candidate.span.clone(),
                score,
            })
        })
        .collect();
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.span.id.cmp(&b.span.id))
    });
    hits.truncate(options.k);
    hits
}

pub(crate) fn search_spans_keyword<'a>(
    query_text: &str,
    spans: impl IntoIterator<Item = &'a SpanRecord>,
    options: SpanSearchOptions,
) -> Vec<SpanSearchHit> {
    if query_text.trim().is_empty() || options.k == 0 {
        return Vec::new();
    }
    let query_terms = unique_terms(lexical_terms(query_text));
    if query_terms.is_empty() {
        return Vec::new();
    }

    let docs = spans
        .into_iter()
        .filter(|span| {
            (options.include_inactive || matches!(span.status, MemoryStatus::Active))
                && (options.include_historical || span_active_at(span, options.at_ms))
                && options
                    .scope
                    .as_ref()
                    .is_none_or(|scope| span.scope.matches_filter(scope))
        })
        .map(|span| {
            let text = if span.lexical_text.is_empty() {
                &span.text
            } else {
                &span.lexical_text
            };
            (span, lexical_terms(text))
        })
        .filter(|(_, terms)| !terms.is_empty())
        .collect::<Vec<_>>();
    if docs.is_empty() {
        return Vec::new();
    }

    let doc_count = docs.len();
    let avg_len =
        docs.iter().map(|(_, terms)| terms.len()).sum::<usize>() as f32 / doc_count as f32;
    let mut doc_freq = HashMap::<String, usize>::new();
    for (_, terms) in &docs {
        let mut seen = HashSet::<&str>::new();
        for term in terms {
            if query_terms.iter().any(|query| query == term) && seen.insert(term.as_str()) {
                *doc_freq.entry(term.clone()).or_default() += 1;
            }
        }
    }

    let mut hits = docs
        .into_iter()
        .filter_map(|(span, terms)| {
            let score = bm25_score(&query_terms, &terms, &doc_freq, doc_count, avg_len);
            (score > 0.0).then(|| SpanSearchHit {
                span: span.clone(),
                score,
            })
        })
        .collect::<Vec<_>>();
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.span.id.cmp(&b.span.id))
    });
    hits.truncate(options.k);
    hits
}

pub(crate) fn hybrid_fuse_rrf(
    vector_hits: &[SpanSearchHit],
    keyword_hits: &[SpanSearchHit],
    k: usize,
) -> Vec<SpanSearchHit> {
    if k == 0 {
        return Vec::new();
    }
    let mut by_id = HashMap::<String, (SpanRecord, f32)>::new();
    add_rrf_scores(&mut by_id, vector_hits);
    add_rrf_scores(&mut by_id, keyword_hits);
    let mut hits = by_id
        .into_iter()
        .map(|(_, (span, score))| SpanSearchHit { span, score })
        .collect::<Vec<_>>();
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.span.id.cmp(&b.span.id))
    });
    hits.truncate(k);
    hits
}

fn add_rrf_scores(by_id: &mut HashMap<String, (SpanRecord, f32)>, hits: &[SpanSearchHit]) {
    const RRF_K: f32 = 60.0;
    for (rank, hit) in hits.iter().enumerate() {
        let score = 1.0 / (RRF_K + rank as f32 + 1.0);
        by_id
            .entry(hit.span.id.clone())
            .and_modify(|(_, total)| *total += score)
            .or_insert_with(|| (hit.span.clone(), score));
    }
}

#[cfg(test)]
pub(crate) fn apply_corrections_to_span_hits(
    hits: &[SpanSearchHit],
    corrections: &[CorrectionRecord],
) -> Vec<SpanSearchHit> {
    apply_corrections_to_span_hits_at(hits, corrections, None)
}

pub(crate) fn apply_corrections_to_span_hits_at(
    hits: &[SpanSearchHit],
    corrections: &[CorrectionRecord],
    at_ms: Option<i64>,
) -> Vec<SpanSearchHit> {
    let mut ordered = corrections
        .iter()
        .filter(|correction| matches!(correction.status, MemoryStatus::Active))
        .filter(|correction| correction.target_type == "span")
        .filter(|correction| at_ms.is_none_or(|at| correction.effective_at_ms <= at))
        .collect::<Vec<_>>();
    ordered.sort_by(|a, b| {
        a.effective_at_ms
            .cmp(&b.effective_at_ms)
            .then_with(|| a.id.cmp(&b.id))
    });

    let mut excluded = HashSet::new();
    for correction in ordered {
        let excludes = match correction.operation {
            CorrectionOperation::Restore => false,
            CorrectionOperation::Retract
            | CorrectionOperation::Replace
            | CorrectionOperation::Tombstone
            | CorrectionOperation::Forget
            | CorrectionOperation::MarkStale => true,
            CorrectionOperation::Assert
            | CorrectionOperation::Merge
            | CorrectionOperation::Split => continue,
        };
        for hit in hits
            .iter()
            .filter(|hit| correction_targets_span_hit(correction, hit))
        {
            if excludes {
                excluded.insert(hit.span.id.clone());
            } else {
                excluded.remove(&hit.span.id);
            }
        }
    }

    hits.iter()
        .filter(|hit| !excluded.contains(&hit.span.id))
        .cloned()
        .collect()
}

fn correction_targets_span_hit(correction: &CorrectionRecord, hit: &SpanSearchHit) -> bool {
    // A selector sweeps text, so it only reaches spans of the correction's exact scope.
    correction.target_ids.iter().any(|id| id == &hit.span.id)
        || correction
            .target_selector
            .as_deref()
            .is_some_and(|selector| {
                hit.span.scope == correction.scope && span_matches_selector(&hit.span, selector)
            })
}

fn span_matches_selector(span: &SpanRecord, selector: &str) -> bool {
    let selector = selector.trim().to_lowercase();
    if selector.is_empty() {
        return false;
    }
    span.text.to_lowercase().contains(&selector)
        || span.lexical_text.to_lowercase().contains(&selector)
}

fn bm25_score(
    query_terms: &[String],
    doc_terms: &[String],
    doc_freq: &HashMap<String, usize>,
    doc_count: usize,
    avg_len: f32,
) -> f32 {
    let k1 = 1.2;
    let b = 0.75;
    let mut term_freq = HashMap::<&str, usize>::new();
    for term in doc_terms {
        *term_freq.entry(term).or_default() += 1;
    }
    let doc_len = doc_terms.len() as f32;
    query_terms
        .iter()
        .filter_map(|term| {
            let tf = *term_freq.get(term.as_str())? as f32;
            let df = *doc_freq.get(term.as_str()).unwrap_or(&0) as f32;
            let idf = ((doc_count as f32 - df + 0.5) / (df + 0.5) + 1.0).ln();
            let denom = tf + k1 * (1.0 - b + b * doc_len / avg_len.max(1.0));
            Some(idf * (tf * (k1 + 1.0)) / denom)
        })
        .sum()
}

/// Splits text into lowercase alphanumeric terms.
pub fn lexical_terms(text: &str) -> Vec<String> {
    text.split(|ch: char| !ch.is_alphanumeric())
        .filter(|term| !term.is_empty())
        .map(str::to_lowercase)
        .collect()
}

fn unique_terms(terms: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut unique = Vec::new();
    for term in terms {
        if seen.insert(term.clone()) {
            unique.push(term);
        }
    }
    unique
}

#[cfg(test)]
fn cosine_similarity(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return None;
    }
    Some(dot / (norm_a.sqrt() * norm_b.sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        ActorKind, CorrectionAuthority, CorrectionOperation, MemoryScope, SourceType, Visibility,
    };

    fn span(id: &str, status: MemoryStatus, valid_from_ms: Option<i64>) -> SpanRecord {
        SpanRecord {
            id: id.to_string(),
            scope: MemoryScope::new("default"),
            status,
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            source_type: SourceType::Episode,
            source_id: "ep_1".to_string(),
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
            valid_from_ms,
            valid_to_ms: None,
            provenance: Vec::new(),
        }
    }

    #[test]
    fn search_spans_filters_inactive_and_future_spans() {
        let active = span("active", MemoryStatus::Active, Some(1));
        let future = span("future", MemoryStatus::Active, Some(10));
        let tombstoned = span("gone", MemoryStatus::Tombstoned, Some(1));
        let candidates = [
            SpanSearchCandidate {
                span: &active,
                embedding: &[1.0, 0.0],
            },
            SpanSearchCandidate {
                span: &future,
                embedding: &[1.0, 0.0],
            },
            SpanSearchCandidate {
                span: &tombstoned,
                embedding: &[1.0, 0.0],
            },
        ];
        let hits = search_spans(
            &[1.0, 0.0],
            candidates,
            SpanSearchOptions {
                k: 10,
                at_ms: Some(5),
                ..SpanSearchOptions::default()
            },
        );
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].span.id, "active");
    }

    #[test]
    fn search_spans_scores_by_cosine() {
        let close = span("close", MemoryStatus::Active, None);
        let far = span("far", MemoryStatus::Active, None);
        let candidates = [
            SpanSearchCandidate {
                span: &far,
                embedding: &[0.0, 1.0],
            },
            SpanSearchCandidate {
                span: &close,
                embedding: &[1.0, 0.0],
            },
        ];
        let hits = search_spans(&[1.0, 0.0], candidates, SpanSearchOptions::default());
        assert_eq!(hits[0].span.id, "close");
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn search_spans_filters_by_scope() {
        let mut wanted = span("wanted", MemoryStatus::Active, None);
        wanted.scope.user_id = Some("user_1".to_string());
        let mut other = span("other", MemoryStatus::Active, None);
        other.scope.user_id = Some("user_2".to_string());
        let mut scope = MemoryScope::new("default");
        scope.user_id = Some("user_1".to_string());
        let hits = search_spans(
            &[1.0, 0.0],
            [
                SpanSearchCandidate {
                    span: &wanted,
                    embedding: &[1.0, 0.0],
                },
                SpanSearchCandidate {
                    span: &other,
                    embedding: &[1.0, 0.0],
                },
            ],
            SpanSearchOptions {
                scope: Some(scope),
                ..SpanSearchOptions::default()
            },
        );
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].span.id, "wanted");
    }

    #[test]
    fn keyword_search_finds_exact_terms() {
        let mut exact = span("exact", MemoryStatus::Active, None);
        exact.lexical_text = "invoice x123 deadline".to_string();
        let mut semantic_only = span("semantic", MemoryStatus::Active, None);
        semantic_only.lexical_text = "billing schedule".to_string();
        let hits = search_spans_keyword(
            "x123",
            [&exact, &semantic_only],
            SpanSearchOptions::default(),
        );
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].span.id, "exact");
    }

    #[test]
    fn hybrid_rrf_favors_hits_seen_by_both_signals() {
        let overlap = span("overlap", MemoryStatus::Active, None);
        let vector_only = span("vector", MemoryStatus::Active, None);
        let keyword_only = span("keyword", MemoryStatus::Active, None);
        let vector_hits = vec![
            SpanSearchHit {
                span: overlap.clone(),
                score: 0.9,
            },
            SpanSearchHit {
                span: vector_only,
                score: 0.8,
            },
        ];
        let keyword_hits = vec![
            SpanSearchHit {
                span: keyword_only,
                score: 2.0,
            },
            SpanSearchHit {
                span: overlap,
                score: 1.0,
            },
        ];
        let hits = hybrid_fuse_rrf(&vector_hits, &keyword_hits, 3);
        assert_eq!(hits[0].span.id, "overlap");
    }

    #[test]
    fn corrections_remove_and_restore_span_hits() {
        let removed = span("removed", MemoryStatus::Active, None);
        let kept = span("kept", MemoryStatus::Active, None);
        let hits = vec![
            SpanSearchHit {
                span: removed,
                score: 1.0,
            },
            SpanSearchHit {
                span: kept,
                score: 0.5,
            },
        ];
        let tombstone = correction("c1", CorrectionOperation::Tombstone, "removed", 10);
        let filtered = apply_corrections_to_span_hits(&hits, &[tombstone]);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].span.id, "kept");

        let tombstone = correction("c1", CorrectionOperation::Tombstone, "removed", 10);
        let restore = correction("c2", CorrectionOperation::Restore, "removed", 11);
        let restored = apply_corrections_to_span_hits(&hits, &[restore, tombstone]);
        assert_eq!(restored.len(), 2);
    }

    #[test]
    fn corrections_can_target_span_hits_by_selector() {
        let mut removed = span("removed", MemoryStatus::Active, None);
        removed.text = "Temporary API key sk-test-old was rotated.".to_string();
        removed.lexical_text = removed.text.clone();
        let kept = span("kept", MemoryStatus::Active, None);
        let hits = vec![
            SpanSearchHit {
                span: removed,
                score: 1.0,
            },
            SpanSearchHit {
                span: kept,
                score: 0.5,
            },
        ];
        let mut tombstone = correction("c1", CorrectionOperation::Tombstone, "unused", 10);
        tombstone.target_ids.clear();
        tombstone.target_selector = Some("SK-TEST-OLD".to_string());

        let filtered = apply_corrections_to_span_hits(&hits, &[tombstone]);

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].span.id, "kept");
    }

    #[test]
    fn corrections_respect_effective_time_for_historical_queries() {
        let removed = span("removed", MemoryStatus::Active, None);
        let hits = vec![SpanSearchHit {
            span: removed,
            score: 1.0,
        }];
        let tombstone = correction("c1", CorrectionOperation::Tombstone, "removed", 20);

        let before =
            apply_corrections_to_span_hits_at(&hits, std::slice::from_ref(&tombstone), Some(10));
        let after = apply_corrections_to_span_hits_at(&hits, &[tombstone], Some(21));

        assert_eq!(before.len(), 1);
        assert!(after.is_empty());
    }

    fn correction(
        id: &str,
        operation: CorrectionOperation,
        target_id: &str,
        effective_at_ms: i64,
    ) -> CorrectionRecord {
        CorrectionRecord {
            id: id.to_string(),
            scope: MemoryScope::new("default"),
            status: MemoryStatus::Active,
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            operation,
            target_type: "span".to_string(),
            target_match: crate::CorrectionTargetMatch::ClaimVersions,
            target_ids: vec![target_id.to_string()],
            target_selector: None,
            target_subject_key: None,
            target_predicate_key: None,
            target_subject_entity_id: None,
            target_slot_id: None,
            new_value: None,
            reason: None,
            actor: ActorKind::User,
            authority: CorrectionAuthority::User,
            created_at_ms: effective_at_ms,
            effective_at_ms,
            applies_valid_from_ms: None,
            applies_valid_to_ms: None,
            cascade_policy: None,
            source_span_ids: Vec::new(),
            source_episode_ids: Vec::new(),
            source_sequence_no: None,
            metadata_json: None,
        }
    }
}

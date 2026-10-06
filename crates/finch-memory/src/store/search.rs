use super::*;

use crate::source_index::{SourceLexicalHit, SourceLexicalIndex};
use crate::CurrentStateSlotRankings;

/// Options for hybrid dense-plus-lexical source-diverse retrieval.
#[derive(Debug, Clone, Copy)]
pub struct HybridSourceDiverseOptions {
    /// Dense span candidates fetched from the vector index.
    pub candidate_k: usize,
    /// Source candidates fetched from the lexical index.
    pub lexical_source_k: usize,
    /// Maximum distinct sources in the fused output.
    pub max_sources: usize,
    /// Maximum spans contributed by any one source.
    pub max_spans_per_source: usize,
    /// Adjacent spans hydrated around each dense hit.
    pub neighbor_radius: usize,
    /// Lexical terms present in more than this fraction of sources are skipped.
    pub lexical_max_df_ratio: f32,
    /// Weight applied to each expansion-query lexical ranking during fusion.
    pub expansion_weight: f32,
    /// Maximum distinct sources appended by a supplemental retrieval round.
    pub supplemental_sources: usize,
    /// Maximum spans contributed by each supplemental source.
    pub supplemental_spans_per_source: usize,
    /// Dense span candidates fetched for each embedded supplemental query.
    pub supplemental_candidate_k: usize,
}

/// The query side of one hybrid source-diverse retrieval.
pub struct HybridSourceDiverseQuery<'a> {
    pub query_embedding: Vec<f32>,
    pub query_text: &'a str,
    /// Lexical expansion queries fused with the main query.
    pub expansions: &'a [ExpansionQuery],
    /// Queries for the supplemental retrieval round.
    pub supplemental_queries: &'a [ExpansionQuery],
    pub index: &'a SourceLexicalIndex,
    pub at_ms: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceSpanSelection {
    pub source_id: String,
    pub max_spans: usize,
}

#[derive(Debug, Clone)]
pub struct ExpansionQuery {
    pub text: String,
    pub embedding: Option<Vec<f32>>,
}

const RRF_K: f32 = 60.0;

/// A ranking's reciprocal-rank-fusion share at `rank`, scaled by `weight`.
fn rrf_share(weight: f32, rank: usize) -> f32 {
    weight / (RRF_K + rank as f32 + 1.0)
}

/// One source's spans before hydration: its ranked hits, then the span ids to fetch after them.
struct SourcePlan {
    score: f32,
    spans: Vec<SpanSearchHit>,
    fill_ids: Vec<MemoryId>,
}

impl SourcePlan {
    /// The plan's spans followed by the fill spans `hydrate` supplies, scored with the plan.
    fn into_hits(
        self,
        mut hydrate: impl FnMut(&MemoryId) -> Option<SpanRecord>,
    ) -> Vec<SpanSearchHit> {
        let mut hits = self.spans;
        for span_id in &self.fill_ids {
            if let Some(span) = hydrate(span_id) {
                hits.push(SpanSearchHit {
                    span,
                    score: self.score,
                });
            }
        }
        hits
    }
}

struct SupplementalSearch<'a> {
    queries: &'a [ExpansionQuery],
    index: &'a SourceLexicalIndex,
    at_ms: Option<i64>,
    primary_source_ids: &'a BTreeSet<String>,
}

/// A selected source's plan: its prior hits, then neighbors of those hits and its leading spans.
fn selected_source_plan(
    selection: &SourceSpanSelection,
    prior_hits: &[SpanSearchHit],
    index: &SourceLexicalIndex,
) -> SourcePlan {
    let mut seen_prior_ids = BTreeSet::new();
    let spans = prior_hits
        .iter()
        .filter(|hit| hit.span.source_id == selection.source_id)
        .filter(|hit| seen_prior_ids.insert(hit.span.id.as_str()))
        .take(selection.max_spans)
        .cloned()
        .collect::<Vec<_>>();
    let score = spans.first().map_or(0.0, |hit| hit.score);
    let fill_ids = index
        .span_ids_for_source(&selection.source_id)
        .map(|span_ids| neighbor_fill_ids(&spans, span_ids, 1, selection.max_spans))
        .unwrap_or_default();
    SourcePlan {
        score,
        spans,
        fill_ids,
    }
}

/// A primary source's plan: its dense hits, then their neighbors and its leading spans.
fn primary_source_plan(
    source_id: String,
    score: f32,
    mut spans: Vec<SpanSearchHit>,
    index: &SourceLexicalIndex,
    options: &HybridSourceDiverseOptions,
) -> SourcePlan {
    let span_cap = options.max_spans_per_source;
    spans.truncate(span_cap);
    let fill_ids = index
        .span_ids_for_source(&source_id)
        .map(|span_ids| neighbor_fill_ids(&spans, span_ids, options.neighbor_radius, span_cap))
        .unwrap_or_default();
    SourcePlan {
        score,
        spans,
        fill_ids,
    }
}

/// Span ids that top `spans` up to `span_cap`: neighbors within `radius` of each hit, then the
/// source's leading spans, skipping spans already present.
fn neighbor_fill_ids(
    spans: &[SpanSearchHit],
    span_ids: &[String],
    radius: usize,
    span_cap: usize,
) -> Vec<MemoryId> {
    let present = spans
        .iter()
        .map(|hit| hit.span.id.as_str())
        .collect::<BTreeSet<_>>();
    let neighbors = spans.iter().flat_map(|hit| {
        let position = hit.span.span_index.max(0) as usize;
        (1..=radius).flat_map(move |delta| {
            [
                position.saturating_add(delta),
                position.saturating_sub(delta),
            ]
        })
    });
    let mut seen = BTreeSet::new();
    neighbors
        .chain(0..span_ids.len().min(span_cap))
        .filter_map(|position| span_ids.get(position))
        .filter(|span_id| !present.contains(span_id.as_str()) && seen.insert(span_id.as_str()))
        .take(span_cap.saturating_sub(spans.len()))
        .cloned()
        .collect()
}

/// The source's leading span ids not already in `spans`, up to `span_cap` spans in total.
fn leading_fill_ids(
    spans: &[SpanSearchHit],
    span_ids: &[String],
    span_cap: usize,
) -> Vec<MemoryId> {
    let present = spans
        .iter()
        .map(|hit| hit.span.id.as_str())
        .collect::<BTreeSet<_>>();
    span_ids
        .iter()
        .filter(|span_id| !present.contains(span_id.as_str()))
        .take(span_cap.saturating_sub(spans.len()))
        .cloned()
        .collect()
}

/// Dense hits grouped by source, with the sources in first-hit order.
fn group_hits_by_source(
    hits: Vec<SpanSearchHit>,
) -> (BTreeMap<String, Vec<SpanSearchHit>>, Vec<String>) {
    let mut by_source = BTreeMap::<String, Vec<SpanSearchHit>>::new();
    let mut source_order = Vec::new();
    for hit in hits {
        let source_hits = by_source.entry(hit.span.source_id.clone()).or_default();
        if source_hits.is_empty() {
            source_order.push(hit.span.source_id.clone());
        }
        source_hits.push(hit);
    }
    (by_source, source_order)
}

/// Primary sources ranked by reciprocal-rank fusion of the dense source order, the lexical
/// ranking, and each weighted expansion ranking; best first, at most `max_sources`.
fn fuse_primary_sources(
    dense_source_order: &[String],
    lexical_hits: &[SourceLexicalHit],
    expansions: &[ExpansionQuery],
    index: &SourceLexicalIndex,
    options: &HybridSourceDiverseOptions,
) -> ZResult<Vec<(String, f32)>> {
    let mut fused = BTreeMap::<String, f32>::new();
    for (rank, source_id) in dense_source_order.iter().enumerate() {
        *fused.entry(source_id.clone()).or_insert(0.0) += rrf_share(1.0, rank);
    }
    for (rank, hit) in lexical_hits.iter().enumerate() {
        *fused.entry(hit.source_id.clone()).or_insert(0.0) += rrf_share(1.0, rank);
    }
    for expansion in expansions {
        if expansion.text.trim().is_empty() {
            continue;
        }
        let expansion_hits = index.search(
            &expansion.text,
            options.lexical_source_k,
            options.lexical_max_df_ratio,
        )?;
        for (rank, hit) in expansion_hits.iter().enumerate() {
            *fused.entry(hit.source_id.clone()).or_insert(0.0) +=
                rrf_share(options.expansion_weight, rank);
        }
    }
    let mut fused = fused.into_iter().collect::<Vec<_>>();
    sort_fused_sources(&mut fused);
    fused.truncate(options.max_sources);
    Ok(fused)
}

fn sort_fused_sources(fused: &mut [(String, f32)]) {
    fused.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
}

/// Adds one supplemental dense ranking: its source order joins the fusion and each source keeps
/// its distinct hits.
fn fuse_supplemental_dense_hits(
    hits: Vec<SpanSearchHit>,
    fused: &mut BTreeMap<String, f32>,
    dense_by_source: &mut BTreeMap<String, Vec<SpanSearchHit>>,
) {
    let mut source_order = Vec::new();
    let mut seen_sources = BTreeSet::new();
    for hit in hits {
        let source_id = hit.span.source_id.clone();
        if seen_sources.insert(source_id.clone()) {
            source_order.push(source_id.clone());
        }
        let source_hits = dense_by_source.entry(source_id).or_default();
        if !source_hits
            .iter()
            .any(|existing| existing.span.id == hit.span.id)
        {
            source_hits.push(hit);
        }
    }
    for (rank, source_id) in source_order.into_iter().enumerate() {
        *fused.entry(source_id).or_insert(0.0) += rrf_share(1.0, rank);
    }
}

/// Round-robin over the sources by rank: every source's first span, then every second span, ...
fn interleave_by_rank(per_source: Vec<Vec<SpanSearchHit>>, max_rank: usize) -> Vec<SpanSearchHit> {
    let mut sources = per_source
        .into_iter()
        .map(Vec::into_iter)
        .collect::<Vec<_>>();
    let mut output = Vec::new();
    for _ in 0..max_rank {
        output.extend(sources.iter_mut().filter_map(Iterator::next));
    }
    output
}

impl MemoryStore {
    pub(super) fn index_span_terms(&self, span: &SpanRecord) -> ZResult<()> {
        insert_many(&self.terms, term_docs_for_span(span))
    }

    pub fn ingest_episode(
        &self,
        input: EpisodeInput,
        ingested_at_ms: i64,
        chunk_options: &ChunkOptions,
    ) -> ZResult<IngestedEpisode> {
        let ingested = ingest_episode(input, ingested_at_ms, chunk_options);
        self.append_episode(&ingested.episode)?;
        for span in &ingested.spans {
            self.append_span(span, None)?;
        }
        Ok(ingested)
    }

    pub fn query_spans(
        &self,
        scope: &MemoryScope,
        query_embedding: Vec<f32>,
        k: usize,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<SpanSearchHit>> {
        let _state_guard = self.lock_state_read()?;
        let fetch_k = expanded_span_vector_fetch_k(k);
        let mut hits = self.query_span_candidates_with_corrections(
            scope,
            query_embedding,
            fetch_k,
            fetch_k,
            at_ms,
            &[],
        )?;
        let corrections = self.scan_span_corrections_for_hits(scope, &hits, at_ms)?;
        hits = apply_corrections_to_span_hits_at(&hits, &corrections, at_ms);
        hits.truncate(k);
        Ok(hits)
    }

    pub fn query_source_diverse_spans(
        &self,
        scope: &MemoryScope,
        query_embedding: Vec<f32>,
        candidate_k: usize,
        max_sources: usize,
        max_spans_per_source: usize,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<SpanSearchHit>> {
        let _state_guard = self.lock_state_read()?;
        if candidate_k == 0 || max_sources == 0 || max_spans_per_source == 0 {
            return Ok(Vec::new());
        }
        let candidates = self.query_spans(scope, query_embedding, candidate_k, at_ms)?;
        Ok(source_diverse_span_hits(
            candidates,
            max_sources,
            max_spans_per_source,
        ))
    }

    /// Select matching passages within already ranked sources, retaining dense
    /// matches and filling remaining passage slots with surrounding evidence.
    pub fn search_selected_sources(
        &self,
        scope: &MemoryScope,
        query_text: &str,
        selections: &[SourceSpanSelection],
        dense_hits: &[SpanSearchHit],
        index: &SourceLexicalIndex,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<SpanSearchHit>> {
        let _state_guard = self.lock_state_read()?;
        let span_ids = selections
            .iter()
            .filter(|selection| selection.max_spans > 0)
            .filter_map(|selection| index.span_ids_for_source(&selection.source_id))
            .flatten()
            .cloned()
            .collect::<Vec<_>>();
        let spans = self.fetch_uncorrected_spans_by_ids(scope, &span_ids, at_ms)?;
        let allowed_ids = spans
            .iter()
            .map(|span| span.id.as_str())
            .collect::<BTreeSet<_>>();
        let dense_hits = dense_hits
            .iter()
            .filter(|hit| allowed_ids.contains(hit.span.id.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        let lexical_hits = crate::retrieval::search_spans_keyword(
            query_text,
            &spans,
            crate::retrieval::SpanSearchOptions {
                k: spans.len(),
                scope: Some(scope.clone()),
                at_ms,
                ..Default::default()
            },
        );
        let ranked = hybrid_fuse_rrf(&dense_hits, &lexical_hits, spans.len());
        self.hydrate_selected_sources(scope, selections, &ranked, index, at_ms)
    }

    pub fn hydrate_selected_sources(
        &self,
        scope: &MemoryScope,
        selections: &[SourceSpanSelection],
        prior_hits: &[SpanSearchHit],
        index: &SourceLexicalIndex,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<SpanSearchHit>> {
        let plans = selections
            .iter()
            .map(|selection| selected_source_plan(selection, prior_hits, index))
            .collect::<Vec<_>>();
        let fill_span_ids = plans
            .iter()
            .flat_map(|plan| plan.fill_ids.iter().cloned())
            .collect::<Vec<_>>();
        let hydrated_by_id = self
            .fetch_uncorrected_spans_by_ids(scope, &fill_span_ids, at_ms)?
            .into_iter()
            .map(|span| (span.id.clone(), span))
            .collect::<BTreeMap<_, _>>();
        let per_source = plans
            .into_iter()
            .map(|plan| plan.into_hits(|span_id| hydrated_by_id.get(span_id).cloned()))
            .collect::<Vec<_>>();
        let max_spans = per_source.iter().map(Vec::len).max().unwrap_or(0);
        Ok(interleave_by_rank(per_source, max_spans))
    }

    /// Hybrid company-knowledge retrieval: dense span candidates and
    /// source-level lexical candidates are fused at source granularity with
    /// reciprocal-rank fusion, then each selected source is hydrated into
    /// spans through the index's source-to-span registry (dense hits first,
    /// then their neighbors, then the source's leading spans).
    pub fn query_hybrid_source_diverse_spans(
        &self,
        scope: &MemoryScope,
        query: HybridSourceDiverseQuery<'_>,
        options: HybridSourceDiverseOptions,
    ) -> ZResult<Vec<SpanSearchHit>> {
        let _state_guard = self.lock_state_read()?;
        let HybridSourceDiverseQuery {
            query_embedding,
            query_text,
            expansions,
            supplemental_queries,
            index,
            at_ms,
        } = query;
        if options.candidate_k == 0 || options.max_sources == 0 || options.max_spans_per_source == 0
        {
            return Ok(Vec::new());
        }
        let dense_hits = self.query_spans(scope, query_embedding, options.candidate_k, at_ms)?;
        let lexical_hits = index.search(
            query_text,
            options.lexical_source_k,
            options.lexical_max_df_ratio,
        )?;
        let (mut dense_by_source, dense_source_order) = group_hits_by_source(dense_hits);
        let fused = fuse_primary_sources(
            &dense_source_order,
            &lexical_hits,
            expansions,
            index,
            &options,
        )?;
        let primary_source_ids = fused
            .iter()
            .map(|(source_id, _)| source_id.clone())
            .collect::<BTreeSet<_>>();
        let mut plans = fused
            .into_iter()
            .map(|(source_id, score)| {
                let spans = dense_by_source.remove(&source_id).unwrap_or_default();
                primary_source_plan(source_id, score, spans, index, &options)
            })
            .collect::<Vec<_>>();
        let supplemental = !supplemental_queries.is_empty()
            && options.supplemental_sources > 0
            && options.supplemental_spans_per_source > 0;
        if supplemental {
            let search = SupplementalSearch {
                queries: supplemental_queries,
                index,
                at_ms,
                primary_source_ids: &primary_source_ids,
            };
            plans.extend(self.supplemental_source_plans(scope, &search, &options)?);
        }
        let per_source = self.hydrate_source_plans(scope, plans, at_ms)?;
        let max_span_cap = options.max_spans_per_source.max(
            if supplemental_queries.is_empty() || options.supplemental_sources == 0 {
                0
            } else {
                options.supplemental_spans_per_source
            },
        );
        Ok(interleave_by_rank(per_source, max_span_cap))
    }

    /// Sources the supplemental queries rank outside the primary sources, each planned with its
    /// dense hits and then its leading spans.
    fn supplemental_source_plans(
        &self,
        scope: &MemoryScope,
        search: &SupplementalSearch<'_>,
        options: &HybridSourceDiverseOptions,
    ) -> ZResult<Vec<SourcePlan>> {
        let mut fused = BTreeMap::<String, f32>::new();
        let mut dense_by_source = BTreeMap::<String, Vec<SpanSearchHit>>::new();
        for query in search.queries {
            if query.text.trim().is_empty() {
                continue;
            }
            let lexical_hits = search.index.search(
                &query.text,
                options.lexical_source_k,
                options.lexical_max_df_ratio,
            )?;
            for (rank, hit) in lexical_hits.iter().enumerate() {
                *fused.entry(hit.source_id.clone()).or_insert(0.0) += rrf_share(1.0, rank);
            }
            let Some(embedding) = query
                .embedding
                .as_ref()
                .filter(|_| options.supplemental_candidate_k > 0)
            else {
                continue;
            };
            let dense_hits = self.query_spans(
                scope,
                embedding.clone(),
                options.supplemental_candidate_k,
                search.at_ms,
            )?;
            fuse_supplemental_dense_hits(dense_hits, &mut fused, &mut dense_by_source);
        }
        let mut fused = fused
            .into_iter()
            .filter(|(source_id, _)| !search.primary_source_ids.contains(source_id))
            .collect::<Vec<_>>();
        sort_fused_sources(&mut fused);
        fused.truncate(options.supplemental_sources);
        let span_cap = options.supplemental_spans_per_source;
        Ok(fused
            .into_iter()
            .map(|(source_id, score)| {
                let mut spans = dense_by_source.remove(&source_id).unwrap_or_default();
                spans.truncate(span_cap);
                let fill_ids = search
                    .index
                    .span_ids_for_source(&source_id)
                    .map(|span_ids| leading_fill_ids(&spans, span_ids, span_cap))
                    .unwrap_or_default();
                SourcePlan {
                    score,
                    spans,
                    fill_ids,
                }
            })
            .collect())
    }

    /// Each plan's spans followed by its fetched fill spans (scored with the source's fused
    /// score). A fill span is placed once, in the first plan that asks for it.
    fn hydrate_source_plans(
        &self,
        scope: &MemoryScope,
        plans: Vec<SourcePlan>,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<Vec<SpanSearchHit>>> {
        let fill_span_ids = plans
            .iter()
            .flat_map(|plan| plan.fill_ids.iter().cloned())
            .collect::<Vec<_>>();
        let mut hydrated_by_id = self
            .fetch_uncorrected_spans_by_ids(scope, &fill_span_ids, at_ms)?
            .into_iter()
            .map(|span| (span.id.clone(), span))
            .collect::<BTreeMap<_, _>>();
        Ok(plans
            .into_iter()
            .map(|plan| plan.into_hits(|span_id| hydrated_by_id.remove(span_id)))
            .collect())
    }

    pub fn query_claim_slot_ids(
        &self,
        scope: &MemoryScope,
        query_embedding: Vec<f32>,
        k: usize,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<MemoryId>> {
        let _state_guard = self.lock_state_read()?;
        self.ranked_claim_vector_slot_ids(scope, query_embedding, k, at_ms)
    }

    pub fn query_current_state_slot_ids(
        &self,
        scope: &MemoryScope,
        query_embedding: Vec<f32>,
        query_text: &str,
        k: usize,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<MemoryId>> {
        let _state_guard = self.lock_state_read()?;
        Ok(self
            .query_current_state_slot_rankings(scope, query_embedding, query_text, k, at_ms)?
            .fused)
    }

    /// Per-modality slot rankings plus their fusion. The separate lists let a caller apply an
    /// agreement gate (both modalities naming the same top slot) before spending a model call on
    /// slot selection.
    pub fn query_current_state_slot_rankings(
        &self,
        scope: &MemoryScope,
        query_embedding: Vec<f32>,
        query_text: &str,
        k: usize,
        at_ms: Option<i64>,
    ) -> ZResult<CurrentStateSlotRankings> {
        let _state_guard = self.lock_state_read()?;
        let vector = self.ranked_claim_vector_slot_ids(scope, query_embedding, k, at_ms)?;
        let lexical = self.ranked_current_state_lexical_slot_ids(scope, query_text, k, at_ms)?;
        let fused = if lexical.is_empty() {
            vector.clone()
        } else {
            fuse_ranked_slot_ids(&vector, &lexical, k)
        };
        Ok(CurrentStateSlotRankings {
            vector,
            lexical,
            fused,
        })
    }

    fn ranked_claim_vector_slot_ids(
        &self,
        scope: &MemoryScope,
        query_embedding: Vec<f32>,
        k: usize,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<MemoryId>> {
        if query_embedding.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        let valid_from_clause = at_ms.map_or_else(String::new, |at| {
            format!(" AND (valid_from_ms IS NULL OR valid_from_ms <= {at})")
        });
        let query = VectorQuery::new(
            "embedding",
            query_embedding,
            expanded_claim_slot_vector_fetch_k(k),
        )
        .with_filter(format!("{}{valid_from_clause}", scope_filter(scope)))
        .with_output_fields(output_fields(CLAIM_OUTPUT_FIELDS));
        let hits = self
            .claims
            .query(query)?
            .into_iter()
            .map(|doc| claim_from_doc(&doc))
            .collect::<ZResult<Vec<_>>>()?
            .into_iter()
            .filter(|claim| claim.scope.matches_filter(scope))
            .filter(|claim| {
                at_ms.is_none_or(|at| claim.valid_from_ms.is_none_or(|from| from <= at))
            })
            .collect::<Vec<_>>();
        let corrections = self.scan_claim_corrections_for_claims(scope, &hits, at_ms)?;
        let hits = apply_corrections_to_claims(hits, &corrections, at_ms);
        let mut seen = BTreeSet::new();
        Ok(hits
            .into_iter()
            .filter_map(|claim| claim.slot_id)
            .filter(|slot_id| seen.insert(slot_id.clone()))
            .take(k)
            .collect())
    }

    fn ranked_current_state_lexical_slot_ids(
        &self,
        scope: &MemoryScope,
        query_text: &str,
        k: usize,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<MemoryId>> {
        let normalized_query = canonical_slot_part(query_text);
        if normalized_query.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        let fetch_k = expanded_claim_slot_vector_fetch_k(k);
        let temporal = BiTemporalQuery {
            valid_at_ms: at_ms,
            transaction_at_ms: None,
        };
        // Only records sharing a usable term with the query can score by term overlap.
        let (terms, records) = self.scan_state_records_sharing_rare_terms(
            scope,
            &lexical_query_terms(query_text),
            temporal,
        )?;
        let mut scored = records
            .into_iter()
            .filter_map(|record| {
                let slot_id = record.slot_id.clone()?;
                let score =
                    current_state_query_overlap(query_text, &terms, &state_lexical_fields(&record));
                (score > 0).then_some((score, slot_id, record.id))
            })
            .collect::<Vec<_>>();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.2.cmp(&b.2)));
        let mut seen = BTreeSet::new();
        Ok(scored
            .into_iter()
            .filter_map(|(_, slot_id, _)| seen.insert(slot_id.clone()).then_some(slot_id))
            .take(fetch_k)
            .collect())
    }

    pub fn query_spans_with_corrections(
        &self,
        scope: &MemoryScope,
        query_embedding: Vec<f32>,
        k: usize,
        at_ms: Option<i64>,
        corrections: &[CorrectionRecord],
    ) -> ZResult<Vec<SpanSearchHit>> {
        let _state_guard = self.lock_state_read()?;
        let fetch_k = expanded_span_vector_fetch_k(k);
        self.query_span_candidates_with_corrections(
            scope,
            query_embedding,
            fetch_k,
            k,
            at_ms,
            corrections,
        )
    }

    fn query_span_candidates_with_corrections(
        &self,
        scope: &MemoryScope,
        query_embedding: Vec<f32>,
        fetch_k: usize,
        output_k: usize,
        at_ms: Option<i64>,
        corrections: &[CorrectionRecord],
    ) -> ZResult<Vec<SpanSearchHit>> {
        if query_embedding.is_empty() || fetch_k == 0 || output_k == 0 {
            return Ok(Vec::new());
        }
        let filter = format!(
            "{} AND status = 'active'{}",
            scope_filter(scope),
            interval_filter_clause(at_ms)
        );
        let query = VectorQuery::new("embedding", query_embedding, fetch_k)
            .with_filter(filter)
            .with_output_fields(output_fields(SPAN_OUTPUT_FIELDS));
        let mut hits = self
            .spans
            .query(query)?
            .into_iter()
            .map(|doc| {
                let span = span_from_doc(&doc)?;
                Ok(SpanSearchHit {
                    span,
                    score: doc.score,
                })
            })
            .collect::<ZResult<Vec<_>>>()?;
        hits.retain(|hit| hit.span.scope.matches_filter(scope) && span_active_at(&hit.span, at_ms));
        hits = apply_corrections_to_span_hits_at(&hits, corrections, at_ms);
        hits.truncate(output_k);
        Ok(hits)
    }

    pub fn scan_spans(
        &self,
        scope: &MemoryScope,
        limit: usize,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<SpanRecord>> {
        self.scan_spans_with_filter(scope, limit, at_ms, None)
    }

    fn scan_spans_with_filter(
        &self,
        scope: &MemoryScope,
        limit: usize,
        at_ms: Option<i64>,
        extra_filter: Option<String>,
    ) -> ZResult<Vec<SpanRecord>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let filter = extra_filter
            .map(|extra| format!("{} AND {extra}", scope_filter(scope)))
            .unwrap_or_else(|| scope_filter(scope));
        let query = VectorQuery::new("", Vec::new(), limit)
            .with_filter(filter)
            .with_output_fields(output_fields(SPAN_OUTPUT_FIELDS));
        let mut spans = self
            .spans
            .scan_filter_only(query)?
            .into_iter()
            .map(|doc| span_from_doc(&doc))
            .collect::<ZResult<Vec<_>>>()?;
        spans.retain(|span| span.scope.matches_filter(scope) && span_active_at(span, at_ms));
        spans.truncate(limit);
        Ok(spans)
    }

    pub fn complete_span_evidence(
        &self,
        scope: &MemoryScope,
        hits: &[SpanSearchHit],
        before: usize,
        after: usize,
        scan_limit: usize,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<SpanSearchHit>> {
        let _state_guard = self.lock_state_read()?;
        if hits.is_empty() || (before == 0 && after == 0) {
            return Ok(hits.to_vec());
        }
        // Only the hits' sources are read, so the scan limit is not spent on unrelated spans.
        let source_ids = sql_string_list(hits.iter().map(|hit| hit.span.source_id.as_str()));
        let source_filter = format!("source_id IN ({source_ids})");
        let mut spans_by_source = BTreeMap::<String, Vec<SpanRecord>>::new();
        for span in self.scan_spans_with_filter(scope, scan_limit, at_ms, Some(source_filter))? {
            spans_by_source
                .entry(span.source_id.clone())
                .or_default()
                .push(span);
        }
        for hit in hits {
            spans_by_source
                .entry(hit.span.source_id.clone())
                .or_default()
                .push(hit.span.clone());
        }
        for spans in spans_by_source.values_mut() {
            spans.sort_by(|a, b| {
                a.span_index
                    .cmp(&b.span_index)
                    .then_with(|| a.id.cmp(&b.id))
            });
            spans.dedup_by(|a, b| a.id == b.id);
        }

        let mut out = Vec::new();
        let mut seen = BTreeSet::new();
        for hit in hits {
            let Some(spans) = spans_by_source.get(&hit.span.source_id) else {
                continue;
            };
            let start = hit.span.span_index.saturating_sub(before as i64);
            let end = hit.span.span_index.saturating_add(after as i64);
            let neighborhood = spans
                .iter()
                .filter(|span| (start..=end).contains(&span.span_index));
            for span in neighborhood {
                if !seen.insert(span.id.clone()) {
                    continue;
                }
                let score = neighbor_score(hit, span);
                out.push(SpanSearchHit {
                    span: span.clone(),
                    score,
                });
            }
        }
        let corrections = self.scan_span_corrections_for_hits(scope, &out, at_ms)?;
        Ok(apply_corrections_to_span_hits_at(&out, &corrections, at_ms))
    }

    pub fn scan_corrections(
        &self,
        scope: &MemoryScope,
        limit: usize,
    ) -> ZResult<Vec<CorrectionRecord>> {
        let _state_guard = self.lock_state_read()?;
        self.scan_corrections_with_filter(scope, limit, None)
    }

    pub(super) fn scan_claim_corrections_for_claims(
        &self,
        scope: &MemoryScope,
        claims: &[ClaimRecord],
        at_ms: Option<i64>,
    ) -> ZResult<Vec<CorrectionRecord>> {
        let mut ids = claims
            .iter()
            .map(|claim| claim.id.as_str())
            .collect::<Vec<_>>();
        ids.sort_unstable();
        ids.dedup();
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let active_filter = active_correction_filter(at_ms);
        let mut corrections = BTreeMap::<String, CorrectionRecord>::new();
        for chunk in ids.chunks(MAX_CONTAINS_FILTER_VALUES) {
            let target_filter = format!(
                "target_ids contain_any ({})",
                sql_string_list(chunk.iter().copied())
            );
            let query = claim_correction_query(scope, &active_filter, &target_filter);
            for doc in self.corrections.scan_filter_only(query)? {
                insert_scoped_correction(&mut corrections, scope, correction_from_doc(&doc)?);
            }
        }
        let mut slot_ids = claims
            .iter()
            .filter_map(|claim| claim.slot_id.as_deref())
            .collect::<Vec<_>>();
        slot_ids.sort_unstable();
        slot_ids.dedup();
        for chunk in slot_ids.chunks(MAX_CONTAINS_FILTER_VALUES) {
            let target_filter = format!(
                "({})",
                sql_or_eq_list("target_slot_id", chunk.iter().copied())
            );
            let query = claim_correction_query(scope, &active_filter, &target_filter);
            for doc in self.corrections.scan_filter_only(query)? {
                insert_scoped_correction(&mut corrections, scope, correction_from_doc(&doc)?);
            }
        }
        let selector_filter =
            format!("{active_filter} AND target_type = 'claim' AND target_selector IS NOT NULL");
        for correction in
            self.scan_corrections_with_filter(scope, usize::MAX, Some(selector_filter))?
        {
            if correction.target_type == "claim" && correction.target_selector.is_some() {
                corrections.insert(correction.id.clone(), correction);
            }
        }
        let mut corrections = corrections.into_values().collect::<Vec<_>>();
        sort_corrections_by_effect(&mut corrections);
        Ok(corrections)
    }

    pub fn scan_context_corrections(
        &self,
        scope: &MemoryScope,
        claims: &[ClaimRecord],
        hits: &[SpanSearchHit],
        at_ms: Option<i64>,
    ) -> ZResult<Vec<CorrectionRecord>> {
        let _state_guard = self.lock_state_read()?;
        let mut corrections = BTreeMap::<String, CorrectionRecord>::new();
        for correction in self.scan_claim_corrections_for_claims(scope, claims, at_ms)? {
            corrections.insert(correction.id.clone(), correction);
        }
        for correction in self.scan_span_corrections_for_hits(scope, hits, at_ms)? {
            corrections.insert(correction.id.clone(), correction);
        }
        let mut corrections = corrections.into_values().collect::<Vec<_>>();
        sort_corrections_by_effect(&mut corrections);
        Ok(corrections)
    }

    pub(super) fn scan_corrections_with_filter(
        &self,
        scope: &MemoryScope,
        limit: usize,
        extra_filter: Option<String>,
    ) -> ZResult<Vec<CorrectionRecord>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let filter = extra_filter
            .map(|extra| format!("{} AND {extra}", scope_filter(scope)))
            .unwrap_or_else(|| scope_filter(scope));
        let query = VectorQuery::new("", Vec::new(), limit)
            .with_filter(filter)
            .with_output_fields(output_fields(CORRECTION_OUTPUT_FIELDS));
        self.corrections
            .scan_filter_only(query)?
            .into_iter()
            .map(|doc| correction_from_doc(&doc))
            .filter(|record| {
                record
                    .as_ref()
                    .map_or(true, |correction| correction.scope.matches_filter(scope))
            })
            .take(limit)
            .collect()
    }

    pub fn scan_artifacts(
        &self,
        scope: &MemoryScope,
        limit: usize,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<ArtifactRecord>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let query = VectorQuery::new("", Vec::new(), limit)
            .with_filter(scope_filter(scope))
            .with_output_fields(output_fields(ARTIFACT_OUTPUT_FIELDS));
        let mut artifacts = self
            .artifacts
            .scan_filter_only(query)?
            .into_iter()
            .map(|doc| artifact_from_doc(&doc))
            .collect::<ZResult<Vec<_>>>()?;
        artifacts.retain(|artifact| {
            matches!(artifact.status, MemoryStatus::Active)
                && artifact.scope.matches_filter(scope)
                && interval_holds_at(artifact.valid_from_ms, artifact.valid_to_ms, at_ms)
        });
        artifacts.truncate(limit);
        Ok(artifacts)
    }

    pub fn scan_claims(
        &self,
        scope: &MemoryScope,
        limit: usize,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<ClaimRecord>> {
        let _state_guard = self.lock_state_read()?;
        if limit == 0 {
            return Ok(Vec::new());
        }
        let query = VectorQuery::new("", Vec::new(), limit)
            .with_filter(scope_filter(scope))
            .with_output_fields(output_fields(CLAIM_OUTPUT_FIELDS));
        let mut claims = self
            .claims
            .scan_filter_only(query)?
            .into_iter()
            .map(|doc| claim_from_doc(&doc))
            .collect::<ZResult<Vec<_>>>()?;
        claims.retain(|claim| {
            matches!(claim.status, MemoryStatus::Active)
                && claim.scope.matches_filter(scope)
                && interval_holds_at(claim.valid_from_ms, claim.valid_to_ms, at_ms)
        });
        claims.truncate(limit);
        Ok(claims)
    }

    pub(crate) fn claim_slot_ids_for_evidence(
        &self,
        scope: &MemoryScope,
        source_span_ids: &BTreeSet<&str>,
        source_episode_ids: &BTreeSet<&str>,
    ) -> ZResult<BTreeSet<MemoryId>> {
        let evidence_ids = source_span_ids
            .iter()
            .chain(source_episode_ids)
            .copied()
            .collect::<Vec<_>>();
        if evidence_ids.is_empty() {
            return Ok(BTreeSet::new());
        }
        // Claims store the ids they cite, so only claims citing the evidence are read.
        let docs = scan_in_chunks(
            &self.claims,
            &evidence_ids,
            usize::MAX,
            CLAIM_OUTPUT_FIELDS,
            |chunk| {
                format!(
                    "{} AND evidence_ids contain_any ({})",
                    scope_filter(scope),
                    sql_string_list(chunk.iter().copied())
                )
            },
        )?;
        let mut slot_ids = BTreeSet::new();
        for doc in &docs {
            let claim = claim_from_doc(doc)?;
            if matches!(claim.status, MemoryStatus::Active)
                && claim.scope.matches_filter(scope)
                && claim_cites_any(&claim, source_span_ids, source_episode_ids)
            {
                slot_ids.extend(claim.slot_id);
            }
        }
        Ok(slot_ids)
    }

    pub fn scan_current_claims(
        &self,
        scope: &MemoryScope,
        limit: usize,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<ClaimRecord>> {
        let _state_guard = self.lock_state_read()?;
        if limit == 0 {
            return Ok(Vec::new());
        }
        // Every version must reach newest-wins resolution before the limit applies.
        let claims = self.scan_claims(scope, usize::MAX, at_ms)?;
        let corrections = self.scan_claim_corrections_for_claims(scope, &claims, at_ms)?;
        let claims = apply_corrections_to_claims(claims, &corrections, at_ms);
        Ok(resolve_current_claims(claims, limit))
    }

    pub(crate) fn scan_current_claims_for_slot_ids(
        &self,
        scope: &MemoryScope,
        slot_ids: &BTreeSet<MemoryId>,
        limit: usize,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<ClaimRecord>> {
        let claims = self.scan_claim_versions_for_slot_ids(scope, slot_ids, usize::MAX, at_ms)?;
        let corrections = self.scan_claim_corrections_for_claims(scope, &claims, at_ms)?;
        Ok(resolve_current_claims(
            apply_corrections_to_claims(claims, &corrections, at_ms),
            limit,
        ))
    }

    pub(crate) fn scan_claim_versions_for_slot_ids(
        &self,
        scope: &MemoryScope,
        slot_ids: &BTreeSet<MemoryId>,
        limit: usize,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<ClaimRecord>> {
        if slot_ids.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let values = slot_ids.iter().map(String::as_str).collect::<Vec<_>>();
        let docs = scan_in_chunks(&self.claims, &values, limit, CLAIM_OUTPUT_FIELDS, |chunk| {
            scoped_field_in(scope, "slot_id", chunk)
        })?;
        let by_id = docs
            .iter()
            .map(|doc| claim_from_doc(doc))
            .collect::<ZResult<Vec<_>>>()?
            .into_iter()
            .filter(|claim| {
                let on_slot = claim
                    .slot_id
                    .as_ref()
                    .is_some_and(|id| slot_ids.contains(id));
                on_slot
                    && claim.scope.matches_filter(scope)
                    && interval_holds_at(claim.valid_from_ms, claim.valid_to_ms, at_ms)
            })
            .map(|claim| (claim.id.clone(), claim))
            .collect::<BTreeMap<_, _>>();
        Ok(by_id.into_values().collect())
    }

    pub(crate) fn scan_current_claims_for_subjects(
        &self,
        scope: &MemoryScope,
        subjects: &BTreeSet<(Option<MemoryId>, String)>,
        limit: usize,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<ClaimRecord>> {
        if subjects.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let mut clauses = subjects
            .iter()
            .map(|(entity_id, subject_key)| match entity_id {
                Some(entity_id) => {
                    format!("subject_entity_id = '{}'", sql_escape(entity_id))
                }
                None => format!("subject_key = '{}'", sql_escape(subject_key)),
            })
            .collect::<Vec<_>>();
        clauses.sort_unstable();
        let docs = scan_in_chunks(
            &self.claims,
            &clauses,
            usize::MAX,
            CLAIM_OUTPUT_FIELDS,
            |chunk| format!("{} AND ({})", scope_filter(scope), chunk.join(" OR ")),
        )?;
        let by_id = docs
            .iter()
            .map(|doc| claim_from_doc(doc))
            .collect::<ZResult<Vec<_>>>()?
            .into_iter()
            .filter(|claim| {
                claim.scope.matches_filter(scope)
                    && claim_has_subject(claim, subjects)
                    && interval_holds_at(claim.valid_from_ms, claim.valid_to_ms, at_ms)
            })
            .map(|claim| (claim.id.clone(), claim))
            .collect::<BTreeMap<_, _>>();
        let claims = by_id.into_values().collect::<Vec<_>>();
        let corrections = self.scan_claim_corrections_for_claims(scope, &claims, at_ms)?;
        Ok(resolve_current_claims(
            apply_corrections_to_claims(claims, &corrections, at_ms),
            limit,
        ))
    }

    pub(crate) fn claims_by_ids(
        &self,
        scope: &MemoryScope,
        claim_ids: &[MemoryId],
    ) -> ZResult<Vec<ClaimRecord>> {
        let mut claims = self
            .claims
            .fetch(claim_ids.to_vec())?
            .values()
            .map(|doc| claim_from_doc(doc))
            .collect::<ZResult<Vec<_>>>()?;
        claims.retain(|claim| claim.scope.matches_filter(scope));
        claims.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(claims)
    }

    pub(super) fn current_claims_for_rule_targets<'c>(
        &self,
        scope: &MemoryScope,
        rules: &[&RuleRecord],
        generated_claims: impl Iterator<Item = &'c ClaimRecord>,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<ClaimRecord>> {
        let mut slot_ids = rules
            .iter()
            .filter_map(|rule| rule.target_slot_id.clone())
            .collect::<BTreeSet<_>>();
        let generic_subjects = rules
            .iter()
            .filter(|rule| matches!(rule.target_match, RuleTargetMatch::AnyActiveSlotForSubject))
            .map(|rule| {
                (
                    rule.target_subject_entity_id.clone(),
                    rule.target_subject_key.clone(),
                )
            })
            .collect::<BTreeSet<_>>();
        if !generic_subjects.is_empty() {
            for slot in self.scan_slots(scope, usize::MAX, at_ms)? {
                if generic_subjects.contains(&(slot.subject_entity_id, slot.subject_key)) {
                    slot_ids.insert(slot.slot_key);
                }
            }
        }
        let mut claims =
            self.scan_current_claims_for_slot_ids(scope, &slot_ids, usize::MAX, at_ms)?;
        if !generic_subjects.is_empty() {
            claims.extend(self.scan_current_claims_for_subjects(
                scope,
                &generic_subjects,
                usize::MAX,
                at_ms,
            )?);
        }
        claims.extend(
            generated_claims
                .filter(|claim| rules.iter().any(|rule| rule_targets_claim(rule, claim)))
                .cloned(),
        );
        // Rules fire within one scope, so only that scope's targets can go stale.
        claims.retain(|claim| claim.scope == *scope);
        dedupe_claims_by_id(&mut claims);
        Ok(claims)
    }

    pub(super) fn current_claims_for_rule_triggers(
        &self,
        scope: &MemoryScope,
        rules: &[RuleRecord],
        at_ms: Option<i64>,
    ) -> ZResult<Vec<ClaimRecord>> {
        let slot_ids = rules
            .iter()
            .filter_map(|rule| rule.trigger_slot_id.clone())
            .collect::<BTreeSet<_>>();
        let mut claims =
            self.scan_current_claims_for_slot_ids(scope, &slot_ids, usize::MAX, at_ms)?;
        let resolved_slot_ids = claims
            .iter()
            .map(claim_slot_lifecycle_key)
            .collect::<BTreeSet<_>>();
        if rules.iter().all(|rule| {
            rule.trigger_slot_id
                .as_ref()
                .is_some_and(|slot_id| resolved_slot_ids.contains(slot_id))
                || claims
                    .iter()
                    .any(|claim| rule_trigger_matches_claim(rule, claim))
        }) {
            return Ok(claims);
        }

        let broad_claims = self.scan_current_claims(scope, usize::MAX, at_ms)?;
        claims.extend(broad_claims.into_iter().filter(|claim| {
            rules
                .iter()
                .any(|rule| rule_trigger_matches_claim(rule, claim))
        }));
        dedupe_claims_by_id(&mut claims);
        Ok(claims)
    }

    pub fn scan_profiles(
        &self,
        scope: &MemoryScope,
        limit: usize,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<ProfileRecord>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let filter = format!(
            "{} AND status = 'active'{}",
            scope_filter(scope),
            interval_filter_clause(at_ms)
        );
        let query = VectorQuery::new("", Vec::new(), limit)
            .with_filter(filter)
            .with_output_fields(output_fields(PROFILE_OUTPUT_FIELDS));
        let mut profiles = self
            .profiles
            .query(query)?
            .into_iter()
            .map(|doc| profile_from_doc(&doc))
            .collect::<ZResult<Vec<_>>>()?;
        profiles.retain(|profile| {
            matches!(profile.status, MemoryStatus::Active)
                && profile.scope.matches_filter(scope)
                && interval_holds_at(profile.valid_from_ms, profile.valid_to_ms, at_ms)
        });
        Ok(profiles)
    }
}

/// A neighbor span's score decays with its distance from the hit it completes.
fn neighbor_score(hit: &SpanSearchHit, span: &SpanRecord) -> f32 {
    if span.id == hit.span.id {
        return hit.score;
    }
    let distance = (span.span_index - hit.span.span_index).abs() as f32;
    hit.score / (1.0 + distance)
}

fn claim_correction_query(
    scope: &MemoryScope,
    active_filter: &str,
    target_filter: &str,
) -> VectorQuery {
    VectorQuery::new("", Vec::new(), usize::MAX)
        .with_filter(format!(
            "{} AND {active_filter} AND target_type = 'claim' AND {target_filter}",
            scope_filter(scope)
        ))
        .with_output_fields(output_fields(CORRECTION_OUTPUT_FIELDS))
}

/// Whether the claim cites one of the spans or episodes.
fn claim_cites_any(
    claim: &ClaimRecord,
    span_ids: &BTreeSet<&str>,
    episode_ids: &BTreeSet<&str>,
) -> bool {
    claim
        .source_span_ids
        .iter()
        .any(|span_id| span_ids.contains(span_id.as_str()))
        || claim
            .source_episode_ids
            .iter()
            .any(|episode_id| episode_ids.contains(episode_id.as_str()))
}

/// Whether the claim is about one of the subjects: by entity id when the subject has one,
/// otherwise by canonical subject key.
fn claim_has_subject(claim: &ClaimRecord, subjects: &BTreeSet<(Option<MemoryId>, String)>) -> bool {
    let subject_key = canonical_slot_part(claim.subject.as_deref().unwrap_or_default());
    subjects.iter().any(|(entity_id, key)| match entity_id {
        Some(_) => *entity_id == claim.subject_entity_id,
        None => subject_key == *key,
    })
}

/// Whether a generated claim lands on the rule's target: its target slot, its exact surface,
/// or (subject-wide) any slot of its subject.
fn rule_targets_claim(rule: &RuleRecord, claim: &ClaimRecord) -> bool {
    if rule.target_slot_id.as_ref() == claim.slot_id.as_ref() {
        return true;
    }
    let subject_key = canonical_slot_part(claim.subject.as_deref().unwrap_or_default());
    match rule.target_match {
        RuleTargetMatch::ExactSlot => {
            subject_key == rule.target_subject_key
                && canonical_slot_part(claim.predicate.as_deref().unwrap_or_default())
                    == rule.target_predicate_key
        }
        RuleTargetMatch::AnyActiveSlotForSubject => match rule.target_subject_entity_id {
            Some(_) => rule.target_subject_entity_id == claim.subject_entity_id,
            None => subject_key == rule.target_subject_key,
        },
    }
}

fn fuse_ranked_slot_ids(vector: &[MemoryId], lexical: &[MemoryId], k: usize) -> Vec<MemoryId> {
    const RRF_K: f32 = 60.0;
    if k == 0 {
        return Vec::new();
    }
    let mut scores = BTreeMap::<MemoryId, f32>::new();
    for (rank, slot_id) in vector.iter().enumerate() {
        *scores.entry(slot_id.clone()).or_default() += 1.0 / (RRF_K + rank as f32 + 1.0);
    }
    for (rank, slot_id) in lexical.iter().enumerate() {
        *scores.entry(slot_id.clone()).or_default() += 1.0 / (RRF_K + rank as f32 + 1.0);
    }
    let mut ranked = scores.into_iter().collect::<Vec<_>>();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    ranked
        .into_iter()
        .map(|(slot_id, _)| slot_id)
        .take(k)
        .collect()
}

fn source_diverse_span_hits(
    hits: Vec<SpanSearchHit>,
    max_sources: usize,
    max_spans_per_source: usize,
) -> Vec<SpanSearchHit> {
    let mut source_order = Vec::new();
    let mut by_source = BTreeMap::<String, Vec<SpanSearchHit>>::new();
    for hit in hits {
        if !by_source.contains_key(&hit.span.source_id) {
            if source_order.len() >= max_sources {
                continue;
            }
            source_order.push(hit.span.source_id.clone());
        }
        let source_hits = by_source.entry(hit.span.source_id.clone()).or_default();
        if source_hits.len() < max_spans_per_source {
            source_hits.push(hit);
        }
    }

    let mut output = Vec::new();
    for rank_within_source in 0..max_spans_per_source {
        for source_id in &source_order {
            if let Some(hit) = by_source
                .get(source_id)
                .and_then(|hits| hits.get(rank_within_source))
            {
                output.push(hit.clone());
            }
        }
    }
    output
}

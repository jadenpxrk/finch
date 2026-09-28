use super::*;

impl MemoryStore {
    pub fn keyword_search_spans(
        &self,
        scope: &MemoryScope,
        query_text: &str,
        k: usize,
        scan_limit: usize,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<SpanSearchHit>> {
        let _state_guard = self.lock_state_read()?;
        let mut hits = self.keyword_search_spans_with_corrections(
            scope,
            query_text,
            k.saturating_mul(8).max(k),
            scan_limit,
            at_ms,
            &[],
        )?;
        let corrections = self.scan_span_corrections_for_hits(scope, &hits, at_ms)?;
        hits = apply_corrections_to_span_hits_at(&hits, &corrections, at_ms);
        hits.truncate(k);
        Ok(hits)
    }

    pub(crate) fn keyword_search_spans_with_corrections(
        &self,
        scope: &MemoryScope,
        query_text: &str,
        k: usize,
        scan_limit: usize,
        at_ms: Option<i64>,
        corrections: &[CorrectionRecord],
    ) -> ZResult<Vec<SpanSearchHit>> {
        if query_text.trim().is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        let hits = self.keyword_candidates_from_terms(scope, query_text, k, scan_limit, at_ms)?;
        Ok(apply_corrections_to_span_hits_at(&hits, corrections, at_ms))
    }

    fn keyword_candidates_from_terms(
        &self,
        scope: &MemoryScope,
        query_text: &str,
        k: usize,
        scan_limit: usize,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<SpanSearchHit>> {
        let query_terms = unique_strings(lexical_terms(query_text));
        if query_terms.is_empty() {
            return Ok(Vec::new());
        }
        let posting_chunks = self.term_postings(scope, &query_terms)?;
        let span_ids = posting_chunks
            .iter()
            .flatten()
            .map(|posting| posting.span_id.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let by_id = self
            .fetch_spans_by_ids(scope, &span_ids, at_ms)?
            .into_iter()
            .map(|span| (span.id.clone(), span))
            .collect::<BTreeMap<_, _>>();
        let postings = posting_chunks
            .into_iter()
            .flat_map(|chunk| live_posting_prefix(chunk, &by_id, scan_limit.max(k)))
            .collect::<Vec<_>>();
        if postings.is_empty() {
            return Ok(Vec::new());
        }
        let scores = bm25_posting_scores(&query_terms, &postings);
        let mut hits = scores
            .into_iter()
            .filter_map(|(span_id, score)| {
                by_id
                    .get(&span_id)
                    .cloned()
                    .map(|span| SpanSearchHit { span, score })
            })
            .collect::<Vec<_>>();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.span.id.cmp(&b.span.id))
        });
        hits.truncate(k);
        Ok(hits)
    }

    /// Every posting of `terms`, one list per filter chunk, in storage order.
    fn term_postings(
        &self,
        scope: &MemoryScope,
        terms: &[String],
    ) -> ZResult<Vec<Vec<TermPosting>>> {
        let mut chunks = Vec::new();
        for chunk in terms.chunks(MAX_CONTAINS_FILTER_VALUES) {
            let filter = format!(
                "{} AND ({})",
                scope_filter(scope),
                sql_or_eq_list("term", chunk.iter().map(String::as_str))
            );
            let query = VectorQuery::new("", Vec::new(), usize::MAX)
                .with_filter(filter)
                .with_output_fields(output_fields(TERM_OUTPUT_FIELDS));
            chunks.push(
                self.terms
                    .scan_filter_only(query)?
                    .iter()
                    .map(|doc| term_posting_from_doc(doc))
                    .collect::<ZResult<Vec<_>>>()?,
            );
        }
        Ok(chunks)
    }

    pub fn fetch_spans_by_ids(
        &self,
        scope: &MemoryScope,
        span_ids: &[String],
        at_ms: Option<i64>,
    ) -> ZResult<Vec<SpanRecord>> {
        self.ensure_not_poisoned()?;
        if span_ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut spans = Vec::with_capacity(span_ids.len());
        for ids in span_ids.chunks(1024) {
            spans.extend(
                self.spans
                    .fetch(ids.to_vec())?
                    .values()
                    .map(|doc| span_from_doc(doc))
                    .collect::<ZResult<Vec<_>>>()?,
            );
        }
        spans.retain(|span| span.scope.matches_filter(scope) && span_active_at(span, at_ms));
        Ok(spans)
    }

    /// Spans by id that are also not hidden by a span correction effective at `at_ms`.
    pub(super) fn fetch_uncorrected_spans_by_ids(
        &self,
        scope: &MemoryScope,
        span_ids: &[String],
        at_ms: Option<i64>,
    ) -> ZResult<Vec<SpanRecord>> {
        let hits = self
            .fetch_spans_by_ids(scope, span_ids, at_ms)?
            .into_iter()
            .map(|span| SpanSearchHit { span, score: 0.0 })
            .collect::<Vec<_>>();
        let corrections = self.scan_span_corrections_for_hits(scope, &hits, at_ms)?;
        Ok(
            apply_corrections_to_span_hits_at(&hits, &corrections, at_ms)
                .into_iter()
                .map(|hit| hit.span)
                .collect(),
        )
    }

    pub fn hybrid_search_spans(
        &self,
        scope: &MemoryScope,
        search: HybridSpanSearch<'_>,
    ) -> ZResult<HybridSearchDebug> {
        let _state_guard = self.lock_state_read()?;
        let HybridSpanSearch {
            query_embedding,
            query_text,
            k,
            scan_limit,
            at_ms,
        } = search;
        if k == 0 {
            return Ok(HybridSearchDebug {
                vector_hits: Vec::new(),
                keyword_hits: Vec::new(),
                fused_hits: Vec::new(),
            });
        }
        let vector_hits = self.query_spans(scope, query_embedding, k, at_ms)?;
        let keyword_hits = self.keyword_search_spans(scope, query_text, k, scan_limit, at_ms)?;
        let fused_hits = hybrid_fuse_rrf(&vector_hits, &keyword_hits, k);
        Ok(HybridSearchDebug {
            vector_hits,
            keyword_hits,
            fused_hits,
        })
    }

    pub(super) fn scan_span_corrections_for_hits(
        &self,
        scope: &MemoryScope,
        hits: &[SpanSearchHit],
        at_ms: Option<i64>,
    ) -> ZResult<Vec<CorrectionRecord>> {
        let mut ids = hits
            .iter()
            .map(|hit| hit.span.id.as_str())
            .collect::<Vec<_>>();
        ids.sort_unstable();
        ids.dedup();
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut corrections = BTreeMap::<String, CorrectionRecord>::new();
        let active_filter = active_correction_filter(at_ms);
        for chunk in ids.chunks(MAX_CONTAINS_FILTER_VALUES) {
            let filter = format!(
                "{} AND {active_filter} AND target_type = 'span' AND target_ids contain_any ({})",
                scope_filter(scope),
                sql_string_list(chunk.iter().copied())
            );
            let query = VectorQuery::new("", Vec::new(), usize::MAX)
                .with_filter(filter)
                .with_output_fields(output_fields(CORRECTION_OUTPUT_FIELDS));
            for doc in self.corrections.scan_filter_only(query)? {
                insert_scoped_correction(&mut corrections, scope, correction_from_doc(&doc)?);
            }
        }
        let selector_filter =
            format!("{active_filter} AND target_type = 'span' AND target_selector IS NOT NULL");
        for correction in
            self.scan_corrections_with_filter(scope, usize::MAX, Some(selector_filter))?
        {
            if correction.target_type == "span" && correction.target_selector.is_some() {
                corrections.insert(correction.id.clone(), correction);
            }
        }
        let mut corrections = corrections.into_values().collect::<Vec<_>>();
        sort_corrections_by_effect(&mut corrections);
        Ok(corrections)
    }
}

/// The shortest prefix of `postings` that holds `limit` postings of `live` spans.
// Postings carry no status or validity; dead ones inside the prefix still feed BM25 as before.
fn live_posting_prefix(
    postings: Vec<TermPosting>,
    live: &BTreeMap<String, SpanRecord>,
    limit: usize,
) -> Vec<TermPosting> {
    let mut remaining = limit;
    postings
        .into_iter()
        .take_while(|posting| {
            if remaining == 0 {
                return false;
            }
            if live.contains_key(&posting.span_id) {
                remaining -= 1;
            }
            true
        })
        .collect()
}

fn term_posting_from_doc(doc: &Doc) -> ZResult<TermPosting> {
    Ok(TermPosting {
        term: required_doc_string(doc, "term")?,
        span_id: required_doc_string(doc, "span_id")?,
        tf: required_doc_i64(doc, "tf")? as f32,
        doc_len: required_doc_i64(doc, "doc_len")? as f32,
    })
}

use crate::{
    add_correction, build_answer_ready_state_context, build_context, build_query_state_context,
    evaluate_retrieval_baseline, evaluate_span_hits_with_admission, lexical_terms, AnswerEmission,
    AnswerReadyStateRequest, AnswerSupportContract, ArtifactRecord, BiTemporalQuery, ChunkOptions,
    ClaimRecord, ContextInput, ContextOptions, CorrectionInput, EdgeInput, EntityInput,
    EpisodeInput, HybridSpanSearch, ManualClaimInput, MemoryScope, MemoryStore, ProfileInput,
    SlotAliasInput, SpanSearchHit, StateMutationBatch, StateReadSelection,
};
use finch_types::{Status, ZResult};
use serde::Deserialize;
use std::collections::HashSet;

impl MemoryStore {
    pub fn validate_answer_emission_json(
        &self,
        scope_json: &str,
        support_json: &str,
        emission_json: &str,
    ) -> ZResult<String> {
        let scope = serde_json::from_str::<MemoryScope>(scope_json).map_err(json_status)?;
        let support =
            serde_json::from_str::<AnswerSupportContract>(support_json).map_err(json_status)?;
        let emission =
            serde_json::from_str::<AnswerEmission>(emission_json).map_err(json_status)?;
        let validated = self.validate_answer_emission(&scope, &support, emission)?;
        serde_json::to_string(&validated).map_err(json_status)
    }

    pub fn validate_answer_context_json(
        &self,
        scope_json: &str,
        context_json: &str,
        emission_json: &str,
    ) -> ZResult<String> {
        let scope = serde_json::from_str::<MemoryScope>(scope_json).map_err(json_status)?;
        let context = serde_json::from_str::<crate::CompiledMemoryContext>(context_json)
            .map_err(json_status)?;
        let emission =
            serde_json::from_str::<AnswerEmission>(emission_json).map_err(json_status)?;
        let validated = self.validate_answer_context(&scope, &context, emission)?;
        serde_json::to_string(&validated).map_err(json_status)
    }

    pub fn finalize_answer_context_json(
        &self,
        scope_json: &str,
        context_json: &str,
        emission_json: &str,
    ) -> ZResult<String> {
        let scope = serde_json::from_str::<MemoryScope>(scope_json).map_err(json_status)?;
        let context = serde_json::from_str::<crate::CompiledMemoryContext>(context_json)
            .map_err(json_status)?;
        let emission =
            serde_json::from_str::<AnswerEmission>(emission_json).map_err(json_status)?;
        let grounded = self.finalize_answer_context(&scope, &context, emission)?;
        serde_json::to_string(&grounded).map_err(json_status)
    }

    pub fn apply_state_mutation_batch_json(&self, batch_json: &str) -> ZResult<String> {
        let batch = serde_json::from_str::<StateMutationBatch>(batch_json).map_err(json_status)?;
        let result = self.apply_state_mutation_batch(batch)?;
        serde_json::to_string(&result).map_err(json_status)
    }

    pub fn project_answer_ready_state_json(&self, request_json: &str) -> ZResult<String> {
        let request = serde_json::from_str::<AnswerReadyStateJsonRequest>(request_json)
            .map_err(json_status)?;
        let projection = self.project_answer_ready_state(
            &request.scope,
            &AnswerReadyStateRequest {
                projection_seed: &request.projection_seed,
                query_text: &request.query_text,
                evidence_hits: &request.evidence_hits,
                selections: &request.selections,
                target_selections: request.target_selections.as_deref(),
                claim_limit: request.claim_limit,
                entity_limit: request.entity_limit,
                temporal: request.temporal,
            },
        )?;
        serde_json::to_string(&projection).map_err(json_status)
    }

    pub fn ingest_episode_json(
        &self,
        input_json: &str,
        ingested_at_ms: i64,
        chunk_options_json: Option<&str>,
    ) -> ZResult<String> {
        let input = serde_json::from_str::<EpisodeInput>(input_json).map_err(json_status)?;
        let options = match chunk_options_json {
            Some(json) => serde_json::from_str::<ChunkOptions>(json).map_err(json_status)?,
            None => ChunkOptions::default(),
        };
        let ingested = self.ingest_episode(input, ingested_at_ms, &options)?;
        serde_json::to_string(&ingested).map_err(json_status)
    }

    pub fn append_artifact_json(&self, record_json: &str) -> ZResult<String> {
        let artifact = serde_json::from_str::<ArtifactRecord>(record_json).map_err(json_status)?;
        self.append_artifact(&artifact)?;
        serde_json::to_string(&artifact).map_err(json_status)
    }

    pub fn ingest_artifact_text_json(
        &self,
        record_json: &str,
        text: &str,
        chunk_options_json: Option<&str>,
    ) -> ZResult<String> {
        let artifact = serde_json::from_str::<ArtifactRecord>(record_json).map_err(json_status)?;
        let options = match chunk_options_json {
            Some(json) => serde_json::from_str::<ChunkOptions>(json).map_err(json_status)?,
            None => ChunkOptions::default(),
        };
        let ingested = self.ingest_artifact_text(artifact, text, &options)?;
        serde_json::to_string(&ingested).map_err(json_status)
    }

    pub fn add_correction_json(&self, input_json: &str, created_at_ms: i64) -> ZResult<String> {
        let input = serde_json::from_str::<CorrectionInput>(input_json).map_err(json_status)?;
        let correction = add_correction(input, created_at_ms);
        self.add_correction(&correction)?;
        serde_json::to_string(&correction).map_err(json_status)
    }

    pub fn add_manual_claim_json(
        &self,
        input_json: &str,
        embedding: Option<Vec<f32>>,
    ) -> ZResult<String> {
        let input = serde_json::from_str::<ManualClaimInput>(input_json).map_err(json_status)?;
        let claim = self.add_manual_claim(input, embedding.as_deref())?;
        serde_json::to_string(&claim).map_err(json_status)
    }

    pub fn add_profile_json(&self, input_json: &str) -> ZResult<String> {
        let input = serde_json::from_str::<ProfileInput>(input_json).map_err(json_status)?;
        let profile = self.add_profile(input)?;
        serde_json::to_string(&profile).map_err(json_status)
    }

    pub fn add_entity_json(
        &self,
        input_json: &str,
        embedding: Option<Vec<f32>>,
    ) -> ZResult<String> {
        let input = serde_json::from_str::<EntityInput>(input_json).map_err(json_status)?;
        let entity = self.add_entity(input, embedding.as_deref())?;
        serde_json::to_string(&entity).map_err(json_status)
    }

    pub fn add_edge_json(&self, input_json: &str) -> ZResult<String> {
        let input = serde_json::from_str::<EdgeInput>(input_json).map_err(json_status)?;
        let edge = self.add_edge(input)?;
        serde_json::to_string(&edge).map_err(json_status)
    }

    pub fn add_slot_alias_json(&self, input_json: &str, recorded_at_ms: i64) -> ZResult<String> {
        let input = serde_json::from_str::<SlotAliasInput>(input_json).map_err(json_status)?;
        let alias = self.add_slot_alias(input, recorded_at_ms)?;
        serde_json::to_string(&alias).map_err(json_status)
    }

    pub fn scan_slot_aliases_json(
        &self,
        scope_json: &str,
        limit: usize,
        at_ms: Option<i64>,
    ) -> ZResult<String> {
        let scope = serde_json::from_str::<MemoryScope>(scope_json).map_err(json_status)?;
        let aliases = self.scan_slot_aliases(&scope, limit, at_ms)?;
        serde_json::to_string(&aliases).map_err(json_status)
    }

    pub fn query_spans_json(
        &self,
        scope_json: &str,
        query_embedding: Vec<f32>,
        k: usize,
        at_ms: Option<i64>,
    ) -> ZResult<String> {
        let scope = serde_json::from_str::<MemoryScope>(scope_json).map_err(json_status)?;
        let hits = self.query_spans(&scope, query_embedding, k, at_ms)?;
        serde_json::to_string(&hits).map_err(json_status)
    }

    pub fn keyword_search_spans_json(
        &self,
        scope_json: &str,
        query_text: &str,
        k: usize,
        scan_limit: usize,
        at_ms: Option<i64>,
    ) -> ZResult<String> {
        let scope = serde_json::from_str::<MemoryScope>(scope_json).map_err(json_status)?;
        let hits = self.keyword_search_spans(&scope, query_text, k, scan_limit, at_ms)?;
        serde_json::to_string(&hits).map_err(json_status)
    }

    pub fn hybrid_search_spans_json(
        &self,
        scope_json: &str,
        query_embedding: Vec<f32>,
        query_text: &str,
        k: usize,
        scan_limit: usize,
        at_ms: Option<i64>,
    ) -> ZResult<String> {
        let scope = serde_json::from_str::<MemoryScope>(scope_json).map_err(json_status)?;
        let hits = self
            .hybrid_search_spans(
                &scope,
                HybridSpanSearch {
                    query_embedding,
                    query_text,
                    k,
                    scan_limit,
                    at_ms,
                },
            )?
            .fused_hits;
        serde_json::to_string(&hits).map_err(json_status)
    }

    pub fn hybrid_search_spans_debug_json(
        &self,
        scope_json: &str,
        query_embedding: Vec<f32>,
        query_text: &str,
        k: usize,
        scan_limit: usize,
        at_ms: Option<i64>,
    ) -> ZResult<String> {
        let scope = serde_json::from_str::<MemoryScope>(scope_json).map_err(json_status)?;
        let debug = self.hybrid_search_spans(
            &scope,
            HybridSpanSearch {
                query_embedding,
                query_text,
                k,
                scan_limit,
                at_ms,
            },
        )?;
        serde_json::to_string(&debug).map_err(json_status)
    }

    pub fn complete_span_evidence_json(
        &self,
        scope_json: &str,
        hits_json: &str,
        before: usize,
        after: usize,
        scan_limit: usize,
        at_ms: Option<i64>,
    ) -> ZResult<String> {
        let scope = serde_json::from_str::<MemoryScope>(scope_json).map_err(json_status)?;
        let hits = serde_json::from_str::<Vec<SpanSearchHit>>(hits_json).map_err(json_status)?;
        let completed =
            self.complete_span_evidence(&scope, &hits, before, after, scan_limit, at_ms)?;
        serde_json::to_string(&completed).map_err(json_status)
    }

    pub fn evaluate_span_hits_json(
        &self,
        hits_json: &str,
        relevant_span_ids_json: &str,
        required_exact_tokens_json: &str,
        stale_or_inadmissible_span_ids_json: &str,
        k: usize,
    ) -> ZResult<String> {
        let hits = serde_json::from_str::<Vec<SpanSearchHit>>(hits_json).map_err(json_status)?;
        let relevant =
            serde_json::from_str::<Vec<String>>(relevant_span_ids_json).map_err(json_status)?;
        let exact_tokens =
            serde_json::from_str::<Vec<String>>(required_exact_tokens_json).map_err(json_status)?;
        let stale = serde_json::from_str::<Vec<String>>(stale_or_inadmissible_span_ids_json)
            .map_err(json_status)?;
        let metrics = evaluate_span_hits_with_admission(&hits, &relevant, &exact_tokens, &stale, k);
        serde_json::to_string(&metrics).map_err(json_status)
    }

    pub fn evaluate_retrieval_baseline_json(&self, request_json: &str) -> ZResult<String> {
        let request = serde_json::from_str::<RetrievalBaselineJsonRequest>(request_json)
            .map_err(json_status)?;
        let report = evaluate_retrieval_baseline(
            &crate::RetrievalBaselineHits {
                vector: &request.vector_hits,
                keyword: &request.keyword_hits,
                hybrid: &request.hybrid_hits,
            },
            &request.relevant_span_ids,
            &request.required_exact_tokens,
            &request.stale_or_inadmissible_span_ids,
            request.k,
        );
        serde_json::to_string(&report).map_err(json_status)
    }

    pub fn scan_corrections_json(&self, scope_json: &str, limit: usize) -> ZResult<String> {
        let scope = serde_json::from_str::<MemoryScope>(scope_json).map_err(json_status)?;
        let records = self.scan_corrections(&scope, limit)?;
        serde_json::to_string(&records).map_err(json_status)
    }

    pub fn scan_claims_json(
        &self,
        scope_json: &str,
        limit: usize,
        at_ms: Option<i64>,
    ) -> ZResult<String> {
        let scope = serde_json::from_str::<MemoryScope>(scope_json).map_err(json_status)?;
        let records = self.scan_claims(&scope, limit, at_ms)?;
        serde_json::to_string(&records).map_err(json_status)
    }

    pub fn scan_current_claims_json(
        &self,
        scope_json: &str,
        limit: usize,
        at_ms: Option<i64>,
    ) -> ZResult<String> {
        let scope = serde_json::from_str::<MemoryScope>(scope_json).map_err(json_status)?;
        let records = self.scan_current_claims(&scope, limit, at_ms)?;
        serde_json::to_string(&records).map_err(json_status)
    }

    pub fn scan_artifacts_json(
        &self,
        scope_json: &str,
        limit: usize,
        at_ms: Option<i64>,
    ) -> ZResult<String> {
        let scope = serde_json::from_str::<MemoryScope>(scope_json).map_err(json_status)?;
        let records = self.scan_artifacts(&scope, limit, at_ms)?;
        serde_json::to_string(&records).map_err(json_status)
    }

    pub fn scan_profiles_json(
        &self,
        scope_json: &str,
        limit: usize,
        at_ms: Option<i64>,
    ) -> ZResult<String> {
        let scope = serde_json::from_str::<MemoryScope>(scope_json).map_err(json_status)?;
        let records = self.scan_profiles(&scope, limit, at_ms)?;
        serde_json::to_string(&records).map_err(json_status)
    }

    pub fn scan_entities_json(&self, scope_json: &str, limit: usize) -> ZResult<String> {
        let scope = serde_json::from_str::<MemoryScope>(scope_json).map_err(json_status)?;
        let records = self.scan_entities(&scope, limit)?;
        serde_json::to_string(&records).map_err(json_status)
    }

    pub fn expand_edges_json(
        &self,
        scope_json: &str,
        seed_entity_ids_json: &str,
        max_depth: usize,
        limit: usize,
        at_ms: Option<i64>,
    ) -> ZResult<String> {
        let scope = serde_json::from_str::<MemoryScope>(scope_json).map_err(json_status)?;
        let seed_entity_ids =
            serde_json::from_str::<Vec<String>>(seed_entity_ids_json).map_err(json_status)?;
        let hits = self.expand_edges_ranked(&scope, &seed_entity_ids, max_depth, limit, at_ms)?;
        serde_json::to_string(&hits).map_err(json_status)
    }

    pub fn build_context_json(
        &self,
        hits_json: &str,
        token_budget: usize,
        include_provenance: bool,
    ) -> ZResult<String> {
        let hits = serde_json::from_str::<Vec<SpanSearchHit>>(hits_json).map_err(json_status)?;
        let context = build_context(
            &hits,
            ContextOptions {
                token_budget,
                include_provenance,
            },
        );
        serde_json::to_string(&context).map_err(json_status)
    }

    pub fn build_memory_context_json(&self, request_json: &str) -> ZResult<String> {
        let _state_guard = self.lock_state_read();
        let request =
            serde_json::from_str::<MemoryContextJsonRequest>(request_json).map_err(json_status)?;
        let scope = &request.scope;
        let limits = &request.limits;
        let profiles = self.scan_profiles(scope, limits.profile_limit, limits.at_ms)?;
        let claims = self.scan_current_claims(scope, limits.claim_limit, limits.at_ms)?;
        let corrections =
            self.scan_context_corrections(scope, &claims, &request.hits, limits.at_ms)?;
        let artifacts = self.scan_artifacts(scope, limits.artifact_limit, limits.at_ms)?;
        let context = build_query_state_context(
            &ContextInput {
                profiles: &profiles,
                claims: &claims,
                corrections: &corrections,
                artifacts: &artifacts,
                hits: &request.hits,
                ..Default::default()
            },
            limits.context_options(),
        );
        serde_json::to_string(&context).map_err(json_status)
    }

    /// Evidence hits and answer-target selections a searched context request retrieves.
    fn searched_answer_evidence(
        &self,
        scope: &MemoryScope,
        query_text: &str,
        search: AnswerContextSearchJson,
        at_ms: Option<i64>,
    ) -> ZResult<(Vec<SpanSearchHit>, Vec<StateReadSelection>)> {
        let preferred_slot_ids = self.query_current_state_slot_ids(
            scope,
            search.query_embedding.clone(),
            query_text,
            search.k,
            at_ms,
        )?;
        let span_search = HybridSpanSearch {
            query_embedding: search.query_embedding,
            query_text,
            k: search.k,
            scan_limit: search.scan_limit,
            at_ms,
        };
        let hits = self.hybrid_search_spans(scope, span_search)?.fused_hits;
        let hits = self.complete_span_evidence(scope, &hits, 1, 1, search.scan_limit, at_ms)?;
        Ok((
            hits,
            StateReadSelection::answer_targets(&preferred_slot_ids),
        ))
    }

    /// Answer-ready context for one request. With `search` set, the evidence and the answer
    /// target slots are retrieved for the query; otherwise `hits` and `selections` are used as
    /// given.
    pub fn build_answer_ready_context_json(&self, request_json: &str) -> ZResult<String> {
        let _state_guard = self.lock_state_read();
        let request =
            serde_json::from_str::<AnswerContextJsonRequest>(request_json).map_err(json_status)?;
        let scope = &request.scope;
        let limits = &request.limits;
        let query_text = request.query_text.as_str();
        let (hits, selections) = match request.search {
            Some(search) => {
                self.searched_answer_evidence(scope, query_text, search, limits.at_ms)?
            }
            None => (request.hits, request.selections),
        };
        let state_projection = self.project_answer_ready_state(
            scope,
            &AnswerReadyStateRequest {
                projection_seed: "json_api",
                query_text,
                evidence_hits: &hits,
                selections: &selections,
                target_selections: None,
                claim_limit: limits.claim_limit,
                entity_limit: limits.claim_limit,
                temporal: BiTemporalQuery {
                    valid_at_ms: limits.at_ms,
                    transaction_at_ms: None,
                },
            },
        )?;
        let context_hits =
            self.hydrate_answer_ready_state_evidence(scope, &state_projection, &hits)?;
        let profiles = self.scan_profiles(scope, limits.profile_limit, limits.at_ms)?;
        let terms = unique_terms(lexical_terms(query_text));
        let artifacts = filter_artifacts_by_query(
            self.scan_artifacts(scope, expanded_limit(limits.artifact_limit), limits.at_ms)?,
            &terms,
            limits.artifact_limit,
        );
        let context = build_answer_ready_state_context(
            &profiles,
            &state_projection,
            &artifacts,
            &context_hits,
            limits.context_options(),
        );
        serde_json::to_string(&context).map_err(json_status)
    }
}

/// Request of `project_answer_ready_state_json`: the JSON form of `AnswerReadyStateRequest`.
#[derive(Debug, Deserialize)]
pub struct AnswerReadyStateJsonRequest {
    pub projection_seed: String,
    pub scope: MemoryScope,
    pub query_text: String,
    #[serde(default)]
    pub evidence_hits: Vec<SpanSearchHit>,
    #[serde(default)]
    pub selections: Vec<StateReadSelection>,
    #[serde(default)]
    pub target_selections: Option<Vec<StateReadSelection>>,
    #[serde(default = "default_state_limit")]
    pub claim_limit: usize,
    #[serde(default = "default_state_limit")]
    pub entity_limit: usize,
    #[serde(default)]
    pub temporal: BiTemporalQuery,
}

fn default_state_limit() -> usize {
    256
}

/// Record limits, valid time, and packing budget shared by the context requests.
#[derive(Debug, Deserialize)]
pub struct ContextLimitsJson {
    pub profile_limit: usize,
    pub claim_limit: usize,
    pub artifact_limit: usize,
    #[serde(default)]
    pub at_ms: Option<i64>,
    pub token_budget: usize,
    pub include_provenance: bool,
}

impl ContextLimitsJson {
    fn context_options(&self) -> ContextOptions {
        ContextOptions {
            token_budget: self.token_budget,
            include_provenance: self.include_provenance,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct MemoryContextJsonRequest {
    pub scope: MemoryScope,
    #[serde(default)]
    pub hits: Vec<SpanSearchHit>,
    #[serde(flatten)]
    pub limits: ContextLimitsJson,
}

/// Evidence retrieval for an answer context: hybrid span search plus answer-target slots.
#[derive(Debug, Deserialize)]
pub struct AnswerContextSearchJson {
    pub query_embedding: Vec<f32>,
    pub k: usize,
    pub scan_limit: usize,
}

#[derive(Debug, Deserialize)]
pub struct AnswerContextJsonRequest {
    pub scope: MemoryScope,
    pub query_text: String,
    #[serde(default)]
    pub hits: Vec<SpanSearchHit>,
    #[serde(default)]
    pub selections: Vec<StateReadSelection>,
    #[serde(default)]
    pub search: Option<AnswerContextSearchJson>,
    #[serde(flatten)]
    pub limits: ContextLimitsJson,
}

#[derive(Debug, Deserialize)]
pub struct RetrievalBaselineJsonRequest {
    pub vector_hits: Vec<SpanSearchHit>,
    pub keyword_hits: Vec<SpanSearchHit>,
    pub hybrid_hits: Vec<SpanSearchHit>,
    #[serde(default)]
    pub relevant_span_ids: Vec<String>,
    #[serde(default)]
    pub required_exact_tokens: Vec<String>,
    #[serde(default)]
    pub stale_or_inadmissible_span_ids: Vec<String>,
    pub k: usize,
}

fn json_status(err: serde_json::Error) -> Status {
    Status::invalid_argument(format!("memory JSON API error: {err}"))
}

fn expanded_limit(limit: usize) -> usize {
    limit.saturating_mul(8).max(limit)
}

pub fn filter_claims_by_query_text(
    claims: Vec<ClaimRecord>,
    query_text: &str,
    limit: usize,
) -> Vec<ClaimRecord> {
    let terms = unique_terms(lexical_terms(query_text));
    filter_claims_by_query(claims, &terms, limit)
}

fn filter_claims_by_query(
    claims: Vec<ClaimRecord>,
    terms: &[String],
    limit: usize,
) -> Vec<ClaimRecord> {
    rank_and_filter(claims, terms, limit, |claim| {
        format!(
            "{} {} {} {}",
            claim.claim_text,
            claim.subject.as_deref().unwrap_or(""),
            claim.predicate.as_deref().unwrap_or(""),
            claim.object_value.as_deref().unwrap_or("")
        )
    })
}

fn filter_artifacts_by_query(
    artifacts: Vec<ArtifactRecord>,
    terms: &[String],
    limit: usize,
) -> Vec<ArtifactRecord> {
    rank_and_filter(artifacts, terms, limit, |artifact| {
        format!(
            "{} {} {} {}",
            artifact.title,
            artifact.uri.as_deref().unwrap_or(""),
            artifact.blob_ref.as_deref().unwrap_or(""),
            artifact.metadata_json.as_deref().unwrap_or("")
        )
    })
}

fn rank_and_filter<T>(
    records: Vec<T>,
    terms: &[String],
    limit: usize,
    text: impl Fn(&T) -> String,
) -> Vec<T> {
    if limit == 0 {
        return Vec::new();
    }
    if terms.is_empty() {
        return records.into_iter().take(limit).collect();
    }
    let mut scored = records
        .into_iter()
        .filter_map(|record| {
            let haystack = text(&record).to_lowercase();
            let score = terms
                .iter()
                .filter(|term| haystack.contains(term.as_str()))
                .count();
            (score > 0).then_some((record, score))
        })
        .collect::<Vec<_>>();
    scored.sort_by(|(_, a), (_, b)| b.cmp(a));
    scored
        .into_iter()
        .take(limit)
        .map(|(record, _)| record)
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
mod tests {
    use super::*;
    use crate::{
        ActorKind, ClaimKind, ClaimPolarity, CorrectionAuthority, CorrectionOperation, EdgeInput,
        EntityInput, ManualClaimInput, ProfileInput, SourceKind, Visibility,
    };
    use finch_types::CollectionOptions;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn temp_dir() -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "finch_memory_json_api_{}_{}",
            std::process::id(),
            n
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn json_api_ingests_and_searches_spans() {
        let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
        let dir = temp_dir();
        let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
        let scope = MemoryScope::new("json");
        let input = EpisodeInput {
            id: Some("ep_json_api".to_string()),
            scope: scope.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            source_kind: SourceKind::UserMessage,
            actor: ActorKind::User,
            sequence_no: 1,
            event_time_ms: Some(10),
            valid_from_ms: Some(10),
            valid_to_ms: None,
            raw_text: "Remember ticket JSON-42 for bindings.".to_string(),
            blob_ref: None,
            mime_type: Some("text/plain".to_string()),
            causal_parent_ids: Vec::new(),
            metadata_json: None,
        };
        store
            .ingest_episode_json(&serde_json::to_string(&input).unwrap(), 10, None)
            .unwrap();
        let hits = store
            .keyword_search_spans_json(
                &serde_json::to_string(&scope).unwrap(),
                "JSON-42",
                10,
                100,
                Some(20),
            )
            .unwrap();
        assert!(hits.contains("JSON-42"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn json_api_semantic_claim_hit_projects_current_canonical_slot_state() {
        let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
        let dir = temp_dir();
        let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
        let scope = MemoryScope::new("json_state_projection");
        for (id, value, observed_at_ms, embedding) in [
            ("owner_previous", "Ana", 10, vec![1.0, 0.0, 0.0]),
            ("owner_current", "Bea", 20, vec![0.0, 1.0, 0.0]),
        ] {
            let episode_id = format!("episode_{id}");
            let episode = EpisodeInput {
                id: Some(episode_id.clone()),
                scope: scope.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                source_kind: SourceKind::UserMessage,
                actor: ActorKind::User,
                sequence_no: observed_at_ms,
                event_time_ms: Some(observed_at_ms),
                valid_from_ms: Some(observed_at_ms),
                valid_to_ms: None,
                raw_text: format!("Responsable: {value}"),
                blob_ref: None,
                mime_type: Some("text/plain".to_string()),
                causal_parent_ids: Vec::new(),
                metadata_json: None,
            };
            let ingested = store
                .ingest_episode_json(
                    &serde_json::to_string(&episode).unwrap(),
                    observed_at_ms,
                    None,
                )
                .unwrap();
            let ingested = serde_json::from_str::<crate::IngestedEpisode>(&ingested).unwrap();
            let claim = ManualClaimInput {
                id: Some(id.to_string()),
                scope: scope.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                claim_text: format!("Responsable: {value}"),
                subject: Some("proyecto".to_string()),
                predicate: Some("responsable".to_string()),
                object_value: Some(value.to_string()),
                claim_kind: ClaimKind::Fact,
                polarity: ClaimPolarity::Affirmative,
                source_span_ids: ingested.spans.iter().map(|span| span.id.clone()).collect(),
                source_episode_ids: vec![episode_id],
                asserted_by: "user".to_string(),
                confidence: Some(1.0),
                observed_at_ms,
                valid_from_ms: Some(observed_at_ms),
                valid_to_ms: None,
            };
            store
                .add_manual_claim_json(&serde_json::to_string(&claim).unwrap(), Some(embedding))
                .unwrap();
        }

        let context = store
            .build_answer_ready_context_json(
                &serde_json::json!({
                    "scope": scope,
                    "query_text": "誰ですか？",
                    "search": {"query_embedding": [1.0, 0.0, 0.0], "k": 1, "scan_limit": 20},
                    "profile_limit": 0,
                    "claim_limit": 10,
                    "artifact_limit": 0,
                    "at_ms": 30,
                    "token_budget": 256,
                    "include_provenance": true,
                })
                .to_string(),
            )
            .unwrap();

        assert!(context.contains("value: Bea"));
        assert!(!context.contains("value: Ana"));

        let owner_slot = store
            .scan_slots(&scope, 10, Some(30))
            .unwrap()
            .iter()
            .find(|slot| slot.predicate.as_deref() == Some("responsable"))
            .map(|slot| slot.slot_key.clone())
            .unwrap();
        let selections = serde_json::json!([{
            "slot_id": owner_slot,
            "view": "timeline"
        }]);
        let timeline_context = store
            .build_answer_ready_context_json(
                &serde_json::json!({
                    "scope": scope,
                    "query_text": "誰が担当していましたか？",
                    "selections": selections,
                    "profile_limit": 0,
                    "claim_limit": 10,
                    "artifact_limit": 0,
                    "at_ms": 30,
                    "token_budget": 4_096,
                    "include_provenance": true,
                })
                .to_string(),
            )
            .unwrap();
        assert!(
            timeline_context.contains("### SlotHistory"),
            "{timeline_context}"
        );
        assert!(timeline_context.contains("Ana"));
        assert!(timeline_context.contains("Bea"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn json_api_exposes_active_records_and_context() {
        let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
        let dir = temp_dir();
        let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
        let scope = MemoryScope::new("json_records");
        let scope_json = serde_json::to_string(&scope).unwrap();

        let claim = ManualClaimInput {
            id: Some("claim_json_api".to_string()),
            scope: scope.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            claim_text: "User prefers exact ticket IDs.".to_string(),
            subject: Some("user".to_string()),
            predicate: Some("prefers".to_string()),
            object_value: Some("exact ticket IDs".to_string()),
            claim_kind: ClaimKind::Preference,
            polarity: ClaimPolarity::Affirmative,
            source_span_ids: Vec::new(),
            source_episode_ids: Vec::new(),
            asserted_by: "user".to_string(),
            confidence: Some(1.0),
            observed_at_ms: 10,
            valid_from_ms: Some(10),
            valid_to_ms: None,
        };
        store
            .add_manual_claim_json(&serde_json::to_string(&claim).unwrap(), None)
            .unwrap();
        assert!(store
            .scan_claims_json(&scope_json, 10, Some(20))
            .unwrap()
            .contains("exact ticket IDs"));

        let profile = ProfileInput {
            id: Some("profile_json_api".to_string()),
            scope: scope.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            subject_id: Some("user".to_string()),
            profile_key: "style".to_string(),
            profile_text: "Use exact identifiers.".to_string(),
            value_json: None,
            evidence_claim_ids: vec!["claim_json_api".to_string()],
            source_span_ids: Vec::new(),
            generated_at_ms: 20,
            generator_version: "manual".to_string(),
            valid_from_ms: Some(20),
            valid_to_ms: None,
            correction_watermark: 0,
        };
        store
            .add_profile_json(&serde_json::to_string(&profile).unwrap())
            .unwrap();
        assert!(store
            .scan_profiles_json(&scope_json, 10, Some(20))
            .unwrap()
            .contains("Use exact identifiers"));

        let artifact = ArtifactRecord {
            id: "artifact_json_api".to_string(),
            scope: scope.clone(),
            status: crate::MemoryStatus::Active,
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            artifact_kind: crate::ArtifactKind::File,
            title: "bindings-runbook.md".to_string(),
            uri: Some("file:///bindings-runbook.md".to_string()),
            blob_ref: None,
            mime_type: Some("text/markdown".to_string()),
            content_hash: "artifact_hash".to_string(),
            source_created_at_ms: None,
            source_modified_at_ms: None,
            created_at_ms: 25,
            ingested_at_ms: 25,
            valid_from_ms: Some(25),
            valid_to_ms: None,
            extracted_text_ref: None,
            metadata_json: None,
        };
        store
            .ingest_artifact_text_json(
                &serde_json::to_string(&artifact).unwrap(),
                "Artifact text includes BINDINGS-RUNBOOK.",
                None,
            )
            .unwrap();
        assert!(store
            .scan_artifacts_json(&scope_json, 10, Some(30))
            .unwrap()
            .contains("bindings-runbook.md"));
        assert!(store
            .keyword_search_spans_json(&scope_json, "BINDINGS-RUNBOOK", 10, 100, Some(30))
            .unwrap()
            .contains("BINDINGS-RUNBOOK"));

        let entity_a = EntityInput {
            id: Some("entity_user".to_string()),
            scope: scope.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            entity_type: "person".to_string(),
            canonical_name: "User".to_string(),
            aliases: Vec::new(),
            source_claim_ids: Vec::new(),
            merge_parent_ids: Vec::new(),
            split_from_id: None,
            confidence: Some(1.0),
        };
        let entity_b = EntityInput {
            id: Some("entity_ticket".to_string()),
            canonical_name: "Ticket JSON-42".to_string(),
            ..entity_a.clone()
        };
        store
            .add_entity_json(&serde_json::to_string(&entity_a).unwrap(), None)
            .unwrap();
        store
            .add_entity_json(&serde_json::to_string(&entity_b).unwrap(), None)
            .unwrap();
        let edge = EdgeInput {
            id: Some("edge_json_api".to_string()),
            scope: scope.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            src_entity_id: "entity_user".to_string(),
            dst_entity_id: "entity_ticket".to_string(),
            relation_type: "mentions".to_string(),
            claim_id: None,
            source_span_ids: Vec::new(),
            valid_from_ms: Some(20),
            valid_to_ms: None,
            confidence: Some(0.8),
        };
        store
            .add_edge_json(&serde_json::to_string(&edge).unwrap())
            .unwrap();
        assert!(store
            .expand_edges_json(
                &scope_json,
                &serde_json::to_string(&vec!["entity_user".to_string()]).unwrap(),
                2,
                10,
                Some(30),
            )
            .unwrap()
            .contains("edge_json_api"));

        let input = EpisodeInput {
            id: Some("ep_json_context".to_string()),
            scope: scope.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            source_kind: SourceKind::UserMessage,
            actor: ActorKind::User,
            sequence_no: 2,
            event_time_ms: Some(30),
            valid_from_ms: Some(30),
            valid_to_ms: None,
            raw_text: "The useful exact identifier is JSON-42.".to_string(),
            blob_ref: None,
            mime_type: Some("text/plain".to_string()),
            causal_parent_ids: Vec::new(),
            metadata_json: None,
        };
        store
            .ingest_episode_json(&serde_json::to_string(&input).unwrap(), 30, None)
            .unwrap();
        let hits = store
            .keyword_search_spans_json(&scope_json, "JSON-42", 10, 100, Some(30))
            .unwrap();
        let hit_records = serde_json::from_str::<Vec<SpanSearchHit>>(&hits).unwrap();
        let hit_values = serde_json::from_str::<serde_json::Value>(&hits).unwrap();
        let relevant_ids = vec![hit_records[0].span.id.clone()];
        let metrics = store
            .evaluate_span_hits_json(
                &hits,
                &serde_json::to_string(&relevant_ids).unwrap(),
                &serde_json::to_string(&vec!["JSON-42".to_string()]).unwrap(),
                "[]",
                10,
            )
            .unwrap();
        assert!(metrics.contains("\"recall_at_k\":1.0"));
        assert!(metrics.contains("\"exact_token_miss_rate\":0.0"));
        let report = store
            .evaluate_retrieval_baseline_json(
                &serde_json::json!({
                    "vector_hits": hit_values,
                    "keyword_hits": hit_values,
                    "hybrid_hits": hit_values,
                    "relevant_span_ids": relevant_ids,
                    "required_exact_tokens": ["JSON-42"],
                    "k": 10,
                })
                .to_string(),
            )
            .unwrap();
        assert!(report.contains("\"hybrid_vs_vector\""));
        let packed = store
            .build_memory_context_json(
                &serde_json::json!({
                    "scope": scope,
                    "hits": hit_values,
                    "profile_limit": 10,
                    "claim_limit": 10,
                    "artifact_limit": 10,
                    "at_ms": 30,
                    "token_budget": 256,
                    "include_provenance": true,
                })
                .to_string(),
            )
            .unwrap();
        assert!(packed.contains("Use exact identifiers"));
        assert!(packed.contains("exact ticket IDs"));
        assert!(packed.contains("bindings-runbook.md"));
        assert!(packed.contains("JSON-42"));
        let one_shot = store
            .build_answer_ready_context_json(
                &serde_json::json!({
                    "scope": scope,
                    "query_text": "JSON-42",
                    "search": {"query_embedding": [0.0, 0.0, 0.0], "k": 10, "scan_limit": 100},
                    "profile_limit": 10,
                    "claim_limit": 10,
                    "artifact_limit": 10,
                    "at_ms": 30,
                    "token_budget": 256,
                    "include_provenance": true,
                })
                .to_string(),
            )
            .unwrap();
        assert!(one_shot.contains("JSON-42"));
        assert!(one_shot.contains("Use exact identifiers"));
        assert!(!one_shot.contains("exact ticket IDs"));
        assert!(!one_shot.contains("bindings-runbook.md"));
        let debug = store
            .hybrid_search_spans_debug_json(
                &scope_json,
                vec![0.0, 0.0, 0.0],
                "JSON-42",
                10,
                100,
                Some(30),
            )
            .unwrap();
        assert!(debug.contains("vector_hits"));
        assert!(debug.contains("keyword_hits"));
        assert!(debug.contains("fused_hits"));
        assert!(store
            .build_context_json(&hits, 128, true)
            .unwrap()
            .contains("JSON-42"));
        let answer_ready = store
            .build_answer_ready_context_json(
                &serde_json::json!({
                    "scope": scope,
                    "query_text": "exact ticket IDs",
                    "hits": hit_values,
                    "profile_limit": 10,
                    "claim_limit": 10,
                    "artifact_limit": 10,
                    "at_ms": 30,
                    "token_budget": 256,
                    "include_provenance": true,
                })
                .to_string(),
            )
            .unwrap();
        let answer_ready = serde_json::from_str::<serde_json::Value>(&answer_ready).unwrap();
        assert_eq!(answer_ready["support"]["state"], "no_evidence");
        assert_eq!(
            answer_ready["support"]["source_claim_ids"],
            serde_json::json!([])
        );

        let correction = CorrectionInput {
            id: Some("corr_json_api".to_string()),
            scope,
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            operation: CorrectionOperation::Tombstone,
            target_type: "span".to_string(),
            target_ids: vec!["span_missing".to_string()],
            target_selector: None,
            new_value: None,
            reason: Some("test".to_string()),
            actor: ActorKind::User,
            authority: CorrectionAuthority::User,
            effective_at_ms: None,
            applies_valid_from_ms: None,
            applies_valid_to_ms: None,
            cascade_policy: None,
            metadata_json: None,
        };
        store
            .add_correction_json(&serde_json::to_string(&correction).unwrap(), 40)
            .unwrap();
        assert!(store
            .scan_corrections_json(&scope_json, 10)
            .unwrap()
            .contains("corr_json_api"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}

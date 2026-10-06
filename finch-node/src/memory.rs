use crate::{finch_err, BlockingTask};
use finch_memory::MemoryStore as RustMemoryStore;
use finch_types::ZResult;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::sync::Arc;

#[napi]
pub struct MemoryStore {
    inner: Option<Arc<RustMemoryStore>>,
}

impl MemoryStore {
    fn inner(&self) -> Result<&Arc<RustMemoryStore>> {
        self.inner
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("MemoryStore is closed/destroyed"))
    }

    /// Runs `work` off the JS thread; the returned promise settles with its result.
    fn spawn<T: ToNapiValue + TypeName + Send + 'static>(
        &self,
        work: impl FnOnce(&RustMemoryStore) -> ZResult<T> + Send + 'static,
    ) -> Result<AsyncTask<BlockingTask<T>>> {
        let inner = self.inner()?.clone();
        Ok(BlockingTask::spawn(move || work(&inner)))
    }
}

#[napi]
impl MemoryStore {
    /// Creates a store in the new Postgres schema `name` at `url`.
    #[napi]
    pub fn create(
        url: String,
        name: String,
        embedding_dim: u32,
        vector_index: Option<bool>,
    ) -> AsyncTask<BlockingTask<MemoryStore>> {
        BlockingTask::spawn(move || {
            let store = RustMemoryStore::create(
                &url,
                &name,
                embedding_dim as usize,
                vector_index.unwrap_or(false),
            )?;
            Ok(Self {
                inner: Some(Arc::new(store)),
            })
        })
    }

    /// Opens the store in the Postgres schema `name` at `url`.
    #[napi]
    pub fn open(url: String, name: String) -> AsyncTask<BlockingTask<MemoryStore>> {
        BlockingTask::spawn(move || {
            let store = RustMemoryStore::open(&url, &name)?;
            Ok(Self {
                inner: Some(Arc::new(store)),
            })
        })
    }

    #[napi]
    pub fn apply_state_mutation_batch_json(
        &self,
        batch_json: String,
    ) -> Result<AsyncTask<BlockingTask<String>>> {
        self.spawn(move |m| m.apply_state_mutation_batch_json(&batch_json))
    }

    #[napi]
    pub fn project_answer_ready_state_json(
        &self,
        env: Env,
        request_json: String,
    ) -> Result<String> {
        self.inner()?
            .project_answer_ready_state_json(&request_json)
            .map_err(|e| finch_err(&env, e))
    }

    #[napi]
    pub fn ingest_episode_json(
        &self,
        input_json: String,
        ingested_at_ms: i64,
        chunk_options_json: Option<String>,
    ) -> Result<AsyncTask<BlockingTask<String>>> {
        self.spawn(move |m| {
            m.ingest_episode_json(&input_json, ingested_at_ms, chunk_options_json.as_deref())
        })
    }

    #[napi]
    pub fn append_artifact_json(
        &self,
        record_json: String,
    ) -> Result<AsyncTask<BlockingTask<String>>> {
        self.spawn(move |m| m.append_artifact_json(&record_json))
    }

    #[napi]
    pub fn ingest_artifact_text_json(
        &self,
        record_json: String,
        text: String,
        chunk_options_json: Option<String>,
    ) -> Result<AsyncTask<BlockingTask<String>>> {
        self.spawn(move |m| {
            m.ingest_artifact_text_json(&record_json, &text, chunk_options_json.as_deref())
        })
    }

    #[napi]
    pub fn add_correction_json(
        &self,
        input_json: String,
        created_at_ms: i64,
    ) -> Result<AsyncTask<BlockingTask<String>>> {
        self.spawn(move |m| m.add_correction_json(&input_json, created_at_ms))
    }

    #[napi]
    pub fn add_manual_claim_json(
        &self,
        input_json: String,
        embedding: Option<Vec<f64>>,
    ) -> Result<AsyncTask<BlockingTask<String>>> {
        self.spawn(move |m| {
            m.add_manual_claim_json(
                &input_json,
                embedding.map(|values| values.into_iter().map(|value| value as f32).collect()),
            )
        })
    }

    #[napi]
    pub fn add_profile_json(&self, input_json: String) -> Result<AsyncTask<BlockingTask<String>>> {
        self.spawn(move |m| m.add_profile_json(&input_json))
    }

    #[napi]
    pub fn add_entity_json(
        &self,
        input_json: String,
        embedding: Option<Vec<f64>>,
    ) -> Result<AsyncTask<BlockingTask<String>>> {
        self.spawn(move |m| {
            m.add_entity_json(
                &input_json,
                embedding.map(|values| values.into_iter().map(|value| value as f32).collect()),
            )
        })
    }

    #[napi]
    pub fn add_edge_json(&self, input_json: String) -> Result<AsyncTask<BlockingTask<String>>> {
        self.spawn(move |m| m.add_edge_json(&input_json))
    }

    #[napi]
    pub fn add_slot_alias_json(
        &self,
        input_json: String,
        recorded_at_ms: i64,
    ) -> Result<AsyncTask<BlockingTask<String>>> {
        self.spawn(move |m| m.add_slot_alias_json(&input_json, recorded_at_ms))
    }

    #[napi]
    pub fn scan_slot_aliases_json(
        &self,
        env: Env,
        scope_json: String,
        limit: u32,
        at_ms: Option<i64>,
    ) -> Result<String> {
        self.inner()?
            .scan_slot_aliases_json(&scope_json, limit as usize, at_ms)
            .map_err(|e| finch_err(&env, e))
    }

    #[napi]
    pub fn query_spans_json(
        &self,
        env: Env,
        scope_json: String,
        query_embedding: Vec<f64>,
        k: u32,
        at_ms: Option<i64>,
    ) -> Result<String> {
        self.inner()?
            .query_spans_json(
                &scope_json,
                query_embedding
                    .into_iter()
                    .map(|value| value as f32)
                    .collect(),
                k as usize,
                at_ms,
            )
            .map_err(|e| finch_err(&env, e))
    }

    #[napi]
    pub fn keyword_search_spans_json(
        &self,
        env: Env,
        scope_json: String,
        query_text: String,
        k: u32,
        scan_limit: u32,
        at_ms: Option<i64>,
    ) -> Result<String> {
        self.inner()?
            .keyword_search_spans_json(
                &scope_json,
                &query_text,
                k as usize,
                scan_limit as usize,
                at_ms,
            )
            .map_err(|e| finch_err(&env, e))
    }

    #[napi]
    pub fn hybrid_search_spans_json(
        &self,
        env: Env,
        scope_json: String,
        query_embedding: Vec<f64>,
        query_text: String,
        k: u32,
        scan_limit: u32,
        at_ms: Option<i64>,
    ) -> Result<String> {
        self.inner()?
            .hybrid_search_spans_json(
                &scope_json,
                query_embedding
                    .into_iter()
                    .map(|value| value as f32)
                    .collect(),
                &query_text,
                k as usize,
                scan_limit as usize,
                at_ms,
            )
            .map_err(|e| finch_err(&env, e))
    }

    #[napi]
    pub fn hybrid_search_spans_debug_json(
        &self,
        env: Env,
        scope_json: String,
        query_embedding: Vec<f64>,
        query_text: String,
        k: u32,
        scan_limit: u32,
        at_ms: Option<i64>,
    ) -> Result<String> {
        self.inner()?
            .hybrid_search_spans_debug_json(
                &scope_json,
                query_embedding
                    .into_iter()
                    .map(|value| value as f32)
                    .collect(),
                &query_text,
                k as usize,
                scan_limit as usize,
                at_ms,
            )
            .map_err(|e| finch_err(&env, e))
    }

    #[napi]
    pub fn complete_span_evidence_json(
        &self,
        env: Env,
        scope_json: String,
        hits_json: String,
        before: u32,
        after: u32,
        scan_limit: u32,
        at_ms: Option<i64>,
    ) -> Result<String> {
        self.inner()?
            .complete_span_evidence_json(
                &scope_json,
                &hits_json,
                before as usize,
                after as usize,
                scan_limit as usize,
                at_ms,
            )
            .map_err(|e| finch_err(&env, e))
    }

    #[napi]
    pub fn evaluate_span_hits_json(
        &self,
        env: Env,
        hits_json: String,
        relevant_span_ids_json: String,
        required_exact_tokens_json: String,
        stale_or_inadmissible_span_ids_json: String,
        k: u32,
    ) -> Result<String> {
        self.inner()?
            .evaluate_span_hits_json(
                &hits_json,
                &relevant_span_ids_json,
                &required_exact_tokens_json,
                &stale_or_inadmissible_span_ids_json,
                k as usize,
            )
            .map_err(|e| finch_err(&env, e))
    }

    #[napi]
    pub fn evaluate_retrieval_baseline_json(
        &self,
        env: Env,
        request_json: String,
    ) -> Result<String> {
        self.inner()?
            .evaluate_retrieval_baseline_json(&request_json)
            .map_err(|e| finch_err(&env, e))
    }

    #[napi]
    pub fn scan_corrections_json(
        &self,
        env: Env,
        scope_json: String,
        limit: u32,
    ) -> Result<String> {
        self.inner()?
            .scan_corrections_json(&scope_json, limit as usize)
            .map_err(|e| finch_err(&env, e))
    }

    #[napi]
    pub fn scan_claims_json(
        &self,
        env: Env,
        scope_json: String,
        limit: u32,
        at_ms: Option<i64>,
    ) -> Result<String> {
        self.inner()?
            .scan_claims_json(&scope_json, limit as usize, at_ms)
            .map_err(|e| finch_err(&env, e))
    }

    #[napi]
    pub fn scan_current_claims_json(
        &self,
        env: Env,
        scope_json: String,
        limit: u32,
        at_ms: Option<i64>,
    ) -> Result<String> {
        self.inner()?
            .scan_current_claims_json(&scope_json, limit as usize, at_ms)
            .map_err(|e| finch_err(&env, e))
    }

    #[napi]
    pub fn scan_artifacts_json(
        &self,
        env: Env,
        scope_json: String,
        limit: u32,
        at_ms: Option<i64>,
    ) -> Result<String> {
        self.inner()?
            .scan_artifacts_json(&scope_json, limit as usize, at_ms)
            .map_err(|e| finch_err(&env, e))
    }

    #[napi]
    pub fn scan_profiles_json(
        &self,
        env: Env,
        scope_json: String,
        limit: u32,
        at_ms: Option<i64>,
    ) -> Result<String> {
        self.inner()?
            .scan_profiles_json(&scope_json, limit as usize, at_ms)
            .map_err(|e| finch_err(&env, e))
    }

    #[napi]
    pub fn scan_entities_json(&self, env: Env, scope_json: String, limit: u32) -> Result<String> {
        self.inner()?
            .scan_entities_json(&scope_json, limit as usize)
            .map_err(|e| finch_err(&env, e))
    }

    #[napi]
    pub fn expand_edges_json(
        &self,
        env: Env,
        scope_json: String,
        seed_entity_ids_json: String,
        max_depth: u32,
        limit: u32,
        at_ms: Option<i64>,
    ) -> Result<String> {
        self.inner()?
            .expand_edges_json(
                &scope_json,
                &seed_entity_ids_json,
                max_depth as usize,
                limit as usize,
                at_ms,
            )
            .map_err(|e| finch_err(&env, e))
    }

    #[napi]
    pub fn build_context_json(
        &self,
        env: Env,
        hits_json: String,
        token_budget: u32,
        include_provenance: bool,
    ) -> Result<String> {
        self.inner()?
            .build_context_json(&hits_json, token_budget as usize, include_provenance)
            .map_err(|e| finch_err(&env, e))
    }

    #[napi]
    pub fn build_memory_context_json(&self, env: Env, request_json: String) -> Result<String> {
        self.inner()?
            .build_memory_context_json(&request_json)
            .map_err(|e| finch_err(&env, e))
    }

    #[napi]
    pub fn build_answer_ready_context_json(
        &self,
        env: Env,
        request_json: String,
    ) -> Result<String> {
        self.inner()?
            .build_answer_ready_context_json(&request_json)
            .map_err(|e| finch_err(&env, e))
    }

    #[napi]
    pub fn validate_answer_emission_json(
        &self,
        env: Env,
        scope_json: String,
        support_json: String,
        emission_json: String,
    ) -> Result<String> {
        self.inner()?
            .validate_answer_emission_json(&scope_json, &support_json, &emission_json)
            .map_err(|e| finch_err(&env, e))
    }

    #[napi]
    pub fn validate_answer_context_json(
        &self,
        env: Env,
        scope_json: String,
        context_json: String,
        emission_json: String,
    ) -> Result<String> {
        self.inner()?
            .validate_answer_context_json(&scope_json, &context_json, &emission_json)
            .map_err(|e| finch_err(&env, e))
    }

    #[napi]
    pub fn finalize_answer_context_json(
        &self,
        env: Env,
        scope_json: String,
        context_json: String,
        emission_json: String,
    ) -> Result<String> {
        self.inner()?
            .finalize_answer_context_json(&scope_json, &context_json, &emission_json)
            .map_err(|e| finch_err(&env, e))
    }

    #[napi]
    pub fn close(&mut self) {
        self.inner = None;
    }
}

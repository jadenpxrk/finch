use finch_memory::MemoryStore as RustMemoryStore;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::path::Path;

fn to_napi_err(e: finch_types::Status) -> napi::Error {
    napi::Error::from_reason(e.to_string())
}

#[napi(object)]
pub struct MemoryStoreOptions {
    pub read_only: Option<bool>,
    pub enable_mmap: Option<bool>,
    pub max_buffer_size: Option<u32>,
}

fn options_from_node(options: Option<MemoryStoreOptions>) -> finch_types::CollectionOptions {
    let mut out = finch_types::CollectionOptions::default();
    if let Some(options) = options {
        if let Some(read_only) = options.read_only {
            out.read_only = read_only;
        }
        if let Some(enable_mmap) = options.enable_mmap {
            out.enable_mmap = enable_mmap;
        }
        if let Some(max_buffer_size) = options.max_buffer_size {
            out.max_buffer_size = max_buffer_size;
        }
    }
    out
}

#[napi]
pub struct MemoryStore {
    inner: Option<RustMemoryStore>,
}

impl MemoryStore {
    fn inner(&self) -> Result<&RustMemoryStore> {
        self.inner
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("MemoryStore is closed/destroyed"))
    }
}

#[napi]
impl MemoryStore {
    #[napi(factory)]
    pub fn create(
        path: String,
        embedding_dim: u32,
        options: Option<MemoryStoreOptions>,
    ) -> Result<Self> {
        Ok(Self {
            inner: Some(
                RustMemoryStore::create(
                    Path::new(&path),
                    embedding_dim as usize,
                    options_from_node(options),
                )
                .map_err(to_napi_err)?,
            ),
        })
    }

    #[napi(factory)]
    pub fn open(path: String, options: Option<MemoryStoreOptions>) -> Result<Self> {
        Ok(Self {
            inner: Some(
                RustMemoryStore::open(Path::new(&path), options_from_node(options))
                    .map_err(to_napi_err)?,
            ),
        })
    }

    #[napi]
    pub fn apply_state_mutation_batch_json(&self, batch_json: String) -> Result<String> {
        self.inner()?
            .apply_state_mutation_batch_json(&batch_json)
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn project_answer_ready_state_json(&self, request_json: String) -> Result<String> {
        self.inner()?
            .project_answer_ready_state_json(&request_json)
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn ingest_episode_json(
        &self,
        input_json: String,
        ingested_at_ms: i64,
        chunk_options_json: Option<String>,
    ) -> Result<String> {
        self.inner()?
            .ingest_episode_json(&input_json, ingested_at_ms, chunk_options_json.as_deref())
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn append_artifact_json(&self, record_json: String) -> Result<String> {
        self.inner()?
            .append_artifact_json(&record_json)
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn ingest_artifact_text_json(
        &self,
        record_json: String,
        text: String,
        chunk_options_json: Option<String>,
    ) -> Result<String> {
        self.inner()?
            .ingest_artifact_text_json(&record_json, &text, chunk_options_json.as_deref())
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn add_correction_json(&self, input_json: String, created_at_ms: i64) -> Result<String> {
        self.inner()?
            .add_correction_json(&input_json, created_at_ms)
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn add_manual_claim_json(
        &self,
        input_json: String,
        embedding: Option<Vec<f64>>,
    ) -> Result<String> {
        self.inner()?
            .add_manual_claim_json(
                &input_json,
                embedding.map(|values| values.into_iter().map(|value| value as f32).collect()),
            )
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn add_profile_json(&self, input_json: String) -> Result<String> {
        self.inner()?
            .add_profile_json(&input_json)
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn add_entity_json(
        &self,
        input_json: String,
        embedding: Option<Vec<f64>>,
    ) -> Result<String> {
        self.inner()?
            .add_entity_json(
                &input_json,
                embedding.map(|values| values.into_iter().map(|value| value as f32).collect()),
            )
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn add_edge_json(&self, input_json: String) -> Result<String> {
        self.inner()?
            .add_edge_json(&input_json)
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn add_slot_alias_json(&self, input_json: String, recorded_at_ms: i64) -> Result<String> {
        self.inner()?
            .add_slot_alias_json(&input_json, recorded_at_ms)
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn scan_slot_aliases_json(
        &self,
        scope_json: String,
        limit: u32,
        at_ms: Option<i64>,
    ) -> Result<String> {
        self.inner()?
            .scan_slot_aliases_json(&scope_json, limit as usize, at_ms)
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn query_spans_json(
        &self,
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
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn keyword_search_spans_json(
        &self,
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
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn hybrid_search_spans_json(
        &self,
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
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn hybrid_search_spans_debug_json(
        &self,
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
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn complete_span_evidence_json(
        &self,
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
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn evaluate_span_hits_json(
        &self,
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
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn evaluate_retrieval_baseline_json(&self, request_json: String) -> Result<String> {
        self.inner()?
            .evaluate_retrieval_baseline_json(&request_json)
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn scan_corrections_json(&self, scope_json: String, limit: u32) -> Result<String> {
        self.inner()?
            .scan_corrections_json(&scope_json, limit as usize)
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn scan_claims_json(
        &self,
        scope_json: String,
        limit: u32,
        at_ms: Option<i64>,
    ) -> Result<String> {
        self.inner()?
            .scan_claims_json(&scope_json, limit as usize, at_ms)
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn scan_current_claims_json(
        &self,
        scope_json: String,
        limit: u32,
        at_ms: Option<i64>,
    ) -> Result<String> {
        self.inner()?
            .scan_current_claims_json(&scope_json, limit as usize, at_ms)
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn scan_artifacts_json(
        &self,
        scope_json: String,
        limit: u32,
        at_ms: Option<i64>,
    ) -> Result<String> {
        self.inner()?
            .scan_artifacts_json(&scope_json, limit as usize, at_ms)
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn scan_profiles_json(
        &self,
        scope_json: String,
        limit: u32,
        at_ms: Option<i64>,
    ) -> Result<String> {
        self.inner()?
            .scan_profiles_json(&scope_json, limit as usize, at_ms)
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn scan_entities_json(&self, scope_json: String, limit: u32) -> Result<String> {
        self.inner()?
            .scan_entities_json(&scope_json, limit as usize)
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn expand_edges_json(
        &self,
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
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn build_context_json(
        &self,
        hits_json: String,
        token_budget: u32,
        include_provenance: bool,
    ) -> Result<String> {
        self.inner()?
            .build_context_json(&hits_json, token_budget as usize, include_provenance)
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn build_memory_context_json(&self, request_json: String) -> Result<String> {
        self.inner()?
            .build_memory_context_json(&request_json)
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn build_answer_ready_context_json(&self, request_json: String) -> Result<String> {
        self.inner()?
            .build_answer_ready_context_json(&request_json)
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn validate_answer_emission_json(
        &self,
        scope_json: String,
        support_json: String,
        emission_json: String,
    ) -> Result<String> {
        self.inner()?
            .validate_answer_emission_json(&scope_json, &support_json, &emission_json)
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn validate_answer_context_json(
        &self,
        scope_json: String,
        context_json: String,
        emission_json: String,
    ) -> Result<String> {
        self.inner()?
            .validate_answer_context_json(&scope_json, &context_json, &emission_json)
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn finalize_answer_context_json(
        &self,
        scope_json: String,
        context_json: String,
        emission_json: String,
    ) -> Result<String> {
        self.inner()?
            .finalize_answer_context_json(&scope_json, &context_json, &emission_json)
            .map_err(to_napi_err)
    }

    #[napi]
    pub fn close(&mut self) {
        self.inner = None;
    }
}

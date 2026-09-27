use crate::schema::PyCollectionOption;
use crate::{to_py_err, PyResult};
use finch_memory::MemoryStore;
use pyo3::prelude::*;
use std::path::Path;

#[pyclass(module = "finch._finch")]
pub struct PyMemoryStore {
    inner: MemoryStore,
}

#[pymethods]
impl PyMemoryStore {
    #[staticmethod]
    #[pyo3(signature = (path, embedding_dim, options=None))]
    fn create(
        path: &str,
        embedding_dim: usize,
        options: Option<&PyCollectionOption>,
    ) -> PyResult<Self> {
        let options = options
            .map(|option| option.inner.clone())
            .unwrap_or_default();
        Ok(Self {
            inner: MemoryStore::create(Path::new(path), embedding_dim, options)
                .map_err(to_py_err)?,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (path, options=None))]
    fn open(path: &str, options: Option<&PyCollectionOption>) -> PyResult<Self> {
        let options = options
            .map(|option| option.inner.clone())
            .unwrap_or_default();
        Ok(Self {
            inner: MemoryStore::open(Path::new(path), options).map_err(to_py_err)?,
        })
    }

    #[pyo3(signature = (batch_json))]
    fn apply_state_mutation_batch_json(&self, batch_json: &str) -> PyResult<String> {
        self.inner
            .apply_state_mutation_batch_json(batch_json)
            .map_err(to_py_err)
    }

    fn project_answer_ready_state_json(&self, request_json: &str) -> PyResult<String> {
        self.inner
            .project_answer_ready_state_json(request_json)
            .map_err(to_py_err)
    }

    #[pyo3(signature = (input_json, ingested_at_ms, chunk_options_json=None))]
    fn ingest_episode_json(
        &self,
        input_json: &str,
        ingested_at_ms: i64,
        chunk_options_json: Option<&str>,
    ) -> PyResult<String> {
        self.inner
            .ingest_episode_json(input_json, ingested_at_ms, chunk_options_json)
            .map_err(to_py_err)
    }

    fn append_artifact_json(&self, record_json: &str) -> PyResult<String> {
        self.inner
            .append_artifact_json(record_json)
            .map_err(to_py_err)
    }

    #[pyo3(signature = (record_json, text, chunk_options_json=None))]
    fn ingest_artifact_text_json(
        &self,
        record_json: &str,
        text: &str,
        chunk_options_json: Option<&str>,
    ) -> PyResult<String> {
        self.inner
            .ingest_artifact_text_json(record_json, text, chunk_options_json)
            .map_err(to_py_err)
    }

    fn add_correction_json(&self, input_json: &str, created_at_ms: i64) -> PyResult<String> {
        self.inner
            .add_correction_json(input_json, created_at_ms)
            .map_err(to_py_err)
    }

    #[pyo3(signature = (input_json, embedding=None))]
    fn add_manual_claim_json(
        &self,
        input_json: &str,
        embedding: Option<Vec<f32>>,
    ) -> PyResult<String> {
        self.inner
            .add_manual_claim_json(input_json, embedding)
            .map_err(to_py_err)
    }

    fn add_profile_json(&self, input_json: &str) -> PyResult<String> {
        self.inner.add_profile_json(input_json).map_err(to_py_err)
    }

    #[pyo3(signature = (input_json, embedding=None))]
    fn add_entity_json(&self, input_json: &str, embedding: Option<Vec<f32>>) -> PyResult<String> {
        self.inner
            .add_entity_json(input_json, embedding)
            .map_err(to_py_err)
    }

    fn add_edge_json(&self, input_json: &str) -> PyResult<String> {
        self.inner.add_edge_json(input_json).map_err(to_py_err)
    }

    fn add_slot_alias_json(&self, input_json: &str, recorded_at_ms: i64) -> PyResult<String> {
        self.inner
            .add_slot_alias_json(input_json, recorded_at_ms)
            .map_err(to_py_err)
    }

    #[pyo3(signature = (scope_json, limit=100, at_ms=None))]
    fn scan_slot_aliases_json(
        &self,
        scope_json: &str,
        limit: usize,
        at_ms: Option<i64>,
    ) -> PyResult<String> {
        self.inner
            .scan_slot_aliases_json(scope_json, limit, at_ms)
            .map_err(to_py_err)
    }

    #[pyo3(signature = (scope_json, query_embedding, k=10, at_ms=None))]
    fn query_spans_json(
        &self,
        scope_json: &str,
        query_embedding: Vec<f32>,
        k: usize,
        at_ms: Option<i64>,
    ) -> PyResult<String> {
        self.inner
            .query_spans_json(scope_json, query_embedding, k, at_ms)
            .map_err(to_py_err)
    }

    #[pyo3(signature = (scope_json, query_text, k=10, scan_limit=1000, at_ms=None))]
    fn keyword_search_spans_json(
        &self,
        scope_json: &str,
        query_text: &str,
        k: usize,
        scan_limit: usize,
        at_ms: Option<i64>,
    ) -> PyResult<String> {
        self.inner
            .keyword_search_spans_json(scope_json, query_text, k, scan_limit, at_ms)
            .map_err(to_py_err)
    }

    #[pyo3(signature = (scope_json, query_embedding, query_text, k=10, scan_limit=1000, at_ms=None))]
    fn hybrid_search_spans_json(
        &self,
        scope_json: &str,
        query_embedding: Vec<f32>,
        query_text: &str,
        k: usize,
        scan_limit: usize,
        at_ms: Option<i64>,
    ) -> PyResult<String> {
        self.inner
            .hybrid_search_spans_json(
                scope_json,
                query_embedding,
                query_text,
                k,
                scan_limit,
                at_ms,
            )
            .map_err(to_py_err)
    }

    #[pyo3(signature = (scope_json, query_embedding, query_text, k=10, scan_limit=1000, at_ms=None))]
    fn hybrid_search_spans_debug_json(
        &self,
        scope_json: &str,
        query_embedding: Vec<f32>,
        query_text: &str,
        k: usize,
        scan_limit: usize,
        at_ms: Option<i64>,
    ) -> PyResult<String> {
        self.inner
            .hybrid_search_spans_debug_json(
                scope_json,
                query_embedding,
                query_text,
                k,
                scan_limit,
                at_ms,
            )
            .map_err(to_py_err)
    }

    #[pyo3(signature = (scope_json, hits_json, before=1, after=1, scan_limit=1000, at_ms=None))]
    fn complete_span_evidence_json(
        &self,
        scope_json: &str,
        hits_json: &str,
        before: usize,
        after: usize,
        scan_limit: usize,
        at_ms: Option<i64>,
    ) -> PyResult<String> {
        self.inner
            .complete_span_evidence_json(scope_json, hits_json, before, after, scan_limit, at_ms)
            .map_err(to_py_err)
    }

    fn evaluate_span_hits_json(
        &self,
        hits_json: &str,
        relevant_span_ids_json: &str,
        required_exact_tokens_json: &str,
        stale_or_inadmissible_span_ids_json: &str,
        k: usize,
    ) -> PyResult<String> {
        self.inner
            .evaluate_span_hits_json(
                hits_json,
                relevant_span_ids_json,
                required_exact_tokens_json,
                stale_or_inadmissible_span_ids_json,
                k,
            )
            .map_err(to_py_err)
    }

    fn evaluate_retrieval_baseline_json(&self, request_json: &str) -> PyResult<String> {
        self.inner
            .evaluate_retrieval_baseline_json(request_json)
            .map_err(to_py_err)
    }

    fn scan_corrections_json(&self, scope_json: &str, limit: usize) -> PyResult<String> {
        self.inner
            .scan_corrections_json(scope_json, limit)
            .map_err(to_py_err)
    }

    #[pyo3(signature = (scope_json, limit=1000, at_ms=None))]
    fn scan_claims_json(
        &self,
        scope_json: &str,
        limit: usize,
        at_ms: Option<i64>,
    ) -> PyResult<String> {
        self.inner
            .scan_claims_json(scope_json, limit, at_ms)
            .map_err(to_py_err)
    }

    #[pyo3(signature = (scope_json, limit=1000, at_ms=None))]
    fn scan_current_claims_json(
        &self,
        scope_json: &str,
        limit: usize,
        at_ms: Option<i64>,
    ) -> PyResult<String> {
        self.inner
            .scan_current_claims_json(scope_json, limit, at_ms)
            .map_err(to_py_err)
    }

    #[pyo3(signature = (scope_json, limit=1000, at_ms=None))]
    fn scan_artifacts_json(
        &self,
        scope_json: &str,
        limit: usize,
        at_ms: Option<i64>,
    ) -> PyResult<String> {
        self.inner
            .scan_artifacts_json(scope_json, limit, at_ms)
            .map_err(to_py_err)
    }

    #[pyo3(signature = (scope_json, limit=1000, at_ms=None))]
    fn scan_profiles_json(
        &self,
        scope_json: &str,
        limit: usize,
        at_ms: Option<i64>,
    ) -> PyResult<String> {
        self.inner
            .scan_profiles_json(scope_json, limit, at_ms)
            .map_err(to_py_err)
    }

    fn scan_entities_json(&self, scope_json: &str, limit: usize) -> PyResult<String> {
        self.inner
            .scan_entities_json(scope_json, limit)
            .map_err(to_py_err)
    }

    #[pyo3(signature = (scope_json, seed_entity_ids_json, max_depth=1, limit=100, at_ms=None))]
    fn expand_edges_json(
        &self,
        scope_json: &str,
        seed_entity_ids_json: &str,
        max_depth: usize,
        limit: usize,
        at_ms: Option<i64>,
    ) -> PyResult<String> {
        self.inner
            .expand_edges_json(scope_json, seed_entity_ids_json, max_depth, limit, at_ms)
            .map_err(to_py_err)
    }

    #[pyo3(signature = (hits_json, token_budget=4096, include_provenance=true))]
    fn build_context_json(
        &self,
        hits_json: &str,
        token_budget: usize,
        include_provenance: bool,
    ) -> PyResult<String> {
        self.inner
            .build_context_json(hits_json, token_budget, include_provenance)
            .map_err(to_py_err)
    }

    fn build_memory_context_json(&self, request_json: &str) -> PyResult<String> {
        self.inner
            .build_memory_context_json(request_json)
            .map_err(to_py_err)
    }

    fn build_answer_ready_context_json(&self, request_json: &str) -> PyResult<String> {
        self.inner
            .build_answer_ready_context_json(request_json)
            .map_err(to_py_err)
    }

    fn validate_answer_emission_json(
        &self,
        scope_json: &str,
        support_json: &str,
        emission_json: &str,
    ) -> PyResult<String> {
        self.inner
            .validate_answer_emission_json(scope_json, support_json, emission_json)
            .map_err(to_py_err)
    }

    fn validate_answer_context_json(
        &self,
        scope_json: &str,
        context_json: &str,
        emission_json: &str,
    ) -> PyResult<String> {
        self.inner
            .validate_answer_context_json(scope_json, context_json, emission_json)
            .map_err(to_py_err)
    }

    fn finalize_answer_context_json(
        &self,
        scope_json: &str,
        context_json: &str,
        emission_json: &str,
    ) -> PyResult<String> {
        self.inner
            .finalize_answer_context_json(scope_json, context_json, emission_json)
            .map_err(to_py_err)
    }
}

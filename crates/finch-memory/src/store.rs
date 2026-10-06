use crate::ingest::{
    chunk_artifact_text, create_edge, create_entity, create_manual_claim, ingest_episode,
    stable_hash_hex, ChunkOptions, EdgeInput, EntityInput, EpisodeInput, IngestedArtifact,
    IngestedEpisode, ManualClaimInput, ProfileInput,
};
use crate::postgres::PgSession;
use crate::retrieval::{
    apply_corrections_to_span_hits_at, hybrid_fuse_rrf, lexical_terms, span_active_at,
    SpanSearchHit,
};
use crate::row::{
    artifact_doc, artifact_from_doc, claim_doc, claim_from_doc, correction_doc,
    correction_from_doc, dependency_trace_doc, dependency_trace_from_doc, edge_doc, edge_from_doc,
    entity_alias_doc, entity_alias_from_doc, entity_doc, entity_from_doc, episode_doc, profile_doc,
    profile_from_doc, rule_doc, rule_from_doc, slot_alias_doc, slot_alias_from_doc, slot_doc,
    slot_from_doc, span_doc, span_from_doc, state_record_doc, state_record_from_doc,
};
use crate::schema::{
    active_collection_schemas, ARTIFACTS_COLLECTION, CLAIMS_COLLECTION, CORRECTIONS_COLLECTION,
    DEPENDENCY_TRACES_COLLECTION, EDGES_COLLECTION, ENTITIES_COLLECTION, ENTITY_ALIASES_COLLECTION,
    EPISODES_COLLECTION, PROFILES_COLLECTION, RULES_COLLECTION, SLOTS_COLLECTION,
    SLOT_ALIASES_COLLECTION, SPANS_COLLECTION, STATE_RECORDS_COLLECTION, TERMS_COLLECTION,
};
use crate::state::{
    aggregate_support_state, AnswerSlotSupport, AnswerSupportContract, AnswerSupportState,
    BiTemporalQuery,
};
use crate::table::{MemoryTable, ReadCachedTable};
use crate::types::{
    canonical_slot_part, AnswerReadyStateRequest, ArtifactRecord, CanonicalSlot,
    CanonicalSlotBindingContext, CanonicalSlotRecord, ClaimKind, ClaimPolarity, ClaimRecord,
    CorrectionOperation, CorrectionRecord, CorrectionTargetMatch, DependencyTraceRecord,
    EdgeRecord, EntityAliasRecord, EntityRecord, EpisodeRecord, HybridSpanSearch, MemoryId,
    MemoryScope, MemoryStatus, ProfileRecord, RepairableTriggerSlot, ResolvedRuleApplication,
    RuleAction, RuleActivation, RuleBindingStatus, RuleInput, RuleRecord, RuleResolutionOutcome,
    RuleResolutionStatus, RuleTargetMatch, SetStateRecord, SlotAliasInput, SlotAliasRecord,
    SlotHistoryRecord, SlotHistoryVersion, SlotSurfaceAlias, SlotTemporalState, SpanRecord,
    StateReadRole, StateReadSelection, StateReadView, StateRecord, StateRecordKind,
    StateRecordScan, Visibility,
};
use finch_types::{CollectionSchema, Doc, HnswIndexParams, Status, Value, VectorQuery, ZResult};
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub(crate) mod canonical;
mod corrections;
mod graph;
mod hybrid;
mod lifecycle;
mod projection;
mod records;
mod rules;
mod search;
pub(crate) mod state_records;
mod timeline;

pub use search::{
    ExpansionQuery, HybridSourceDiverseOptions, HybridSourceDiverseQuery, SourceSpanSelection,
};

use canonical::{
    canonical_slot_id, canonicalize_claim, canonicalize_correction, canonicalize_rule,
    entity_names, CanonicalRegistry,
};
pub(crate) use lifecycle::dedupe_claims_by_id;
use lifecycle::{
    active_correction_filter, apply_corrections_to_claims, claim_facet_lifecycle_key,
    claim_is_newer, claim_matches_selector, claim_slot, claim_slot_lifecycle_key,
    correction_already_projected, correction_applies_to_claim_time, insert_scoped_correction,
    interval_filter_clause, interval_holds_at, materialize_rule_application, newest_claim,
    resolve_current_claims, rule_target_matches_claim, rule_trigger_interval,
    rule_trigger_matches_claim, sort_corrections_by_effect, temporal_position_is_after,
};
pub(crate) use projection::StateProjectionFrontier;
pub(crate) use state_records::ProjectionWrite;
use state_records::{
    assess_state_records, claim_state_record_kind, current_state_query_overlap,
    dependency_trace_from_application, dependency_trace_id, lexical_query_terms,
    mentioned_entity_ids, merge_slot_identity_records, merge_trace_rule_ids,
    normalized_phrase_present, prepare_slot_identity_revisions, proof_span_ids_for_claims,
    reconcile_set_states_with_slot_lifecycle, select_state_records, slot_records_from_projection,
    sorted_unique_ids, state_lexical_fields, state_record_priority, state_record_recorded_at,
    state_record_sets_semantically_equal, state_record_valid_at, state_record_visible_at,
    state_records_from_claims, state_records_semantically_equal, state_records_to_projection,
    state_records_to_slot_histories, support_contract_for_claims, support_state_for_claim_refs,
    system_time_ms, transaction_visible_at, versioned_projection_id, StateSelectionAssessment,
    StateSelectionQuery,
};
use timeline::{reconcile_dependency_trace_intervals, reconcile_state_record_intervals};

pub struct MemoryStore {
    id: u64,
    pub(crate) episodes: Arc<dyn MemoryTable>,
    pub(crate) spans: Arc<dyn MemoryTable>,
    artifacts: Arc<dyn MemoryTable>,
    pub(crate) corrections: Arc<dyn MemoryTable>,
    terms: Arc<dyn MemoryTable>,
    pub(crate) claims: Arc<dyn MemoryTable>,
    profiles: Arc<dyn MemoryTable>,
    pub(crate) entities: Arc<dyn MemoryTable>,
    entity_aliases: Arc<dyn MemoryTable>,
    edges: Arc<dyn MemoryTable>,
    pub(crate) rules: Arc<dyn MemoryTable>,
    state_records: Arc<dyn MemoryTable>,
    dependency_traces: Arc<dyn MemoryTable>,
    slots: Arc<dyn MemoryTable>,
    pub(crate) slot_aliases: Arc<dyn MemoryTable>,
    session: Arc<PgSession>,
    state_mutation_lock: RwLock<()>,
}

static NEXT_STORE_ID: AtomicU64 = AtomicU64::new(0);

thread_local! {
    // Stores whose write lock this thread holds; a write calls public reads, which must not relock.
    static STATE_WRITE_HELD: RefCell<BTreeSet<u64>> = const { RefCell::new(BTreeSet::new()) };
}

pub(crate) struct StateWriteGuard<'a> {
    _lock: RwLockWriteGuard<'a, ()>,
    store_id: u64,
}

pub(crate) struct StateReadGuard<'a> {
    _lock: Option<RwLockReadGuard<'a, ()>>,
    ends_cache: bool,
}

#[cfg(test)]
impl StateReadGuard<'_> {
    /// False when the calling thread already holds this store's write lock.
    pub(crate) fn holds_lock(&self) -> bool {
        self._lock.is_some()
    }
}

impl Drop for StateReadGuard<'_> {
    fn drop(&mut self) {
        if self.ends_cache {
            crate::table::end_read_cache();
        }
    }
}

impl Drop for StateWriteGuard<'_> {
    fn drop(&mut self) {
        STATE_WRITE_HELD.with(|held| held.borrow_mut().remove(&self.store_id));
    }
}

impl MemoryStore {
    pub(crate) fn lock_state_mutation(&self) -> StateWriteGuard<'_> {
        let lock = self.state_mutation_lock.write();
        STATE_WRITE_HELD.with(|held| held.borrow_mut().insert(self.id));
        StateWriteGuard {
            _lock: lock,
            store_id: self.id,
        }
    }

    /// Public reads hold this so they see a state mutation batch whole or not at all. The
    /// outermost read also serves its repeated state-table reads from memory.
    pub(crate) fn lock_state_read(&self) -> ZResult<StateReadGuard<'_>> {
        if STATE_WRITE_HELD.with(|held| held.borrow().contains(&self.id)) {
            return Ok(StateReadGuard {
                _lock: None,
                ends_cache: false,
            });
        }
        // Recursive so a public read that calls another cannot deadlock behind a waiting writer.
        let guard = self.state_mutation_lock.read_recursive();
        Ok(StateReadGuard {
            _lock: Some(guard),
            ends_cache: crate::table::begin_read_cache(),
        })
    }

    /// The one entry point of a state mutation: holds the write lock and runs every write of
    /// `mutate` in one transaction, committing on success and rolling back on error.
    pub(crate) fn with_state_mutation<T>(
        &self,
        _scope: &MemoryScope,
        mutate: impl FnOnce() -> ZResult<T>,
    ) -> ZResult<T> {
        let _mutation_guard = self.lock_state_mutation();
        self.session.begin()?;
        match mutate() {
            Ok(value) => {
                self.session.commit()?;
                Ok(value)
            }
            Err(error) => match self.session.rollback() {
                Ok(()) => Err(error),
                Err(rollback) => Err(Status::internal(format!(
                    "{error}; rolling it back failed: {rollback}"
                ))),
            },
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AnswerReadyStateProjection {
    pub claims: Vec<ClaimRecord>,
    pub set_states: Vec<SetStateRecord>,
    #[serde(default)]
    pub slot_histories: Vec<SlotHistoryRecord>,
    pub rules: Vec<RuleRecord>,
    #[serde(default)]
    pub rule_outcomes: Vec<RuleResolutionOutcome>,
    pub corrections: Vec<CorrectionRecord>,
    pub entities: Vec<EntityRecord>,
    pub proof_span_ids: Vec<MemoryId>,
    pub support: AnswerSupportContract,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedAnswerSlot {
    pub slot_id: MemoryId,
    pub read_view: StateReadView,
    pub support_state: AnswerSupportState,
    pub state_kind: Option<StateRecordKind>,
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub current_value: Option<String>,
    pub members: Vec<String>,
    /// Co-current surface facets of the slot family, newest first. A slot with more than one
    /// supported facet value has no single authoritative `current_value`; the reader sees every
    /// facet with its own validity and proof instead of a newest-wins guess.
    #[serde(default)]
    pub facets: Vec<ResolvedSlotFacet>,
    pub source_claim_ids: Vec<MemoryId>,
    pub source_span_ids: Vec<MemoryId>,
    pub dependency_rule_ids: Vec<MemoryId>,
    pub dependency_trigger_slot_ids: Vec<MemoryId>,
    pub valid_from_ms: Option<i64>,
    pub valid_to_ms: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedSlotFacet {
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub support_state: AnswerSupportState,
    pub state_kind: Option<StateRecordKind>,
    pub value: Option<String>,
    /// Evidence-side statement text of the newest claim in the facet.
    #[serde(default)]
    pub state_text: Option<String>,
    pub claim_ids: Vec<MemoryId>,
    pub source_span_ids: Vec<MemoryId>,
    pub valid_from_ms: Option<i64>,
    pub valid_to_ms: Option<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateSelectionDisposition {
    Selected,
    PreferredSlotFiltered,
    CandidateFiltered,
    NoSelectionSignal,
    ClaimLimit,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateSelectionSignals {
    pub preferred_slot: bool,
    pub evidence: bool,
    pub anchored_slot: bool,
    pub set_entity: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateSelectionTraceItem {
    pub state_id: MemoryId,
    pub state_kind: StateRecordKind,
    pub slot_id: Option<MemoryId>,
    pub subject_entity_id: Option<MemoryId>,
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub score: u8,
    pub signals: StateSelectionSignals,
    pub disposition: StateSelectionDisposition,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotHistoryTrace {
    pub slot_id: MemoryId,
    pub subject_entity_id: Option<MemoryId>,
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub version_count: usize,
    pub distinct_value_count: usize,
    pub first_observed_at_ms: i64,
    pub last_observed_at_ms: i64,
    pub first_valid_from_ms: Option<i64>,
    pub last_valid_from_ms: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnswerReadyStateProjectionTrace {
    pub preferred_slot_ids: Vec<MemoryId>,
    #[serde(default)]
    pub answer_target_slot_ids: Vec<MemoryId>,
    pub evidence_span_ids: Vec<MemoryId>,
    pub candidates: Vec<StateSelectionTraceItem>,
    pub slot_histories: Vec<SlotHistoryTrace>,
    pub projected_claim_ids: Vec<MemoryId>,
    pub projected_set_state_ids: Vec<MemoryId>,
    pub projected_rule_ids: Vec<MemoryId>,
}

const RULE_OUTPUT_FIELDS: &[&str] = &[
    "id",
    "space_id",
    "tenant_id",
    "user_id",
    "agent_id",
    "project_id",
    "thread_id",
    "visibility",
    "policy_tags_json",
    "status",
    "trigger_subject_key",
    "trigger_predicate_key",
    "target_subject_key",
    "target_predicate_key",
    "trigger_subject_entity_id",
    "target_subject_entity_id",
    "trigger_slot_id",
    "target_slot_id",
    "trigger_binding_status",
    "target_binding_status",
    "trigger_subject",
    "trigger_predicate",
    "target_subject",
    "target_predicate",
    "target_match",
    "activation",
    "action",
    "value_template",
    "value",
    "source_span_ids_json",
    "source_eps_json",
    "source_sequence_no",
    "valid_from_ms",
    "valid_to_ms",
    "confidence",
];

const SLOT_ALIAS_OUTPUT_FIELDS: &[&str] = &[
    "id",
    "space_id",
    "tenant_id",
    "user_id",
    "agent_id",
    "project_id",
    "thread_id",
    "visibility",
    "policy_tags_json",
    "status",
    "alias_subject_key",
    "alias_predicate_key",
    "alias_subject_entity_id",
    "canonical_slot_id",
    "source_claims_json",
    "valid_from_ms",
    "valid_to_ms",
    "recorded_at_ms",
    "superseded_at_ms",
];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphExpansionHit {
    pub edge: EdgeRecord,
    pub depth: usize,
    pub score: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HybridSearchDebug {
    pub vector_hits: Vec<SpanSearchHit>,
    pub keyword_hits: Vec<SpanSearchHit>,
    pub fused_hits: Vec<SpanSearchHit>,
}

const SPAN_OUTPUT_FIELDS: &[&str] = &[
    "id",
    "space_id",
    "tenant_id",
    "user_id",
    "agent_id",
    "project_id",
    "thread_id",
    "visibility",
    "policy_tags_json",
    "status",
    "source_type",
    "source_id",
    "byte_start",
    "byte_end",
    "char_start",
    "char_end",
    "token_start",
    "token_end",
    "span_index",
    "text",
    "text_hash",
    "chunker_version",
    "embedding_model",
    "embedding_version",
    "lexical_text",
    "created_at_ms",
    "valid_from_ms",
    "valid_to_ms",
    "provenance_json",
];

const CORRECTION_OUTPUT_FIELDS: &[&str] = &[
    "id",
    "space_id",
    "tenant_id",
    "user_id",
    "agent_id",
    "project_id",
    "thread_id",
    "visibility",
    "policy_tags_json",
    "status",
    "operation",
    "target_type",
    "target_match",
    "target_ids_json",
    "target_selector",
    "target_subject_key",
    "target_predicate_key",
    "target_subject_entity_id",
    "target_slot_id",
    "new_value",
    "reason",
    "actor_kind",
    "authority",
    "created_at_ms",
    "effective_at_ms",
    "applies_from_ms",
    "applies_to_ms",
    "cascade_policy",
    "source_spans_json",
    "source_eps_json",
    "source_sequence_no",
    "metadata_json",
];

const MAX_VECTOR_QUERY_TOPK: usize = 1024;
const MAX_CONTAINS_FILTER_VALUES: usize = 31;
const SPAN_VECTOR_FETCH_MULTIPLIER: usize = 8;
const CLAIM_SLOT_VECTOR_FETCH_MULTIPLIER: usize = 8;
const MAX_RULE_HOPS: usize = 8;

const TERM_OUTPUT_FIELDS: &[&str] = &["term", "span_id", "tf", "doc_len"];

fn expanded_span_vector_fetch_k(k: usize) -> usize {
    k.saturating_mul(SPAN_VECTOR_FETCH_MULTIPLIER)
        .max(k)
        .min(MAX_VECTOR_QUERY_TOPK)
}

fn expanded_claim_slot_vector_fetch_k(k: usize) -> usize {
    k.saturating_mul(CLAIM_SLOT_VECTOR_FETCH_MULTIPLIER)
        .max(k)
        .min(MAX_VECTOR_QUERY_TOPK)
}

const ARTIFACT_OUTPUT_FIELDS: &[&str] = &[
    "id",
    "space_id",
    "tenant_id",
    "user_id",
    "agent_id",
    "project_id",
    "thread_id",
    "visibility",
    "policy_tags_json",
    "status",
    "artifact_kind",
    "title",
    "uri",
    "blob_ref",
    "mime_type",
    "content_hash",
    "source_created_ms",
    "source_modified_ms",
    "created_at_ms",
    "ingested_at_ms",
    "valid_from_ms",
    "valid_to_ms",
    "extracted_text_ref",
    "metadata_json",
];

const CLAIM_OUTPUT_FIELDS: &[&str] = &[
    "id",
    "space_id",
    "tenant_id",
    "user_id",
    "agent_id",
    "project_id",
    "thread_id",
    "visibility",
    "policy_tags_json",
    "status",
    "claim_text",
    "subject",
    "predicate",
    "object_value",
    "subject_entity_id",
    "slot_id",
    "slot_facet",
    "claim_kind",
    "polarity",
    "source_spans_json",
    "source_eps_json",
    "source_sequence_no",
    "asserted_by",
    "extractor_version",
    "confidence",
    "observed_at_ms",
    "valid_from_ms",
    "valid_to_ms",
    "correction_ids_json",
];

const PROFILE_OUTPUT_FIELDS: &[&str] = &[
    "id",
    "space_id",
    "tenant_id",
    "user_id",
    "agent_id",
    "project_id",
    "thread_id",
    "visibility",
    "policy_tags_json",
    "status",
    "subject_id",
    "profile_key",
    "profile_text",
    "value_json",
    "evidence_claims_json",
    "source_spans_json",
    "generated_at_ms",
    "generator_version",
    "valid_from_ms",
    "valid_to_ms",
    "correction_watermark",
];

const ENTITY_OUTPUT_FIELDS: &[&str] = &[
    "id",
    "space_id",
    "tenant_id",
    "user_id",
    "agent_id",
    "project_id",
    "thread_id",
    "visibility",
    "policy_tags_json",
    "status",
    "entity_type",
    "canonical_name",
    "aliases_json",
    "source_claims_json",
    "merge_parents_json",
    "split_from_id",
    "confidence",
];

const ENTITY_ALIAS_OUTPUT_FIELDS: &[&str] = &[
    "id",
    "space_id",
    "tenant_id",
    "user_id",
    "agent_id",
    "project_id",
    "thread_id",
    "visibility",
    "policy_tags_json",
    "status",
    "alias_key",
    "entity_id",
    "entity_type",
    "source_claims_json",
    "recorded_at_ms",
    "superseded_at_ms",
];

const EDGE_OUTPUT_FIELDS: &[&str] = &[
    "id",
    "space_id",
    "tenant_id",
    "user_id",
    "agent_id",
    "project_id",
    "thread_id",
    "visibility",
    "policy_tags_json",
    "status",
    "src_entity_id",
    "dst_entity_id",
    "relation_type",
    "claim_id",
    "source_spans_json",
    "valid_from_ms",
    "valid_to_ms",
    "confidence",
];

const STATE_RECORD_OUTPUT_FIELDS: &[&str] = &[
    "id",
    "state_key",
    "space_id",
    "tenant_id",
    "user_id",
    "agent_id",
    "project_id",
    "thread_id",
    "visibility",
    "policy_tags_json",
    "status",
    "state_kind",
    "claim_kind",
    "subject",
    "predicate",
    "object_value",
    "subject_entity_id",
    "slot_id",
    "slot_facet",
    "state_text",
    "members_json",
    "claim_ids_json",
    "correction_ids_json",
    "rule_ids_json",
    "source_spans_json",
    "source_eps_json",
    "source_sequence_no",
    "observed_at_ms",
    "valid_from_ms",
    "valid_to_ms",
    "projected_at_ms",
    "recorded_at_ms",
    "superseded_at_ms",
];

const DEPENDENCY_TRACE_OUTPUT_FIELDS: &[&str] = &[
    "id",
    "trace_key",
    "space_id",
    "tenant_id",
    "user_id",
    "agent_id",
    "project_id",
    "thread_id",
    "visibility",
    "policy_tags_json",
    "status",
    "rule_id",
    "trigger_claim_id",
    "prior_trigger_claim_id",
    "derived_claim_id",
    "target_subject_key",
    "target_predicate_key",
    "target_subject_entity_id",
    "target_slot_id",
    "target_subject",
    "target_predicate",
    "state_kind",
    "hop",
    "parent_trace_ids_json",
    "source_spans_json",
    "source_eps_json",
    "observed_at_ms",
    "valid_from_ms",
    "valid_to_ms",
    "projected_at_ms",
    "recorded_at_ms",
    "superseded_at_ms",
];

const SLOT_OUTPUT_FIELDS: &[&str] = &[
    "id",
    "slot_key",
    "space_id",
    "tenant_id",
    "user_id",
    "agent_id",
    "project_id",
    "thread_id",
    "visibility",
    "policy_tags_json",
    "status",
    "subject_key",
    "predicate_key",
    "subject_entity_id",
    "subject",
    "predicate",
    "source_claim_ids_json",
    "source_rule_ids_json",
    "source_entity_ids_json",
    "observed_at_ms",
    "valid_from_ms",
    "valid_to_ms",
    "projected_at_ms",
    "recorded_at_ms",
    "superseded_at_ms",
];

impl MemoryStore {
    pub fn ensure_span_hnsw_index(
        &self,
        hnsw_params: HnswIndexParams,
        concurrency: Option<usize>,
    ) -> ZResult<()> {
        self.spans
            .create_hnsw_index("embedding", hnsw_params, concurrency)
    }

    /// Builds the store from one table per collection of the session's schema.
    pub(crate) fn from_tables(
        session: Arc<PgSession>,
        mut table: impl FnMut(&'static str) -> ZResult<Arc<dyn MemoryTable>>,
    ) -> ZResult<Self> {
        let id = NEXT_STORE_ID.fetch_add(1, Ordering::Relaxed);
        Ok(Self {
            id,
            episodes: table(EPISODES_COLLECTION)?,
            spans: table(SPANS_COLLECTION)?,
            artifacts: table(ARTIFACTS_COLLECTION)?,
            corrections: read_cached(id, CORRECTIONS_COLLECTION, table(CORRECTIONS_COLLECTION)?),
            terms: table(TERMS_COLLECTION)?,
            claims: read_cached(id, CLAIMS_COLLECTION, table(CLAIMS_COLLECTION)?),
            profiles: table(PROFILES_COLLECTION)?,
            entities: read_cached(id, ENTITIES_COLLECTION, table(ENTITIES_COLLECTION)?),
            entity_aliases: read_cached(
                id,
                ENTITY_ALIASES_COLLECTION,
                table(ENTITY_ALIASES_COLLECTION)?,
            ),
            edges: table(EDGES_COLLECTION)?,
            rules: read_cached(id, RULES_COLLECTION, table(RULES_COLLECTION)?),
            state_records: read_cached(
                id,
                STATE_RECORDS_COLLECTION,
                table(STATE_RECORDS_COLLECTION)?,
            ),
            dependency_traces: read_cached(
                id,
                DEPENDENCY_TRACES_COLLECTION,
                table(DEPENDENCY_TRACES_COLLECTION)?,
            ),
            slots: read_cached(id, SLOTS_COLLECTION, table(SLOTS_COLLECTION)?),
            slot_aliases: read_cached(id, SLOT_ALIASES_COLLECTION, table(SLOT_ALIASES_COLLECTION)?),
            session,
            state_mutation_lock: RwLock::new(()),
        })
    }

    pub(crate) fn append_episode(&self, record: &EpisodeRecord) -> ZResult<()> {
        insert_one(&self.episodes, episode_doc(record).map_err(json_error)?)
    }

    pub(crate) fn append_span(
        &self,
        record: &SpanRecord,
        embedding: Option<&[f32]>,
    ) -> ZResult<()> {
        insert_one(
            &self.spans,
            span_doc(record, embedding).map_err(json_error)?,
        )?;
        self.index_span_terms(record)
    }

    pub fn append_ingested_episodes_with_embeddings(
        &self,
        records: &[(IngestedEpisode, Vec<Vec<f32>>)],
    ) -> ZResult<()> {
        let mut episode_docs = Vec::new();
        let mut span_docs = Vec::new();
        let mut term_docs = Vec::new();

        for (ingested, embeddings) in records {
            if ingested.spans.len() != embeddings.len() {
                return Err(Status::invalid_argument(
                    "span embedding count does not match ingested span count",
                ));
            }
            episode_docs.push(episode_doc(&ingested.episode).map_err(json_error)?);
            for (span, embedding) in ingested.spans.iter().zip(embeddings) {
                span_docs.push(span_doc(span, Some(embedding)).map_err(json_error)?);
                term_docs.extend(term_docs_for_span(span));
            }
        }

        insert_many(&self.episodes, episode_docs)?;
        insert_many(&self.spans, span_docs)?;
        insert_many(&self.terms, term_docs)
    }

    pub fn append_vector_spans(&self, records: &[(SpanRecord, Vec<f32>)]) -> ZResult<()> {
        let docs = records
            .iter()
            .map(|(span, embedding)| span_doc(span, Some(embedding)).map_err(json_error))
            .collect::<ZResult<Vec<_>>>()?;
        insert_many(&self.spans, docs)
    }

    pub(crate) fn append_artifact(&self, record: &ArtifactRecord) -> ZResult<()> {
        insert_one(&self.artifacts, artifact_doc(record).map_err(json_error)?)
    }

    pub fn ingest_artifact_text(
        &self,
        record: ArtifactRecord,
        text: &str,
        chunk_options: &ChunkOptions,
    ) -> ZResult<IngestedArtifact> {
        let spans = chunk_artifact_text(&record, text, chunk_options);
        self.append_artifact(&record)?;
        for span in &spans {
            self.append_span(span, None)?;
        }
        Ok(IngestedArtifact {
            artifact: record,
            spans,
        })
    }

    pub fn add_correction(&self, record: &CorrectionRecord) -> ZResult<()> {
        self.with_state_mutation(&record.scope, || {
            if !record.source_span_ids.is_empty() || !record.source_episode_ids.is_empty() {
                self.validate_evidence_references(
                    &record.scope,
                    &record.source_span_ids,
                    &record.source_episode_ids,
                )?;
            } else if !matches!(record.actor, crate::ActorKind::User) {
                return Err(Status::invalid_argument(
                    "non-user corrections require source evidence",
                ));
            }
            let mut record = self.resolve_existing_correction_target(record)?;
            record.source_sequence_no =
                self.source_sequence_no_for_episode_ids(&record.scope, &record.source_episode_ids)?;
            let correction_docs = vec![correction_doc(&record).map_err(json_error)?];
            insert_many(&self.corrections, correction_docs)?;
            if record.target_type != "claim" {
                return Ok(());
            }
            let affected_slot_ids = self.correction_target_slot_ids_of(&record)?;
            if affected_slot_ids.is_empty() {
                if record.target_selector.is_some() {
                    self.rebuild_state_projection(&record.scope, Some(record.effective_at_ms))?;
                }
                return Ok(());
            }
            let projected =
                self.current_state_records_for_slot_ids(&record.scope, &affected_slot_ids)?;
            if correction_already_projected(&record, &projected) {
                return Ok(());
            }
            self.project_correction(&record, affected_slot_ids)
        })
    }

    /// The slot a stored claim correction names plus the slots of the claims it targets.
    fn correction_target_slot_ids_of(
        &self,
        record: &CorrectionRecord,
    ) -> ZResult<BTreeSet<MemoryId>> {
        let mut slot_ids = record
            .target_slot_id
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        if !record.target_ids.is_empty() {
            slot_ids.extend(
                self.claims_by_ids(&record.scope, &record.target_ids)?
                    .into_iter()
                    .filter_map(|claim| claim.slot_id),
            );
        }
        Ok(slot_ids)
    }

    /// Re-evaluates the rules over the corrected slots' current claims and projects the slots
    /// at the correction's effective time.
    fn project_correction(
        &self,
        record: &CorrectionRecord,
        mut affected_slot_ids: BTreeSet<MemoryId>,
    ) -> ZResult<()> {
        let transaction_time_ms = system_time_ms();
        let mut changed_claims = self.scan_current_claims_for_slot_ids(
            &record.scope,
            &affected_slot_ids,
            usize::MAX,
            Some(record.effective_at_ms),
        )?;
        changed_claims.retain(|claim| claim.scope == record.scope);
        let known_ids = changed_claims
            .iter()
            .map(|claim| claim.id.clone())
            .collect::<BTreeSet<_>>();
        let applications =
            self.resolve_rules_for_changed_claims(&changed_claims, &known_ids, MAX_RULE_HOPS)?;
        let derived_docs = applications
            .iter()
            .map(|application| claim_doc(&application.claim, None).map_err(json_error))
            .collect::<ZResult<Vec<_>>>()?;
        // Surviving triggers re-derive claims already stored under the same deterministic ids.
        upsert_many(&self.claims, derived_docs)?;
        affected_slot_ids.extend(
            applications
                .iter()
                .filter_map(|application| application.claim.slot_id.clone()),
        );
        self.project_state_slots(
            &record.scope,
            StateProjectionFrontier {
                slot_ids: &affected_slot_ids,
                claims: None,
                applications: &applications,
                valid_at_ms: Some(record.effective_at_ms),
                write: ProjectionWrite {
                    time_ms: transaction_time_ms,
                    key: &record.id,
                },
            },
        )?;
        Ok(())
    }
}

fn insert_one(collection: &dyn MemoryTable, doc: Doc) -> ZResult<()> {
    let statuses = collection.insert(vec![doc])?;
    match statuses.into_iter().next() {
        Some(status) if status.ok() => Ok(()),
        Some(status) => Err(status),
        None => Err(Status::internal("insert returned no status")),
    }
}

pub(crate) fn insert_many(collection: &dyn MemoryTable, docs: Vec<Doc>) -> ZResult<()> {
    for chunk in docs.chunks(1024) {
        let statuses = collection.insert(chunk.to_vec())?;
        for status in statuses {
            if !status.ok() {
                return Err(status);
            }
        }
    }
    Ok(())
}

pub(crate) fn upsert_many(collection: &dyn MemoryTable, docs: Vec<Doc>) -> ZResult<()> {
    for chunk in docs.chunks(1024) {
        let statuses = collection.upsert(chunk.to_vec())?;
        for status in statuses {
            if !status.ok() {
                return Err(status);
            }
        }
    }
    Ok(())
}

fn term_docs_for_span(span: &SpanRecord) -> Vec<Doc> {
    let text = if span.lexical_text.is_empty() {
        &span.text
    } else {
        &span.lexical_text
    };
    let terms = lexical_terms(text);
    if terms.is_empty() {
        return Vec::new();
    }
    let doc_len = terms.len() as i64;
    let mut counts = BTreeMap::<String, i64>::new();
    for term in terms {
        *counts.entry(term).or_default() += 1;
    }
    counts
        .into_iter()
        .map(|(term, tf)| {
            let id = stable_hash_hex(&[span.id.as_str(), term.as_str()]);
            term_doc(&id, span, &term, tf, doc_len)
        })
        .collect()
}

fn term_doc(id: &str, span: &SpanRecord, term: &str, tf: i64, doc_len: i64) -> Doc {
    Doc::new(id.to_string())
        .set("id", Value::String(id.to_string()))
        .set("space_id", Value::String(span.scope.space_id.clone()))
        .set(
            "tenant_id",
            span.scope
                .tenant_id
                .clone()
                .map(Value::String)
                .unwrap_or(Value::Null),
        )
        .set(
            "user_id",
            span.scope
                .user_id
                .clone()
                .map(Value::String)
                .unwrap_or(Value::Null),
        )
        .set(
            "agent_id",
            span.scope
                .agent_id
                .clone()
                .map(Value::String)
                .unwrap_or(Value::Null),
        )
        .set(
            "project_id",
            span.scope
                .project_id
                .clone()
                .map(Value::String)
                .unwrap_or(Value::Null),
        )
        .set(
            "thread_id",
            span.scope
                .thread_id
                .clone()
                .map(Value::String)
                .unwrap_or(Value::Null),
        )
        .set(
            "visibility",
            Value::String(span.visibility.as_str().to_string()),
        )
        .set(
            "policy_tags_json",
            Value::String(serde_json::to_string(&span.policy_tags).unwrap_or_default()),
        )
        .set("term", Value::String(term.to_string()))
        .set("span_id", Value::String(span.id.clone()))
        .set("tf", Value::I64(tf))
        .set("doc_len", Value::I64(doc_len))
}

struct TermPosting {
    term: String,
    span_id: String,
    tf: f32,
    doc_len: f32,
}

fn bm25_posting_scores(query_terms: &[String], postings: &[TermPosting]) -> BTreeMap<String, f32> {
    let doc_count = postings
        .iter()
        .map(|posting| posting.span_id.as_str())
        .collect::<BTreeSet<_>>()
        .len()
        .max(1) as f32;
    let avg_len = (postings
        .iter()
        .map(|posting| (posting.span_id.as_str(), posting.doc_len))
        .collect::<BTreeMap<_, _>>()
        .values()
        .sum::<f32>()
        / doc_count)
        .max(1.0);
    let mut df = BTreeMap::<&str, usize>::new();
    for term in query_terms {
        let docs = postings
            .iter()
            .filter(|posting| posting.term == *term)
            .map(|posting| posting.span_id.as_str())
            .collect::<BTreeSet<_>>();
        df.insert(term.as_str(), docs.len());
    }
    let mut scores = BTreeMap::<String, f32>::new();
    for posting in postings {
        if !query_terms.iter().any(|term| term == &posting.term) {
            continue;
        }
        let doc_freq = *df.get(posting.term.as_str()).unwrap_or(&0) as f32;
        let idf = ((doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0).ln();
        let k1 = 1.2;
        let b = 0.75;
        let denom = posting.tf + k1 * (1.0 - b + b * posting.doc_len / avg_len);
        *scores.entry(posting.span_id.clone()).or_default() +=
            idf * (posting.tf * (k1 + 1.0)) / denom;
    }
    scores
}

/// Keeps the first item of each id, in order.
pub(crate) fn retain_first_by_id<T>(items: &mut Vec<T>, id: impl Fn(&T) -> &str) {
    let first_of_id = {
        let mut seen = BTreeSet::new();
        items
            .iter()
            .map(|item| seen.insert(id(item)))
            .collect::<Vec<_>>()
    };
    let mut first_of_id = first_of_id.into_iter();
    items.retain(|_| first_of_id.next().unwrap_or(false));
}

fn unique_strings(items: Vec<String>) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for item in items {
        if seen.insert(item.clone()) {
            out.push(item);
        }
    }
    out
}

// The state tables change only under the write lock, so a read may cache them.
fn read_cached(id: u64, name: &str, table: Arc<dyn MemoryTable>) -> Arc<dyn MemoryTable> {
    Arc::new(ReadCachedTable::new(table, id, name))
}

/// Every collection's schema by name, with `spans` as the span collection's schema.
pub(crate) fn collection_schemas(
    embedding_dim: usize,
    spans: CollectionSchema,
) -> BTreeMap<String, CollectionSchema> {
    let mut schemas = active_collection_schemas(embedding_dim)
        .into_iter()
        .map(|schema| (schema.name.clone(), schema))
        .collect::<BTreeMap<_, _>>();
    schemas.insert(SPANS_COLLECTION.to_string(), spans);
    schemas
}

pub(crate) fn take_schema(
    schemas: &mut BTreeMap<String, CollectionSchema>,
    name: &str,
) -> ZResult<CollectionSchema> {
    schemas
        .remove(name)
        .ok_or_else(|| Status::internal(format!("no schema for collection {name}")))
}

fn output_fields(fields: &[&str]) -> Vec<String> {
    fields.iter().map(|field| (*field).to_string()).collect()
}

/// Scans `collection` once per chunk of `values`, each chunk small enough for one filter.
fn scan_in_chunks<T>(
    collection: &dyn MemoryTable,
    values: &[T],
    limit: usize,
    fields: &[&str],
    chunk_filter: impl Fn(&[T]) -> String,
) -> ZResult<Vec<Arc<Doc>>> {
    let mut docs = Vec::new();
    for chunk in values.chunks(MAX_CONTAINS_FILTER_VALUES) {
        let query = VectorQuery::new("", Vec::new(), limit)
            .with_filter(chunk_filter(chunk))
            .with_output_fields(output_fields(fields));
        docs.extend(collection.scan_filter_only(query)?);
    }
    Ok(docs)
}

/// `scope`'s filter AND `field` equal to one of `values`.
fn scoped_field_in(scope: &MemoryScope, field: &str, values: &[&str]) -> String {
    format!(
        "{} AND ({})",
        scope_filter(scope),
        sql_or_eq_list(field, values.iter().copied()),
    )
}

fn scope_filter(scope: &MemoryScope) -> String {
    let mut clauses = vec![format!("space_id = '{}'", sql_escape(&scope.space_id))];
    for (field, value) in [
        ("tenant_id", &scope.tenant_id),
        ("user_id", &scope.user_id),
        ("agent_id", &scope.agent_id),
        ("project_id", &scope.project_id),
        ("thread_id", &scope.thread_id),
    ] {
        if let Some(value) = value {
            clauses.push(format!("{field} = '{}'", sql_escape(value)));
        }
    }
    clauses.join(" AND ")
}

fn sql_string_list<'a>(items: impl Iterator<Item = &'a str>) -> String {
    items
        .map(|item| format!("'{}'", sql_escape(item)))
        .collect::<Vec<_>>()
        .join(", ")
}

fn sql_or_eq_list<'a>(field: &str, items: impl Iterator<Item = &'a str>) -> String {
    items
        .map(|item| format!("{field} = '{}'", sql_escape(item)))
        .collect::<Vec<_>>()
        .join(" OR ")
}

// The filter dialect escapes quotes with a backslash and rejects SQL quote doubling.
// Backslashes that reach a quote or the end of the literal are read in pairs by the filter parser.
fn sql_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut run = 0;
    for c in value.chars() {
        match c {
            '\\' => run += 1,
            '\'' => {
                out.push_str(&"\\".repeat(2 * run + 1));
                out.push('\'');
                run = 0;
            }
            _ => {
                out.push_str(&"\\".repeat(run));
                out.push(c);
                run = 0;
            }
        }
    }
    out.push_str(&"\\".repeat(2 * run));
    out
}

fn required_doc_string(doc: &Doc, field: &str) -> ZResult<String> {
    doc.fields
        .get(field)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| Status::invalid_argument(format!("memory term row missing `{field}`")))
}

fn required_doc_i64(doc: &Doc, field: &str) -> ZResult<i64> {
    doc.fields
        .get(field)
        .and_then(Value::as_i64)
        .ok_or_else(|| Status::invalid_argument(format!("memory term row missing `{field}`")))
}

pub(crate) fn json_error(err: serde_json::Error) -> Status {
    Status::invalid_argument(format!("memory record JSON encoding failed: {err}"))
}

#[cfg(test)]
mod tests;

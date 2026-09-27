use serde::{Deserialize, Serialize};

use crate::retrieval::SpanSearchHit;
use crate::state::BiTemporalQuery;

/// Caller-owned ID. Use UUIDv7/ULID or another sortable unique string.
pub type MemoryId = String;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MemoryScope {
    pub space_id: String,
    pub tenant_id: Option<String>,
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
    pub project_id: Option<String>,
    pub thread_id: Option<String>,
}

impl MemoryScope {
    pub fn new(space_id: impl Into<String>) -> Self {
        Self {
            space_id: space_id.into(),
            tenant_id: None,
            user_id: None,
            agent_id: None,
            project_id: None,
            thread_id: None,
        }
    }

    pub fn matches_filter(&self, filter: &MemoryScope) -> bool {
        self.space_id == filter.space_id
            && optional_scope_matches(&self.tenant_id, &filter.tenant_id)
            && optional_scope_matches(&self.user_id, &filter.user_id)
            && optional_scope_matches(&self.agent_id, &filter.agent_id)
            && optional_scope_matches(&self.project_id, &filter.project_id)
            && optional_scope_matches(&self.thread_id, &filter.thread_id)
    }
}

fn optional_scope_matches(value: &Option<String>, filter: &Option<String>) -> bool {
    filter
        .as_ref()
        .is_none_or(|expected| value.as_ref() == Some(expected))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryStatus {
    Active,
    Superseded,
    Retracted,
    Tombstoned,
    Expired,
    Redacted,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CanonicalSlot {
    pub subject_key: String,
    pub predicate_key: String,
}

impl CanonicalSlot {
    pub fn new(subject: &str, predicate: &str) -> Self {
        Self {
            subject_key: canonical_slot_part(subject),
            predicate_key: canonical_slot_part(predicate),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleAction {
    DeriveValue,
    MarkUnsupported,
}

impl RuleAction {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DeriveValue => "derive_value",
            Self::MarkUnsupported => "mark_unsupported",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleActivation {
    ContinuousProjection,
    #[default]
    OnChange,
}

impl RuleActivation {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ContinuousProjection => "continuous_projection",
            Self::OnChange => "on_change",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleTargetMatch {
    ExactSlot,
    AnyActiveSlotForSubject,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleBindingStatus {
    Pending,
    Bound,
    Ambiguous,
    Invalid,
}

impl RuleBindingStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Bound => "bound",
            Self::Ambiguous => "ambiguous",
            Self::Invalid => "invalid",
        }
    }
}

impl RuleTargetMatch {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ExactSlot => "exact_slot",
            Self::AnyActiveSlotForSubject => "any_active_slot_for_subject",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuleInput {
    pub id: Option<MemoryId>,
    pub scope: MemoryScope,
    pub status: MemoryStatus,
    pub visibility: Visibility,
    pub policy_tags: Vec<String>,
    pub trigger_subject_key: String,
    pub trigger_predicate_key: String,
    pub target_subject_key: String,
    pub target_predicate_key: String,
    #[serde(default)]
    pub trigger_slot_id: Option<MemoryId>,
    #[serde(default)]
    pub target_slot_id: Option<MemoryId>,
    pub trigger_subject: Option<String>,
    pub trigger_predicate: Option<String>,
    pub target_subject: Option<String>,
    pub target_predicate: Option<String>,
    pub target_match: RuleTargetMatch,
    #[serde(default)]
    pub activation: RuleActivation,
    pub action: RuleAction,
    pub value_template: Option<String>,
    pub value: Option<String>,
    pub source_span_ids: Vec<MemoryId>,
    pub source_episode_ids: Vec<MemoryId>,
    pub valid_from_ms: Option<i64>,
    pub valid_to_ms: Option<i64>,
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuleRecord {
    pub id: MemoryId,
    pub scope: MemoryScope,
    pub status: MemoryStatus,
    pub visibility: Visibility,
    pub policy_tags: Vec<String>,
    pub trigger_subject_key: String,
    pub trigger_predicate_key: String,
    pub target_subject_key: String,
    pub target_predicate_key: String,
    pub trigger_subject_entity_id: Option<MemoryId>,
    pub target_subject_entity_id: Option<MemoryId>,
    pub trigger_slot_id: Option<MemoryId>,
    pub target_slot_id: Option<MemoryId>,
    pub trigger_binding_status: RuleBindingStatus,
    pub target_binding_status: RuleBindingStatus,
    pub trigger_subject: Option<String>,
    pub trigger_predicate: Option<String>,
    pub target_subject: Option<String>,
    pub target_predicate: Option<String>,
    pub target_match: RuleTargetMatch,
    #[serde(default)]
    pub activation: RuleActivation,
    pub action: RuleAction,
    pub value_template: Option<String>,
    pub value: Option<String>,
    pub source_span_ids: Vec<MemoryId>,
    pub source_episode_ids: Vec<MemoryId>,
    #[serde(default)]
    pub source_sequence_no: Option<i64>,
    pub valid_from_ms: Option<i64>,
    pub valid_to_ms: Option<i64>,
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedRuleApplication {
    pub rule_id: MemoryId,
    pub trace_id: MemoryId,
    pub trigger_claim_id: MemoryId,
    #[serde(default)]
    pub prior_trigger_claim_id: Option<MemoryId>,
    pub claim: ClaimRecord,
    pub target_subject_key: String,
    pub target_predicate_key: String,
    pub hop: usize,
    pub parent_trace_ids: Vec<MemoryId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleResolutionStatus {
    AppliedDerived,
    AppliedUnsupported,
    MissingTrigger,
    MissingTriggerValue,
    OutsideValidityWindow,
    /// The trigger slot has no earlier value to compare against, so no transition is provable.
    NoPriorValue,
    /// The trigger was restated with the same value, status, and polarity.
    NoValueChange,
    NotApplicable,
}

impl RuleResolutionStatus {
    pub(crate) fn is_applied(self) -> bool {
        matches!(self, Self::AppliedDerived | Self::AppliedUnsupported)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuleResolutionOutcome {
    pub rule_id: MemoryId,
    pub trigger_slot_id: Option<MemoryId>,
    pub target_slot_id: Option<MemoryId>,
    pub status: RuleResolutionStatus,
    pub resolved_claim_id: Option<MemoryId>,
}

impl MemoryStatus {
    pub fn is_active_for_default_retrieval(self) -> bool {
        matches!(self, MemoryStatus::Active)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            MemoryStatus::Active => "active",
            MemoryStatus::Superseded => "superseded",
            MemoryStatus::Retracted => "retracted",
            MemoryStatus::Tombstoned => "tombstoned",
            MemoryStatus::Expired => "expired",
            MemoryStatus::Redacted => "redacted",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Visibility {
    Private,
    Shared,
    System,
    Tool,
}

impl Visibility {
    pub fn as_str(self) -> &'static str {
        match self {
            Visibility::Private => "private",
            Visibility::Shared => "shared",
            Visibility::System => "system",
            Visibility::Tool => "tool",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    UserMessage,
    AssistantMessage,
    ToolCall,
    ToolResult,
    Import,
    SystemEvent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActorKind {
    User,
    Assistant,
    Tool,
    System,
    External,
}

impl ActorKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
            Self::System => "system",
            Self::External => "external",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    File,
    Url,
    ToolOutput,
    Document,
    Image,
    Code,
    Email,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorrectionOperation {
    Assert,
    Retract,
    Replace,
    Tombstone,
    Restore,
    Forget,
    Merge,
    Split,
    MarkStale,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorrectionTargetMatch {
    #[default]
    ClaimVersions,
    CanonicalSlot,
}

impl CorrectionTargetMatch {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ClaimVersions => "claim_versions",
            Self::CanonicalSlot => "canonical_slot",
        }
    }
}

impl CorrectionOperation {
    pub fn requires_existing_claim_target(self) -> bool {
        matches!(
            self,
            Self::Retract
                | Self::Replace
                | Self::Tombstone
                | Self::Restore
                | Self::Forget
                | Self::MarkStale
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorrectionAuthority {
    User,
    Admin,
    System,
    Policy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceType {
    Episode,
    Artifact,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalFields {
    pub created_at_ms: i64,
    pub ingested_at_ms: i64,
    pub event_time_ms: Option<i64>,
    pub valid_from_ms: Option<i64>,
    pub valid_to_ms: Option<i64>,
}

impl TemporalFields {
    pub fn valid_at(&self, at_ms: i64) -> bool {
        self.valid_from_ms.is_none_or(|from| from <= at_ms)
            && self.valid_to_ms.is_none_or(|to| at_ms < to)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceRef {
    pub source_type: SourceType,
    pub source_id: MemoryId,
    pub span_id: Option<MemoryId>,
    #[serde(default)]
    pub actor: Option<ActorKind>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpisodeRecord {
    pub id: MemoryId,
    pub scope: MemoryScope,
    pub status: MemoryStatus,
    pub visibility: Visibility,
    pub policy_tags: Vec<String>,
    pub source_kind: SourceKind,
    pub actor: ActorKind,
    pub sequence_no: i64,
    pub temporal: TemporalFields,
    pub raw_text: String,
    pub blob_ref: Option<String>,
    pub mime_type: Option<String>,
    pub content_hash: String,
    pub causal_parent_ids: Vec<MemoryId>,
    pub metadata_json: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArtifactRecord {
    pub id: MemoryId,
    pub scope: MemoryScope,
    pub status: MemoryStatus,
    pub visibility: Visibility,
    pub policy_tags: Vec<String>,
    pub artifact_kind: ArtifactKind,
    pub title: String,
    pub uri: Option<String>,
    pub blob_ref: Option<String>,
    pub mime_type: Option<String>,
    pub content_hash: String,
    pub source_created_at_ms: Option<i64>,
    pub source_modified_at_ms: Option<i64>,
    pub created_at_ms: i64,
    pub ingested_at_ms: i64,
    pub valid_from_ms: Option<i64>,
    pub valid_to_ms: Option<i64>,
    pub extracted_text_ref: Option<String>,
    pub metadata_json: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpanRecord {
    pub id: MemoryId,
    pub scope: MemoryScope,
    pub status: MemoryStatus,
    pub visibility: Visibility,
    pub policy_tags: Vec<String>,
    pub source_type: SourceType,
    pub source_id: MemoryId,
    pub byte_start: i64,
    pub byte_end: i64,
    pub char_start: i64,
    pub char_end: i64,
    pub token_start: Option<i64>,
    pub token_end: Option<i64>,
    pub span_index: i64,
    pub text: String,
    pub text_hash: String,
    pub chunker_version: String,
    pub embedding_model: Option<String>,
    pub embedding_version: Option<String>,
    pub lexical_text: String,
    pub created_at_ms: i64,
    pub valid_from_ms: Option<i64>,
    pub valid_to_ms: Option<i64>,
    pub provenance: Vec<ProvenanceRef>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorrectionRecord {
    pub id: MemoryId,
    pub scope: MemoryScope,
    pub status: MemoryStatus,
    pub visibility: Visibility,
    pub policy_tags: Vec<String>,
    pub operation: CorrectionOperation,
    pub target_type: String,
    #[serde(default)]
    pub target_match: CorrectionTargetMatch,
    pub target_ids: Vec<MemoryId>,
    pub target_selector: Option<String>,
    pub target_subject_key: Option<String>,
    pub target_predicate_key: Option<String>,
    pub target_subject_entity_id: Option<MemoryId>,
    pub target_slot_id: Option<MemoryId>,
    pub new_value: Option<String>,
    pub reason: Option<String>,
    pub actor: ActorKind,
    pub authority: CorrectionAuthority,
    pub created_at_ms: i64,
    pub effective_at_ms: i64,
    pub applies_valid_from_ms: Option<i64>,
    pub applies_valid_to_ms: Option<i64>,
    pub cascade_policy: Option<String>,
    pub source_span_ids: Vec<MemoryId>,
    pub source_episode_ids: Vec<MemoryId>,
    #[serde(default)]
    pub source_sequence_no: Option<i64>,
    pub metadata_json: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimKind {
    Preference,
    Fact,
    Event,
    Relationship,
    Constraint,
    TaskState,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimPolarity {
    Affirmative,
    Negative,
    Uncertain,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClaimRecord {
    pub id: MemoryId,
    pub scope: MemoryScope,
    pub status: MemoryStatus,
    pub visibility: Visibility,
    pub policy_tags: Vec<String>,
    pub claim_text: String,
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object_value: Option<String>,
    pub subject_entity_id: Option<MemoryId>,
    pub slot_id: Option<MemoryId>,
    /// Surface identity of this claim inside its canonical slot family. `None` means the claim's
    /// own (subject, predicate) surface is the slot's identity (or an evidence-backed alias of it);
    /// `Some(own_slot_id)` means the claim was bound into the family under a different surface, so
    /// it is a co-current facet that does not supersede other facets.
    #[serde(default)]
    pub slot_facet: Option<MemoryId>,
    pub claim_kind: ClaimKind,
    pub polarity: ClaimPolarity,
    pub source_span_ids: Vec<MemoryId>,
    pub source_episode_ids: Vec<MemoryId>,
    #[serde(default)]
    pub source_sequence_no: Option<i64>,
    pub asserted_by: String,
    pub extractor_version: Option<String>,
    pub confidence: Option<f32>,
    pub observed_at_ms: i64,
    pub valid_from_ms: Option<i64>,
    pub valid_to_ms: Option<i64>,
    pub correction_ids: Vec<MemoryId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SetStateRecord {
    pub id: MemoryId,
    pub scope: MemoryScope,
    pub status: MemoryStatus,
    pub claim_kind: ClaimKind,
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub subject_entity_id: Option<MemoryId>,
    pub slot_id: Option<MemoryId>,
    pub members: Vec<String>,
    pub claim_ids: Vec<MemoryId>,
    pub source_span_ids: Vec<MemoryId>,
    pub source_episode_ids: Vec<MemoryId>,
    pub observed_at_ms: i64,
    pub valid_from_ms: Option<i64>,
    pub valid_to_ms: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SlotHistoryVersion {
    pub state_id: MemoryId,
    pub state_kind: StateRecordKind,
    pub object_value: Option<String>,
    pub members: Vec<String>,
    pub claim_ids: Vec<MemoryId>,
    pub correction_ids: Vec<MemoryId>,
    pub rule_ids: Vec<MemoryId>,
    pub source_span_ids: Vec<MemoryId>,
    pub source_episode_ids: Vec<MemoryId>,
    pub observed_at_ms: i64,
    pub valid_from_ms: Option<i64>,
    pub valid_to_ms: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurrentStateSlotRankings {
    pub vector: Vec<MemoryId>,
    pub lexical: Vec<MemoryId>,
    pub fused: Vec<MemoryId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SlotHistoryRecord {
    pub id: MemoryId,
    pub slot_id: MemoryId,
    pub subject_entity_id: Option<MemoryId>,
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub versions: Vec<SlotHistoryVersion>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateReadView {
    #[default]
    Current,
    Timeline,
    Set,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateReadRole {
    #[default]
    AnswerTarget,
    Supporting,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateReadSelection {
    pub slot_id: MemoryId,
    #[serde(default)]
    pub view: StateReadView,
    #[serde(default)]
    pub role: StateReadRole,
}

impl StateReadSelection {
    pub fn answer_targets(slot_ids: &[MemoryId]) -> Vec<StateReadSelection> {
        slot_ids
            .iter()
            .cloned()
            .map(|slot_id| StateReadSelection {
                slot_id,
                view: StateReadView::Current,
                role: StateReadRole::AnswerTarget,
            })
            .collect()
    }
}

#[derive(Debug, Clone, Default)]
pub struct AnswerReadyStateRequest<'a> {
    pub projection_seed: &'a str,
    pub query_text: &'a str,
    pub evidence_hits: &'a [SpanSearchHit],
    /// Coverage selections. Roles are honored via `state_read_selection_plan`.
    pub selections: &'a [StateReadSelection],
    /// `Some` switches to target-and-coverage semantics: both lists' views are merged, the
    /// targets are the target selections' slots (or every selection when empty), and roles are
    /// ignored.
    pub target_selections: Option<&'a [StateReadSelection]>,
    pub claim_limit: usize,
    pub entity_limit: usize,
    pub temporal: BiTemporalQuery,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct StateRecordScan {
    pub limit: usize,
    pub temporal: BiTemporalQuery,
}

#[derive(Debug, Clone, Default)]
pub struct HybridSpanSearch<'a> {
    pub query_embedding: Vec<f32>,
    pub query_text: &'a str,
    pub k: usize,
    pub scan_limit: usize,
    pub at_ms: Option<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateRecordKind {
    Current,
    Set,
    Tombstone,
    Unsupported,
    Derived,
    Rule,
}

impl StateRecordKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Current => "current",
            Self::Set => "set",
            Self::Tombstone => "tombstone",
            Self::Unsupported => "unsupported",
            Self::Derived => "derived",
            Self::Rule => "rule",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateRecord {
    pub id: MemoryId,
    pub state_key: MemoryId,
    pub scope: MemoryScope,
    pub status: MemoryStatus,
    pub visibility: Visibility,
    pub policy_tags: Vec<String>,
    pub state_kind: StateRecordKind,
    pub claim_kind: ClaimKind,
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object_value: Option<String>,
    pub subject_entity_id: Option<MemoryId>,
    pub slot_id: Option<MemoryId>,
    #[serde(default)]
    pub slot_facet: Option<MemoryId>,
    pub state_text: String,
    pub members: Vec<String>,
    pub claim_ids: Vec<MemoryId>,
    pub correction_ids: Vec<MemoryId>,
    pub rule_ids: Vec<MemoryId>,
    pub source_span_ids: Vec<MemoryId>,
    pub source_episode_ids: Vec<MemoryId>,
    #[serde(default)]
    pub source_sequence_no: Option<i64>,
    pub observed_at_ms: i64,
    pub valid_from_ms: Option<i64>,
    pub valid_to_ms: Option<i64>,
    pub projected_at_ms: i64,
    pub recorded_at_ms: i64,
    pub superseded_at_ms: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DependencyTraceRecord {
    pub id: MemoryId,
    pub trace_key: MemoryId,
    pub scope: MemoryScope,
    pub status: MemoryStatus,
    pub visibility: Visibility,
    pub policy_tags: Vec<String>,
    pub rule_id: MemoryId,
    pub trigger_claim_id: MemoryId,
    #[serde(default)]
    pub prior_trigger_claim_id: Option<MemoryId>,
    pub derived_claim_id: MemoryId,
    pub target_subject_key: String,
    pub target_predicate_key: String,
    pub target_subject_entity_id: Option<MemoryId>,
    pub target_slot_id: Option<MemoryId>,
    pub target_subject: Option<String>,
    pub target_predicate: Option<String>,
    pub state_kind: StateRecordKind,
    pub hop: usize,
    pub parent_trace_ids: Vec<MemoryId>,
    pub source_span_ids: Vec<MemoryId>,
    pub source_episode_ids: Vec<MemoryId>,
    pub observed_at_ms: i64,
    pub valid_from_ms: Option<i64>,
    pub valid_to_ms: Option<i64>,
    pub projected_at_ms: i64,
    pub recorded_at_ms: i64,
    pub superseded_at_ms: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CanonicalSlotRecord {
    pub id: MemoryId,
    pub slot_key: MemoryId,
    pub scope: MemoryScope,
    pub status: MemoryStatus,
    pub visibility: Visibility,
    pub policy_tags: Vec<String>,
    pub subject_key: String,
    pub predicate_key: String,
    pub subject_entity_id: Option<MemoryId>,
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub source_claim_ids: Vec<MemoryId>,
    pub source_rule_ids: Vec<MemoryId>,
    pub source_entity_ids: Vec<MemoryId>,
    pub observed_at_ms: i64,
    pub valid_from_ms: Option<i64>,
    pub valid_to_ms: Option<i64>,
    pub projected_at_ms: i64,
    pub recorded_at_ms: i64,
    pub superseded_at_ms: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotSurfaceAlias {
    pub subject: String,
    pub predicate: String,
}

/// A canonical slot an active `on_change` rule watches, plus the directly asserted state a later
/// transition would supersede.
///
/// `on_change` rules apply to every proven transition, so a slot stays repairable after it has
/// already changed once. The current state is the comparison baseline a proposed transition must
/// differ from, so a slot without directly asserted state is not repairable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepairableTriggerSlot {
    pub slot_id: MemoryId,
    pub subject_key: String,
    pub predicate_key: String,
    pub subject_entity_id: Option<MemoryId>,
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub current_value: Option<String>,
    pub current_state_text: String,
    pub current_valid_from_ms: Option<i64>,
    pub current_observed_at_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SlotTemporalState {
    pub state_kind: StateRecordKind,
    pub state_text: String,
    pub observed_at_ms: i64,
    pub valid_from_ms: Option<i64>,
    pub valid_to_ms: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CanonicalSlotBindingContext {
    pub slot_id: MemoryId,
    pub subject: String,
    pub predicate: String,
    pub subject_entity_id: Option<MemoryId>,
    pub entity_type: Option<String>,
    pub canonical_entity_name: Option<String>,
    pub entity_aliases: Vec<String>,
    pub slot_aliases: Vec<SlotSurfaceAlias>,
    pub temporal_state: Vec<SlotTemporalState>,
    pub dependency_trigger_slot_ids: Vec<MemoryId>,
    pub dependency_target_slot_ids: Vec<MemoryId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SlotAliasInput {
    pub id: Option<MemoryId>,
    pub scope: MemoryScope,
    pub visibility: Visibility,
    pub policy_tags: Vec<String>,
    pub alias_subject: String,
    pub alias_predicate: String,
    pub canonical_slot_id: Option<MemoryId>,
    pub target_claim_id: Option<MemoryId>,
    pub source_claim_ids: Vec<MemoryId>,
    pub valid_from_ms: Option<i64>,
    pub valid_to_ms: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SlotAliasRecord {
    pub id: MemoryId,
    pub scope: MemoryScope,
    pub status: MemoryStatus,
    pub visibility: Visibility,
    pub policy_tags: Vec<String>,
    pub alias_subject_key: String,
    pub alias_predicate_key: String,
    pub alias_subject_entity_id: Option<MemoryId>,
    pub canonical_slot_id: MemoryId,
    pub source_claim_ids: Vec<MemoryId>,
    pub valid_from_ms: Option<i64>,
    pub valid_to_ms: Option<i64>,
    pub recorded_at_ms: i64,
    pub superseded_at_ms: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfileRecord {
    pub id: MemoryId,
    pub scope: MemoryScope,
    pub status: MemoryStatus,
    pub visibility: Visibility,
    pub policy_tags: Vec<String>,
    pub subject_id: Option<String>,
    pub profile_key: String,
    pub profile_text: String,
    pub value_json: Option<String>,
    pub evidence_claim_ids: Vec<MemoryId>,
    pub source_span_ids: Vec<MemoryId>,
    pub generated_at_ms: i64,
    pub generator_version: String,
    pub valid_from_ms: Option<i64>,
    pub valid_to_ms: Option<i64>,
    pub correction_watermark: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileInput {
    pub id: Option<MemoryId>,
    pub scope: MemoryScope,
    pub visibility: Visibility,
    pub policy_tags: Vec<String>,
    pub subject_id: Option<String>,
    pub profile_key: String,
    pub profile_text: String,
    pub value_json: Option<String>,
    pub evidence_claim_ids: Vec<MemoryId>,
    pub source_span_ids: Vec<MemoryId>,
    pub generated_at_ms: i64,
    pub generator_version: String,
    pub valid_from_ms: Option<i64>,
    pub valid_to_ms: Option<i64>,
    pub correction_watermark: i64,
}

impl ProfileInput {
    /// The active record this input creates under `id`.
    pub(crate) fn into_record(self, id: MemoryId) -> ProfileRecord {
        ProfileRecord {
            id,
            scope: self.scope,
            status: MemoryStatus::Active,
            visibility: self.visibility,
            policy_tags: self.policy_tags,
            subject_id: self.subject_id,
            profile_key: self.profile_key,
            profile_text: self.profile_text,
            value_json: self.value_json,
            evidence_claim_ids: self.evidence_claim_ids,
            source_span_ids: self.source_span_ids,
            generated_at_ms: self.generated_at_ms,
            generator_version: self.generator_version,
            valid_from_ms: self.valid_from_ms,
            valid_to_ms: self.valid_to_ms,
            correction_watermark: self.correction_watermark,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityRecord {
    pub id: MemoryId,
    pub scope: MemoryScope,
    pub status: MemoryStatus,
    pub visibility: Visibility,
    pub policy_tags: Vec<String>,
    pub entity_type: String,
    pub canonical_name: String,
    pub aliases: Vec<String>,
    pub source_claim_ids: Vec<MemoryId>,
    pub merge_parent_ids: Vec<MemoryId>,
    pub split_from_id: Option<MemoryId>,
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityInput {
    pub id: Option<MemoryId>,
    pub scope: MemoryScope,
    pub visibility: Visibility,
    pub policy_tags: Vec<String>,
    pub entity_type: String,
    pub canonical_name: String,
    pub aliases: Vec<String>,
    pub source_claim_ids: Vec<MemoryId>,
    pub merge_parent_ids: Vec<MemoryId>,
    pub split_from_id: Option<MemoryId>,
    pub confidence: Option<f32>,
}

impl EntityInput {
    /// The active record this input creates under `id`.
    pub(crate) fn into_record(self, id: MemoryId) -> EntityRecord {
        EntityRecord {
            id,
            scope: self.scope,
            status: MemoryStatus::Active,
            visibility: self.visibility,
            policy_tags: self.policy_tags,
            entity_type: self.entity_type,
            canonical_name: self.canonical_name,
            aliases: self.aliases,
            source_claim_ids: self.source_claim_ids,
            merge_parent_ids: self.merge_parent_ids,
            split_from_id: self.split_from_id,
            confidence: self.confidence,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityAliasRecord {
    pub id: MemoryId,
    pub scope: MemoryScope,
    pub status: MemoryStatus,
    pub visibility: Visibility,
    pub policy_tags: Vec<String>,
    pub alias_key: String,
    pub entity_id: MemoryId,
    pub entity_type: String,
    pub source_claim_ids: Vec<MemoryId>,
    pub recorded_at_ms: i64,
    pub superseded_at_ms: Option<i64>,
}

impl EntityRecord {
    /// Folds an earlier version of this entity into it: aliases, source claims, and merge
    /// parents accumulate and the higher confidence wins. Names are the caller's choice.
    pub(crate) fn absorb_provenance(&mut self, existing: EntityRecord) {
        self.aliases.extend(existing.aliases);
        self.aliases.sort();
        self.aliases.dedup();
        self.source_claim_ids.extend(existing.source_claim_ids);
        self.source_claim_ids.sort();
        self.source_claim_ids.dedup();
        self.merge_parent_ids.extend(existing.merge_parent_ids);
        self.merge_parent_ids.sort();
        self.merge_parent_ids.dedup();
        self.confidence = match (self.confidence, existing.confidence) {
            (Some(left), Some(right)) => Some(left.max(right)),
            (left, right) => left.or(right),
        };
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EdgeRecord {
    pub id: MemoryId,
    pub scope: MemoryScope,
    pub status: MemoryStatus,
    pub visibility: Visibility,
    pub policy_tags: Vec<String>,
    pub src_entity_id: MemoryId,
    pub dst_entity_id: MemoryId,
    pub relation_type: String,
    pub claim_id: Option<MemoryId>,
    pub source_span_ids: Vec<MemoryId>,
    pub valid_from_ms: Option<i64>,
    pub valid_to_ms: Option<i64>,
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeInput {
    pub id: Option<MemoryId>,
    pub scope: MemoryScope,
    pub visibility: Visibility,
    pub policy_tags: Vec<String>,
    pub src_entity_id: MemoryId,
    pub dst_entity_id: MemoryId,
    pub relation_type: String,
    pub claim_id: Option<MemoryId>,
    pub source_span_ids: Vec<MemoryId>,
    pub valid_from_ms: Option<i64>,
    pub valid_to_ms: Option<i64>,
    pub confidence: Option<f32>,
}

impl EdgeInput {
    /// The active record this input creates under `id`.
    pub(crate) fn into_record(self, id: MemoryId) -> EdgeRecord {
        EdgeRecord {
            id,
            scope: self.scope,
            status: MemoryStatus::Active,
            visibility: self.visibility,
            policy_tags: self.policy_tags,
            src_entity_id: self.src_entity_id,
            dst_entity_id: self.dst_entity_id,
            relation_type: self.relation_type,
            claim_id: self.claim_id,
            source_span_ids: self.source_span_ids,
            valid_from_ms: self.valid_from_ms,
            valid_to_ms: self.valid_to_ms,
            confidence: self.confidence,
        }
    }
}

pub fn active_at(status: MemoryStatus, temporal: &TemporalFields, at_ms: i64) -> bool {
    status.is_active_for_default_retrieval() && temporal.valid_at(at_ms)
}

pub fn canonical_slot_part(value: &str) -> String {
    let mut out = String::new();
    let mut needs_space = false;
    for ch in value.trim().chars().flat_map(char::to_lowercase) {
        if ch.is_alphanumeric() {
            if needs_space && !out.is_empty() {
                out.push(' ');
            }
            out.push(ch);
            needs_space = false;
        } else {
            needs_space = !out.is_empty();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_retrieval_only_accepts_active_status() {
        assert!(MemoryStatus::Active.is_active_for_default_retrieval());
        assert!(!MemoryStatus::Superseded.is_active_for_default_retrieval());
        assert!(!MemoryStatus::Tombstoned.is_active_for_default_retrieval());
        assert!(!MemoryStatus::Redacted.is_active_for_default_retrieval());
    }

    #[test]
    fn temporal_validity_is_inclusive_exclusive() {
        let temporal = TemporalFields {
            created_at_ms: 0,
            ingested_at_ms: 0,
            event_time_ms: None,
            valid_from_ms: Some(10),
            valid_to_ms: Some(20),
        };
        assert!(!temporal.valid_at(9));
        assert!(temporal.valid_at(10));
        assert!(temporal.valid_at(19));
        assert!(!temporal.valid_at(20));
    }

    #[test]
    fn active_filter_combines_status_and_time() {
        let temporal = TemporalFields {
            created_at_ms: 0,
            ingested_at_ms: 0,
            event_time_ms: None,
            valid_from_ms: Some(10),
            valid_to_ms: None,
        };
        assert!(active_at(MemoryStatus::Active, &temporal, 10));
        assert!(!active_at(MemoryStatus::Superseded, &temporal, 10));
        assert!(!active_at(MemoryStatus::Active, &temporal, 9));
    }

    #[test]
    fn scope_filter_matches_only_specified_dimensions() {
        let mut scope = MemoryScope::new("space");
        scope.user_id = Some("user_1".to_string());
        scope.project_id = Some("project_1".to_string());

        assert!(scope.matches_filter(&MemoryScope::new("space")));

        let mut matching = MemoryScope::new("space");
        matching.user_id = Some("user_1".to_string());
        assert!(scope.matches_filter(&matching));

        let mut other_user = MemoryScope::new("space");
        other_user.user_id = Some("user_2".to_string());
        assert!(!scope.matches_filter(&other_user));
        assert!(!scope.matches_filter(&MemoryScope::new("other")));
    }
}

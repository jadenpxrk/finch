use serde::{Deserialize, Serialize};

use crate::retrieval::SpanSearchHit;
use crate::state::BiTemporalQuery;

/// Caller-owned ID. Use UUIDv7/ULID or another sortable unique string.
pub type MemoryId = String;

/// Ids that set which memory a read or a write uses.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MemoryScope {
    /// Id of the memory space. Every record has one.
    pub space_id: String,
    /// Id of the tenant. `None` in a read filter matches any tenant.
    pub tenant_id: Option<String>,
    /// Id of the user. `None` in a read filter matches any user.
    pub user_id: Option<String>,
    /// Id of the agent. `None` in a read filter matches any agent.
    pub agent_id: Option<String>,
    /// Id of the project. `None` in a read filter matches any project.
    pub project_id: Option<String>,
    /// Id of the conversation thread. `None` in a read filter matches any thread.
    pub thread_id: Option<String>,
}

impl MemoryScope {
    /// Returns a scope for `space_id` with no other ids.
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

    pub(crate) fn matches_filter(&self, filter: &MemoryScope) -> bool {
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

/// Lifecycle status of a record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryStatus {
    /// Current and visible to default reads.
    Active,
    /// A newer version replaces it.
    Superseded,
    /// A correction withdrew it.
    Retracted,
    /// A correction deleted it.
    Tombstoned,
    /// Its validity window ended.
    Expired,
    /// Its content was removed.
    Redacted,
}

/// Normalized subject and predicate keys that identify a slot.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CanonicalSlot {
    /// Normalized subject.
    pub subject_key: String,
    /// Normalized predicate.
    pub predicate_key: String,
}

impl CanonicalSlot {
    /// Normalizes the subject and predicate into slot keys.
    pub fn new(subject: &str, predicate: &str) -> Self {
        Self {
            subject_key: canonical_slot_part(subject),
            predicate_key: canonical_slot_part(predicate),
        }
    }
}

/// What a rule does to its target slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleAction {
    /// Derives a new value for the target slot.
    DeriveValue,
    /// Marks the target slot value as unsupported.
    MarkUnsupported,
}

impl RuleAction {
    /// Returns the snake_case name of the value.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DeriveValue => "derive_value",
            Self::MarkUnsupported => "mark_unsupported",
        }
    }
}

/// When a rule applies.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleActivation {
    /// Applies while the trigger state holds.
    ContinuousProjection,
    /// Applies when the trigger slot changes value.
    #[default]
    OnChange,
}

impl RuleActivation {
    /// Returns the snake_case name of the value.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ContinuousProjection => "continuous_projection",
            Self::OnChange => "on_change",
        }
    }
}

/// How a rule finds its target slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleTargetMatch {
    /// Targets the one slot the rule names.
    ExactSlot,
    /// Targets every active slot of the target subject.
    AnyActiveSlotForSubject,
}

/// Whether a rule endpoint binds to one slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleBindingStatus {
    /// Not bound yet.
    Pending,
    /// Bound to exactly one slot.
    Bound,
    /// Matches more than one slot.
    Ambiguous,
    /// Cannot bind to a slot.
    Invalid,
}

impl RuleBindingStatus {
    /// Returns the snake_case name of the value.
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
    /// Returns the snake_case name of the value.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ExactSlot => "exact_slot",
            Self::AnyActiveSlotForSubject => "any_active_slot_for_subject",
        }
    }
}

/// Input for one dependency rule.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuleInput {
    /// Id for the rule. `None` makes the store generate one.
    pub id: Option<MemoryId>,
    /// Memory scope that owns the record.
    pub scope: MemoryScope,
    /// Lifecycle status of the rule.
    pub status: MemoryStatus,
    /// Visibility label of the record.
    pub visibility: Visibility,
    /// Caller-defined policy labels that stay with the record.
    pub policy_tags: Vec<String>,
    /// Normalized subject key of the trigger slot.
    pub trigger_subject_key: String,
    /// Normalized predicate key of the trigger slot.
    pub trigger_predicate_key: String,
    /// Normalized subject key of the target slot.
    pub target_subject_key: String,
    /// Normalized predicate key of the target slot.
    pub target_predicate_key: String,
    /// Id of the trigger slot. `None` means the trigger has no bound slot.
    #[serde(default)]
    pub trigger_slot_id: Option<MemoryId>,
    /// Id of the target slot. `None` means the target has no bound slot.
    #[serde(default)]
    pub target_slot_id: Option<MemoryId>,
    /// Trigger subject text as the evidence states it.
    pub trigger_subject: Option<String>,
    /// Trigger predicate text as the evidence states it.
    pub trigger_predicate: Option<String>,
    /// Target subject text as the evidence states it.
    pub target_subject: Option<String>,
    /// Target predicate text as the evidence states it.
    pub target_predicate: Option<String>,
    /// How the rule finds its target slots.
    pub target_match: RuleTargetMatch,
    /// When the rule applies.
    #[serde(default)]
    pub activation: RuleActivation,
    /// What the rule does to the target slot.
    pub action: RuleAction,
    /// Template for the derived value. `None` means the rule uses `value`.
    pub value_template: Option<String>,
    /// Fixed value for a derived claim. `None` means the rule uses `value_template`.
    pub value: Option<String>,
    /// Ids of the evidence spans that support the record.
    pub source_span_ids: Vec<MemoryId>,
    /// Ids of the episodes that support the record.
    pub source_episode_ids: Vec<MemoryId>,
    /// Start of the valid-time interval in Unix ms, inclusive. `None` means no lower bound.
    pub valid_from_ms: Option<i64>,
    /// End of the valid-time interval in Unix ms, exclusive. `None` means no upper bound.
    pub valid_to_ms: Option<i64>,
    /// Confidence score. `None` means the source gave no score.
    pub confidence: Option<f32>,
}

/// A dependency rule: a change to the trigger slot acts on the target slot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuleRecord {
    /// Unique id of the record.
    pub id: MemoryId,
    /// Memory scope that owns the record.
    pub scope: MemoryScope,
    /// Lifecycle status of the record.
    pub status: MemoryStatus,
    /// Visibility label of the record.
    pub visibility: Visibility,
    /// Caller-defined policy labels that stay with the record.
    pub policy_tags: Vec<String>,
    /// Normalized subject key of the trigger slot.
    pub trigger_subject_key: String,
    /// Normalized predicate key of the trigger slot.
    pub trigger_predicate_key: String,
    /// Normalized subject key of the target slot.
    pub target_subject_key: String,
    /// Normalized predicate key of the target slot.
    pub target_predicate_key: String,
    /// Id of the entity of the trigger subject. `None` means no entity matches.
    pub trigger_subject_entity_id: Option<MemoryId>,
    /// Id of the entity of the target subject. `None` means no entity matches.
    pub target_subject_entity_id: Option<MemoryId>,
    /// Id of the trigger slot. `None` means the trigger has no bound slot.
    pub trigger_slot_id: Option<MemoryId>,
    /// Id of the target slot. `None` means the target has no bound slot.
    pub target_slot_id: Option<MemoryId>,
    /// Whether the trigger binds to one slot.
    pub trigger_binding_status: RuleBindingStatus,
    /// Whether the target binds to one slot.
    pub target_binding_status: RuleBindingStatus,
    /// Trigger subject text as the evidence states it.
    pub trigger_subject: Option<String>,
    /// Trigger predicate text as the evidence states it.
    pub trigger_predicate: Option<String>,
    /// Target subject text as the evidence states it.
    pub target_subject: Option<String>,
    /// Target predicate text as the evidence states it.
    pub target_predicate: Option<String>,
    /// How the rule finds its target slots.
    pub target_match: RuleTargetMatch,
    /// When the rule applies.
    #[serde(default)]
    pub activation: RuleActivation,
    /// What the rule does to the target slot.
    pub action: RuleAction,
    /// Template for the derived value. `None` means the rule uses `value`.
    pub value_template: Option<String>,
    /// Fixed value for a derived claim. `None` means the rule uses `value_template`.
    pub value: Option<String>,
    /// Ids of the evidence spans that support the record.
    pub source_span_ids: Vec<MemoryId>,
    /// Ids of the episodes that support the record.
    pub source_episode_ids: Vec<MemoryId>,
    /// Episode sequence number of the source evidence. `None` means unknown.
    #[serde(default)]
    pub source_sequence_no: Option<i64>,
    /// Start of the valid-time interval in Unix ms, inclusive. `None` means no lower bound.
    pub valid_from_ms: Option<i64>,
    /// End of the valid-time interval in Unix ms, exclusive. `None` means no upper bound.
    pub valid_to_ms: Option<i64>,
    /// Confidence score. `None` means the source gave no score.
    pub confidence: Option<f32>,
}

/// One rule application and the claim it derived.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedRuleApplication {
    /// Id of the rule that applied.
    pub rule_id: MemoryId,
    /// Id of the dependency trace of the application.
    pub trace_id: MemoryId,
    /// Id of the trigger claim that activated the rule.
    pub trigger_claim_id: MemoryId,
    /// Id of the earlier trigger claim the change replaced. `None` means no earlier claim.
    #[serde(default)]
    pub prior_trigger_claim_id: Option<MemoryId>,
    /// Claim the rule derived.
    pub claim: ClaimRecord,
    /// Normalized subject key of the target slot.
    pub target_subject_key: String,
    /// Normalized predicate key of the target slot.
    pub target_predicate_key: String,
    /// Position in the chain of rule applications, from 0.
    pub hop: usize,
    /// Ids of the applications that led to this application.
    pub parent_trace_ids: Vec<MemoryId>,
}

/// Result of the resolution of a rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleResolutionStatus {
    /// The rule derived a value.
    AppliedDerived,
    /// The rule marked the target unsupported.
    AppliedUnsupported,
    /// The trigger slot has no state.
    MissingTrigger,
    /// The trigger state has no value.
    MissingTriggerValue,
    /// The trigger is outside the validity window of the rule.
    OutsideValidityWindow,
    /// The trigger slot has no earlier value to compare against, so no transition is provable.
    NoPriorValue,
    /// The trigger was restated with the same value, status, and polarity.
    NoValueChange,
    /// The rule does not apply to the trigger state.
    NotApplicable,
}

impl RuleResolutionStatus {
    pub(crate) fn is_applied(self) -> bool {
        matches!(self, Self::AppliedDerived | Self::AppliedUnsupported)
    }
}

/// Result of the resolution of one rule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuleResolutionOutcome {
    /// Id of the rule.
    pub rule_id: MemoryId,
    /// Id of the trigger slot. `None` means the trigger has no bound slot.
    pub trigger_slot_id: Option<MemoryId>,
    /// Id of the target slot. `None` means the target has no bound slot.
    pub target_slot_id: Option<MemoryId>,
    /// Result of the resolution.
    pub status: RuleResolutionStatus,
    /// Id of the claim the rule produced. `None` means the rule produced no claim.
    pub resolved_claim_id: Option<MemoryId>,
}

impl MemoryStatus {
    #[cfg(test)]
    pub(crate) fn is_active_for_default_retrieval(self) -> bool {
        matches!(self, MemoryStatus::Active)
    }

    /// Returns the snake_case name of the value.
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

/// Visibility label of a record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Visibility {
    /// Private to its owner.
    Private,
    /// Shared with other readers.
    Shared,
    /// Owned by the system.
    System,
    /// Produced by a tool.
    Tool,
}

impl Visibility {
    /// Returns the snake_case name of the value.
    pub fn as_str(self) -> &'static str {
        match self {
            Visibility::Private => "private",
            Visibility::Shared => "shared",
            Visibility::System => "system",
            Visibility::Tool => "tool",
        }
    }
}

/// Kind of event an episode records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    /// A message from the user.
    UserMessage,
    /// A message from the assistant.
    AssistantMessage,
    /// A call to a tool.
    ToolCall,
    /// A result from a tool.
    ToolResult,
    /// Content from a bulk import.
    Import,
    /// An event from the system.
    SystemEvent,
}

/// Who produced content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActorKind {
    /// The user.
    User,
    /// The assistant.
    Assistant,
    /// A tool.
    Tool,
    /// The system.
    System,
    /// A party outside the conversation.
    External,
}

impl ActorKind {
    /// Returns the snake_case name of the value.
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

/// Kind of artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    /// A file.
    File,
    /// A web page.
    Url,
    /// Output of a tool.
    ToolOutput,
    /// A document.
    Document,
    /// An image.
    Image,
    /// Source code.
    Code,
    /// An email.
    Email,
    /// Any other kind.
    Other,
}

/// Change that a correction makes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorrectionOperation {
    /// Asserts a value.
    Assert,
    /// Withdraws the target claims.
    Retract,
    /// Replaces the target value with the new value.
    Replace,
    /// Deletes the target claims.
    Tombstone,
    /// Restores withdrawn or deleted claims.
    Restore,
    /// Removes the targets from memory.
    Forget,
    /// Merges the targets into one record.
    Merge,
    /// Splits a target into separate records.
    Split,
    /// Marks the targets as out of date.
    MarkStale,
}

/// How a correction finds its target claims.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorrectionTargetMatch {
    /// Targets the named claim versions.
    #[default]
    ClaimVersions,
    /// Targets every claim in the slot of the named claims.
    CanonicalSlot,
}

impl CorrectionTargetMatch {
    /// Returns the snake_case name of the value.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ClaimVersions => "claim_versions",
            Self::CanonicalSlot => "canonical_slot",
        }
    }
}

impl CorrectionOperation {
    /// Returns true when the operation acts on claims that exist already.
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

/// Who made a correction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorrectionAuthority {
    /// The user.
    User,
    /// An administrator.
    Admin,
    /// The system.
    System,
    /// A policy.
    Policy,
}

/// Kind of record a span comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceType {
    /// An episode.
    Episode,
    /// An artifact.
    Artifact,
}

/// Time fields of an episode, in Unix ms.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalFields {
    /// Time the record was created, in Unix ms.
    pub created_at_ms: i64,
    /// Time the store ingested the source, in Unix ms.
    pub ingested_at_ms: i64,
    /// Time the event occurred, in Unix ms. `None` means unknown.
    pub event_time_ms: Option<i64>,
    /// Start of the valid-time interval in Unix ms, inclusive. `None` means no lower bound.
    pub valid_from_ms: Option<i64>,
    /// End of the valid-time interval in Unix ms, exclusive. `None` means no upper bound.
    pub valid_to_ms: Option<i64>,
}

impl TemporalFields {
    /// Returns true when `at_ms` is in the valid-time interval.
    pub fn valid_at(&self, at_ms: i64) -> bool {
        self.valid_from_ms.is_none_or(|from| from <= at_ms)
            && self.valid_to_ms.is_none_or(|to| at_ms < to)
    }
}

/// Reference to the source of a span.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceRef {
    /// Kind of record the source is.
    pub source_type: SourceType,
    /// Id of the source episode or artifact.
    pub source_id: MemoryId,
    /// Id of the span in the source. `None` means the whole source.
    pub span_id: Option<MemoryId>,
    /// Who produced the source. `None` means unknown.
    #[serde(default)]
    pub actor: Option<ActorKind>,
}

/// One message or event as raw text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpisodeRecord {
    /// Unique id of the record.
    pub id: MemoryId,
    /// Memory scope that owns the record.
    pub scope: MemoryScope,
    /// Lifecycle status of the record.
    pub status: MemoryStatus,
    /// Visibility label of the record.
    pub visibility: Visibility,
    /// Caller-defined policy labels that stay with the record.
    pub policy_tags: Vec<String>,
    /// Kind of event the episode records.
    pub source_kind: SourceKind,
    /// Who produced the content.
    pub actor: ActorKind,
    /// Position of the episode in its conversation.
    pub sequence_no: i64,
    /// Time fields of the episode.
    pub temporal: TemporalFields,
    /// Full text of the episode.
    pub raw_text: String,
    /// Reference to the raw content in external storage. `None` means the text holds all content.
    pub blob_ref: Option<String>,
    /// MIME type of the raw content.
    pub mime_type: Option<String>,
    /// Hash of the raw content, from `stable_hash_hex`.
    pub content_hash: String,
    /// Ids of the episodes that caused this episode.
    pub causal_parent_ids: Vec<MemoryId>,
    /// Caller metadata as a JSON string.
    pub metadata_json: Option<String>,
}

/// A file, page, or other artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArtifactRecord {
    /// Unique id of the record.
    pub id: MemoryId,
    /// Memory scope that owns the record.
    pub scope: MemoryScope,
    /// Lifecycle status of the record.
    pub status: MemoryStatus,
    /// Visibility label of the record.
    pub visibility: Visibility,
    /// Caller-defined policy labels that stay with the record.
    pub policy_tags: Vec<String>,
    /// Kind of artifact.
    pub artifact_kind: ArtifactKind,
    /// Title of the artifact.
    pub title: String,
    /// Location of the artifact. `None` means no location.
    pub uri: Option<String>,
    /// Reference to the raw content in external storage. `None` means the text holds all content.
    pub blob_ref: Option<String>,
    /// MIME type of the raw content.
    pub mime_type: Option<String>,
    /// Hash of the raw content, from `stable_hash_hex`.
    pub content_hash: String,
    /// Time the source system created the artifact, in Unix ms.
    pub source_created_at_ms: Option<i64>,
    /// Time the source system last changed the artifact, in Unix ms.
    pub source_modified_at_ms: Option<i64>,
    /// Time the record was created, in Unix ms.
    pub created_at_ms: i64,
    /// Time the store ingested the source, in Unix ms.
    pub ingested_at_ms: i64,
    /// Start of the valid-time interval in Unix ms, inclusive. `None` means no lower bound.
    pub valid_from_ms: Option<i64>,
    /// End of the valid-time interval in Unix ms, exclusive. `None` means no upper bound.
    pub valid_to_ms: Option<i64>,
    /// Reference to the text extracted from the artifact.
    pub extracted_text_ref: Option<String>,
    /// Caller metadata as a JSON string.
    pub metadata_json: Option<String>,
}

/// A part of episode or artifact text that retrieval returns.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpanRecord {
    /// Unique id of the record.
    pub id: MemoryId,
    /// Memory scope that owns the record.
    pub scope: MemoryScope,
    /// Lifecycle status of the record.
    pub status: MemoryStatus,
    /// Visibility label of the record.
    pub visibility: Visibility,
    /// Caller-defined policy labels that stay with the record.
    pub policy_tags: Vec<String>,
    /// Kind of record the span comes from.
    pub source_type: SourceType,
    /// Id of the episode or artifact the span comes from.
    pub source_id: MemoryId,
    /// Start of the span in the source text, as a byte offset.
    pub byte_start: i64,
    /// End of the span in the source text, as an exclusive byte offset.
    pub byte_end: i64,
    /// Start of the span in the source text, as a character offset.
    pub char_start: i64,
    /// End of the span in the source text, as an exclusive character offset.
    pub char_end: i64,
    /// Start of the span as a token offset. `None` means unknown.
    pub token_start: Option<i64>,
    /// End of the span as an exclusive token offset. `None` means unknown.
    pub token_end: Option<i64>,
    /// Position of the span in its source, from 0.
    pub span_index: i64,
    /// Text of the span.
    pub text: String,
    /// Hash of the span text, from `stable_hash_hex`.
    pub text_hash: String,
    /// Version label of the chunker that cut the spans.
    pub chunker_version: String,
    /// Name of the embedding model. `None` means the span has no embedding.
    pub embedding_model: Option<String>,
    /// Version of the embedding model. `None` means the span has no embedding.
    pub embedding_version: Option<String>,
    /// Text that keyword search indexes.
    pub lexical_text: String,
    /// Time the record was created, in Unix ms.
    pub created_at_ms: i64,
    /// Start of the valid-time interval in Unix ms, inclusive. `None` means no lower bound.
    pub valid_from_ms: Option<i64>,
    /// End of the valid-time interval in Unix ms, exclusive. `None` means no upper bound.
    pub valid_to_ms: Option<i64>,
    /// Sources of the span.
    pub provenance: Vec<ProvenanceRef>,
}

/// A correction to earlier memory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorrectionRecord {
    /// Unique id of the record.
    pub id: MemoryId,
    /// Memory scope that owns the record.
    pub scope: MemoryScope,
    /// Lifecycle status of the record.
    pub status: MemoryStatus,
    /// Visibility label of the record.
    pub visibility: Visibility,
    /// Caller-defined policy labels that stay with the record.
    pub policy_tags: Vec<String>,
    /// Change the correction makes.
    pub operation: CorrectionOperation,
    /// Kind of record the correction targets, such as `claim`.
    pub target_type: String,
    /// How the correction finds its target claims.
    #[serde(default)]
    pub target_match: CorrectionTargetMatch,
    /// Ids of the records the correction targets.
    pub target_ids: Vec<MemoryId>,
    /// Text that selects target records. `None` means the target ids alone select them.
    pub target_selector: Option<String>,
    /// Normalized subject key of the target slot. `None` means not resolved.
    pub target_subject_key: Option<String>,
    /// Normalized predicate key of the target slot. `None` means not resolved.
    pub target_predicate_key: Option<String>,
    /// Id of the entity of the target subject. `None` means no entity matches.
    pub target_subject_entity_id: Option<MemoryId>,
    /// Id of the target slot. `None` means the target has no bound slot.
    pub target_slot_id: Option<MemoryId>,
    /// Replacement value. `None` means the operation sets no value.
    pub new_value: Option<String>,
    /// Reason for the correction, in free text.
    pub reason: Option<String>,
    /// Who produced the content.
    pub actor: ActorKind,
    /// Who made the correction.
    pub authority: CorrectionAuthority,
    /// Time the record was created, in Unix ms.
    pub created_at_ms: i64,
    /// Time the correction takes effect, in Unix ms.
    pub effective_at_ms: i64,
    /// Start of the valid time the correction changes, in Unix ms. `None` means no lower bound.
    pub applies_valid_from_ms: Option<i64>,
    /// End of the valid time the correction changes, in Unix ms. `None` means no upper bound.
    pub applies_valid_to_ms: Option<i64>,
    /// How the correction spreads to dependent records. `None` means the default policy.
    pub cascade_policy: Option<String>,
    /// Ids of the evidence spans that support the record.
    pub source_span_ids: Vec<MemoryId>,
    /// Ids of the episodes that support the record.
    pub source_episode_ids: Vec<MemoryId>,
    /// Episode sequence number of the source evidence. `None` means unknown.
    #[serde(default)]
    pub source_sequence_no: Option<i64>,
    /// Caller metadata as a JSON string.
    pub metadata_json: Option<String>,
}

/// Category of a claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimKind {
    /// What someone likes or wants.
    Preference,
    /// A fact about a person or the world.
    Fact,
    /// Something that occurred.
    Event,
    /// A link between two entities.
    Relationship,
    /// A limit to obey.
    Constraint,
    /// Progress of a task.
    TaskState,
    /// Any other claim.
    Other,
}

/// Whether a claim asserts, denies, or is not certain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimPolarity {
    /// The claim states that something is true.
    Affirmative,
    /// The claim states that something is false.
    Negative,
    /// The claim is not certain.
    Uncertain,
}

/// One statement about a subject and predicate, with its evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClaimRecord {
    /// Unique id of the record.
    pub id: MemoryId,
    /// Memory scope that owns the record.
    pub scope: MemoryScope,
    /// Lifecycle status of the record.
    pub status: MemoryStatus,
    /// Visibility label of the record.
    pub visibility: Visibility,
    /// Caller-defined policy labels that stay with the record.
    pub policy_tags: Vec<String>,
    /// Statement of the claim in natural language.
    pub claim_text: String,
    /// Subject text as the evidence states it.
    pub subject: Option<String>,
    /// Predicate text as the evidence states it.
    pub predicate: Option<String>,
    /// Value the record states for its slot.
    pub object_value: Option<String>,
    /// Id of the entity that the subject resolves to. `None` means no entity matches.
    pub subject_entity_id: Option<MemoryId>,
    /// Id of the canonical slot of the record. `None` means the record has no slot.
    pub slot_id: Option<MemoryId>,
    /// Surface identity of this claim inside its canonical slot family. `None` means the claim's
    /// own (subject, predicate) surface is the slot's identity (or an evidence-backed alias of it);
    /// `Some(own_slot_id)` means the claim was bound into the family under a different surface, so
    /// it is a co-current facet that does not supersede other facets.
    #[serde(default)]
    pub slot_facet: Option<MemoryId>,
    /// Category of the claim.
    pub claim_kind: ClaimKind,
    /// Whether the claim asserts, denies, or is not certain.
    pub polarity: ClaimPolarity,
    /// Ids of the evidence spans that support the record.
    pub source_span_ids: Vec<MemoryId>,
    /// Ids of the episodes that support the record.
    pub source_episode_ids: Vec<MemoryId>,
    /// Episode sequence number of the source evidence. `None` means unknown.
    #[serde(default)]
    pub source_sequence_no: Option<i64>,
    /// Name of the extractor or person that made the claim.
    pub asserted_by: String,
    /// Version of the extractor that made the claim. `None` means a person made it.
    pub extractor_version: Option<String>,
    /// Confidence score. `None` means the source gave no score.
    pub confidence: Option<f32>,
    /// Time the evidence was observed, in Unix ms.
    pub observed_at_ms: i64,
    /// Start of the valid-time interval in Unix ms, inclusive. `None` means no lower bound.
    pub valid_from_ms: Option<i64>,
    /// End of the valid-time interval in Unix ms, exclusive. `None` means no upper bound.
    pub valid_to_ms: Option<i64>,
    /// Ids of the corrections that changed the record.
    pub correction_ids: Vec<MemoryId>,
}

/// Members of a set-valued slot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SetStateRecord {
    /// Unique id of the set state.
    pub id: MemoryId,
    /// Memory scope that owns the record.
    pub scope: MemoryScope,
    /// Lifecycle status of the record.
    pub status: MemoryStatus,
    /// Category of the member claims.
    pub claim_kind: ClaimKind,
    /// Subject text as the evidence states it.
    pub subject: Option<String>,
    /// Predicate text as the evidence states it.
    pub predicate: Option<String>,
    /// Id of the entity that the subject resolves to. `None` means no entity matches.
    pub subject_entity_id: Option<MemoryId>,
    /// Id of the canonical slot of the record. `None` means the record has no slot.
    pub slot_id: Option<MemoryId>,
    /// Members of a set-valued slot.
    pub members: Vec<String>,
    /// Ids of the claims behind the state.
    pub claim_ids: Vec<MemoryId>,
    /// Ids of the evidence spans that support the record.
    pub source_span_ids: Vec<MemoryId>,
    /// Ids of the episodes that support the record.
    pub source_episode_ids: Vec<MemoryId>,
    /// Time the evidence was observed, in Unix ms.
    pub observed_at_ms: i64,
    /// Start of the valid-time interval in Unix ms, inclusive. `None` means no lower bound.
    pub valid_from_ms: Option<i64>,
    /// End of the valid-time interval in Unix ms, exclusive. `None` means no upper bound.
    pub valid_to_ms: Option<i64>,
}

/// One version in the history of a slot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SlotHistoryVersion {
    /// Id of the state record.
    pub state_id: MemoryId,
    /// Kind of state in this version.
    pub state_kind: StateRecordKind,
    /// Value of the slot in this version.
    pub object_value: Option<String>,
    /// Members of a set-valued slot.
    pub members: Vec<String>,
    /// Ids of the claims behind the state.
    pub claim_ids: Vec<MemoryId>,
    /// Ids of the corrections that changed the record.
    pub correction_ids: Vec<MemoryId>,
    /// Ids of the rules that produced or changed the state.
    pub rule_ids: Vec<MemoryId>,
    /// Ids of the evidence spans that support the record.
    pub source_span_ids: Vec<MemoryId>,
    /// Ids of the episodes that support the record.
    pub source_episode_ids: Vec<MemoryId>,
    /// Time the evidence was observed, in Unix ms.
    pub observed_at_ms: i64,
    /// Start of the valid-time interval in Unix ms, inclusive. `None` means no lower bound.
    pub valid_from_ms: Option<i64>,
    /// End of the valid-time interval in Unix ms, exclusive. `None` means no upper bound.
    pub valid_to_ms: Option<i64>,
}

/// Slot ids ranked by vector, keyword, and fused scores.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurrentStateSlotRankings {
    /// Slot ids ranked by vector similarity.
    pub vector: Vec<MemoryId>,
    /// Slot ids ranked by keyword match.
    pub lexical: Vec<MemoryId>,
    /// Slot ids ranked by reciprocal rank fusion of both lists.
    pub fused: Vec<MemoryId>,
}

/// Every version of one slot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SlotHistoryRecord {
    /// Unique id of the history.
    pub id: MemoryId,
    /// Id of the slot.
    pub slot_id: MemoryId,
    /// Id of the entity that the subject resolves to. `None` means no entity matches.
    pub subject_entity_id: Option<MemoryId>,
    /// Subject text as the evidence states it.
    pub subject: Option<String>,
    /// Predicate text as the evidence states it.
    pub predicate: Option<String>,
    /// Versions of the slot.
    pub versions: Vec<SlotHistoryVersion>,
}

/// How to read a slot.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateReadView {
    /// The current value only.
    #[default]
    Current,
    /// Every version over time.
    Timeline,
    /// Every member of a set-valued slot.
    Set,
}

/// Why the reader wants a slot.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateReadRole {
    /// The answer is about the slot.
    #[default]
    AnswerTarget,
    /// The slot gives context for the answer.
    Supporting,
}

/// A slot to read, with its view and role.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateReadSelection {
    /// Id of the slot to read.
    pub slot_id: MemoryId,
    /// How to read the slot.
    #[serde(default)]
    pub view: StateReadView,
    /// Why the reader wants the slot.
    #[serde(default)]
    pub role: StateReadRole,
}

impl StateReadSelection {
    /// Returns a current-view answer-target selection for each slot id.
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

/// Query and limits for an answer-ready state projection.
#[derive(Debug, Clone, Default)]
pub struct AnswerReadyStateRequest<'a> {
    /// Seed that makes the projection repeatable.
    pub projection_seed: &'a str,
    /// Query text for keyword matching.
    pub query_text: &'a str,
    /// Retrieved spans that point to relevant slots.
    pub evidence_hits: &'a [SpanSearchHit],
    /// Coverage selections. Roles are honored via `state_read_selection_plan`.
    pub selections: &'a [StateReadSelection],
    /// `Some` switches to target-and-coverage semantics: both lists' views are merged, the
    /// targets are the target selections' slots (or every selection when empty), and roles are
    /// ignored.
    pub target_selections: Option<&'a [StateReadSelection]>,
    /// Maximum number of claims in the projection.
    pub claim_limit: usize,
    /// Maximum number of entities in the projection.
    pub entity_limit: usize,
    /// Valid time and transaction time to read at.
    pub temporal: BiTemporalQuery,
}

/// Limit and read time for a scan of state records.
#[derive(Debug, Clone, Copy, Default)]
pub struct StateRecordScan {
    /// Maximum number of records to return.
    pub limit: usize,
    /// Valid time and transaction time to read at.
    pub temporal: BiTemporalQuery,
}

/// Query and limits for a hybrid span search.
#[derive(Debug, Clone, Default)]
pub struct HybridSpanSearch<'a> {
    /// Embedding of the query.
    pub query_embedding: Vec<f32>,
    /// Query text for keyword matching.
    pub query_text: &'a str,
    /// Number of hits to return.
    pub k: usize,
    /// Maximum number of candidates to read.
    pub scan_limit: usize,
    /// Valid time to read at, in Unix ms. `None` means no time filter.
    pub at_ms: Option<i64>,
}

/// Kind of projected state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateRecordKind {
    /// Current single value of a slot.
    Current,
    /// Members of a set-valued slot.
    Set,
    /// A correction deleted the slot value.
    Tombstone,
    /// The slot value has no support.
    Unsupported,
    /// A rule derived the value.
    Derived,
    /// A dependency rule.
    Rule,
}

impl StateRecordKind {
    /// Returns the snake_case name of the value.
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

/// One projected state version of a slot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateRecord {
    /// Unique id of this version.
    pub id: MemoryId,
    /// Id shared by every version of the same state.
    pub state_key: MemoryId,
    /// Memory scope that owns the record.
    pub scope: MemoryScope,
    /// Lifecycle status of the record.
    pub status: MemoryStatus,
    /// Visibility label of the record.
    pub visibility: Visibility,
    /// Caller-defined policy labels that stay with the record.
    pub policy_tags: Vec<String>,
    /// Kind of state the record holds.
    pub state_kind: StateRecordKind,
    /// Category of the claim.
    pub claim_kind: ClaimKind,
    /// Subject text as the evidence states it.
    pub subject: Option<String>,
    /// Predicate text as the evidence states it.
    pub predicate: Option<String>,
    /// Value the record states for its slot.
    pub object_value: Option<String>,
    /// Id of the entity that the subject resolves to. `None` means no entity matches.
    pub subject_entity_id: Option<MemoryId>,
    /// Id of the canonical slot of the record. `None` means the record has no slot.
    pub slot_id: Option<MemoryId>,
    /// Id of the facet surface in the slot family. `None` means the slot identity.
    #[serde(default)]
    pub slot_facet: Option<MemoryId>,
    /// Statement of the state in natural language.
    pub state_text: String,
    /// Members of a set-valued slot.
    pub members: Vec<String>,
    /// Ids of the claims behind the state.
    pub claim_ids: Vec<MemoryId>,
    /// Ids of the corrections that changed the record.
    pub correction_ids: Vec<MemoryId>,
    /// Ids of the rules that produced or changed the state.
    pub rule_ids: Vec<MemoryId>,
    /// Ids of the evidence spans that support the record.
    pub source_span_ids: Vec<MemoryId>,
    /// Ids of the episodes that support the record.
    pub source_episode_ids: Vec<MemoryId>,
    /// Episode sequence number of the source evidence. `None` means unknown.
    #[serde(default)]
    pub source_sequence_no: Option<i64>,
    /// Time the evidence was observed, in Unix ms.
    pub observed_at_ms: i64,
    /// Start of the valid-time interval in Unix ms, inclusive. `None` means no lower bound.
    pub valid_from_ms: Option<i64>,
    /// End of the valid-time interval in Unix ms, exclusive. `None` means no upper bound.
    pub valid_to_ms: Option<i64>,
    /// Time the projection produced the record, in Unix ms.
    pub projected_at_ms: i64,
    /// Transaction time the store wrote the record, in Unix ms.
    pub recorded_at_ms: i64,
    /// Transaction time a newer version replaced the record, in Unix ms. `None` means the record is
    /// current.
    pub superseded_at_ms: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct DependencyTraceRecord {
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

/// A canonical slot: one normalized subject and predicate pair.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CanonicalSlotRecord {
    /// Unique id of this version.
    pub id: MemoryId,
    /// Id shared by every version of the slot.
    pub slot_key: MemoryId,
    /// Memory scope that owns the record.
    pub scope: MemoryScope,
    /// Lifecycle status of the record.
    pub status: MemoryStatus,
    /// Visibility label of the record.
    pub visibility: Visibility,
    /// Caller-defined policy labels that stay with the record.
    pub policy_tags: Vec<String>,
    /// Normalized subject key, from `canonical_slot_part`.
    pub subject_key: String,
    /// Normalized predicate key, from `canonical_slot_part`.
    pub predicate_key: String,
    /// Id of the entity that the subject resolves to. `None` means no entity matches.
    pub subject_entity_id: Option<MemoryId>,
    /// Subject text as the evidence states it.
    pub subject: Option<String>,
    /// Predicate text as the evidence states it.
    pub predicate: Option<String>,
    /// Ids of the claims that support the record.
    pub source_claim_ids: Vec<MemoryId>,
    /// Ids of the rules that name the slot.
    pub source_rule_ids: Vec<MemoryId>,
    /// Ids of the entities the slot subject resolves to.
    pub source_entity_ids: Vec<MemoryId>,
    /// Observation time of the newest source, in Unix ms.
    pub observed_at_ms: i64,
    /// Start of the valid-time interval in Unix ms, inclusive. `None` means no lower bound.
    pub valid_from_ms: Option<i64>,
    /// End of the valid-time interval in Unix ms, exclusive. `None` means no upper bound.
    pub valid_to_ms: Option<i64>,
    /// Time the projection produced the record, in Unix ms.
    pub projected_at_ms: i64,
    /// Transaction time the store wrote the record, in Unix ms.
    pub recorded_at_ms: i64,
    /// Transaction time a newer version replaced the record, in Unix ms. `None` means the record is
    /// current.
    pub superseded_at_ms: Option<i64>,
}

/// Other subject and predicate text that names a slot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotSurfaceAlias {
    /// Alias subject text.
    pub subject: String,
    /// Alias predicate text.
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
    /// Id of the trigger slot.
    pub slot_id: MemoryId,
    /// Normalized subject key, from `canonical_slot_part`.
    pub subject_key: String,
    /// Normalized predicate key, from `canonical_slot_part`.
    pub predicate_key: String,
    /// Id of the entity that the subject resolves to. `None` means no entity matches.
    pub subject_entity_id: Option<MemoryId>,
    /// Subject text as the evidence states it.
    pub subject: Option<String>,
    /// Predicate text as the evidence states it.
    pub predicate: Option<String>,
    /// Current directly asserted value. `None` means the state has no value.
    pub current_value: Option<String>,
    /// Statement of the current state.
    pub current_state_text: String,
    /// Valid-from time of the current state, in Unix ms. `None` means no lower bound.
    pub current_valid_from_ms: Option<i64>,
    /// Observation time of the current state, in Unix ms.
    pub current_observed_at_ms: i64,
}

/// One state of a slot in time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SlotTemporalState {
    /// Kind of state.
    pub state_kind: StateRecordKind,
    /// Statement of the state in natural language.
    pub state_text: String,
    /// Time the evidence was observed, in Unix ms.
    pub observed_at_ms: i64,
    /// Start of the valid-time interval in Unix ms, inclusive. `None` means no lower bound.
    pub valid_from_ms: Option<i64>,
    /// End of the valid-time interval in Unix ms, exclusive. `None` means no upper bound.
    pub valid_to_ms: Option<i64>,
}

/// Data an extractor uses to bind new claims to an existing slot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CanonicalSlotBindingContext {
    /// Id of the slot.
    pub slot_id: MemoryId,
    /// Subject text of the slot.
    pub subject: String,
    /// Predicate text of the slot.
    pub predicate: String,
    /// Id of the entity that the subject resolves to. `None` means no entity matches.
    pub subject_entity_id: Option<MemoryId>,
    /// Category of the subject entity. `None` means no entity matches.
    pub entity_type: Option<String>,
    /// Preferred name of the subject entity. `None` means no entity matches.
    pub canonical_entity_name: Option<String>,
    /// Other names of the subject entity.
    pub entity_aliases: Vec<String>,
    /// Other subject and predicate texts that name the slot.
    pub slot_aliases: Vec<SlotSurfaceAlias>,
    /// States of the slot over time.
    pub temporal_state: Vec<SlotTemporalState>,
    /// Ids of the slots whose rules can change this slot.
    pub dependency_trigger_slot_ids: Vec<MemoryId>,
    /// Ids of the slots that rules on this slot can change.
    pub dependency_target_slot_ids: Vec<MemoryId>,
}

/// Input for one slot alias.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SlotAliasInput {
    /// Id for the alias. `None` makes the store generate one.
    pub id: Option<MemoryId>,
    /// Memory scope that owns the record.
    pub scope: MemoryScope,
    /// Visibility label of the record.
    pub visibility: Visibility,
    /// Caller-defined policy labels that stay with the record.
    pub policy_tags: Vec<String>,
    /// Subject text of the alias.
    pub alias_subject: String,
    /// Predicate text of the alias.
    pub alias_predicate: String,
    /// Id of the slot the alias names. `None` means the slot of `target_claim_id`.
    pub canonical_slot_id: Option<MemoryId>,
    /// Id of a claim whose slot the alias names. `None` means `canonical_slot_id` names the slot.
    pub target_claim_id: Option<MemoryId>,
    /// Ids of the claims that support the record.
    pub source_claim_ids: Vec<MemoryId>,
    /// Start of the valid-time interval in Unix ms, inclusive. `None` means no lower bound.
    pub valid_from_ms: Option<i64>,
    /// End of the valid-time interval in Unix ms, exclusive. `None` means no upper bound.
    pub valid_to_ms: Option<i64>,
}

/// An alias that maps a subject and predicate to a canonical slot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SlotAliasRecord {
    /// Unique id of the record.
    pub id: MemoryId,
    /// Memory scope that owns the record.
    pub scope: MemoryScope,
    /// Lifecycle status of the record.
    pub status: MemoryStatus,
    /// Visibility label of the record.
    pub visibility: Visibility,
    /// Caller-defined policy labels that stay with the record.
    pub policy_tags: Vec<String>,
    /// Normalized subject key of the alias.
    pub alias_subject_key: String,
    /// Normalized predicate key of the alias.
    pub alias_predicate_key: String,
    /// Id of the entity of the alias subject. `None` means no entity matches.
    pub alias_subject_entity_id: Option<MemoryId>,
    /// Id of the slot the alias names.
    pub canonical_slot_id: MemoryId,
    /// Ids of the claims that support the record.
    pub source_claim_ids: Vec<MemoryId>,
    /// Start of the valid-time interval in Unix ms, inclusive. `None` means no lower bound.
    pub valid_from_ms: Option<i64>,
    /// End of the valid-time interval in Unix ms, exclusive. `None` means no upper bound.
    pub valid_to_ms: Option<i64>,
    /// Transaction time the store wrote the record, in Unix ms.
    pub recorded_at_ms: i64,
    /// Transaction time a newer version replaced the record, in Unix ms. `None` means the record is
    /// current.
    pub superseded_at_ms: Option<i64>,
}

/// A summary of a subject, built from claims.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfileRecord {
    /// Unique id of the record.
    pub id: MemoryId,
    /// Memory scope that owns the record.
    pub scope: MemoryScope,
    /// Lifecycle status of the record.
    pub status: MemoryStatus,
    /// Visibility label of the record.
    pub visibility: Visibility,
    /// Caller-defined policy labels that stay with the record.
    pub policy_tags: Vec<String>,
    /// Id of the subject the profile describes. `None` means no specific subject.
    pub subject_id: Option<String>,
    /// Key of the profile entry.
    pub profile_key: String,
    /// Profile text in natural language.
    pub profile_text: String,
    /// Structured profile value as a JSON string. `None` means text only.
    pub value_json: Option<String>,
    /// Ids of the claims the profile summarizes.
    pub evidence_claim_ids: Vec<MemoryId>,
    /// Ids of the evidence spans that support the record.
    pub source_span_ids: Vec<MemoryId>,
    /// Time the profile was generated, in Unix ms.
    pub generated_at_ms: i64,
    /// Version of the profile generator.
    pub generator_version: String,
    /// Start of the valid-time interval in Unix ms, inclusive. `None` means no lower bound.
    pub valid_from_ms: Option<i64>,
    /// End of the valid-time interval in Unix ms, exclusive. `None` means no upper bound.
    pub valid_to_ms: Option<i64>,
    /// Watermark of the newest correction the profile includes.
    pub correction_watermark: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ProfileInput {
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

/// A person, place, thing, or idea that memory knows.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityRecord {
    /// Unique id of the record.
    pub id: MemoryId,
    /// Memory scope that owns the record.
    pub scope: MemoryScope,
    /// Lifecycle status of the record.
    pub status: MemoryStatus,
    /// Visibility label of the record.
    pub visibility: Visibility,
    /// Caller-defined policy labels that stay with the record.
    pub policy_tags: Vec<String>,
    /// Category of the entity, such as person or place.
    pub entity_type: String,
    /// Preferred name of the entity.
    pub canonical_name: String,
    /// Other names of the entity.
    pub aliases: Vec<String>,
    /// Ids of the claims that support the record.
    pub source_claim_ids: Vec<MemoryId>,
    /// Ids of the entities merged into this entity.
    pub merge_parent_ids: Vec<MemoryId>,
    /// Id of the entity this entity was split from. `None` means no split.
    pub split_from_id: Option<MemoryId>,
    /// Confidence score. `None` means the source gave no score.
    pub confidence: Option<f32>,
}

/// Input for one entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityInput {
    /// Id for the entity. `None` makes the store generate one.
    pub id: Option<MemoryId>,
    /// Memory scope that owns the record.
    pub scope: MemoryScope,
    /// Visibility label of the record.
    pub visibility: Visibility,
    /// Caller-defined policy labels that stay with the record.
    pub policy_tags: Vec<String>,
    /// Category of the entity, such as person or place.
    pub entity_type: String,
    /// Preferred name of the entity.
    pub canonical_name: String,
    /// Other names of the entity.
    pub aliases: Vec<String>,
    /// Ids of the claims that support the record.
    pub source_claim_ids: Vec<MemoryId>,
    /// Ids of the entities merged into this entity.
    pub merge_parent_ids: Vec<MemoryId>,
    /// Id of the entity this entity was split from. `None` means no split.
    pub split_from_id: Option<MemoryId>,
    /// Confidence score. `None` means the source gave no score.
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
pub(crate) struct EntityAliasRecord {
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
pub(crate) struct EdgeRecord {
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
pub(crate) struct EdgeInput {
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

#[cfg(test)]
pub(crate) fn active_at(status: MemoryStatus, temporal: &TemporalFields, at_ms: i64) -> bool {
    status.is_active_for_default_retrieval() && temporal.valid_at(at_ms)
}

/// Normalizes text into a slot key: lowercase alphanumeric words with one space between them.
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

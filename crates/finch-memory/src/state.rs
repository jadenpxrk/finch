use crate::ingest::{create_entity, stable_hash_hex};
use crate::row::{claim_doc, correction_doc, entity_doc, rule_doc, slot_alias_doc};
use crate::store::canonical::{canonicalize_claim, canonicalize_correction};
use crate::store::{
    dedupe_claims_by_id, insert_many, json_error, retain_first_by_id, upsert_many, MemoryStore,
    ProjectionWrite, StateProjectionFrontier,
};
use crate::{
    ActorKind, ClaimRecord, CompiledMemoryContext, CorrectionRecord, EntityInput, EntityRecord,
    MemoryId, MemoryScope, MemoryStatus, ResolvedRuleApplication, RuleInput, RuleRecord,
    SlotAliasInput, SlotAliasRecord, StateReadView, StateRecord,
};
use finch_types::{Doc, Status, Value, ZResult};
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::collections::{BTreeMap, BTreeSet};

mod answer_support;
mod batch_writes;
mod boundaries;
use batch_writes::*;
use boundaries::*;

/// Valid time and transaction time at which to read state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BiTemporalQuery {
    /// Valid time to read at, in Unix ms. `None` means the current time.
    pub valid_at_ms: Option<i64>,
    /// Transaction time to read at, in Unix ms. `None` means the newest write.
    pub transaction_at_ms: Option<i64>,
}

impl BiTemporalQuery {
    /// Reads the current state.
    pub fn current() -> Self {
        Self::default()
    }

    /// Reads the state valid at `valid_at_ms`, as the newest writes record it.
    pub fn valid_at(valid_at_ms: i64) -> Self {
        Self {
            valid_at_ms: Some(valid_at_ms),
            transaction_at_ms: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn as_of(valid_at_ms: i64, transaction_at_ms: i64) -> Self {
        Self {
            valid_at_ms: Some(valid_at_ms),
            transaction_at_ms: Some(transaction_at_ms),
        }
    }
}

/// Claims, entities, rules, aliases, and corrections to write together.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateMutationBatch {
    /// Scope that every record in the batch must use.
    pub scope: MemoryScope,
    /// Transaction time of the batch, in Unix ms.
    pub transaction_time_ms: i64,
    /// Seed that makes rule propagation repeatable.
    pub propagation_seed: String,
    /// Maximum length of a chain of rule applications.
    pub max_rule_hops: usize,
    /// Claims to write.
    pub claims: Vec<ClaimRecord>,
    /// Entities to write.
    pub entities: Vec<EntityInput>,
    /// Rules to write.
    pub rules: Vec<RuleInput>,
    /// Slot aliases to write.
    #[serde(default)]
    pub slot_aliases: Vec<crate::SlotAliasInput>,
    /// Corrections to write.
    pub corrections: Vec<CorrectionRecord>,
}

/// Records that a state mutation batch wrote, and the state it projected.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateMutationResult {
    /// Claims the batch wrote.
    pub claims: Vec<ClaimRecord>,
    /// Entities the batch wrote.
    pub entities: Vec<EntityRecord>,
    /// Rules the batch wrote.
    pub rules: Vec<RuleRecord>,
    /// Slot aliases the batch wrote.
    pub slot_aliases: Vec<crate::SlotAliasRecord>,
    /// Corrections the batch wrote, bound to their targets.
    pub corrections: Vec<CorrectionRecord>,
    /// Claims that rules derived from the batch.
    pub derived_claims: Vec<ClaimRecord>,
    /// Rule applications that produced the derived claims.
    pub rule_applications: Vec<ResolvedRuleApplication>,
    /// State records the batch projected.
    pub state_records: Vec<StateRecord>,
}

/// Whether evidence can set shared state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SharedStateEvidenceRole {
    /// The evidence can set state.
    StateBearing,
    /// Only the assistant said it, so it can only advise.
    AdvisoryOnly,
}

/// Returns `StateBearing` when any actor is not the assistant, else `AdvisoryOnly`.
pub fn shared_state_evidence_role(
    actors: impl IntoIterator<Item = ActorKind>,
) -> SharedStateEvidenceRole {
    for actor in actors {
        if !matches!(actor, ActorKind::Assistant) {
            return SharedStateEvidenceRole::StateBearing;
        }
    }
    SharedStateEvidenceRole::AdvisoryOnly
}

/// `asserted_by` marker for state a dependency rule projected instead of evidence asserting it.
pub(crate) const DERIVED_STATE_ASSERTED_BY: &str = "extractor_derived";

/// Whether a claim carries directly asserted state rather than rule-derived projection output.
///
/// Derived state restates whatever a rule projected from another slot, so it cannot prove a
/// transition or act as the baseline one is compared against.
pub(crate) fn is_direct_state_claim(claim: &ClaimRecord) -> bool {
    claim.asserted_by != DERIVED_STATE_ASSERTED_BY
}

/// Whether evidence supports a state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnswerSupportState {
    /// Evidence supports the state.
    Supported,
    /// The state is marked unsupported.
    Unsupported,
    /// A correction deleted the state.
    Deleted,
    /// No evidence exists for the state.
    NoEvidence,
}

/// Evidence an answer may cite, and the support of each slot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnswerSupportContract {
    /// Support state of the answer as a whole.
    pub state: AnswerSupportState,
    /// Ids of the claims an answer may rely on.
    pub source_claim_ids: Vec<MemoryId>,
    /// Ids of the spans an answer may cite.
    pub source_span_ids: Vec<MemoryId>,
    /// Support of each answer slot.
    #[serde(default)]
    pub slots: Vec<AnswerSlotSupport>,
}

/// Support state and evidence of one slot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnswerSlotSupport {
    /// Id of the slot.
    pub slot_id: MemoryId,
    /// Subject text of the slot.
    #[serde(default)]
    pub subject: Option<String>,
    /// Predicate text of the slot.
    #[serde(default)]
    pub predicate: Option<String>,
    /// How the answer reads the slot.
    #[serde(default)]
    pub view: StateReadView,
    /// Whether evidence supports the slot.
    pub state: AnswerSupportState,
    /// Ids of the claims behind the slot.
    pub source_claim_ids: Vec<MemoryId>,
    /// Ids of the spans an answer about the slot may cite.
    pub source_span_ids: Vec<MemoryId>,
}

/// Whether to answer or refuse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnswerDisposition {
    /// Give an answer.
    Answer,
    /// Refuse to answer.
    Refuse,
}

/// A quote from an evidence span.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnswerEvidence {
    /// Id of the cited span.
    pub span_id: MemoryId,
    /// Text quoted from the span.
    pub quote: String,
}

/// One claim in an answer, with the evidence it cites.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroundedAnswerClaim {
    /// Statement the answer makes.
    pub text: String,
    /// Id of the slot the claim is about. `None` lets validation find the slot.
    #[serde(default)]
    pub slot_id: Option<MemoryId>,
    /// Quotes from the evidence spans the claim cites.
    pub evidence: Vec<AnswerEvidence>,
}

/// An answer a model proposes, before validation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnswerEmission {
    /// Whether the model answers or refuses.
    pub disposition: AnswerDisposition,
    /// Claims the model proposes.
    pub claims: Vec<GroundedAnswerClaim>,
}

/// An answer after validation against a support contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidatedAnswerEmission {
    /// Whether the result answers or refuses.
    pub disposition: AnswerDisposition,
    /// Claims that passed validation.
    pub claims: Vec<GroundedAnswerClaim>,
    /// Ids of the spans the claims cite.
    pub source_span_ids: Vec<MemoryId>,
}

/// Whether a claim stays in the answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnswerClaimDecision {
    /// The claim stays in the answer.
    Emitted,
    /// Validation removed the claim.
    Rejected,
}

/// Reason that validation rejects a claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnswerClaimRejectionReason {
    /// A refusal contains claims.
    RefusalContainsClaims,
    /// The claim has no text.
    MissingText,
    /// The claim cites no evidence.
    MissingEvidence,
    /// The claim matches more than one slot.
    AmbiguousSlot,
    /// The claim names a slot outside the support contract.
    UnknownSlot,
    /// The slot of the claim is unsupported.
    UnsupportedSlot,
    /// The slot of the claim is deleted.
    DeletedSlot,
    /// The support contract has no evidence.
    NoEvidence,
    /// The claim cites a span outside the support contract.
    EvidenceOutsideSupport,
    /// A cited span cannot be read.
    EvidenceUnavailable,
    /// A quote does not occur in its cited span.
    QuoteMismatch,
}

/// Reason that an answer is a refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnswerRefusalReason {
    /// The emission asked to refuse.
    Requested,
    /// The state is unsupported.
    UnsupportedState,
    /// The state is deleted.
    DeletedState,
    /// No evidence supports an answer.
    NoEvidence,
    /// Validation rejected every claim.
    NoValidClaims,
}

/// Validation result of one proposed claim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnswerClaimValidation {
    /// Position of the claim in the emission.
    pub input_index: usize,
    /// Whether the claim stays in the answer.
    pub decision: AnswerClaimDecision,
    /// Id of the slot the claim matched. `None` means no slot matched.
    pub slot_id: Option<MemoryId>,
    /// Reason for a rejection. `None` means the claim stays.
    pub rejection_reason: Option<AnswerClaimRejectionReason>,
}

/// Final answer, with each claim validation and the reason for a refusal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroundedAnswerResult {
    /// Whether the result answers or refuses.
    pub disposition: AnswerDisposition,
    /// Claims that passed validation.
    pub claims: Vec<GroundedAnswerClaim>,
    /// Ids of the spans the claims cite.
    pub source_span_ids: Vec<MemoryId>,
    /// Reason for a refusal. `None` means the result answers.
    pub refusal_reason: Option<AnswerRefusalReason>,
    /// Validation result of each proposed claim.
    pub claim_validations: Vec<AnswerClaimValidation>,
}

impl Default for AnswerSupportContract {
    fn default() -> Self {
        Self {
            state: AnswerSupportState::NoEvidence,
            source_claim_ids: Vec::new(),
            source_span_ids: Vec::new(),
            slots: Vec::new(),
        }
    }
}

impl AnswerSlotSupport {
    fn restrict_to_spans(&self, included_span_ids: &BTreeSet<MemoryId>) -> Self {
        let source_span_ids = included_ids(&self.source_span_ids, included_span_ids);
        Self {
            slot_id: self.slot_id.clone(),
            subject: self.subject.clone(),
            predicate: self.predicate.clone(),
            view: self.view,
            state: state_with_evidence_in_context(self.state, &source_span_ids),
            source_claim_ids: self.source_claim_ids.clone(),
            source_span_ids,
        }
    }
}

fn included_ids(ids: &[MemoryId], included: &BTreeSet<MemoryId>) -> Vec<MemoryId> {
    ids.iter()
        .filter(|id| included.contains(*id))
        .cloned()
        .collect()
}

/// A supported state whose evidence is all outside the context has no evidence in it.
fn state_with_evidence_in_context(
    state: AnswerSupportState,
    source_span_ids: &[MemoryId],
) -> AnswerSupportState {
    if matches!(state, AnswerSupportState::Supported) && source_span_ids.is_empty() {
        AnswerSupportState::NoEvidence
    } else {
        state
    }
}

impl AnswerSupportContract {
    pub(crate) fn restrict_to_context(
        &self,
        included_slot_ids: &BTreeSet<MemoryId>,
        included_span_ids: &BTreeSet<MemoryId>,
    ) -> Self {
        if self.slots.is_empty() {
            let source_span_ids = included_ids(&self.source_span_ids, included_span_ids);
            return Self {
                state: state_with_evidence_in_context(self.state, &source_span_ids),
                source_claim_ids: self.source_claim_ids.clone(),
                source_span_ids,
                slots: Vec::new(),
            };
        }

        let slots = self
            .slots
            .iter()
            .filter(|slot| included_slot_ids.contains(&slot.slot_id))
            .map(|slot| slot.restrict_to_spans(included_span_ids))
            .collect::<Vec<_>>();
        let mut source_claim_ids = slots
            .iter()
            .flat_map(|slot| slot.source_claim_ids.iter().cloned())
            .collect::<Vec<_>>();
        let mut source_span_ids = slots
            .iter()
            .flat_map(|slot| slot.source_span_ids.iter().cloned())
            .collect::<Vec<_>>();
        source_claim_ids.sort();
        source_claim_ids.dedup();
        source_span_ids.sort();
        source_span_ids.dedup();
        Self {
            state: aggregate_support_state(&slots),
            source_claim_ids,
            source_span_ids,
            slots,
        }
    }
}

pub(crate) fn aggregate_support_state(slots: &[AnswerSlotSupport]) -> AnswerSupportState {
    if slots
        .iter()
        .any(|slot| matches!(slot.state, AnswerSupportState::Deleted))
    {
        AnswerSupportState::Deleted
    } else if slots
        .iter()
        .any(|slot| matches!(slot.state, AnswerSupportState::Unsupported))
    {
        AnswerSupportState::Unsupported
    } else if slots
        .iter()
        .any(|slot| matches!(slot.state, AnswerSupportState::Supported))
    {
        AnswerSupportState::Supported
    } else {
        AnswerSupportState::NoEvidence
    }
}

/// Valid times a batch changes something at, and the claims each boundary re-evaluates beyond
/// the batch's own.
struct BoundaryPlan {
    valid_boundaries: Vec<i64>,
    /// Current claims on the reconciled rules' trigger slots.
    rule_activation_claims: Vec<ClaimRecord>,
    /// Trigger claims current at each reconciled rule start.
    rule_start_trigger_claims: BTreeMap<i64, Vec<ClaimRecord>>,
    correction_target_slot_ids: BTreeSet<MemoryId>,
}

/// Batch-wide inputs of the boundary propagation.
struct BoundaryInputs<'a> {
    scope: &'a MemoryScope,
    max_rule_hops: usize,
    claims: &'a [ClaimRecord],
    corrections: &'a [CorrectionRecord],
    /// The reconciled rules.
    rules: &'a [RuleRecord],
    plan: &'a BoundaryPlan,
}

struct BatchPropagation {
    changed_claims: Vec<ClaimRecord>,
    rule_applications: Vec<ResolvedRuleApplication>,
    /// Each boundary's applications as a `start..end` range of `rule_applications`.
    boundary_application_ranges: BTreeMap<i64, (usize, usize)>,
}

/// Records one batch writes.
struct BatchRecords<'a> {
    claims: &'a [ClaimRecord],
    rules: &'a [RuleRecord],
    corrections: &'a [CorrectionRecord],
    entities: &'a [EntityRecord],
    slot_aliases: &'a [SlotAliasRecord],
}

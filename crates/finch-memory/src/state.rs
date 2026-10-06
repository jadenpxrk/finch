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
pub use answer_support::validate_packed_answer_emission;
use batch_writes::*;
use boundaries::*;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BiTemporalQuery {
    pub valid_at_ms: Option<i64>,
    pub transaction_at_ms: Option<i64>,
}

impl BiTemporalQuery {
    pub fn current() -> Self {
        Self::default()
    }

    pub fn valid_at(valid_at_ms: i64) -> Self {
        Self {
            valid_at_ms: Some(valid_at_ms),
            transaction_at_ms: None,
        }
    }

    pub fn as_of(valid_at_ms: i64, transaction_at_ms: i64) -> Self {
        Self {
            valid_at_ms: Some(valid_at_ms),
            transaction_at_ms: Some(transaction_at_ms),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateMutationBatch {
    pub scope: MemoryScope,
    pub transaction_time_ms: i64,
    pub propagation_seed: String,
    pub max_rule_hops: usize,
    pub claims: Vec<ClaimRecord>,
    pub entities: Vec<EntityInput>,
    pub rules: Vec<RuleInput>,
    #[serde(default)]
    pub slot_aliases: Vec<crate::SlotAliasInput>,
    pub corrections: Vec<CorrectionRecord>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateMutationResult {
    pub claims: Vec<ClaimRecord>,
    pub entities: Vec<EntityRecord>,
    pub rules: Vec<RuleRecord>,
    pub slot_aliases: Vec<crate::SlotAliasRecord>,
    pub corrections: Vec<CorrectionRecord>,
    pub derived_claims: Vec<ClaimRecord>,
    pub rule_applications: Vec<ResolvedRuleApplication>,
    pub state_records: Vec<StateRecord>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SharedStateEvidenceRole {
    StateBearing,
    AdvisoryOnly,
}

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
pub const DERIVED_STATE_ASSERTED_BY: &str = "extractor_derived";

/// Whether a claim carries directly asserted state rather than rule-derived projection output.
///
/// Derived state restates whatever a rule projected from another slot, so it cannot prove a
/// transition or act as the baseline one is compared against.
pub fn is_direct_state_claim(claim: &ClaimRecord) -> bool {
    claim.asserted_by != DERIVED_STATE_ASSERTED_BY
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnswerSupportState {
    Supported,
    Unsupported,
    Deleted,
    NoEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnswerSupportContract {
    pub state: AnswerSupportState,
    pub source_claim_ids: Vec<MemoryId>,
    pub source_span_ids: Vec<MemoryId>,
    #[serde(default)]
    pub slots: Vec<AnswerSlotSupport>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnswerSlotSupport {
    pub slot_id: MemoryId,
    #[serde(default)]
    pub subject: Option<String>,
    #[serde(default)]
    pub predicate: Option<String>,
    #[serde(default)]
    pub view: StateReadView,
    pub state: AnswerSupportState,
    pub source_claim_ids: Vec<MemoryId>,
    pub source_span_ids: Vec<MemoryId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnswerDisposition {
    Answer,
    Refuse,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnswerEvidence {
    pub span_id: MemoryId,
    pub quote: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroundedAnswerClaim {
    pub text: String,
    #[serde(default)]
    pub slot_id: Option<MemoryId>,
    pub evidence: Vec<AnswerEvidence>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnswerEmission {
    pub disposition: AnswerDisposition,
    pub claims: Vec<GroundedAnswerClaim>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidatedAnswerEmission {
    pub disposition: AnswerDisposition,
    pub claims: Vec<GroundedAnswerClaim>,
    pub source_span_ids: Vec<MemoryId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnswerClaimDecision {
    Emitted,
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnswerClaimRejectionReason {
    RefusalContainsClaims,
    MissingText,
    MissingEvidence,
    AmbiguousSlot,
    UnknownSlot,
    UnsupportedSlot,
    DeletedSlot,
    NoEvidence,
    EvidenceOutsideSupport,
    EvidenceUnavailable,
    QuoteMismatch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnswerRefusalReason {
    Requested,
    UnsupportedState,
    DeletedState,
    NoEvidence,
    NoValidClaims,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnswerClaimValidation {
    pub input_index: usize,
    pub decision: AnswerClaimDecision,
    pub slot_id: Option<MemoryId>,
    pub rejection_reason: Option<AnswerClaimRejectionReason>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroundedAnswerResult {
    pub disposition: AnswerDisposition,
    pub claims: Vec<GroundedAnswerClaim>,
    pub source_span_ids: Vec<MemoryId>,
    pub refusal_reason: Option<AnswerRefusalReason>,
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
    pub fn restrict_to_context(
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

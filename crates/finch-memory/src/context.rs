use crate::retrieval::SpanSearchHit;
use crate::state::AnswerSupportContract;
use crate::store::{AnswerReadyStateProjection, ResolvedAnswerSlot};
use crate::types::{
    canonical_slot_part, ArtifactRecord, ClaimKind, ClaimPolarity, ClaimRecord,
    CorrectionOperation, CorrectionRecord, EntityRecord, MemoryId, MemoryStatus, ProfileRecord,
    RuleRecord, RuleResolutionOutcome, RuleResolutionStatus, SetStateRecord, SlotHistoryRecord,
    StateRecordKind,
};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashSet};

mod render;

use render::*;

/// Limits and options for a context packet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextOptions {
    /// Maximum number of estimated tokens in the packet.
    pub token_budget: usize,
    /// Whether span items show their source.
    pub include_provenance: bool,
}

impl Default for ContextOptions {
    fn default() -> Self {
        Self {
            token_budget: 4_096,
            include_provenance: true,
        }
    }
}

/// One item in a context packet.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextItem {
    /// Id of the record the item shows.
    pub id: MemoryId,
    /// Kind of record the item shows, such as `span` or `claim`.
    pub kind: String,
    /// Estimated token count.
    pub estimated_tokens: usize,
    /// Ids of the slots the item belongs to.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub slot_ids: Vec<MemoryId>,
    /// Evidence text the item quotes. `None` means the item quotes no evidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_text: Option<String>,
}

/// Context packet for a model prompt, with the evidence an answer may cite.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledMemoryContext {
    /// Rendered text of the packet.
    pub body: String,
    /// Estimated token count of the body.
    pub estimated_tokens: usize,
    /// Token budget the packet was built for.
    pub budget: usize,
    /// Items the body includes, in order.
    pub included: Vec<ContextItem>,
    /// Whether the budget cut items from the packet.
    pub truncated: bool,
    /// Evidence that an answer built from the packet may cite.
    #[serde(default)]
    pub support: AnswerSupportContract,
    /// Result of each rule the projection resolved.
    #[serde(default)]
    pub rule_outcomes: Vec<RuleResolutionOutcome>,
}

impl CompiledMemoryContext {
    /// Ids of the raw evidence spans the packet includes.
    pub(crate) fn included_span_ids(&self) -> BTreeSet<MemoryId> {
        self.included
            .iter()
            .filter(|item| item.kind == "span")
            .map(|item| item.id.clone())
            .collect()
    }
}

/// Builds a context packet from span hits within the token budget.
pub fn build_context(hits: &[SpanSearchHit], options: ContextOptions) -> CompiledMemoryContext {
    let items = hits
        .iter()
        .map(|hit| span_item(hit, options.include_provenance))
        .collect::<Vec<_>>();
    let mut context = build_rendered_context(items, options.token_budget);
    let source_span_ids = context
        .included
        .iter()
        .filter(|item| item.kind == "span" && item.source_text.is_some())
        .map(|item| item.id.clone())
        .collect::<Vec<_>>();
    context.support = AnswerSupportContract {
        state: if source_span_ids.is_empty() {
            crate::AnswerSupportState::NoEvidence
        } else {
            crate::AnswerSupportState::Supported
        },
        source_claim_ids: Vec::new(),
        source_span_ids,
        slots: Vec::new(),
    };
    context
}

/// Every record source the context assembler can pack. Empty slices contribute nothing.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ContextInput<'a> {
    pub profiles: &'a [ProfileRecord],
    pub claims: &'a [ClaimRecord],
    pub set_states: &'a [SetStateRecord],
    pub slot_histories: &'a [SlotHistoryRecord],
    pub rules: &'a [RuleRecord],
    pub rule_outcomes: &'a [RuleResolutionOutcome],
    pub entities: &'a [EntityRecord],
    pub corrections: &'a [CorrectionRecord],
    pub artifacts: &'a [ArtifactRecord],
    pub hits: &'a [SpanSearchHit],
}

/// Packs claims grounded in the retrieved hits, corrections, profiles, artifacts, and hits.
#[cfg(test)]
pub(crate) fn build_state_context(
    input: &ContextInput<'_>,
    options: ContextOptions,
) -> CompiledMemoryContext {
    let mut items = Vec::<RenderedContextItem>::new();
    let hit_ids = input
        .hits
        .iter()
        .map(|hit| hit.span.id.as_str())
        .collect::<HashSet<_>>();
    for claim in input.claims {
        if claim_is_grounded_in_hits(claim, &hit_ids) {
            items.push(
                (
                    claim.id.clone(),
                    claim_context_kind(claim),
                    render_claim(claim, options.include_provenance),
                )
                    .into(),
            );
        }
    }
    for correction in input.corrections {
        if matches!(correction.status, MemoryStatus::Active) {
            items.push(
                (
                    correction.id.clone(),
                    correction_context_kind(correction),
                    render_correction(correction, options.include_provenance),
                )
                    .into(),
            );
        }
    }
    for profile in input.profiles {
        items.push(
            (
                profile.id.clone(),
                "profile",
                render_profile(profile, options.include_provenance),
            )
                .into(),
        );
    }
    for artifact in input.artifacts {
        items.push(
            (
                artifact.id.clone(),
                "artifact",
                render_artifact(artifact, options.include_provenance),
            )
                .into(),
        );
    }
    items.extend(
        input
            .hits
            .iter()
            .map(|hit| span_item(hit, options.include_provenance)),
    );
    build_rendered_context(items, options.token_budget)
}

/// Packs state first, then raw evidence, with no answer targets.
pub(crate) fn build_query_state_context(
    input: &ContextInput<'_>,
    options: ContextOptions,
) -> CompiledMemoryContext {
    build_query_state_context_for_targets(input, &[], options)
}

/// Builds a context packet from a state projection, profiles, artifacts, and span hits.
pub fn build_answer_ready_state_context(
    profiles: &[ProfileRecord],
    projection: &AnswerReadyStateProjection,
    artifacts: &[ArtifactRecord],
    hits: &[SpanSearchHit],
    options: ContextOptions,
) -> CompiledMemoryContext {
    let resolved_answer_slots = projection.resolved_answer_slots();
    let mut context = build_query_state_context_for_targets(
        &ContextInput {
            profiles,
            claims: &projection.claims,
            set_states: &projection.set_states,
            slot_histories: &projection.slot_histories,
            rules: &projection.rules,
            rule_outcomes: &projection.rule_outcomes,
            entities: &projection.entities,
            corrections: &projection.corrections,
            artifacts,
            hits,
        },
        &resolved_answer_slots,
        options,
    );
    let included_slot_ids = context
        .included
        .iter()
        .filter(|item| item.kind != "state_history")
        .flat_map(|item| item.slot_ids.iter().cloned())
        .collect::<BTreeSet<_>>();
    let included_span_ids = context.included_span_ids();
    context.support = projection
        .support
        .restrict_to_context(&included_slot_ids, &included_span_ids);
    context.rule_outcomes = projection.rule_outcomes.clone();
    context
}

fn build_query_state_context_for_targets(
    input: &ContextInput<'_>,
    resolved_answer_slots: &[ResolvedAnswerSlot],
    options: ContextOptions,
) -> CompiledMemoryContext {
    let include_provenance = options.include_provenance;
    let targets = AnswerTargets::new(resolved_answer_slots);
    let state_items = collect_state_items(input, &targets, include_provenance);
    let (mut items, mut coverage_items) = order_state_items(state_items, &targets);
    if targets.fail_closed_current_read {
        let raw_start = items.len();
        return build_rendered_context_with_raw_reserve(items, options.token_budget, raw_start, 0);
    }
    // Without resolved targets the coverage items are the read's only state, so they pack as
    // state before `raw_start`. With targets they are secondary: they pack after `raw_start`,
    // behind the targets' reserved proof spans and capped, so they cannot displace that proof.
    let has_resolved_targets = !resolved_answer_slots.is_empty();
    if !has_resolved_targets {
        items.append(&mut coverage_items);
    }

    let raw_start = items.len();
    let hits_by_id = input
        .hits
        .iter()
        .map(|hit| (hit.span.id.as_str(), hit))
        .collect::<BTreeMap<_, _>>();
    items.extend(
        targets
            .proof_span_ids
            .iter()
            .filter_map(|span_id| hits_by_id.get(span_id.as_str()))
            .map(|hit| span_item(hit, include_provenance)),
    );
    if has_resolved_targets {
        push_capped_coverage(&mut items, coverage_items, options.token_budget);
    }
    items.extend(evidence_span_items(
        input.hits,
        &targets,
        &support_span_ids(input, resolved_answer_slots),
        include_provenance,
    ));
    items.extend(slot_history_items(
        input.slot_histories,
        &targets,
        include_provenance,
    ));
    items.extend(profile_and_artifact_items(input, include_provenance));
    build_rendered_context_with_raw_reserve(
        items,
        options.token_budget,
        raw_start,
        targets.required_proof_count,
    )
}

/// What the answer targets of one read decide about packing.
struct AnswerTargets<'a> {
    slots: &'a [ResolvedAnswerSlot],
    slot_ids: HashSet<MemoryId>,
    claim_ids: HashSet<&'a str>,
    rule_ids: HashSet<&'a str>,
    /// Every target is a current read that must refuse, so only the targets are packed.
    fail_closed_current_read: bool,
    /// Proof spans of refusing current targets, kept out of the raw evidence.
    withheld_raw_span_ids: HashSet<&'a str>,
    /// Proof spans of the targets that do not withhold raw evidence: each target's first span
    /// (the first `required_proof_count`, which are required), then the rest.
    proof_span_ids: Vec<MemoryId>,
    required_proof_count: usize,
}

impl<'a> AnswerTargets<'a> {
    fn new(slots: &'a [ResolvedAnswerSlot]) -> Self {
        let proving_slots = slots
            .iter()
            .filter(|slot| !current_target_withholds_competing_raw(slot))
            .collect::<Vec<_>>();
        let mut seen = HashSet::new();
        let mut proof_span_ids = proving_slots
            .iter()
            .filter_map(|slot| slot.source_span_ids.first())
            .filter(|span_id| seen.insert(span_id.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        let required_proof_count = proof_span_ids.len();
        let rest = proving_slots
            .iter()
            .flat_map(|slot| slot.source_span_ids.iter())
            .filter(|span_id| seen.insert(span_id.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        proof_span_ids.extend(rest);
        Self {
            slots,
            slot_ids: slots.iter().map(|slot| slot.slot_id.clone()).collect(),
            claim_ids: slots
                .iter()
                .flat_map(|slot| slot.source_claim_ids.iter().map(String::as_str))
                .collect(),
            rule_ids: slots
                .iter()
                .flat_map(|slot| slot.dependency_rule_ids.iter().map(String::as_str))
                .collect(),
            fail_closed_current_read: !slots.is_empty()
                && slots.iter().all(current_target_withholds_competing_raw),
            withheld_raw_span_ids: slots
                .iter()
                .filter(|slot| current_target_withholds_competing_raw(slot))
                .flat_map(|slot| slot.source_span_ids.iter().map(String::as_str))
                .collect(),
            proof_span_ids,
            required_proof_count,
        }
    }

    fn has_slot(&self, slot_id: Option<&MemoryId>) -> bool {
        slot_id.is_some_and(|slot_id| self.slot_ids.contains(slot_id))
    }
}

/// A state item before ordering: `priority`, then answer-target rank, then `order` rank it.
struct StateItem {
    priority: u8,
    order: usize,
    id: String,
    kind: &'static str,
    rendered: String,
    subject_key: Option<String>,
    slot_ids: Vec<MemoryId>,
}

impl StateItem {
    fn new(order: usize, id: String, kind: &'static str, rendered: String) -> Self {
        Self {
            priority: state_context_priority(kind),
            order,
            id,
            kind,
            rendered,
            subject_key: None,
            slot_ids: Vec::new(),
        }
    }

    fn about(mut self, subject_key: Option<String>, slot_ids: Vec<MemoryId>) -> Self {
        self.subject_key = subject_key;
        self.slot_ids = slot_ids;
        self
    }
}

/// Every state item in packing-insertion order: resolved targets, entity cards, claims, set
/// claims, set states, corrections, then rules.
fn collect_state_items(
    input: &ContextInput<'_>,
    targets: &AnswerTargets<'_>,
    include_provenance: bool,
) -> Vec<StateItem> {
    let alias_map = entity_alias_map(input.entities);
    let mut items = targets
        .slots
        .iter()
        .enumerate()
        .map(|(order, slot)| resolved_target_item(order, slot, include_provenance))
        .collect::<Vec<_>>();
    items.extend(
        render_entity_state_cards(input.claims, &alias_map, include_provenance)
            .into_iter()
            .map(|(order, subject_key, id, rendered, slot_ids)| {
                StateItem::new(order, id, "state_entity_card", rendered)
                    .about(Some(subject_key), slot_ids)
            }),
    );
    items.extend(claim_state_items(
        input,
        targets,
        &alias_map,
        include_provenance,
    ));
    items.extend(set_state_items(input, targets, include_provenance));
    items.extend(correction_state_items(input, targets, include_provenance));
    items.extend(rule_state_items(input, targets, include_provenance));
    items
}

fn resolved_target_item(
    order: usize,
    slot: &ResolvedAnswerSlot,
    include_provenance: bool,
) -> StateItem {
    StateItem::new(
        order,
        format!("resolved_answer_{}", slot.slot_id),
        "state_resolved",
        render_resolved_answer_slot(slot, include_provenance),
    )
    .about(
        subject_key_of(slot.subject.as_deref()),
        vec![slot.slot_id.clone()],
    )
}

/// The canonical subject key of a surface, when it has one.
fn subject_key_of(subject: Option<&str>) -> Option<String> {
    subject
        .map(canonical_slot_part)
        .filter(|key| !key.is_empty())
}

/// Items of the claims no answer target covers. Active set-member claims are grouped into one
/// set item per subject, predicate, and kind, and only when no set state is projected.
fn claim_state_items(
    input: &ContextInput<'_>,
    targets: &AnswerTargets<'_>,
    alias_map: &BTreeMap<String, (String, String)>,
    include_provenance: bool,
) -> Vec<StateItem> {
    let mut items = Vec::new();
    let mut set_claims =
        BTreeMap::<(Option<String>, Option<String>, u8), Vec<(usize, &ClaimRecord)>>::new();
    for (order, claim) in input.claims.iter().enumerate() {
        if targets.has_slot(claim.slot_id.as_ref()) || targets.claim_ids.contains(claim.id.as_str())
        {
            continue;
        }
        if is_active_set_claim(claim) {
            if input.set_states.is_empty() {
                set_claims
                    .entry(set_claim_group_key(claim))
                    .or_default()
                    .push((order, claim));
            }
            continue;
        }
        let kind = claim_context_kind(claim);
        items.push(
            StateItem::new(
                order,
                claim.id.clone(),
                kind,
                render_claim(claim, include_provenance),
            )
            .about(
                claim_subject_key(claim, alias_map),
                claim.slot_id.iter().cloned().collect(),
            ),
        );
    }
    items.extend(
        set_claims
            .into_iter()
            .map(|((subject, predicate, _), claims)| {
                set_claims_item(subject, predicate, &claims, include_provenance)
            }),
    );
    items
}

fn set_claim_group_key(claim: &ClaimRecord) -> (Option<String>, Option<String>, u8) {
    (
        Some(canonical_slot_part(
            claim.subject.as_deref().unwrap_or_default(),
        )),
        Some(canonical_slot_part(
            claim.predicate.as_deref().unwrap_or_default(),
        )),
        set_claim_kind_key(claim.claim_kind),
    )
}

/// One set item over a group of active set-member claims, ordered at its earliest claim.
fn set_claims_item(
    subject: Option<String>,
    predicate: Option<String>,
    ordered_claims: &[(usize, &ClaimRecord)],
    include_provenance: bool,
) -> StateItem {
    let first_order = ordered_claims
        .iter()
        .map(|(order, _)| *order)
        .min()
        .unwrap_or(0);
    let claims = ordered_claims
        .iter()
        .map(|(_, claim)| *claim)
        .collect::<Vec<_>>();
    let slot_ids = claims
        .iter()
        .filter_map(|claim| claim.slot_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let id = set_state_id(
        subject.as_deref(),
        predicate.as_deref(),
        claims[0].claim_kind, // groups are created with their first claim
    );
    StateItem::new(
        first_order,
        id,
        "state_set",
        render_set_claims(&claims, include_provenance),
    )
    .about(subject, slot_ids)
}

fn set_state_items(
    input: &ContextInput<'_>,
    targets: &AnswerTargets<'_>,
    include_provenance: bool,
) -> Vec<StateItem> {
    input
        .set_states
        .iter()
        .enumerate()
        .filter(|(_, set_state)| {
            !targets.has_slot(set_state.slot_id.as_ref())
                && !set_state
                    .claim_ids
                    .iter()
                    .any(|claim_id| targets.claim_ids.contains(claim_id.as_str()))
        })
        .map(|(order, set_state)| {
            StateItem::new(
                input.claims.len() + order,
                set_state.id.clone(),
                "state_set",
                render_set_state(set_state, include_provenance),
            )
            .about(
                subject_key_of(set_state.subject.as_deref()),
                set_state.slot_id.iter().cloned().collect(),
            )
        })
        .collect()
}

fn correction_state_items(
    input: &ContextInput<'_>,
    targets: &AnswerTargets<'_>,
    include_provenance: bool,
) -> Vec<StateItem> {
    input
        .corrections
        .iter()
        .enumerate()
        .filter(|(_, correction)| {
            matches!(correction.status, MemoryStatus::Active)
                && !targets.has_slot(correction.target_slot_id.as_ref())
        })
        .map(|(order, correction)| {
            StateItem::new(
                input.claims.len() + order,
                correction.id.clone(),
                correction_context_kind(correction),
                render_correction(correction, include_provenance),
            )
        })
        .collect()
}

/// Items of the rules outside the targets' dependency chains whose outcome resolved a
/// projected claim and does not land on an answer target.
fn rule_state_items(
    input: &ContextInput<'_>,
    targets: &AnswerTargets<'_>,
    include_provenance: bool,
) -> Vec<StateItem> {
    let projected_state_claim_ids = input
        .claims
        .iter()
        .map(|claim| claim.id.as_str())
        .chain(
            input
                .set_states
                .iter()
                .flat_map(|state| state.claim_ids.iter().map(String::as_str)),
        )
        .collect::<HashSet<_>>();
    let order_base = input.claims.len()
        + input.set_states.len()
        + input.slot_histories.len()
        + input.corrections.len();
    let mut items = Vec::new();
    for (order, rule) in input.rules.iter().enumerate() {
        if targets.rule_ids.contains(rule.id.as_str()) {
            continue;
        }
        let outcome = rule_outcome_for(rule, input.rule_outcomes);
        let target_slot_id = outcome.and_then(|outcome| outcome.target_slot_id.as_ref());
        let resolves_projected_claim = outcome
            .and_then(|outcome| outcome.resolved_claim_id.as_deref())
            .is_some_and(|claim_id| projected_state_claim_ids.contains(claim_id));
        if targets.has_slot(target_slot_id) || !resolves_projected_claim {
            continue;
        }
        let Some(rendered) = render_rule(rule, outcome, include_provenance) else {
            continue;
        };
        items.push(
            StateItem::new(order_base + order, rule.id.clone(), "state_rule", rendered)
                .about(None, target_slot_id.into_iter().cloned().collect()),
        );
    }
    items
}

/// The rule's outcome on its own target, else its only outcome.
fn rule_outcome_for<'a>(
    rule: &RuleRecord,
    rule_outcomes: &'a [RuleResolutionOutcome],
) -> Option<&'a RuleResolutionOutcome> {
    rule_outcomes
        .iter()
        .find(|outcome| outcome.rule_id == rule.id && outcome.target_slot_id == rule.target_slot_id)
        .or_else(|| {
            let mut candidates = rule_outcomes
                .iter()
                .filter(|outcome| outcome.rule_id == rule.id);
            let first = candidates.next()?;
            candidates.next().is_none().then_some(first)
        })
}

/// Orders state items by priority, answer-target rank, and order, marks answer-target items,
/// and splits the resolved targets from the coverage items.
fn order_state_items(
    mut state_items: Vec<StateItem>,
    targets: &AnswerTargets<'_>,
) -> (Vec<RenderedContextItem>, Vec<RenderedContextItem>) {
    let targets_any =
        |slot_ids: &[MemoryId]| slot_ids.iter().any(|id| targets.slot_ids.contains(id));
    state_items.sort_by_key(|item| {
        let target_rank = usize::from(!targets_any(&item.slot_ids));
        (item.priority, target_rank, item.order)
    });
    state_items
        .into_iter()
        .map(|mut item| {
            if targets_any(&item.slot_ids) {
                append_answer_target_role(&mut item.rendered);
            }
            RenderedContextItem {
                id: item.id,
                kind: item.kind,
                rendered: item.rendered,
                subject_key: item.subject_key,
                slot_ids: item.slot_ids,
                source_text: None,
            }
        })
        .partition(|item| item.kind == "state_resolved")
}

/// Adds coverage items after the answer targets and their proof, taking at most half of the
/// remaining budget so retrieved evidence keeps the other half of the packet.
fn push_capped_coverage(
    items: &mut Vec<RenderedContextItem>,
    coverage_items: Vec<RenderedContextItem>,
    token_budget: usize,
) {
    let used = items
        .iter()
        .map(|item| estimate_tokens(&item.rendered))
        .sum::<usize>();
    let coverage_cap = token_budget.saturating_sub(used) / 2;
    let mut coverage_used = 0usize;
    for item in coverage_items {
        let tokens = estimate_tokens(&item.rendered);
        if coverage_used + tokens > coverage_cap {
            continue;
        }
        coverage_used += tokens;
        items.push(item);
    }
}

/// Spans cited by the targets, claims, and set states of the packet.
fn support_span_ids<'a>(
    input: &ContextInput<'a>,
    resolved_answer_slots: &'a [ResolvedAnswerSlot],
) -> HashSet<&'a str> {
    resolved_answer_slots
        .iter()
        .flat_map(|slot| &slot.source_span_ids)
        .chain(input.claims.iter().flat_map(|claim| &claim.source_span_ids))
        .chain(
            input
                .set_states
                .iter()
                .flat_map(|state| &state.source_span_ids),
        )
        .map(String::as_str)
        .collect()
}

/// Retrieved hits after the proof: the ones supporting the packed state, then every hit, each
/// group chronological. Withheld spans and already packed proof spans are skipped.
fn evidence_span_items(
    hits: &[SpanSearchHit],
    targets: &AnswerTargets<'_>,
    support_span_ids: &HashSet<&str>,
    include_provenance: bool,
) -> Vec<RenderedContextItem> {
    let target_proof_span_ids = targets
        .proof_span_ids
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    let mut raw_hits = hits
        .iter()
        .filter(|hit| !targets.withheld_raw_span_ids.contains(hit.span.id.as_str()))
        .collect::<Vec<_>>();
    let mut supporting_hits = raw_hits
        .iter()
        .copied()
        .filter(|hit| {
            support_span_ids.contains(hit.span.id.as_str())
                && !target_proof_span_ids.contains(hit.span.id.as_str())
        })
        .collect::<Vec<_>>();
    sort_hits_chronologically(&mut supporting_hits);
    sort_hits_chronologically(&mut raw_hits);
    supporting_hits
        .into_iter()
        .chain(raw_hits)
        .map(|hit| span_item(hit, include_provenance))
        .collect()
}

fn slot_history_items(
    slot_histories: &[SlotHistoryRecord],
    targets: &AnswerTargets<'_>,
    include_provenance: bool,
) -> Vec<RenderedContextItem> {
    slot_histories
        .iter()
        .map(|history| {
            let mut rendered = render_slot_history(history, include_provenance);
            if targets.slot_ids.contains(&history.slot_id) {
                append_answer_target_role(&mut rendered);
            }
            RenderedContextItem {
                id: history.id.clone(),
                kind: "state_history",
                rendered,
                subject_key: subject_key_of(history.subject.as_deref()),
                slot_ids: vec![history.slot_id.clone()],
                source_text: None,
            }
        })
        .collect()
}

fn profile_and_artifact_items(
    input: &ContextInput<'_>,
    include_provenance: bool,
) -> Vec<RenderedContextItem> {
    let profiles = input.profiles.iter().map(|profile| {
        let rendered = render_profile(profile, include_provenance);
        RenderedContextItem::from((profile.id.clone(), "profile", rendered))
    });
    let artifacts = input.artifacts.iter().map(|artifact| {
        let rendered = render_artifact(artifact, include_provenance);
        RenderedContextItem::from((artifact.id.clone(), "artifact", rendered))
    });
    profiles.chain(artifacts).collect()
}

fn span_item(hit: &SpanSearchHit, include_provenance: bool) -> RenderedContextItem {
    RenderedContextItem {
        id: hit.span.id.clone(),
        kind: "span",
        rendered: render_hit(hit, include_provenance),
        subject_key: None,
        slot_ids: Vec::new(),
        source_text: Some(hit.span.text.clone()),
    }
}

fn current_target_withholds_competing_raw(slot: &ResolvedAnswerSlot) -> bool {
    matches!(slot.read_view, crate::StateReadView::Current)
        && !matches!(slot.support_state, crate::AnswerSupportState::Supported)
}

fn append_answer_target_role(rendered: &mut String) {
    if !rendered.ends_with('\n') {
        rendered.push('\n');
    }
    rendered.push_str("read_role: answer_target\n");
}

fn sort_hits_chronologically(hits: &mut [&SpanSearchHit]) {
    hits.sort_by(|a, b| {
        a.span
            .valid_from_ms
            .cmp(&b.span.valid_from_ms)
            .then_with(|| a.span.source_id.cmp(&b.span.source_id))
            .then_with(|| a.span.span_index.cmp(&b.span.span_index))
            .then_with(|| a.span.id.cmp(&b.span.id))
    });
}

#[cfg(test)]
fn claim_is_grounded_in_hits(claim: &ClaimRecord, hit_ids: &HashSet<&str>) -> bool {
    claim.source_span_ids.is_empty()
        || claim
            .source_span_ids
            .iter()
            .any(|source_span_id| hit_ids.contains(source_span_id.as_str()))
}

fn claim_subject_key(
    claim: &ClaimRecord,
    alias_map: &BTreeMap<String, (String, String)>,
) -> Option<String> {
    let subject_key = canonical_slot_part(claim.subject.as_deref()?);
    if subject_key.is_empty() {
        return None;
    }
    Some(
        alias_map
            .get(&subject_key)
            .map(|(group_key, _)| group_key.clone())
            .unwrap_or(subject_key),
    )
}

fn is_specific_state_kind(kind: &str) -> bool {
    matches!(
        kind,
        "state_resolved"
            | "state_tombstone"
            | "state_unsupported"
            | "state_derived"
            | "state_current"
            | "state_set"
            | "state_history"
    )
}

pub(crate) fn estimate_tokens(text: &str) -> usize {
    let by_words = text.split_whitespace().count();
    let by_chars = text.chars().count().div_ceil(4);
    by_words.max(by_chars).max(1)
}

#[cfg(test)]
mod tests;

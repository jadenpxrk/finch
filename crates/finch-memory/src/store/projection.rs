use super::*;

mod answer_slots;
mod dependency_closure;
mod projector;
mod read_plan;
mod scans;
mod selection_trace;
use dependency_closure::{persisted_rule_outcomes, resolution_root_slot_ids};
use read_plan::{answer_ready_read_plan, query_state_identity_matches};
use selection_trace::slot_shares_subject;

/// Evidence and identity resolved for one answer-ready read before any state is loaded.
struct StateReadPlan {
    explicit_selection: bool,
    preferred_slot_ids: BTreeSet<MemoryId>,
    answer_target_slot_ids: BTreeSet<MemoryId>,
    candidate_slot_ids: BTreeSet<MemoryId>,
    evidence_slot_ids: BTreeSet<MemoryId>,
    mentioned_entity_ids: BTreeSet<MemoryId>,
    entities: Vec<EntityRecord>,
    canonical_slots: Vec<CanonicalSlotRecord>,
}

/// Selected durable states closed over their dependency rules.
struct SelectedReadStates {
    states: Vec<StateRecord>,
    rules: Vec<RuleRecord>,
    dependency_slot_ids: BTreeSet<MemoryId>,
}

/// A read no durable state answers. Explicit targets still report their (empty) support; an
/// unselected read is explicitly empty rather than recomputed from raw claims by a second
/// read-time engine.
fn empty_read_projection(
    plan: &StateReadPlan,
    selection_views: &BTreeMap<MemoryId, StateReadView>,
) -> AnswerReadyStateProjection {
    let mut support = AnswerSupportContract::default();
    if plan.explicit_selection {
        support = support_contract_for_target_slots(
            &[],
            &[],
            &plan.answer_target_slot_ids,
            &plan.canonical_slots,
        );
        apply_read_views_to_support(&mut support, selection_views, &[]);
    }
    AnswerReadyStateProjection {
        support,
        ..AnswerReadyStateProjection::default()
    }
}

/// Source spans proving the projection's claims, set states, and slot histories.
fn projection_proof_span_ids(projection: &AnswerReadyStateProjection) -> Vec<MemoryId> {
    let mut span_ids = proof_span_ids_for_claims(&projection.claims);
    span_ids.extend(
        projection
            .set_states
            .iter()
            .flat_map(|state| state.source_span_ids.iter().cloned()),
    );
    span_ids.extend(
        projection
            .slot_histories
            .iter()
            .flat_map(|history| history.versions.iter())
            .flat_map(|version| version.source_span_ids.iter().cloned()),
    );
    span_ids.sort();
    span_ids.dedup();
    span_ids
}

impl MemoryStore {
    /// Adds the proof spans of the answer slots of a projection to the retrieved hits.
    pub fn hydrate_answer_ready_state_evidence(
        &self,
        scope: &MemoryScope,
        projection: &AnswerReadyStateProjection,
        retrieved_hits: &[SpanSearchHit],
    ) -> ZResult<Vec<SpanSearchHit>> {
        let _state_guard = self.lock_state_read()?;
        let mut required_span_ids = projection
            .resolved_answer_slots()
            .into_iter()
            .flat_map(|slot| slot.source_span_ids)
            .collect::<Vec<_>>();
        required_span_ids.sort();
        required_span_ids.dedup();

        let retrieved_by_id = retrieved_hits
            .iter()
            .map(|hit| (hit.span.id.as_str(), hit))
            .collect::<BTreeMap<_, _>>();
        let missing_span_ids = required_span_ids
            .iter()
            .filter(|span_id| !retrieved_by_id.contains_key(span_id.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        let fetched_by_id = self
            .fetch_spans_by_ids(scope, &missing_span_ids, None)?
            .into_iter()
            .map(|span| (span.id.clone(), span))
            .collect::<BTreeMap<_, _>>();

        let mut hydrated = Vec::with_capacity(retrieved_hits.len() + fetched_by_id.len());
        let mut seen = BTreeSet::new();
        for span_id in required_span_ids {
            if let Some(hit) = retrieved_by_id.get(span_id.as_str()) {
                hydrated.push((*hit).clone());
                seen.insert(span_id);
            } else if let Some(span) = fetched_by_id.get(&span_id) {
                hydrated.push(SpanSearchHit {
                    span: span.clone(),
                    score: 1.0,
                });
                seen.insert(span_id);
            }
        }
        hydrated.extend(
            retrieved_hits
                .iter()
                .filter(|hit| seen.insert(hit.span.id.clone()))
                .cloned(),
        );
        Ok(hydrated)
    }

    /// Selects the claims, sets, histories, rules, and corrections that answer one query.
    pub fn project_answer_ready_state(
        &self,
        scope: &MemoryScope,
        request: &AnswerReadyStateRequest<'_>,
    ) -> ZResult<AnswerReadyStateProjection> {
        let _state_guard = self.lock_state_read()?;
        let (selection_views, answer_target_slot_ids) = answer_ready_read_plan(request)?;
        self.project_answer_ready_state_from_views(
            scope,
            request,
            &selection_views,
            &answer_target_slot_ids,
        )
    }

    fn project_answer_ready_state_from_views(
        &self,
        scope: &MemoryScope,
        request: &AnswerReadyStateRequest<'_>,
        selection_views: &BTreeMap<MemoryId, StateReadView>,
        answer_target_slot_ids: &BTreeSet<MemoryId>,
    ) -> ZResult<AnswerReadyStateProjection> {
        if request.claim_limit == 0 {
            return Ok(AnswerReadyStateProjection::default());
        }
        let plan = self.plan_state_read(scope, request, selection_views, answer_target_slot_ids)?;
        let persisted_states = self.load_candidate_states(scope, &plan, request.temporal)?;
        if persisted_states.is_empty() {
            return Ok(empty_read_projection(&plan, selection_views));
        }
        let states =
            self.select_read_states(scope, request, &plan, selection_views, persisted_states)?;
        let roots = resolution_root_slot_ids(&plan, &states);
        let selected =
            self.close_state_dependencies(scope, &plan, request.temporal, &roots, states)?;
        let projection =
            self.project_selected_states(scope, request, &plan, selection_views, selected)?;
        self.attach_answer_support(scope, request, plan, selection_views, &roots, projection)
    }

    /// The selected states as a projection with their histories, the claims their dependency
    /// rules need, and the rule outcomes a read-time re-evaluation settles.
    fn project_selected_states(
        &self,
        scope: &MemoryScope,
        request: &AnswerReadyStateRequest<'_>,
        plan: &StateReadPlan,
        selection_views: &BTreeMap<MemoryId, StateReadView>,
        selected: SelectedReadStates,
    ) -> ZResult<AnswerReadyStateProjection> {
        let at_ms = request.temporal.valid_at_ms;
        let slot_histories =
            self.load_read_histories(scope, plan, selection_views, request.temporal)?;
        let dependency_claims = self.load_dependency_claims(scope, &selected, at_ms)?;
        let persisted_outcomes = persisted_rule_outcomes(&selected.states, &selected.rules);
        let mut projection = state_records_to_projection(selected.states);
        projection.slot_histories = slot_histories;
        projection.claims.extend(dependency_claims);
        projection.claims = resolve_current_claims(projection.claims, usize::MAX);
        projection.rules = selected.rules;
        self.resolve_read_time_rules(scope, at_ms, persisted_outcomes, &mut projection)?;
        reconcile_set_states_with_slot_lifecycle(&projection.claims, &mut projection.set_states);
        Ok(projection)
    }

    /// Attaches the answer support to the projection: corrections, proof spans, the support
    /// contract (per read view), and entities.
    fn attach_answer_support(
        &self,
        scope: &MemoryScope,
        request: &AnswerReadyStateRequest<'_>,
        plan: StateReadPlan,
        selection_views: &BTreeMap<MemoryId, StateReadView>,
        resolution_root_slot_ids: &BTreeSet<MemoryId>,
        mut projection: AnswerReadyStateProjection,
    ) -> ZResult<AnswerReadyStateProjection> {
        let at_ms = request.temporal.valid_at_ms;
        projection.corrections =
            self.scan_context_corrections(scope, &projection.claims, request.evidence_hits, at_ms)?;
        projection.proof_span_ids = projection_proof_span_ids(&projection);
        let support_target_slot_ids = if plan.explicit_selection {
            &plan.answer_target_slot_ids
        } else {
            resolution_root_slot_ids
        };
        projection.support = support_contract_for_target_slots(
            &projection.claims,
            &projection.set_states,
            support_target_slot_ids,
            &plan.canonical_slots,
        );
        apply_read_views_to_support(
            &mut projection.support,
            selection_views,
            &projection.slot_histories,
        );
        projection.entities = plan
            .entities
            .into_iter()
            .take(request.entity_limit)
            .collect();
        Ok(projection)
    }
}

fn support_contract_for_target_slots(
    claims: &[ClaimRecord],
    set_states: &[SetStateRecord],
    target_slot_ids: &BTreeSet<MemoryId>,
    canonical_slots: &[CanonicalSlotRecord],
) -> AnswerSupportContract {
    if target_slot_ids.is_empty() {
        return support_contract_for_claims(claims, set_states);
    }
    let target_claims = claims
        .iter()
        .filter(|claim| target_slot_ids.contains(&claim_slot_lifecycle_key(claim)))
        .collect::<Vec<_>>();
    let target_sets = set_states
        .iter()
        .filter(|state| {
            state
                .slot_id
                .as_ref()
                .is_some_and(|slot_id| target_slot_ids.contains(slot_id))
        })
        .collect::<Vec<_>>();
    let mut support = support_contract_for_claims(&target_claims, &target_sets);
    let slots_by_id = canonical_slots
        .iter()
        .map(|slot| (slot.slot_key.as_str(), slot))
        .collect::<BTreeMap<_, _>>();
    for slot in &mut support.slots {
        let Some(canonical) = slots_by_id.get(slot.slot_id.as_str()) else {
            continue;
        };
        if slot.subject.is_none() {
            slot.subject = canonical.subject.clone();
        }
        if slot.predicate.is_none() {
            slot.predicate = canonical.predicate.clone();
        }
    }
    let present_slot_ids = support
        .slots
        .iter()
        .map(|slot| slot.slot_id.as_str())
        .collect::<BTreeSet<_>>();
    let mut missing_slots = target_slot_ids
        .iter()
        .filter(|slot_id| !present_slot_ids.contains(slot_id.as_str()))
        .map(|slot_id| {
            let canonical = slots_by_id.get(slot_id.as_str());
            AnswerSlotSupport {
                slot_id: slot_id.clone(),
                subject: canonical.and_then(|slot| slot.subject.clone()),
                predicate: canonical.and_then(|slot| slot.predicate.clone()),
                view: StateReadView::Current,
                state: AnswerSupportState::NoEvidence,
                source_claim_ids: Vec::new(),
                source_span_ids: Vec::new(),
            }
        })
        .collect::<Vec<_>>();
    support.slots.append(&mut missing_slots);
    support.slots.sort_by(|a, b| a.slot_id.cmp(&b.slot_id));
    support.state = aggregate_support_state(&support.slots);
    support.source_claim_ids = support
        .slots
        .iter()
        .flat_map(|slot| slot.source_claim_ids.iter().cloned())
        .collect();
    support.source_claim_ids.sort();
    support.source_claim_ids.dedup();
    support.source_span_ids = support
        .slots
        .iter()
        .flat_map(|slot| slot.source_span_ids.iter().cloned())
        .collect();
    support.source_span_ids.sort();
    support.source_span_ids.dedup();
    support
}

/// Distinct supported values a slot has held, oldest first: the synthesized member list for a
/// set read over scalar history. Tombstoned/unsupported versions contribute nothing.
fn distinct_history_values(versions: &[SlotHistoryVersion]) -> Vec<String> {
    let mut ordered = versions
        .iter()
        .filter(|version| {
            matches!(
                version.state_kind,
                StateRecordKind::Current | StateRecordKind::Derived | StateRecordKind::Set
            )
        })
        .collect::<Vec<_>>();
    ordered.sort_by_key(|version| (version.valid_from_ms, version.observed_at_ms));
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for version in ordered {
        for value in version.object_value.iter().chain(version.members.iter()) {
            if seen.insert(canonical_slot_part(value)) {
                out.push(value.clone());
            }
        }
    }
    out
}

fn apply_read_views_to_support(
    support: &mut AnswerSupportContract,
    selection_views: &BTreeMap<MemoryId, StateReadView>,
    histories: &[SlotHistoryRecord],
) {
    let histories_by_slot = histories
        .iter()
        .map(|history| (history.slot_id.as_str(), history))
        .collect::<BTreeMap<_, _>>();

    for slot in &mut support.slots {
        slot.view = selection_views
            .get(&slot.slot_id)
            .copied()
            .unwrap_or(StateReadView::Current);
        if !matches!(slot.view, StateReadView::Timeline | StateReadView::Set) {
            continue;
        }
        let Some(history) = histories_by_slot.get(slot.slot_id.as_str()) else {
            continue;
        };
        let supported_versions = history.versions.iter().filter(|version| {
            matches!(
                version.state_kind,
                StateRecordKind::Current | StateRecordKind::Derived | StateRecordKind::Set
            ) && (version.object_value.is_some() || !version.members.is_empty())
        });
        let mut found_supported = false;
        for version in supported_versions {
            found_supported = true;
            slot.source_claim_ids
                .extend(version.claim_ids.iter().cloned());
            slot.source_span_ids
                .extend(version.source_span_ids.iter().cloned());
        }
        if found_supported {
            slot.state = AnswerSupportState::Supported;
            slot.source_claim_ids.sort();
            slot.source_claim_ids.dedup();
            slot.source_span_ids.sort();
            slot.source_span_ids.dedup();
        }
    }

    support.state = aggregate_support_state(&support.slots);
    support.source_claim_ids = support
        .slots
        .iter()
        .flat_map(|slot| slot.source_claim_ids.iter().cloned())
        .collect();
    support.source_claim_ids.sort();
    support.source_claim_ids.dedup();
    support.source_span_ids = support
        .slots
        .iter()
        .flat_map(|slot| slot.source_span_ids.iter().cloned())
        .collect();
    support.source_span_ids.sort();
    support.source_span_ids.dedup();
}

/// The slots one projector run covers. `claims` defaults to the current claims of those slots;
/// the scope rebuild passes its re-canonicalized current claims instead.
pub(crate) struct StateProjectionFrontier<'a> {
    pub slot_ids: &'a BTreeSet<MemoryId>,
    pub claims: Option<Vec<ClaimRecord>>,
    pub applications: &'a [ResolvedRuleApplication],
    pub valid_at_ms: Option<i64>,
    pub write: ProjectionWrite<'a>,
}

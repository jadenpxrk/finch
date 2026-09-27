use super::*;

/// States on a candidate slot or about a candidate entity (a mentioned entity or the subject of
/// a candidate-slot state); every state when none qualifies.
fn eligible_trace_states(
    scope_states: &[StateRecord],
    candidate_slot_ids: &BTreeSet<MemoryId>,
    mut candidate_entity_ids: BTreeSet<MemoryId>,
) -> Vec<StateRecord> {
    let on_candidate_slot = |state: &StateRecord| {
        state
            .slot_id
            .as_ref()
            .is_some_and(|slot_id| candidate_slot_ids.contains(slot_id))
    };
    candidate_entity_ids.extend(
        scope_states
            .iter()
            .filter(|state| on_candidate_slot(state))
            .filter_map(|state| state.subject_entity_id.clone()),
    );
    let eligible = scope_states
        .iter()
        .filter(|state| {
            on_candidate_slot(state)
                || state
                    .subject_entity_id
                    .as_ref()
                    .is_some_and(|entity_id| candidate_entity_ids.contains(entity_id))
        })
        .cloned()
        .collect::<Vec<_>>();
    if eligible.is_empty() {
        scope_states.to_vec()
    } else {
        eligible
    }
}

fn state_selection_trace_item(
    state: StateRecord,
    assessment: Option<&StateSelectionAssessment>,
    eligible: bool,
    selected: bool,
) -> StateSelectionTraceItem {
    let score = assessment.map_or(0, |assessment| assessment.score);
    let signals = assessment.map_or(
        StateSelectionSignals {
            preferred_slot: false,
            evidence: false,
            anchored_slot: false,
            set_entity: false,
        },
        |assessment| assessment.signals.clone(),
    );
    let disposition = if !eligible {
        StateSelectionDisposition::CandidateFiltered
    } else if selected {
        StateSelectionDisposition::Selected
    } else if score == 0 {
        StateSelectionDisposition::NoSelectionSignal
    } else {
        StateSelectionDisposition::ClaimLimit
    };
    StateSelectionTraceItem {
        state_id: state.id,
        state_kind: state.state_kind,
        slot_id: state.slot_id,
        subject_entity_id: state.subject_entity_id,
        subject: state.subject,
        predicate: state.predicate,
        score,
        signals,
        disposition,
    }
}

/// Version and value counts of one slot's claims; `None` for a slot never revised.
fn slot_history_trace(slot_id: MemoryId, mut claims: Vec<ClaimRecord>) -> Option<SlotHistoryTrace> {
    claims.sort_by(|a, b| {
        a.valid_from_ms
            .unwrap_or(a.observed_at_ms)
            .cmp(&b.valid_from_ms.unwrap_or(b.observed_at_ms))
            .then_with(|| a.observed_at_ms.cmp(&b.observed_at_ms))
            .then_with(|| a.id.cmp(&b.id))
    });
    let [first, .., last] = claims.as_slice() else {
        return None;
    };
    Some(SlotHistoryTrace {
        subject_entity_id: first.subject_entity_id.clone(),
        subject: first.subject.clone(),
        predicate: first.predicate.clone(),
        version_count: claims.len(),
        distinct_value_count: claims
            .iter()
            .filter_map(|claim| claim.object_value.as_deref())
            .map(|value| stable_hash_hex(&[value]))
            .collect::<BTreeSet<_>>()
            .len(),
        first_observed_at_ms: first.observed_at_ms,
        last_observed_at_ms: last.observed_at_ms,
        first_valid_from_ms: claims.iter().filter_map(|claim| claim.valid_from_ms).min(),
        last_valid_from_ms: claims.iter().filter_map(|claim| claim.valid_from_ms).max(),
        slot_id,
    })
}

/// Whether `slot` has the subject entity or subject key of one of `subjects`.
pub(super) fn slot_shares_subject(
    slot: &CanonicalSlotRecord,
    subjects: &[(Option<&str>, &str)],
) -> bool {
    subjects.iter().any(|(entity_id, subject_key)| {
        (entity_id.is_some() && *entity_id == slot.subject_entity_id.as_deref())
            || *subject_key == slot.subject_key.as_str()
    })
}

impl MemoryStore {
    /// Selection diagnostics for the projection `request` produced.
    pub fn explain_answer_ready_state(
        &self,
        scope: &MemoryScope,
        request: &AnswerReadyStateRequest<'_>,
        projection: &AnswerReadyStateProjection,
    ) -> ZResult<AnswerReadyStateProjectionTrace> {
        let (selection_views, answer_target_slot_ids) = answer_ready_read_plan(request)?;
        let preferred_slot_ids = selection_views.into_keys().collect::<BTreeSet<_>>();
        let candidates = self.trace_state_selection(scope, request, &preferred_slot_ids)?;
        let slot_histories = self.trace_slot_histories(scope)?;
        Ok(AnswerReadyStateProjectionTrace {
            preferred_slot_ids: preferred_slot_ids.into_iter().collect(),
            answer_target_slot_ids: answer_target_slot_ids.into_iter().collect(),
            evidence_span_ids: sorted_unique_ids(
                request.evidence_hits.iter().map(|hit| &hit.span.id),
            ),
            candidates,
            slot_histories,
            projected_claim_ids: sorted_unique_ids(projection.claims.iter().map(|claim| &claim.id)),
            projected_set_state_ids: sorted_unique_ids(
                projection.set_states.iter().map(|state| &state.id),
            ),
            projected_rule_ids: sorted_unique_ids(projection.rules.iter().map(|rule| &rule.id)),
        })
    }

    /// Every in-scope state with the score, signals, and disposition the request's selection
    /// gave it.
    fn trace_state_selection(
        &self,
        scope: &MemoryScope,
        request: &AnswerReadyStateRequest<'_>,
        preferred_slot_ids: &BTreeSet<MemoryId>,
    ) -> ZResult<Vec<StateSelectionTraceItem>> {
        let temporal = BiTemporalQuery {
            valid_at_ms: request.temporal.valid_at_ms,
            transaction_at_ms: None,
        };
        let evidence_slot_ids = self.evidence_slot_ids(scope, request.evidence_hits)?;
        let entities = self.scan_entities(scope, usize::MAX)?;
        let scope_states = self.scan_state_records(
            scope,
            StateRecordScan {
                limit: usize::MAX,
                temporal,
            },
        )?;
        let slots = self.scan_slots(scope, usize::MAX, temporal.valid_at_ms)?;
        let (query_slot_ids, mentioned_entity_ids) =
            query_state_identity_matches(request.query_text, &entities, &slots);
        let mut candidate_slot_ids = preferred_slot_ids.clone();
        candidate_slot_ids.extend(evidence_slot_ids.iter().cloned());
        candidate_slot_ids.extend(query_slot_ids);
        let eligible_states =
            eligible_trace_states(&scope_states, &candidate_slot_ids, mentioned_entity_ids);
        let eligible_state_ids = eligible_states
            .iter()
            .map(|state| state.id.clone())
            .collect::<BTreeSet<_>>();
        let query = StateSelectionQuery {
            query_text: request.query_text,
            evidence_hits: request.evidence_hits,
            entities: &entities,
            preferred_slot_ids,
            evidence_slot_ids: &evidence_slot_ids,
        };
        let assessments = assess_state_records(&eligible_states, &query);
        let selected_state_ids = select_state_records(eligible_states, &query, request.claim_limit)
            .into_iter()
            .map(|state| state.id)
            .collect::<BTreeSet<_>>();
        let mut candidates = scope_states
            .into_iter()
            .map(|state| {
                let eligible = eligible_state_ids.contains(&state.id);
                let selected = selected_state_ids.contains(&state.id);
                let assessment = assessments.get(&state.id);
                state_selection_trace_item(state, assessment, eligible, selected)
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|a, b| a.state_id.cmp(&b.state_id));
        Ok(candidates)
    }

    /// Every slot the scope's claims revised at least once, with its version and value counts.
    fn trace_slot_histories(&self, scope: &MemoryScope) -> ZResult<Vec<SlotHistoryTrace>> {
        let mut claims_by_slot = BTreeMap::<MemoryId, Vec<ClaimRecord>>::new();
        for claim in self.scan_claims(scope, usize::MAX, None)? {
            claims_by_slot
                .entry(claim_slot_lifecycle_key(&claim))
                .or_default()
                .push(claim);
        }
        Ok(claims_by_slot
            .into_iter()
            .filter_map(|(slot_id, claims)| slot_history_trace(slot_id, claims))
            .collect())
    }
}

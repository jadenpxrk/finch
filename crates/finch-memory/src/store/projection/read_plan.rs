use super::*;

fn state_read_selection_views(
    selections: &[StateReadSelection],
) -> ZResult<BTreeMap<MemoryId, StateReadView>> {
    let mut views = BTreeMap::new();
    for selection in selections {
        if selection.slot_id.trim().is_empty() {
            return Err(Status::invalid_argument(
                "state read selection requires a non-empty slot_id",
            ));
        }
        if let Some(existing) = views.insert(selection.slot_id.clone(), selection.view) {
            if existing != selection.view {
                return Err(Status::invalid_argument(format!(
                    "state read selection repeats slot {} with conflicting views",
                    selection.slot_id
                )));
            }
        }
    }
    Ok(views)
}

fn state_read_selection_plan(
    selections: &[StateReadSelection],
) -> ZResult<(BTreeMap<MemoryId, StateReadView>, BTreeSet<MemoryId>)> {
    let views = state_read_selection_views(selections)?;
    let targets = selections
        .iter()
        .filter(|selection| matches!(selection.role, StateReadRole::AnswerTarget))
        .map(|selection| selection.slot_id.clone())
        .collect::<BTreeSet<_>>();
    if !views.is_empty() && targets.is_empty() {
        return Err(Status::invalid_argument(
            "state read selections require at least one answer_target",
        ));
    }
    Ok((views, targets))
}

/// Read views of every selected slot and the answer-target slot ids of one request.
pub(super) fn answer_ready_read_plan(
    request: &AnswerReadyStateRequest<'_>,
) -> ZResult<(BTreeMap<MemoryId, StateReadView>, BTreeSet<MemoryId>)> {
    match request.target_selections {
        None => state_read_selection_plan(request.selections),
        Some(target_selections) => {
            let target_views = state_read_selection_views(target_selections)?;
            let mut selection_views = state_read_selection_views(request.selections)?;
            let target_slot_ids = if target_views.is_empty() {
                selection_views.keys().cloned().collect::<BTreeSet<_>>()
            } else {
                target_views.keys().cloned().collect::<BTreeSet<_>>()
            };
            selection_views.extend(target_views);
            Ok((selection_views, target_slot_ids))
        }
    }
}

fn state_record_visible_for_read_view(kind: StateRecordKind, view: StateReadView) -> bool {
    match view {
        StateReadView::Current | StateReadView::Timeline => true,
        StateReadView::Set => matches!(
            kind,
            StateRecordKind::Set
                | StateRecordKind::Tombstone
                | StateRecordKind::Unsupported
                | StateRecordKind::Rule
        ),
    }
}

fn state_visible_in_selection(
    state: &StateRecord,
    selection_views: &BTreeMap<MemoryId, StateReadView>,
) -> bool {
    state.slot_id.as_ref().is_some_and(|slot_id| {
        selection_views
            .get(slot_id)
            .is_none_or(|view| state_record_visible_for_read_view(state.state_kind, *view))
    })
}

impl MemoryStore {
    fn derived_consequences_of_selected_targets_in_read_set(
        &self,
        scope: &MemoryScope,
        read_set_slot_ids: &BTreeSet<MemoryId>,
        selected_answer_target_slot_ids: &BTreeSet<MemoryId>,
        temporal: BiTemporalQuery,
    ) -> ZResult<BTreeSet<MemoryId>> {
        if read_set_slot_ids.is_empty() || selected_answer_target_slot_ids.is_empty() {
            return Ok(BTreeSet::new());
        }
        let candidate_slot_ids = self
            .scan_rules(scope, usize::MAX, temporal.valid_at_ms)?
            .into_iter()
            .filter(|rule| {
                matches!(rule.trigger_binding_status, RuleBindingStatus::Bound)
                    && matches!(rule.target_binding_status, RuleBindingStatus::Bound)
                    && matches!(rule.target_match, RuleTargetMatch::ExactSlot)
                    && matches!(rule.action, RuleAction::DeriveValue)
            })
            .filter_map(|rule| {
                let trigger_id = rule.trigger_slot_id.as_ref()?;
                let target_id = rule.target_slot_id?;
                (selected_answer_target_slot_ids.contains(trigger_id)
                    && read_set_slot_ids.contains(&target_id))
                .then_some(target_id)
            })
            .collect::<BTreeSet<_>>();
        if candidate_slot_ids.is_empty() {
            return Ok(BTreeSet::new());
        }
        Ok(self
            .current_state_kinds(scope, &candidate_slot_ids, temporal)?
            .into_iter()
            .filter(|(_, kind)| matches!(kind, StateRecordKind::Derived))
            .map(|((slot_id, _), _)| slot_id)
            .collect())
    }

    fn query_matched_lifecycle_answer_target_slot_ids(
        &self,
        scope: &MemoryScope,
        query_slot_ids: &BTreeSet<MemoryId>,
        selected_answer_target_slot_ids: &BTreeSet<MemoryId>,
        read_set_slot_ids: &BTreeSet<MemoryId>,
        canonical_slots: &[CanonicalSlotRecord],
        temporal: BiTemporalQuery,
    ) -> ZResult<BTreeSet<MemoryId>> {
        if query_slot_ids.is_empty() {
            return Ok(BTreeSet::new());
        }
        let lifecycle_ids = self
            .current_state_kinds(scope, query_slot_ids, temporal)?
            .into_iter()
            .filter(|(_, kind)| is_lifecycle_state_kind(*kind))
            .map(|((slot_id, _), _)| slot_id)
            .collect::<BTreeSet<_>>();
        if lifecycle_ids.is_empty() {
            return Ok(BTreeSet::new());
        }
        let slots_by_id = canonical_slots
            .iter()
            .map(|slot| (slot.slot_key.as_str(), slot))
            .collect::<BTreeMap<_, _>>();
        let selected_subjects = selected_answer_target_slot_ids
            .iter()
            .filter_map(|slot_id| slots_by_id.get(slot_id.as_str()).copied())
            .map(|slot| (slot.subject_entity_id.as_deref(), slot.subject_key.as_str()))
            .collect::<Vec<_>>();
        let subject_sharing_slot_ids = slots_by_id
            .iter()
            .filter(|(_, slot)| slot_shares_subject(slot, &selected_subjects))
            .map(|(slot_id, _)| *slot_id)
            .collect::<BTreeSet<_>>();
        Ok(lifecycle_ids
            .into_iter()
            .filter(|slot_id| {
                read_set_slot_ids.contains(slot_id)
                    || subject_sharing_slot_ids.contains(slot_id.as_str())
            })
            .collect())
    }

    /// The highest-priority current state kind of each of `slot_ids` in each scope that has one;
    /// one scope's state must not stand for another's.
    fn current_state_kinds(
        &self,
        scope: &MemoryScope,
        slot_ids: &BTreeSet<MemoryId>,
        temporal: BiTemporalQuery,
    ) -> ZResult<BTreeMap<(MemoryId, MemoryScope), StateRecordKind>> {
        let records =
            self.scan_state_records_bitemporal_for_slot_ids(scope, slot_ids, usize::MAX, temporal)?;
        let mut kinds = BTreeMap::<(MemoryId, MemoryScope), StateRecordKind>::new();
        for record in records {
            let Some(slot_id) = record.slot_id else {
                continue;
            };
            if !slot_ids.contains(&slot_id) {
                continue;
            }
            let key = (slot_id, record.scope);
            let replace = kinds.get(&key).is_none_or(|existing| {
                state_record_priority(record.state_kind) < state_record_priority(*existing)
            });
            if replace {
                kinds.insert(key, record.state_kind);
            }
        }
        Ok(kinds)
    }

    /// Slots the evidence hits' claims project into.
    pub(super) fn evidence_slot_ids(
        &self,
        scope: &MemoryScope,
        evidence_hits: &[SpanSearchHit],
    ) -> ZResult<BTreeSet<MemoryId>> {
        let span_ids = evidence_hits
            .iter()
            .map(|hit| hit.span.id.as_str())
            .collect::<BTreeSet<_>>();
        let episode_ids = evidence_hits
            .iter()
            .map(|hit| hit.span.source_id.as_str())
            .collect::<BTreeSet<_>>();
        self.claim_slot_ids_for_evidence(scope, &span_ids, &episode_ids)
    }

    /// Resolves the evidence, entities, canonical slots, answer targets, and candidate slots of
    /// one read before any state is loaded.
    pub(super) fn plan_state_read(
        &self,
        scope: &MemoryScope,
        request: &AnswerReadyStateRequest<'_>,
        selection_views: &BTreeMap<MemoryId, StateReadView>,
        answer_target_slot_ids: &BTreeSet<MemoryId>,
    ) -> ZResult<StateReadPlan> {
        let temporal = request.temporal;
        let evidence_slot_ids = self.evidence_slot_ids(scope, request.evidence_hits)?;
        let entities = if request.entity_limit == 0 {
            Vec::new()
        } else {
            self.scan_entities(scope, usize::MAX)?
        };
        let canonical_slots = self.scan_slots(scope, usize::MAX, temporal.valid_at_ms)?;
        let (query_slot_ids, mentioned_entity_ids) =
            query_state_identity_matches(request.query_text, &entities, &canonical_slots);
        let preferred_slot_ids = selection_views.keys().cloned().collect::<BTreeSet<_>>();
        let explicit_selection = !selection_views.is_empty();
        let (answer_target_slot_ids, mut candidate_slot_ids) = if explicit_selection {
            let targets = self.expand_selected_answer_targets(
                scope,
                &preferred_slot_ids,
                answer_target_slot_ids.clone(),
                &query_slot_ids,
                &canonical_slots,
                temporal,
            )?;
            let (_, dependency_slot_ids) = self.dependency_closure(
                scope,
                &preferred_slot_ids,
                &targets,
                &BTreeSet::new(),
                temporal.valid_at_ms,
                MAX_RULE_HOPS,
            )?;
            (targets, dependency_slot_ids)
        } else {
            let mut slot_ids = evidence_slot_ids.clone();
            slot_ids.extend(query_slot_ids);
            (answer_target_slot_ids.clone(), slot_ids)
        };
        candidate_slot_ids.extend(preferred_slot_ids.iter().cloned());
        candidate_slot_ids.extend(answer_target_slot_ids.iter().cloned());
        Ok(StateReadPlan {
            explicit_selection,
            preferred_slot_ids,
            answer_target_slot_ids,
            candidate_slot_ids,
            evidence_slot_ids,
            mentioned_entity_ids,
            entities,
            canonical_slots,
        })
    }

    /// Adds to the selected answer targets the derived consequences of those targets inside the
    /// read set and the query-matched slots whose current state is a lifecycle state.
    fn expand_selected_answer_targets(
        &self,
        scope: &MemoryScope,
        read_set_slot_ids: &BTreeSet<MemoryId>,
        mut answer_target_slot_ids: BTreeSet<MemoryId>,
        query_slot_ids: &BTreeSet<MemoryId>,
        canonical_slots: &[CanonicalSlotRecord],
        temporal: BiTemporalQuery,
    ) -> ZResult<BTreeSet<MemoryId>> {
        answer_target_slot_ids.extend(self.derived_consequences_of_selected_targets_in_read_set(
            scope,
            read_set_slot_ids,
            &answer_target_slot_ids,
            temporal,
        )?);
        answer_target_slot_ids.extend(self.query_matched_lifecycle_answer_target_slot_ids(
            scope,
            query_slot_ids,
            &answer_target_slot_ids,
            read_set_slot_ids,
            canonical_slots,
            temporal,
        )?);
        Ok(answer_target_slot_ids)
    }

    /// Durable states on the read's candidate slots. An unselected read also takes the states of
    /// the entities it mentions or its states name, and reads every state in scope when nothing
    /// matched.
    pub(super) fn load_candidate_states(
        &self,
        scope: &MemoryScope,
        plan: &StateReadPlan,
        temporal: BiTemporalQuery,
    ) -> ZResult<Vec<StateRecord>> {
        let mut states = self.scan_state_records_bitemporal_for_slot_ids(
            scope,
            &plan.candidate_slot_ids,
            usize::MAX,
            temporal,
        )?;
        if !plan.explicit_selection {
            let entity_ids = read_entity_ids(plan, &states);
            states.extend(self.scan_state_records_bitemporal_for_subject_entity_ids(
                scope,
                &entity_ids,
                usize::MAX,
                temporal,
            )?);
        }
        if states.is_empty() && !plan.explicit_selection {
            let every_state = StateRecordScan {
                limit: usize::MAX,
                temporal,
            };
            return self.scan_state_records(scope, every_state);
        }
        retain_first_by_id(&mut states, |state| &state.id);
        Ok(states)
    }

    /// The candidate states the read's signals select, limited to what each selected slot's view
    /// shows, with the rule ids of their dependency traces attached.
    pub(super) fn select_read_states(
        &self,
        scope: &MemoryScope,
        request: &AnswerReadyStateRequest<'_>,
        plan: &StateReadPlan,
        selection_views: &BTreeMap<MemoryId, StateReadView>,
        persisted_states: Vec<StateRecord>,
    ) -> ZResult<Vec<StateRecord>> {
        let selection_signal_slot_ids = if plan.explicit_selection {
            &plan.candidate_slot_ids
        } else {
            &plan.preferred_slot_ids
        };
        let query = StateSelectionQuery {
            query_text: request.query_text,
            evidence_hits: request.evidence_hits,
            entities: &plan.entities,
            preferred_slot_ids: selection_signal_slot_ids,
            evidence_slot_ids: &plan.evidence_slot_ids,
        };
        let mut states = select_state_records(persisted_states, &query, request.claim_limit);
        if plan.explicit_selection {
            states.retain(|state| state_visible_in_selection(state, selection_views));
        }
        let traces = self.scan_dependency_traces_bitemporal(scope, usize::MAX, request.temporal)?;
        merge_trace_rule_ids(&mut states, &traces);
        Ok(states)
    }
}

fn is_lifecycle_state_kind(kind: StateRecordKind) -> bool {
    matches!(
        kind,
        StateRecordKind::Derived | StateRecordKind::Tombstone | StateRecordKind::Unsupported
    )
}

/// Entities an unselected read covers: the ones its query mentions and its states' subjects.
fn read_entity_ids(plan: &StateReadPlan, states: &[StateRecord]) -> BTreeSet<MemoryId> {
    let state_entity_ids = states
        .iter()
        .filter_map(|state| state.subject_entity_id.clone());
    plan.mentioned_entity_ids
        .iter()
        .cloned()
        .chain(state_entity_ids)
        .collect()
}

/// Whether the query names the slot's predicate and its subject (by name or mentioned entity).
fn query_names_slot(
    normalized_query: &str,
    mentioned_entity_ids: &BTreeSet<MemoryId>,
    slot: &CanonicalSlotRecord,
) -> bool {
    let names = |surface: Option<&str>, key: &str| {
        surface.is_some_and(|surface| normalized_phrase_present(normalized_query, surface))
            || normalized_phrase_present(normalized_query, key)
    };
    let subject_entity_mentioned = slot
        .subject_entity_id
        .as_ref()
        .is_some_and(|entity_id| mentioned_entity_ids.contains(entity_id));
    names(slot.predicate.as_deref(), &slot.predicate_key)
        && (names(slot.subject.as_deref(), &slot.subject_key) || subject_entity_mentioned)
}

pub(super) fn query_state_identity_matches(
    query_text: &str,
    entities: &[EntityRecord],
    slots: &[CanonicalSlotRecord],
) -> (BTreeSet<MemoryId>, BTreeSet<MemoryId>) {
    let normalized_query = canonical_slot_part(query_text);
    let entity_ids = mentioned_entity_ids(query_text, entities);
    let slot_ids = slots
        .iter()
        .filter(|slot| query_names_slot(&normalized_query, &entity_ids, slot))
        .map(|slot| slot.slot_key.clone())
        .collect();
    (slot_ids, entity_ids)
}

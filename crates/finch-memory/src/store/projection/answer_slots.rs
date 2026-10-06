use super::*;

impl AnswerReadyStateProjection {
    /// Returns the answer-ready view of each answer slot.
    pub fn resolved_answer_slots(&self) -> Vec<ResolvedAnswerSlot> {
        let mut resolved = self
            .support
            .slots
            .iter()
            .map(|support| self.resolve_answer_slot(support))
            .collect::<Vec<_>>();
        resolved.sort_by(|a, b| a.slot_id.cmp(&b.slot_id));
        resolved
    }

    fn resolve_answer_slot(&self, support: &AnswerSlotSupport) -> ResolvedAnswerSlot {
        let claims = self
            .claims
            .iter()
            .filter(|claim| claim_slot_lifecycle_key(claim) == support.slot_id)
            .collect::<Vec<_>>();
        let sources = SlotStateSources {
            current_claim: newest_claim(claims.iter().copied()),
            set_state: self
                .set_states
                .iter()
                .filter(|state| state.slot_id.as_ref() == Some(&support.slot_id))
                .max_by_key(|state| (state.valid_from_ms, state.observed_at_ms, &state.id)),
            history: self
                .slot_histories
                .iter()
                .find(|history| history.slot_id == support.slot_id),
        };
        let facets = resolved_slot_facets(&claims);
        let (dependency_rule_ids, dependency_trigger_slot_ids) =
            self.slot_dependency_chain(support);
        let (valid_from_ms, valid_to_ms) = sources.validity();
        ResolvedAnswerSlot {
            slot_id: support.slot_id.clone(),
            read_view: support.view,
            support_state: support.state,
            state_kind: sources.state_kind(support.state),
            subject: support.subject.clone().or_else(|| sources.subject()),
            predicate: support.predicate.clone().or_else(|| sources.predicate()),
            current_value: sources.current_value(support.state, &facets),
            members: sources.members(support),
            facets,
            source_claim_ids: support.source_claim_ids.clone(),
            source_span_ids: support.source_span_ids.clone(),
            dependency_rule_ids,
            dependency_trigger_slot_ids,
            valid_from_ms,
            valid_to_ms,
        }
    }

    /// Applied rules reachable backwards from the slot through rule targets, and their trigger
    /// slots. At the slot itself only outcomes resolving one of its visible claims count.
    fn slot_dependency_chain(&self, support: &AnswerSlotSupport) -> (Vec<MemoryId>, Vec<MemoryId>) {
        let source_claim_ids = support
            .source_claim_ids
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let mut rule_ids = Vec::new();
        let mut trigger_slot_ids = Vec::new();
        let mut frontier = vec![support.slot_id.as_str()];
        let mut visited_slots = BTreeSet::new();
        while let Some(target_slot_id) = frontier.pop() {
            if !visited_slots.insert(target_slot_id) {
                continue;
            }
            for outcome in &self.rule_outcomes {
                if !outcome.status.is_applied()
                    || outcome.target_slot_id.as_deref() != Some(target_slot_id)
                {
                    continue;
                }
                let resolves_visible_claim = outcome.resolved_claim_id.as_deref();
                if target_slot_id == support.slot_id
                    && !resolves_visible_claim.is_some_and(|id| source_claim_ids.contains(id))
                {
                    continue;
                }
                rule_ids.push(outcome.rule_id.clone());
                if let Some(trigger_slot_id) = outcome.trigger_slot_id.as_deref() {
                    trigger_slot_ids.push(trigger_slot_id.to_owned());
                    frontier.push(trigger_slot_id);
                }
            }
        }
        rule_ids.sort();
        rule_ids.dedup();
        trigger_slot_ids.sort();
        trigger_slot_ids.dedup();
        (rule_ids, trigger_slot_ids)
    }
}

/// The projected records one answer slot resolves from, each optional.
struct SlotStateSources<'a> {
    current_claim: Option<&'a ClaimRecord>,
    set_state: Option<&'a SetStateRecord>,
    history: Option<&'a SlotHistoryRecord>,
}

impl SlotStateSources<'_> {
    fn subject(&self) -> Option<String> {
        self.current_claim
            .and_then(|claim| claim.subject.clone())
            .or_else(|| self.set_state.and_then(|state| state.subject.clone()))
            .or_else(|| self.history.and_then(|history| history.subject.clone()))
    }

    fn predicate(&self) -> Option<String> {
        self.current_claim
            .and_then(|claim| claim.predicate.clone())
            .or_else(|| self.set_state.and_then(|state| state.predicate.clone()))
            .or_else(|| self.history.and_then(|history| history.predicate.clone()))
    }

    fn state_kind(&self, support_state: AnswerSupportState) -> Option<StateRecordKind> {
        match support_state {
            AnswerSupportState::Deleted => Some(StateRecordKind::Tombstone),
            AnswerSupportState::Unsupported => Some(StateRecordKind::Unsupported),
            AnswerSupportState::Supported if self.set_state.is_some() => Some(StateRecordKind::Set),
            AnswerSupportState::Supported => self.current_claim.map(claim_state_record_kind),
            AnswerSupportState::NoEvidence => None,
        }
    }

    fn current_value(
        &self,
        support_state: AnswerSupportState,
        facets: &[ResolvedSlotFacet],
    ) -> Option<String> {
        if !matches!(support_state, AnswerSupportState::Supported) {
            return None;
        }
        let distinct_supported_values = facets
            .iter()
            .filter(|facet| matches!(facet.support_state, AnswerSupportState::Supported))
            .filter_map(|facet| facet.value.as_deref().map(canonical_slot_part))
            .collect::<BTreeSet<_>>();
        // Several co-current supported facets with different values mean the store has no
        // evidence that one replaced the others; expose the facets, not a guess.
        if distinct_supported_values.len() > 1 {
            return None;
        }
        self.current_claim
            .and_then(|claim| claim.object_value.clone())
    }

    fn members(&self, support: &AnswerSlotSupport) -> Vec<String> {
        if !matches!(support.state, AnswerSupportState::Supported) {
            return Vec::new();
        }
        let explicit = self
            .set_state
            .map(|state| state.members.clone())
            .unwrap_or_default();
        // A set read over a slot with no explicit set state is a question about every value the
        // slot has held: its versioned history is that set. Values whose only record is a
        // tombstone or unsupported state stay excluded.
        if explicit.is_empty() && matches!(support.view, StateReadView::Set) {
            return self
                .history
                .map(|history| distinct_history_values(&history.versions))
                .unwrap_or_default();
        }
        explicit
    }

    fn validity(&self) -> (Option<i64>, Option<i64>) {
        self.current_claim
            .map(|claim| (claim.valid_from_ms, claim.valid_to_ms))
            .or_else(|| {
                self.set_state
                    .map(|state| (state.valid_from_ms, state.valid_to_ms))
            })
            .unwrap_or((None, None))
    }
}

/// One entry per surface facet projected for a slot family, newest first.
fn resolved_slot_facets(claims: &[&ClaimRecord]) -> Vec<ResolvedSlotFacet> {
    let mut by_facet = BTreeMap::<String, Vec<&ClaimRecord>>::new();
    for claim in claims {
        by_facet
            .entry(claim_facet_lifecycle_key(claim))
            .or_default()
            .push(claim);
    }
    let mut facets = by_facet
        .into_values()
        .filter_map(|group| {
            let newest = newest_claim(group.iter().copied())?;
            let support_state = support_state_for_claim_refs(&group, false);
            let mut claim_ids = group
                .iter()
                .map(|claim| claim.id.clone())
                .collect::<Vec<_>>();
            claim_ids.sort();
            claim_ids.dedup();
            let mut source_span_ids = group
                .iter()
                .flat_map(|claim| claim.source_span_ids.iter().cloned())
                .collect::<Vec<_>>();
            source_span_ids.sort();
            source_span_ids.dedup();
            Some(ResolvedSlotFacet {
                subject: newest.subject.clone(),
                predicate: newest.predicate.clone(),
                support_state,
                state_kind: Some(claim_state_record_kind(newest)),
                value: matches!(support_state, AnswerSupportState::Supported)
                    .then(|| newest.object_value.clone())
                    .flatten(),
                state_text: Some(newest.claim_text.clone()),
                claim_ids,
                source_span_ids,
                valid_from_ms: newest.valid_from_ms,
                valid_to_ms: newest.valid_to_ms,
            })
        })
        .collect::<Vec<_>>();
    facets.sort_by(|a, b| {
        b.valid_from_ms
            .cmp(&a.valid_from_ms)
            .then_with(|| a.predicate.cmp(&b.predicate))
            .then_with(|| a.claim_ids.cmp(&b.claim_ids))
    });
    facets
}

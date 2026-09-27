use super::*;

/// The slots a read resolves its rules from: the selected slots of an explicit selection,
/// else the slots of the selected states.
pub(super) fn resolution_root_slot_ids(
    plan: &StateReadPlan,
    states: &[StateRecord],
) -> BTreeSet<MemoryId> {
    if plan.explicit_selection {
        plan.preferred_slot_ids.clone()
    } else {
        states
            .iter()
            .filter_map(|state| state.slot_id.clone())
            .collect()
    }
}

/// Rule outcomes the selected derived and unsupported states already record.
pub(super) fn persisted_rule_outcomes(
    states: &[StateRecord],
    rules: &[RuleRecord],
) -> Vec<RuleResolutionOutcome> {
    let mut outcomes = Vec::new();
    for state in states {
        let status = match state.state_kind {
            StateRecordKind::Derived => RuleResolutionStatus::AppliedDerived,
            StateRecordKind::Unsupported => RuleResolutionStatus::AppliedUnsupported,
            _ => continue,
        };
        for rule_id in &state.rule_ids {
            outcomes.push(RuleResolutionOutcome {
                rule_id: rule_id.clone(),
                trigger_slot_id: rules
                    .iter()
                    .find(|rule| rule.id == *rule_id)
                    .and_then(|rule| rule.trigger_slot_id.clone()),
                target_slot_id: state.slot_id.clone(),
                status,
                resolved_claim_id: state.claim_ids.first().cloned(),
            });
        }
    }
    outcomes
}

/// Read-time outcomes, where a durable outcome fills a rule/target pair the read did not apply.
/// Durable outcomes resolved by a claim the read suppressed are dropped.
fn merge_rule_outcomes(
    mut outcomes: Vec<RuleResolutionOutcome>,
    persisted_outcomes: Vec<RuleResolutionOutcome>,
    suppressed_claim_ids: &BTreeSet<MemoryId>,
) -> Vec<RuleResolutionOutcome> {
    for persisted in persisted_outcomes {
        if persisted
            .resolved_claim_id
            .as_ref()
            .is_some_and(|claim_id| suppressed_claim_ids.contains(claim_id))
        {
            continue;
        }
        let existing = outcomes.iter_mut().find(|outcome| {
            outcome.rule_id == persisted.rule_id
                && outcome.target_slot_id == persisted.target_slot_id
        });
        match existing {
            Some(outcome) if !outcome.status.is_applied() => *outcome = persisted,
            Some(_) => {}
            None => outcomes.push(persisted),
        }
    }
    outcomes.sort_by(|a, b| {
        a.rule_id
            .cmp(&b.rule_id)
            .then_with(|| a.target_slot_id.cmp(&b.target_slot_id))
    });
    outcomes
}

impl MemoryStore {
    /// Closes the selected states over their dependency rules and adds the states of every
    /// dependency slot the selection did not already hold.
    pub(super) fn close_state_dependencies(
        &self,
        scope: &MemoryScope,
        plan: &StateReadPlan,
        temporal: BiTemporalQuery,
        resolution_root_slot_ids: &BTreeSet<MemoryId>,
        mut states: Vec<StateRecord>,
    ) -> ZResult<SelectedReadStates> {
        let target_rule_ids = states
            .iter()
            .flat_map(|state| state.rule_ids.iter().cloned())
            .collect::<BTreeSet<_>>();
        let (rules, dependency_slot_ids) = self.dependency_closure(
            scope,
            resolution_root_slot_ids,
            &plan.answer_target_slot_ids,
            &target_rule_ids,
            temporal.valid_at_ms,
            MAX_RULE_HOPS,
        )?;
        if !dependency_slot_ids.is_empty() {
            let known_state_ids = states
                .iter()
                .map(|state| state.id.as_str())
                .collect::<BTreeSet<_>>();
            let dependency_states = self
                .scan_state_records_bitemporal_for_slot_ids(
                    scope,
                    &dependency_slot_ids,
                    usize::MAX,
                    temporal,
                )?
                .into_iter()
                .filter(|state| !known_state_ids.contains(state.id.as_str()))
                .collect::<Vec<_>>();
            states.extend(dependency_states);
        }
        Ok(SelectedReadStates {
            states,
            rules,
            dependency_slot_ids,
        })
    }

    /// Timeline and set reads carry their history; a current read of an answer target also
    /// carries its version history (non-authoritative) so supersession is visible, not implied.
    pub(super) fn load_read_histories(
        &self,
        scope: &MemoryScope,
        plan: &StateReadPlan,
        selection_views: &BTreeMap<MemoryId, StateReadView>,
        temporal: BiTemporalQuery,
    ) -> ZResult<Vec<SlotHistoryRecord>> {
        let slot_ids = selection_views
            .iter()
            .filter(|(slot_id, view)| match view {
                StateReadView::Timeline | StateReadView::Set => true,
                StateReadView::Current => plan.answer_target_slot_ids.contains(*slot_id),
            })
            .map(|(slot_id, _)| slot_id.clone())
            .collect::<BTreeSet<_>>();
        Ok(state_records_to_slot_histories(
            self.scan_state_record_history_for_slot_ids(scope, &slot_ids, temporal)?,
        ))
    }

    /// Current claims of the dependency slots no selected state covers and of the dependency
    /// rules' triggers.
    pub(super) fn load_dependency_claims(
        &self,
        scope: &MemoryScope,
        selected: &SelectedReadStates,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<ClaimRecord>> {
        let projected_slot_ids = selected
            .states
            .iter()
            .filter_map(|state| state.slot_id.as_ref())
            .collect::<BTreeSet<_>>();
        let missing_slot_ids = selected
            .dependency_slot_ids
            .iter()
            .filter(|slot_id| !projected_slot_ids.contains(slot_id))
            .cloned()
            .collect::<BTreeSet<_>>();
        let mut claims =
            self.scan_current_claims_for_slot_ids(scope, &missing_slot_ids, usize::MAX, at_ms)?;
        claims.extend(self.current_claims_for_rule_triggers(scope, &selected.rules, at_ms)?);
        Ok(claims)
    }

    /// Re-evaluates the read set's rules at read time, completes their dependency chains, and
    /// merges the result with the outcomes durable state already recorded.
    pub(super) fn resolve_read_time_rules(
        &self,
        scope: &MemoryScope,
        at_ms: Option<i64>,
        persisted_outcomes: Vec<RuleResolutionOutcome>,
        projection: &mut AnswerReadyStateProjection,
    ) -> ZResult<()> {
        let applications = self.read_time_rule_applications(projection)?;
        self.add_applied_rules(scope, at_ms, &applications, &mut projection.rules)?;
        let completion = self.complete_target_dependency_chains(
            &projection.rules,
            &projection.claims,
            &applications,
            MAX_RULE_HOPS,
        )?;
        projection.rule_outcomes = merge_rule_outcomes(
            completion.outcomes,
            persisted_outcomes,
            &completion.suppressed_claim_ids,
        );
        let visible_rule_ids = projection
            .rule_outcomes
            .iter()
            .filter(|outcome| outcome.status.is_applied())
            .map(|outcome| outcome.rule_id.as_str())
            .collect::<BTreeSet<_>>();
        projection
            .rules
            .retain(|rule| visible_rule_ids.contains(rule.id.as_str()));

        let mut claims = std::mem::take(&mut projection.claims);
        claims.retain(|claim| !completion.suppressed_claim_ids.contains(&claim.id));
        claims.extend(completion.projected_claims);
        let mut applications = completion.applications;
        retain_first_by_id(&mut applications, |application| &application.claim.id);
        claims.extend(
            applications
                .into_iter()
                .map(|application| application.claim),
        );
        projection.claims = resolve_current_claims(claims, usize::MAX);
        Ok(())
    }

    /// Rule applications a read-time re-evaluation of the projection's claims produces, limited
    /// to the dependency rules when the read has any.
    fn read_time_rule_applications(
        &self,
        projection: &AnswerReadyStateProjection,
    ) -> ZResult<Vec<ResolvedRuleApplication>> {
        let known_ids = projection
            .claims
            .iter()
            .map(|claim| claim.id.clone())
            .collect::<BTreeSet<_>>();
        let mut applications =
            self.resolve_rules_for_changed_claims(&projection.claims, &known_ids, MAX_RULE_HOPS)?;
        if !projection.rules.is_empty() {
            let relevant_rule_ids = projection
                .rules
                .iter()
                .map(|rule| rule.id.as_str())
                .collect::<BTreeSet<_>>();
            applications
                .retain(|application| relevant_rule_ids.contains(application.rule_id.as_str()));
        }
        Ok(applications)
    }

    /// Adds the stored rules that fired at read time but were outside the dependency closure.
    fn add_applied_rules(
        &self,
        scope: &MemoryScope,
        at_ms: Option<i64>,
        applications: &[ResolvedRuleApplication],
        rules: &mut Vec<RuleRecord>,
    ) -> ZResult<()> {
        let applied_rule_ids = applications
            .iter()
            .map(|application| application.rule_id.as_str())
            .collect::<BTreeSet<_>>();
        let known_rule_ids = rules
            .iter()
            .map(|rule| rule.id.as_str())
            .collect::<BTreeSet<_>>();
        let applied_rules = self
            .scan_rules(scope, usize::MAX, at_ms)?
            .into_iter()
            .filter(|rule| {
                applied_rule_ids.contains(rule.id.as_str())
                    && !known_rule_ids.contains(rule.id.as_str())
            })
            .collect::<Vec<_>>();
        rules.extend(applied_rules);
        rules.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(())
    }
}

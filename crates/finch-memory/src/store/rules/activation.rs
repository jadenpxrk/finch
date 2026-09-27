use super::*;

impl MemoryStore {
    pub(super) fn rule_activation_evidence(
        &self,
        rule: &RuleRecord,
        trigger: &ClaimRecord,
    ) -> ZResult<(bool, Option<ClaimRecord>)> {
        let (activated, prior, _) =
            self.rule_activation_evidence_with_reason_for_target(rule, trigger, None)?;
        Ok((activated, prior))
    }

    /// A standing dependency governs its target from the moment the dependency holds, not from
    /// the moment it was uttered: when the rule is stated after the trigger's transition but the
    /// target's current value was asserted BEFORE that transition, the assertion predates the
    /// world it depends on and the rule still applies to that transition.
    pub(super) fn rule_activation_evidence_with_reason_for_target(
        &self,
        rule: &RuleRecord,
        trigger: &ClaimRecord,
        target_asserted_at_ms: Option<i64>,
    ) -> ZResult<(bool, Option<ClaimRecord>, RuleResolutionStatus)> {
        let trigger_time = claim_effective_from_ms(trigger);
        let rule_time = rule.valid_from_ms.unwrap_or(i64::MIN);
        if matches!(rule.activation, RuleActivation::ContinuousProjection) {
            let active =
                trigger_time >= rule_time || rule_trigger_interval(rule, trigger).is_some();
            return Ok((active, None, RuleResolutionStatus::OutsideValidityWindow));
        }
        let retroactive =
            target_asserted_at_ms.is_some_and(|target_time| target_time < trigger_time);
        if !temporal_position_is_after(
            trigger_time,
            trigger.source_sequence_no,
            rule_time,
            rule.source_sequence_no,
        ) && !retroactive
        {
            return Ok((false, None, RuleResolutionStatus::OutsideValidityWindow));
        }
        let Some(prior) = self.prior_trigger_claim(rule, trigger)? else {
            return Ok((false, None, RuleResolutionStatus::NoPriorValue));
        };
        let changed = prior.status != trigger.status
            || prior.polarity != trigger.polarity
            || prior.object_value != trigger.object_value;
        let reason = if changed {
            RuleResolutionStatus::OutsideValidityWindow
        } else {
            RuleResolutionStatus::NoValueChange
        };
        Ok((changed, changed.then_some(prior), reason))
    }

    /// The newest claim of the rule's trigger (same facet) from before the trigger's change.
    fn prior_trigger_claim(
        &self,
        rule: &RuleRecord,
        trigger: &ClaimRecord,
    ) -> ZResult<Option<ClaimRecord>> {
        let trigger_time = claim_effective_from_ms(trigger);
        Ok(self
            .prior_trigger_candidates(rule, trigger)?
            .into_iter()
            .filter(|claim| rule_trigger_matches_claim(rule, claim))
            .filter(|claim| claim.slot_facet == trigger.slot_facet)
            .filter(|claim| {
                temporal_position_is_after(
                    trigger_time,
                    trigger.source_sequence_no,
                    claim_effective_from_ms(claim),
                    claim.source_sequence_no,
                )
            })
            .reduce(|current, candidate| {
                if claim_is_newer(&candidate, &current) {
                    candidate
                } else {
                    current
                }
            }))
    }

    /// Claims that may hold the trigger's prior value: the trigger slots' versions (or claims)
    /// current just before the change, widened to every current claim and then re-canonicalised
    /// until one matches the rule's trigger.
    fn prior_trigger_candidates(
        &self,
        rule: &RuleRecord,
        trigger: &ClaimRecord,
    ) -> ZResult<Vec<ClaimRecord>> {
        let trigger_time = claim_effective_from_ms(trigger);
        let sequence_aware_change = trigger.source_sequence_no.is_some();
        let prior_at_ms = Some(if sequence_aware_change {
            trigger_time
        } else {
            trigger_time.saturating_sub(1)
        });
        let scope = &trigger.scope;
        let slot_ids = [rule.trigger_slot_id.clone(), trigger.slot_id.clone()]
            .into_iter()
            .flatten()
            .collect::<BTreeSet<_>>();
        let mut candidates = if !sequence_aware_change {
            self.scan_current_claims_for_slot_ids(scope, &slot_ids, usize::MAX, prior_at_ms)?
        } else if slot_ids.is_empty() {
            self.scan_claims(scope, usize::MAX, prior_at_ms)?
        } else {
            self.scan_claim_versions_for_slot_ids(scope, &slot_ids, usize::MAX, prior_at_ms)?
        };
        if !any_rule_trigger(rule, &candidates) {
            candidates.extend(self.scan_current_claims(scope, usize::MAX, prior_at_ms)?);
        }
        if any_rule_trigger(rule, &candidates) {
            return Ok(candidates);
        }
        let endpoint_names = candidates
            .iter()
            .filter_map(|claim| claim.subject.as_deref())
            .chain(rule.trigger_subject.as_deref())
            .chain(trigger.subject.as_deref())
            .collect::<Vec<_>>();
        let registry =
            self.canonical_registry_for_names_at(scope, endpoint_names, Some(trigger_time))?;
        Ok(resolve_current_claims(
            candidates
                .into_iter()
                .map(|claim| canonicalize_claim(claim, &registry))
                .collect(),
            usize::MAX,
        ))
    }
}

pub(super) fn merge_prior_trigger_evidence(claim: &mut ClaimRecord, prior: Option<&ClaimRecord>) {
    let Some(prior) = prior else {
        return;
    };
    claim.source_span_ids.extend(prior.source_span_ids.clone());
    claim.source_span_ids.sort();
    claim.source_span_ids.dedup();
    claim
        .source_episode_ids
        .extend(prior.source_episode_ids.clone());
    claim.source_episode_ids.sort();
    claim.source_episode_ids.dedup();
}

fn any_rule_trigger(rule: &RuleRecord, claims: &[ClaimRecord]) -> bool {
    claims
        .iter()
        .any(|claim| rule_trigger_matches_claim(rule, claim))
}

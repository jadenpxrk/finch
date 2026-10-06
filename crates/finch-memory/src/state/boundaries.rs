use super::*;

impl MemoryStore {
    pub(super) fn plan_batch_boundaries(
        &self,
        scope: &MemoryScope,
        claims: &[ClaimRecord],
        corrections: &[CorrectionRecord],
        rules: &[RuleRecord],
    ) -> ZResult<BoundaryPlan> {
        let rule_trigger_slot_ids = rules
            .iter()
            .filter_map(|rule| rule.trigger_slot_id.clone())
            .collect::<BTreeSet<_>>();
        let mut rule_activation_claims =
            self.scan_current_claims_for_slot_ids(scope, &rule_trigger_slot_ids, usize::MAX, None)?;
        rule_activation_claims.retain(|claim| claim.scope == *scope);
        let rule_start_trigger_claims = self.rule_start_trigger_claims(scope, rules)?;
        let mut valid_boundaries = batch_valid_boundaries(claims, corrections, rules);
        valid_boundaries.extend(rule_activation_claims.iter().map(claim_start_ms));
        valid_boundaries.sort_unstable();
        valid_boundaries.dedup();
        let correction_target_slot_ids = self.correction_target_slot_ids(scope, corrections)?;
        Ok(BoundaryPlan {
            valid_boundaries,
            rule_activation_claims,
            rule_start_trigger_claims,
            correction_target_slot_ids,
        })
    }

    /// For each reconciled rule start, the claims current on the triggers of the rules that
    /// start then.
    pub(super) fn rule_start_trigger_claims(
        &self,
        scope: &MemoryScope,
        rules: &[RuleRecord],
    ) -> ZResult<BTreeMap<i64, Vec<ClaimRecord>>> {
        let mut trigger_slots_by_rule_from = BTreeMap::<i64, BTreeSet<MemoryId>>::new();
        for rule in rules {
            if let (Some(from), Some(slot_id)) = (rule.valid_from_ms, rule.trigger_slot_id.clone())
            {
                trigger_slots_by_rule_from
                    .entry(from)
                    .or_default()
                    .insert(slot_id);
            }
        }
        trigger_slots_by_rule_from
            .iter()
            .map(|(from, slot_ids)| {
                let mut claims = self.scan_current_claims_for_slot_ids(
                    scope,
                    slot_ids,
                    usize::MAX,
                    Some(*from),
                )?;
                claims.retain(|claim| claim.scope == *scope);
                Ok((*from, claims))
            })
            .collect()
    }

    /// Slots the corrections target directly or through the claims they target.
    pub(super) fn correction_target_slot_ids(
        &self,
        scope: &MemoryScope,
        corrections: &[CorrectionRecord],
    ) -> ZResult<BTreeSet<MemoryId>> {
        let target_ids = corrections
            .iter()
            .flat_map(|correction| correction.target_ids.iter().cloned())
            .collect::<BTreeSet<_>>();
        let target_claims =
            self.claims_by_ids(scope, &target_ids.into_iter().collect::<Vec<_>>())?;
        Ok(corrections
            .iter()
            .filter_map(|correction| correction.target_slot_id.clone())
            .chain(target_claims.into_iter().filter_map(|claim| claim.slot_id))
            .collect())
    }

    /// Evaluates the rules at every boundary in valid-time order, writing each boundary's
    /// derived claims before the next boundary evaluates.
    pub(super) fn propagate_batch_boundaries(
        &self,
        inputs: &BoundaryInputs<'_>,
        claim_embeddings: &mut BTreeMap<MemoryId, Vec<f32>>,
    ) -> ZResult<BatchPropagation> {
        let mut changed_claims = Vec::new();
        let mut known_ids = inputs
            .claims
            .iter()
            .map(|claim| claim.id.clone())
            .collect::<BTreeSet<_>>();
        let mut rule_applications = Vec::new();
        let mut boundary_application_ranges = BTreeMap::new();
        for &boundary in &inputs.plan.valid_boundaries {
            let boundary_claims = self.claims_changed_at_boundary(inputs, boundary)?;
            let applications =
                self.rule_applications_at_boundary(inputs, boundary, &boundary_claims, &known_ids)?;
            changed_claims.extend(boundary_claims);
            let derived = applications
                .iter()
                .map(|application| &application.claim)
                .collect::<Vec<_>>();
            self.write_derived_claims(inputs.scope, &derived, claim_embeddings)?;
            known_ids.extend(derived.into_iter().map(|claim| claim.id.clone()));
            let start = rule_applications.len();
            rule_applications.extend(applications);
            boundary_application_ranges.insert(boundary, (start, rule_applications.len()));
        }
        dedupe_claims_by_id(&mut changed_claims);
        Ok(BatchPropagation {
            changed_claims,
            rule_applications,
            boundary_application_ranges,
        })
    }

    /// Claims whose state changes at `boundary`: batch claims starting or ending there, trigger
    /// claims starting there under an active reconciled rule, triggers current at a rule start,
    /// and the current claims of correction targets when a correction takes effect there.
    fn claims_changed_at_boundary(
        &self,
        inputs: &BoundaryInputs<'_>,
        boundary: i64,
    ) -> ZResult<Vec<ClaimRecord>> {
        let plan = inputs.plan;
        let mut changed = inputs
            .claims
            .iter()
            .filter(|claim| claim_start_ms(claim) == boundary)
            .cloned()
            .collect::<Vec<_>>();
        changed.extend(
            inputs
                .claims
                .iter()
                .filter(|claim| claim.valid_to_ms == Some(boundary))
                .map(|claim| expired_at(claim.clone(), boundary)),
        );
        let activated_triggers = plan.rule_activation_claims.iter().filter(|claim| {
            claim_start_ms(claim) == boundary
                && trigger_rule_active_at(inputs.rules, claim, boundary)
        });
        changed.extend(activated_triggers.cloned());
        if let Some(current_triggers) = plan.rule_start_trigger_claims.get(&boundary) {
            changed.extend(current_triggers.iter().cloned());
        }
        if inputs
            .corrections
            .iter()
            .any(|correction| correction.effective_at_ms == boundary)
        {
            changed.extend(
                self.scan_current_claims_for_slot_ids(
                    inputs.scope,
                    &plan.correction_target_slot_ids,
                    usize::MAX,
                    Some(boundary),
                )?
                .into_iter()
                .filter(|claim| claim.scope == *inputs.scope),
            );
        }
        dedupe_claims_by_id(&mut changed);
        Ok(changed)
    }

    /// Rule applications at `boundary`: the changed claims', plus the retraction of whatever a
    /// rule ending there had derived from its trigger.
    fn rule_applications_at_boundary(
        &self,
        inputs: &BoundaryInputs<'_>,
        boundary: i64,
        boundary_claims: &[ClaimRecord],
        known_ids: &BTreeSet<MemoryId>,
    ) -> ZResult<Vec<ResolvedRuleApplication>> {
        let mut applications = self.resolve_rules_for_changed_claims_at(
            boundary_claims,
            known_ids,
            inputs.max_rule_hops,
            Some(boundary),
        )?;
        let expiring_trigger_slot_ids = inputs
            .rules
            .iter()
            .filter(|rule| rule.valid_to_ms == Some(boundary))
            .filter_map(|rule| rule.trigger_slot_id.clone())
            .collect::<BTreeSet<_>>();
        if expiring_trigger_slot_ids.is_empty() {
            return Ok(applications);
        }
        let before_expiry = boundary.saturating_sub(1);
        let expired_triggers = self
            .scan_current_claims_for_slot_ids(
                inputs.scope,
                &expiring_trigger_slot_ids,
                usize::MAX,
                Some(before_expiry),
            )?
            .into_iter()
            .filter(|claim| claim.scope == *inputs.scope)
            .map(|claim| expired_at(claim, boundary))
            .collect::<Vec<_>>();
        applications.extend(self.resolve_rules_for_changed_claims_at(
            &expired_triggers,
            known_ids,
            inputs.max_rule_hops,
            Some(before_expiry),
        )?);
        retain_first_by_id(&mut applications, |application| &application.claim.id);
        Ok(applications)
    }

    fn write_derived_claims(
        &self,
        scope: &MemoryScope,
        derived: &[&ClaimRecord],
        claim_embeddings: &mut BTreeMap<MemoryId, Vec<f32>>,
    ) -> ZResult<()> {
        self.extend_claim_embeddings_from_sources(scope, derived, claim_embeddings)?;
        let derived_docs = claim_docs_with_embeddings(derived, claim_embeddings)?;
        upsert_many(&self.claims, derived_docs)
    }

    /// One batch is one write: every boundary shares its key. The versions it starts from are
    /// part of its identity, so repeating a batch never reuses a key.
    pub(super) fn batch_write_key(
        &self,
        scope: &MemoryScope,
        propagation_seed: &str,
        written: &BatchRecords<'_>,
        affected_slot_ids: &BTreeSet<MemoryId>,
    ) -> ZResult<String> {
        let mut write_ids = written
            .claims
            .iter()
            .map(|claim| claim.id.as_str())
            .chain(written.rules.iter().map(|rule| rule.id.as_str()))
            .chain(
                written
                    .corrections
                    .iter()
                    .map(|correction| correction.id.as_str()),
            )
            .chain(written.entities.iter().map(|entity| entity.id.as_str()))
            .chain(written.slot_aliases.iter().map(|alias| alias.id.as_str()))
            .collect::<Vec<_>>();
        write_ids.sort_unstable();
        let prior_ids = self
            .current_state_records_for_slot_ids(scope, affected_slot_ids)?
            .into_iter()
            .map(|record| record.id)
            .collect::<Vec<_>>();
        Ok(stable_hash_hex(&[
            propagation_seed,
            &write_ids.join("\u{0}"),
            &prior_ids.join("\u{0}"),
        ]))
    }

    /// Projects the affected slots once per boundary with that boundary's rule applications, or
    /// once at the current time when the batch has no boundary.
    pub(super) fn project_batch_boundaries(
        &self,
        scope: &MemoryScope,
        affected_slot_ids: &BTreeSet<MemoryId>,
        valid_boundaries: &[i64],
        propagation: &BatchPropagation,
        write: ProjectionWrite<'_>,
    ) -> ZResult<Vec<StateRecord>> {
        if valid_boundaries.is_empty() {
            return self.project_state_slots(
                scope,
                StateProjectionFrontier {
                    slot_ids: affected_slot_ids,
                    claims: None,
                    applications: &propagation.rule_applications,
                    valid_at_ms: None,
                    write,
                },
            );
        }
        let mut state_records = Vec::new();
        for &boundary in valid_boundaries {
            let applications = propagation
                .boundary_application_ranges
                .get(&boundary)
                .and_then(|&(start, end)| propagation.rule_applications.get(start..end))
                .unwrap_or_default();
            state_records.extend(self.project_state_slots(
                scope,
                StateProjectionFrontier {
                    slot_ids: affected_slot_ids,
                    claims: None,
                    applications,
                    valid_at_ms: Some(boundary),
                    write,
                },
            )?);
        }
        Ok(state_records)
    }
}

fn claim_start_ms(claim: &ClaimRecord) -> i64 {
    claim.valid_from_ms.unwrap_or(claim.observed_at_ms)
}

/// `claim` restated as having expired at `boundary`.
fn expired_at(mut claim: ClaimRecord, boundary: i64) -> ClaimRecord {
    claim.status = MemoryStatus::Expired;
    claim.observed_at_ms = boundary;
    claim.valid_from_ms = Some(boundary);
    claim.valid_to_ms = None;
    claim
}

/// Whether a rule triggered by the claim's slot is in force at `boundary`.
fn trigger_rule_active_at(rules: &[RuleRecord], claim: &ClaimRecord, boundary: i64) -> bool {
    rules.iter().any(|rule| {
        rule.trigger_slot_id.as_ref() == claim.slot_id.as_ref()
            && rule.valid_from_ms.is_none_or(|from| from <= boundary)
            && rule.valid_to_ms.is_none_or(|to| boundary < to)
    })
}

/// Starts and ends of the batch's claims, correction times, and reconciled rule windows.
fn batch_valid_boundaries(
    claims: &[ClaimRecord],
    corrections: &[CorrectionRecord],
    rules: &[RuleRecord],
) -> Vec<i64> {
    claims
        .iter()
        .map(claim_start_ms)
        .chain(claims.iter().filter_map(|claim| claim.valid_to_ms))
        .chain(
            corrections
                .iter()
                .map(|correction| correction.effective_at_ms),
        )
        .chain(rules.iter().filter_map(|rule| rule.valid_from_ms))
        .chain(rules.iter().filter_map(|rule| rule.valid_to_ms))
        .collect()
}

/// Every slot the batch touches: those of its changed and derived claims, its reconciled
/// rules' endpoints, and its corrections' targets.
pub(super) fn batch_affected_slot_ids(
    inputs: &BoundaryInputs<'_>,
    propagation: &BatchPropagation,
    derived_claims: &[ClaimRecord],
) -> BTreeSet<MemoryId> {
    propagation
        .changed_claims
        .iter()
        .chain(derived_claims)
        .filter_map(|claim| claim.slot_id.clone())
        .chain(rule_endpoint_slot_ids(inputs.rules))
        .chain(inputs.plan.correction_target_slot_ids.iter().cloned())
        .collect()
}

fn rule_endpoint_slot_ids(rules: &[RuleRecord]) -> impl Iterator<Item = MemoryId> + '_ {
    rules
        .iter()
        .filter_map(|rule| rule.trigger_slot_id.clone())
        .chain(rules.iter().filter_map(|rule| rule.target_slot_id.clone()))
}

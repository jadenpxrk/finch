use super::*;

impl MemoryStore {
    pub(in crate::store) fn dependency_closure(
        &self,
        scope: &MemoryScope,
        selected_slot_ids: &BTreeSet<MemoryId>,
        subject_wide_target_slot_ids: &BTreeSet<MemoryId>,
        selected_rule_ids: &BTreeSet<MemoryId>,
        at_ms: Option<i64>,
        max_hops: usize,
    ) -> ZResult<(Vec<RuleRecord>, BTreeSet<MemoryId>)> {
        if max_hops == 0 || (selected_slot_ids.is_empty() && selected_rule_ids.is_empty()) {
            return Ok((Vec::new(), BTreeSet::new()));
        }
        let rules = self
            .scan_rules(scope, usize::MAX, at_ms)?
            .into_iter()
            .filter(rule_endpoints_bound)
            .collect::<Vec<_>>();
        let mut frontier = selected_slot_ids.clone();
        let mut closure = DependencyClosure {
            slot_ids: selected_slot_ids.clone(),
            rule_targets: BTreeSet::new(),
            rules: Vec::new(),
        };
        let subject_wide_target_slots = self
            .scan_slots(scope, usize::MAX, at_ms)?
            .into_iter()
            .filter(|slot| subject_wide_target_slot_ids.contains(&slot.slot_key))
            .collect::<Vec<_>>();

        for _ in 0..max_hops {
            let mut next = BTreeSet::new();
            for rule in &rules {
                let candidates = closure_candidates(
                    rule,
                    &frontier,
                    selected_rule_ids,
                    &subject_wide_target_slots,
                );
                for candidate in candidates {
                    closure.add(candidate, &mut next);
                }
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }
        closure.rules.sort_by(|a, b| a.id.cmp(&b.id));
        Ok((closure.rules, closure.slot_ids))
    }

    /// Completes the read set's dependency chains once per scope: a rule completes over its own
    /// scope's claims, so a read over several scopes never pairs one scope's rule with another's
    /// trigger or target.
    pub(in crate::store) fn complete_target_dependency_chains(
        &self,
        rules: &[RuleRecord],
        available_claims: &[ClaimRecord],
        existing_applications: &[ResolvedRuleApplication],
        max_hops: usize,
    ) -> ZResult<DependencyChainResolution> {
        let scopes = rules
            .iter()
            .map(|rule| &rule.scope)
            .chain(available_claims.iter().map(|claim| &claim.scope))
            .chain(
                existing_applications
                    .iter()
                    .map(|application| &application.claim.scope),
            )
            .collect::<BTreeSet<_>>();
        let mut resolution = DependencyChainResolution {
            applications: Vec::new(),
            projected_claims: Vec::new(),
            suppressed_claim_ids: BTreeSet::new(),
            outcomes: Vec::new(),
        };
        for scope in scopes {
            let rules = rules
                .iter()
                .filter(|rule| rule.scope == *scope)
                .cloned()
                .collect::<Vec<_>>();
            let claims = available_claims
                .iter()
                .filter(|claim| claim.scope == *scope)
                .cloned()
                .collect::<Vec<_>>();
            let applications = existing_applications
                .iter()
                .filter(|application| application.claim.scope == *scope)
                .cloned()
                .collect::<Vec<_>>();
            let scoped =
                self.complete_scope_dependency_chains(&rules, &claims, &applications, max_hops)?;
            resolution.applications.extend(scoped.applications);
            resolution.projected_claims.extend(scoped.projected_claims);
            resolution
                .suppressed_claim_ids
                .extend(scoped.suppressed_claim_ids);
            resolution.outcomes.extend(scoped.outcomes);
        }
        resolution.outcomes.sort_by(|a, b| {
            a.rule_id
                .cmp(&b.rule_id)
                .then_with(|| a.target_slot_id.cmp(&b.target_slot_id))
        });
        Ok(resolution)
    }

    fn complete_scope_dependency_chains(
        &self,
        rules: &[RuleRecord],
        available_claims: &[ClaimRecord],
        existing_applications: &[ResolvedRuleApplication],
        max_hops: usize,
    ) -> ZResult<DependencyChainResolution> {
        let mut completion = ChainCompletion::new(rules, available_claims, existing_applications);
        for hop in 0..max_hops {
            let mut progressed = false;
            for rule in rules {
                progressed |= completion.apply_rule(self, rule, hop)?;
            }
            completion.suppress_redundant_unsupported();
            completion.drop_superseded_unresolved();
            let newly_unsupported_slot_ids = completion.mark_missing_trigger_targets();
            progressed |= !newly_unsupported_slot_ids.is_empty();
            completion.invalidate_derivations_from(&newly_unsupported_slot_ids);
            if !progressed {
                break;
            }
        }
        Ok(completion.finish())
    }
}

/// Working state of the read-time completion of a read set's dependency chains.
struct ChainCompletion<'r> {
    rules: &'r [RuleRecord],
    applications: Vec<ResolvedRuleApplication>,
    /// Unsupported claims materialized for targets whose trigger is missing.
    unresolved_claims: Vec<ClaimRecord>,
    suppressed_claim_ids: BTreeSet<MemoryId>,
    /// Current claims, including every claim applied so far.
    claims: Vec<ClaimRecord>,
    known_ids: BTreeSet<MemoryId>,
    outcomes: BTreeMap<OutcomeKey, RuleResolutionOutcome>,
}

/// A rule outcome is kept per rule and target slot.
type OutcomeKey = (MemoryId, Option<MemoryId>);

impl<'r> ChainCompletion<'r> {
    fn new(
        rules: &'r [RuleRecord],
        available_claims: &[ClaimRecord],
        existing_applications: &[ResolvedRuleApplication],
    ) -> Self {
        let mut claims = available_claims.to_vec();
        claims.extend(
            existing_applications
                .iter()
                .map(|application| application.claim.clone()),
        );
        let claims = resolve_current_claims(claims, usize::MAX);
        let known_ids = claims
            .iter()
            .map(|claim| claim.id.clone())
            .collect::<BTreeSet<_>>();
        let outcomes = existing_applications
            .iter()
            .map(|application| {
                let outcome =
                    resolution_outcome(rule_of_application(rules, application), application);
                (
                    (outcome.rule_id.clone(), outcome.target_slot_id.clone()),
                    outcome,
                )
            })
            .collect();
        Self {
            rules,
            applications: existing_applications.to_vec(),
            unresolved_claims: Vec::new(),
            suppressed_claim_ids: BTreeSet::new(),
            claims,
            known_ids,
            outcomes,
        }
    }

    fn resolve_claims(&mut self) {
        self.claims = resolve_current_claims(std::mem::take(&mut self.claims), usize::MAX);
    }

    fn forget_claims(&mut self, claim_ids: &BTreeSet<MemoryId>) {
        self.claims.retain(|claim| !claim_ids.contains(&claim.id));
        for claim_id in claim_ids {
            self.known_ids.remove(claim_id);
        }
    }

    fn record_unresolved(
        &mut self,
        outcome_key: OutcomeKey,
        rule: &RuleRecord,
        status: RuleResolutionStatus,
    ) {
        self.outcomes
            .insert(outcome_key, unresolved_rule_outcome(rule, status));
    }

    /// Fires one rule whose outcome is not applied yet. Returns whether it added a new claim.
    fn apply_rule(&mut self, store: &MemoryStore, rule: &RuleRecord, hop: usize) -> ZResult<bool> {
        let outcome_key = (rule.id.clone(), rule.target_slot_id.clone());
        if self
            .outcomes
            .get(&outcome_key)
            .is_some_and(|outcome| outcome.status.is_applied())
        {
            return Ok(false);
        }
        if !matches!(rule.target_match, RuleTargetMatch::ExactSlot) {
            self.outcomes.entry(outcome_key).or_insert_with(|| {
                unresolved_rule_outcome(rule, RuleResolutionStatus::NotApplicable)
            });
            return Ok(false);
        }
        let Some(trigger) = self
            .claims
            .iter()
            .find(|claim| rule_trigger_matches_claim(rule, claim))
            .cloned()
        else {
            self.record_unresolved(outcome_key, rule, RuleResolutionStatus::MissingTrigger);
            return Ok(false);
        };
        let target = self
            .claims
            .iter()
            .find(|claim| rule_target_matches_claim(rule, claim));
        let (activated, prior_trigger, block_reason) = store
            .rule_activation_evidence_with_reason_for_target(
                rule,
                &trigger,
                target.map(claim_effective_from_ms),
            )?;
        if !activated {
            self.record_unresolved(outcome_key, rule, block_reason);
            return Ok(false);
        }
        let firing = RuleFiring::new(rule, &trigger, target).materialize(&self.known_ids)?;
        let mut claim = match firing {
            Ok(Some(claim)) => claim,
            Ok(None) => return Ok(false),
            Err(status) => {
                self.record_unresolved(outcome_key, rule, status);
                return Ok(false);
            }
        };
        merge_prior_trigger_evidence(&mut claim, prior_trigger.as_ref());
        let application = ResolvedRuleApplication {
            rule_id: rule.id.clone(),
            trace_id: dependency_trace_id(&rule.id, &trigger.id, &claim.id),
            trigger_claim_id: trigger.id,
            prior_trigger_claim_id: prior_trigger.map(|claim| claim.id),
            claim,
            target_subject_key: rule.target_subject_key.clone(),
            target_predicate_key: rule.target_predicate_key.clone(),
            hop: hop + 1,
            parent_trace_ids: Vec::new(),
        };
        Ok(self.record_application(outcome_key, rule, application))
    }

    /// Records a rule's application as its outcome and, when its claim is new, adds both.
    fn record_application(
        &mut self,
        outcome_key: OutcomeKey,
        rule: &RuleRecord,
        application: ResolvedRuleApplication,
    ) -> bool {
        let is_new_claim = self.known_ids.insert(application.claim.id.clone());
        self.outcomes
            .insert(outcome_key, resolution_outcome(Some(rule), &application));
        if !is_new_claim {
            return false;
        }
        self.claims.push(application.claim.clone());
        self.applications.push(application);
        self.resolve_claims();
        true
    }

    /// Drops unsupported applications that restate a transition a derivation already covers.
    fn suppress_redundant_unsupported(&mut self) {
        let redundant_claim_ids =
            redundant_same_transition_unsupported_claim_ids(self.rules, &self.applications);
        if redundant_claim_ids.is_empty() {
            return;
        }
        self.suppressed_claim_ids
            .extend(redundant_claim_ids.iter().cloned());
        self.applications
            .retain(|application| !redundant_claim_ids.contains(&application.claim.id));
        self.forget_claims(&redundant_claim_ids);
        for outcome in self.outcomes.values_mut() {
            let redundant = outcome
                .resolved_claim_id
                .as_ref()
                .is_some_and(|claim_id| redundant_claim_ids.contains(claim_id));
            if matches!(outcome.status, RuleResolutionStatus::AppliedUnsupported) && redundant {
                outcome.status = RuleResolutionStatus::NotApplicable;
                outcome.resolved_claim_id = None;
            }
        }
        self.resolve_claims();
    }

    /// Drops missing-trigger unsupported claims whose target a derivation has since resolved.
    fn drop_superseded_unresolved(&mut self) {
        let derived_target_slot_ids = self
            .outcomes
            .values()
            .filter(|outcome| matches!(outcome.status, RuleResolutionStatus::AppliedDerived))
            .filter_map(|outcome| outcome.target_slot_id.clone())
            .collect::<BTreeSet<_>>();
        let superseded_ids = self
            .unresolved_claims
            .iter()
            .filter(|claim| derived_target_slot_ids.contains(&claim_slot_lifecycle_key(claim)))
            .map(|claim| claim.id.clone())
            .collect::<BTreeSet<_>>();
        if superseded_ids.is_empty() {
            return;
        }
        self.unresolved_claims
            .retain(|claim| !superseded_ids.contains(&claim.id));
        self.forget_claims(&superseded_ids);
        self.resolve_claims();
    }

    /// Only a derivation (the target is a function of the trigger) makes an unknown trigger
    /// retire an earlier target assertion. A mark_unsupported rule waits for an observed
    /// transition of its trigger; a trigger that was never observed is not one. Returns the
    /// targets newly marked unsupported.
    fn mark_missing_trigger_targets(&mut self) -> BTreeSet<MemoryId> {
        let mut newly_unsupported_slot_ids = BTreeSet::new();
        for (target_slot_id, missing_rules) in self.missing_trigger_rules_by_target() {
            if self.target_resolved(&target_slot_id) {
                continue;
            }
            let Some(claim) = self.missing_trigger_claim(&target_slot_id, &missing_rules) else {
                continue;
            };
            if !self.known_ids.insert(claim.id.clone()) {
                continue;
            }
            for rule in missing_rules {
                let outcome = RuleResolutionOutcome {
                    rule_id: rule.id.clone(),
                    trigger_slot_id: rule.trigger_slot_id.clone(),
                    target_slot_id: Some(target_slot_id.clone()),
                    status: RuleResolutionStatus::AppliedUnsupported,
                    resolved_claim_id: Some(claim.id.clone()),
                };
                let outcome_key = (rule.id.clone(), Some(target_slot_id.clone()));
                self.outcomes.insert(outcome_key, outcome);
            }
            self.claims.push(claim.clone());
            self.resolve_claims();
            self.unresolved_claims.push(claim);
            newly_unsupported_slot_ids.insert(target_slot_id);
        }
        newly_unsupported_slot_ids
    }

    fn missing_trigger_rules_by_target(&self) -> BTreeMap<MemoryId, Vec<&'r RuleRecord>> {
        let mut grouped = BTreeMap::<MemoryId, Vec<&RuleRecord>>::new();
        for rule in self.rules {
            let Some(target_slot_id) = rule.target_slot_id.as_ref() else {
                continue;
            };
            let outcome_key = (rule.id.clone(), Some(target_slot_id.clone()));
            if self.outcomes.get(&outcome_key).is_some_and(|outcome| {
                matches!(outcome.status, RuleResolutionStatus::MissingTrigger)
            }) {
                grouped
                    .entry(target_slot_id.clone())
                    .or_default()
                    .push(rule);
            }
        }
        grouped
    }

    fn target_resolved(&self, target_slot_id: &MemoryId) -> bool {
        self.outcomes.values().any(|outcome| {
            outcome.target_slot_id.as_ref() == Some(target_slot_id) && outcome.status.is_applied()
        })
    }

    /// The unsupported claim a target gets when its rules' triggers are missing, unless the
    /// current target is independently re-asserted after the dependency took hold.
    fn missing_trigger_claim(
        &self,
        target_slot_id: &MemoryId,
        missing_rules: &[&RuleRecord],
    ) -> Option<ClaimRecord> {
        let current_target = newest_claim(
            self.claims
                .iter()
                .filter(|claim| claim_slot_lifecycle_key(claim) == *target_slot_id),
        );
        let (boundary_ms, boundary_sequence_no) = missing_rules
            .iter()
            .filter_map(|rule| {
                rule.valid_from_ms
                    .map(|at_ms| (at_ms, rule.source_sequence_no))
            })
            .max()
            .or_else(|| {
                current_target
                    .map(|claim| (claim_effective_from_ms(claim), claim.source_sequence_no))
            })?;
        if current_target.is_some_and(|claim| {
            independently_supports_slot_after(claim, boundary_ms, boundary_sequence_no)
        }) {
            return None;
        }
        materialize_missing_trigger_state(
            target_slot_id,
            missing_rules,
            current_target,
            boundary_ms,
            boundary_sequence_no,
        )
    }

    /// Retracts the derivations whose trigger slot was just marked unsupported.
    fn invalidate_derivations_from(&mut self, unsupported_slot_ids: &BTreeSet<MemoryId>) {
        if unsupported_slot_ids.is_empty() {
            return;
        }
        let invalidated_claim_ids = self
            .outcomes
            .values()
            .filter(|outcome| matches!(outcome.status, RuleResolutionStatus::AppliedDerived))
            .filter(|outcome| {
                outcome
                    .trigger_slot_id
                    .as_ref()
                    .is_some_and(|slot_id| unsupported_slot_ids.contains(slot_id))
            })
            .filter_map(|outcome| outcome.resolved_claim_id.clone())
            .collect::<BTreeSet<_>>();
        if invalidated_claim_ids.is_empty() {
            return;
        }
        self.suppressed_claim_ids
            .extend(invalidated_claim_ids.iter().cloned());
        self.outcomes.retain(|_, outcome| {
            !outcome
                .resolved_claim_id
                .as_ref()
                .is_some_and(|claim_id| invalidated_claim_ids.contains(claim_id))
        });
        self.applications
            .retain(|application| !invalidated_claim_ids.contains(&application.claim.id));
        self.forget_claims(&invalidated_claim_ids);
        self.resolve_claims();
    }

    fn finish(mut self) -> DependencyChainResolution {
        for rule in self.rules {
            self.outcomes
                .entry((rule.id.clone(), rule.target_slot_id.clone()))
                .or_insert_with(|| {
                    unresolved_rule_outcome(rule, RuleResolutionStatus::NotApplicable)
                });
        }
        let mut outcomes = self.outcomes.into_values().collect::<Vec<_>>();
        outcomes.sort_by(|a, b| {
            a.rule_id
                .cmp(&b.rule_id)
                .then_with(|| a.target_slot_id.cmp(&b.target_slot_id))
        });
        DependencyChainResolution {
            applications: self.applications,
            projected_claims: self.unresolved_claims,
            suppressed_claim_ids: self.suppressed_claim_ids,
            outcomes,
        }
    }
}

/// The rule an existing application came from: the one on the claim's slot, else any with
/// its id.
fn rule_of_application<'a>(
    rules: &'a [RuleRecord],
    application: &ResolvedRuleApplication,
) -> Option<&'a RuleRecord> {
    rules
        .iter()
        .find(|rule| {
            rule.id == application.rule_id && rule.target_slot_id == application.claim.slot_id
        })
        .or_else(|| rules.iter().find(|rule| rule.id == application.rule_id))
}

/// What firing a rule yields: a claim, nothing, or the status recorded for not firing.
type RuleFiringResult = Result<Option<ClaimRecord>, RuleResolutionStatus>;

/// One activated exact-slot rule with the trigger and target claims it fires on.
struct RuleFiring<'a> {
    rule: &'a RuleRecord,
    trigger: &'a ClaimRecord,
    target: Option<&'a ClaimRecord>,
    trigger_time: i64,
    rule_time: i64,
    /// The trigger changed after the rule was stated, or the rule reaches back over a
    /// transition that postdates the target's value.
    governs: bool,
}

impl<'a> RuleFiring<'a> {
    fn new(
        rule: &'a RuleRecord,
        trigger: &'a ClaimRecord,
        target: Option<&'a ClaimRecord>,
    ) -> Self {
        let trigger_time = claim_effective_from_ms(trigger);
        let rule_time = rule.valid_from_ms.unwrap_or(i64::MIN);
        let trigger_after_rule = temporal_position_is_after(
            trigger_time,
            trigger.source_sequence_no,
            rule_time,
            rule.source_sequence_no,
        );
        // The stated dependency reaches a transition that happened before the rule was uttered
        // whenever the target's value predates that transition.
        let retroactively_governs = !trigger_after_rule
            && target.is_some_and(|claim| claim_effective_from_ms(claim) < trigger_time);
        Self {
            rule,
            trigger,
            target,
            trigger_time,
            rule_time,
            governs: trigger_after_rule || retroactively_governs,
        }
    }

    fn materialize(&self, known_ids: &BTreeSet<MemoryId>) -> ZResult<RuleFiringResult> {
        let trigger_supports_value = matches!(self.trigger.status, MemoryStatus::Active)
            && matches!(self.trigger.polarity, ClaimPolarity::Affirmative)
            && self.trigger.object_value.is_some();
        match self.rule.action {
            RuleAction::DeriveValue if trigger_supports_value => self.derive_from_value(),
            RuleAction::DeriveValue => self.derive_without_value(),
            RuleAction::MarkUnsupported => self.mark_unsupported(known_ids),
        }
    }

    fn derive_from_value(&self) -> ZResult<RuleFiringResult> {
        if rule_trigger_interval(self.rule, self.trigger).is_none() {
            return Ok(Err(RuleResolutionStatus::OutsideValidityWindow));
        }
        Ok(Ok(materialize_rule_application(
            self.trigger,
            self.rule,
            false,
            &BTreeSet::new(),
        )?))
    }

    /// A trigger with no value leaves the target unsupported, unless the target was
    /// independently re-asserted after the value was lost or the rule does not govern.
    fn derive_without_value(&self) -> ZResult<RuleFiringResult> {
        let (boundary_ms, boundary_sequence_no) = if matches!(
            (self.trigger.status, self.trigger.polarity),
            (MemoryStatus::Active, ClaimPolarity::Affirmative)
        ) {
            (self.rule_time, self.rule.source_sequence_no)
        } else {
            (self.trigger_time, self.trigger.source_sequence_no)
        };
        let independently_reasserted = self.target.is_some_and(|claim| {
            independently_supports_slot_after(claim, boundary_ms, boundary_sequence_no)
        });
        if independently_reasserted || !self.governs {
            return Ok(Err(RuleResolutionStatus::MissingTriggerValue));
        }
        Ok(Ok(materialize_rule_application(
            self.trigger,
            self.rule,
            true,
            &BTreeSet::new(),
        )?))
    }

    /// An unsupported mark already applied stands; otherwise it applies only over a target
    /// the trigger's transition made stale.
    fn mark_unsupported(&self, known_ids: &BTreeSet<MemoryId>) -> ZResult<RuleFiringResult> {
        let materialized =
            materialize_rule_application(self.trigger, self.rule, true, &BTreeSet::new())?;
        if materialized
            .as_ref()
            .is_some_and(|claim| known_ids.contains(&claim.id))
        {
            return Ok(Ok(materialized));
        }
        let stale_target = self.target.is_some_and(|claim| {
            matches!(claim.status, MemoryStatus::Active)
                && matches!(claim.polarity, ClaimPolarity::Affirmative)
                && temporal_position_is_after(
                    self.trigger_time,
                    self.trigger.source_sequence_no,
                    claim_effective_from_ms(claim),
                    claim.source_sequence_no,
                )
        });
        if stale_target && self.governs {
            Ok(Ok(materialized))
        } else {
            Ok(Err(RuleResolutionStatus::NotApplicable))
        }
    }
}

fn redundant_same_transition_unsupported_claim_ids(
    rules: &[RuleRecord],
    applications: &[ResolvedRuleApplication],
) -> BTreeSet<MemoryId> {
    let rules_by_id = rules
        .iter()
        .map(|rule| (rule.id.as_str(), rule))
        .collect::<BTreeMap<_, _>>();
    let derived_transitions = applications
        .iter()
        .filter_map(|application| {
            let rule = rules_by_id.get(application.rule_id.as_str())?;
            (matches!(rule.action, RuleAction::DeriveValue)
                && matches!(application.claim.status, MemoryStatus::Active)
                && matches!(application.claim.polarity, ClaimPolarity::Affirmative)
                && application.claim.object_value.is_some())
            .then(|| {
                (
                    rule.trigger_slot_id.clone(),
                    application.claim.slot_id.clone(),
                    application.trigger_claim_id.clone(),
                )
            })
        })
        .collect::<BTreeSet<_>>();

    applications
        .iter()
        .filter_map(|application| {
            let rule = rules_by_id.get(application.rule_id.as_str())?;
            if !matches!(rule.action, RuleAction::MarkUnsupported) {
                return None;
            }
            derived_transitions
                .contains(&(
                    rule.trigger_slot_id.clone(),
                    application.claim.slot_id.clone(),
                    application.trigger_claim_id.clone(),
                ))
                .then(|| application.claim.id.clone())
        })
        .collect()
}

fn independently_supports_slot_after(
    claim: &ClaimRecord,
    boundary_ms: i64,
    boundary_sequence_no: Option<i64>,
) -> bool {
    !matches!(claim.claim_kind, ClaimKind::Relationship | ClaimKind::Event)
        && matches!(claim.status, MemoryStatus::Active)
        && matches!(claim.polarity, ClaimPolarity::Affirmative)
        && claim.object_value.is_some()
        && crate::is_direct_state_claim(claim)
        && temporal_position_is_after(
            claim_effective_from_ms(claim),
            claim.source_sequence_no,
            boundary_ms,
            boundary_sequence_no,
        )
}

fn missing_trigger_claim_text(subject: &str, predicate: &str, rule_ids: &[MemoryId]) -> String {
    serde_json::json!({
        "subject": subject,
        "predicate": predicate,
        "support_state": "unsupported",
        "reason": "missing_dependency_trigger",
        "rule_ids": rule_ids,
    })
    .to_string()
}

fn missing_trigger_claim_id(
    target_slot_id: &str,
    rule_ids: &[MemoryId],
    dependency_boundary: i64,
    dependency_sequence_no: Option<i64>,
) -> MemoryId {
    let hash = stable_hash_hex(&[
        "missing_dependency_trigger",
        target_slot_id,
        &rule_ids.join("\u{1f}"),
        &dependency_boundary.to_string(),
        &dependency_sequence_no.unwrap_or(i64::MIN).to_string(),
    ]);
    format!("derived_{hash}")
}

fn materialize_missing_trigger_state(
    target_slot_id: &MemoryId,
    rules: &[&RuleRecord],
    current_target: Option<&ClaimRecord>,
    dependency_boundary: i64,
    dependency_sequence_no: Option<i64>,
) -> Option<ClaimRecord> {
    let representative = rules
        .iter()
        .copied()
        .max_by_key(|rule| (rule.valid_from_ms, &rule.id))?;
    let subject = current_target
        .and_then(|claim| claim.subject.clone())
        .or_else(|| representative.target_subject.clone())?;
    let predicate = current_target
        .and_then(|claim| claim.predicate.clone())
        .or_else(|| representative.target_predicate.clone())?;
    let rule_ids = sorted_unique_ids(rules.iter().map(|rule| &rule.id));
    let source_span_ids = sorted_unique_ids(
        rules.iter().flat_map(|rule| &rule.source_span_ids).chain(
            current_target
                .into_iter()
                .flat_map(|claim| &claim.source_span_ids),
        ),
    );
    let source_episode_ids = sorted_unique_ids(
        rules
            .iter()
            .flat_map(|rule| &rule.source_episode_ids)
            .chain(
                current_target
                    .into_iter()
                    .flat_map(|claim| &claim.source_episode_ids),
            ),
    );
    let id = missing_trigger_claim_id(
        target_slot_id,
        &rule_ids,
        dependency_boundary,
        dependency_sequence_no,
    );
    Some(ClaimRecord {
        id,
        scope: representative.scope.clone(),
        status: MemoryStatus::Active,
        visibility: current_target
            .map(|claim| claim.visibility)
            .unwrap_or(representative.visibility),
        policy_tags: representative.policy_tags.clone(),
        claim_text: missing_trigger_claim_text(&subject, &predicate, &rule_ids),
        subject: Some(subject),
        predicate: Some(predicate),
        object_value: None,
        subject_entity_id: current_target
            .and_then(|claim| claim.subject_entity_id.clone())
            .or_else(|| representative.target_subject_entity_id.clone()),
        slot_id: Some(target_slot_id.clone()),
        slot_facet: current_target.and_then(|claim| claim.slot_facet.clone()),
        claim_kind: current_target
            .map(|claim| claim.claim_kind)
            .unwrap_or(ClaimKind::Fact),
        polarity: ClaimPolarity::Uncertain,
        source_span_ids,
        source_episode_ids,
        source_sequence_no: dependency_sequence_no,
        asserted_by: "dependency_resolution".to_string(),
        extractor_version: None,
        confidence: representative.confidence,
        observed_at_ms: dependency_boundary,
        valid_from_ms: Some(dependency_boundary),
        valid_to_ms: rules.iter().filter_map(|rule| rule.valid_to_ms).min(),
        correction_ids: Vec::new(),
    })
}

fn rule_target_subject_matches_slot(rule: &RuleRecord, slot: &CanonicalSlotRecord) -> bool {
    match (
        rule.target_subject_entity_id.as_deref(),
        slot.subject_entity_id.as_deref(),
    ) {
        (Some(expected), Some(actual)) => expected == actual,
        (Some(_), None) => false,
        (None, _) => rule.target_subject_key == slot.subject_key,
    }
}

fn concretize_subject_wide_rule_target(
    rule: &RuleRecord,
    slot: &CanonicalSlotRecord,
) -> RuleRecord {
    let mut concrete = rule.clone();
    concrete.target_subject_key = slot.subject_key.clone();
    concrete.target_predicate_key = slot.predicate_key.clone();
    concrete.target_subject_entity_id = slot.subject_entity_id.clone();
    concrete.target_slot_id = Some(slot.slot_key.clone());
    concrete.target_subject = slot.subject.clone().or(concrete.target_subject);
    concrete.target_predicate = slot.predicate.clone();
    concrete.target_match = RuleTargetMatch::ExactSlot;
    concrete.target_binding_status = RuleBindingStatus::Bound;
    concrete
}

fn resolution_outcome(
    rule: Option<&RuleRecord>,
    application: &ResolvedRuleApplication,
) -> RuleResolutionOutcome {
    RuleResolutionOutcome {
        rule_id: application.rule_id.clone(),
        trigger_slot_id: rule.and_then(|rule| rule.trigger_slot_id.clone()),
        target_slot_id: application
            .claim
            .slot_id
            .clone()
            .or_else(|| rule.and_then(|rule| rule.target_slot_id.clone())),
        status: if matches!(application.claim.polarity, ClaimPolarity::Uncertain) {
            RuleResolutionStatus::AppliedUnsupported
        } else {
            RuleResolutionStatus::AppliedDerived
        },
        resolved_claim_id: Some(application.claim.id.clone()),
    }
}

fn unresolved_rule_outcome(
    rule: &RuleRecord,
    status: RuleResolutionStatus,
) -> RuleResolutionOutcome {
    RuleResolutionOutcome {
        rule_id: rule.id.clone(),
        trigger_slot_id: rule.trigger_slot_id.clone(),
        target_slot_id: rule.target_slot_id.clone(),
        status,
        resolved_claim_id: None,
    }
}

/// Rules and slots accumulated by a dependency closure.
struct DependencyClosure {
    slot_ids: BTreeSet<MemoryId>,
    rule_targets: BTreeSet<(MemoryId, Option<MemoryId>)>,
    rules: Vec<RuleRecord>,
}

impl DependencyClosure {
    /// Adds a rule once per target; a trigger slot new to the closure joins the next frontier.
    fn add(&mut self, rule: RuleRecord, next_frontier: &mut BTreeSet<MemoryId>) {
        if !self
            .rule_targets
            .insert((rule.id.clone(), rule.target_slot_id.clone()))
        {
            return;
        }
        if let Some(slot_id) = rule.target_slot_id.as_ref() {
            self.slot_ids.insert(slot_id.clone());
        }
        if let Some(slot_id) = rule.trigger_slot_id.as_ref() {
            if self.slot_ids.insert(slot_id.clone()) {
                next_frontier.insert(slot_id.clone());
            }
        }
        self.rules.push(rule);
    }
}

/// The closure entries one rule contributes this hop: an exact-slot rule resolving a frontier
/// slot (or requested by id), or a subject-wide unsupported mark concretised per target slot.
fn closure_candidates(
    rule: &RuleRecord,
    frontier: &BTreeSet<MemoryId>,
    requested_rule_ids: &BTreeSet<MemoryId>,
    subject_wide_target_slots: &[CanonicalSlotRecord],
) -> Vec<RuleRecord> {
    if matches!(rule.target_match, RuleTargetMatch::ExactSlot) {
        let resolves_frontier = rule
            .target_slot_id
            .as_ref()
            .is_some_and(|slot_id| frontier.contains(slot_id));
        if resolves_frontier || requested_rule_ids.contains(&rule.id) {
            return vec![rule.clone()];
        }
        return Vec::new();
    }
    if !matches!(rule.action, RuleAction::MarkUnsupported) {
        return Vec::new();
    }
    subject_wide_target_slots
        .iter()
        .filter(|slot| rule_target_subject_matches_slot(rule, slot))
        .map(|slot| concretize_subject_wide_rule_target(rule, slot))
        .collect()
}

/// Whether an active, bound rule in force at `at_ms` targets `slot`, exactly or subject-wide.
pub(super) fn rule_governs_target_slot(
    rule: &RuleRecord,
    scope: &MemoryScope,
    slot: &CanonicalSlot,
    at_ms: Option<i64>,
) -> bool {
    let subject_wide_match = matches!(rule.target_match, RuleTargetMatch::AnyActiveSlotForSubject)
        && rule
            .target_subject_entity_id
            .as_deref()
            .unwrap_or(&rule.target_subject_key)
            == slot.subject_key;
    rule.scope.matches_filter(scope)
        && matches!(rule.status, MemoryStatus::Active)
        && rule_bound_and_in_force(rule, at_ms)
        && (rule_target_slot(rule) == *slot || subject_wide_match)
}

use super::*;

impl MemoryStore {
    /// Fires a newly stored rule on its trigger's current claims and projects what it derives.
    /// A rule whose trigger has no claim and whose endpoint slots already exist changes nothing.
    pub(super) fn propagate_new_rule(&self, record: &RuleRecord) -> ZResult<()> {
        let trigger_slot_ids = record
            .trigger_slot_id
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        if trigger_slot_ids.is_empty() {
            return Ok(());
        }
        let valid_at_ms = record.valid_from_ms;
        let mut changed_claims = self.scan_current_claims_for_slot_ids(
            &record.scope,
            &trigger_slot_ids,
            usize::MAX,
            valid_at_ms,
        )?;
        changed_claims.retain(|claim| claim.scope == record.scope);
        if changed_claims.is_empty() && self.rule_endpoint_slots_exist(record)? {
            return Ok(());
        }
        let known_ids = changed_claims
            .iter()
            .map(|claim| claim.id.clone())
            .collect::<BTreeSet<_>>();
        let applications = self.resolve_rules_for_changed_claims_at(
            &changed_claims,
            &known_ids,
            MAX_RULE_HOPS,
            valid_at_ms,
        )?;
        self.store_and_project_applications(
            &record.scope,
            record.target_slot_id.as_ref(),
            &applications,
            valid_at_ms,
            &record.id,
        )
    }

    fn rule_endpoint_slots_exist(&self, record: &RuleRecord) -> ZResult<bool> {
        let endpoint_slot_ids = record
            .trigger_slot_id
            .iter()
            .chain(record.target_slot_id.iter())
            .cloned()
            .collect::<BTreeSet<_>>();
        let existing_slot_ids = self
            .current_slots_for_slot_keys(&record.scope, &endpoint_slot_ids)?
            .into_iter()
            .map(|slot| slot.slot_key)
            .collect::<BTreeSet<_>>();
        Ok(endpoint_slot_ids.is_subset(&existing_slot_ids))
    }

    pub(crate) fn resolve_rules_for_changed_claims(
        &self,
        changed_claims: &[ClaimRecord],
        existing_ids: &BTreeSet<String>,
        max_hops: usize,
    ) -> ZResult<Vec<ResolvedRuleApplication>> {
        let at_ms = changed_claims
            .iter()
            .map(|claim| claim.valid_from_ms.unwrap_or(claim.observed_at_ms))
            .max();
        self.resolve_rules_for_changed_claims_at(changed_claims, existing_ids, max_hops, at_ms)
    }

    pub(crate) fn resolve_rules_for_changed_claims_at(
        &self,
        changed_claims: &[ClaimRecord],
        existing_ids: &BTreeSet<String>,
        max_hops: usize,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<ResolvedRuleApplication>> {
        if max_hops == 0 {
            return Ok(Vec::new());
        }
        // A rule fires only on claims of its own scope, so each scope propagates on its own.
        let mut claims_by_scope = BTreeMap::<&MemoryScope, Vec<ClaimRecord>>::new();
        for claim in changed_claims {
            claims_by_scope
                .entry(&claim.scope)
                .or_default()
                .push(claim.clone());
        }
        let mut applications = Vec::new();
        for (scope, claims) in claims_by_scope {
            applications.extend(self.resolve_rules_in_scope(
                scope,
                &claims,
                existing_ids,
                max_hops,
                at_ms,
            )?);
        }
        Ok(applications)
    }

    /// Propagation from changed claims that all belong to `scope`.
    fn resolve_rules_in_scope(
        &self,
        scope: &MemoryScope,
        changed_claims: &[ClaimRecord],
        existing_ids: &BTreeSet<String>,
        max_hops: usize,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<ResolvedRuleApplication>> {
        let registry = self.canonical_registry_for_names_at(
            scope,
            changed_claims
                .iter()
                .filter_map(|claim| claim.subject.as_deref()),
            at_ms,
        )?;
        let mut propagation = RulePropagation {
            scope,
            registry: &registry,
            at_ms,
            known_ids: existing_ids.clone(),
            queue: changed_claims
                .iter()
                .filter_map(PropagationStep::initial)
                .collect(),
            visited: BTreeSet::new(),
            changed_claims,
            applications: Vec::new(),
        };
        while let Some(step) = propagation.queue.pop_front() {
            if step.hop < max_hops {
                self.propagate_from_trigger(&mut propagation, &step)?;
            }
        }
        Ok(propagation.applications)
    }

    /// Resolves every target slot the rules of one trigger claim reach: a derivation where the
    /// trigger carries a value, otherwise an unsupported mark on the targets it made stale.
    fn propagate_from_trigger(
        &self,
        propagation: &mut RulePropagation<'_>,
        step: &PropagationStep,
    ) -> ZResult<()> {
        let Some(trigger_slot) = claim_slot(&step.trigger) else {
            return Ok(());
        };
        let rules = self
            .rules_for_trigger_claim(
                propagation.scope,
                &step.trigger,
                usize::MAX,
                propagation.at_ms,
            )?
            .into_iter()
            .map(|rule| canonicalize_rule(rule, propagation.registry))
            .collect::<Vec<_>>();
        if rules.is_empty() {
            return Ok(());
        }
        let trigger = TriggerStep {
            step,
            trigger_slot,
            trigger_supports_value: matches!(step.trigger.status, MemoryStatus::Active)
                && matches!(step.trigger.polarity, ClaimPolarity::Affirmative)
                && step.trigger.object_value.is_some(),
            specific_target_slots: rules
                .iter()
                .filter(|rule| matches!(rule.target_match, RuleTargetMatch::ExactSlot))
                .map(rule_target_slot)
                .collect(),
        };
        let mut rules_by_target = BTreeMap::<CanonicalSlot, Vec<&RuleRecord>>::new();
        for rule in &rules {
            rules_by_target
                .entry(rule_target_slot(rule))
                .or_default()
                .push(rule);
        }
        for (target_slot, target_rules) in rules_by_target {
            if step.path.contains(&target_slot)
                || !propagation.visited.insert(trigger.visit_key(&target_slot))
            {
                continue;
            }
            if self.derive_target(propagation, &trigger, &target_slot, &target_rules)? {
                continue;
            }
            self.mark_stale_targets_unsupported(
                propagation,
                &trigger,
                &target_slot,
                &target_rules,
            )?;
        }
        Ok(())
    }

    /// Applies the target's derivation rules to a trigger that carries a value. Returns whether
    /// any of them derived the target.
    fn derive_target(
        &self,
        propagation: &mut RulePropagation<'_>,
        trigger: &TriggerStep<'_>,
        target_slot: &CanonicalSlot,
        target_rules: &[&RuleRecord],
    ) -> ZResult<bool> {
        let mut derived_target = false;
        for rule in target_rules
            .iter()
            .filter(|rule| matches!(rule.action, RuleAction::DeriveValue))
        {
            let (activated, prior_trigger) =
                self.rule_activation_evidence(rule, &trigger.step.trigger)?;
            if !activated || !trigger.trigger_supports_value {
                continue;
            }
            let Some(mut record) =
                materialize_rule_application(&trigger.step.trigger, rule, false, &BTreeSet::new())?
            else {
                continue;
            };
            derived_target = true;
            merge_prior_trigger_evidence(&mut record, prior_trigger.as_ref());
            propagation.record(
                trigger.step,
                rule,
                target_slot,
                prior_trigger.as_ref(),
                record,
            );
        }
        Ok(derived_target)
    }

    /// Marks unsupported the target slots whose active assertion predates the trigger's change,
    /// unless a subject-wide rule's own target slot is still actively asserted.
    fn mark_stale_targets_unsupported(
        &self,
        propagation: &mut RulePropagation<'_>,
        trigger: &TriggerStep<'_>,
        target_slot: &CanonicalSlot,
        target_rules: &[&RuleRecord],
    ) -> ZResult<()> {
        let target_is_generic = target_rules
            .iter()
            .any(|rule| matches!(rule.target_match, RuleTargetMatch::AnyActiveSlotForSubject));
        let target_claims = self.current_claims_for_rule_targets(
            propagation.scope,
            target_rules,
            propagation.generated_claims(),
            propagation.at_ms,
        )?;
        if target_is_generic
            && target_claims.iter().any(|claim| {
                claim_slot(claim).is_some_and(|slot| slot == *target_slot)
                    && claim_actively_affirmed(claim)
            })
        {
            return Ok(());
        }
        let stale_slots =
            trigger.stale_target_slots(target_slot, target_is_generic, &target_claims);
        if stale_slots.is_empty() {
            return Ok(());
        }
        let Some(rule) = unsupported_mark_rule(target_rules, trigger.trigger_supports_value) else {
            return Ok(());
        };
        let (activated, prior_trigger) =
            self.rule_activation_evidence(rule, &trigger.step.trigger)?;
        if !activated {
            return Ok(());
        }
        for stale_slot in stale_slots {
            let concrete_rule = if target_is_generic {
                exact_rule_for_slot(rule, &stale_slot, &target_claims, propagation.registry)
            } else {
                rule.clone()
            };
            let Some(mut record) = materialize_rule_application(
                &trigger.step.trigger,
                &concrete_rule,
                true,
                &propagation.known_ids,
            )?
            else {
                continue;
            };
            merge_prior_trigger_evidence(&mut record, prior_trigger.as_ref());
            propagation.record(
                trigger.step,
                &concrete_rule,
                &stale_slot,
                prior_trigger.as_ref(),
                record,
            );
        }
        Ok(())
    }
}

/// Queue entry of the write-time propagation: a claim to evaluate as a trigger.
struct PropagationStep {
    trigger: ClaimRecord,
    hop: usize,
    /// Target slots already reached along this chain; a chain never revisits one.
    path: Vec<CanonicalSlot>,
    parent_trace_ids: Vec<MemoryId>,
}

/// Working state of the write-time rule propagation from a set of changed claims.
struct RulePropagation<'a> {
    scope: &'a MemoryScope,
    registry: &'a CanonicalRegistry,
    at_ms: Option<i64>,
    known_ids: BTreeSet<MemoryId>,
    queue: VecDeque<PropagationStep>,
    visited: BTreeSet<String>,
    changed_claims: &'a [ClaimRecord],
    applications: Vec<ResolvedRuleApplication>,
}

impl PropagationStep {
    fn initial(claim: &ClaimRecord) -> Option<Self> {
        Some(Self {
            path: vec![claim_slot(claim)?],
            trigger: claim.clone(),
            hop: 0,
            parent_trace_ids: Vec::new(),
        })
    }
}

impl RulePropagation<'_> {
    /// The changed claims and every claim derived from them so far.
    fn generated_claims(&self) -> impl Iterator<Item = &ClaimRecord> {
        self.changed_claims.iter().chain(
            self.applications
                .iter()
                .map(|application| &application.claim),
        )
    }

    /// Records a newly derived claim: queues it as the next trigger and emits its application.
    fn record(
        &mut self,
        step: &PropagationStep,
        rule: &RuleRecord,
        target_slot: &CanonicalSlot,
        prior_trigger: Option<&ClaimRecord>,
        claim: ClaimRecord,
    ) {
        if !self.known_ids.insert(claim.id.clone()) {
            return;
        }
        let trace_id = dependency_trace_id(&rule.id, &step.trigger.id, &claim.id);
        let mut path = step.path.clone();
        path.push(target_slot.clone());
        let mut parent_trace_ids = step.parent_trace_ids.clone();
        parent_trace_ids.push(trace_id.clone());
        self.queue.push_back(PropagationStep {
            trigger: claim.clone(),
            hop: step.hop + 1,
            path,
            parent_trace_ids,
        });
        self.applications.push(ResolvedRuleApplication {
            rule_id: rule.id.clone(),
            trace_id,
            trigger_claim_id: step.trigger.id.clone(),
            prior_trigger_claim_id: prior_trigger.map(|claim| claim.id.clone()),
            claim,
            target_subject_key: rule.target_subject_key.clone(),
            target_predicate_key: target_slot.predicate_key.clone(),
            hop: step.hop + 1,
            parent_trace_ids: step.parent_trace_ids.clone(),
        });
    }
}

/// One dequeued trigger claim with the facts every target of its rules is resolved against.
struct TriggerStep<'a> {
    step: &'a PropagationStep,
    trigger_slot: CanonicalSlot,
    trigger_supports_value: bool,
    /// Target slots some exact-slot rule of the trigger names.
    specific_target_slots: BTreeSet<CanonicalSlot>,
}

impl TriggerStep<'_> {
    fn visit_key(&self, target_slot: &CanonicalSlot) -> String {
        format!(
            "{}\u{0}{}\u{0}{}\u{0}{}\u{0}{}\u{0}{}",
            self.step.trigger.id,
            self.trigger_slot.subject_key,
            self.trigger_slot.predicate_key,
            target_slot.subject_key,
            target_slot.predicate_key,
            self.step.hop
        )
    }

    /// Slots of the target's subject whose active assertion predates the trigger's change: the
    /// rules' own target slot, or for a subject-wide rule every slot no exact rule names.
    fn stale_target_slots(
        &self,
        target_slot: &CanonicalSlot,
        target_is_generic: bool,
        target_claims: &[ClaimRecord],
    ) -> BTreeSet<CanonicalSlot> {
        let trigger = &self.step.trigger;
        let trigger_change_at = claim_effective_from_ms(trigger);
        target_claims
            .iter()
            .filter_map(|claim| {
                let slot = claim_slot(claim)?;
                let in_rule_scope = if target_is_generic {
                    !self.specific_target_slots.contains(&slot)
                } else {
                    slot == *target_slot
                };
                let predates_trigger = temporal_position_is_after(
                    trigger_change_at,
                    trigger.source_sequence_no,
                    claim_effective_from_ms(claim),
                    claim.source_sequence_no,
                );
                (slot != self.trigger_slot
                    && slot.subject_key == target_slot.subject_key
                    && claim_actively_affirmed(claim)
                    && predates_trigger
                    && in_rule_scope)
                    .then_some(slot)
            })
            .collect()
    }
}

pub(super) fn claim_actively_affirmed(claim: &ClaimRecord) -> bool {
    matches!(claim.status, MemoryStatus::Active)
        && matches!(claim.polarity, ClaimPolarity::Affirmative)
}

pub(super) fn rule_target_slot(rule: &RuleRecord) -> CanonicalSlot {
    CanonicalSlot {
        subject_key: rule
            .target_subject_entity_id
            .clone()
            .unwrap_or_else(|| rule.target_subject_key.clone()),
        predicate_key: rule.target_predicate_key.clone(),
    }
}

/// The rule that marks a stale target unsupported: an explicit mark, or a derivation whose
/// trigger no longer carries a value.
fn unsupported_mark_rule<'a>(
    target_rules: &[&'a RuleRecord],
    trigger_supports_value: bool,
) -> Option<&'a RuleRecord> {
    let with_action = |action: RuleAction| {
        target_rules
            .iter()
            .copied()
            .find(|rule| rule.action == action)
    };
    with_action(RuleAction::MarkUnsupported).or_else(|| {
        (!trigger_supports_value)
            .then(|| with_action(RuleAction::DeriveValue))
            .flatten()
    })
}

/// A subject-wide rule narrowed to one target slot, labelled with the surface that slot's
/// claim uses.
fn exact_rule_for_slot(
    rule: &RuleRecord,
    target_slot: &CanonicalSlot,
    target_claims: &[ClaimRecord],
    registry: &CanonicalRegistry,
) -> RuleRecord {
    let (subject, predicate) = target_claims
        .iter()
        .find(|claim| claim_slot(claim).is_some_and(|slot| slot == *target_slot))
        .map(|claim| {
            (
                claim.subject.as_deref().unwrap_or_default().to_string(),
                claim.predicate.as_deref().unwrap_or_default().to_string(),
            )
        })
        .unwrap_or_else(|| {
            (
                target_slot.subject_key.clone(),
                target_slot.predicate_key.clone(),
            )
        });
    let mut concrete_rule = rule.clone();
    concrete_rule.target_subject = Some(subject);
    concrete_rule.target_predicate = Some(predicate);
    concrete_rule.target_subject_key = target_slot.subject_key.clone();
    concrete_rule.target_predicate_key = target_slot.predicate_key.clone();
    concrete_rule.target_match = RuleTargetMatch::ExactSlot;
    concrete_rule.target_slot_id = Some(canonical_slot_id(
        concrete_rule.target_subject_entity_id.as_deref(),
        &target_slot.subject_key,
        &target_slot.predicate_key,
    ));
    concrete_rule.target_binding_status = RuleBindingStatus::Bound;
    canonicalize_rule(concrete_rule, registry)
}

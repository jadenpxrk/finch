use super::*;

impl MemoryStore {
    /// Stamps the rule's source order, canonicalises it at its start, and binds its endpoints.
    pub(super) fn sequence_and_bind_rule(&self, mut record: RuleRecord) -> ZResult<RuleRecord> {
        record.source_sequence_no =
            self.source_sequence_no_for_episode_ids(&record.scope, &record.source_episode_ids)?;
        let at_ms = record.valid_from_ms;
        let record = self.canonicalize_rule_at(record, at_ms)?;
        self.bind_rule_endpoints(record, at_ms)
    }

    fn canonicalize_rule_at(&self, rule: RuleRecord, at_ms: Option<i64>) -> ZResult<RuleRecord> {
        let endpoint_subjects = [
            rule.trigger_subject.as_deref(),
            rule.target_subject.as_deref(),
        ];
        let registry = self.canonical_registry_for_names_at(
            &rule.scope,
            endpoint_subjects.into_iter().flatten(),
            at_ms,
        )?;
        Ok(canonicalize_rule(rule, &registry))
    }

    fn bind_rule_endpoints(&self, rule: RuleRecord, at_ms: Option<i64>) -> ZResult<RuleRecord> {
        let registry = self.canonical_registry_for_names_at(
            &rule.scope,
            [
                rule.trigger_subject.as_deref(),
                rule.target_subject.as_deref(),
            ]
            .into_iter()
            .flatten(),
            at_ms,
        )?;
        let mut rule = canonicalize_rule(rule, &registry);
        let claims = self
            .scan_claims(&rule.scope, usize::MAX, at_ms)?
            .into_iter()
            .filter(|claim| claim.scope == rule.scope)
            .map(|claim| canonicalize_claim(claim, &registry))
            .collect::<Vec<_>>();
        bind_trigger_endpoint(&mut rule, &claims);
        bind_target_endpoint(&mut rule, &claims);
        self.adopt_bound_slot_identities(&mut rule)?;
        Ok(rule)
    }

    /// An endpoint bound to a slot other than its own surface's canonical slot takes that
    /// slot's identity (entity, keys, and any labels it has).
    fn adopt_bound_slot_identities(&self, rule: &mut RuleRecord) -> ZResult<()> {
        let trigger_lookup = rule.trigger_slot_id.as_ref().filter(|slot_id| {
            **slot_id
                != canonical_slot_id(
                    rule.trigger_subject_entity_id.as_deref(),
                    &rule.trigger_subject_key,
                    &rule.trigger_predicate_key,
                )
        });
        let target_lookup = rule.target_slot_id.as_ref().filter(|slot_id| {
            **slot_id
                != canonical_slot_id(
                    rule.target_subject_entity_id.as_deref(),
                    &rule.target_subject_key,
                    &rule.target_predicate_key,
                )
        });
        let bound_slot_ids = trigger_lookup
            .into_iter()
            .chain(target_lookup)
            .cloned()
            .collect::<BTreeSet<_>>();
        let slots = self.current_slots_for_slot_keys(&rule.scope, &bound_slot_ids)?;
        if let Some(slot) = slot_with_key(&slots, rule.trigger_slot_id.as_ref()) {
            rule.trigger_subject_entity_id = slot.subject_entity_id.clone();
            rule.trigger_subject_key = slot.subject_key.clone();
            rule.trigger_predicate_key = slot.predicate_key.clone();
            rule.trigger_subject = slot.subject.clone().or(rule.trigger_subject.take());
            rule.trigger_predicate = slot.predicate.clone().or(rule.trigger_predicate.take());
        }
        if let Some(slot) = slot_with_key(&slots, rule.target_slot_id.as_ref()) {
            rule.target_subject_entity_id = slot.subject_entity_id.clone();
            rule.target_subject_key = slot.subject_key.clone();
            rule.target_predicate_key = slot.predicate_key.clone();
            rule.target_subject = slot.subject.clone().or(rule.target_subject.take());
            rule.target_predicate = slot.predicate.clone().or(rule.target_predicate.take());
        }
        Ok(())
    }

    pub(crate) fn rebind_rules_for_scope(
        &self,
        scope: &MemoryScope,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<RuleRecord>> {
        let mut rules = self.scan_rules(scope, usize::MAX, at_ms)?;
        rules.retain(|rule| rule.scope == *scope);
        let mut rebound = Vec::with_capacity(rules.len());
        for rule in rules {
            rebound.push(self.bind_rule_endpoints(rule, at_ms)?);
        }
        let rule_docs = rebound
            .iter()
            .map(|rule| rule_doc(rule).map_err(json_error))
            .collect::<ZResult<Vec<_>>>()?;
        upsert_many(&self.rules, rule_docs)?;
        Ok(rebound)
    }

    /// Canonicalises rules against the entity registry of their endpoint subjects (plus
    /// `extra_subject`).
    pub(super) fn canonicalize_rules_at(
        &self,
        scope: &MemoryScope,
        rules: Vec<RuleRecord>,
        extra_subject: Option<&str>,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<RuleRecord>> {
        let endpoint_names = rules
            .iter()
            .flat_map(|rule| {
                [
                    rule.trigger_subject.as_deref(),
                    rule.target_subject.as_deref(),
                ]
            })
            .flatten()
            .chain(extra_subject)
            .collect::<Vec<_>>();
        let registry = self.canonical_registry_for_names_at(scope, endpoint_names, at_ms)?;
        Ok(rules
            .into_iter()
            .map(|rule| canonicalize_rule(rule, &registry))
            .collect())
    }
}

/// Whether any claim carries this endpoint surface (same subject identity and predicate key).
/// A rule record for `input`, with endpoint keys canonical and bindings pending.
pub(super) fn rule_record_from_input(input: RuleInput) -> ZResult<RuleRecord> {
    if input.source_span_ids.is_empty() || input.source_episode_ids.is_empty() {
        return Err(Status::invalid_argument(
            "explicit rule record requires source evidence ids",
        ));
    }
    if matches!(input.activation, RuleActivation::ContinuousProjection)
        && matches!(input.action, RuleAction::MarkUnsupported)
    {
        return Err(Status::invalid_argument(
            "continuous rules must derive a supported value",
        ));
    }
    let trigger_subject_key = canonical_slot_part(&input.trigger_subject_key);
    let trigger_predicate_key = canonical_slot_part(&input.trigger_predicate_key);
    let target_subject_key = canonical_slot_part(&input.target_subject_key);
    let target_predicate_key = canonical_slot_part(&input.target_predicate_key);
    let id = input.id.unwrap_or_else(|| {
        let mut provenance_keys = input.source_span_ids.clone();
        provenance_keys.extend_from_slice(&input.source_episode_ids);
        provenance_keys.sort_unstable();
        provenance_keys.dedup();
        crate::ingest::generated_id(
            "rule",
            &input.scope,
            &[
                &trigger_subject_key,
                &trigger_predicate_key,
                &target_subject_key,
                &target_predicate_key,
                input.target_match.as_str(),
                input.activation.as_str(),
                input.action.as_str(),
                &provenance_keys.join(","),
            ],
        )
    });
    let target_slot_id = matches!(input.target_match, RuleTargetMatch::ExactSlot)
        .then_some(input.target_slot_id)
        .flatten();
    Ok(RuleRecord {
        id,
        scope: input.scope,
        status: input.status,
        visibility: input.visibility,
        policy_tags: input.policy_tags,
        trigger_subject_key,
        trigger_predicate_key,
        target_subject_key,
        target_predicate_key,
        trigger_subject_entity_id: None,
        target_subject_entity_id: None,
        trigger_slot_id: input.trigger_slot_id,
        target_slot_id,
        trigger_binding_status: RuleBindingStatus::Pending,
        target_binding_status: RuleBindingStatus::Pending,
        trigger_subject: input.trigger_subject,
        trigger_predicate: input.trigger_predicate,
        target_subject: input.target_subject,
        target_predicate: input.target_predicate,
        target_match: input.target_match,
        activation: input.activation,
        action: input.action,
        value_template: input.value_template,
        value: input.value,
        source_span_ids: input.source_span_ids,
        source_episode_ids: input.source_episode_ids,
        source_sequence_no: None,
        valid_from_ms: input.valid_from_ms,
        valid_to_ms: input.valid_to_ms,
        confidence: input.confidence,
    })
}

fn bind_trigger_endpoint(rule: &mut RuleRecord, claims: &[ClaimRecord]) {
    if matches!(rule.trigger_binding_status, RuleBindingStatus::Ambiguous) {
        return;
    }
    let (status, slot_id) = bind_exact_rule_endpoint(
        claims,
        rule.trigger_subject_entity_id.as_deref(),
        &rule.trigger_subject_key,
        &rule.trigger_predicate_key,
        rule.trigger_slot_id.as_deref(),
    );
    rule.trigger_binding_status = status;
    rule.trigger_slot_id = slot_id;
}

/// A subject-wide target is a proposal about scope. When the rule's own target surface is
/// already a concrete slot with claim evidence, that evidence fixes the scope: the rule is
/// about that slot, not about every slot of the subject.
fn bind_target_endpoint(rule: &mut RuleRecord, claims: &[ClaimRecord]) {
    if matches!(rule.target_match, RuleTargetMatch::AnyActiveSlotForSubject)
        && endpoint_surface_has_evidence(
            claims,
            rule.target_subject_entity_id.as_deref(),
            &rule.target_subject_key,
            &rule.target_predicate_key,
        )
    {
        rule.target_match = RuleTargetMatch::ExactSlot;
    }
    let subject_wide = matches!(rule.target_match, RuleTargetMatch::AnyActiveSlotForSubject);
    let (status, slot_id) = if subject_wide {
        bind_rule_subject(
            claims,
            rule.target_subject_entity_id.as_deref(),
            &rule.target_subject_key,
        )
    } else {
        bind_exact_rule_endpoint(
            claims,
            rule.target_subject_entity_id.as_deref(),
            &rule.target_subject_key,
            &rule.target_predicate_key,
            rule.target_slot_id.as_deref(),
        )
    };
    if matches!(rule.target_binding_status, RuleBindingStatus::Ambiguous) {
        return;
    }
    rule.target_binding_status = status;
    if !subject_wide {
        rule.target_slot_id = slot_id;
    }
}

fn slot_with_key<'a>(
    slots: &'a [CanonicalSlotRecord],
    slot_id: Option<&MemoryId>,
) -> Option<&'a CanonicalSlotRecord> {
    let slot_id = slot_id?;
    slots.iter().find(|slot| slot.slot_key == *slot_id)
}

fn endpoint_surface_has_evidence(
    claims: &[ClaimRecord],
    subject_entity_id: Option<&str>,
    subject_key: &str,
    predicate_key: &str,
) -> bool {
    claims.iter().any(|claim| {
        canonical_slot_part(claim.predicate.as_deref().unwrap_or_default()) == predicate_key
            && match (subject_entity_id, claim.subject_entity_id.as_deref()) {
                (Some(expected), Some(actual)) => expected == actual,
                _ => {
                    canonical_slot_part(claim.subject.as_deref().unwrap_or_default()) == subject_key
                }
            }
    })
}

fn bind_exact_rule_endpoint(
    claims: &[ClaimRecord],
    subject_entity_id: Option<&str>,
    subject_key: &str,
    predicate_key: &str,
    existing_slot_id: Option<&str>,
) -> (RuleBindingStatus, Option<MemoryId>) {
    let mut slot_ids = existing_slot_id
        .map(str::to_string)
        .into_iter()
        .collect::<BTreeSet<_>>();
    for claim in claims {
        let claim_subject_key = canonical_slot_part(claim.subject.as_deref().unwrap_or_default());
        let same_subject = match (subject_entity_id, claim.subject_entity_id.as_deref()) {
            (Some(expected), Some(actual)) => expected == actual,
            _ => claim_subject_key == subject_key,
        };
        if same_subject
            && canonical_slot_part(claim.predicate.as_deref().unwrap_or_default()) == predicate_key
        {
            if let Some(slot_id) = claim.slot_id.as_ref() {
                slot_ids.insert(slot_id.clone());
            }
        }
    }
    match slot_ids.len() {
        0 if !subject_key.is_empty() && !predicate_key.is_empty() => (
            RuleBindingStatus::Bound,
            Some(canonical_slot_id(
                subject_entity_id,
                subject_key,
                predicate_key,
            )),
        ),
        0 => (RuleBindingStatus::Pending, None),
        1 => (RuleBindingStatus::Bound, slot_ids.into_iter().next()),
        _ => (RuleBindingStatus::Ambiguous, None),
    }
}

fn bind_rule_subject(
    claims: &[ClaimRecord],
    subject_entity_id: Option<&str>,
    subject_key: &str,
) -> (RuleBindingStatus, Option<MemoryId>) {
    if subject_entity_id.is_some() {
        return (RuleBindingStatus::Bound, None);
    }
    let identities = claims
        .iter()
        .filter(|claim| {
            canonical_slot_part(claim.subject.as_deref().unwrap_or_default()) == subject_key
        })
        .map(|claim| {
            claim
                .subject_entity_id
                .clone()
                .unwrap_or_else(|| subject_key.to_string())
        })
        .collect::<BTreeSet<_>>();
    match identities.len() {
        0 if !subject_key.is_empty() => (RuleBindingStatus::Bound, None),
        0 => (RuleBindingStatus::Pending, None),
        1 => (RuleBindingStatus::Bound, None),
        _ => (RuleBindingStatus::Ambiguous, None),
    }
}

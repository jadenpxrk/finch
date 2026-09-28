use super::*;

mod activation;
mod binding;
mod outcomes;
mod propagation;
use activation::*;
use binding::*;
use outcomes::*;
use propagation::*;

pub(super) struct DependencyChainResolution {
    pub applications: Vec<ResolvedRuleApplication>,
    pub projected_claims: Vec<ClaimRecord>,
    pub suppressed_claim_ids: BTreeSet<MemoryId>,
    pub outcomes: Vec<RuleResolutionOutcome>,
}

impl MemoryStore {
    pub(crate) fn append_rule(&self, record: &RuleRecord) -> ZResult<()> {
        self.with_state_mutation(&record.scope, || {
            self.validate_evidence_references(
                &record.scope,
                &record.source_span_ids,
                &record.source_episode_ids,
            )?;
            let record = self.sequence_and_bind_rule(record.clone())?;
            let rule_docs = vec![rule_doc(&record).map_err(json_error)?];
            self.capture_state_mutation_docs(RULES_COLLECTION, &self.rules, &rule_docs)?;
            insert_many(&self.rules, rule_docs)?;
            self.propagate_new_rule(&record)
        })
    }

    pub fn add_rule(&self, input: RuleInput) -> ZResult<RuleRecord> {
        let record = self.build_rule_record(input)?;
        self.append_rule(&record)?;
        Ok(record)
    }

    pub(crate) fn build_rule_record(&self, input: RuleInput) -> ZResult<RuleRecord> {
        self.sequence_and_bind_rule(rule_record_from_input(input)?)
    }

    pub fn scan_rules(
        &self,
        scope: &MemoryScope,
        limit: usize,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<RuleRecord>> {
        let _state_guard = self.lock_state_read();
        if limit == 0 {
            return Ok(Vec::new());
        }
        let query = VectorQuery::new("", Vec::new(), limit)
            .with_filter(scope_filter(scope))
            .with_output_fields(output_fields(RULE_OUTPUT_FIELDS));
        let mut rules = self
            .rules
            .scan_filter_only(query)?
            .into_iter()
            .map(|doc| rule_from_doc(&doc))
            .collect::<ZResult<Vec<_>>>()?
            .into_iter()
            .filter(|rule| rule.scope.matches_filter(scope))
            .filter(|rule| matches!(rule.status, MemoryStatus::Active))
            .filter(|rule| rule_in_force_at(rule, at_ms))
            .collect::<Vec<_>>();
        rules.sort_by(|a, b| {
            a.id.cmp(&b.id)
                .then_with(|| a.trigger_subject_key.cmp(&b.trigger_subject_key))
        });
        rules.truncate(limit);
        Ok(rules)
    }

    /// Canonical trigger slots whose transitions an active `on_change` rule still propagates.
    ///
    /// `on_change` rules apply to every proven transition, so a slot stays listed after it has
    /// already changed: only the comparison baseline moves. A slot is listed when its rule is
    /// active at `at_ms`, the rule names a canonical trigger slot, that slot exists, and the slot
    /// carries directly asserted current state that supplies a value or state text to compare a
    /// later transition against. Results are deduplicated and ordered by slot ID.
    pub fn repairable_trigger_slots(
        &self,
        scope: &MemoryScope,
        limit: usize,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<RepairableTriggerSlot>> {
        let _state_guard = self.lock_state_read();
        if limit == 0 {
            return Ok(Vec::new());
        }
        // A rule's trigger slot is keyed with its scope: each scope repairs its own slot.
        let trigger_slots = self
            .scan_rules(scope, usize::MAX, at_ms)?
            .into_iter()
            .filter(|rule| matches!(rule.activation, RuleActivation::OnChange))
            .filter_map(|rule| Some((rule.trigger_slot_id?, rule.scope)))
            .collect::<BTreeSet<_>>();
        if trigger_slots.is_empty() {
            return Ok(Vec::new());
        }
        let trigger_slot_ids = trigger_slots
            .iter()
            .map(|(slot_id, _)| slot_id.clone())
            .collect::<BTreeSet<_>>();
        let mut slots = self
            .scan_slots(scope, usize::MAX, at_ms)?
            .into_iter()
            .map(|slot| ((slot.slot_key.clone(), slot.scope.clone()), slot))
            .filter(|(key, _)| trigger_slots.contains(key))
            .collect::<BTreeMap<_, _>>();
        let current = latest_comparable_claims_by_slot(self.scan_current_claims_for_slot_ids(
            scope,
            &trigger_slot_ids,
            usize::MAX,
            at_ms,
        )?);

        let mut repairable = current
            .into_iter()
            .filter_map(|((slot_id, claim_scope), claim)| {
                let slot = slots.remove(&(slot_id.clone(), claim_scope))?;
                Some(RepairableTriggerSlot {
                    slot_id,
                    subject_key: slot.subject_key,
                    predicate_key: slot.predicate_key,
                    subject_entity_id: slot.subject_entity_id,
                    subject: slot.subject,
                    predicate: slot.predicate,
                    current_value: claim.object_value,
                    current_state_text: claim.claim_text,
                    current_valid_from_ms: claim.valid_from_ms,
                    current_observed_at_ms: claim.observed_at_ms,
                })
            })
            .collect::<Vec<_>>();
        repairable.sort_by(|a, b| a.slot_id.cmp(&b.slot_id));
        repairable.truncate(limit);
        Ok(repairable)
    }

    pub fn rules_for_trigger_slot(
        &self,
        scope: &MemoryScope,
        trigger_subject: &str,
        trigger_predicate: &str,
        limit: usize,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<RuleRecord>> {
        let _state_guard = self.lock_state_read();
        let trigger_subject_key = canonical_slot_part(trigger_subject);
        let trigger_predicate_key = canonical_slot_part(trigger_predicate);
        if limit == 0 {
            return Ok(Vec::new());
        }
        let filter = format!(
            "{} AND status = 'active' AND trigger_subject_key = '{}' AND trigger_predicate_key = '{}'",
            scope_filter(scope),
            sql_escape(&trigger_subject_key),
            sql_escape(&trigger_predicate_key),
        );
        let query = VectorQuery::new("", Vec::new(), limit)
            .with_filter(filter)
            .with_output_fields(output_fields(RULE_OUTPUT_FIELDS));
        let mut rules = self
            .rules
            .scan_filter_only(query)?
            .into_iter()
            .map(|doc| rule_from_doc(&doc))
            .collect::<ZResult<Vec<_>>>()?
            .into_iter()
            .filter(|rule| {
                rule.trigger_subject_key == trigger_subject_key
                    && rule.trigger_predicate_key == trigger_predicate_key
            })
            .filter(|rule| rule_bound_and_in_force(rule, at_ms))
            .collect::<Vec<_>>();
        rules.sort_by(|a, b| a.id.cmp(&b.id));
        rules.truncate(limit);
        Ok(rules)
    }

    pub fn rules_for_target_slot(
        &self,
        scope: &MemoryScope,
        slot: &CanonicalSlot,
        limit: usize,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<RuleRecord>> {
        let _state_guard = self.lock_state_read();
        if limit == 0 {
            return Ok(Vec::new());
        }
        let target_subject_key = canonical_slot_part(&slot.subject_key);
        let target_predicate_key = canonical_slot_part(&slot.predicate_key);
        let target_slot_id = canonical_slot_id(None, &target_subject_key, &target_predicate_key);
        let target_entity = self
            .entities_by_ids(scope, std::slice::from_ref(&slot.subject_key))?
            .into_iter()
            .next();
        let mut subject_keys = target_entity
            .iter()
            .flat_map(entity_names)
            .map(canonical_slot_part)
            .chain(std::iter::once(target_subject_key))
            .filter(|key| !key.is_empty())
            .collect::<Vec<_>>();
        subject_keys.sort();
        subject_keys.dedup();
        let surfaces =
            self.target_alias_surfaces(scope, target_slot_id, target_predicate_key, at_ms)?;
        let filter = format!(
            "{} AND status = 'active' AND (({}) OR (({}) AND (({}) OR target_match = 'any_active_slot_for_subject')))",
            scope_filter(scope),
            sql_or_eq_list("target_slot_id", surfaces.slot_ids.iter().map(String::as_str)),
            sql_or_eq_list("target_subject_key", subject_keys.iter().map(String::as_str)),
            sql_or_eq_list(
                "target_predicate_key",
                surfaces.predicate_keys.iter().map(String::as_str)
            ),
        );
        let query = VectorQuery::new("", Vec::new(), limit)
            .with_filter(filter)
            .with_output_fields(output_fields(RULE_OUTPUT_FIELDS));
        let rules = self
            .rules
            .scan_filter_only(query)?
            .into_iter()
            .map(|doc| rule_from_doc(&doc))
            .collect::<ZResult<Vec<_>>>()?;
        let mut rules = self
            .canonicalize_rules_at(scope, rules, None, at_ms)?
            .into_iter()
            .filter(|rule| rule_governs_target_slot(rule, scope, slot, at_ms))
            .collect::<Vec<_>>();
        rules.sort_by(|a, b| a.id.cmp(&b.id));
        rules.truncate(limit);
        Ok(rules)
    }

    /// Alias surfaces are the same slot under another name: a rule bound to an alias surface
    /// governs the canonical slot and vice versa, so both directions join the target filter.
    fn target_alias_surfaces(
        &self,
        scope: &MemoryScope,
        target_slot_id: MemoryId,
        target_predicate_key: String,
        at_ms: Option<i64>,
    ) -> ZResult<AliasSurfaces> {
        let mut slot_ids = Vec::new();
        let mut predicate_keys = vec![target_predicate_key];
        for alias in self.scan_slot_aliases(scope, usize::MAX, at_ms)? {
            let alias_surface_id =
                canonical_slot_id(None, &alias.alias_subject_key, &alias.alias_predicate_key);
            if alias.canonical_slot_id == target_slot_id {
                slot_ids.push(alias_surface_id);
                predicate_keys.push(alias.alias_predicate_key);
            } else if alias_surface_id == target_slot_id {
                slot_ids.push(alias.canonical_slot_id);
            }
        }
        slot_ids.push(target_slot_id);
        slot_ids.sort();
        slot_ids.dedup();
        predicate_keys.sort();
        predicate_keys.dedup();
        Ok(AliasSurfaces {
            slot_ids,
            predicate_keys,
        })
    }

    pub(super) fn rules_for_trigger_claim(
        &self,
        scope: &MemoryScope,
        trigger: &ClaimRecord,
        limit: usize,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<RuleRecord>> {
        let trigger_registry = self.canonical_registry_for_names_at(
            scope,
            trigger.subject.as_deref().into_iter(),
            at_ms,
        )?;
        let trigger = canonicalize_claim(trigger.clone(), &trigger_registry);
        let subject_keys = self.trigger_subject_keys(scope, &trigger)?;
        let Some(endpoint_filter) = trigger_endpoint_filter(&trigger, &subject_keys) else {
            return Ok(Vec::new());
        };
        let filter = format!(
            "{} AND status = 'active' AND ({endpoint_filter})",
            scope_filter(scope)
        );
        let query = VectorQuery::new("", Vec::new(), limit)
            .with_filter(filter)
            .with_output_fields(output_fields(RULE_OUTPUT_FIELDS));
        let rules = self
            .rules
            .scan_filter_only(query)?
            .into_iter()
            .map(|doc| rule_from_doc(&doc))
            .collect::<ZResult<Vec<_>>>()?
            .into_iter()
            .filter(|rule| rule.scope == trigger.scope && rule_bound_and_in_force(rule, at_ms))
            .collect::<Vec<_>>();
        let trigger_slot = claim_slot(&trigger);
        let mut rules = self
            .canonicalize_rules_at(scope, rules, trigger.subject.as_deref(), at_ms)?
            .into_iter()
            .filter(|rule| {
                trigger_slot
                    .as_ref()
                    .is_some_and(|slot| rule_triggered_by(rule, &trigger, slot))
            })
            .collect::<Vec<_>>();
        rules.sort_by(|a, b| a.id.cmp(&b.id));
        rules.truncate(limit);
        Ok(rules)
    }

    /// Canonical keys the trigger's subject goes by: its own surface and every name of its
    /// entity.
    fn trigger_subject_keys(
        &self,
        scope: &MemoryScope,
        trigger: &ClaimRecord,
    ) -> ZResult<Vec<String>> {
        let mut subject_keys = trigger
            .subject
            .as_deref()
            .map(canonical_slot_part)
            .into_iter()
            .collect::<Vec<_>>();
        if let Some(entity_id) = trigger.subject_entity_id.as_ref() {
            for entity in self.entities_by_ids(scope, std::slice::from_ref(entity_id))? {
                subject_keys.extend(entity_names(&entity).map(canonical_slot_part));
            }
        }
        subject_keys.retain(|key| !key.is_empty());
        subject_keys.sort();
        subject_keys.dedup();
        Ok(subject_keys)
    }
}

fn claim_effective_from_ms(claim: &ClaimRecord) -> i64 {
    claim.valid_from_ms.unwrap_or(claim.observed_at_ms)
}

/// Whether the rule's trigger is the claim's slot, by slot id or by canonical keys.
fn rule_triggered_by(
    rule: &RuleRecord,
    trigger: &ClaimRecord,
    trigger_slot: &CanonicalSlot,
) -> bool {
    rule.trigger_slot_id.as_ref() == trigger.slot_id.as_ref()
        || (rule.trigger_subject_key == trigger_slot.subject_key
            && rule.trigger_predicate_key == trigger_slot.predicate_key)
}

/// Slot ids and predicate keys a target slot goes by, its alias surfaces included.
struct AliasSurfaces {
    slot_ids: Vec<MemoryId>,
    predicate_keys: Vec<String>,
}

fn rule_in_force_at(rule: &RuleRecord, at_ms: Option<i64>) -> bool {
    at_ms.is_none_or(|at| {
        rule.valid_from_ms.unwrap_or(i64::MIN) <= at && rule.valid_to_ms.is_none_or(|to| at < to)
    })
}

fn rule_endpoints_bound(rule: &RuleRecord) -> bool {
    matches!(rule.trigger_binding_status, RuleBindingStatus::Bound)
        && matches!(rule.target_binding_status, RuleBindingStatus::Bound)
}

fn rule_bound_and_in_force(rule: &RuleRecord, at_ms: Option<i64>) -> bool {
    rule_endpoints_bound(rule) && rule_in_force_at(rule, at_ms)
}

/// The latest directly asserted, active claim per slot and scope that carries a value or state
/// text a later transition can be compared against.
fn latest_comparable_claims_by_slot(
    claims: Vec<ClaimRecord>,
) -> BTreeMap<(MemoryId, MemoryScope), ClaimRecord> {
    let mut latest = BTreeMap::<(MemoryId, MemoryScope), ClaimRecord>::new();
    for claim in claims {
        let Some(slot_id) = claim.slot_id.clone() else {
            continue;
        };
        let key = (slot_id, claim.scope.clone());
        let comparable = claim_actively_affirmed(&claim)
            && crate::is_direct_state_claim(&claim)
            && (claim.object_value.is_some() || !claim.claim_text.trim().is_empty());
        let newer = latest.get(&key).is_none_or(|held| {
            (claim_effective_from_ms(&claim), &claim.id) > (claim_effective_from_ms(held), &held.id)
        });
        if comparable && newer {
            latest.insert(key, claim);
        }
    }
    latest
}

/// Filter clauses naming a trigger claim as a rule trigger: by its slot id, or by its subject
/// keys and predicate. `None` when neither is known.
fn trigger_endpoint_filter(trigger: &ClaimRecord, subject_keys: &[String]) -> Option<String> {
    let predicate_key = canonical_slot_part(trigger.predicate.as_deref().unwrap_or_default());
    let mut clauses = Vec::new();
    if let Some(slot_id) = trigger.slot_id.as_ref() {
        clauses.push(format!("trigger_slot_id = '{}'", sql_escape(slot_id)));
    }
    if !subject_keys.is_empty() {
        clauses.push(format!(
            "(({}) AND trigger_predicate_key = '{}')",
            sql_or_eq_list(
                "trigger_subject_key",
                subject_keys.iter().map(String::as_str)
            ),
            sql_escape(&predicate_key),
        ));
    }
    (!clauses.is_empty()).then(|| clauses.join(" OR "))
}

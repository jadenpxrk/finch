use super::*;

#[derive(Debug, Clone)]
pub(crate) struct CanonicalRegistry {
    entity_by_alias: BTreeMap<String, MemoryId>,
    ambiguous_aliases: BTreeSet<String>,
    slot_by_alias: BTreeMap<(String, String), MemoryId>,
    ambiguous_slot_aliases: BTreeSet<(String, String)>,
}

impl CanonicalRegistry {
    pub(super) fn from_alias_records(
        aliases: &[EntityAliasRecord],
        slot_aliases: &[SlotAliasRecord],
    ) -> Self {
        let mut entity_by_alias = BTreeMap::<String, MemoryId>::new();
        let mut ambiguous_aliases = BTreeSet::new();
        for alias in aliases {
            if !matches!(alias.status, MemoryStatus::Active)
                || alias.superseded_at_ms.is_some()
                || alias.alias_key.is_empty()
                || ambiguous_aliases.contains(&alias.alias_key)
            {
                continue;
            }
            if let Some(existing) = entity_by_alias.get(&alias.alias_key) {
                if existing != &alias.entity_id {
                    entity_by_alias.remove(&alias.alias_key);
                    ambiguous_aliases.insert(alias.alias_key.clone());
                }
            } else {
                entity_by_alias.insert(alias.alias_key.clone(), alias.entity_id.clone());
            }
        }
        let mut slot_by_alias = BTreeMap::<(String, String), MemoryId>::new();
        let mut ambiguous_slot_aliases = BTreeSet::new();
        for alias in slot_aliases {
            if !matches!(alias.status, MemoryStatus::Active)
                || alias.superseded_at_ms.is_some()
                || alias.source_claim_ids.is_empty()
            {
                continue;
            }
            let subject_key = alias
                .alias_subject_entity_id
                .clone()
                .unwrap_or_else(|| alias.alias_subject_key.clone());
            let key = (subject_key, alias.alias_predicate_key.clone());
            if key.0.is_empty() || key.1.is_empty() || ambiguous_slot_aliases.contains(&key) {
                continue;
            }
            if let Some(existing) = slot_by_alias.get(&key) {
                if existing != &alias.canonical_slot_id {
                    slot_by_alias.remove(&key);
                    ambiguous_slot_aliases.insert(key);
                }
            } else {
                slot_by_alias.insert(key, alias.canonical_slot_id.clone());
            }
        }
        Self {
            entity_by_alias,
            ambiguous_aliases,
            slot_by_alias,
            ambiguous_slot_aliases,
        }
    }

    pub(crate) fn subject_entity_id(&self, subject: Option<&str>) -> Option<MemoryId> {
        let key = canonical_slot_part(subject?);
        if self.ambiguous_aliases.contains(&key) {
            return None;
        }
        self.entity_by_alias.get(&key).cloned()
    }

    pub(crate) fn subject_is_ambiguous(&self, subject: Option<&str>) -> bool {
        subject
            .map(canonical_slot_part)
            .is_some_and(|key| self.ambiguous_aliases.contains(&key))
    }

    pub(super) fn slot_id(
        &self,
        subject_entity_id: Option<&str>,
        subject_key: &str,
        predicate_key: &str,
    ) -> Option<MemoryId> {
        let mut subject_identities = subject_entity_id
            .into_iter()
            .chain(std::iter::once(subject_key))
            .collect::<Vec<_>>();
        subject_identities.sort_unstable();
        subject_identities.dedup();
        let mut resolved = None;
        for subject_identity in subject_identities {
            let key = (subject_identity.to_string(), predicate_key.to_string());
            if self.ambiguous_slot_aliases.contains(&key) {
                return None;
            }
            if let Some(slot_id) = self.slot_by_alias.get(&key) {
                if resolved
                    .as_ref()
                    .is_some_and(|existing| existing != slot_id)
                {
                    return None;
                }
                resolved = Some(slot_id.clone());
            }
        }
        resolved
    }

    pub(crate) fn slot_is_ambiguous(
        &self,
        subject_entity_id: Option<&str>,
        subject_key: &str,
        predicate_key: &str,
    ) -> bool {
        subject_entity_id
            .into_iter()
            .chain(std::iter::once(subject_key))
            .map(|subject| (subject.to_string(), predicate_key.to_string()))
            .any(|key| self.ambiguous_slot_aliases.contains(&key))
    }
}

impl MemoryStore {
    /// Returns binding data for up to `limit` slots: names, aliases, states, and rule links.
    pub fn canonical_slot_binding_contexts(
        &self,
        scope: &MemoryScope,
        limit: usize,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<CanonicalSlotBindingContext>> {
        let _state_guard = self.lock_state_read()?;
        if limit == 0 {
            return Ok(Vec::new());
        }
        let entities = self.scan_entities(scope, usize::MAX)?;
        let slot_aliases = self.scan_slot_aliases(scope, usize::MAX, at_ms)?;
        let states = self.scan_state_records(
            scope,
            StateRecordScan {
                limit: usize::MAX,
                temporal: BiTemporalQuery {
                    valid_at_ms: at_ms,
                    transaction_at_ms: None,
                },
            },
        )?;
        let rules = self.scan_rules(scope, usize::MAX, at_ms)?;
        let mut slots_by_scope = BTreeMap::<MemoryScope, Vec<CanonicalSlotRecord>>::new();
        for slot in self.scan_slots(scope, usize::MAX, at_ms)? {
            slots_by_scope
                .entry(slot.scope.clone())
                .or_default()
                .push(slot);
        }
        // Each scope's slots take their entities, aliases, states, and rules from that scope only.
        let mut contexts = Vec::new();
        for (slot_scope, slots) in slots_by_scope {
            let scope_entities = entities
                .iter()
                .filter(|entity| entity.scope == slot_scope)
                .cloned()
                .collect::<Vec<_>>();
            let mut sources = SlotBindingSources {
                entities_by_id: scope_entities
                    .iter()
                    .map(|entity| (entity.id.as_str(), entity))
                    .collect(),
                entity_alias_owners: entity_alias_owners(&scope_entities),
                aliases_by_slot: unambiguous_slot_aliases(
                    slot_aliases
                        .iter()
                        .filter(|alias| alias.scope == slot_scope)
                        .cloned()
                        .collect(),
                ),
                states_by_slot: recent_states_by_slot(
                    states.iter().filter(|state| state.scope == slot_scope),
                ),
                rule_neighbors: RuleNeighbors::from_rules(
                    rules
                        .iter()
                        .filter(|rule| rule.scope == slot_scope)
                        .cloned()
                        .collect(),
                ),
            };
            contexts.extend(slots.into_iter().map(|slot| sources.context_for(slot)));
        }
        contexts.sort_by(|a, b| a.slot_id.cmp(&b.slot_id));
        contexts.truncate(limit);
        Ok(contexts)
    }
}

/// Up to eight distinct recent states of every slot, newest first.
fn recent_states_by_slot<'s>(
    states: impl Iterator<Item = &'s StateRecord>,
) -> BTreeMap<MemoryId, Vec<SlotTemporalState>> {
    let mut states_by_slot = BTreeMap::<MemoryId, Vec<SlotTemporalState>>::new();
    for state in states {
        let Some(slot_id) = state.slot_id.clone() else {
            continue;
        };
        states_by_slot
            .entry(slot_id)
            .or_default()
            .push(SlotTemporalState {
                state_kind: state.state_kind,
                state_text: state.state_text.clone(),
                observed_at_ms: state.observed_at_ms,
                valid_from_ms: state.valid_from_ms,
                valid_to_ms: state.valid_to_ms,
            });
    }
    for states in states_by_slot.values_mut() {
        states.sort_by(|a, b| {
            b.observed_at_ms
                .cmp(&a.observed_at_ms)
                .then_with(|| a.state_kind.as_str().cmp(b.state_kind.as_str()))
                .then_with(|| a.state_text.cmp(&b.state_text))
        });
        states.dedup_by(|a, b| {
            a.state_kind == b.state_kind
                && a.state_text == b.state_text
                && a.valid_from_ms == b.valid_from_ms
                && a.valid_to_ms == b.valid_to_ms
        });
        states.truncate(8);
    }
    states_by_slot
}

/// Everything a slot's binding context is assembled from, keyed for per-slot lookup.
struct SlotBindingSources<'a> {
    entities_by_id: BTreeMap<&'a str, &'a EntityRecord>,
    /// Canonical alias key to the entities that claim it.
    entity_alias_owners: BTreeMap<String, BTreeSet<MemoryId>>,
    aliases_by_slot: BTreeMap<MemoryId, Vec<SlotSurfaceAlias>>,
    states_by_slot: BTreeMap<MemoryId, Vec<SlotTemporalState>>,
    rule_neighbors: RuleNeighbors,
}

impl SlotBindingSources<'_> {
    fn context_for(&mut self, slot: CanonicalSlotRecord) -> CanonicalSlotBindingContext {
        let entity = slot
            .subject_entity_id
            .as_deref()
            .and_then(|entity_id| self.entities_by_id.get(entity_id).copied());
        let mut slot_aliases = self
            .aliases_by_slot
            .remove(&slot.slot_key)
            .unwrap_or_default();
        slot_aliases.sort_by(|a, b| {
            a.subject
                .cmp(&b.subject)
                .then_with(|| a.predicate.cmp(&b.predicate))
        });
        slot_aliases.dedup();
        CanonicalSlotBindingContext {
            entity_aliases: self.unambiguous_entity_aliases(entity),
            slot_aliases,
            temporal_state: self
                .states_by_slot
                .remove(&slot.slot_key)
                .unwrap_or_default(),
            dependency_trigger_slot_ids: take_ids(
                &mut self.rule_neighbors.triggers_by_target,
                &slot.slot_key,
            ),
            dependency_target_slot_ids: take_ids(
                &mut self.rule_neighbors.targets_by_trigger,
                &slot.slot_key,
            ),
            entity_type: entity.map(|entity| entity.entity_type.clone()),
            canonical_entity_name: entity.map(|entity| entity.canonical_name.clone()),
            slot_id: slot.slot_key,
            subject: slot.subject.unwrap_or(slot.subject_key),
            predicate: slot.predicate.unwrap_or(slot.predicate_key),
            subject_entity_id: slot.subject_entity_id,
        }
    }

    /// The entity's names that no other entity also claims, sorted.
    fn unambiguous_entity_aliases(&self, entity: Option<&EntityRecord>) -> Vec<String> {
        let mut aliases = entity
            .into_iter()
            .flat_map(entity_names)
            .filter(|alias| {
                self.entity_alias_owners
                    .get(&canonical_slot_part(alias))
                    .is_some_and(|owners| owners.len() == 1)
            })
            .map(str::to_string)
            .collect::<Vec<_>>();
        aliases.sort();
        aliases.dedup();
        aliases
    }
}

/// Rule endpoints linking slots: the trigger slots of each target and the targets of each
/// trigger.
struct RuleNeighbors {
    triggers_by_target: BTreeMap<MemoryId, BTreeSet<MemoryId>>,
    targets_by_trigger: BTreeMap<MemoryId, BTreeSet<MemoryId>>,
}

impl RuleNeighbors {
    fn from_rules(rules: Vec<RuleRecord>) -> Self {
        let mut neighbors = Self {
            triggers_by_target: BTreeMap::new(),
            targets_by_trigger: BTreeMap::new(),
        };
        for rule in rules {
            let (Some(trigger_slot_id), Some(target_slot_id)) =
                (rule.trigger_slot_id, rule.target_slot_id)
            else {
                continue;
            };
            neighbors
                .triggers_by_target
                .entry(target_slot_id.clone())
                .or_default()
                .insert(trigger_slot_id.clone());
            neighbors
                .targets_by_trigger
                .entry(trigger_slot_id)
                .or_default()
                .insert(target_slot_id);
        }
        neighbors
    }
}

fn take_ids(
    ids_by_slot: &mut BTreeMap<MemoryId, BTreeSet<MemoryId>>,
    slot_id: &str,
) -> Vec<MemoryId> {
    ids_by_slot
        .remove(slot_id)
        .unwrap_or_default()
        .into_iter()
        .collect()
}

pub(super) fn entity_names(entity: &EntityRecord) -> impl Iterator<Item = &str> {
    std::iter::once(entity.canonical_name.as_str()).chain(entity.aliases.iter().map(String::as_str))
}

fn entity_alias_owners(entities: &[EntityRecord]) -> BTreeMap<String, BTreeSet<MemoryId>> {
    let mut owners = BTreeMap::<String, BTreeSet<MemoryId>>::new();
    for entity in entities {
        for key in entity_names(entity).map(canonical_slot_part) {
            if !key.is_empty() {
                owners.entry(key).or_default().insert(entity.id.clone());
            }
        }
    }
    owners
}

/// Slot aliases grouped by canonical slot, keeping only surfaces that alias exactly one slot.
fn unambiguous_slot_aliases(
    slot_aliases: Vec<SlotAliasRecord>,
) -> BTreeMap<MemoryId, Vec<SlotSurfaceAlias>> {
    let mut owners = BTreeMap::<(&str, &str), BTreeSet<&str>>::new();
    for alias in &slot_aliases {
        owners
            .entry((
                alias.alias_subject_key.as_str(),
                alias.alias_predicate_key.as_str(),
            ))
            .or_default()
            .insert(alias.canonical_slot_id.as_str());
    }
    let unambiguous = slot_aliases
        .iter()
        .map(|alias| {
            owners
                .get(&(
                    alias.alias_subject_key.as_str(),
                    alias.alias_predicate_key.as_str(),
                ))
                .is_some_and(|slot_ids| slot_ids.len() == 1)
        })
        .collect::<Vec<_>>();
    let mut aliases_by_slot = BTreeMap::<MemoryId, Vec<SlotSurfaceAlias>>::new();
    for (alias, unambiguous) in slot_aliases.into_iter().zip(unambiguous) {
        if unambiguous {
            aliases_by_slot
                .entry(alias.canonical_slot_id)
                .or_default()
                .push(SlotSurfaceAlias {
                    subject: alias.alias_subject_key,
                    predicate: alias.alias_predicate_key,
                });
        }
    }
    aliases_by_slot
}

impl MemoryStore {
    /// Admits or clears caller-proposed `slot_id` bindings on incoming claims. A binding is kept
    /// only when the bound slot exists and shares the claim's subject identity (entity id, or
    /// canonical subject key); an admitted binding under another predicate surface yields an
    /// evidence-backed slot alias sourced by the claim, so the surface becomes identity now and
    /// for later claims.
    pub(crate) fn admit_bound_claim_slots(
        &self,
        scope: &MemoryScope,
        claims: &mut [ClaimRecord],
    ) -> ZResult<Vec<SlotAliasInput>> {
        let preset = claims
            .iter()
            .filter_map(|claim| claim.slot_id.clone())
            .collect::<BTreeSet<_>>();
        let proposed_slots = self
            .current_slots_for_slot_keys(scope, &preset)?
            .into_iter()
            .map(|slot| (slot.slot_key.clone(), slot))
            .collect::<BTreeMap<_, _>>();
        let mut aliases = Vec::new();
        let mut seen = BTreeSet::new();
        for claim in claims.iter_mut() {
            let admitted = self.admitted_claim_binding(scope, claim, &proposed_slots)?;
            claim.slot_id = admitted.as_ref().map(|binding| binding.slot_id.clone());
            let Some(binding) = admitted else {
                continue;
            };
            let bound_predicate_key = proposed_slots
                .get(&binding.slot_id)
                .map(|slot| slot.predicate_key.as_str())
                .unwrap_or_default();
            if bound_predicate_key == binding.predicate_key {
                continue;
            }
            let seen_key = (
                binding.subject_key,
                binding.predicate_key,
                binding.slot_id.clone(),
            );
            if seen.insert(seen_key) {
                aliases.push(slot_alias_for_binding(scope, claim, binding.slot_id));
            }
        }
        Ok(aliases)
    }

    /// The proposed slot binding a claim may keep: the bound slot exists, shares the claim's
    /// subject identity, and is not the claim's own canonical slot. Surfaces the registry
    /// already resolves need no binding.
    fn admitted_claim_binding(
        &self,
        scope: &MemoryScope,
        claim: &ClaimRecord,
        proposed_slots: &BTreeMap<MemoryId, CanonicalSlotRecord>,
    ) -> ZResult<Option<AdmittedBinding>> {
        let (Some(subject), Some(predicate)) =
            (claim.subject.as_deref(), claim.predicate.as_deref())
        else {
            return Ok(None);
        };
        let subject_key = canonical_slot_part(subject);
        let predicate_key = canonical_slot_part(predicate);
        if subject_key.is_empty() || predicate_key.is_empty() {
            return Ok(None);
        }
        let registry = self.canonical_registry_for_names_at(
            scope,
            std::iter::once(subject),
            claim.valid_from_ms.or(Some(claim.observed_at_ms)),
        )?;
        let entity = claim
            .subject_entity_id
            .clone()
            .or_else(|| registry.subject_entity_id(Some(subject)));
        if registry
            .slot_id(entity.as_deref(), &subject_key, &predicate_key)
            .is_some()
        {
            return Ok(None);
        }
        let Some(bound) = claim.slot_id.as_ref().filter(|slot_id| {
            proposed_slots
                .get(*slot_id)
                .is_some_and(|slot| slot_has_subject(slot, entity.as_deref(), &subject_key))
        }) else {
            return Ok(None);
        };
        if *bound == canonical_slot_id(entity.as_deref(), &subject_key, &predicate_key) {
            return Ok(None);
        }
        Ok(Some(AdmittedBinding {
            slot_id: bound.clone(),
            subject_key,
            predicate_key,
        }))
    }
}

struct AdmittedBinding {
    slot_id: MemoryId,
    subject_key: String,
    predicate_key: String,
}

/// Whether `slot` belongs to the subject: the same entity when both have one, otherwise the
/// same canonical subject key.
fn slot_has_subject(
    slot: &CanonicalSlotRecord,
    entity_id: Option<&str>,
    subject_key: &str,
) -> bool {
    match (entity_id, slot.subject_entity_id.as_deref()) {
        (Some(expected), Some(actual)) => expected == actual,
        _ => {
            slot.subject_key == subject_key
                || slot.subject.as_deref().map(canonical_slot_part).as_deref() == Some(subject_key)
        }
    }
}

/// The evidence-backed alias that makes the claim's surface an identity of `bound_slot_id`.
fn slot_alias_for_binding(
    scope: &MemoryScope,
    claim: &ClaimRecord,
    bound_slot_id: MemoryId,
) -> SlotAliasInput {
    SlotAliasInput {
        id: None,
        scope: scope.clone(),
        visibility: claim.visibility,
        policy_tags: claim.policy_tags.clone(),
        alias_subject: claim.subject.clone().unwrap_or_default(),
        alias_predicate: claim.predicate.clone().unwrap_or_default(),
        canonical_slot_id: Some(bound_slot_id),
        target_claim_id: None,
        source_claim_ids: vec![claim.id.clone()],
        valid_from_ms: claim.valid_from_ms.or(Some(claim.observed_at_ms)),
        valid_to_ms: None,
    }
}

pub(super) fn canonical_slot_id(
    subject_entity_id: Option<&str>,
    subject_key: &str,
    predicate_key: &str,
) -> MemoryId {
    let subject_part = subject_entity_id.unwrap_or(subject_key);
    format!("slot_{}", stable_hash_hex(&[subject_part, predicate_key]))
}

pub(crate) fn canonicalize_claim(
    mut claim: ClaimRecord,
    registry: &CanonicalRegistry,
) -> ClaimRecord {
    let (Some(subject), Some(predicate)) = (claim.subject.as_deref(), claim.predicate.as_deref())
    else {
        return claim;
    };
    let subject_key = canonical_slot_part(subject);
    let predicate_key = canonical_slot_part(predicate);
    let resolved_entity_id = registry.subject_entity_id(Some(subject));
    let entity_newly_resolved = claim.subject_entity_id.is_none() && resolved_entity_id.is_some();
    let subject_entity_id = claim
        .subject_entity_id
        .take()
        .or_else(|| resolved_entity_id.clone());
    let aliased_slot_id =
        registry.slot_id(subject_entity_id.as_deref(), &subject_key, &predicate_key);
    let alias_resolved = aliased_slot_id.is_some();
    let slot_id = if alias_resolved {
        aliased_slot_id
    } else if entity_newly_resolved {
        Some(canonical_slot_id(
            resolved_entity_id.as_deref(),
            &subject_key,
            &predicate_key,
        ))
    } else {
        claim.slot_id.take().or_else(|| {
            Some(canonical_slot_id(
                subject_entity_id.as_deref(),
                &subject_key,
                &predicate_key,
            ))
        })
    };
    let own_slot_id = canonical_slot_id(subject_entity_id.as_deref(), &subject_key, &predicate_key);
    // An alias-resolved binding is evidence-backed identity; any other preset slot that differs
    // from the claim's own surface slot is a proposed merge and stays a co-current facet.
    let slot_facet = if alias_resolved {
        None
    } else {
        slot_id
            .as_deref()
            .filter(|bound| *bound != own_slot_id)
            .map(|_| own_slot_id)
    };
    claim.subject_entity_id = subject_entity_id;
    claim.slot_id = slot_id;
    claim.slot_facet = slot_facet;
    claim
}

pub(crate) fn canonicalize_rule(mut rule: RuleRecord, registry: &CanonicalRegistry) -> RuleRecord {
    let resolved_trigger_entity_id = registry.subject_entity_id(rule.trigger_subject.as_deref());
    let resolved_target_entity_id = registry.subject_entity_id(rule.target_subject.as_deref());
    let trigger_entity_newly_resolved =
        rule.trigger_subject_entity_id.is_none() && resolved_trigger_entity_id.is_some();
    let target_entity_newly_resolved =
        rule.target_subject_entity_id.is_none() && resolved_target_entity_id.is_some();
    rule.trigger_subject_entity_id = rule
        .trigger_subject_entity_id
        .take()
        .or(resolved_trigger_entity_id);
    rule.target_subject_entity_id = rule
        .target_subject_entity_id
        .take()
        .or(resolved_target_entity_id);
    let aliased_trigger_slot_id = registry.slot_id(
        rule.trigger_subject_entity_id.as_deref(),
        &rule.trigger_subject_key,
        &rule.trigger_predicate_key,
    );
    rule.trigger_slot_id = aliased_trigger_slot_id.or_else(|| {
        if trigger_entity_newly_resolved {
            Some(canonical_slot_id(
                rule.trigger_subject_entity_id.as_deref(),
                &rule.trigger_subject_key,
                &rule.trigger_predicate_key,
            ))
        } else {
            rule.trigger_slot_id.take()
        }
    });
    rule.trigger_binding_status = if registry.subject_is_ambiguous(rule.trigger_subject.as_deref())
        || registry.slot_is_ambiguous(
            rule.trigger_subject_entity_id.as_deref(),
            &rule.trigger_subject_key,
            &rule.trigger_predicate_key,
        ) {
        RuleBindingStatus::Ambiguous
    } else if rule.trigger_slot_id.is_some() {
        RuleBindingStatus::Bound
    } else {
        RuleBindingStatus::Pending
    };
    let aliased_target_slot_id = registry.slot_id(
        rule.target_subject_entity_id.as_deref(),
        &rule.target_subject_key,
        &rule.target_predicate_key,
    );
    rule.target_slot_id = aliased_target_slot_id.or_else(|| {
        if target_entity_newly_resolved {
            Some(canonical_slot_id(
                rule.target_subject_entity_id.as_deref(),
                &rule.target_subject_key,
                &rule.target_predicate_key,
            ))
        } else {
            rule.target_slot_id.take()
        }
    });
    rule.target_binding_status = if registry.subject_is_ambiguous(rule.target_subject.as_deref())
        || registry.slot_is_ambiguous(
            rule.target_subject_entity_id.as_deref(),
            &rule.target_subject_key,
            &rule.target_predicate_key,
        ) {
        RuleBindingStatus::Ambiguous
    } else if rule.target_slot_id.is_some()
        || (matches!(rule.target_match, RuleTargetMatch::AnyActiveSlotForSubject)
            && rule.target_subject_entity_id.is_some())
    {
        RuleBindingStatus::Bound
    } else {
        RuleBindingStatus::Pending
    };
    rule
}

pub(crate) fn canonicalize_correction(
    mut correction: CorrectionRecord,
    registry: &CanonicalRegistry,
) -> CorrectionRecord {
    correction.target_subject_key = correction
        .target_subject_key
        .as_deref()
        .map(canonical_slot_part)
        .filter(|value| !value.is_empty());
    correction.target_predicate_key = correction
        .target_predicate_key
        .as_deref()
        .map(canonical_slot_part)
        .filter(|value| !value.is_empty());
    correction.target_subject_entity_id = correction
        .target_subject_entity_id
        .take()
        .or_else(|| registry.subject_entity_id(correction.target_subject_key.as_deref()));
    correction.target_slot_id = correction.target_slot_id.take().or_else(|| {
        let subject = correction.target_subject_key.as_deref()?;
        let predicate = correction.target_predicate_key.as_deref()?;
        registry.slot_id(
            correction.target_subject_entity_id.as_deref(),
            subject,
            predicate,
        )
    });
    correction
}

use super::*;

impl MemoryStore {
    pub fn apply_state_mutation_batch(
        &self,
        batch: StateMutationBatch,
    ) -> ZResult<StateMutationResult> {
        self.apply_state_mutation_batch_with_claim_embeddings(batch, BTreeMap::new())
    }

    pub fn apply_state_mutation_batch_with_claim_embeddings(
        &self,
        batch: StateMutationBatch,
        claim_embeddings: BTreeMap<MemoryId, Vec<f32>>,
    ) -> ZResult<StateMutationResult> {
        validate_batch_scope(&batch)?;
        self.validate_batch_evidence(&batch)?;
        self.apply_validated_state_mutation_batch(batch, claim_embeddings)
    }

    fn apply_validated_state_mutation_batch(
        &self,
        mut batch: StateMutationBatch,
        claim_embeddings: BTreeMap<MemoryId, Vec<f32>>,
    ) -> ZResult<StateMutationResult> {
        let scope = batch.scope.clone();
        self.with_state_mutation(&scope, || {
            for correction in &mut batch.corrections {
                *correction =
                    self.resolve_correction_target_with_pending_claims(correction, &batch.claims)?;
            }
            self.apply_state_mutation_batch_inner(batch, claim_embeddings)
        })
    }

    pub(crate) fn validate_batch_evidence(&self, batch: &StateMutationBatch) -> ZResult<()> {
        let span_ids = batch
            .claims
            .iter()
            .flat_map(|record| record.source_span_ids.iter())
            .chain(
                batch
                    .rules
                    .iter()
                    .flat_map(|record| record.source_span_ids.iter()),
            )
            .chain(
                batch
                    .corrections
                    .iter()
                    .flat_map(|record| record.source_span_ids.iter()),
            )
            .cloned()
            .collect::<BTreeSet<_>>();
        let episode_ids = batch
            .claims
            .iter()
            .flat_map(|record| record.source_episode_ids.iter())
            .chain(
                batch
                    .rules
                    .iter()
                    .flat_map(|record| record.source_episode_ids.iter()),
            )
            .chain(
                batch
                    .corrections
                    .iter()
                    .flat_map(|record| record.source_episode_ids.iter()),
            )
            .cloned()
            .collect::<BTreeSet<_>>();
        if !span_ids.is_empty() || !episode_ids.is_empty() {
            self.validate_evidence_references(
                &batch.scope,
                &span_ids.into_iter().collect::<Vec<_>>(),
                &episode_ids.into_iter().collect::<Vec<_>>(),
            )?;
        }

        let source_claim_ids = batch
            .entities
            .iter()
            .flat_map(|entity| entity.source_claim_ids.iter())
            .chain(
                batch
                    .slot_aliases
                    .iter()
                    .flat_map(|alias| alias.source_claim_ids.iter()),
            )
            .cloned()
            .collect::<BTreeSet<_>>();
        let incoming_claim_ids = batch
            .claims
            .iter()
            .map(|claim| claim.id.clone())
            .collect::<BTreeSet<_>>();
        if !self.validate_claim_references(
            &batch.scope,
            &source_claim_ids.into_iter().collect::<Vec<_>>(),
            &incoming_claim_ids,
        )? {
            return Err(Status::invalid_argument(
                "state mutation entity provenance claims must exist in the batch scope",
            ));
        }
        Ok(())
    }

    pub(crate) fn validate_evidence_references(
        &self,
        scope: &MemoryScope,
        span_ids: &[MemoryId],
        episode_ids: &[MemoryId],
    ) -> ZResult<()> {
        let span_ids = span_ids.iter().cloned().collect::<BTreeSet<_>>();
        let episode_ids = episode_ids.iter().cloned().collect::<BTreeSet<_>>();
        if span_ids.is_empty() || episode_ids.is_empty() {
            return Err(Status::invalid_argument(
                "evidence references require both span and episode ids",
            ));
        }
        let spans =
            self.fetch_spans_by_ids(scope, &span_ids.iter().cloned().collect::<Vec<_>>(), None)?;
        if spans
            .iter()
            .map(|span| span.id.clone())
            .collect::<BTreeSet<_>>()
            != span_ids
        {
            return Err(Status::invalid_argument(
                "evidence spans must exist in the requested scope",
            ));
        }
        if spans.iter().any(|span| {
            !matches!(span.source_type, crate::SourceType::Episode)
                || !episode_ids.contains(&span.source_id)
        }) {
            return Err(Status::invalid_argument(
                "evidence spans must reference a cited episode",
            ));
        }
        let episode_docs = self
            .episodes
            .fetch(episode_ids.iter().cloned().collect::<Vec<_>>())?;
        let found_episode_ids = episode_docs
            .values()
            .filter(|doc| doc_matches_scope(doc, scope))
            .map(|doc| doc.pk.clone())
            .collect::<BTreeSet<_>>();
        if found_episode_ids != episode_ids {
            return Err(Status::invalid_argument(
                "evidence episodes must exist in the requested scope",
            ));
        }
        Ok(())
    }

    pub(crate) fn validate_claim_references(
        &self,
        scope: &MemoryScope,
        claim_ids: &[MemoryId],
        additional_valid_ids: &BTreeSet<MemoryId>,
    ) -> ZResult<bool> {
        let requested = claim_ids.iter().cloned().collect::<BTreeSet<_>>();
        let mut found = self
            .claims_by_ids(scope, &requested.iter().cloned().collect::<Vec<_>>())?
            .into_iter()
            .map(|claim| claim.id)
            .collect::<BTreeSet<_>>();
        found.extend(additional_valid_ids.iter().cloned());
        Ok(requested.is_subset(&found))
    }

    pub(crate) fn source_sequence_no_for_episode_ids(
        &self,
        scope: &MemoryScope,
        episode_ids: &[MemoryId],
    ) -> ZResult<Option<i64>> {
        if episode_ids.is_empty() {
            return Ok(None);
        }
        let episode_docs = self.episodes.fetch(episode_ids.to_vec())?;
        Ok(episode_docs
            .values()
            .filter(|doc| doc_matches_scope(doc, scope))
            .filter_map(|doc| doc.fields.get("sequence_no").and_then(Value::as_i64))
            .max())
    }

    fn apply_state_mutation_batch_inner(
        &self,
        batch: StateMutationBatch,
        mut claim_embeddings: BTreeMap<MemoryId, Vec<f32>>,
    ) -> ZResult<StateMutationResult> {
        let scope = &batch.scope;
        let entities =
            self.write_batch_entities(scope, batch.entities, batch.transaction_time_ms)?;
        let mut claims = batch.claims;
        let mut alias_inputs = batch.slot_aliases;
        alias_inputs.extend(self.admit_bound_claim_slots(scope, &mut claims)?);
        let claims = self.insert_batch_claims(scope, claims, &mut claim_embeddings)?;
        let slot_aliases =
            self.write_batch_slot_aliases(alias_inputs, &claims, batch.transaction_time_ms)?;
        let claims = self.recanonicalize_batch_claims(scope, claims, &claim_embeddings)?;
        let (rules, reconciliation_rules) = self.write_batch_rules(scope, batch.rules)?;
        let corrections = self.write_batch_corrections(scope, batch.corrections)?;

        let boundaries =
            self.plan_batch_boundaries(scope, &claims, &corrections, &reconciliation_rules)?;
        let inputs = BoundaryInputs {
            scope,
            max_rule_hops: batch.max_rule_hops,
            claims: &claims,
            corrections: &corrections,
            rules: &reconciliation_rules,
            plan: &boundaries,
        };
        let propagation = self.propagate_batch_boundaries(&inputs, &mut claim_embeddings)?;
        let mut derived_claims = propagation
            .rule_applications
            .iter()
            .map(|application| application.claim.clone())
            .collect::<Vec<_>>();
        dedupe_claims_by_id(&mut derived_claims);
        let affected_slot_ids = batch_affected_slot_ids(&inputs, &propagation, &derived_claims);
        let written = BatchRecords {
            claims: &claims,
            rules: &rules,
            corrections: &corrections,
            entities: &entities,
            slot_aliases: &slot_aliases,
        };
        let write_key =
            self.batch_write_key(scope, &batch.propagation_seed, &written, &affected_slot_ids)?;
        let write = ProjectionWrite {
            time_ms: batch.transaction_time_ms,
            key: &write_key,
        };
        let state_records = self.project_batch_boundaries(
            scope,
            &affected_slot_ids,
            &boundaries.valid_boundaries,
            &propagation,
            write,
        )?;

        Ok(StateMutationResult {
            claims,
            entities,
            rules,
            slot_aliases,
            corrections,
            derived_claims,
            rule_applications: propagation.rule_applications,
            state_records,
        })
    }

    /// Resolves each incoming entity to the one existing entity its names identify, merges it
    /// with that entity's history (and with earlier entities of the batch), and writes it.
    fn write_batch_entities(
        &self,
        scope: &MemoryScope,
        inputs: Vec<EntityInput>,
        transaction_time_ms: i64,
    ) -> ZResult<Vec<EntityRecord>> {
        let mut incoming_entities = Vec::with_capacity(inputs.len());
        for mut input in inputs {
            input.id = self.resolved_entity_id(scope, &input)?.or(input.id);
            incoming_entities.push(create_entity(input));
        }
        let candidate_entity_ids = incoming_entities
            .iter()
            .map(|entity| entity.id.clone())
            .collect::<Vec<_>>();
        let mut entities_by_id = self
            .entities_by_ids(scope, &candidate_entity_ids)?
            .into_iter()
            .map(|entity| (entity.id.clone(), entity))
            .collect::<BTreeMap<_, _>>();
        let mut entities = Vec::with_capacity(incoming_entities.len());
        for mut entity in incoming_entities {
            if let Some(existing) = entities_by_id.remove(&entity.id) {
                merge_entity_history(&mut entity, existing);
            }
            entities_by_id.insert(entity.id.clone(), entity.clone());
            entities.push(entity);
        }
        self.ensure_entity_ids_owned_by_scope(&entities)?;
        let entity_docs = entities
            .iter()
            .map(|entity| entity_doc(entity, None).map_err(json_error))
            .collect::<ZResult<Vec<_>>>()?;
        upsert_many(&self.entities, entity_docs)?;
        self.upsert_entity_aliases(&entities, transaction_time_ms)?;
        Ok(entities)
    }

    /// The existing entity id every name of `input` resolves to, when they agree on exactly one.
    fn resolved_entity_id(
        &self,
        scope: &MemoryScope,
        input: &EntityInput,
    ) -> ZResult<Option<MemoryId>> {
        let names = || {
            std::iter::once(input.canonical_name.as_str())
                .chain(input.aliases.iter().map(String::as_str))
        };
        let registry = self.canonical_registry_for_names(scope, names())?;
        let resolved_ids = names()
            .filter_map(|name| registry.subject_entity_id(Some(name)))
            .collect::<BTreeSet<_>>();
        if resolved_ids.len() != 1 {
            return Ok(None);
        }
        Ok(resolved_ids.into_iter().next())
    }

    /// Canonicalises `claim` against the entity and alias registry at its own valid time.
    pub(crate) fn canonicalize_claim_at_valid_time(
        &self,
        scope: &MemoryScope,
        claim: ClaimRecord,
    ) -> ZResult<ClaimRecord> {
        let registry = self.canonical_registry_for_names_at(
            scope,
            claim.subject.as_deref().into_iter(),
            claim.valid_from_ms.or(Some(claim.observed_at_ms)),
        )?;
        Ok(canonicalize_claim(claim, &registry))
    }

    /// Stamps each claim's source order, canonicalises it, and inserts it with its embedding.
    fn insert_batch_claims(
        &self,
        scope: &MemoryScope,
        incoming_claims: Vec<ClaimRecord>,
        claim_embeddings: &mut BTreeMap<MemoryId, Vec<f32>>,
    ) -> ZResult<Vec<ClaimRecord>> {
        let mut claims = Vec::with_capacity(incoming_claims.len());
        for mut claim in incoming_claims {
            claim.source_sequence_no =
                self.source_sequence_no_for_episode_ids(scope, &claim.source_episode_ids)?;
            claims.push(self.canonicalize_claim_at_valid_time(scope, claim)?);
        }
        self.extend_claim_embeddings_from_sources(scope, &claims, claim_embeddings)?;
        let claim_docs = claim_docs_with_embeddings(&claims, claim_embeddings)?;
        insert_many(&self.claims, claim_docs)?;
        Ok(claims)
    }

    fn write_batch_slot_aliases(
        &self,
        alias_inputs: Vec<SlotAliasInput>,
        claims: &[ClaimRecord],
        transaction_time_ms: i64,
    ) -> ZResult<Vec<SlotAliasRecord>> {
        let target_slot_ids = claims
            .iter()
            .filter_map(|claim| claim.slot_id.clone())
            .collect::<BTreeSet<_>>();
        let slot_aliases = alias_inputs
            .into_iter()
            .map(|input| self.build_slot_alias_record(input, transaction_time_ms, &target_slot_ids))
            .collect::<ZResult<Vec<_>>>()?;
        let slot_alias_docs = slot_aliases
            .iter()
            .map(slot_alias_doc)
            .collect::<Result<Vec<_>, _>>()
            .map_err(json_error)?;
        upsert_many(&self.slot_aliases, slot_alias_docs)?;
        Ok(slot_aliases)
    }

    /// Re-canonicalises the batch's claims now that its slot aliases exist, and rewrites them.
    fn recanonicalize_batch_claims(
        &self,
        scope: &MemoryScope,
        claims: Vec<ClaimRecord>,
        claim_embeddings: &BTreeMap<MemoryId, Vec<f32>>,
    ) -> ZResult<Vec<ClaimRecord>> {
        let claims = claims
            .into_iter()
            .map(|claim| self.canonicalize_claim_at_valid_time(scope, claim))
            .collect::<ZResult<Vec<_>>>()?;
        upsert_many(
            &self.claims,
            claim_docs_with_embeddings(&claims, claim_embeddings)?,
        )?;
        Ok(claims)
    }

    /// Inserts the batch's rules and rebinds every rule of the scope. Returns the batch's rules
    /// as rebound and the rules whose projection must be reconciled: the batch's own and every
    /// rule whose binding the rebind changed.
    fn write_batch_rules(
        &self,
        scope: &MemoryScope,
        inputs: Vec<RuleInput>,
    ) -> ZResult<(Vec<RuleRecord>, Vec<RuleRecord>)> {
        let mut rules = inputs
            .into_iter()
            .map(|input| self.build_rule_record(input))
            .collect::<ZResult<Vec<_>>>()?;
        let rule_docs = rules
            .iter()
            .map(|rule| rule_doc(rule).map_err(json_error))
            .collect::<ZResult<Vec<_>>>()?;
        insert_many(&self.rules, rule_docs)?;
        let rules_before_rebind = self
            .scan_rules(scope, usize::MAX, None)?
            .into_iter()
            .map(|rule| (rule.id.clone(), rule))
            .collect::<BTreeMap<_, _>>();
        let rebound_rule_records = self.rebind_rules_for_scope(scope, None)?;
        let incoming_rule_ids = rules
            .iter()
            .map(|rule| rule.id.as_str())
            .collect::<BTreeSet<_>>();
        let reconciliation_rules = rebound_rule_records
            .iter()
            .filter(|rule| {
                incoming_rule_ids.contains(rule.id.as_str())
                    || rebind_changed_binding(&rules_before_rebind, rule)
            })
            .cloned()
            .collect::<Vec<_>>();
        let rebound_rules = rebound_rule_records
            .into_iter()
            .map(|rule| (rule.id.clone(), rule))
            .collect::<BTreeMap<_, _>>();
        for rule in &mut rules {
            if let Some(rebound) = rebound_rules.get(&rule.id) {
                *rule = rebound.clone();
            }
        }
        Ok((rules, reconciliation_rules))
    }

    fn write_batch_corrections(
        &self,
        scope: &MemoryScope,
        inputs: Vec<CorrectionRecord>,
    ) -> ZResult<Vec<CorrectionRecord>> {
        let mut corrections = Vec::with_capacity(inputs.len());
        for mut correction in inputs {
            correction.source_sequence_no =
                self.source_sequence_no_for_episode_ids(scope, &correction.source_episode_ids)?;
            let registry = self.canonical_registry_for_names_at(
                scope,
                correction.target_subject_key.as_deref().into_iter(),
                Some(correction.effective_at_ms),
            )?;
            corrections.push(canonicalize_correction(correction, &registry));
        }
        let correction_docs = corrections
            .iter()
            .map(|correction| correction_doc(correction).map_err(json_error))
            .collect::<ZResult<Vec<_>>>()?;
        insert_many(&self.corrections, correction_docs)?;
        Ok(corrections)
    }

    pub(super) fn extend_claim_embeddings_from_sources<C: Borrow<ClaimRecord>>(
        &self,
        scope: &MemoryScope,
        claims: &[C],
        embeddings: &mut BTreeMap<MemoryId, Vec<f32>>,
    ) -> ZResult<()> {
        let source_span_ids = claims
            .iter()
            .map(Borrow::borrow)
            .filter(|claim| !embeddings.contains_key(&claim.id))
            .flat_map(|claim| claim.source_span_ids.iter().cloned())
            .collect::<BTreeSet<_>>();
        if source_span_ids.is_empty() {
            return Ok(());
        }
        let docs = self
            .spans
            .fetch(source_span_ids.into_iter().collect::<Vec<_>>())?;
        let source_embeddings = docs
            .into_iter()
            .filter(|(_, doc)| doc_matches_scope(doc, scope))
            .filter_map(|(span_id, doc)| {
                doc.get_vec_f32("embedding")
                    .filter(|embedding| !embedding.is_empty())
                    .map(|embedding| (span_id, embedding.to_vec()))
            })
            .collect::<BTreeMap<_, _>>();

        for claim in claims.iter().map(Borrow::borrow) {
            if embeddings.contains_key(&claim.id) {
                continue;
            }
            let vectors = claim
                .source_span_ids
                .iter()
                .filter_map(|span_id| source_embeddings.get(span_id))
                .collect::<Vec<_>>();
            let Some(first) = vectors.first() else {
                continue;
            };
            let mut mean = vec![0.0f32; first.len()];
            let mut count = 0usize;
            for vector in vectors {
                if vector.len() != mean.len() {
                    continue;
                }
                for (sum, value) in mean.iter_mut().zip(vector.iter()) {
                    *sum += value;
                }
                count += 1;
            }
            if count == 0 {
                continue;
            }
            for value in &mut mean {
                *value /= count as f32;
            }
            let norm = mean.iter().map(|value| value * value).sum::<f32>().sqrt();
            if norm > 0.0 {
                for value in &mut mean {
                    *value /= norm;
                }
                embeddings.insert(claim.id.clone(), mean);
            }
        }
        Ok(())
    }
}

fn rule_binding_changed(before: &RuleRecord, after: &RuleRecord) -> bool {
    before.trigger_subject_entity_id != after.trigger_subject_entity_id
        || before.target_subject_entity_id != after.target_subject_entity_id
        || before.trigger_slot_id != after.trigger_slot_id
        || before.target_slot_id != after.target_slot_id
        || before.trigger_binding_status != after.trigger_binding_status
        || before.target_binding_status != after.target_binding_status
}

fn doc_matches_scope(doc: &Doc, scope: &MemoryScope) -> bool {
    let string_field = |name: &str| doc.fields.get(name).and_then(Value::as_str);
    string_field("space_id") == Some(scope.space_id.as_str())
        && string_field("tenant_id") == scope.tenant_id.as_deref()
        && string_field("user_id") == scope.user_id.as_deref()
        && string_field("agent_id") == scope.agent_id.as_deref()
        && string_field("project_id") == scope.project_id.as_deref()
        && string_field("thread_id") == scope.thread_id.as_deref()
}

fn validate_batch_scope(batch: &StateMutationBatch) -> ZResult<()> {
    let wrong_scope = batch
        .claims
        .iter()
        .any(|record| record.scope != batch.scope)
        || batch.rules.iter().any(|record| record.scope != batch.scope)
        || batch
            .slot_aliases
            .iter()
            .any(|record| record.scope != batch.scope)
        || batch
            .entities
            .iter()
            .any(|record| record.scope != batch.scope)
        || batch
            .corrections
            .iter()
            .any(|record| record.scope != batch.scope);
    if wrong_scope {
        return Err(Status::invalid_argument(
            "state mutation batch records must share one scope",
        ));
    }
    if batch
        .claims
        .iter()
        .any(|record| record.source_span_ids.is_empty() || record.source_episode_ids.is_empty())
    {
        return Err(Status::invalid_argument(
            "state mutation claims require source evidence ids",
        ));
    }
    if batch
        .rules
        .iter()
        .any(|record| record.source_span_ids.is_empty() || record.source_episode_ids.is_empty())
    {
        return Err(Status::invalid_argument(
            "state mutation rules require source evidence ids",
        ));
    }
    if batch
        .entities
        .iter()
        .any(|record| record.source_claim_ids.is_empty())
    {
        return Err(Status::invalid_argument(
            "state mutation entities require source claim ids",
        ));
    }
    if batch
        .slot_aliases
        .iter()
        .any(|record| record.source_claim_ids.is_empty())
    {
        return Err(Status::invalid_argument(
            "state mutation slot aliases require source claim ids",
        ));
    }
    if batch
        .corrections
        .iter()
        .any(|record| record.source_span_ids.is_empty() || record.source_episode_ids.is_empty())
    {
        return Err(Status::invalid_argument(
            "state mutation corrections require source evidence ids",
        ));
    }
    Ok(())
}

fn rebind_changed_binding(
    before_rebind: &BTreeMap<MemoryId, RuleRecord>,
    rule: &RuleRecord,
) -> bool {
    before_rebind
        .get(&rule.id)
        .is_some_and(|before| rule_binding_changed(before, rule))
}

pub(super) fn claim_docs_with_embeddings<C: Borrow<ClaimRecord>>(
    claims: &[C],
    claim_embeddings: &BTreeMap<MemoryId, Vec<f32>>,
) -> ZResult<Vec<Doc>> {
    claims
        .iter()
        .map(|claim| {
            let claim = claim.borrow();
            claim_doc(claim, claim_embeddings.get(&claim.id).map(Vec::as_slice)).map_err(json_error)
        })
        .collect()
}

fn merge_entity_history(entity: &mut EntityRecord, mut existing: EntityRecord) {
    let incoming_canonical_name = std::mem::replace(
        &mut entity.canonical_name,
        std::mem::take(&mut existing.canonical_name),
    );
    if crate::canonical_slot_part(&incoming_canonical_name)
        != crate::canonical_slot_part(&entity.canonical_name)
    {
        entity.aliases.push(incoming_canonical_name);
    }
    entity.absorb_provenance(existing);
}

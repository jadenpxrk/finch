use super::*;

impl MemoryStore {
    pub fn append_claim(&self, record: &ClaimRecord, embedding: Option<&[f32]>) -> ZResult<()> {
        let _mutation_guard = self.lock_state_mutation();
        #[cfg(not(test))]
        if !record.source_span_ids.is_empty() || !record.source_episode_ids.is_empty() {
            self.validate_evidence_references(
                &record.scope,
                &record.source_span_ids,
                &record.source_episode_ids,
            )?;
        }
        let scope = &record.scope;
        let mut record = record.clone();
        record.source_sequence_no =
            self.source_sequence_no_for_episode_ids(scope, &record.source_episode_ids)?;
        let alias_inputs =
            self.admit_bound_claim_slots(scope, std::slice::from_mut(&mut record))?;
        let mut record = self.canonicalize_claim_at_valid_time(scope, record)?;
        insert_one(
            &self.claims,
            claim_doc(&record, embedding).map_err(json_error)?,
        )?;
        if !alias_inputs.is_empty() {
            self.write_claim_slot_aliases(alias_inputs, &record)?;
            record = self.canonicalize_claim_at_valid_time(scope, record)?;
            upsert_many(
                &self.claims,
                vec![claim_doc(&record, embedding).map_err(json_error)?],
            )?;
        }
        let valid_at_ms = record.valid_from_ms.or(Some(record.observed_at_ms));
        self.rebind_rules_for_scope(scope, valid_at_ms)?;
        let known_ids = BTreeSet::from([record.id.clone()]);
        let applications = self.resolve_rules_for_changed_claims(
            std::slice::from_ref(&record),
            &known_ids,
            MAX_RULE_HOPS,
        )?;
        self.store_and_project_applications(
            scope,
            record.slot_id.as_ref(),
            &applications,
            valid_at_ms,
            &record.id,
        )
    }

    fn write_claim_slot_aliases(
        &self,
        alias_inputs: Vec<SlotAliasInput>,
        record: &ClaimRecord,
    ) -> ZResult<()> {
        let recorded_at_ms = system_time_ms();
        let target_slot_ids = record.slot_id.iter().cloned().collect::<BTreeSet<_>>();
        let alias_docs = alias_inputs
            .into_iter()
            .map(|input| {
                let alias =
                    self.build_slot_alias_record(input, recorded_at_ms, &target_slot_ids)?;
                slot_alias_doc(&alias).map_err(json_error)
            })
            .collect::<ZResult<Vec<_>>>()?;
        upsert_many(&self.slot_aliases, alias_docs)
    }

    /// Stores the claims rule applications derived and projects them, with `anchor_slot_id`,
    /// as one write keyed by `write_key`.
    pub(super) fn store_and_project_applications(
        &self,
        scope: &MemoryScope,
        anchor_slot_id: Option<&MemoryId>,
        applications: &[ResolvedRuleApplication],
        valid_at_ms: Option<i64>,
        write_key: &str,
    ) -> ZResult<()> {
        let derived_docs = applications
            .iter()
            .map(|application| claim_doc(&application.claim, None).map_err(json_error))
            .collect::<ZResult<Vec<_>>>()?;
        insert_many(&self.claims, derived_docs)?;
        let affected_slot_ids = anchor_slot_id
            .into_iter()
            .cloned()
            .chain(
                applications
                    .iter()
                    .filter_map(|application| application.claim.slot_id.clone()),
            )
            .collect::<BTreeSet<_>>();
        self.project_state_slots(
            scope,
            StateProjectionFrontier {
                slot_ids: &affected_slot_ids,
                claims: None,
                applications,
                valid_at_ms,
                write: ProjectionWrite {
                    time_ms: system_time_ms(),
                    key: write_key,
                },
            },
        )?;
        Ok(())
    }

    pub fn add_manual_claim(
        &self,
        input: ManualClaimInput,
        embedding: Option<&[f32]>,
    ) -> ZResult<ClaimRecord> {
        let mut claim = create_manual_claim(input);
        claim.source_sequence_no =
            self.source_sequence_no_for_episode_ids(&claim.scope, &claim.source_episode_ids)?;
        let registry = self.canonical_registry_for_names_at(
            &claim.scope,
            claim.subject.iter().map(String::as_str),
            claim.valid_from_ms.or(Some(claim.observed_at_ms)),
        )?;
        let claim = canonicalize_claim(claim, &registry);
        self.append_claim(&claim, embedding)?;
        Ok(claim)
    }

    pub(crate) fn append_profile(&self, record: &ProfileRecord) -> ZResult<()> {
        insert_one(&self.profiles, profile_doc(record).map_err(json_error)?)
    }

    pub fn add_profile(&self, input: ProfileInput) -> ZResult<ProfileRecord> {
        let profile = crate::ingest::create_profile(input);
        self.append_profile(&profile)?;
        Ok(profile)
    }

    pub(crate) fn append_entity(
        &self,
        record: &EntityRecord,
        embedding: Option<&[f32]>,
    ) -> ZResult<()> {
        let _mutation_guard = self.lock_state_mutation();
        #[cfg(not(test))]
        if !record.aliases.is_empty() && record.source_claim_ids.is_empty() {
            return Err(Status::invalid_argument(
                "entity aliases require source claim evidence",
            ));
        }
        #[cfg(not(test))]
        if !record.source_claim_ids.is_empty()
            && !self.validate_claim_references(
                &record.scope,
                &record.source_claim_ids,
                &BTreeSet::new(),
            )?
        {
            return Err(Status::invalid_argument(
                "entity source claims must exist in the entity scope",
            ));
        }
        self.ensure_entity_ids_owned_by_scope(std::slice::from_ref(record))?;
        upsert_many(
            &self.entities,
            vec![entity_doc(record, embedding).map_err(json_error)?],
        )?;
        self.upsert_entity_aliases(std::slice::from_ref(record), system_time_ms())?;
        self.rebind_rules_for_scope(&record.scope, None)?;
        self.refresh_state_projection(&record.scope, None)?;
        Ok(())
    }

    pub fn add_entity(
        &self,
        input: EntityInput,
        embedding: Option<&[f32]>,
    ) -> ZResult<EntityRecord> {
        let mut entity = create_entity(input);
        if let Some(mut existing) = self
            .entities_by_ids(&entity.scope, std::slice::from_ref(&entity.id))?
            .into_iter()
            .next()
        {
            entity.canonical_name = std::mem::take(&mut existing.canonical_name);
            entity.absorb_provenance(existing);
        }
        self.append_entity(&entity, embedding)?;
        Ok(entity)
    }

    pub(crate) fn append_edge(&self, record: &EdgeRecord) -> ZResult<()> {
        insert_one(&self.edges, edge_doc(record).map_err(json_error)?)
    }

    pub fn add_edge(&self, input: EdgeInput) -> ZResult<EdgeRecord> {
        let edge = create_edge(input);
        self.append_edge(&edge)?;
        Ok(edge)
    }
}

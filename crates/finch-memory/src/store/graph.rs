use super::*;

impl MemoryStore {
    pub fn add_slot_alias(
        &self,
        input: SlotAliasInput,
        recorded_at_ms: i64,
    ) -> ZResult<SlotAliasRecord> {
        let _mutation_guard = self.lock_state_mutation();
        let record = self.build_slot_alias_record(input, recorded_at_ms, &BTreeSet::new())?;
        upsert_many(
            &self.slot_aliases,
            vec![slot_alias_doc(&record).map_err(json_error)?],
        )?;
        self.rebind_rules_for_scope(&record.scope, record.valid_from_ms)?;
        self.refresh_state_projection(&record.scope, record.valid_from_ms)?;
        Ok(record)
    }

    pub(crate) fn build_slot_alias_record(
        &self,
        input: SlotAliasInput,
        recorded_at_ms: i64,
        additional_target_slot_ids: &BTreeSet<MemoryId>,
    ) -> ZResult<SlotAliasRecord> {
        if input.source_claim_ids.is_empty() {
            return Err(Status::invalid_argument(
                "slot aliases require source claim evidence",
            ));
        }
        let mut source_claims = self.claims_by_ids(&input.scope, &input.source_claim_ids)?;
        source_claims.retain(|claim| claim.scope == input.scope);
        if source_claims.len() != input.source_claim_ids.iter().collect::<BTreeSet<_>>().len() {
            return Err(Status::invalid_argument(
                "slot alias source claims must exist in the alias scope",
            ));
        }
        let canonical_slot_id = self.slot_alias_target(&input, additional_target_slot_ids)?;
        let registry = self.canonical_registry_for_names_at(
            &input.scope,
            std::iter::once(input.alias_subject.as_str()),
            input.valid_from_ms,
        )?;
        let alias_subject_key = canonical_slot_part(&input.alias_subject);
        let alias_predicate_key = canonical_slot_part(&input.alias_predicate);
        if alias_subject_key.is_empty() || alias_predicate_key.is_empty() {
            return Err(Status::invalid_argument(
                "slot alias subject and predicate must be non-empty",
            ));
        }
        let alias_subject_entity_id = registry.subject_entity_id(Some(&input.alias_subject));
        let subject_identity = alias_subject_entity_id
            .as_deref()
            .unwrap_or(&alias_subject_key);
        // Generated ids hash the scope; an explicit id stored under another scope would be overwritten.
        if let Some(id) = &input.id {
            if let Some(doc) = self.slot_aliases.fetch(vec![id.clone()])?.get(id) {
                if slot_alias_from_doc(doc)?.scope != input.scope {
                    return Err(Status::already_exists(format!(
                        "slot alias id {id} already belongs to another scope"
                    )));
                }
            }
        }
        let id = input.id.unwrap_or_else(|| {
            let hash = stable_hash_hex(&[
                &scope_filter(&input.scope),
                subject_identity,
                &alias_predicate_key,
                &canonical_slot_id,
                &input.source_claim_ids.join("\0"),
            ]);
            format!("slot_alias_{hash}")
        });
        Ok(SlotAliasRecord {
            id,
            scope: input.scope,
            status: MemoryStatus::Active,
            visibility: input.visibility,
            policy_tags: input.policy_tags,
            alias_subject_key,
            alias_predicate_key,
            alias_subject_entity_id,
            canonical_slot_id,
            source_claim_ids: input.source_claim_ids,
            valid_from_ms: input.valid_from_ms,
            valid_to_ms: input.valid_to_ms,
            recorded_at_ms,
            superseded_at_ms: None,
        })
    }

    /// The canonical slot an alias points at: the one named explicitly or through its target
    /// claim (they must agree), which must exist or be one of `additional_target_slot_ids`.
    fn slot_alias_target(
        &self,
        input: &SlotAliasInput,
        additional_target_slot_ids: &BTreeSet<MemoryId>,
    ) -> ZResult<MemoryId> {
        let target_from_claim = match input.target_claim_id.as_ref() {
            Some(claim_id) => self
                .claims_by_ids(&input.scope, std::slice::from_ref(claim_id))?
                .into_iter()
                .find(|claim| claim.scope == input.scope)
                .and_then(|claim| claim.slot_id),
            None => None,
        };
        let canonical_slot_id = match (input.canonical_slot_id.clone(), target_from_claim) {
            (Some(explicit), Some(from_claim)) if explicit != from_claim => {
                return Err(Status::invalid_argument(
                    "slot alias target claim and canonical slot disagree",
                ));
            }
            (Some(explicit), _) => explicit,
            (None, Some(from_claim)) => from_claim,
            (None, None) => {
                return Err(Status::invalid_argument(
                    "slot alias requires a target claim or canonical slot",
                ));
            }
        };
        let slot_exists = additional_target_slot_ids.contains(&canonical_slot_id)
            || self
                .scan_slots(&input.scope, usize::MAX, None)?
                .iter()
                .any(|slot| slot.scope == input.scope && slot.slot_key == canonical_slot_id);
        if !slot_exists {
            return Err(Status::invalid_argument(
                "slot alias target must be an existing canonical slot",
            ));
        }
        Ok(canonical_slot_id)
    }

    pub fn scan_slot_aliases(
        &self,
        scope: &MemoryScope,
        limit: usize,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<SlotAliasRecord>> {
        let _state_guard = self.lock_state_read();
        if limit == 0 {
            return Ok(Vec::new());
        }
        let query = VectorQuery::new("", Vec::new(), limit)
            .with_filter(scope_filter(scope))
            .with_output_fields(output_fields(SLOT_ALIAS_OUTPUT_FIELDS));
        let mut aliases = self
            .slot_aliases
            .scan_filter_only(query)?
            .into_iter()
            .map(|doc| slot_alias_from_doc(&doc))
            .collect::<ZResult<Vec<_>>>()?;
        aliases.retain(|alias| {
            matches!(alias.status, MemoryStatus::Active)
                && alias.superseded_at_ms.is_none()
                && interval_holds_at(alias.valid_from_ms, alias.valid_to_ms, at_ms)
        });
        aliases.sort_by(|a, b| a.id.cmp(&b.id));
        aliases.truncate(limit);
        Ok(aliases)
    }

    pub fn scan_entities(&self, scope: &MemoryScope, limit: usize) -> ZResult<Vec<EntityRecord>> {
        let _state_guard = self.lock_state_read();
        if limit == 0 {
            return Ok(Vec::new());
        }
        let query = VectorQuery::new("", Vec::new(), limit)
            .with_filter(scope_filter(scope))
            .with_output_fields(output_fields(ENTITY_OUTPUT_FIELDS));
        let mut entities = self
            .entities
            .scan_filter_only(query)?
            .into_iter()
            .map(|doc| entity_from_doc(&doc))
            .collect::<ZResult<Vec<_>>>()?;
        entities.retain(|entity| {
            matches!(entity.status, MemoryStatus::Active) && entity.scope.matches_filter(scope)
        });
        entities.sort_by(|a, b| a.id.cmp(&b.id));
        entities.truncate(limit);
        Ok(entities)
    }

    pub(crate) fn upsert_entity_aliases(
        &self,
        entities: &[EntityRecord],
        transaction_time_ms: i64,
    ) -> ZResult<Vec<EntityAliasRecord>> {
        let mut aliases = Vec::new();
        for entity in entities {
            let mut alias_keys = std::iter::once(entity.canonical_name.as_str())
                .chain(entity.aliases.iter().map(String::as_str))
                .map(canonical_slot_part)
                .filter(|alias| !alias.is_empty())
                .collect::<Vec<_>>();
            alias_keys.sort();
            alias_keys.dedup();
            for alias_key in alias_keys {
                let id = format!(
                    "entity_alias_{}",
                    stable_hash_hex(&[&scope_filter(&entity.scope), &alias_key, &entity.id])
                );
                aliases.push(EntityAliasRecord {
                    id,
                    scope: entity.scope.clone(),
                    status: entity.status,
                    visibility: entity.visibility,
                    policy_tags: entity.policy_tags.clone(),
                    alias_key,
                    entity_id: entity.id.clone(),
                    entity_type: entity.entity_type.clone(),
                    source_claim_ids: entity.source_claim_ids.clone(),
                    recorded_at_ms: transaction_time_ms,
                    superseded_at_ms: None,
                });
            }
        }
        let docs = aliases
            .iter()
            .map(entity_alias_doc)
            .collect::<Result<Vec<_>, _>>()
            .map_err(json_error)?;
        self.capture_state_mutation_docs(ENTITY_ALIASES_COLLECTION, &self.entity_aliases, &docs)?;
        upsert_many(&self.entity_aliases, docs)?;
        Ok(aliases)
    }

    pub(crate) fn entity_aliases_for_keys(
        &self,
        scope: &MemoryScope,
        alias_keys: &[String],
    ) -> ZResult<Vec<EntityAliasRecord>> {
        let mut alias_keys = alias_keys
            .iter()
            .map(|alias| canonical_slot_part(alias))
            .filter(|alias| !alias.is_empty())
            .collect::<Vec<_>>();
        alias_keys.sort();
        alias_keys.dedup();
        if alias_keys.is_empty() {
            return Ok(Vec::new());
        }
        let mut by_id = BTreeMap::new();
        for chunk in alias_keys.chunks(MAX_CONTAINS_FILTER_VALUES) {
            let query = VectorQuery::new("", Vec::new(), MAX_VECTOR_QUERY_TOPK)
                .with_filter(format!(
                    "{} AND status = 'active' AND ({})",
                    scope_filter(scope),
                    sql_or_eq_list("alias_key", chunk.iter().map(String::as_str)),
                ))
                .with_output_fields(output_fields(ENTITY_ALIAS_OUTPUT_FIELDS));
            for record in self
                .entity_aliases
                .query(query)?
                .into_iter()
                .map(|doc| entity_alias_from_doc(&doc))
                .collect::<ZResult<Vec<_>>>()?
            {
                by_id.insert(record.id.clone(), record);
            }
        }
        let mut records = by_id.into_values().collect::<Vec<_>>();
        records.retain(|record| record.superseded_at_ms.is_none());
        records.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(records)
    }

    pub(crate) fn entities_by_ids(
        &self,
        scope: &MemoryScope,
        entity_ids: &[String],
    ) -> ZResult<Vec<EntityRecord>> {
        let mut entity_ids = entity_ids.to_vec();
        entity_ids.sort();
        entity_ids.dedup();
        if entity_ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut by_id = BTreeMap::new();
        for chunk in entity_ids.chunks(MAX_CONTAINS_FILTER_VALUES) {
            let query = VectorQuery::new("", Vec::new(), MAX_VECTOR_QUERY_TOPK)
                .with_filter(format!(
                    "{} AND status = 'active' AND ({})",
                    scope_filter(scope),
                    sql_or_eq_list("id", chunk.iter().map(String::as_str)),
                ))
                .with_output_fields(output_fields(ENTITY_OUTPUT_FIELDS));
            for entity in self
                .entities
                .query(query)?
                .into_iter()
                .map(|doc| entity_from_doc(&doc))
                .collect::<ZResult<Vec<_>>>()?
            {
                by_id.insert(entity.id.clone(), entity);
            }
        }
        let mut entities = by_id.into_values().collect::<Vec<_>>();
        entities.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(entities)
    }

    /// Fails when an entity id is already stored under another scope, which an upsert would overwrite.
    pub(crate) fn ensure_entity_ids_owned_by_scope(
        &self,
        entities: &[EntityRecord],
    ) -> ZResult<()> {
        let stored = self
            .entities
            .fetch(entities.iter().map(|entity| entity.id.clone()).collect())?;
        for entity in entities {
            let Some(doc) = stored.get(&entity.id) else {
                continue;
            };
            if entity_from_doc(doc)?.scope != entity.scope {
                return Err(Status::already_exists(format!(
                    "entity id {} already belongs to another scope",
                    entity.id
                )));
            }
        }
        Ok(())
    }

    pub(crate) fn expand_edges_one_hop(
        &self,
        scope: &MemoryScope,
        seed_entity_ids: &[String],
        limit: usize,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<EdgeRecord>> {
        if limit == 0 || seed_entity_ids.is_empty() {
            return Ok(Vec::new());
        }
        let id_list = sql_string_list(seed_entity_ids.iter().map(String::as_str));
        let filter = format!(
            "{} AND status = 'active' AND (src_entity_id IN ({id_list}) OR dst_entity_id IN ({id_list})){}",
            scope_filter(scope),
            interval_filter_clause(at_ms)
        );
        let query = VectorQuery::new("", Vec::new(), limit)
            .with_filter(filter)
            .with_output_fields(output_fields(EDGE_OUTPUT_FIELDS));
        let mut edges = self
            .edges
            .scan_filter_only(query)?
            .into_iter()
            .map(|doc| edge_from_doc(&doc))
            .collect::<ZResult<Vec<_>>>()?;
        edges.retain(|edge| {
            matches!(edge.status, MemoryStatus::Active)
                && edge.scope.matches_filter(scope)
                && interval_holds_at(edge.valid_from_ms, edge.valid_to_ms, at_ms)
        });
        edges.truncate(limit);
        Ok(edges)
    }

    pub fn expand_edges_ranked(
        &self,
        scope: &MemoryScope,
        seed_entity_ids: &[String],
        max_depth: usize,
        limit: usize,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<GraphExpansionHit>> {
        let _state_guard = self.lock_state_read();
        if max_depth == 0 || limit == 0 || seed_entity_ids.is_empty() {
            return Ok(Vec::new());
        }

        let mut frontier = seed_entity_ids.to_vec();
        let mut seen_entities = frontier.iter().cloned().collect::<BTreeSet<_>>();
        let mut seen_edges = BTreeSet::<String>::new();
        let mut hits = Vec::new();

        // Scores are known only after scanning, so every reachable edge is ranked before the limit.
        for depth in 1..=max_depth {
            let edges = self.expand_edges_one_hop(scope, &frontier, usize::MAX, at_ms)?;
            frontier.clear();
            for edge in edges {
                if !seen_edges.insert(edge.id.clone()) {
                    continue;
                }
                let endpoints = [&edge.src_entity_id, &edge.dst_entity_id];
                frontier.extend(unseen_entity_ids(endpoints, &mut seen_entities));
                let score = edge.confidence.unwrap_or(1.0).max(0.0) / depth as f32;
                hits.push(GraphExpansionHit { edge, depth, score });
            }
            if frontier.is_empty() {
                break;
            }
        }

        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.depth.cmp(&b.depth))
                .then_with(|| a.edge.id.cmp(&b.edge.id))
        });
        hits.truncate(limit);
        Ok(hits)
    }
}

/// The endpoint ids not seen before, recording them as seen.
fn unseen_entity_ids(endpoints: [&String; 2], seen: &mut BTreeSet<String>) -> Vec<String> {
    endpoints
        .into_iter()
        .filter(|entity_id| seen.insert((*entity_id).clone()))
        .cloned()
        .collect()
}

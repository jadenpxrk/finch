use super::*;

impl MemoryStore {
    /// Maintenance rebuild of a whole scope: the write-time projector run over every slot that
    /// has current evidence or a rule endpoint. State and traces of any other slot are retired.
    pub fn refresh_state_projection(
        &self,
        scope: &MemoryScope,
        valid_at_ms: Option<i64>,
    ) -> ZResult<()> {
        let transaction_time_ms = system_time_ms();
        let (claims, applications) = self.rebuild_current_claims(scope, valid_at_ms)?;
        let slot_ids = self.rebuild_frontier_slot_ids(scope, &claims, valid_at_ms)?;
        let active_states = self.scan_state_records(
            scope,
            StateRecordScan {
                limit: usize::MAX,
                temporal: BiTemporalQuery::default(),
            },
        )?;
        let active_traces = self.scan_dependency_traces(scope, usize::MAX, None)?;
        let write_key =
            rebuild_write_key(scope, valid_at_ms, &claims, &active_states, &active_traces);
        self.retire_outside_frontier(&slot_ids, active_states, active_traces, transaction_time_ms)?;
        self.project_state_slots(
            scope,
            StateProjectionFrontier {
                slot_ids: &slot_ids,
                claims: Some(claims),
                applications: &applications,
                valid_at_ms,
                write: ProjectionWrite {
                    time_ms: transaction_time_ms,
                    key: &write_key,
                },
            },
        )?;
        Ok(())
    }

    /// The scope's current claims re-canonicalised, plus the claims its rules derive from them.
    fn rebuild_current_claims(
        &self,
        scope: &MemoryScope,
        valid_at_ms: Option<i64>,
    ) -> ZResult<(Vec<ClaimRecord>, Vec<ResolvedRuleApplication>)> {
        let claims = self.scan_current_claims(scope, usize::MAX, valid_at_ms)?;
        let registry = self.canonical_registry_for_names_at(
            scope,
            claims.iter().filter_map(|claim| claim.subject.as_deref()),
            valid_at_ms,
        )?;
        let mut claims = claims
            .into_iter()
            .map(|claim| canonicalize_claim(claim, &registry))
            .collect::<Vec<_>>();
        let applications =
            self.resolve_rules_for_changed_claims(&claims, &BTreeSet::new(), MAX_RULE_HOPS)?;
        claims.extend(
            applications
                .iter()
                .map(|application| application.claim.clone()),
        );
        dedupe_claims_by_id(&mut claims);
        Ok((claims, applications))
    }

    /// Every slot with current evidence or a rule endpoint.
    fn rebuild_frontier_slot_ids(
        &self,
        scope: &MemoryScope,
        claims: &[ClaimRecord],
        valid_at_ms: Option<i64>,
    ) -> ZResult<BTreeSet<MemoryId>> {
        let rule_endpoints = self
            .scan_rules(scope, usize::MAX, valid_at_ms)?
            .into_iter()
            .flat_map(|rule| rule.trigger_slot_id.into_iter().chain(rule.target_slot_id));
        Ok(claims
            .iter()
            .filter_map(|claim| claim.slot_id.clone())
            .chain(rule_endpoints)
            .collect())
    }

    /// Supersedes the active states and traces of every slot outside the rebuild frontier.
    fn retire_outside_frontier(
        &self,
        slot_ids: &BTreeSet<MemoryId>,
        active_states: Vec<StateRecord>,
        active_traces: Vec<DependencyTraceRecord>,
        transaction_time_ms: i64,
    ) -> ZResult<()> {
        let in_frontier =
            |slot_id: &Option<MemoryId>| slot_id.as_ref().is_some_and(|id| slot_ids.contains(id));
        let retired_states = active_states
            .into_iter()
            .filter(|record| !in_frontier(&record.slot_id))
            .map(|mut record| {
                record.status = MemoryStatus::Superseded;
                record.projected_at_ms = transaction_time_ms;
                record.superseded_at_ms = Some(transaction_time_ms);
                state_record_doc(&record).map_err(json_error)
            })
            .collect::<ZResult<Vec<_>>>()?;
        upsert_many(&self.state_records, retired_states)?;
        let retired_traces = active_traces
            .into_iter()
            .filter(|record| !in_frontier(&record.target_slot_id))
            .map(|mut record| {
                record.status = MemoryStatus::Superseded;
                record.projected_at_ms = transaction_time_ms;
                record.superseded_at_ms = Some(transaction_time_ms);
                dependency_trace_doc(&record).map_err(json_error)
            })
            .collect::<ZResult<Vec<_>>>()?;
        upsert_many(&self.dependency_traces, retired_traces)
    }

    /// The one state projector: projects the current claims of `frontier.slot_ids` into durable
    /// state and dependency traces, reconciling validity intervals against the active versions.
    /// Writes and the scope rebuild both call it; only the frontier differs.
    pub(crate) fn project_state_slots(
        &self,
        scope: &MemoryScope,
        frontier: StateProjectionFrontier<'_>,
    ) -> ZResult<Vec<StateRecord>> {
        let StateProjectionFrontier {
            slot_ids,
            claims,
            applications,
            valid_at_ms,
            write,
        } = frontier;
        if slot_ids.is_empty() {
            return Ok(Vec::new());
        }
        let claims = match claims {
            Some(claims) => claims,
            None => {
                self.scan_current_claims_for_slot_ids(scope, slot_ids, usize::MAX, valid_at_ms)?
            }
        };
        let rules = self.rules_touching_slots(scope, slot_ids, valid_at_ms)?;
        let rule_ids_by_claim = rule_ids_by_claim(applications);
        let mut records = versioned_state_records(scope, &claims, &rule_ids_by_claim, write);
        let mut slots = slot_records_from_projection(
            scope,
            &claims,
            &rules,
            applications,
            write.time_ms,
            valid_at_ms,
        );
        self.write_slot_identities(scope, slot_ids, &mut slots, write)?;

        let active_records = self.current_state_records_for_slot_ids(scope, slot_ids)?;
        records.extend(self.project_future_set_snapshots(
            scope,
            slot_ids,
            &active_records,
            &rule_ids_by_claim,
            valid_at_ms,
            write,
        )?);
        let records_at_valid_time = active_records
            .iter()
            .filter(|record| state_record_valid_at(record, valid_at_ms))
            .cloned()
            .collect::<Vec<_>>();
        if applications.is_empty()
            && state_record_sets_semantically_equal(&records, &records_at_valid_time)
        {
            return Ok(records_at_valid_time);
        }
        self.write_state_versions(&mut records, active_records, write, valid_at_ms)?;
        self.write_dependency_traces(scope, slot_ids, &claims, applications, valid_at_ms, write)?;
        Ok(records)
    }

    /// Rules whose trigger or target is one of `slot_ids`.
    fn rules_touching_slots(
        &self,
        scope: &MemoryScope,
        slot_ids: &BTreeSet<MemoryId>,
        valid_at_ms: Option<i64>,
    ) -> ZResult<Vec<RuleRecord>> {
        let touches = |slot_id: &Option<MemoryId>| {
            slot_id
                .as_ref()
                .is_some_and(|slot_id| slot_ids.contains(slot_id))
        };
        Ok(self
            .scan_rules(scope, usize::MAX, valid_at_ms)?
            .into_iter()
            .filter(|rule| touches(&rule.trigger_slot_id) || touches(&rule.target_slot_id))
            .collect())
    }

    /// Merges the projected slot identities with the active ones and writes their revisions.
    fn write_slot_identities(
        &self,
        scope: &MemoryScope,
        slot_ids: &BTreeSet<MemoryId>,
        slots: &mut Vec<CanonicalSlotRecord>,
        write: ProjectionWrite<'_>,
    ) -> ZResult<()> {
        let active_slots = self.current_slots_for_slot_keys(scope, slot_ids)?;
        merge_slot_identity_records(slots, active_slots);
        prepare_slot_identity_revisions(slots, write);
        let slot_docs = slots
            .iter()
            .map(|record| slot_doc(record).map_err(json_error))
            .collect::<ZResult<Vec<_>>>()?;
        if !slot_docs.is_empty() {
            self.capture_state_mutation_docs(SLOTS_COLLECTION, &self.slots, &slot_docs)?;
            upsert_many(&self.slots, slot_docs)?;
        }
        Ok(())
    }

    /// Reconciles the new state versions' validity intervals against the active versions and
    /// writes the new, historical, and superseded versions.
    fn write_state_versions(
        &self,
        records: &mut Vec<StateRecord>,
        active_records: Vec<StateRecord>,
        write: ProjectionWrite<'_>,
        valid_at_ms: Option<i64>,
    ) -> ZResult<()> {
        let (historical_revisions, superseded_records) =
            reconcile_state_record_intervals(records, active_records, write, valid_at_ms);
        let state_docs = records
            .iter()
            .chain(historical_revisions.iter())
            .chain(superseded_records.iter())
            .map(|record| state_record_doc(record).map_err(json_error))
            .collect::<ZResult<Vec<_>>>()?;
        self.capture_state_mutation_docs(
            STATE_RECORDS_COLLECTION,
            &self.state_records,
            &state_docs,
        )?;
        upsert_many(&self.state_records, state_docs)
    }

    /// Writes one dependency trace per rule application on the frontier, reconciled against the
    /// active traces of those slots.
    fn write_dependency_traces(
        &self,
        scope: &MemoryScope,
        slot_ids: &BTreeSet<MemoryId>,
        claims: &[ClaimRecord],
        applications: &[ResolvedRuleApplication],
        valid_at_ms: Option<i64>,
        write: ProjectionWrite<'_>,
    ) -> ZResult<()> {
        let mut traces = versioned_dependency_traces(scope, slot_ids, applications, write);
        let active_traces = self.current_dependency_traces_for_slot_ids(scope, slot_ids)?;
        let active_claim_ids = claims
            .iter()
            .map(|claim| claim.id.clone())
            .collect::<BTreeSet<_>>();
        let (historical_traces, superseded_traces) = reconcile_dependency_trace_intervals(
            &mut traces,
            active_traces,
            &active_claim_ids,
            write,
            valid_at_ms,
        );
        let trace_docs = traces
            .iter()
            .chain(historical_traces.iter())
            .chain(superseded_traces.iter())
            .map(|record| dependency_trace_doc(record).map_err(json_error))
            .collect::<ZResult<Vec<_>>>()?;
        self.capture_state_mutation_docs(
            DEPENDENCY_TRACES_COLLECTION,
            &self.dependency_traces,
            &trace_docs,
        )?;
        upsert_many(&self.dependency_traces, trace_docs)
    }
}

/// A rebuild is named by its inputs and by the versions it starts from, so two rebuilds over the
/// same claims still write distinct versions.
fn rebuild_write_key(
    scope: &MemoryScope,
    valid_at_ms: Option<i64>,
    claims: &[ClaimRecord],
    active_states: &[StateRecord],
    active_traces: &[DependencyTraceRecord],
) -> String {
    let mut claim_ids = claims
        .iter()
        .map(|claim| claim.id.as_str())
        .collect::<Vec<_>>();
    claim_ids.sort_unstable();
    let prior_ids = active_states
        .iter()
        .map(|record| record.id.as_str())
        .chain(active_traces.iter().map(|record| record.id.as_str()))
        .collect::<Vec<_>>();
    let valid_at = valid_at_ms.map(|at| at.to_string()).unwrap_or_default();
    stable_hash_hex(&[
        &scope.space_id,
        &valid_at,
        &claim_ids.join("\u{0}"),
        &prior_ids.join("\u{0}"),
    ])
}

fn rule_ids_by_claim(
    applications: &[ResolvedRuleApplication],
) -> BTreeMap<MemoryId, Vec<MemoryId>> {
    let mut rule_ids = BTreeMap::<MemoryId, Vec<MemoryId>>::new();
    for application in applications {
        rule_ids
            .entry(application.claim.id.clone())
            .or_default()
            .push(application.rule_id.clone());
    }
    rule_ids
}

/// State records of `claims` as new versions of this write, sorted by id.
fn versioned_state_records(
    scope: &MemoryScope,
    claims: &[ClaimRecord],
    rule_ids_by_claim: &BTreeMap<MemoryId, Vec<MemoryId>>,
    write: ProjectionWrite<'_>,
) -> Vec<StateRecord> {
    let mut records = state_records_from_claims(scope, claims, rule_ids_by_claim, write.time_ms);
    for record in &mut records {
        record.projected_at_ms = write.time_ms;
        record.recorded_at_ms = write.time_ms;
        record.superseded_at_ms = None;
        record.id = versioned_projection_id(
            &record.state_key,
            write.key,
            &[
                record.state_kind.as_str(),
                record.object_value.as_deref().unwrap_or_default(),
                &record.members.join("\u{0}"),
                &record.claim_ids.join("\u{0}"),
            ],
        );
    }
    records.sort_by(|a, b| a.id.cmp(&b.id));
    records
}

/// Dependency traces of the applications that derived a claim on one of `slot_ids`, as new
/// versions of this write.
fn versioned_dependency_traces(
    scope: &MemoryScope,
    slot_ids: &BTreeSet<MemoryId>,
    applications: &[ResolvedRuleApplication],
    write: ProjectionWrite<'_>,
) -> Vec<DependencyTraceRecord> {
    applications
        .iter()
        .filter(|application| {
            application
                .claim
                .slot_id
                .as_ref()
                .is_some_and(|slot_id| slot_ids.contains(slot_id))
        })
        .map(|application| {
            let mut record = dependency_trace_from_application(scope, application, write.time_ms);
            record.recorded_at_ms = write.time_ms;
            record.superseded_at_ms = None;
            record.id = versioned_projection_id(
                &record.trace_key,
                write.key,
                &[
                    &record.rule_id,
                    &record.trigger_claim_id,
                    &record.derived_claim_id,
                ],
            );
            record
        })
        .collect()
}

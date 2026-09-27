use super::*;

impl MemoryStore {
    pub fn scan_state_records(
        &self,
        scope: &MemoryScope,
        scan: StateRecordScan,
    ) -> ZResult<Vec<StateRecord>> {
        let StateRecordScan { limit, temporal } = scan;
        if limit == 0 {
            return Ok(Vec::new());
        }
        let query = VectorQuery::new("", Vec::new(), limit)
            .with_filter(scope_filter(scope))
            .with_output_fields(output_fields(STATE_RECORD_OUTPUT_FIELDS));
        let mut records = self
            .state_records
            .scan_filter_only(query)?
            .into_iter()
            .map(|doc| state_record_from_doc(&doc))
            .collect::<ZResult<Vec<_>>>()?
            .into_iter()
            .filter(|record| state_record_visible_at(record, scope, temporal))
            .collect::<Vec<_>>();
        sort_by_state_priority(&mut records);
        records.truncate(limit);
        Ok(records)
    }

    pub(crate) fn scan_state_records_bitemporal_for_slot_ids(
        &self,
        scope: &MemoryScope,
        slot_ids: &BTreeSet<MemoryId>,
        limit: usize,
        temporal: BiTemporalQuery,
    ) -> ZResult<Vec<StateRecord>> {
        self.scan_visible_state_records_where_in(scope, "slot_id", slot_ids, limit, temporal)
    }

    pub(super) fn scan_state_records_bitemporal_for_subject_entity_ids(
        &self,
        scope: &MemoryScope,
        subject_entity_ids: &BTreeSet<MemoryId>,
        limit: usize,
        temporal: BiTemporalQuery,
    ) -> ZResult<Vec<StateRecord>> {
        self.scan_visible_state_records_where_in(
            scope,
            "subject_entity_id",
            subject_entity_ids,
            limit,
            temporal,
        )
    }

    /// Visible states whose `field` is one of `values`, highest priority first.
    fn scan_visible_state_records_where_in(
        &self,
        scope: &MemoryScope,
        field: &str,
        values: &BTreeSet<MemoryId>,
        limit: usize,
        temporal: BiTemporalQuery,
    ) -> ZResult<Vec<StateRecord>> {
        if values.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let values = values.iter().map(String::as_str).collect::<Vec<_>>();
        let docs = scan_in_chunks(
            &self.state_records,
            &values,
            limit,
            STATE_RECORD_OUTPUT_FIELDS,
            |chunk| scoped_field_in(scope, field, chunk),
        )?;
        let by_id = docs
            .iter()
            .map(|doc| state_record_from_doc(doc))
            .collect::<ZResult<Vec<_>>>()?
            .into_iter()
            .filter(|record| state_record_visible_at(record, scope, temporal))
            .map(|record| (record.id.clone(), record))
            .collect::<BTreeMap<_, _>>();
        let mut records = by_id.into_values().collect::<Vec<_>>();
        sort_by_state_priority(&mut records);
        records.truncate(limit);
        Ok(records)
    }

    /// Number of state versions a slot has ever held (all kinds). A slot with more than one
    /// version has a real read-view choice (current vs history), so callers gating a
    /// selection fast path must see the full history, not just currently visible state.
    pub fn slot_state_version_count(
        &self,
        scope: &MemoryScope,
        slot_id: &MemoryId,
        at_ms: Option<i64>,
    ) -> ZResult<usize> {
        let ids = BTreeSet::from([slot_id.clone()]);
        let temporal = BiTemporalQuery {
            valid_at_ms: at_ms,
            transaction_at_ms: None,
        };
        Ok(self
            .scan_state_record_history_for_slot_ids(scope, &ids, temporal)?
            .len())
    }

    /// Every version of the slots' states recorded by `temporal.transaction_at_ms` that began by
    /// `temporal.valid_at_ms`, oldest first.
    pub(super) fn scan_state_record_history_for_slot_ids(
        &self,
        scope: &MemoryScope,
        slot_ids: &BTreeSet<MemoryId>,
        temporal: BiTemporalQuery,
    ) -> ZResult<Vec<StateRecord>> {
        if slot_ids.is_empty() {
            return Ok(Vec::new());
        }
        let slot_ids = slot_ids.iter().map(String::as_str).collect::<Vec<_>>();
        let docs = scan_in_chunks(
            &self.state_records,
            &slot_ids,
            usize::MAX,
            STATE_RECORD_OUTPUT_FIELDS,
            |chunk| scoped_field_in(scope, "slot_id", chunk),
        )?;
        let by_id = docs
            .iter()
            .map(|doc| state_record_from_doc(doc))
            .collect::<ZResult<Vec<_>>>()?
            .into_iter()
            .filter(|record| state_record_in_history_at(record, scope, temporal))
            .map(|record| (record.id.clone(), record))
            .collect::<BTreeMap<_, _>>();
        let mut records = by_id.into_values().collect::<Vec<_>>();
        records.sort_by(|a, b| {
            a.observed_at_ms
                .cmp(&b.observed_at_ms)
                .then_with(|| a.id.cmp(&b.id))
        });
        Ok(records)
    }

    pub(crate) fn current_state_records_for_slot_ids(
        &self,
        scope: &MemoryScope,
        slot_ids: &BTreeSet<MemoryId>,
    ) -> ZResult<Vec<StateRecord>> {
        if slot_ids.is_empty() {
            return Ok(Vec::new());
        }
        let values = slot_ids.iter().map(String::as_str).collect::<Vec<_>>();
        let docs = scan_in_chunks(
            &self.state_records,
            &values,
            usize::MAX,
            STATE_RECORD_OUTPUT_FIELDS,
            |chunk| scoped_field_in(scope, "slot_id", chunk),
        )?;
        let by_id = docs
            .iter()
            .map(|doc| state_record_from_doc(doc))
            .collect::<ZResult<Vec<_>>>()?
            .into_iter()
            .filter(|record| current_state_on_slots(record, scope, slot_ids))
            .map(|record| (record.id.clone(), record))
            .collect::<BTreeMap<_, _>>();
        Ok(by_id.into_values().collect())
    }

    pub fn scan_dependency_traces(
        &self,
        scope: &MemoryScope,
        limit: usize,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<DependencyTraceRecord>> {
        self.scan_dependency_traces_bitemporal(
            scope,
            limit,
            BiTemporalQuery {
                valid_at_ms: at_ms,
                transaction_at_ms: None,
            },
        )
    }

    pub(crate) fn scan_dependency_traces_bitemporal(
        &self,
        scope: &MemoryScope,
        limit: usize,
        temporal: BiTemporalQuery,
    ) -> ZResult<Vec<DependencyTraceRecord>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let query = VectorQuery::new("", Vec::new(), limit)
            .with_filter(scope_filter(scope))
            .with_output_fields(output_fields(DEPENDENCY_TRACE_OUTPUT_FIELDS));
        let mut records = self
            .dependency_traces
            .scan_filter_only(query)?
            .into_iter()
            .map(|doc| dependency_trace_from_doc(&doc))
            .collect::<ZResult<Vec<_>>>()?
            .into_iter()
            .filter(|record| dependency_trace_visible_at(record, scope, temporal))
            .collect::<Vec<_>>();
        records.sort_by(|a, b| {
            a.hop
                .cmp(&b.hop)
                .then_with(|| a.observed_at_ms.cmp(&b.observed_at_ms))
                .then_with(|| a.id.cmp(&b.id))
        });
        records.truncate(limit);
        Ok(records)
    }

    pub(crate) fn current_dependency_traces_for_slot_ids(
        &self,
        scope: &MemoryScope,
        slot_ids: &BTreeSet<MemoryId>,
    ) -> ZResult<Vec<DependencyTraceRecord>> {
        if slot_ids.is_empty() {
            return Ok(Vec::new());
        }
        let values = slot_ids.iter().map(String::as_str).collect::<Vec<_>>();
        let docs = scan_in_chunks(
            &self.dependency_traces,
            &values,
            usize::MAX,
            DEPENDENCY_TRACE_OUTPUT_FIELDS,
            |chunk| scoped_field_in(scope, "target_slot_id", chunk),
        )?;
        let by_id = docs
            .iter()
            .map(|doc| dependency_trace_from_doc(doc))
            .collect::<ZResult<Vec<_>>>()?
            .into_iter()
            .filter(|record| current_trace_on_slots(record, scope, slot_ids))
            .map(|record| (record.id.clone(), record))
            .collect::<BTreeMap<_, _>>();
        let mut records = by_id.into_values().collect::<Vec<_>>();
        records.sort_by(|a, b| {
            a.hop
                .cmp(&b.hop)
                .then_with(|| a.observed_at_ms.cmp(&b.observed_at_ms))
                .then_with(|| a.id.cmp(&b.id))
        });
        Ok(records)
    }

    pub fn scan_slots(
        &self,
        scope: &MemoryScope,
        limit: usize,
        at_ms: Option<i64>,
    ) -> ZResult<Vec<CanonicalSlotRecord>> {
        self.scan_slots_bitemporal(
            scope,
            limit,
            BiTemporalQuery {
                valid_at_ms: at_ms,
                transaction_at_ms: None,
            },
        )
    }

    pub(crate) fn scan_slots_bitemporal(
        &self,
        scope: &MemoryScope,
        limit: usize,
        temporal: BiTemporalQuery,
    ) -> ZResult<Vec<CanonicalSlotRecord>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let query = VectorQuery::new("", Vec::new(), usize::MAX)
            .with_filter(scope_filter(scope))
            .with_output_fields(output_fields(SLOT_OUTPUT_FIELDS));
        let stored_records = self
            .slots
            .scan_filter_only(query)?
            .into_iter()
            .map(|doc| slot_from_doc(&doc))
            .collect::<ZResult<Vec<_>>>()?
            .into_iter()
            .filter(|record| record.scope.matches_filter(scope))
            .filter(|record| {
                temporal
                    .transaction_at_ms
                    .is_none_or(|at| record.recorded_at_ms <= at)
            })
            .collect::<Vec<_>>();
        let mut records = Vec::new();
        merge_slot_identity_records(&mut records, stored_records);
        records.sort_by(|a, b| a.slot_key.cmp(&b.slot_key));
        records.truncate(limit);
        Ok(records)
    }

    pub(in crate::store) fn current_slots_for_slot_keys(
        &self,
        scope: &MemoryScope,
        slot_keys: &BTreeSet<MemoryId>,
    ) -> ZResult<Vec<CanonicalSlotRecord>> {
        if slot_keys.is_empty() {
            return Ok(Vec::new());
        }
        let mut records = Vec::new();
        let slot_keys = slot_keys.iter().cloned().collect::<Vec<_>>();
        for chunk in slot_keys.chunks(MAX_CONTAINS_FILTER_VALUES) {
            let filter = format!(
                "{} AND ({})",
                scope_filter(scope),
                sql_or_eq_list("slot_key", chunk.iter().map(String::as_str)),
            );
            let query = VectorQuery::new("", Vec::new(), usize::MAX)
                .with_filter(filter)
                .with_output_fields(output_fields(SLOT_OUTPUT_FIELDS));
            for record in self
                .slots
                .scan_filter_only(query)?
                .into_iter()
                .map(|doc| slot_from_doc(&doc))
                .collect::<ZResult<Vec<_>>>()?
            {
                if record.scope.matches_filter(scope) && slot_keys.contains(&record.slot_key) {
                    records.push(record);
                }
            }
        }
        let mut identities = Vec::new();
        merge_slot_identity_records(&mut identities, records);
        Ok(identities)
    }

    pub(crate) fn canonical_registry_for_names<I, S>(
        &self,
        scope: &MemoryScope,
        names: I,
    ) -> ZResult<CanonicalRegistry>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.canonical_registry_for_names_at(scope, names, None)
    }

    pub(crate) fn canonical_registry_for_names_at<I, S>(
        &self,
        scope: &MemoryScope,
        names: I,
        at_ms: Option<i64>,
    ) -> ZResult<CanonicalRegistry>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let alias_keys = names
            .into_iter()
            .map(|name| canonical_slot_part(name.as_ref()))
            .filter(|name| !name.is_empty())
            .collect::<Vec<_>>();
        Ok(CanonicalRegistry::from_alias_records(
            &self.entity_aliases_for_keys(scope, &alias_keys)?,
            &self.scan_slot_aliases(scope, usize::MAX, at_ms)?,
        ))
    }
}

fn sort_by_state_priority(records: &mut [StateRecord]) {
    records.sort_by(|a, b| {
        state_record_priority(a.state_kind)
            .cmp(&state_record_priority(b.state_kind))
            .then_with(|| a.observed_at_ms.cmp(&b.observed_at_ms))
            .then_with(|| a.id.cmp(&b.id))
    });
}

/// A state version recorded by `temporal.transaction_at_ms` that began by `temporal.valid_at_ms`,
/// whether or not it has ended since.
fn state_record_in_history_at(
    record: &StateRecord,
    scope: &MemoryScope,
    temporal: BiTemporalQuery,
) -> bool {
    state_record_recorded_at(record, scope, temporal)
        && temporal
            .valid_at_ms
            .is_none_or(|at| record.valid_from_ms.unwrap_or(record.observed_at_ms) <= at)
}

fn current_state_on_slots(
    record: &StateRecord,
    scope: &MemoryScope,
    slot_ids: &BTreeSet<MemoryId>,
) -> bool {
    record.scope.matches_filter(scope)
        && record
            .slot_id
            .as_ref()
            .is_some_and(|slot_id| slot_ids.contains(slot_id))
        && transaction_visible_at(record.recorded_at_ms, record.superseded_at_ms, None)
}

fn current_trace_on_slots(
    record: &DependencyTraceRecord,
    scope: &MemoryScope,
    slot_ids: &BTreeSet<MemoryId>,
) -> bool {
    record.scope.matches_filter(scope)
        && record
            .target_slot_id
            .as_ref()
            .is_some_and(|slot_id| slot_ids.contains(slot_id))
        && transaction_visible_at(record.recorded_at_ms, record.superseded_at_ms, None)
}

fn dependency_trace_visible_at(
    record: &DependencyTraceRecord,
    scope: &MemoryScope,
    temporal: BiTemporalQuery,
) -> bool {
    record.scope.matches_filter(scope)
        && transaction_visible_at(
            record.recorded_at_ms,
            record.superseded_at_ms,
            temporal.transaction_at_ms,
        )
        && temporal.valid_at_ms.is_none_or(|at| {
            record.valid_from_ms.is_none_or(|from| from <= at)
                && record.valid_to_ms.is_none_or(|to| at < to)
        })
}

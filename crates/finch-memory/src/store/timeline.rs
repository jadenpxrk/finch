use super::*;

impl MemoryStore {
    pub(super) fn project_future_set_snapshots(
        &self,
        scope: &MemoryScope,
        affected_slot_ids: &BTreeSet<MemoryId>,
        active_records: &[StateRecord],
        rule_ids_by_claim: &BTreeMap<MemoryId, Vec<MemoryId>>,
        incoming_boundary: Option<i64>,
        write: ProjectionWrite<'_>,
    ) -> ZResult<Vec<StateRecord>> {
        let Some(incoming_boundary) = incoming_boundary else {
            return Ok(Vec::new());
        };
        let mut boundaries = active_records
            .iter()
            .filter(|record| matches!(record.state_kind, StateRecordKind::Set))
            .map(|record| record.valid_from_ms.unwrap_or(record.observed_at_ms))
            .filter(|boundary| *boundary > incoming_boundary)
            .collect::<Vec<_>>();
        boundaries.sort_unstable();
        boundaries.dedup();

        let mut snapshots = Vec::new();
        for boundary in boundaries {
            let claims = self.scan_current_claims_for_slot_ids(
                scope,
                affected_slot_ids,
                usize::MAX,
                Some(boundary),
            )?;
            let set_states =
                state_records_from_claims(scope, &claims, rule_ids_by_claim, write.time_ms)
                    .into_iter()
                    .filter(|record| matches!(record.state_kind, StateRecordKind::Set));
            snapshots.extend(set_states.filter_map(|snapshot| {
                future_set_snapshot(snapshot, boundary, active_records, write)
            }));
        }
        Ok(snapshots)
    }
}

/// The set state as it stands from `boundary`, unless the version active then already says the
/// same; it holds until that version would have ended.
fn future_set_snapshot(
    mut snapshot: StateRecord,
    boundary: i64,
    active_records: &[StateRecord],
    write: ProjectionWrite<'_>,
) -> Option<StateRecord> {
    let existing = active_records.iter().find(|record| {
        record.state_key == snapshot.state_key && state_record_valid_at(record, Some(boundary))
    });
    if existing.is_some_and(|record| state_records_semantically_equal(&snapshot, record)) {
        return None;
    }
    snapshot.valid_from_ms = Some(boundary);
    snapshot.valid_to_ms = existing.and_then(|record| record.valid_to_ms);
    snapshot.projected_at_ms = write.time_ms;
    snapshot.recorded_at_ms = write.time_ms;
    snapshot.superseded_at_ms = None;
    snapshot.id = versioned_projection_id(
        &snapshot.state_key,
        write.key,
        &[
            snapshot.state_kind.as_str(),
            &snapshot.members.join("\u{0}"),
            &snapshot.claim_ids.join("\u{0}"),
            &boundary.to_string(),
        ],
    );
    Some(snapshot)
}

pub(super) fn reconcile_state_record_intervals(
    records: &mut Vec<StateRecord>,
    active_records: Vec<StateRecord>,
    write: ProjectionWrite<'_>,
    incoming_boundary: Option<i64>,
) -> (Vec<StateRecord>, Vec<StateRecord>) {
    let unchanged_record_ids = active_records
        .iter()
        .filter(|active| {
            records
                .iter()
                .any(|record| state_records_semantically_equal(record, active))
        })
        .map(|record| record.id.clone())
        .collect::<BTreeSet<_>>();
    records.retain(|record| {
        !active_records
            .iter()
            .any(|active| state_records_semantically_equal(record, active))
    });
    for record in records.iter_mut() {
        close_before_next_active(record, &active_records);
    }
    let mut historical_revisions = Vec::new();
    let mut superseded_records = Vec::new();
    for stale in active_records {
        if unchanged_record_ids.contains(&stale.id) {
            continue;
        }
        let stale_from = state_record_start_ms(&stale);
        let Some(boundary) = stale_boundary(&stale, records, incoming_boundary) else {
            continue;
        };
        if stale_from < boundary {
            historical_revisions.push(historical_revision(&stale, boundary, write));
        }
        let mut superseded = stale;
        superseded.status = MemoryStatus::Superseded;
        superseded.projected_at_ms = write.time_ms;
        superseded.superseded_at_ms = Some(write.time_ms);
        superseded_records.push(superseded);
    }
    (historical_revisions, superseded_records)
}

fn state_record_start_ms(record: &StateRecord) -> i64 {
    record.valid_from_ms.unwrap_or(record.observed_at_ms)
}

/// Ends a new version where the next later active version of its slot facet begins.
fn close_before_next_active(record: &mut StateRecord, active_records: &[StateRecord]) {
    let record_from = state_record_start_ms(record);
    let next_boundary = active_records
        .iter()
        .filter(|active| same_slot_facet(active, record))
        .map(state_record_start_ms)
        .filter(|active_from| *active_from > record_from)
        .min();
    if let Some(next_boundary) = next_boundary {
        record.valid_to_ms = Some(
            record
                .valid_to_ms
                .map_or(next_boundary, |to| to.min(next_boundary)),
        );
    }
}

/// Where a stale active version stops holding: the earliest new version of its slot facet
/// inside its interval, else the incoming boundary when that falls inside it.
fn stale_boundary(
    stale: &StateRecord,
    records: &[StateRecord],
    incoming_boundary: Option<i64>,
) -> Option<i64> {
    let stale_from = state_record_start_ms(stale);
    let inside = |boundary: &i64| {
        *boundary >= stale_from && stale.valid_to_ms.is_none_or(|to| *boundary < to)
    };
    records
        .iter()
        .filter(|record| same_slot_facet(record, stale))
        .map(state_record_start_ms)
        .filter(inside)
        .min()
        .or_else(|| incoming_boundary.filter(inside))
}

/// The stale version's part before `boundary`, kept as an active historical version.
fn historical_revision(
    stale: &StateRecord,
    boundary: i64,
    write: ProjectionWrite<'_>,
) -> StateRecord {
    let mut revision = stale.clone();
    revision.status = MemoryStatus::Active;
    revision.valid_to_ms = Some(boundary);
    revision.projected_at_ms = write.time_ms;
    revision.recorded_at_ms = write.time_ms;
    revision.superseded_at_ms = None;
    revision.id = versioned_projection_id(
        &revision.state_key,
        write.key,
        &[
            revision.state_kind.as_str(),
            revision.object_value.as_deref().unwrap_or_default(),
            &boundary.to_string(),
            &revision.claim_ids.join("\u{0}"),
        ],
    );
    revision
}

/// State records compete for validity only inside one surface facet of a slot family.
fn same_slot_facet(left: &StateRecord, right: &StateRecord) -> bool {
    left.slot_id == right.slot_id && left.slot_facet == right.slot_facet
}

pub(super) fn reconcile_dependency_trace_intervals(
    records: &mut [DependencyTraceRecord],
    active_records: Vec<DependencyTraceRecord>,
    active_claim_ids: &BTreeSet<MemoryId>,
    write: ProjectionWrite<'_>,
    incoming_boundary: Option<i64>,
) -> (Vec<DependencyTraceRecord>, Vec<DependencyTraceRecord>) {
    for record in records.iter_mut() {
        let record_from = record.valid_from_ms.unwrap_or(record.observed_at_ms);
        if let Some(next_boundary) = active_records
            .iter()
            .filter(|active| active.target_slot_id == record.target_slot_id)
            .map(|active| active.valid_from_ms.unwrap_or(active.observed_at_ms))
            .filter(|active_from| *active_from > record_from)
            .min()
        {
            record.valid_to_ms = Some(
                record
                    .valid_to_ms
                    .map_or(next_boundary, |to| to.min(next_boundary)),
            );
        }
    }

    let mut historical_revisions = Vec::new();
    let mut superseded_records = Vec::new();
    for stale in active_records {
        if active_claim_ids.contains(&stale.derived_claim_id)
            && !records
                .iter()
                .any(|record| record.trace_key == stale.trace_key)
        {
            continue;
        }
        let stale_from = stale.valid_from_ms.unwrap_or(stale.observed_at_ms);
        let boundary = records
            .iter()
            .filter(|record| record.target_slot_id == stale.target_slot_id)
            .map(|record| record.valid_from_ms.unwrap_or(record.observed_at_ms))
            .filter(|boundary| {
                *boundary >= stale_from && stale.valid_to_ms.is_none_or(|to| *boundary < to)
            })
            .min()
            .or_else(|| {
                incoming_boundary.filter(|boundary| {
                    *boundary >= stale_from && stale.valid_to_ms.is_none_or(|to| *boundary < to)
                })
            });
        let Some(boundary) = boundary else {
            continue;
        };
        if stale_from < boundary {
            let mut revision = stale.clone();
            revision.status = MemoryStatus::Active;
            revision.valid_to_ms = Some(boundary);
            revision.projected_at_ms = write.time_ms;
            revision.recorded_at_ms = write.time_ms;
            revision.superseded_at_ms = None;
            revision.id = versioned_projection_id(
                &revision.trace_key,
                write.key,
                &[
                    &revision.rule_id,
                    &revision.derived_claim_id,
                    &boundary.to_string(),
                ],
            );
            historical_revisions.push(revision);
        }
        let mut superseded = stale;
        superseded.status = MemoryStatus::Superseded;
        superseded.projected_at_ms = write.time_ms;
        superseded.superseded_at_ms = Some(write.time_ms);
        superseded_records.push(superseded);
    }
    (historical_revisions, superseded_records)
}

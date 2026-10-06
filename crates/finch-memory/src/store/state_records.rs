use super::*;
use std::borrow::Borrow;

pub(super) fn state_records_from_claims(
    scope: &MemoryScope,
    claims: &[ClaimRecord],
    rule_ids_by_claim: &BTreeMap<MemoryId, Vec<MemoryId>>,
    projected_at_ms: i64,
) -> Vec<StateRecord> {
    let mut set_states = project_set_states(scope, claims);
    let scalar_claims = current_scalar_lifecycle_claims(claims);
    reconcile_set_states_with_slot_lifecycle(&scalar_claims, &mut set_states);
    let mut records = set_states
        .into_iter()
        .map(|state| state_record_from_set_state(state, projected_at_ms))
        .collect::<Vec<_>>();
    records.extend(
        scalar_claims
            .into_iter()
            .filter(|claim| !is_projectable_set_member(claim))
            .map(|claim| state_record_from_claim(claim, rule_ids_by_claim, projected_at_ms)),
    );
    records
}

pub(super) fn sorted_unique_ids<'a>(ids: impl IntoIterator<Item = &'a MemoryId>) -> Vec<MemoryId> {
    ids.into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .cloned()
        .collect()
}

pub(super) fn current_scalar_lifecycle_claims(claims: &[ClaimRecord]) -> Vec<ClaimRecord> {
    let mut by_slot = BTreeMap::<String, &ClaimRecord>::new();
    for claim in claims {
        if is_projectable_set_member(claim) {
            continue;
        }
        by_slot
            .entry(claim_facet_lifecycle_key(claim))
            .and_modify(|current| {
                if claim_is_newer(claim, current) {
                    *current = claim;
                }
            })
            .or_insert(claim);
    }
    let mut current = by_slot.into_values().cloned().collect::<Vec<_>>();
    current.sort_by(|a, b| {
        claim_facet_lifecycle_key(a)
            .cmp(&claim_facet_lifecycle_key(b))
            .then_with(|| a.id.cmp(&b.id))
    });
    current
}

pub(super) fn slot_records_from_projection(
    scope: &MemoryScope,
    claims: &[ClaimRecord],
    rules: &[RuleRecord],
    applications: &[ResolvedRuleApplication],
    projected_at_ms: i64,
    _valid_at_ms: Option<i64>,
) -> Vec<CanonicalSlotRecord> {
    let mut by_id = BTreeMap::<MemoryId, CanonicalSlotRecord>::new();
    for claim in claims {
        add_claim_slot_source(&mut by_id, scope, claim, projected_at_ms);
    }
    for rule in rules {
        add_rule_slot_sources(&mut by_id, scope, rule, projected_at_ms);
    }
    for application in applications {
        let Some((slot_id, _)) = claim_slot_identity(&application.claim) else {
            continue;
        };
        if let Some(record) = by_id.get_mut(&slot_id) {
            record.source_rule_ids.push(application.rule_id.clone());
        }
    }
    let mut records = by_id.into_values().collect::<Vec<_>>();
    records.iter_mut().for_each(dedup_slot_sources);
    records
}

/// The slot a claim projects into and its canonical keys; `None` for a claim without both a
/// subject and a predicate.
fn claim_slot_identity(claim: &ClaimRecord) -> Option<(MemoryId, CanonicalSlot)> {
    let slot = claim_slot(claim)?;
    let slot_id = claim.slot_id.clone().unwrap_or_else(|| {
        canonical_slot_id(
            claim.subject_entity_id.as_deref(),
            &slot.subject_key,
            &slot.predicate_key,
        )
    });
    Some((slot_id, slot))
}

fn add_claim_slot_source(
    by_id: &mut BTreeMap<MemoryId, CanonicalSlotRecord>,
    scope: &MemoryScope,
    claim: &ClaimRecord,
    projected_at_ms: i64,
) {
    let Some((slot_id, slot)) = claim_slot_identity(claim) else {
        return;
    };
    let endpoint = SlotEndpoint {
        subject_key: &slot.subject_key,
        predicate_key: &slot.predicate_key,
        subject_entity_id: claim.subject_entity_id.as_ref(),
        subject: claim.subject.as_ref(),
        predicate: claim.predicate.as_ref(),
        visibility: claim.visibility,
        policy_tags: &claim.policy_tags,
        observed_at_ms: claim.observed_at_ms,
    };
    let record = by_id
        .entry(slot_id.clone())
        .or_insert_with(|| new_slot_identity(scope, slot_id, &endpoint, projected_at_ms));
    record.source_claim_ids.push(claim.id.clone());
    record.observed_at_ms = record.observed_at_ms.min(claim.observed_at_ms);
}

fn add_rule_slot_sources(
    by_id: &mut BTreeMap<MemoryId, CanonicalSlotRecord>,
    scope: &MemoryScope,
    rule: &RuleRecord,
    projected_at_ms: i64,
) {
    let observed_at_ms = rule.valid_from_ms.unwrap_or(projected_at_ms);
    let trigger = SlotEndpoint {
        subject_key: &rule.trigger_subject_key,
        predicate_key: &rule.trigger_predicate_key,
        subject_entity_id: rule.trigger_subject_entity_id.as_ref(),
        subject: rule.trigger_subject.as_ref(),
        predicate: rule.trigger_predicate.as_ref(),
        visibility: rule.visibility,
        policy_tags: &rule.policy_tags,
        observed_at_ms,
    };
    let target = SlotEndpoint {
        subject_key: &rule.target_subject_key,
        predicate_key: &rule.target_predicate_key,
        subject_entity_id: rule.target_subject_entity_id.as_ref(),
        subject: rule.target_subject.as_ref(),
        predicate: rule.target_predicate.as_ref(),
        ..trigger
    };
    for (slot_id, endpoint) in [
        (rule.trigger_slot_id.as_ref(), trigger),
        (rule.target_slot_id.as_ref(), target),
    ] {
        let Some(slot_id) = slot_id else {
            continue;
        };
        let record = by_id.entry(slot_id.clone()).or_insert_with(|| {
            new_slot_identity(scope, slot_id.clone(), &endpoint, projected_at_ms)
        });
        record.source_rule_ids.push(rule.id.clone());
        record.observed_at_ms = record.observed_at_ms.min(observed_at_ms);
    }
}

/// Identity and provenance of one slot as the claim or rule endpoint that names it sees it.
#[derive(Clone, Copy)]
struct SlotEndpoint<'a> {
    subject_key: &'a str,
    predicate_key: &'a str,
    subject_entity_id: Option<&'a MemoryId>,
    subject: Option<&'a String>,
    predicate: Option<&'a String>,
    visibility: Visibility,
    policy_tags: &'a [String],
    observed_at_ms: i64,
}

fn new_slot_identity(
    scope: &MemoryScope,
    slot_id: MemoryId,
    endpoint: &SlotEndpoint<'_>,
    projected_at_ms: i64,
) -> CanonicalSlotRecord {
    CanonicalSlotRecord {
        id: slot_id.clone(),
        slot_key: slot_id,
        scope: scope.clone(),
        status: MemoryStatus::Active,
        visibility: endpoint.visibility,
        policy_tags: endpoint.policy_tags.to_vec(),
        subject_key: endpoint.subject_key.to_owned(),
        predicate_key: endpoint.predicate_key.to_owned(),
        subject_entity_id: endpoint.subject_entity_id.cloned(),
        subject: endpoint.subject.cloned(),
        predicate: endpoint.predicate.cloned(),
        source_claim_ids: Vec::new(),
        source_rule_ids: Vec::new(),
        source_entity_ids: endpoint.subject_entity_id.cloned().into_iter().collect(),
        observed_at_ms: endpoint.observed_at_ms,
        valid_from_ms: None,
        valid_to_ms: None,
        projected_at_ms,
        recorded_at_ms: projected_at_ms,
        superseded_at_ms: None,
    }
}

fn dedup_slot_sources(record: &mut CanonicalSlotRecord) {
    record.source_claim_ids.sort();
    record.source_claim_ids.dedup();
    record.source_rule_ids.sort();
    record.source_rule_ids.dedup();
    record.source_entity_ids.sort();
    record.source_entity_ids.dedup();
}

pub(super) fn merge_slot_identity_records(
    records: &mut Vec<CanonicalSlotRecord>,
    existing: Vec<CanonicalSlotRecord>,
) {
    let mut by_slot = records
        .drain(..)
        .map(|record| (record.slot_key.clone(), record))
        .collect::<BTreeMap<_, _>>();
    for prior in existing {
        match by_slot.get_mut(&prior.slot_key) {
            Some(current) => absorb_prior_slot_identity(current, prior),
            None => {
                by_slot.insert(prior.slot_key.clone(), prior);
            }
        }
    }
    for record in by_slot.values_mut() {
        record.id = record.slot_key.clone();
        record.status = MemoryStatus::Active;
        record.valid_from_ms = None;
        record.valid_to_ms = None;
        record.superseded_at_ms = None;
        dedup_slot_sources(record);
    }
    *records = by_slot.into_values().collect();
}

/// Folds a prior version of a slot identity into the current one: sources accumulate, the
/// earliest times win, and labels the current version lacks come from the prior one.
fn absorb_prior_slot_identity(current: &mut CanonicalSlotRecord, prior: CanonicalSlotRecord) {
    current.source_claim_ids.extend(prior.source_claim_ids);
    current.source_rule_ids.extend(prior.source_rule_ids);
    current.source_entity_ids.extend(prior.source_entity_ids);
    current.observed_at_ms = current.observed_at_ms.min(prior.observed_at_ms);
    current.recorded_at_ms = current.recorded_at_ms.min(prior.recorded_at_ms);
    if current.subject_entity_id.is_none() {
        current.subject_entity_id = prior.subject_entity_id;
    }
    if current.subject.is_none() {
        current.subject = prior.subject;
    }
    if current.predicate.is_none() {
        current.predicate = prior.predicate;
    }
}

pub(super) fn prepare_slot_identity_revisions(
    records: &mut [CanonicalSlotRecord],
    write: ProjectionWrite<'_>,
) {
    for record in records {
        record.id = versioned_projection_id(
            &record.slot_key,
            write.key,
            &[
                &record.source_claim_ids.join("\u{0}"),
                &record.source_rule_ids.join("\u{0}"),
                &record.source_entity_ids.join("\u{0}"),
            ],
        );
        record.projected_at_ms = write.time_ms;
        record.recorded_at_ms = write.time_ms;
        record.superseded_at_ms = None;
    }
}

pub(super) fn state_record_from_claim(
    claim: ClaimRecord,
    rule_ids_by_claim: &BTreeMap<MemoryId, Vec<MemoryId>>,
    projected_at_ms: i64,
) -> StateRecord {
    let state_kind = claim_state_record_kind(&claim);
    let state_key = scalar_state_record_id(&claim, state_kind);
    let hides_value = matches!(
        state_kind,
        StateRecordKind::Tombstone | StateRecordKind::Unsupported
    );
    let state_text = if hides_value {
        serde_json::json!({
            "subject": &claim.subject,
            "predicate": &claim.predicate,
            "support_state": match state_kind {
                StateRecordKind::Tombstone => "deleted",
                _ => "unsupported",
            },
        })
        .to_string()
    } else {
        claim.claim_text
    };
    let rule_ids = rule_ids_by_claim
        .get(&claim.id)
        .cloned()
        .unwrap_or_default();
    StateRecord {
        id: state_key.clone(),
        state_key,
        scope: claim.scope,
        status: MemoryStatus::Active,
        visibility: claim.visibility,
        policy_tags: claim.policy_tags,
        state_kind,
        claim_kind: claim.claim_kind,
        subject: claim.subject,
        predicate: claim.predicate,
        object_value: if hides_value {
            None
        } else {
            claim.object_value
        },
        subject_entity_id: claim.subject_entity_id,
        slot_id: claim.slot_id,
        slot_facet: claim.slot_facet,
        state_text,
        members: Vec::new(),
        claim_ids: vec![claim.id],
        correction_ids: claim.correction_ids,
        rule_ids,
        source_span_ids: claim.source_span_ids,
        source_episode_ids: claim.source_episode_ids,
        source_sequence_no: claim.source_sequence_no,
        observed_at_ms: claim.observed_at_ms,
        valid_from_ms: claim.valid_from_ms.or(Some(claim.observed_at_ms)),
        valid_to_ms: claim.valid_to_ms,
        projected_at_ms,
        recorded_at_ms: projected_at_ms,
        superseded_at_ms: None,
    }
}

pub(super) fn state_record_from_set_state(
    state: SetStateRecord,
    projected_at_ms: i64,
) -> StateRecord {
    let slot_key = state.slot_id.clone().unwrap_or_else(|| {
        let subject_key = canonical_slot_part(state.subject.as_deref().unwrap_or_default());
        let predicate_key = canonical_slot_part(state.predicate.as_deref().unwrap_or_default());
        canonical_slot_id(
            state.subject_entity_id.as_deref(),
            &subject_key,
            &predicate_key,
        )
    });
    let slot_hash =
        stable_hash_hex(&[&slot_key, &set_claim_kind_key(state.claim_kind).to_string()]);
    let state_key = format!("state_set_{slot_hash}");
    let state_text = serde_json::json!({
        "subject": &state.subject,
        "predicate": &state.predicate,
        "members": &state.members,
    })
    .to_string();
    StateRecord {
        id: state_key.clone(),
        state_key,
        scope: state.scope,
        status: MemoryStatus::Active,
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        state_kind: StateRecordKind::Set,
        claim_kind: state.claim_kind,
        subject: state.subject,
        predicate: state.predicate,
        object_value: None,
        subject_entity_id: state.subject_entity_id,
        slot_id: state.slot_id,
        slot_facet: None,
        state_text,
        members: state.members,
        claim_ids: state.claim_ids,
        correction_ids: Vec::new(),
        rule_ids: Vec::new(),
        source_span_ids: state.source_span_ids,
        source_episode_ids: state.source_episode_ids,
        source_sequence_no: None,
        observed_at_ms: state.observed_at_ms,
        valid_from_ms: state.valid_from_ms,
        valid_to_ms: state.valid_to_ms,
        projected_at_ms,
        recorded_at_ms: projected_at_ms,
        superseded_at_ms: None,
    }
}

pub(super) fn claim_state_record_kind(claim: &ClaimRecord) -> StateRecordKind {
    if matches!(
        claim.status,
        MemoryStatus::Retracted | MemoryStatus::Tombstoned | MemoryStatus::Redacted
    ) {
        return StateRecordKind::Tombstone;
    }
    if !matches!(claim.status, MemoryStatus::Active)
        || matches!(
            claim.polarity,
            ClaimPolarity::Uncertain | ClaimPolarity::Negative
        )
    {
        return StateRecordKind::Unsupported;
    }
    if !crate::is_direct_state_claim(claim) {
        return StateRecordKind::Derived;
    }
    if matches!(claim.claim_kind, ClaimKind::Constraint) {
        return StateRecordKind::Rule;
    }
    StateRecordKind::Current
}

pub(super) fn scalar_state_record_id(claim: &ClaimRecord, state_kind: StateRecordKind) -> MemoryId {
    let slot_key = claim.slot_id.clone().unwrap_or_else(|| {
        let subject_key = canonical_slot_part(claim.subject.as_deref().unwrap_or_default());
        let predicate_key = canonical_slot_part(claim.predicate.as_deref().unwrap_or_default());
        canonical_slot_id(
            claim.subject_entity_id.as_deref(),
            &subject_key,
            &predicate_key,
        )
    });
    let slot_hash = match claim.slot_facet.as_deref() {
        Some(facet) => stable_hash_hex(&[&slot_key, state_kind.as_str(), facet]),
        None => stable_hash_hex(&[&slot_key, state_kind.as_str()]),
    };
    match state_kind {
        StateRecordKind::Tombstone => format!("state_tombstone_{}", stable_hash_hex(&[&claim.id])),
        StateRecordKind::Unsupported => format!("state_unsupported_{slot_hash}"),
        StateRecordKind::Derived => format!("state_derived_{slot_hash}"),
        StateRecordKind::Rule => format!("state_rule_{slot_hash}"),
        StateRecordKind::Current | StateRecordKind::Set => format!("state_current_{slot_hash}"),
    }
}

pub(super) fn dependency_trace_id(
    rule_id: &str,
    trigger_claim_id: &str,
    derived_claim_id: &str,
) -> MemoryId {
    format!(
        "dependency_trace_{}",
        stable_hash_hex(&[rule_id, trigger_claim_id, derived_claim_id])
    )
}

pub(super) fn dependency_trace_from_application(
    scope: &MemoryScope,
    application: &ResolvedRuleApplication,
    projected_at_ms: i64,
) -> DependencyTraceRecord {
    let claim = &application.claim;
    DependencyTraceRecord {
        id: application.trace_id.clone(),
        trace_key: application.trace_id.clone(),
        scope: scope.clone(),
        status: MemoryStatus::Active,
        visibility: claim.visibility,
        policy_tags: claim.policy_tags.clone(),
        rule_id: application.rule_id.clone(),
        trigger_claim_id: application.trigger_claim_id.clone(),
        prior_trigger_claim_id: application.prior_trigger_claim_id.clone(),
        derived_claim_id: claim.id.clone(),
        target_subject_key: application.target_subject_key.clone(),
        target_predicate_key: application.target_predicate_key.clone(),
        target_subject_entity_id: claim.subject_entity_id.clone(),
        target_slot_id: claim.slot_id.clone(),
        target_subject: claim.subject.clone(),
        target_predicate: claim.predicate.clone(),
        state_kind: claim_state_record_kind(claim),
        hop: application.hop,
        parent_trace_ids: application.parent_trace_ids.clone(),
        source_span_ids: claim.source_span_ids.clone(),
        source_episode_ids: claim.source_episode_ids.clone(),
        observed_at_ms: claim.observed_at_ms,
        valid_from_ms: claim.valid_from_ms,
        valid_to_ms: claim.valid_to_ms,
        projected_at_ms,
        recorded_at_ms: projected_at_ms,
        superseded_at_ms: None,
    }
}

pub(super) fn state_record_priority(kind: StateRecordKind) -> u8 {
    match kind {
        StateRecordKind::Tombstone => 0,
        StateRecordKind::Unsupported => 1,
        StateRecordKind::Derived => 2,
        StateRecordKind::Current => 3,
        StateRecordKind::Set => 4,
        StateRecordKind::Rule => 5,
    }
}

pub(super) fn transaction_visible_at(
    recorded_at_ms: i64,
    superseded_at_ms: Option<i64>,
    transaction_at_ms: Option<i64>,
) -> bool {
    match transaction_at_ms {
        Some(at) => recorded_at_ms <= at && superseded_at_ms.is_none_or(|to| at < to),
        None => superseded_at_ms.is_none(),
    }
}

pub(super) fn state_record_valid_at(record: &StateRecord, valid_at_ms: Option<i64>) -> bool {
    match valid_at_ms {
        Some(at) => {
            record.valid_from_ms.is_none_or(|from| from <= at)
                && record.valid_to_ms.is_none_or(|to| at < to)
        }
        None => record.valid_to_ms.is_none(),
    }
}

pub(super) fn state_record_sets_semantically_equal(
    left: &[StateRecord],
    right: &[StateRecord],
) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter().all(|record| {
        right
            .iter()
            .any(|existing| state_records_semantically_equal(record, existing))
    })
}

pub(super) fn state_records_semantically_equal(left: &StateRecord, right: &StateRecord) -> bool {
    left.state_key == right.state_key
        && left.status == right.status
        && left.state_kind == right.state_kind
        && left.claim_kind == right.claim_kind
        && left.subject == right.subject
        && left.predicate == right.predicate
        && left.object_value == right.object_value
        && left.subject_entity_id == right.subject_entity_id
        && left.slot_id == right.slot_id
        && left.slot_facet == right.slot_facet
        && left.members == right.members
        && left.claim_ids == right.claim_ids
        && left.rule_ids == right.rule_ids
        && left.source_span_ids == right.source_span_ids
        && left.source_episode_ids == right.source_episode_ids
        && left.observed_at_ms == right.observed_at_ms
        && left.valid_from_ms == right.valid_from_ms
        && left.valid_to_ms == right.valid_to_ms
}

/// One projection write: its transaction time and a deterministic key naming the write (the
/// batch or record that caused it). Version ids hash the key, never the clock, so replaying the
/// same writes yields the same ids.
#[derive(Clone, Copy)]
pub(crate) struct ProjectionWrite<'a> {
    pub time_ms: i64,
    pub key: &'a str,
}

pub(super) fn versioned_projection_id(key: &str, write_key: &str, parts: &[&str]) -> MemoryId {
    let mut hash_parts = Vec::with_capacity(parts.len() + 2);
    hash_parts.push(key);
    hash_parts.push(write_key);
    hash_parts.extend_from_slice(parts);
    format!("{}_v_{}", key, stable_hash_hex(&hash_parts))
}

pub(super) fn system_time_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(i64::MAX as u128) as i64)
        .unwrap_or(0)
}

pub(super) fn merge_trace_rule_ids(records: &mut [StateRecord], traces: &[DependencyTraceRecord]) {
    if records.is_empty() || traces.is_empty() {
        return;
    }
    let mut rule_ids_by_claim = BTreeMap::<MemoryId, Vec<MemoryId>>::new();
    for trace in traces {
        rule_ids_by_claim
            .entry(trace.derived_claim_id.clone())
            .or_default()
            .push(trace.rule_id.clone());
    }
    for record in records {
        for claim_id in &record.claim_ids {
            if let Some(rule_ids) = rule_ids_by_claim.get(claim_id) {
                record.rule_ids.extend(rule_ids.iter().cloned());
            }
        }
        record.rule_ids.sort();
        record.rule_ids.dedup();
    }
}

/// A state version in `scope` recorded, and not yet superseded, at the transaction time.
pub(super) fn state_record_recorded_at(
    record: &StateRecord,
    scope: &MemoryScope,
    temporal: BiTemporalQuery,
) -> bool {
    record.scope.matches_filter(scope)
        && transaction_visible_at(
            record.recorded_at_ms,
            record.superseded_at_ms,
            temporal.transaction_at_ms,
        )
}

pub(super) fn state_record_visible_at(
    record: &StateRecord,
    scope: &MemoryScope,
    temporal: BiTemporalQuery,
) -> bool {
    state_record_recorded_at(record, scope, temporal)
        && temporal.valid_at_ms.is_none_or(|at| {
            record.valid_from_ms.is_none_or(|from| from <= at)
                && record.valid_to_ms.is_none_or(|to| at < to)
        })
        && (temporal.valid_at_ms.is_some() || record.valid_to_ms.is_none())
}

/// The query-side signals state records are scored against.
pub(super) struct StateSelectionQuery<'a> {
    pub query_text: &'a str,
    pub evidence_hits: &'a [SpanSearchHit],
    pub entities: &'a [EntityRecord],
    pub preferred_slot_ids: &'a BTreeSet<MemoryId>,
    pub evidence_slot_ids: &'a BTreeSet<MemoryId>,
}

pub(super) fn select_state_records(
    records: Vec<StateRecord>,
    query: &StateSelectionQuery<'_>,
    limit: usize,
) -> Vec<StateRecord> {
    if limit == 0 {
        return Vec::new();
    }
    let assessments = assess_state_records(&records, query);
    let mut scored = records
        .into_iter()
        .map(|record| {
            let score = assessments
                .get(&record.id)
                .map_or(0, |assessment| assessment.score);
            (score, record)
        })
        .collect::<Vec<_>>();
    scored.sort_by(|(a_score, a), (b_score, b)| {
        b_score
            .cmp(a_score)
            .then_with(|| {
                state_record_priority(a.state_kind).cmp(&state_record_priority(b.state_kind))
            })
            .then_with(|| b.observed_at_ms.cmp(&a.observed_at_ms))
            .then_with(|| a.id.cmp(&b.id))
    });
    scored
        .into_iter()
        .filter(|(score, _)| *score > 0)
        .map(|(_, record)| record)
        .take(limit)
        .collect()
}

#[derive(Debug, Clone)]
pub(super) struct StateSelectionAssessment {
    pub signals: StateSelectionSignals,
    pub score: u8,
}

pub(super) fn mentioned_entity_ids(
    query_text: &str,
    entities: &[EntityRecord],
) -> BTreeSet<MemoryId> {
    let normalized_query = canonical_slot_part(query_text);
    entities
        .iter()
        .filter(|entity| {
            std::iter::once(entity.canonical_name.as_str())
                .chain(entity.aliases.iter().map(String::as_str))
                .any(|alias| normalized_phrase_present(&normalized_query, alias))
        })
        .map(|entity| entity.id.clone())
        .collect()
}

pub(super) fn assess_state_records(
    records: &[StateRecord],
    query: &StateSelectionQuery<'_>,
) -> BTreeMap<MemoryId, StateSelectionAssessment> {
    let signals = StateSelectionContext::new(records, query);
    records
        .iter()
        .map(|record| (record.id.clone(), signals.assess(record)))
        .collect()
}

/// Query-side facts every state of one selection is scored against.
struct StateSelectionContext<'a> {
    preferred_slot_ids: &'a BTreeSet<MemoryId>,
    evidence_slot_ids: &'a BTreeSet<MemoryId>,
    evidence_span_ids: BTreeSet<&'a str>,
    normalized_query: String,
    mentioned_entity_ids: BTreeSet<MemoryId>,
    /// Subjects of the states the evidence spans prove.
    evidence_subjects: BTreeSet<String>,
}

impl<'a> StateSelectionContext<'a> {
    fn new(records: &[StateRecord], query: &StateSelectionQuery<'a>) -> Self {
        let evidence_span_ids = query
            .evidence_hits
            .iter()
            .map(|hit| hit.span.id.as_str())
            .collect::<BTreeSet<_>>();
        let evidence_subjects = records
            .iter()
            .filter(|record| cites_any_span(record, &evidence_span_ids))
            .filter_map(state_subject_key)
            .collect();
        Self {
            normalized_query: canonical_slot_part(query.query_text),
            mentioned_entity_ids: mentioned_entity_ids(query.query_text, query.entities),
            preferred_slot_ids: query.preferred_slot_ids,
            evidence_slot_ids: query.evidence_slot_ids,
            evidence_span_ids,
            evidence_subjects,
        }
    }

    fn assess(&self, record: &StateRecord) -> StateSelectionAssessment {
        let entity_match = record
            .subject_entity_id
            .as_ref()
            .is_some_and(|entity_id| self.mentioned_entity_ids.contains(entity_id));
        let subject_match = phrase_named(&self.normalized_query, record.subject.as_deref());
        let linked_subject = state_subject_key(record)
            .is_some_and(|subject| self.evidence_subjects.contains(&subject));
        let signals = StateSelectionSignals {
            preferred_slot: state_on_slot_in(record, self.preferred_slot_ids),
            evidence: cites_any_span(record, &self.evidence_span_ids)
                || state_on_slot_in(record, self.evidence_slot_ids),
            anchored_slot: phrase_named(&self.normalized_query, record.predicate.as_deref())
                && (entity_match || subject_match || linked_subject),
            set_entity: matches!(record.state_kind, StateRecordKind::Set)
                && (entity_match || subject_match),
        };
        let score = u8::from(signals.preferred_slot) * 8
            + u8::from(signals.evidence) * 6
            + u8::from(signals.anchored_slot) * 4
            + u8::from(signals.set_entity) * 2;
        StateSelectionAssessment { signals, score }
    }
}

fn cites_any_span(record: &StateRecord, span_ids: &BTreeSet<&str>) -> bool {
    record
        .source_span_ids
        .iter()
        .any(|span_id| span_ids.contains(span_id.as_str()))
}

/// The state's subject entity, or its canonical subject surface.
fn state_subject_key(record: &StateRecord) -> Option<String> {
    record
        .subject_entity_id
        .clone()
        .or_else(|| record.subject.as_deref().map(canonical_slot_part))
}

fn state_on_slot_in(record: &StateRecord, slot_ids: &BTreeSet<MemoryId>) -> bool {
    record
        .slot_id
        .as_ref()
        .is_some_and(|slot_id| slot_ids.contains(slot_id))
}

fn phrase_named(normalized_query: &str, surface: Option<&str>) -> bool {
    surface.is_some_and(|surface| normalized_phrase_present(normalized_query, surface))
}

pub(super) fn normalized_phrase_present(normalized_query: &str, phrase: &str) -> bool {
    let phrase = canonical_slot_part(phrase);
    if phrase.is_empty() {
        return false;
    }
    let query = format!(" {normalized_query} ");
    query.contains(&format!(" {phrase} "))
        || (!phrase.is_ascii() && normalized_query.contains(&phrase))
}

/// How well a record's fields match the query: 4 for each field the whole query contains as a
/// phrase, and 1 for each field term among `query_terms`.
pub(super) fn current_state_query_overlap(
    query_text: &str,
    query_terms: &BTreeSet<String>,
    fields: &[&str],
) -> usize {
    if query_terms.is_empty() {
        return 0;
    }
    let mut score = 0usize;
    for field in fields {
        if field.trim().is_empty() {
            continue;
        }
        if normalized_phrase_present(&canonical_slot_part(query_text), field) {
            score = score.saturating_add(4);
        }
        for term in lexical_terms(field)
            .into_iter()
            .filter(|term| usable_query_term(term))
        {
            if query_terms.contains(term.as_str()) {
                score = score.saturating_add(1);
            }
        }
    }
    score
}

/// Lexical fields of a state; lifecycle states that hide their value expose only its slot.
pub(crate) fn state_lexical_fields(record: &StateRecord) -> Vec<&str> {
    let mut fields = vec![
        record.subject.as_deref().unwrap_or(""),
        record.predicate.as_deref().unwrap_or(""),
    ];
    if !matches!(
        record.state_kind,
        StateRecordKind::Tombstone | StateRecordKind::Unsupported
    ) {
        fields.push(record.object_value.as_deref().unwrap_or(""));
        fields.push(record.state_text.as_str());
    }
    fields
}

/// The usable terms of the record's lexical fields: a record shares one with every query it
/// overlaps by term.
pub(crate) fn state_lexical_terms(record: &StateRecord) -> Vec<String> {
    state_lexical_fields(record)
        .into_iter()
        .flat_map(lexical_terms)
        .filter(|term| usable_query_term(term))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

pub(super) fn lexical_query_terms(query_text: &str) -> BTreeSet<String> {
    lexical_terms(query_text)
        .into_iter()
        .filter(|term| usable_query_term(term))
        .collect()
}

fn usable_query_term(term: &str) -> bool {
    !term.is_ascii() || term.chars().count() >= 2
}

pub(super) fn state_records_to_projection(records: Vec<StateRecord>) -> AnswerReadyStateProjection {
    let support = support_contract_for_state_records(&records);
    let mut claims = Vec::new();
    let mut set_states = Vec::new();
    for record in records {
        if matches!(record.state_kind, StateRecordKind::Set) {
            set_states.push(SetStateRecord {
                id: record.id,
                scope: record.scope,
                status: record.status,
                claim_kind: record.claim_kind,
                subject: record.subject,
                predicate: record.predicate,
                subject_entity_id: record.subject_entity_id,
                slot_id: record.slot_id,
                members: record.members,
                claim_ids: record.claim_ids,
                source_span_ids: record.source_span_ids,
                source_episode_ids: record.source_episode_ids,
                observed_at_ms: record.observed_at_ms,
                valid_from_ms: record.valid_from_ms,
                valid_to_ms: record.valid_to_ms,
            });
        } else {
            claims.push(claim_from_state_record(record));
        }
    }
    let proof_span_ids = proof_span_ids_for_claims(&claims);
    AnswerReadyStateProjection {
        claims,
        set_states,
        slot_histories: Vec::new(),
        rules: Vec::new(),
        rule_outcomes: Vec::new(),
        corrections: Vec::new(),
        entities: Vec::new(),
        proof_span_ids,
        support,
    }
}

pub(super) fn state_records_to_slot_histories(records: Vec<StateRecord>) -> Vec<SlotHistoryRecord> {
    let mut by_slot = BTreeMap::<MemoryId, Vec<StateRecord>>::new();
    for record in records {
        if matches!(record.state_kind, StateRecordKind::Rule)
            || record.source_span_ids.is_empty() && record.source_episode_ids.is_empty()
        {
            continue;
        }
        if let Some(slot_id) = record.slot_id.clone() {
            by_slot.entry(slot_id).or_default().push(record);
        }
    }
    by_slot
        .into_iter()
        .filter_map(|(slot_id, records)| slot_history(slot_id, records))
        .collect()
}

/// A slot's distinct state versions in validity order; `None` for fewer than two.
fn slot_history(slot_id: MemoryId, mut records: Vec<StateRecord>) -> Option<SlotHistoryRecord> {
    records.sort_by(history_version_order);
    records.dedup_by(|a, b| {
        a.state_kind == b.state_kind
            && a.object_value == b.object_value
            && a.members == b.members
            && a.valid_from_ms == b.valid_from_ms
            && a.valid_to_ms == b.valid_to_ms
    });
    if records.len() < 2 {
        return None;
    }
    let representative = records.last()?;
    Some(SlotHistoryRecord {
        id: format!("slot_history:{slot_id}"),
        slot_id,
        subject_entity_id: representative.subject_entity_id.clone(),
        subject: representative.subject.clone(),
        predicate: representative.predicate.clone(),
        versions: records.into_iter().map(slot_history_version).collect(),
    })
}

fn history_version_order(a: &StateRecord, b: &StateRecord) -> std::cmp::Ordering {
    let start = |record: &StateRecord| record.valid_from_ms.unwrap_or(record.observed_at_ms);
    start(a)
        .cmp(&start(b))
        .then_with(|| a.valid_to_ms.cmp(&b.valid_to_ms))
        .then_with(|| a.observed_at_ms.cmp(&b.observed_at_ms))
        .then_with(|| state_record_priority(a.state_kind).cmp(&state_record_priority(b.state_kind)))
        .then_with(|| a.id.cmp(&b.id))
}

fn slot_history_version(record: StateRecord) -> SlotHistoryVersion {
    SlotHistoryVersion {
        state_id: record.id,
        state_kind: record.state_kind,
        object_value: record.object_value,
        members: record.members,
        claim_ids: record.claim_ids,
        correction_ids: record.correction_ids,
        rule_ids: record.rule_ids,
        source_span_ids: record.source_span_ids,
        source_episode_ids: record.source_episode_ids,
        observed_at_ms: record.observed_at_ms,
        valid_from_ms: record.valid_from_ms,
        valid_to_ms: record.valid_to_ms,
    }
}

pub(super) fn support_contract_for_state_records(records: &[StateRecord]) -> AnswerSupportContract {
    let visible_records = records
        .iter()
        .filter(|record| !matches!(record.state_kind, StateRecordKind::Rule))
        .collect::<Vec<_>>();
    let mut source_claim_ids = records
        .iter()
        .filter(|record| !matches!(record.state_kind, StateRecordKind::Rule))
        .flat_map(|record| record.claim_ids.iter().cloned())
        .collect::<Vec<_>>();
    let mut source_span_ids = records
        .iter()
        .filter(|record| !matches!(record.state_kind, StateRecordKind::Rule))
        .flat_map(|record| record.source_span_ids.iter().cloned())
        .collect::<Vec<_>>();
    source_claim_ids.sort();
    source_claim_ids.dedup();
    source_span_ids.sort();
    source_span_ids.dedup();
    let mut by_slot = BTreeMap::<MemoryId, Vec<&StateRecord>>::new();
    for record in &visible_records {
        if let Some(slot_id) = record.slot_id.as_ref() {
            by_slot.entry(slot_id.clone()).or_default().push(record);
        }
    }
    let slots = by_slot
        .into_iter()
        .map(|(slot_id, records)| {
            let mut source_claim_ids = records
                .iter()
                .flat_map(|record| record.claim_ids.iter().cloned())
                .collect::<Vec<_>>();
            let mut source_span_ids = records
                .iter()
                .flat_map(|record| record.source_span_ids.iter().cloned())
                .collect::<Vec<_>>();
            source_claim_ids.sort();
            source_claim_ids.dedup();
            source_span_ids.sort();
            source_span_ids.dedup();
            let subject = records.iter().find_map(|record| record.subject.clone());
            let predicate = records.iter().find_map(|record| record.predicate.clone());
            AnswerSlotSupport {
                slot_id,
                subject,
                predicate,
                view: StateReadView::Current,
                state: support_state_for_state_records(&records),
                source_claim_ids,
                source_span_ids,
            }
        })
        .collect::<Vec<_>>();
    AnswerSupportContract {
        state: aggregate_support_state(&slots),
        source_claim_ids,
        source_span_ids,
        slots,
    }
}

pub(super) fn support_contract_for_claims<C, S>(
    claims: &[C],
    set_states: &[S],
) -> AnswerSupportContract
where
    C: Borrow<ClaimRecord>,
    S: Borrow<SetStateRecord>,
{
    let claims = claims.iter().map(Borrow::borrow).collect::<Vec<_>>();
    let set_states = set_states.iter().map(Borrow::borrow).collect::<Vec<_>>();
    let source_claim_ids = sorted_unique_ids(
        claims
            .iter()
            .map(|claim| &claim.id)
            .chain(set_states.iter().flat_map(|state| &state.claim_ids)),
    );
    let source_span_ids = sorted_unique_ids(
        claims
            .iter()
            .flat_map(|claim| &claim.source_span_ids)
            .chain(set_states.iter().flat_map(|state| &state.source_span_ids)),
    );
    let mut claims_by_slot = BTreeMap::<MemoryId, Vec<&ClaimRecord>>::new();
    for claim in claims {
        claims_by_slot
            .entry(claim_slot_lifecycle_key(claim))
            .or_default()
            .push(claim);
    }
    let mut sets_by_slot = BTreeMap::<MemoryId, Vec<&SetStateRecord>>::new();
    for set_state in set_states {
        let slot_id = set_state.slot_id.as_ref().unwrap_or(&set_state.id);
        sets_by_slot
            .entry(slot_id.clone())
            .or_default()
            .push(set_state);
    }
    let slot_ids = claims_by_slot
        .keys()
        .chain(sets_by_slot.keys())
        .collect::<BTreeSet<_>>();
    let slots = slot_ids
        .into_iter()
        .map(|slot_id| {
            let slot_claims = claims_by_slot.get(slot_id).map_or(&[][..], Vec::as_slice);
            let slot_sets = sets_by_slot.get(slot_id).map_or(&[][..], Vec::as_slice);
            slot_support_from_claims(slot_id.clone(), slot_claims, slot_sets)
        })
        .collect::<Vec<_>>();
    AnswerSupportContract {
        state: aggregate_support_state(&slots),
        source_claim_ids,
        source_span_ids,
        slots,
    }
}

fn slot_support_from_claims(
    slot_id: MemoryId,
    slot_claims: &[&ClaimRecord],
    slot_sets: &[&SetStateRecord],
) -> AnswerSlotSupport {
    let source_claim_ids = sorted_unique_ids(
        slot_claims
            .iter()
            .map(|claim| &claim.id)
            .chain(slot_sets.iter().flat_map(|state| &state.claim_ids)),
    );
    let source_span_ids = sorted_unique_ids(
        slot_claims
            .iter()
            .flat_map(|claim| &claim.source_span_ids)
            .chain(slot_sets.iter().flat_map(|state| &state.source_span_ids)),
    );
    let current_claim = newest_claim(slot_claims.iter().copied());
    // Label the slot by its identity facet when one is projected; a co-current facet bound
    // under another surface must not rename the slot.
    let label_claim = newest_claim(
        slot_claims
            .iter()
            .copied()
            .filter(|claim| claim.slot_facet.is_none()),
    )
    .or(current_claim);
    let current_set = slot_sets
        .iter()
        .copied()
        .max_by_key(|state| (state.valid_from_ms, state.observed_at_ms, &state.id));
    AnswerSlotSupport {
        slot_id,
        subject: label_claim
            .and_then(|claim| claim.subject.clone())
            .or_else(|| current_set.and_then(|state| state.subject.clone())),
        predicate: label_claim
            .and_then(|claim| claim.predicate.clone())
            .or_else(|| current_set.and_then(|state| state.predicate.clone())),
        view: StateReadView::Current,
        state: support_state_for_claim_refs(slot_claims, !slot_sets.is_empty()),
        source_claim_ids,
        source_span_ids,
    }
}

fn support_state_for_state_records(records: &[&StateRecord]) -> AnswerSupportState {
    if records.iter().any(|record| {
        !matches!(
            record.claim_kind,
            ClaimKind::Relationship | ClaimKind::Event
        ) && matches!(record.state_kind, StateRecordKind::Tombstone)
    }) {
        AnswerSupportState::Deleted
    } else if records.iter().any(|record| {
        !matches!(
            record.claim_kind,
            ClaimKind::Relationship | ClaimKind::Event
        ) && matches!(record.state_kind, StateRecordKind::Unsupported)
    }) {
        AnswerSupportState::Unsupported
    } else if records.iter().any(|record| {
        matches!(record.state_kind, StateRecordKind::Set) && !record.members.is_empty()
    }) {
        AnswerSupportState::Supported
    } else if records
        .iter()
        .any(|record| matches!(record.state_kind, StateRecordKind::Tombstone))
    {
        AnswerSupportState::Deleted
    } else if records
        .iter()
        .any(|record| matches!(record.state_kind, StateRecordKind::Unsupported))
    {
        AnswerSupportState::Unsupported
    } else if records.iter().any(|record| {
        matches!(
            record.state_kind,
            StateRecordKind::Current | StateRecordKind::Derived | StateRecordKind::Set
        )
    }) {
        AnswerSupportState::Supported
    } else {
        AnswerSupportState::NoEvidence
    }
}

pub(super) fn support_state_for_claim_refs(
    claims: &[&ClaimRecord],
    has_set_state: bool,
) -> AnswerSupportState {
    if claims.iter().any(|claim| {
        claim_invalidates_entire_set_slot(claim)
            && matches!(claim_state_record_kind(claim), StateRecordKind::Tombstone)
    }) {
        AnswerSupportState::Deleted
    } else if claims.iter().any(|claim| {
        claim_invalidates_entire_set_slot(claim)
            && matches!(claim_state_record_kind(claim), StateRecordKind::Unsupported)
    }) {
        AnswerSupportState::Unsupported
    } else if has_set_state {
        AnswerSupportState::Supported
    } else if claims.iter().any(|claim| {
        matches!(
            claim.status,
            MemoryStatus::Retracted | MemoryStatus::Tombstoned | MemoryStatus::Redacted
        )
    }) {
        AnswerSupportState::Deleted
    } else if claims.iter().any(|claim| {
        !matches!(claim.status, MemoryStatus::Active)
            || matches!(
                claim.polarity,
                ClaimPolarity::Negative | ClaimPolarity::Uncertain
            )
    }) {
        AnswerSupportState::Unsupported
    } else if !claims.is_empty() {
        AnswerSupportState::Supported
    } else {
        AnswerSupportState::NoEvidence
    }
}

pub(super) fn claim_from_state_record(record: StateRecord) -> ClaimRecord {
    let status = match record.state_kind {
        StateRecordKind::Tombstone => MemoryStatus::Tombstoned,
        _ => MemoryStatus::Active,
    };
    let polarity = match record.state_kind {
        StateRecordKind::Unsupported => ClaimPolarity::Uncertain,
        _ => ClaimPolarity::Affirmative,
    };
    let asserted_by = match record.state_kind {
        StateRecordKind::Derived => crate::DERIVED_STATE_ASSERTED_BY,
        _ => "state_projection",
    };
    ClaimRecord {
        id: record
            .claim_ids
            .first()
            .cloned()
            .unwrap_or_else(|| record.id.clone()),
        scope: record.scope,
        status,
        visibility: record.visibility,
        policy_tags: record.policy_tags,
        claim_text: record.state_text,
        subject: record.subject,
        predicate: record.predicate,
        object_value: record.object_value,
        subject_entity_id: record.subject_entity_id,
        slot_id: record.slot_id,
        slot_facet: record.slot_facet,
        claim_kind: record.claim_kind,
        polarity,
        source_span_ids: record.source_span_ids,
        source_episode_ids: record.source_episode_ids,
        source_sequence_no: record.source_sequence_no,
        asserted_by: asserted_by.to_string(),
        extractor_version: None,
        confidence: None,
        observed_at_ms: record.observed_at_ms,
        valid_from_ms: record.valid_from_ms,
        valid_to_ms: record.valid_to_ms,
        correction_ids: record.correction_ids,
    }
}

pub(super) fn project_set_states(
    scope: &MemoryScope,
    claims: &[ClaimRecord],
) -> Vec<SetStateRecord> {
    let removals = SetMemberRemovals::from_claims(claims);
    let mut grouped = BTreeMap::<(String, String, u8), (&ClaimRecord, Vec<&ClaimRecord>)>::new();
    for claim in claims {
        if !is_projectable_set_member(claim) {
            continue;
        }
        let slot_key = claim_slot_lifecycle_key(claim);
        if removals.removes(&slot_key, claim) {
            continue;
        }
        let predicate_key = canonical_slot_part(claim.predicate.as_deref().unwrap_or_default());
        grouped
            .entry((
                slot_key,
                predicate_key,
                set_claim_kind_key(claim.claim_kind),
            ))
            .and_modify(|(_, members)| members.push(claim))
            .or_insert((claim, vec![claim]));
    }
    grouped
        .into_iter()
        .map(|((slot_id, predicate_key, _), (first, members))| {
            set_state_from_members(scope, slot_id, &predicate_key, first, &members)
        })
        .collect()
}

/// Set-member removals the claims record: whole-slot invalidations and per-member deletions,
/// each with the time it took effect.
struct SetMemberRemovals {
    invalidated_slots: BTreeMap<MemoryId, i64>,
    removed_members: BTreeMap<(String, String, u8), i64>,
}

impl SetMemberRemovals {
    fn from_claims(claims: &[ClaimRecord]) -> Self {
        let mut removals = Self {
            invalidated_slots: BTreeMap::new(),
            removed_members: BTreeMap::new(),
        };
        for claim in claims {
            if claim_invalidates_entire_set_slot(claim) {
                let invalidated_at = claim_lifecycle_time(claim);
                removals
                    .invalidated_slots
                    .entry(claim_slot_lifecycle_key(claim))
                    .and_modify(|at| *at = (*at).max(invalidated_at))
                    .or_insert(invalidated_at);
            }
            if let Some(member_key) = removed_set_member_key(claim) {
                let removed_at = claim.valid_from_ms.unwrap_or(claim.observed_at_ms);
                removals.removed_members.insert(member_key, removed_at);
            }
        }
        removals
    }

    /// Whether a removal at or after the member claim takes it out of the set.
    fn removes(&self, slot_key: &str, claim: &ClaimRecord) -> bool {
        if self
            .invalidated_slots
            .get(slot_key)
            .is_some_and(|invalidated_at| *invalidated_at >= claim_lifecycle_time(claim))
        {
            return true;
        }
        let member_key = (
            slot_key.to_owned(),
            canonical_slot_part(claim.object_value.as_deref().unwrap_or_default()),
            set_claim_kind_key(claim.claim_kind),
        );
        self.removed_members
            .get(&member_key)
            .is_some_and(|removed_at| {
                *removed_at >= claim.valid_from_ms.unwrap_or(claim.observed_at_ms)
            })
    }
}

/// The (slot, member value, kind) a retracted, tombstoned, or redacted relationship or event
/// claim removes from its set.
fn removed_set_member_key(claim: &ClaimRecord) -> Option<(String, String, u8)> {
    let removed = matches!(
        claim.status,
        MemoryStatus::Retracted | MemoryStatus::Tombstoned | MemoryStatus::Redacted
    ) && matches!(claim.claim_kind, ClaimKind::Relationship | ClaimKind::Event);
    if !removed {
        return None;
    }
    let value = claim.object_value.as_deref()?;
    Some((
        claim_slot_lifecycle_key(claim),
        canonical_slot_part(value),
        set_claim_kind_key(claim.claim_kind),
    ))
}

/// One set state over a slot's member claims: distinct values in claim order, validity from
/// the latest member start to the earliest member end.
fn set_state_from_members(
    scope: &MemoryScope,
    slot_id: MemoryId,
    predicate_key: &str,
    first: &ClaimRecord,
    members: &[&ClaimRecord],
) -> SetStateRecord {
    let mut seen_values = BTreeSet::new();
    let values = members
        .iter()
        .filter_map(|claim| claim.object_value.as_deref())
        .filter(|value| seen_values.insert(canonical_slot_part(value)))
        .map(str::to_string)
        .collect();

    SetStateRecord {
        id: format!(
            "set_state:{:?}:{}:{}",
            first.claim_kind, slot_id, predicate_key
        ),
        scope: scope.clone(),
        status: MemoryStatus::Active,
        claim_kind: first.claim_kind,
        subject: first.subject.clone(),
        predicate: first.predicate.clone(),
        subject_entity_id: first.subject_entity_id.clone(),
        slot_id: Some(slot_id),
        members: values,
        claim_ids: sorted_unique_ids(members.iter().map(|claim| &claim.id)),
        source_span_ids: sorted_unique_ids(members.iter().flat_map(|claim| &claim.source_span_ids)),
        source_episode_ids: sorted_unique_ids(
            members.iter().flat_map(|claim| &claim.source_episode_ids),
        ),
        observed_at_ms: members.iter().fold(first.observed_at_ms, |at, claim| {
            at.max(claim.observed_at_ms)
        }),
        valid_from_ms: Some(
            members
                .iter()
                .fold(claim_lifecycle_time(first), |from, claim| {
                    from.max(claim_lifecycle_time(claim))
                }),
        ),
        valid_to_ms: members
            .iter()
            .filter_map(|claim| claim.valid_to_ms)
            .chain(first.valid_to_ms)
            .min(),
    }
}

pub(super) fn reconcile_set_states_with_slot_lifecycle(
    claims: &[ClaimRecord],
    set_states: &mut Vec<SetStateRecord>,
) {
    // Exact scope is part of the key: one scope's invalidation must not drop another's set.
    let mut invalidators = BTreeMap::<(MemoryId, &MemoryScope), ClaimRecord>::new();
    for claim in claims
        .iter()
        .filter(|claim| claim_invalidates_entire_set_slot(claim))
    {
        let slot_id = claim_slot_lifecycle_key(claim);
        invalidators
            .entry((slot_id, &claim.scope))
            .and_modify(|current| {
                if claim_is_newer(claim, current) {
                    *current = claim.clone();
                }
            })
            .or_insert_with(|| claim.clone());
    }

    set_states.retain(|state| {
        let Some(slot_id) = state.slot_id.as_ref() else {
            return true;
        };
        invalidators
            .get(&(slot_id.clone(), &state.scope))
            .is_none_or(|claim| set_state_is_newer_than_claim(state, claim))
    });
}

fn claim_invalidates_entire_set_slot(claim: &ClaimRecord) -> bool {
    !matches!(claim.claim_kind, ClaimKind::Relationship | ClaimKind::Event)
        && matches!(
            claim_state_record_kind(claim),
            StateRecordKind::Tombstone | StateRecordKind::Unsupported
        )
}

fn claim_lifecycle_time(claim: &ClaimRecord) -> i64 {
    claim.valid_from_ms.unwrap_or(claim.observed_at_ms)
}

fn set_state_is_newer_than_claim(state: &SetStateRecord, claim: &ClaimRecord) -> bool {
    let state_time = state.valid_from_ms.unwrap_or(state.observed_at_ms);
    let claim_time = claim_lifecycle_time(claim);
    state_time > claim_time
        || (state_time == claim_time && state.observed_at_ms > claim.observed_at_ms)
}

pub(super) fn is_projectable_set_member(claim: &ClaimRecord) -> bool {
    matches!(claim.status, MemoryStatus::Active)
        && matches!(claim.polarity, ClaimPolarity::Affirmative)
        && crate::is_direct_state_claim(claim)
        && matches!(
            claim.claim_kind,
            ClaimKind::Relationship | ClaimKind::Event | ClaimKind::Preference
        )
        && claim.object_value.is_some()
}

pub(super) fn set_claim_kind_key(claim_kind: ClaimKind) -> u8 {
    match claim_kind {
        ClaimKind::Relationship => 0,
        ClaimKind::Event => 1,
        ClaimKind::Preference => 2,
        _ => 3,
    }
}

pub(super) fn proof_span_ids_for_claims(claims: &[ClaimRecord]) -> Vec<MemoryId> {
    let mut ids = claims
        .iter()
        .flat_map(|claim| claim.source_span_ids.iter().cloned())
        .collect::<Vec<_>>();
    ids.sort();
    ids.dedup();
    ids
}

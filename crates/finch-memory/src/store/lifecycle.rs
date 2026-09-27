use super::*;

pub(crate) fn dedupe_claims_by_id(claims: &mut Vec<ClaimRecord>) {
    retain_first_by_id(claims, |claim| &claim.id);
}

pub(super) fn apply_corrections_to_claims(
    mut claims: Vec<ClaimRecord>,
    corrections: &[CorrectionRecord],
    at_ms: Option<i64>,
) -> Vec<ClaimRecord> {
    let mut effects = BTreeMap::<MemoryId, ClaimCorrectionEffect<'_>>::new();
    for correction in ordered_claim_corrections(corrections, at_ms) {
        let replacements = record_correction_effects(correction, &claims, &mut effects);
        claims.extend(replacements);
    }
    claims
        .into_iter()
        .map(|claim| {
            let effect = effects.get(&claim.id);
            apply_correction_effect(claim, effect)
        })
        .collect()
}

/// Active claim corrections in effect at `at_ms`, in the order they took effect.
fn ordered_claim_corrections(
    corrections: &[CorrectionRecord],
    at_ms: Option<i64>,
) -> Vec<&CorrectionRecord> {
    let mut ordered = corrections
        .iter()
        .filter(|correction| matches!(correction.status, MemoryStatus::Active))
        .filter(|correction| correction.target_type == "claim")
        .filter(|correction| at_ms.is_none_or(|at| correction.effective_at_ms <= at))
        .collect::<Vec<_>>();
    ordered.sort_by(|a, b| {
        a.effective_at_ms
            .cmp(&b.effective_at_ms)
            .then_with(|| match (a.source_sequence_no, b.source_sequence_no) {
                (Some(a), Some(b)) => a.cmp(&b),
                _ => std::cmp::Ordering::Equal,
            })
            .then_with(|| a.id.cmp(&b.id))
    });
    ordered
}

/// What the applied corrections do to one claim.
#[derive(Default)]
struct ClaimCorrectionEffect<'c> {
    status: Option<MemoryStatus>,
    correction_ids: Vec<MemoryId>,
    /// The latest lifecycle correction: the claim takes its effective time and provenance.
    lifecycle_source: Option<&'c CorrectionRecord>,
}

/// Records what `correction` does to each claim it targets. Returns the replacement claims a
/// replace correction introduces.
fn record_correction_effects<'c>(
    correction: &'c CorrectionRecord,
    claims: &[ClaimRecord],
    effects: &mut BTreeMap<MemoryId, ClaimCorrectionEffect<'c>>,
) -> Vec<ClaimRecord> {
    let status = match correction.operation {
        CorrectionOperation::Restore => MemoryStatus::Active,
        CorrectionOperation::Replace
        | CorrectionOperation::Retract
        | CorrectionOperation::Tombstone
        | CorrectionOperation::Forget
        | CorrectionOperation::MarkStale => correction_claim_status(correction),
        CorrectionOperation::Assert | CorrectionOperation::Merge | CorrectionOperation::Split => {
            return Vec::new();
        }
    };
    let mut replacements = Vec::new();
    for claim in claims
        .iter()
        .filter(|claim| correction_targets_claim(correction, claim))
    {
        let effect = effects.entry(claim.id.clone()).or_default();
        effect.status = Some(status);
        effect.correction_ids.push(correction.id.clone());
        if matches!(correction.operation, CorrectionOperation::Replace) {
            replacements.extend(replacement_claim_from_correction(correction, claim));
        } else {
            effect.lifecycle_source = Some(correction);
        }
    }
    replacements
}

fn apply_correction_effect(
    mut claim: ClaimRecord,
    effect: Option<&ClaimCorrectionEffect<'_>>,
) -> ClaimRecord {
    let Some(effect) = effect else {
        return claim;
    };
    if let Some(status) = effect.status {
        claim.status = status;
    }
    if let Some(correction) = effect.lifecycle_source {
        claim.valid_from_ms = Some(correction.effective_at_ms);
        claim.observed_at_ms = claim.observed_at_ms.max(correction.effective_at_ms);
        if !correction.source_span_ids.is_empty() {
            claim.source_span_ids = correction.source_span_ids.clone();
        }
        if !correction.source_episode_ids.is_empty() {
            claim.source_episode_ids = correction.source_episode_ids.clone();
        }
        if correction.source_sequence_no.is_some() {
            claim.source_sequence_no = correction.source_sequence_no;
        }
    }
    claim
        .correction_ids
        .extend(effect.correction_ids.iter().cloned());
    claim.correction_ids.sort();
    claim.correction_ids.dedup();
    claim
}

pub(super) fn replacement_claim_from_correction(
    correction: &CorrectionRecord,
    target: &ClaimRecord,
) -> Option<ClaimRecord> {
    let new_value = correction.new_value.as_ref()?;
    let subject = target.subject.clone();
    let predicate = target.predicate.clone();
    let id = format!(
        "claim_replace_{}",
        stable_hash_hex(&[&correction.id, &target.id, new_value])
    );
    Some(ClaimRecord {
        id,
        scope: target.scope.clone(),
        status: MemoryStatus::Active,
        visibility: target.visibility,
        policy_tags: target.policy_tags.clone(),
        claim_text: serde_json::json!({
            "subject": subject,
            "predicate": predicate,
            "value": new_value,
        })
        .to_string(),
        subject,
        predicate,
        object_value: Some(new_value.clone()),
        subject_entity_id: target.subject_entity_id.clone(),
        slot_id: target.slot_id.clone(),
        slot_facet: target.slot_facet.clone(),
        claim_kind: target.claim_kind,
        polarity: ClaimPolarity::Affirmative,
        source_span_ids: correction.source_span_ids.clone(),
        source_episode_ids: correction.source_episode_ids.clone(),
        source_sequence_no: correction.source_sequence_no,
        asserted_by: "correction_replace".to_string(),
        extractor_version: None,
        confidence: target.confidence,
        observed_at_ms: correction.effective_at_ms,
        valid_from_ms: Some(correction.effective_at_ms),
        valid_to_ms: target.valid_to_ms,
        correction_ids: vec![correction.id.clone()],
    })
}

pub(super) fn materialize_rule_application(
    trigger: &ClaimRecord,
    rule: &RuleRecord,
    force_unsupported: bool,
    known_ids: &BTreeSet<String>,
) -> ZResult<Option<ClaimRecord>> {
    let (observed_at_ms, valid_to_ms) = if force_unsupported {
        (
            trigger.valid_from_ms.unwrap_or(trigger.observed_at_ms),
            None,
        )
    } else {
        let Some(interval) = rule_trigger_interval(rule, trigger) else {
            return Ok(None);
        };
        interval
    };
    let Some((object_value, polarity, claim_text)) =
        materialized_rule_value(rule, trigger, force_unsupported)?
    else {
        return Ok(None);
    };
    let target_subject = rule.target_subject.clone().unwrap_or_default();
    let target_predicate = rule.target_predicate.clone().unwrap_or_default();
    let trigger_subject = trigger.subject.clone().unwrap_or_default();
    let id = format!(
        "derived_{}",
        stable_hash_hex(&[
            &rule.id,
            &trigger.id,
            &trigger_subject,
            &trigger.observed_at_ms.to_string(),
            &target_subject,
            &target_predicate,
            object_value.as_deref().unwrap_or("")
        ])
    );
    if known_ids.contains(&id) {
        return Ok(None);
    }
    let mut source_span_ids = trigger.source_span_ids.clone();
    source_span_ids.extend(rule.source_span_ids.iter().cloned());
    source_span_ids.sort();
    source_span_ids.dedup();
    let mut source_episode_ids = trigger.source_episode_ids.clone();
    source_episode_ids.extend(rule.source_episode_ids.iter().cloned());
    source_episode_ids.sort();
    source_episode_ids.dedup();
    Ok(Some(ClaimRecord {
        id,
        scope: trigger.scope.clone(),
        status: MemoryStatus::Active,
        visibility: trigger.visibility,
        policy_tags: trigger.policy_tags.clone(),
        claim_text,
        subject: Some(target_subject),
        predicate: Some(target_predicate),
        object_value,
        subject_entity_id: rule.target_subject_entity_id.clone(),
        slot_id: rule.target_slot_id.clone(),
        slot_facet: slot_facet_for_surface(
            rule.target_subject_entity_id.as_deref(),
            &rule.target_subject_key,
            &rule.target_predicate_key,
            rule.target_slot_id.as_deref(),
        ),
        claim_kind: ClaimKind::Fact,
        polarity,
        source_span_ids,
        source_episode_ids,
        source_sequence_no: trigger.source_sequence_no,
        asserted_by: crate::DERIVED_STATE_ASSERTED_BY.to_string(),
        extractor_version: None,
        confidence: rule.confidence.or(trigger.confidence),
        observed_at_ms,
        valid_from_ms: Some(observed_at_ms),
        valid_to_ms,
        correction_ids: Vec::new(),
    }))
}

pub(super) fn claim_slot(claim: &ClaimRecord) -> Option<CanonicalSlot> {
    claim
        .subject
        .as_deref()
        .zip(claim.predicate.as_deref())
        .map(|(subject, predicate)| CanonicalSlot {
            subject_key: claim
                .subject_entity_id
                .clone()
                .unwrap_or_else(|| canonical_slot_part(subject)),
            predicate_key: canonical_slot_part(predicate),
        })
}

pub(super) fn rule_trigger_matches_claim(rule: &RuleRecord, claim: &ClaimRecord) -> bool {
    rule_endpoint_matches_claim(
        rule.trigger_slot_id.as_deref(),
        rule.trigger_subject_entity_id.as_deref(),
        &rule.trigger_subject_key,
        &rule.trigger_predicate_key,
        claim,
    )
}

pub(super) fn rule_target_matches_claim(rule: &RuleRecord, claim: &ClaimRecord) -> bool {
    rule_endpoint_matches_claim(
        rule.target_slot_id.as_deref(),
        rule.target_subject_entity_id.as_deref(),
        &rule.target_subject_key,
        &rule.target_predicate_key,
        claim,
    )
}

fn rule_endpoint_matches_claim(
    slot_id: Option<&str>,
    subject_entity_id: Option<&str>,
    subject_key: &str,
    predicate_key: &str,
    claim: &ClaimRecord,
) -> bool {
    if slot_id.is_some() && slot_id == claim.slot_id.as_deref() {
        return true;
    }
    if canonical_slot_part(claim.predicate.as_deref().unwrap_or_default()) != predicate_key {
        return false;
    }
    match (subject_entity_id, claim.subject_entity_id.as_deref()) {
        (Some(expected), Some(actual)) => expected == actual,
        _ => canonical_slot_part(claim.subject.as_deref().unwrap_or_default()) == subject_key,
    }
}

pub(super) fn rule_trigger_interval(
    rule: &RuleRecord,
    trigger: &ClaimRecord,
) -> Option<(i64, Option<i64>)> {
    let trigger_from_ms = trigger.valid_from_ms.unwrap_or(trigger.observed_at_ms);
    let rule_from_ms = rule.valid_from_ms.unwrap_or(i64::MIN);
    let valid_from_ms = match rule.activation {
        RuleActivation::ContinuousProjection => trigger_from_ms.max(rule_from_ms),
        RuleActivation::OnChange => {
            if !temporal_position_is_after(
                trigger_from_ms,
                trigger.source_sequence_no,
                rule_from_ms,
                rule.source_sequence_no,
            ) {
                return None;
            }
            trigger_from_ms
        }
    };
    let valid_to_ms = match (trigger.valid_to_ms, rule.valid_to_ms) {
        (Some(trigger_to), Some(rule_to)) => Some(trigger_to.min(rule_to)),
        (Some(trigger_to), None) => Some(trigger_to),
        (None, Some(rule_to)) => Some(rule_to),
        (None, None) => None,
    };
    valid_to_ms
        .is_none_or(|valid_to_ms| valid_from_ms < valid_to_ms)
        .then_some((valid_from_ms, valid_to_ms))
}

pub(super) fn materialized_rule_value(
    rule: &RuleRecord,
    trigger: &ClaimRecord,
    force_unsupported: bool,
) -> ZResult<Option<(Option<String>, ClaimPolarity, String)>> {
    if force_unsupported || matches!(rule.action, RuleAction::MarkUnsupported) {
        return Ok(Some((
            None,
            ClaimPolarity::Uncertain,
            serde_json::json!({
                "subject": rule.target_subject,
                "predicate": rule.target_predicate,
                "support_state": "unsupported",
                "trigger_subject": rule.trigger_subject,
                "trigger_predicate": rule.trigger_predicate,
            })
            .to_string(),
        )));
    }
    let trigger_value = match trigger.object_value.as_deref() {
        Some(value) => value,
        None => return Ok(None),
    };
    let target_subject = rule.target_subject.clone().unwrap_or_default();
    let target_predicate = rule.target_predicate.clone().unwrap_or_default();
    if target_subject.trim().is_empty() {
        return Ok(None);
    }
    if target_predicate.trim().is_empty() {
        return Ok(None);
    }
    let value = rule
        .value
        .clone()
        .or_else(|| {
            rule.value_template.as_deref().map(|template| {
                template
                    .replace("{value}", trigger_value)
                    .replace("{object_value}", trigger_value)
                    .replace("{subject}", trigger.subject.as_deref().unwrap_or(""))
                    .replace("{predicate}", trigger.predicate.as_deref().unwrap_or(""))
            })
        })
        .unwrap_or_else(|| trigger_value.to_string());
    Ok(Some((
        Some(value.clone()),
        ClaimPolarity::Affirmative,
        serde_json::json!({
            "subject": target_subject,
            "predicate": target_predicate,
            "value": value,
        })
        .to_string(),
    )))
}

pub(super) fn active_correction_filter(at_ms: Option<i64>) -> String {
    match at_ms {
        Some(at) => format!("status = 'active' AND effective_at_ms <= {at}"),
        None => "status = 'active'".to_string(),
    }
}

pub(super) fn insert_scoped_correction(
    corrections: &mut BTreeMap<String, CorrectionRecord>,
    scope: &MemoryScope,
    correction: CorrectionRecord,
) {
    if correction.scope.matches_filter(scope) {
        corrections.insert(correction.id.clone(), correction);
    }
}

pub(super) fn sort_corrections_by_effect(corrections: &mut [CorrectionRecord]) {
    corrections.sort_by(|a, b| {
        a.effective_at_ms
            .cmp(&b.effective_at_ms)
            .then_with(|| a.id.cmp(&b.id))
    });
}

pub(super) fn correction_claim_status(correction: &CorrectionRecord) -> MemoryStatus {
    match correction.operation {
        CorrectionOperation::Retract => MemoryStatus::Retracted,
        CorrectionOperation::Replace | CorrectionOperation::MarkStale => MemoryStatus::Superseded,
        CorrectionOperation::Tombstone | CorrectionOperation::Forget => MemoryStatus::Tombstoned,
        _ => MemoryStatus::Active,
    }
}

pub(super) fn correction_already_projected(
    correction: &CorrectionRecord,
    records: &[StateRecord],
) -> bool {
    let expected_kind = match correction.operation {
        CorrectionOperation::Retract
        | CorrectionOperation::Tombstone
        | CorrectionOperation::Forget => Some(StateRecordKind::Tombstone),
        CorrectionOperation::MarkStale => Some(StateRecordKind::Unsupported),
        _ => None,
    };
    expected_kind.is_some_and(|kind| {
        records.iter().any(|record| {
            record.state_kind == kind
                && record.valid_from_ms == Some(correction.effective_at_ms)
                && correction_targets_state(correction, record)
        })
    })
}

fn correction_targets_state(correction: &CorrectionRecord, record: &StateRecord) -> bool {
    match correction.target_match {
        CorrectionTargetMatch::CanonicalSlot => correction
            .target_slot_id
            .as_ref()
            .is_some_and(|slot_id| record.slot_id.as_ref() == Some(slot_id)),
        CorrectionTargetMatch::ClaimVersions => {
            correction.target_ids.is_empty()
                || correction
                    .target_ids
                    .iter()
                    .any(|id| record.claim_ids.contains(id))
        }
    }
}

/// A correction reaches claims of its exact scope only, whatever its authority: reads never load
/// a parent scope's corrections, so a wider reach would answer differently at each level.
pub(super) fn correction_targets_claim(correction: &CorrectionRecord, claim: &ClaimRecord) -> bool {
    if claim.scope != correction.scope || !correction_applies_to_claim_time(correction, claim) {
        return false;
    }
    match correction.target_match {
        CorrectionTargetMatch::CanonicalSlot => {
            if let Some(target_slot_id) = correction.target_slot_id.as_ref() {
                return claim.slot_id.as_ref() == Some(target_slot_id);
            }
        }
        CorrectionTargetMatch::ClaimVersions if !correction.target_ids.is_empty() => {
            return correction.target_ids.iter().any(|id| id == &claim.id);
        }
        CorrectionTargetMatch::ClaimVersions => {
            if let Some(selector) = correction.target_selector.as_deref() {
                return claim_matches_selector(claim, selector);
            }
            if let Some(target_slot_id) = correction.target_slot_id.as_ref() {
                return claim.slot_id.as_ref() == Some(target_slot_id);
            }
        }
    }
    correction
        .target_subject_key
        .as_deref()
        .zip(correction.target_predicate_key.as_deref())
        .is_some_and(|(subject, predicate)| {
            canonical_slot_part(claim.subject.as_deref().unwrap_or_default()) == subject
                && canonical_slot_part(claim.predicate.as_deref().unwrap_or_default()) == predicate
        })
        || (matches!(
            correction.target_match,
            CorrectionTargetMatch::CanonicalSlot
        ) && correction
            .target_selector
            .as_deref()
            .is_some_and(|selector| claim_matches_selector(claim, selector)))
}

pub(super) fn correction_applies_to_claim_time(
    correction: &CorrectionRecord,
    claim: &ClaimRecord,
) -> bool {
    let claim_from = claim.valid_from_ms.unwrap_or(claim.observed_at_ms);
    if correction.applies_valid_from_ms.is_none() && correction.applies_valid_to_ms.is_none() {
        return claim_from <= correction.effective_at_ms;
    }
    correction
        .applies_valid_from_ms
        .is_none_or(|from| claim_from >= from)
        && correction
            .applies_valid_to_ms
            .is_none_or(|to| claim_from < to)
}

pub(super) fn claim_matches_selector(claim: &ClaimRecord, selector: &str) -> bool {
    let selector = selector.trim().to_lowercase();
    if selector.is_empty() {
        return false;
    }
    claim.claim_text.to_lowercase().contains(&selector)
        || claim
            .subject
            .as_deref()
            .is_some_and(|value| value.to_lowercase().contains(&selector))
        || claim
            .predicate
            .as_deref()
            .is_some_and(|value| value.to_lowercase().contains(&selector))
        || claim
            .object_value
            .as_deref()
            .is_some_and(|value| value.to_lowercase().contains(&selector))
}

pub(super) fn resolve_current_claims(claims: Vec<ClaimRecord>, limit: usize) -> Vec<ClaimRecord> {
    // Exact scope is part of the key: one scope's newer claim must not hide another scope's.
    let mut by_key = BTreeMap::<(String, MemoryScope), ClaimRecord>::new();
    for claim in claims {
        let key = (claim_projection_key(&claim), claim.scope.clone());
        by_key
            .entry(key)
            .and_modify(|current| {
                if claim_is_newer(&claim, current) {
                    *current = claim.clone();
                }
            })
            .or_insert(claim);
    }
    let mut out = by_key.into_values().collect::<Vec<_>>();
    out.sort_by(|a, b| {
        claim_projection_key(a)
            .cmp(&claim_projection_key(b))
            .then_with(|| a.id.cmp(&b.id))
    });
    out.truncate(limit);
    out
}

pub(super) fn claim_projection_key(claim: &ClaimRecord) -> String {
    if matches!(
        claim.claim_kind,
        ClaimKind::Relationship | ClaimKind::Event | ClaimKind::Preference
    ) {
        return format!(
            "{:?}\u{0}{}\u{0}{}",
            claim.claim_kind,
            claim_facet_lifecycle_key(claim),
            canonical_slot_part(claim.object_value.as_deref().unwrap_or(""))
        );
    }
    if claim.subject.is_some() || claim.predicate.is_some() {
        claim_facet_lifecycle_key(claim)
    } else {
        claim.claim_text.clone()
    }
}

/// Lifecycle key of one surface facet inside a slot family. Newest-wins supersession only
/// applies between claims that share this key; facets bound into the family under another
/// surface stay co-current until explicit lifecycle evidence retires them.
pub(super) fn claim_facet_lifecycle_key(claim: &ClaimRecord) -> String {
    match claim.slot_facet.as_deref() {
        Some(facet) => format!("{}\u{0}{facet}", claim_slot_lifecycle_key(claim)),
        None => claim_slot_lifecycle_key(claim),
    }
}

/// Facet identity for a surface bound into `slot_id`: `None` when the surface's own canonical
/// slot is that slot, otherwise the surface's own canonical slot id.
pub(super) fn slot_facet_for_surface(
    subject_entity_id: Option<&str>,
    subject_key: &str,
    predicate_key: &str,
    slot_id: Option<&str>,
) -> Option<MemoryId> {
    let slot_id = slot_id?;
    if subject_key.is_empty() || predicate_key.is_empty() {
        return None;
    }
    let own = canonical_slot_id(subject_entity_id, subject_key, predicate_key);
    (own != slot_id).then_some(own)
}

pub(super) fn claim_slot_lifecycle_key(claim: &ClaimRecord) -> String {
    claim.slot_id.clone().unwrap_or_else(|| {
        let subject_key = canonical_slot_part(claim.subject.as_deref().unwrap_or(""));
        let predicate_key = canonical_slot_part(claim.predicate.as_deref().unwrap_or(""));
        canonical_slot_id(
            claim.subject_entity_id.as_deref(),
            &subject_key,
            &predicate_key,
        )
    })
}

/// Whether a validity interval holds at `at_ms`; every interval holds when no time is given.
pub(super) fn interval_holds_at(
    valid_from_ms: Option<i64>,
    valid_to_ms: Option<i64>,
    at_ms: Option<i64>,
) -> bool {
    at_ms.is_none_or(|at| {
        valid_from_ms.is_none_or(|from| from <= at) && valid_to_ms.is_none_or(|to| at < to)
    })
}

/// `interval_holds_at` as a filter clause, so rows invalid at `at_ms` cannot consume a limit.
pub(super) fn interval_filter_clause(at_ms: Option<i64>) -> String {
    at_ms.map_or_else(String::new, |at| {
        format!(
            " AND (valid_from_ms IS NULL OR valid_from_ms <= {at}) AND (valid_to_ms IS NULL OR valid_to_ms > {at})"
        )
    })
}

/// The newest of `claims` by `claim_is_newer`; a tie keeps the earlier claim.
pub(super) fn newest_claim<'a>(
    claims: impl IntoIterator<Item = &'a ClaimRecord>,
) -> Option<&'a ClaimRecord> {
    claims.into_iter().reduce(|current, candidate| {
        if claim_is_newer(candidate, current) {
            candidate
        } else {
            current
        }
    })
}

pub(super) fn claim_is_newer(candidate: &ClaimRecord, current: &ClaimRecord) -> bool {
    let candidate_time = candidate.valid_from_ms.unwrap_or(candidate.observed_at_ms);
    let current_time = current.valid_from_ms.unwrap_or(current.observed_at_ms);
    if candidate_time != current_time {
        return candidate_time > current_time;
    }
    if let (Some(candidate_sequence), Some(current_sequence)) =
        (candidate.source_sequence_no, current.source_sequence_no)
    {
        if candidate_sequence != current_sequence {
            return candidate_sequence > current_sequence;
        }
    }
    candidate.observed_at_ms > current.observed_at_ms
        || (candidate.observed_at_ms == current.observed_at_ms
            && claim_lifecycle_priority(candidate) > claim_lifecycle_priority(current))
        || (candidate.observed_at_ms == current.observed_at_ms
            && claim_lifecycle_priority(candidate) == claim_lifecycle_priority(current)
            && candidate.confidence.unwrap_or(0.0) > current.confidence.unwrap_or(0.0))
        || (candidate.observed_at_ms == current.observed_at_ms
            && claim_lifecycle_priority(candidate) == claim_lifecycle_priority(current)
            && candidate.confidence.unwrap_or(0.0) == current.confidence.unwrap_or(0.0)
            && candidate.id > current.id)
}

pub(super) fn temporal_position_is_after(
    candidate_ms: i64,
    candidate_sequence: Option<i64>,
    boundary_ms: i64,
    boundary_sequence: Option<i64>,
) -> bool {
    candidate_ms > boundary_ms
        || (candidate_ms == boundary_ms
            && matches!(
                (candidate_sequence, boundary_sequence),
                (Some(candidate), Some(boundary)) if candidate > boundary
            ))
}

fn claim_lifecycle_priority(claim: &ClaimRecord) -> u8 {
    match claim_state_record_kind(claim) {
        StateRecordKind::Tombstone => 4,
        StateRecordKind::Unsupported => 3,
        StateRecordKind::Derived => 2,
        StateRecordKind::Current | StateRecordKind::Set | StateRecordKind::Rule => 1,
    }
}

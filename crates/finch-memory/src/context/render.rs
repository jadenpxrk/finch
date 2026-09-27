use super::*;

pub(super) struct RenderedContextItem {
    pub(super) id: MemoryId,
    pub(super) kind: &'static str,
    pub(super) rendered: String,
    pub(super) subject_key: Option<String>,
    pub(super) slot_ids: Vec<MemoryId>,
    pub(super) source_text: Option<String>,
}

impl From<(MemoryId, &'static str, String)> for RenderedContextItem {
    fn from((id, kind, rendered): (MemoryId, &'static str, String)) -> Self {
        Self {
            id,
            kind,
            rendered,
            subject_key: None,
            slot_ids: Vec::new(),
            source_text: None,
        }
    }
}

pub(super) fn build_rendered_context_with_raw_reserve(
    items: Vec<RenderedContextItem>,
    token_budget: usize,
    raw_start: usize,
    required_raw_items: usize,
) -> CompiledMemoryContext {
    let raw_reserve = raw_evidence_reserve(&items, token_budget, raw_start, required_raw_items);
    let mut seen = HashSet::new();
    let mut subjects_with_specific_state = HashSet::new();
    let mut packet = Packet::default();
    for (idx, item) in items.into_iter().enumerate() {
        let card_superseded = item.kind == "state_entity_card"
            && item
                .subject_key
                .as_ref()
                .is_some_and(|subject_key| subjects_with_specific_state.contains(subject_key));
        if card_superseded || !seen.insert(item.id.clone()) {
            continue;
        }
        let tokens = estimate_tokens(&item.rendered);
        let limit = if idx < raw_start {
            token_budget.saturating_sub(raw_reserve)
        } else {
            token_budget
        };
        if packet.used + tokens > limit {
            packet.truncated = true;
            continue;
        }
        if is_specific_state_kind(item.kind) {
            subjects_with_specific_state.extend(item.subject_key.clone());
        }
        packet.include(item, tokens);
    }
    packet.into_context(token_budget)
}

/// Tokens kept free for raw evidence while packing the items before `raw_start`: the required
/// proof items (or the first raw item), and at least the evidence's share of the budget.
fn raw_evidence_reserve(
    items: &[RenderedContextItem],
    token_budget: usize,
    raw_start: usize,
    required_raw_items: usize,
) -> usize {
    let tokens = |item: &RenderedContextItem| estimate_tokens(&item.rendered);
    let required_reserve = if required_raw_items == 0 {
        items
            .iter()
            .skip(raw_start)
            .map(tokens)
            .next()
            .map(|tokens| tokens.min(token_budget / 2))
            .unwrap_or(0)
    } else {
        let prefix_tokens = items.iter().take(raw_start).map(tokens).sum::<usize>();
        items
            .iter()
            .skip(raw_start)
            .take(required_raw_items)
            .map(tokens)
            .sum::<usize>()
            .min(token_budget.saturating_sub(prefix_tokens))
    };
    // Coverage state must not crowd evidence out of the packet: after the resolved answer
    // targets, raw evidence keeps at least half of the budget (bounded by what was retrieved).
    let target_tokens = items
        .iter()
        .take(raw_start)
        .filter(|item| item.kind == "state_resolved")
        .map(tokens)
        .sum::<usize>();
    let raw_available = items.iter().skip(raw_start).map(tokens).sum::<usize>();
    let evidence_share = (token_budget / 2)
        .min(raw_available)
        .min(token_budget.saturating_sub(target_tokens));
    required_reserve.max(evidence_share)
}

/// A packet being filled: its text, the items it includes, and their token count.
#[derive(Default)]
struct Packet {
    body: String,
    included: Vec<ContextItem>,
    used: usize,
    truncated: bool,
}

impl Packet {
    fn include(&mut self, item: RenderedContextItem, tokens: usize) {
        self.used += tokens;
        self.body.push_str(&item.rendered);
        self.body.push('\n');
        self.included.push(ContextItem {
            id: item.id,
            kind: item.kind.to_string(),
            estimated_tokens: tokens,
            slot_ids: item.slot_ids,
            source_text: item.source_text,
        });
    }

    fn into_context(self, token_budget: usize) -> CompiledMemoryContext {
        CompiledMemoryContext {
            body: self.body,
            estimated_tokens: self.used,
            budget: token_budget,
            included: self.included,
            truncated: self.truncated,
            support: AnswerSupportContract::default(),
            rule_outcomes: Vec::new(),
        }
    }
}

pub(super) fn build_rendered_context(
    items: impl IntoIterator<Item = RenderedContextItem>,
    token_budget: usize,
) -> CompiledMemoryContext {
    let mut seen = HashSet::new();
    let mut packet = Packet::default();
    for item in items {
        if !seen.insert(item.id.clone()) {
            continue;
        }
        let tokens = estimate_tokens(&item.rendered);
        if packet.used + tokens > token_budget {
            packet.truncated = true;
            if packet.included.is_empty() && token_budget > 0 {
                let clipped = clip_to_budget(&item.rendered, token_budget);
                let clipped_tokens = estimate_tokens(&clipped);
                packet.body.push_str(&clipped);
                packet.included.push(ContextItem {
                    id: item.id,
                    kind: item.kind.to_string(),
                    estimated_tokens: clipped_tokens,
                    slot_ids: Vec::new(),
                    source_text: None,
                });
                packet.used += clipped_tokens;
            }
            break;
        }
        packet.include(item, tokens);
    }
    packet.into_context(token_budget)
}

pub(super) fn render_entity_state_cards(
    claims: &[ClaimRecord],
    alias_map: &BTreeMap<String, (String, String)>,
    include_provenance: bool,
) -> Vec<(usize, String, String, String, Vec<MemoryId>)> {
    let mut groups = BTreeMap::<String, (usize, String, Vec<&ClaimRecord>)>::new();
    for (order, claim) in claims.iter().enumerate() {
        if matches!(claim.claim_kind, ClaimKind::Constraint) {
            continue;
        }
        let Some(subject) = claim.subject.as_deref() else {
            continue;
        };
        let subject_key = canonical_slot_part(subject);
        if subject_key.is_empty() {
            continue;
        }
        let (group_key, display_subject) = alias_map
            .get(&subject_key)
            .cloned()
            .unwrap_or_else(|| (subject_key, subject.to_string()));
        groups
            .entry(group_key)
            .and_modify(|(_, _, grouped)| grouped.push(claim))
            .or_insert_with(|| (order, display_subject, vec![claim]));
    }

    groups
        .into_iter()
        .map(|(subject_key, (order, subject, claims))| {
            let id = format!("entity_state:{}", canonical_slot_part(&subject));
            let slot_ids = claims
                .iter()
                .filter_map(|claim| claim.slot_id.clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            (
                order,
                subject_key,
                id.clone(),
                render_entity_state_card(&id, &subject, &claims, include_provenance),
                slot_ids,
            )
        })
        .collect()
}

pub(super) fn entity_alias_map(entities: &[EntityRecord]) -> BTreeMap<String, (String, String)> {
    let mut out = BTreeMap::new();
    let mut ambiguous = BTreeSet::new();
    for entity in entities {
        if !matches!(entity.status, MemoryStatus::Active) || entity.source_claim_ids.is_empty() {
            continue;
        }
        let canonical_key = canonical_slot_part(&entity.canonical_name);
        if canonical_key.is_empty() {
            continue;
        }
        insert_entity_alias(
            &mut out,
            &mut ambiguous,
            canonical_key.clone(),
            (canonical_key.clone(), entity.canonical_name.clone()),
        );
        for alias in &entity.aliases {
            let alias_key = canonical_slot_part(alias);
            if !alias_key.is_empty() {
                insert_entity_alias(
                    &mut out,
                    &mut ambiguous,
                    alias_key,
                    (canonical_key.clone(), entity.canonical_name.clone()),
                );
            }
        }
    }
    out
}

pub(super) fn insert_entity_alias(
    aliases: &mut BTreeMap<String, (String, String)>,
    ambiguous: &mut BTreeSet<String>,
    alias_key: String,
    target: (String, String),
) {
    if ambiguous.contains(&alias_key) {
        return;
    }
    if let Some(existing) = aliases.get(&alias_key) {
        if existing.0 != target.0 {
            aliases.remove(&alias_key);
            ambiguous.insert(alias_key);
        }
        return;
    }
    aliases.insert(alias_key, target);
}

pub(super) fn render_entity_state_card(
    id: &str,
    subject: &str,
    claims: &[&ClaimRecord],
    include_provenance: bool,
) -> String {
    let mut current = Vec::new();
    let mut derived = Vec::new();
    let mut unsupported = Vec::new();
    let mut tombstoned = Vec::new();
    let mut sets = BTreeMap::<(String, u8), (String, ClaimKind, Vec<String>)>::new();
    let mut claim_ids = BTreeSet::new();
    let mut source_span_ids = BTreeSet::new();

    for claim in claims {
        claim_ids.insert(claim.id.as_str());
        source_span_ids.extend(claim.source_span_ids.iter().map(String::as_str));
        let predicate = claim.predicate.as_deref().unwrap_or("value");
        match claim_context_kind(claim) {
            "state_tombstone" => tombstoned.push(format!("- {predicate}: support_state=deleted")),
            "state_unsupported" => {
                unsupported.push(format!("- {predicate}: support_state=unsupported"))
            }
            "state_derived" => derived.push(format!(
                "- {predicate}: {}",
                claim.object_value.as_deref().unwrap_or("derived")
            )),
            "state_set" if is_active_set_claim(claim) => {
                let key = (
                    canonical_slot_part(predicate),
                    set_claim_kind_key(claim.claim_kind),
                );
                let entry = sets
                    .entry(key)
                    .or_insert_with(|| (predicate.to_string(), claim.claim_kind, Vec::new()));
                if let Some(value) = claim.object_value.as_deref() {
                    entry.2.push(value.to_string());
                }
            }
            "state_current" => current.push(format!(
                "- {predicate}: {}",
                claim.object_value.as_deref().unwrap_or("current")
            )),
            _ => {}
        }
    }

    let mut out = if include_provenance {
        format!(
            "### EntityStateCard {id}\nstate_status: active\nentity: {subject:?}\nclaim_ids: {:?}\nsource_span_ids: {:?}\n",
            claim_ids.into_iter().collect::<Vec<_>>(),
            source_span_ids.into_iter().collect::<Vec<_>>(),
        )
    } else {
        format!("### EntityStateCard {id}\nentity: {subject}\n")
    };

    append_card_section(&mut out, "Current slots", &current);
    append_card_section(&mut out, "Derived slots", &derived);
    append_card_section(&mut out, "Unsupported slots", &unsupported);
    append_card_section(&mut out, "Tombstoned slots", &tombstoned);

    if !sets.is_empty() {
        out.push_str("\nSet slots:\n");
        for (_, (predicate, kind, members)) in sets {
            out.push_str(&format!(
                "- {predicate} [{kind:?}]: {}\n",
                members.join(", ")
            ));
        }
    }
    out
}

pub(super) fn append_card_section(out: &mut String, title: &str, lines: &[String]) {
    if lines.is_empty() {
        return;
    }
    out.push('\n');
    out.push_str(title);
    out.push_str(":\n");
    for line in lines {
        out.push_str(line);
        out.push('\n');
    }
}

pub(super) fn render_hit(hit: &SpanSearchHit, include_provenance: bool) -> String {
    render_span_text(hit, Cow::Borrowed(&hit.span.text), include_provenance)
}

pub(super) fn render_span_text(
    hit: &SpanSearchHit,
    text: Cow<'_, str>,
    include_provenance: bool,
) -> String {
    let span = &hit.span;
    if include_provenance {
        let source_actor = span
            .provenance
            .iter()
            .find_map(|source| source.actor)
            .map(|actor| format!("source_actor: {}\n", actor.as_str()))
            .unwrap_or_default();
        format!(
            "### Memory {}\nsource_type: {:?}\nsource_id: {}\n{}span_index: {}\nscore: {:.4}\nvalid_from: {}\nvalid_to: {}\n\n{}\n",
            span.id,
            span.source_type,
            span.source_id,
            source_actor,
            span.span_index,
            hit.score,
            display_time(span.valid_from_ms),
            display_time(span.valid_to_ms),
            text
        )
    } else {
        format!("### Memory {}\n{}\n", span.id, text)
    }
}

pub(super) fn render_profile(profile: &ProfileRecord, include_provenance: bool) -> String {
    if include_provenance {
        format!(
            "### Profile {}\nkey: {}\nsubject_id: {:?}\ngenerated_at_ms: {}\nvalid_from_ms: {:?}\nvalid_to_ms: {:?}\nevidence_claim_ids: {:?}\nsource_span_ids: {:?}\n\n{}\n",
            profile.id,
            profile.profile_key,
            profile.subject_id,
            profile.generated_at_ms,
            profile.valid_from_ms,
            profile.valid_to_ms,
            profile.evidence_claim_ids,
            profile.source_span_ids,
            profile.profile_text
        )
    } else {
        format!(
            "### Profile {}\n{}: {}\n",
            profile.id, profile.profile_key, profile.profile_text
        )
    }
}

/// Epoch-millis rendered with a human-readable UTC date so a reader can reason
/// about ordering and elapsed time ("which came first, March or July?").
pub(super) fn display_time(value: Option<i64>) -> String {
    match value {
        Some(ms) => format!("{} ({})", civil_date_utc(ms), ms),
        None => String::new(),
    }
}

pub(super) fn civil_date_utc(ms: i64) -> String {
    // Howard Hinnant's civil-from-days algorithm; avoids a chrono dependency.
    let days = ms.div_euclid(86_400_000);
    let secs = ms.rem_euclid(86_400_000) / 1000;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}Z",
        y,
        m,
        d,
        secs / 3600,
        (secs % 3600) / 60
    )
}

pub(super) fn render_resolved_answer_slot(
    slot: &ResolvedAnswerSlot,
    include_provenance: bool,
) -> String {
    let support_state = match slot.support_state {
        crate::AnswerSupportState::Supported => "supported",
        crate::AnswerSupportState::Unsupported => "unsupported",
        crate::AnswerSupportState::Deleted => "deleted",
        crate::AnswerSupportState::NoEvidence => "no_evidence",
    };
    let state_kind = slot
        .state_kind
        .map(StateRecordKind::as_str)
        .unwrap_or("none");
    let read_view = match slot.read_view {
        crate::StateReadView::Current => "current",
        crate::StateReadView::Timeline => "timeline",
        crate::StateReadView::Set => "set",
    };
    let answer_disposition = if matches!(slot.support_state, crate::AnswerSupportState::Supported) {
        "answer"
    } else {
        "refuse"
    };
    let rendered_slot = render_slot(slot.subject.as_deref(), slot.predicate.as_deref());
    let current_value = slot.current_value.as_deref().unwrap_or("null");
    let members = if slot.members.is_empty() {
        "[]".to_string()
    } else {
        serde_json::to_string(&slot.members).unwrap_or_else(|_| "[]".to_string())
    };
    let authoritative_value = matches!(slot.support_state, crate::AnswerSupportState::Supported)
        && matches!(slot.read_view, crate::StateReadView::Current)
        && (slot.facets.len() <= 1 || slot.current_value.is_some());
    let mut rendered = format!(
        "### ResolvedAnswerState {}\nslot: {rendered_slot}\nread_view: {read_view}\nsupport_state: {support_state}\nanswer_disposition: {answer_disposition}\nstate_kind: {state_kind}\nauthoritative_state: true\nauthoritative_current_value: {authoritative_value}\ncurrent_value: {current_value}\nmembers: {members}\nvalid_from_ms: {}\nvalid_to_ms: {}\n",
        slot.slot_id,
        display_time(slot.valid_from_ms),
        display_time(slot.valid_to_ms),
    );
    if slot.facets.len() > 1 {
        rendered.push_str(&format!("co_current_facets: {}\n", slot.facets.len()));
        for facet in &slot.facets {
            let facet_state = match facet.support_state {
                crate::AnswerSupportState::Supported => "supported",
                crate::AnswerSupportState::Unsupported => "unsupported",
                crate::AnswerSupportState::Deleted => "deleted",
                crate::AnswerSupportState::NoEvidence => "no_evidence",
            };
            rendered.push_str(&format!(
                "- facet: {} | support_state: {facet_state} | value: {} | valid_from_ms: {} | valid_to_ms: {}\n",
                render_slot(facet.subject.as_deref(), facet.predicate.as_deref()),
                facet.value.as_deref().unwrap_or("null"),
                display_time(facet.valid_from_ms),
                display_time(facet.valid_to_ms),
            ));
            if include_provenance {
                rendered.push_str(&format!(
                    "  source_claim_ids: {}\n  source_span_ids: {}\n",
                    facet.claim_ids.join(", "),
                    facet.source_span_ids.join(", "),
                ));
            }
        }
    }
    if !slot.dependency_rule_ids.is_empty() {
        rendered.push_str("dependency_resolution: applied\n");
    }
    if include_provenance {
        rendered.push_str(&format!(
            "source_claim_ids: {}\nsource_span_ids: {}\ndependency_rule_ids: {}\ndependency_trigger_slot_ids: {}\n",
            slot.source_claim_ids.join(", "),
            slot.source_span_ids.join(", "),
            slot.dependency_rule_ids.join(", "),
            slot.dependency_trigger_slot_ids.join(", "),
        ));
    }
    rendered
}

pub(super) fn render_claim(claim: &ClaimRecord, include_provenance: bool) -> String {
    let title = claim_context_title(claim);
    let slot = render_slot(claim.subject.as_deref(), claim.predicate.as_deref());
    if !matches!(claim.status, MemoryStatus::Active)
        || matches!(claim_context_kind(claim), "state_unsupported")
    {
        let support_state = if matches!(
            claim.status,
            MemoryStatus::Retracted | MemoryStatus::Tombstoned | MemoryStatus::Redacted
        ) {
            "deleted"
        } else {
            "unsupported"
        };
        return if include_provenance {
            format!(
                "### {title} {}\nslot: {slot}\nsupport_state: {support_state}\ncurrent_value: null\nhistorical_evidence_is_current: false\nvalid_from_ms: {}\nvalid_to_ms: {}\nsource_span_ids: {}\nsource_episode_ids: {}\n",
                claim.id,
                display_i64(claim.valid_from_ms),
                display_i64(claim.valid_to_ms),
                claim.source_span_ids.join(", "),
                claim.source_episode_ids.join(", "),
            )
        } else {
            format!(
                "### {title} {}\nslot: {slot}\nsupport_state: {support_state}\ncurrent_value: null\nhistorical_evidence_is_current: false\n",
                claim.id
            )
        };
    }
    let value = claim.object_value.as_deref().unwrap_or_default();
    if include_provenance {
        format!(
            "### {title} {}\nslot: {slot}\nvalue: {value}\nvalid_from_ms: {}\nvalid_to_ms: {}\nsource_span_ids: {}\nsource_episode_ids: {}\n",
            claim.id,
            display_i64(claim.valid_from_ms),
            display_i64(claim.valid_to_ms),
            claim.source_span_ids.join(", "),
            claim.source_episode_ids.join(", "),
        )
    } else {
        format!("### {title} {}\nslot: {slot}\nvalue: {value}\n", claim.id)
    }
}

pub(super) fn render_rule(
    rule: &RuleRecord,
    outcome: Option<&RuleResolutionOutcome>,
    include_provenance: bool,
) -> Option<String> {
    let outcome = outcome?;
    let resolution_status = match outcome.status {
        RuleResolutionStatus::AppliedDerived => "applied_derived",
        RuleResolutionStatus::AppliedUnsupported => "applied_unsupported",
        RuleResolutionStatus::MissingTrigger
        | RuleResolutionStatus::MissingTriggerValue
        | RuleResolutionStatus::OutsideValidityWindow
        | RuleResolutionStatus::NoPriorValue
        | RuleResolutionStatus::NoValueChange
        | RuleResolutionStatus::NotApplicable => return None,
    };
    let trigger = render_slot(
        rule.trigger_subject.as_deref(),
        rule.trigger_predicate.as_deref(),
    );
    let target = render_slot(
        rule.target_subject.as_deref(),
        rule.target_predicate.as_deref(),
    );
    let resolved_claim_id = outcome.resolved_claim_id.as_deref().unwrap_or_default();
    if include_provenance {
        Some(format!(
            "### DependencyProof {}\ntrigger_slot: {trigger}\ntarget_slot: {target}\nactivation: {}\naction: {}\nresolution_status: {resolution_status}\nresolved_claim_id: {resolved_claim_id}\nauthoritative_current_value: false\nsource_span_ids: {}\n",
            rule.id,
            rule.activation.as_str(),
            rule.action.as_str(),
            rule.source_span_ids.join(", "),
        ))
    } else {
        Some(format!(
            "### DependencyProof {}\ntrigger_slot: {trigger}\ntarget_slot: {target}\nactivation: {}\naction: {}\nresolution_status: {resolution_status}\nresolved_claim_id: {resolved_claim_id}\nauthoritative_current_value: false\n",
            rule.id,
            rule.activation.as_str(),
            rule.action.as_str(),
        ))
    }
}

pub(super) fn render_set_claims(claims: &[&ClaimRecord], include_provenance: bool) -> String {
    let first = claims[0];
    let members = claims
        .iter()
        .filter_map(|claim| claim.object_value.as_deref())
        .collect::<Vec<_>>();
    let claim_ids = claims
        .iter()
        .map(|claim| claim.id.as_str())
        .collect::<Vec<_>>();
    let source_span_ids = claims
        .iter()
        .flat_map(|claim| claim.source_span_ids.iter())
        .collect::<Vec<_>>();
    let source_episode_ids = claims
        .iter()
        .flat_map(|claim| claim.source_episode_ids.iter())
        .collect::<Vec<_>>();
    let id = set_state_id(
        first.subject.as_deref(),
        first.predicate.as_deref(),
        first.claim_kind,
    );
    let slot = render_slot(first.subject.as_deref(), first.predicate.as_deref());
    if include_provenance {
        format!(
            "### SetState {id}\nslot: {slot}\nmembers: {}\nclaim_ids: {}\nsource_span_ids: {}\nsource_episode_ids: {}\n",
            members.join(", "),
            claim_ids.join(", "),
            source_span_ids.into_iter().map(String::as_str).collect::<Vec<_>>().join(", "),
            source_episode_ids.into_iter().map(String::as_str).collect::<Vec<_>>().join(", "),
        )
    } else {
        format!(
            "### SetState {id}\nslot: {slot}\nmembers: {}\n",
            members.join(", ")
        )
    }
}

pub(super) fn render_set_state(state: &SetStateRecord, include_provenance: bool) -> String {
    let slot = render_slot(state.subject.as_deref(), state.predicate.as_deref());
    if include_provenance {
        format!(
            "### SetState {}\nslot: {slot}\nmembers: {}\nvalid_from_ms: {}\nvalid_to_ms: {}\nclaim_ids: {}\nsource_span_ids: {}\nsource_episode_ids: {}\n",
            state.id,
            state.members.join(", "),
            display_i64(state.valid_from_ms),
            display_i64(state.valid_to_ms),
            state.claim_ids.join(", "),
            state.source_span_ids.join(", "),
            state.source_episode_ids.join(", "),
        )
    } else {
        format!(
            "### SetState {}\nslot: {slot}\nmembers: {}\n",
            state.id,
            state.members.join(", "),
        )
    }
}

pub(super) fn render_slot_history(history: &SlotHistoryRecord, include_provenance: bool) -> String {
    let slot = render_slot(history.subject.as_deref(), history.predicate.as_deref());
    let mut rendered = format!(
        "### SlotHistory {}\nslot: {slot}\nauthoritative_current_value: false\nversions_oldest_to_newest:\n",
        history.id
    );
    for version in &history.versions {
        let value = serde_json::to_string(&version.object_value).unwrap_or_else(|_| "null".into());
        let members = serde_json::to_string(&version.members).unwrap_or_else(|_| "[]".into());
        rendered.push_str(&format!(
            "- state_kind: {}; value: {value}; members: {members}; observed_at_ms: {}; valid_from_ms: {}; valid_to_ms: {}",
            version.state_kind.as_str(),
            version.observed_at_ms,
            display_i64(version.valid_from_ms),
            display_i64(version.valid_to_ms),
        ));
        if include_provenance {
            rendered.push_str(&format!(
                "; claim_ids: {}; correction_ids: {}; rule_ids: {}; source_span_ids: {}; source_episode_ids: {}",
                version.claim_ids.join(", "),
                version.correction_ids.join(", "),
                version.rule_ids.join(", "),
                version.source_span_ids.join(", "),
                version.source_episode_ids.join(", "),
            ));
        }
        rendered.push('\n');
    }
    rendered
}

fn render_slot(subject: Option<&str>, predicate: Option<&str>) -> String {
    format!(
        "{} / {}",
        subject.unwrap_or_default(),
        predicate.unwrap_or_default()
    )
}

fn display_i64(value: Option<i64>) -> String {
    value.map_or_else(String::new, |value| value.to_string())
}

pub(super) fn is_active_set_claim(claim: &ClaimRecord) -> bool {
    matches!(claim.status, MemoryStatus::Active)
        && matches!(claim.polarity, ClaimPolarity::Affirmative)
        && crate::is_direct_state_claim(claim)
        && matches!(claim.claim_kind, ClaimKind::Relationship | ClaimKind::Event)
}

pub(super) fn set_claim_kind_key(claim_kind: ClaimKind) -> u8 {
    match claim_kind {
        ClaimKind::Relationship => 0,
        ClaimKind::Event => 1,
        _ => 2,
    }
}

pub(super) fn set_state_id(
    subject: Option<&str>,
    predicate: Option<&str>,
    claim_kind: ClaimKind,
) -> String {
    let canonical_subject = canonical_slot_part(subject.unwrap_or(""));
    let canonical_predicate = canonical_slot_part(predicate.unwrap_or(""));
    format!(
        "set_state:{claim_kind:?}:{}:{}",
        canonical_subject, canonical_predicate
    )
}

pub(super) fn render_correction(correction: &CorrectionRecord, include_provenance: bool) -> String {
    let title = correction_context_title(correction);
    let state = correction_state_text(correction.operation);
    if include_provenance {
        format!(
            "### {title} {}\noperation: {:?}\ntarget_type: {}\ntarget_match: {}\ntarget_ids: {:?}\neffective_at_ms: {}\napplies_valid_from_ms: {:?}\napplies_valid_to_ms: {:?}\n\n{state}\n",
            correction.id,
            correction.operation,
            correction.target_type,
            correction.target_match.as_str(),
            correction.target_ids,
            correction.effective_at_ms,
            correction.applies_valid_from_ms,
            correction.applies_valid_to_ms,
        )
    } else {
        format!("### {title} {}\n{state}\n", correction.id)
    }
}

pub(super) fn claim_context_kind(claim: &ClaimRecord) -> &'static str {
    if matches!(
        claim.status,
        MemoryStatus::Retracted | MemoryStatus::Tombstoned | MemoryStatus::Redacted
    ) {
        return "state_tombstone";
    }
    if !matches!(claim.status, MemoryStatus::Active)
        || matches!(claim.polarity, ClaimPolarity::Uncertain)
    {
        return "state_unsupported";
    }
    if !crate::is_direct_state_claim(claim) {
        return "state_derived";
    }
    match claim.claim_kind {
        ClaimKind::Relationship | ClaimKind::Event => "state_set",
        ClaimKind::Constraint => "state_rule",
        _ => "state_current",
    }
}

pub(super) fn claim_context_title(claim: &ClaimRecord) -> &'static str {
    match claim_context_kind(claim) {
        "state_tombstone" => "TombstoneState",
        "state_unsupported" => "UnsupportedState",
        "state_derived" => "DerivedState",
        "state_set" => "SetState",
        "state_rule" => "StateRule",
        _ => "CurrentState",
    }
}

pub(super) fn correction_context_kind(correction: &CorrectionRecord) -> &'static str {
    match correction.operation {
        CorrectionOperation::Retract
        | CorrectionOperation::Tombstone
        | CorrectionOperation::Forget => "state_tombstone",
        CorrectionOperation::Replace | CorrectionOperation::MarkStale => "state_unsupported",
        CorrectionOperation::Restore => "state_restore",
        CorrectionOperation::Assert | CorrectionOperation::Merge | CorrectionOperation::Split => {
            "state_change"
        }
    }
}

pub(super) fn state_context_priority(kind: &str) -> u8 {
    match kind {
        "state_resolved" => 0,
        "state_tombstone" => 1,
        "state_unsupported" => 2,
        "state_derived" => 3,
        "state_current" => 4,
        "state_set" => 5,
        "state_history" => 6,
        "state_entity_card" => 7,
        "state_rule" => 8,
        _ => 9,
    }
}

pub(super) fn correction_context_title(correction: &CorrectionRecord) -> &'static str {
    match correction_context_kind(correction) {
        "state_tombstone" => "TombstoneState",
        "state_unsupported" => "UnsupportedState",
        "state_restore" => "RestoredState",
        _ => "StateChange",
    }
}

pub(super) fn correction_state_text(operation: CorrectionOperation) -> &'static str {
    match operation {
        CorrectionOperation::Retract => {
            "Matching memory is retracted and must not be treated as current."
        }
        CorrectionOperation::Tombstone | CorrectionOperation::Forget => {
            "Matching memory is deleted and must not be treated as current."
        }
        CorrectionOperation::Replace | CorrectionOperation::MarkStale => {
            "Matching memory is stale or unsupported unless a later active value is present."
        }
        CorrectionOperation::Restore => "Matching memory is restored.",
        CorrectionOperation::Assert | CorrectionOperation::Merge | CorrectionOperation::Split => {
            "Matching memory state changed."
        }
    }
}

pub(super) fn render_artifact(artifact: &ArtifactRecord, include_provenance: bool) -> String {
    if include_provenance {
        format!(
            "### Artifact {}\nkind: {:?}\ntitle: {}\nuri: {:?}\nblob_ref: {:?}\nmime_type: {:?}\nvalid_from_ms: {:?}\nvalid_to_ms: {:?}\nextracted_text_ref: {:?}\n\n{}\n",
            artifact.id,
            artifact.artifact_kind,
            artifact.title,
            artifact.uri,
            artifact.blob_ref,
            artifact.mime_type,
            artifact.valid_from_ms,
            artifact.valid_to_ms,
            artifact.extracted_text_ref,
            artifact.metadata_json.as_deref().unwrap_or("")
        )
    } else {
        format!("### Artifact {}\n{}\n", artifact.id, artifact.title)
    }
}

pub(super) fn clip_to_budget(text: &str, budget: usize) -> String {
    if budget == 0 {
        return String::new();
    }
    let mut out = String::new();
    for word in text.split_whitespace() {
        let candidate = if out.is_empty() {
            word.to_string()
        } else {
            format!("{out} {word}")
        };
        if estimate_tokens(&candidate) > budget {
            break;
        }
        out = candidate;
    }
    out
}

#[cfg(test)]
mod time_render_tests {
    #[test]
    fn civil_date_renders_known_epochs() {
        assert_eq!(super::civil_date_utc(1678406400000), "2023-03-10 00:00Z");
        assert_eq!(super::civil_date_utc(0), "1970-01-01 00:00Z");
        assert_eq!(
            super::display_time(Some(1678406400000)),
            "2023-03-10 00:00Z (1678406400000)"
        );
        assert_eq!(super::display_time(None), "");
    }
}

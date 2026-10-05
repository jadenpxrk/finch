use super::*;

impl MemoryStore {
    pub fn validate_answer_context(
        &self,
        scope: &MemoryScope,
        context: &CompiledMemoryContext,
        emission: AnswerEmission,
    ) -> ZResult<ValidatedAnswerEmission> {
        let validated = validate_packed_answer_emission(context, emission)?;
        self.validate_scoped_answer_evidence(scope, &validated)?;
        Ok(validated)
    }

    pub fn finalize_answer_context(
        &self,
        scope: &MemoryScope,
        context: &CompiledMemoryContext,
        emission: AnswerEmission,
    ) -> ZResult<GroundedAnswerResult> {
        let included_slot_ids = context
            .included
            .iter()
            .flat_map(|item| item.slot_ids.iter().cloned())
            .collect::<BTreeSet<_>>();
        let included_span_ids = context.included_span_ids();
        let support = context
            .support
            .restrict_to_context(&included_slot_ids, &included_span_ids);
        let requested_span_ids = emission
            .claims
            .iter()
            .flat_map(|claim| claim.evidence.iter())
            .map(|evidence| evidence.span_id.clone())
            .collect::<BTreeSet<_>>();
        let spans = self.fetch_spans_by_ids(
            scope,
            &requested_span_ids.into_iter().collect::<Vec<_>>(),
            None,
        )?;
        let span_text_by_id = spans
            .into_iter()
            .map(|span| (span.id, span.text))
            .collect::<BTreeMap<_, _>>();
        Ok(finalize_answer_support(&support, emission, |span_id| {
            span_text_by_id.get(span_id).map(String::as_str)
        }))
    }

    pub fn validate_answer_emission(
        &self,
        scope: &MemoryScope,
        support: &AnswerSupportContract,
        emission: AnswerEmission,
    ) -> ZResult<ValidatedAnswerEmission> {
        let validated = validate_answer_support(support, emission)?;
        self.validate_scoped_answer_evidence(scope, &validated)?;
        Ok(validated)
    }

    fn validate_scoped_answer_evidence(
        &self,
        scope: &MemoryScope,
        emission: &ValidatedAnswerEmission,
    ) -> ZResult<()> {
        if emission.source_span_ids.is_empty() {
            return Ok(());
        }
        let spans = self.fetch_spans_by_ids(scope, &emission.source_span_ids, None)?;
        let spans_by_id = spans
            .into_iter()
            .map(|span| (span.id.clone(), span))
            .collect::<BTreeMap<_, _>>();
        validate_evidence_quotes(emission, |span_id| {
            spans_by_id.get(span_id).map(|span| span.text.as_str())
        })
    }
}

pub fn validate_packed_answer_emission(
    context: &CompiledMemoryContext,
    emission: AnswerEmission,
) -> ZResult<ValidatedAnswerEmission> {
    let included_slot_ids = context
        .included
        .iter()
        .flat_map(|item| item.slot_ids.iter().cloned())
        .collect::<BTreeSet<_>>();
    let evidence_by_id = context
        .included
        .iter()
        .filter(|item| item.kind == "span")
        .filter_map(|item| {
            item.source_text
                .as_deref()
                .map(|text| (item.id.as_str(), text))
        })
        .collect::<BTreeMap<_, _>>();
    let included_span_ids = evidence_by_id
        .keys()
        .map(|span_id| (*span_id).to_string())
        .collect::<BTreeSet<_>>();
    let support = context
        .support
        .restrict_to_context(&included_slot_ids, &included_span_ids);
    let validated = validate_answer_support(&support, emission)?;
    validate_evidence_quotes(&validated, |span_id| evidence_by_id.get(span_id).copied())?;
    Ok(validated)
}

fn validate_answer_support(
    support: &AnswerSupportContract,
    emission: AnswerEmission,
) -> ZResult<ValidatedAnswerEmission> {
    if matches!(emission.disposition, AnswerDisposition::Refuse) {
        if !emission.claims.is_empty() {
            return Err(Status::invalid_argument(
                "a structured refusal cannot contain answer claims",
            ));
        }
        return Ok(ValidatedAnswerEmission {
            disposition: AnswerDisposition::Refuse,
            claims: Vec::new(),
            source_span_ids: Vec::new(),
        });
    }
    if support.slots.is_empty() && !matches!(support.state, AnswerSupportState::Supported) {
        return Err(Status::invalid_argument(
            "unsupported, deleted, or absent state requires a structured refusal",
        ));
    }
    if !matches!(emission.disposition, AnswerDisposition::Answer) || emission.claims.is_empty() {
        return Err(Status::invalid_argument(
            "supported state requires at least one grounded answer claim",
        ));
    }
    let slot_support = support
        .slots
        .iter()
        .map(|slot| (slot.slot_id.as_str(), slot))
        .collect::<BTreeMap<_, _>>();
    for claim in &emission.claims {
        validate_claim_grounding(support, &slot_support, claim)?;
    }
    let requested = emission
        .claims
        .iter()
        .flat_map(|claim| claim.evidence.iter().map(|evidence| &evidence.span_id))
        .collect::<BTreeSet<_>>();
    if requested.is_empty() {
        return Err(Status::invalid_argument(
            "answer claims must cite spans from the answer support contract",
        ));
    }
    if emission
        .claims
        .iter()
        .any(|claim| claim.text.trim().is_empty() || claim.evidence.is_empty())
    {
        return Err(Status::invalid_argument(
            "every answer claim requires text and source evidence",
        ));
    }
    let source_span_ids = requested.into_iter().cloned().collect::<Vec<_>>();
    Ok(ValidatedAnswerEmission {
        disposition: AnswerDisposition::Answer,
        claims: emission.claims,
        source_span_ids,
    })
}

/// Checks that an answer claim cites only spans its support allows: the contract's spans when
/// it has no slots, otherwise the spans of the one supported slot the claim identifies.
fn validate_claim_grounding(
    support: &AnswerSupportContract,
    slot_support: &BTreeMap<&str, &AnswerSlotSupport>,
    claim: &GroundedAnswerClaim,
) -> ZResult<()> {
    let cites_outside = |allowed: &[MemoryId]| {
        claim
            .evidence
            .iter()
            .any(|evidence| !allowed.contains(&evidence.span_id))
    };
    if support.slots.is_empty() {
        if cites_outside(&support.source_span_ids) {
            return Err(Status::invalid_argument(
                "answer claims must cite spans from the answer support contract",
            ));
        }
        return Ok(());
    }
    let selected = match claim.slot_id.as_deref() {
        Some(slot_id) => slot_support.get(slot_id).copied(),
        None if support.slots.len() == 1 => support.slots.first(),
        None => None,
    }
    .ok_or_else(|| {
        Status::invalid_argument("answer claims must identify one projected canonical slot")
    })?;
    if !matches!(selected.state, AnswerSupportState::Supported) {
        return Err(Status::invalid_argument(
            "answer claims cannot use an unsupported, deleted, or absent slot",
        ));
    }
    if cites_outside(&selected.source_span_ids) {
        return Err(Status::invalid_argument(
            "answer claim evidence is outside its canonical slot support",
        ));
    }
    Ok(())
}

fn finalize_answer_support<'a>(
    support: &AnswerSupportContract,
    emission: AnswerEmission,
    mut source_text: impl FnMut(&str) -> Option<&'a str>,
) -> GroundedAnswerResult {
    if matches!(emission.disposition, AnswerDisposition::Refuse) {
        return refused_answer(support, &emission.claims);
    }
    let (emitted_claims, claim_validations) =
        check_answer_claims(support, emission.claims, &mut source_text);
    if emitted_claims.is_empty() {
        return GroundedAnswerResult {
            disposition: AnswerDisposition::Refuse,
            claims: Vec::new(),
            source_span_ids: Vec::new(),
            refusal_reason: Some(
                answer_refusal_reason_for_support(support)
                    .unwrap_or(AnswerRefusalReason::NoValidClaims),
            ),
            claim_validations,
        };
    }
    let source_span_ids = emitted_claims
        .iter()
        .flat_map(|claim| claim.evidence.iter().map(|evidence| &evidence.span_id))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .cloned()
        .collect();
    GroundedAnswerResult {
        disposition: AnswerDisposition::Answer,
        claims: emitted_claims,
        source_span_ids,
        refusal_reason: None,
        claim_validations,
    }
}

/// Each claim either emitted (bound to the slot it answers from) or rejected with its reason,
/// with one validation per input claim in input order.
fn check_answer_claims<'a>(
    support: &AnswerSupportContract,
    claims: Vec<GroundedAnswerClaim>,
    source_text: &mut impl FnMut(&str) -> Option<&'a str>,
) -> (Vec<GroundedAnswerClaim>, Vec<AnswerClaimValidation>) {
    let slots_by_id = support
        .slots
        .iter()
        .map(|slot| (slot.slot_id.as_str(), slot))
        .collect::<BTreeMap<_, _>>();
    let mut emitted_claims = Vec::new();
    let mut claim_validations = Vec::with_capacity(claims.len());
    for (input_index, mut claim) in claims.into_iter().enumerate() {
        let slot = match select_claim_slot(support, &slots_by_id, &claim) {
            Ok(slot) => slot,
            Err((slot_id, reason)) => {
                claim_validations.push(rejected_claim_validation(input_index, slot_id, reason));
                continue;
            }
        };
        if let Some(reason) = answer_claim_rejection(support, slot, &claim, source_text) {
            let slot_id = slot.map(|slot| slot.slot_id.clone());
            claim_validations.push(rejected_claim_validation(input_index, slot_id, reason));
            continue;
        }
        if claim.slot_id.is_none() {
            claim.slot_id = slot.map(|slot| slot.slot_id.clone());
        }
        claim_validations.push(AnswerClaimValidation {
            input_index,
            decision: AnswerClaimDecision::Emitted,
            slot_id: claim.slot_id.clone(),
            rejection_reason: None,
        });
        emitted_claims.push(claim);
    }
    (emitted_claims, claim_validations)
}

/// A requested refusal: every claim it carried is rejected.
fn refused_answer(
    support: &AnswerSupportContract,
    claims: &[GroundedAnswerClaim],
) -> GroundedAnswerResult {
    let claim_validations = claims
        .iter()
        .enumerate()
        .map(|(input_index, claim)| {
            rejected_claim_validation(
                input_index,
                claim.slot_id.clone(),
                AnswerClaimRejectionReason::RefusalContainsClaims,
            )
        })
        .collect();
    GroundedAnswerResult {
        disposition: AnswerDisposition::Refuse,
        claims: Vec::new(),
        source_span_ids: Vec::new(),
        refusal_reason: Some(
            answer_refusal_reason_for_support(support).unwrap_or(AnswerRefusalReason::Requested),
        ),
        claim_validations,
    }
}

type ClaimRejection = (Option<MemoryId>, AnswerClaimRejectionReason);

/// The support slot a claim answers from: the slot it names, or the only slot. `None` when the
/// support has no slots.
fn select_claim_slot<'s>(
    support: &'s AnswerSupportContract,
    slots_by_id: &BTreeMap<&str, &'s AnswerSlotSupport>,
    claim: &GroundedAnswerClaim,
) -> Result<Option<&'s AnswerSlotSupport>, ClaimRejection> {
    if support.slots.is_empty() {
        return Ok(None);
    }
    match claim.slot_id.as_deref() {
        Some(slot_id) => slots_by_id.get(slot_id).copied().map(Some).ok_or_else(|| {
            (
                claim.slot_id.clone(),
                AnswerClaimRejectionReason::UnknownSlot,
            )
        }),
        None if support.slots.len() == 1 => Ok(support.slots.first()),
        None => Err((None, AnswerClaimRejectionReason::AmbiguousSlot)),
    }
}

/// Why a claim answering from `slot` (or the whole support) cannot be emitted, if it cannot.
fn answer_claim_rejection<'a>(
    support: &AnswerSupportContract,
    slot: Option<&AnswerSlotSupport>,
    claim: &GroundedAnswerClaim,
    source_text: &mut impl FnMut(&str) -> Option<&'a str>,
) -> Option<AnswerClaimRejectionReason> {
    let support_state = slot.map_or(support.state, |slot| slot.state);
    if let Some(reason) = answer_claim_rejection_for_support_state(support_state) {
        return Some(reason);
    }
    if claim.text.trim().is_empty() {
        return Some(AnswerClaimRejectionReason::MissingText);
    }
    if claim.evidence.is_empty() {
        return Some(AnswerClaimRejectionReason::MissingEvidence);
    }
    let allowed_span_ids = slot.map_or(support.source_span_ids.as_slice(), |slot| {
        slot.source_span_ids.as_slice()
    });
    if claim
        .evidence
        .iter()
        .any(|evidence| !allowed_span_ids.contains(&evidence.span_id))
    {
        return Some(AnswerClaimRejectionReason::EvidenceOutsideSupport);
    }
    claim.evidence.iter().find_map(|evidence| {
        let Some(text) = source_text(&evidence.span_id) else {
            return Some(AnswerClaimRejectionReason::EvidenceUnavailable);
        };
        (evidence.quote.trim().is_empty() || !text.contains(&evidence.quote))
            .then_some(AnswerClaimRejectionReason::QuoteMismatch)
    })
}

fn rejected_claim_validation(
    input_index: usize,
    slot_id: Option<MemoryId>,
    rejection_reason: AnswerClaimRejectionReason,
) -> AnswerClaimValidation {
    AnswerClaimValidation {
        input_index,
        decision: AnswerClaimDecision::Rejected,
        slot_id,
        rejection_reason: Some(rejection_reason),
    }
}

fn answer_claim_rejection_for_support_state(
    state: AnswerSupportState,
) -> Option<AnswerClaimRejectionReason> {
    match state {
        AnswerSupportState::Supported => None,
        AnswerSupportState::Unsupported => Some(AnswerClaimRejectionReason::UnsupportedSlot),
        AnswerSupportState::Deleted => Some(AnswerClaimRejectionReason::DeletedSlot),
        AnswerSupportState::NoEvidence => Some(AnswerClaimRejectionReason::NoEvidence),
    }
}

fn answer_refusal_reason_for_support(
    support: &AnswerSupportContract,
) -> Option<AnswerRefusalReason> {
    let mut states = support
        .slots
        .iter()
        .map(|slot| slot.state)
        .chain(std::iter::once(support.state));
    if states
        .clone()
        .any(|state| matches!(state, AnswerSupportState::Deleted))
    {
        Some(AnswerRefusalReason::DeletedState)
    } else if states
        .clone()
        .any(|state| matches!(state, AnswerSupportState::Unsupported))
    {
        Some(AnswerRefusalReason::UnsupportedState)
    } else if states.any(|state| matches!(state, AnswerSupportState::NoEvidence)) {
        Some(AnswerRefusalReason::NoEvidence)
    } else {
        None
    }
}

fn validate_evidence_quotes<'a>(
    emission: &ValidatedAnswerEmission,
    mut source_text: impl FnMut(&str) -> Option<&'a str>,
) -> ZResult<()> {
    for evidence in emission
        .claims
        .iter()
        .flat_map(|claim| claim.evidence.iter())
    {
        let Some(text) = source_text(&evidence.span_id) else {
            return Err(Status::invalid_argument(
                "answer evidence span is not available in the validated context",
            ));
        };
        if evidence.quote.trim().is_empty() || !text.contains(&evidence.quote) {
            return Err(Status::invalid_argument(
                "answer evidence quote is not present in the cited span",
            ));
        }
    }
    Ok(())
}

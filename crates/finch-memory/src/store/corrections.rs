use super::*;

impl MemoryStore {
    pub fn resolve_existing_correction_target(
        &self,
        correction: &CorrectionRecord,
    ) -> ZResult<CorrectionRecord> {
        self.resolve_correction_target_with_pending_claims(correction, &[])
    }

    pub fn resolve_correction_target_with_pending_claims(
        &self,
        correction: &CorrectionRecord,
        pending_claims: &[ClaimRecord],
    ) -> ZResult<CorrectionRecord> {
        let (mut correction, pending_claims) =
            self.canonicalize_correction_inputs(correction, pending_claims)?;
        if correction.target_type != "claim"
            || !correction.operation.requires_existing_claim_target()
        {
            return Ok(correction);
        }
        let identities = self.scan_slots(&correction.scope, usize::MAX, None)?;
        let mut targets = self.named_target_claims(&correction, &pending_claims)?;
        let structured_slot_ids =
            structured_target_slot_ids(&correction, &identities, &targets.claims, &pending_claims)?;
        self.add_selector_targets(&correction, &mut targets)?;
        let slot_id = single_candidate_slot(
            correction.target_match,
            structured_slot_ids,
            targets.slot_ids,
        )?;
        correction.target_ids = targets.ids.into_iter().collect();
        bind_correction_to_slot(&mut correction, slot_id, &identities, &pending_claims);
        Ok(correction)
    }

    /// The correction and the pending claims of its scope, canonicalised at the correction's
    /// effective time.
    fn canonicalize_correction_inputs(
        &self,
        correction: &CorrectionRecord,
        pending_claims: &[ClaimRecord],
    ) -> ZResult<(CorrectionRecord, Vec<ClaimRecord>)> {
        let pending_claims = pending_claims
            .iter()
            .filter(|claim| claim.scope == correction.scope)
            .collect::<Vec<_>>();
        let names = correction
            .target_subject_key
            .iter()
            .map(String::as_str)
            .chain(
                pending_claims
                    .iter()
                    .filter_map(|claim| claim.subject.as_deref()),
            );
        let registry = self.canonical_registry_for_names_at(
            &correction.scope,
            names,
            Some(correction.effective_at_ms),
        )?;
        let pending_claims = pending_claims
            .into_iter()
            .map(|claim| canonicalize_claim(claim.clone(), &registry))
            .collect();
        Ok((
            canonicalize_correction(correction.clone(), &registry),
            pending_claims,
        ))
    }

    /// Stored and pending claims the correction names by id.
    fn named_target_claims(
        &self,
        correction: &CorrectionRecord,
        pending_claims: &[ClaimRecord],
    ) -> ZResult<CorrectionTargetClaims> {
        let mut claims = self.claims_by_ids(&correction.scope, &correction.target_ids)?;
        let requested_ids = correction
            .target_ids
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        claims.extend(
            pending_claims
                .iter()
                .filter(|claim| requested_ids.contains(claim.id.as_str()))
                .cloned(),
        );
        Ok(CorrectionTargetClaims {
            ids: claims.iter().map(|claim| claim.id.clone()).collect(),
            slot_ids: claims
                .iter()
                .filter_map(|claim| claim.slot_id.clone())
                .collect(),
            claims,
        })
    }

    /// Adds the stored claims the correction's selector matches at its effective time.
    fn add_selector_targets(
        &self,
        correction: &CorrectionRecord,
        targets: &mut CorrectionTargetClaims,
    ) -> ZResult<()> {
        let Some(selector) = correction.target_selector.as_deref() else {
            return Ok(());
        };
        let selector_claims = self
            .scan_claims(&correction.scope, usize::MAX, None)?
            .into_iter()
            .filter(|claim| claim_matches_selector(claim, selector))
            .filter(|claim| correction_applies_to_claim_time(correction, claim))
            .collect::<Vec<_>>();
        for claim in selector_claims {
            targets.slot_ids.extend(claim.slot_id);
            targets.ids.insert(claim.id);
        }
        Ok(())
    }
}

/// Claims a destructive correction targets, with their ids and slots.
struct CorrectionTargetClaims {
    claims: Vec<ClaimRecord>,
    ids: BTreeSet<MemoryId>,
    slot_ids: BTreeSet<MemoryId>,
}

/// Slots the correction names structurally: its target slot (which must exist or be confirmed
/// by a claim) and every slot of its target subject and predicate.
fn structured_target_slot_ids(
    correction: &CorrectionRecord,
    identities: &[CanonicalSlotRecord],
    target_claims: &[ClaimRecord],
    pending_claims: &[ClaimRecord],
) -> ZResult<BTreeSet<MemoryId>> {
    let mut slot_ids = BTreeSet::new();
    if let Some(slot_id) = correction.target_slot_id.as_ref() {
        let claim_confirms_slot = target_claims
            .iter()
            .chain(pending_claims)
            .any(|claim| claim.slot_id.as_ref() == Some(slot_id));
        let slot_exists = identities.iter().any(|slot| slot.slot_key == *slot_id);
        if !slot_exists && !claim_confirms_slot {
            return Err(Status::invalid_argument(
                "destructive correction target slot does not exist",
            ));
        }
        slot_ids.insert(slot_id.clone());
    }
    let Some(predicate_key) = correction.target_predicate_key.as_deref() else {
        return Ok(slot_ids);
    };
    if correction.target_subject_key.is_none() {
        return Ok(slot_ids);
    }
    slot_ids.extend(
        identities
            .iter()
            .filter(|slot| slot.predicate_key == predicate_key)
            .filter(|slot| {
                is_correction_subject(
                    correction,
                    slot.subject_entity_id.as_ref(),
                    &slot.subject_key,
                )
            })
            .map(|slot| slot.slot_key.clone()),
    );
    slot_ids.extend(
        pending_claims
            .iter()
            .filter(|claim| {
                let subject_key = canonical_slot_part(claim.subject.as_deref().unwrap_or_default());
                is_correction_subject(correction, claim.subject_entity_id.as_ref(), &subject_key)
            })
            .filter(|claim| {
                canonical_slot_part(claim.predicate.as_deref().unwrap_or_default()) == predicate_key
            })
            .filter_map(|claim| claim.slot_id.clone()),
    );
    Ok(slot_ids)
}

/// Whether a subject is the correction's target subject: by entity when the correction names
/// one, otherwise by canonical subject key.
fn is_correction_subject(
    correction: &CorrectionRecord,
    subject_entity_id: Option<&MemoryId>,
    subject_key: &str,
) -> bool {
    match correction.target_subject_entity_id.as_ref() {
        Some(entity_id) => subject_entity_id == Some(entity_id),
        None => correction.target_subject_key.as_deref() == Some(subject_key),
    }
}

/// The one slot a destructive correction applies to: its structural slots for a canonical-slot
/// match (else its claims' slots), or both for a claim-versions match.
fn single_candidate_slot(
    target_match: CorrectionTargetMatch,
    structured_slot_ids: BTreeSet<MemoryId>,
    claim_slot_ids: BTreeSet<MemoryId>,
) -> ZResult<MemoryId> {
    let candidate_slot_ids = match target_match {
        CorrectionTargetMatch::CanonicalSlot if !structured_slot_ids.is_empty() => {
            structured_slot_ids
        }
        CorrectionTargetMatch::CanonicalSlot => claim_slot_ids,
        CorrectionTargetMatch::ClaimVersions => structured_slot_ids
            .into_iter()
            .chain(claim_slot_ids)
            .collect(),
    };
    let mut candidates = candidate_slot_ids.into_iter();
    match (candidates.next(), candidates.next()) {
        (Some(slot_id), None) => Ok(slot_id),
        (None, _) => Err(Status::invalid_argument(
            "destructive correction requires an existing canonical slot",
        )),
        (Some(_), Some(_)) => Err(Status::invalid_argument(
            "destructive correction target is ambiguous across canonical slots",
        )),
    }
}

/// Points the correction at `slot_id` and takes the slot's identity from the stored slot, or
/// from a pending claim on it.
fn bind_correction_to_slot(
    correction: &mut CorrectionRecord,
    slot_id: MemoryId,
    identities: &[CanonicalSlotRecord],
    pending_claims: &[ClaimRecord],
) {
    let identity_by_id = identities
        .iter()
        .map(|slot| (slot.slot_key.as_str(), slot))
        .collect::<BTreeMap<_, _>>();
    if let Some(slot) = identity_by_id.get(slot_id.as_str()) {
        correction.target_subject_key = Some(slot.subject_key.clone());
        correction.target_predicate_key = Some(slot.predicate_key.clone());
        correction.target_subject_entity_id = slot.subject_entity_id.clone();
    } else if let Some(claim) = pending_claims
        .iter()
        .find(|claim| claim.slot_id.as_ref() == Some(&slot_id))
    {
        correction.target_subject_key = claim.subject.as_deref().map(canonical_slot_part);
        correction.target_predicate_key = claim.predicate.as_deref().map(canonical_slot_part);
        correction.target_subject_entity_id = claim.subject_entity_id.clone();
    }
    correction.target_slot_id = Some(slot_id);
}

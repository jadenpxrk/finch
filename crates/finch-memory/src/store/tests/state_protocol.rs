use super::*;
use crate::state::{
    AnswerClaimRejectionReason, AnswerDisposition, AnswerEmission, AnswerEvidence,
    AnswerRefusalReason, BiTemporalQuery, GroundedAnswerClaim, StateMutationBatch,
};
use crate::store::state_records::state_record_from_claim;
use crate::{CompiledMemoryContext, ContextItem};

#[test]
fn state_mutation_evidence_must_resolve_in_scope() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("state_mutation_evidence_validation");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let ingested = store
        .ingest_episode(
            crate::ingest::EpisodeInput {
                id: Some("episode_evidence".to_string()),
                scope: scope.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                source_kind: SourceKind::UserMessage,
                actor: ActorKind::User,
                sequence_no: 1,
                event_time_ms: Some(10),
                valid_from_ms: Some(10),
                valid_to_ms: None,
                raw_text: "La responsable es Lucía.".to_string(),
                blob_ref: None,
                mime_type: Some("text/plain".to_string()),
                causal_parent_ids: Vec::new(),
                metadata_json: None,
            },
            10,
            &crate::ingest::ChunkOptions::default(),
        )
        .unwrap();
    let mut claim = make_claim(
        &scope,
        "claim_evidence",
        "proyecto",
        "responsable",
        Some("Lucía"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    claim.source_span_ids = vec![ingested.spans[0].id.clone()];
    claim.source_episode_ids = vec![ingested.episode.id.clone()];
    let valid = mutation(
        &scope,
        10,
        vec![claim.clone()],
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    assert!(store.validate_batch_evidence(&valid).is_ok());

    claim.source_span_ids = vec!["missing_span".to_string()];
    let invalid = mutation(&scope, 10, vec![claim], Vec::new(), Vec::new(), Vec::new());
    assert!(store.validate_batch_evidence(&invalid).is_err());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn evidence_backed_slot_aliases_unify_multilingual_lifecycle_state() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("slot_alias_lifecycle");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let original = make_claim(
        &scope,
        "claim_facturation_original",
        "projet",
        "contact de facturation",
        Some("Alice"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let evidence = make_claim(
        &scope,
        "claim_facturation_alias_evidence",
        "schéma",
        "équivalence",
        Some("contact facturation = contact de facturation"),
        11,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    store.append_claim(&original, None).unwrap();
    store.append_claim(&evidence, None).unwrap();
    let target_slot_id = store
        .claims_by_ids(&scope, std::slice::from_ref(&original.id))
        .unwrap()[0]
        .slot_id
        .clone()
        .unwrap();
    store
        .add_slot_alias(
            SlotAliasInput {
                id: Some("slot_alias_facturation".to_string()),
                scope: scope.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                alias_subject: "projet".to_string(),
                alias_predicate: "contact facturation".to_string(),
                canonical_slot_id: Some(target_slot_id.clone()),
                target_claim_id: None,
                source_claim_ids: vec![evidence.id],
                valid_from_ms: Some(11),
                valid_to_ms: None,
            },
            11,
        )
        .unwrap();
    let replacement = make_claim(
        &scope,
        "claim_facturation_replacement",
        "projet",
        "contact facturation",
        Some("Béatrice"),
        20,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    store.append_claim(&replacement, None).unwrap();

    let stored = store
        .claims_by_ids(&scope, std::slice::from_ref(&replacement.id))
        .unwrap();
    assert_eq!(stored[0].slot_id.as_deref(), Some(target_slot_id.as_str()));
    let current = store.scan_current_claims(&scope, 20, Some(20)).unwrap();
    assert!(current.iter().any(
        |claim| claim.id == replacement.id && claim.object_value.as_deref() == Some("Béatrice")
    ));
    assert!(!current.iter().any(|claim| claim.id == original.id));
    let binding_context = store
        .canonical_slot_binding_contexts(&scope, 20, Some(20))
        .unwrap()
        .into_iter()
        .find(|context| context.slot_id == target_slot_id)
        .unwrap();
    assert!(binding_context
        .slot_aliases
        .iter()
        .any(|alias| { alias.subject == "projet" && alias.predicate == "contact facturation" }));
    assert!(binding_context
        .temporal_state
        .iter()
        .any(|state| state.state_text.contains("Béatrice")));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn conflicting_slot_aliases_fail_closed() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("slot_alias_ambiguity");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let first = make_claim(
        &scope,
        "claim_alias_target_first",
        "servicio",
        "responsable principal",
        Some("Lina"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let second = make_claim(
        &scope,
        "claim_alias_target_second",
        "servicio",
        "responsable suplente",
        Some("Noé"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let evidence_a = make_claim(
        &scope,
        "claim_alias_evidence_a",
        "modelo",
        "equivalencia a",
        Some("encargado"),
        11,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let evidence_b = make_claim(
        &scope,
        "claim_alias_evidence_b",
        "modelo",
        "equivalencia b",
        Some("encargado"),
        12,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    for claim in [&first, &second, &evidence_a, &evidence_b] {
        store.append_claim(claim, None).unwrap();
    }
    let first_slot = store.claims_by_ids(&scope, &[first.id]).unwrap()[0]
        .slot_id
        .clone()
        .unwrap();
    let second_slot = store.claims_by_ids(&scope, &[second.id]).unwrap()[0]
        .slot_id
        .clone()
        .unwrap();
    for (id, slot, source) in [
        ("slot_alias_ambiguous_a", first_slot.clone(), evidence_a.id),
        ("slot_alias_ambiguous_b", second_slot.clone(), evidence_b.id),
    ] {
        store
            .add_slot_alias(
                SlotAliasInput {
                    id: Some(id.to_string()),
                    scope: scope.clone(),
                    visibility: Visibility::Private,
                    policy_tags: Vec::new(),
                    alias_subject: "servicio".to_string(),
                    alias_predicate: "encargado".to_string(),
                    canonical_slot_id: Some(slot),
                    target_claim_id: None,
                    source_claim_ids: vec![source],
                    valid_from_ms: Some(12),
                    valid_to_ms: None,
                },
                12,
            )
            .unwrap();
    }
    let ambiguous = make_claim(
        &scope,
        "claim_alias_ambiguous_value",
        "servicio",
        "encargado",
        Some("Iris"),
        20,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    store.append_claim(&ambiguous, None).unwrap();
    let ambiguous_slot = store.claims_by_ids(&scope, &[ambiguous.id]).unwrap()[0]
        .slot_id
        .clone()
        .unwrap();
    assert_ne!(ambiguous_slot, first_slot);
    assert_ne!(ambiguous_slot, second_slot);
    let binding_contexts = store
        .canonical_slot_binding_contexts(&scope, 20, Some(20))
        .unwrap();
    assert!(binding_contexts.iter().all(|context| {
        context
            .slot_aliases
            .iter()
            .all(|alias| alias.subject != "servicio" || alias.predicate != "encargado")
    }));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn slot_alias_validity_does_not_rewrite_earlier_claims() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("slot_alias_validity");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let target = make_claim(
        &scope,
        "claim_slot_alias_target",
        "sistema",
        "propietario",
        Some("Ada"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let evidence = make_claim(
        &scope,
        "claim_slot_alias_validity_evidence",
        "modelo",
        "equivalencia",
        Some("encargado = propietario"),
        40,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    store.append_claim(&target, None).unwrap();
    store.append_claim(&evidence, None).unwrap();
    let target_slot = store.claims_by_ids(&scope, &[target.id]).unwrap()[0]
        .slot_id
        .clone()
        .unwrap();
    store
        .add_slot_alias(
            SlotAliasInput {
                id: Some("slot_alias_future".to_string()),
                scope: scope.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                alias_subject: "sistema".to_string(),
                alias_predicate: "encargado".to_string(),
                canonical_slot_id: Some(target_slot.clone()),
                target_claim_id: None,
                source_claim_ids: vec![evidence.id],
                valid_from_ms: Some(50),
                valid_to_ms: None,
            },
            40,
        )
        .unwrap();
    let before = make_claim(
        &scope,
        "claim_slot_alias_before",
        "sistema",
        "encargado",
        Some("Bela"),
        45,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let after = make_claim(
        &scope,
        "claim_slot_alias_after",
        "sistema",
        "encargado",
        Some("Cora"),
        60,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    store.append_claim(&before, None).unwrap();
    store.append_claim(&after, None).unwrap();
    let stored_before = store.claims_by_ids(&scope, &[before.id]).unwrap();
    let stored_after = store.claims_by_ids(&scope, &[after.id]).unwrap();
    assert_ne!(
        stored_before[0].slot_id.as_deref(),
        Some(target_slot.as_str())
    );
    assert_eq!(
        stored_after[0].slot_id.as_deref(),
        Some(target_slot.as_str())
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn invalid_same_batch_slot_alias_rolls_back_claims() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("slot_alias_atomic_rollback");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let claim = make_claim(
        &scope,
        "claim_slot_alias_rollback",
        "servicio",
        "campo",
        Some("valor"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let mut batch = mutation(
        &scope,
        10,
        vec![claim.clone()],
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    batch.slot_aliases.push(SlotAliasInput {
        id: Some("slot_alias_invalid_target".to_string()),
        scope: scope.clone(),
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        alias_subject: "servicio".to_string(),
        alias_predicate: "otro campo".to_string(),
        canonical_slot_id: Some("slot_missing".to_string()),
        target_claim_id: None,
        source_claim_ids: vec![claim.id.clone()],
        valid_from_ms: Some(10),
        valid_to_ms: None,
    });

    assert!(store.apply_state_mutation_batch(batch).is_err());
    assert!(store.scan_claims(&scope, 20, None).unwrap().is_empty());
    assert!(store
        .scan_slot_aliases(&scope, 20, None)
        .unwrap()
        .is_empty());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn answer_emission_requires_verbatim_in_scope_evidence() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("validated_answer_emission");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let mut span = span_record("span_answer_evidence", MemoryStatus::Active, Some(10));
    span.scope = scope.clone();
    span.text = "El responsable actual es Lucía.".to_string();
    span.lexical_text = span.text.clone();
    store.append_span(&span, None).unwrap();
    let support = AnswerSupportContract {
        state: AnswerSupportState::Supported,
        source_claim_ids: vec!["claim_owner".to_string()],
        source_span_ids: vec![span.id.clone()],
        slots: Vec::new(),
    };
    let emission = AnswerEmission {
        disposition: AnswerDisposition::Answer,
        claims: vec![GroundedAnswerClaim {
            text: "Lucía".to_string(),
            slot_id: None,
            evidence: vec![AnswerEvidence {
                span_id: span.id.clone(),
                quote: "responsable actual es Lucía".to_string(),
            }],
        }],
    };

    let validated = store
        .validate_answer_emission(&scope, &support, emission)
        .unwrap();
    assert_eq!(validated.source_span_ids, vec![span.id.clone()]);
    assert!(store
        .validate_answer_emission(
            &scope,
            &support,
            AnswerEmission {
                disposition: AnswerDisposition::Refuse,
                claims: Vec::new(),
            },
        )
        .is_ok());

    let invalid = AnswerEmission {
        disposition: AnswerDisposition::Answer,
        claims: vec![GroundedAnswerClaim {
            text: "María".to_string(),
            slot_id: None,
            evidence: vec![AnswerEvidence {
                span_id: span.id,
                quote: "responsable actual es María".to_string(),
            }],
        }],
    };
    assert!(store
        .validate_answer_emission(&scope, &support, invalid)
        .is_err());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn answer_emission_validates_support_per_canonical_slot() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("validated_answer_slot_support");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let mut current_span = span_record("span_current_owner", MemoryStatus::Active, Some(20));
    current_span.scope = scope.clone();
    current_span.text = "La responsable actual es Lucía.".to_string();
    current_span.lexical_text = current_span.text.clone();
    store.append_span(&current_span, None).unwrap();
    let mut deleted_span = span_record("span_deleted_vendor", MemoryStatus::Active, Some(10));
    deleted_span.scope = scope.clone();
    deleted_span.text = "El proveedor anterior era Norte.".to_string();
    deleted_span.lexical_text = deleted_span.text.clone();
    store.append_span(&deleted_span, None).unwrap();
    let support = AnswerSupportContract {
        state: AnswerSupportState::Deleted,
        source_claim_ids: vec!["claim_owner".to_string(), "claim_vendor".to_string()],
        source_span_ids: vec![current_span.id.clone(), deleted_span.id.clone()],
        slots: vec![
            AnswerSlotSupport {
                slot_id: "slot_owner".to_string(),
                subject: None,
                predicate: None,
                view: StateReadView::Current,
                state: AnswerSupportState::Supported,
                source_claim_ids: vec!["claim_owner".to_string()],
                source_span_ids: vec![current_span.id.clone()],
            },
            AnswerSlotSupport {
                slot_id: "slot_vendor".to_string(),
                subject: None,
                predicate: None,
                view: StateReadView::Current,
                state: AnswerSupportState::Deleted,
                source_claim_ids: vec!["claim_vendor".to_string()],
                source_span_ids: vec![deleted_span.id.clone()],
            },
        ],
    };

    let supported = AnswerEmission {
        disposition: AnswerDisposition::Answer,
        claims: vec![GroundedAnswerClaim {
            text: "Lucía".to_string(),
            slot_id: Some("slot_owner".to_string()),
            evidence: vec![AnswerEvidence {
                span_id: current_span.id,
                quote: "responsable actual es Lucía".to_string(),
            }],
        }],
    };
    assert!(store
        .validate_answer_emission(&scope, &support, supported)
        .is_ok());

    let deleted = AnswerEmission {
        disposition: AnswerDisposition::Answer,
        claims: vec![GroundedAnswerClaim {
            text: "Norte".to_string(),
            slot_id: Some("slot_vendor".to_string()),
            evidence: vec![AnswerEvidence {
                span_id: deleted_span.id,
                quote: "proveedor anterior era Norte".to_string(),
            }],
        }],
    };
    assert!(store
        .validate_answer_emission(&scope, &support, deleted)
        .is_err());

    let ambiguous = AnswerEmission {
        disposition: AnswerDisposition::Answer,
        claims: vec![GroundedAnswerClaim {
            text: "Lucía".to_string(),
            slot_id: None,
            evidence: Vec::new(),
        }],
    };
    assert!(store
        .validate_answer_emission(&scope, &support, ambiguous)
        .is_err());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn answer_finalization_strips_unsupported_claims_and_refuses_when_none_remain() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("grounded_answer_finalization");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let mut current_span = span_record("span_current_reviewer", MemoryStatus::Active, Some(20));
    current_span.scope = scope.clone();
    current_span.text = "La revisora confirmada es Lucía.".to_string();
    current_span.lexical_text = current_span.text.clone();
    store.append_span(&current_span, None).unwrap();
    let mut deleted_span = span_record("span_deleted_supplier", MemoryStatus::Active, Some(10));
    deleted_span.scope = scope.clone();
    deleted_span.text = "以前の供給元は北部でした。".to_string();
    deleted_span.lexical_text = deleted_span.text.clone();
    store.append_span(&deleted_span, None).unwrap();

    let context = CompiledMemoryContext {
        body: String::new(),
        estimated_tokens: 0,
        budget: 256,
        included: vec![
            ContextItem {
                id: current_span.id.clone(),
                kind: "span".to_string(),
                estimated_tokens: 8,
                slot_ids: vec!["slot_reviewer".to_string()],
                source_text: Some(current_span.text.clone()),
            },
            ContextItem {
                id: deleted_span.id.clone(),
                kind: "span".to_string(),
                estimated_tokens: 8,
                slot_ids: vec!["slot_supplier".to_string()],
                source_text: Some(deleted_span.text.clone()),
            },
        ],
        truncated: false,
        support: AnswerSupportContract {
            state: AnswerSupportState::Deleted,
            source_claim_ids: vec!["claim_reviewer".to_string(), "claim_supplier".to_string()],
            source_span_ids: vec![current_span.id.clone(), deleted_span.id.clone()],
            slots: vec![
                AnswerSlotSupport {
                    slot_id: "slot_reviewer".to_string(),
                    subject: None,
                    predicate: None,
                    view: StateReadView::Current,
                    state: AnswerSupportState::Supported,
                    source_claim_ids: vec!["claim_reviewer".to_string()],
                    source_span_ids: vec![current_span.id.clone()],
                },
                AnswerSlotSupport {
                    slot_id: "slot_supplier".to_string(),
                    subject: None,
                    predicate: None,
                    view: StateReadView::Current,
                    state: AnswerSupportState::Deleted,
                    source_claim_ids: vec!["claim_supplier".to_string()],
                    source_span_ids: vec![deleted_span.id.clone()],
                },
            ],
        },
        rule_outcomes: Vec::new(),
    };
    let emission = AnswerEmission {
        disposition: AnswerDisposition::Answer,
        claims: vec![
            GroundedAnswerClaim {
                text: "Lucía".to_string(),
                slot_id: Some("slot_reviewer".to_string()),
                evidence: vec![AnswerEvidence {
                    span_id: current_span.id.clone(),
                    quote: "revisora confirmada es Lucía".to_string(),
                }],
            },
            GroundedAnswerClaim {
                text: "北部".to_string(),
                slot_id: Some("slot_supplier".to_string()),
                evidence: vec![AnswerEvidence {
                    span_id: deleted_span.id.clone(),
                    quote: "以前の供給元は北部".to_string(),
                }],
            },
        ],
    };
    let result = store
        .finalize_answer_context(&scope, &context, emission)
        .unwrap();
    assert_eq!(result.disposition, AnswerDisposition::Answer);
    assert_eq!(result.claims.len(), 1);
    assert_eq!(result.claims[0].text, "Lucía");
    assert_eq!(result.source_span_ids, vec![current_span.id.clone()]);
    assert_eq!(result.claim_validations.len(), 2);
    assert_eq!(
        result.claim_validations[1].rejection_reason,
        Some(AnswerClaimRejectionReason::DeletedSlot)
    );

    let refused = store
        .finalize_answer_context(
            &scope,
            &context,
            AnswerEmission {
                disposition: AnswerDisposition::Answer,
                claims: vec![GroundedAnswerClaim {
                    text: "北部".to_string(),
                    slot_id: Some("slot_supplier".to_string()),
                    evidence: vec![AnswerEvidence {
                        span_id: deleted_span.id,
                        quote: "以前の供給元は北部".to_string(),
                    }],
                }],
            },
        )
        .unwrap();
    assert_eq!(refused.disposition, AnswerDisposition::Refuse);
    assert!(refused.claims.is_empty());
    assert_eq!(
        refused.refusal_reason,
        Some(AnswerRefusalReason::DeletedState)
    );

    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn answer_projection_builds_support_per_canonical_slot() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("answer_projection_slot_support");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let owner = make_claim(
        &scope,
        "claim_project_owner",
        "proyecto",
        "responsable",
        Some("Lucía"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let vendor = make_claim(
        &scope,
        "claim_project_vendor",
        "proyecto",
        "proveedor",
        Some("Norte"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    store
        .apply_state_mutation_batch(mutation(
            &scope,
            100,
            vec![owner.clone(), vendor.clone()],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ))
        .unwrap();

    let mut correction = crate::ingest::add_correction(
        CorrectionInput {
            id: Some("correction_project_vendor".to_string()),
            scope: scope.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            operation: CorrectionOperation::Tombstone,
            target_type: "claim".to_string(),
            target_ids: vec![vendor.id.clone()],
            target_selector: None,
            new_value: None,
            reason: None,
            actor: ActorKind::User,
            authority: CorrectionAuthority::User,
            effective_at_ms: Some(20),
            applies_valid_from_ms: None,
            applies_valid_to_ms: None,
            cascade_policy: None,
            metadata_json: None,
        },
        200,
    );
    correction.source_span_ids = vec!["span_vendor_deleted".to_string()];
    correction.source_episode_ids = vec!["episode_vendor_deleted".to_string()];
    store
        .apply_state_mutation_batch(mutation(
            &scope,
            200,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![correction],
        ))
        .unwrap();

    let projection = store
        .project_answer_ready_state(
            &scope,
            &answer_request(
                "slot_support",
                "proyecto responsable proveedor",
                &[],
                10,
                0,
                Some(25),
            ),
        )
        .unwrap();
    let owner_support = projection
        .support
        .slots
        .iter()
        .find(|slot| slot.source_claim_ids.contains(&owner.id))
        .unwrap();
    assert_eq!(owner_support.state, AnswerSupportState::Supported);
    let vendor_support = projection
        .support
        .slots
        .iter()
        .find(|slot| slot.source_claim_ids.contains(&vendor.id))
        .unwrap();
    assert_eq!(vendor_support.state, AnswerSupportState::Deleted);
    assert_ne!(owner_support.slot_id, vendor_support.slot_id);

    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn packed_context_support_requires_admitted_raw_proof() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("packed_context_support_boundary");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let claim = make_claim(
        &scope,
        "claim_current_owner",
        "プロジェクト",
        "責任者",
        Some("愛子"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let mut span = span_record(&claim.source_span_ids[0], MemoryStatus::Active, Some(10));
    span.scope = scope.clone();
    span.source_id = claim.source_episode_ids[0].clone();
    span.text = "プロジェクトの責任者は愛子です。".to_string();
    span.lexical_text = span.text.clone();
    store.append_span(&span, None).unwrap();
    store
        .apply_state_mutation_batch(mutation(
            &scope,
            100,
            vec![claim],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ))
        .unwrap();
    let projection = store
        .project_answer_ready_state(
            &scope,
            &answer_request(
                "packed_support",
                "プロジェクトの責任者",
                &[],
                10,
                0,
                Some(20),
            ),
        )
        .unwrap();

    let without_proof = build_answer_ready_state_context(
        &[],
        &projection,
        &[],
        &[],
        ContextOptions {
            token_budget: 512,
            include_provenance: true,
        },
    );
    assert_eq!(without_proof.support.slots.len(), 1);
    assert_eq!(
        without_proof.support.slots[0].state,
        AnswerSupportState::NoEvidence
    );
    assert!(without_proof.support.slots[0].source_span_ids.is_empty());

    let hit = crate::retrieval::SpanSearchHit {
        score: 1.0,
        span: span.clone(),
    };
    let with_proof = build_answer_ready_state_context(
        &[],
        &projection,
        &[],
        &[hit],
        ContextOptions {
            token_budget: 512,
            include_provenance: true,
        },
    );
    let slot = with_proof.support.slots.first().unwrap();
    assert_eq!(slot.state, AnswerSupportState::Supported);
    assert_eq!(slot.source_span_ids, vec![span.id.clone()]);
    assert!(with_proof
        .included
        .iter()
        .any(|item| item.kind == "span" && item.id == span.id));

    let emission = AnswerEmission {
        disposition: AnswerDisposition::Answer,
        claims: vec![GroundedAnswerClaim {
            text: "愛子".to_string(),
            slot_id: Some(slot.slot_id.clone()),
            evidence: vec![AnswerEvidence {
                span_id: span.id.clone(),
                quote: "責任者は愛子".to_string(),
            }],
        }],
    };
    assert!(store
        .validate_answer_context(&scope, &without_proof, emission.clone())
        .is_err());
    let mut widened_without_proof = without_proof.clone();
    widened_without_proof.support = projection.support.clone();
    assert!(store
        .validate_answer_context(&scope, &widened_without_proof, emission.clone())
        .is_err());
    let validated_json = store
        .validate_answer_context_json(
            &serde_json::to_string(&scope).unwrap(),
            &serde_json::to_string(&with_proof).unwrap(),
            &serde_json::to_string(&emission).unwrap(),
        )
        .unwrap();
    let validated =
        serde_json::from_str::<crate::ValidatedAnswerEmission>(&validated_json).unwrap();
    assert_eq!(validated.source_span_ids, vec![span.id.clone()]);
    assert!(store
        .validate_answer_context(&scope, &with_proof, emission)
        .is_ok());

    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn unsupported_answer_emission_requires_structured_refusal() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("validated_answer_refusal");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let support = AnswerSupportContract {
        state: AnswerSupportState::Unsupported,
        source_claim_ids: Vec::new(),
        source_span_ids: Vec::new(),
        slots: Vec::new(),
    };
    let answer = AnswerEmission {
        disposition: AnswerDisposition::Answer,
        claims: vec![GroundedAnswerClaim {
            text: "ancienne valeur".to_string(),
            slot_id: None,
            evidence: Vec::new(),
        }],
    };
    assert!(store
        .validate_answer_emission(&scope(), &support, answer)
        .is_err());
    let refusal = AnswerEmission {
        disposition: AnswerDisposition::Refuse,
        claims: Vec::new(),
    };
    assert!(store
        .validate_answer_emission(&scope(), &support, refusal)
        .is_ok());
    let _ = std::fs::remove_dir_all(dir);
}

fn exact_rule(
    scope: &MemoryScope,
    id: &str,
    subject: &str,
    trigger_predicate: &str,
    target_predicate: &str,
    action: RuleAction,
) -> RuleInput {
    RuleInput {
        id: Some(id.to_string()),
        scope: scope.clone(),
        status: MemoryStatus::Active,
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        trigger_subject_key: subject.to_string(),
        trigger_predicate_key: trigger_predicate.to_string(),
        target_subject_key: subject.to_string(),
        target_predicate_key: target_predicate.to_string(),
        trigger_slot_id: None,
        target_slot_id: None,
        trigger_subject: Some(subject.to_string()),
        trigger_predicate: Some(trigger_predicate.to_string()),
        target_subject: Some(subject.to_string()),
        target_predicate: Some(target_predicate.to_string()),
        target_match: RuleTargetMatch::ExactSlot,
        activation: if matches!(action, RuleAction::MarkUnsupported) {
            RuleActivation::OnChange
        } else {
            RuleActivation::ContinuousProjection
        },
        action,
        value_template: Some("{value}".to_string()),
        value: None,
        source_span_ids: vec![format!("span_{id}")],
        source_episode_ids: vec![format!("episode_{id}")],
        valid_from_ms: Some(0),
        valid_to_ms: None,
        confidence: Some(1.0),
    }
}

fn mutation(
    scope: &MemoryScope,
    transaction_time_ms: i64,
    claims: Vec<ClaimRecord>,
    entities: Vec<EntityInput>,
    rules: Vec<RuleInput>,
    corrections: Vec<CorrectionRecord>,
) -> StateMutationBatch {
    StateMutationBatch {
        scope: scope.clone(),
        transaction_time_ms,
        propagation_seed: format!("mutation_{transaction_time_ms}"),
        max_rule_hops: 8,
        claims,
        entities,
        rules,
        slot_aliases: Vec::new(),
        corrections,
    }
}

fn projected_value<'a>(
    projection: &'a AnswerReadyStateProjection,
    predicate: &str,
) -> Option<&'a str> {
    projection
        .claims
        .iter()
        .find(|claim| claim.predicate.as_deref() == Some(predicate))
        .and_then(|claim| claim.object_value.as_deref())
}

fn dependency_trace(
    scope: &MemoryScope,
    id: &str,
    derived_claim_id: &str,
    valid_from_ms: i64,
) -> DependencyTraceRecord {
    DependencyTraceRecord {
        id: id.to_string(),
        trace_key: id.to_string(),
        scope: scope.clone(),
        status: MemoryStatus::Active,
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        rule_id: format!("rule_{id}"),
        trigger_claim_id: format!("trigger_{id}"),
        prior_trigger_claim_id: None,
        derived_claim_id: derived_claim_id.to_string(),
        target_subject_key: "equipo".to_string(),
        target_predicate_key: "miembros".to_string(),
        target_subject_entity_id: None,
        target_slot_id: Some("slot_equipo_miembros".to_string()),
        target_subject: Some("equipo".to_string()),
        target_predicate: Some("miembros".to_string()),
        state_kind: StateRecordKind::Derived,
        hop: 1,
        parent_trace_ids: Vec::new(),
        source_span_ids: vec![format!("span_{id}")],
        source_episode_ids: vec![format!("episode_{id}")],
        observed_at_ms: valid_from_ms,
        valid_from_ms: Some(valid_from_ms),
        valid_to_ms: None,
        projected_at_ms: 100,
        recorded_at_ms: 100,
        superseded_at_ms: None,
    }
}

#[test]
fn rejected_mutation_batch_does_not_leave_partial_claims() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("state_protocol_atomic_validation");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let claim = make_claim(
        &scope,
        "claim_atomic",
        "servicio",
        "responsable",
        Some("Ana"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let mut invalid_rule = exact_rule(
        &scope,
        "rule_atomic",
        "servicio",
        "responsable",
        "contacto",
        RuleAction::DeriveValue,
    );
    invalid_rule.source_span_ids.clear();

    assert!(store
        .apply_state_mutation_batch(mutation(
            &scope,
            100,
            vec![claim],
            Vec::new(),
            vec![invalid_rule],
            Vec::new(),
        ))
        .is_err());
    assert!(store.scan_claims(&scope, 10, None).unwrap().is_empty());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn pending_mutation_journal_is_recovered_on_reopen() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("state_protocol_journal_recovery");
    let scope = scope();
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    store
        .append_claim(
            &make_claim(
                &scope,
                "claim_preexisting_unrelated",
                "perfil",
                "zona",
                Some("este"),
                9,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            ),
            None,
        )
        .unwrap();
    store.begin_state_mutation_journal(&scope).unwrap();
    store
        .capture_state_mutation_documents(
            CLAIMS_COLLECTION,
            &store.claims,
            ["claim_interrupted".to_string()],
        )
        .unwrap();
    let journal = std::fs::read_to_string(dir.join(".state-mutation-journal.json")).unwrap();
    assert!(journal.contains("claim_interrupted"));
    assert!(!journal.contains("claim_preexisting_unrelated"));
    // A pending journal refuses journaled writes, so the rows land directly, as a crash leaves them.
    for (id, subject, predicate, value, at) in [
        ("claim_interrupted", "proyecto", "responsable", "Ana", 10),
        ("claim_concurrent_unrelated", "otro", "estado", "activo", 11),
    ] {
        let claim = make_claim(
            &scope,
            id,
            subject,
            predicate,
            Some(value),
            at,
            (ClaimKind::Fact, ClaimPolarity::Affirmative),
        );
        insert_one(&store.claims, claim_doc(&claim, None).unwrap()).unwrap();
    }
    drop(store);

    let store = MemoryStore::open(&dir, CollectionOptions::default()).unwrap();
    let claims = store.scan_claims(&scope, 10, None).unwrap();
    assert!(!claims.iter().any(|claim| claim.id == "claim_interrupted"));
    assert!(claims
        .iter()
        .any(|claim| claim.id == "claim_concurrent_unrelated"));
    assert!(claims
        .iter()
        .any(|claim| claim.id == "claim_preexisting_unrelated"));

    drop(store);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn failed_mutation_rolls_back_writes_completed_before_storage_error() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("state_protocol_runtime_rollback");
    let scope = scope();
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let claim = make_claim(
        &scope,
        "claim_duplicate",
        "proyecto",
        "responsable",
        Some("Ana"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );

    assert!(store
        .apply_state_mutation_batch(mutation(
            &scope,
            100,
            vec![claim.clone(), claim],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ))
        .is_err());
    assert!(store.scan_claims(&scope, 10, None).unwrap().is_empty());
    assert!(!dir.join(".state-mutation-journal.json").exists());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn failed_mutation_restores_existing_rules_rebound_by_the_batch() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("state_protocol_rebind_rollback");
    let scope = scope();
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    store
        .apply_state_mutation_batch(mutation(
            &scope,
            100,
            Vec::new(),
            Vec::new(),
            vec![exact_rule(
                &scope,
                "rule_rebound_then_rolled_back",
                "proyecto",
                "responsable",
                "revisor",
                RuleAction::DeriveValue,
            )],
            Vec::new(),
        ))
        .unwrap();
    let rules_before = store.scan_rules(&scope, 10, None).unwrap();

    let mut correction = crate::ingest::add_correction(
        CorrectionInput {
            id: Some("correction_duplicate".to_string()),
            scope: scope.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            operation: CorrectionOperation::Tombstone,
            target_type: "claim".to_string(),
            target_ids: vec!["claim_owner_rolled_back".to_string()],
            target_selector: None,
            new_value: None,
            reason: None,
            actor: ActorKind::User,
            authority: CorrectionAuthority::User,
            effective_at_ms: Some(30),
            applies_valid_from_ms: None,
            applies_valid_to_ms: None,
            cascade_policy: None,
            metadata_json: None,
        },
        200,
    );
    correction.source_span_ids = vec!["span_correction".to_string()];
    correction.source_episode_ids = vec!["episode_correction".to_string()];
    assert!(store
        .apply_state_mutation_batch(mutation(
            &scope,
            200,
            vec![make_claim(
                &scope,
                "claim_owner_rolled_back",
                "proyecto",
                "responsable",
                Some("Ana"),
                20,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            )],
            vec![EntityInput {
                id: None,
                scope: scope.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                entity_type: "project".to_string(),
                canonical_name: "proyecto".to_string(),
                aliases: Vec::new(),
                source_claim_ids: vec!["claim_owner_rolled_back".to_string()],
                merge_parent_ids: Vec::new(),
                split_from_id: None,
                confidence: None,
            }],
            Vec::new(),
            vec![correction.clone(), correction],
        ))
        .is_err());

    assert!(store.scan_claims(&scope, 10, None).unwrap().is_empty());
    assert_eq!(store.scan_rules(&scope, 10, None).unwrap(), rules_before);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn multilingual_multi_hop_state_survives_reopen_and_bitemporal_queries() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("state_protocol_multilingual_cascade");
    let scope = scope();
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let rules = vec![
        exact_rule(
            &scope,
            "rule_responsable_revisor",
            "プロジェクト",
            "責任者",
            "査読者",
            RuleAction::DeriveValue,
        ),
        exact_rule(
            &scope,
            "rule_revisor_aprobador",
            "プロジェクト",
            "査読者",
            "承認者",
            RuleAction::DeriveValue,
        ),
    ];
    let first = store
        .apply_state_mutation_batch(mutation(
            &scope,
            100,
            vec![make_claim(
                &scope,
                "claim_responsable_ana",
                "プロジェクト",
                "責任者",
                Some("Ana"),
                10,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            )],
            Vec::new(),
            rules,
            Vec::new(),
        ))
        .unwrap();
    assert_eq!(first.rule_applications.len(), 2);
    assert!(first
        .rule_applications
        .iter()
        .any(|application| { application.hop == 2 && !application.parent_trace_ids.is_empty() }));

    store
        .apply_state_mutation_batch(mutation(
            &scope,
            200,
            vec![make_claim(
                &scope,
                "claim_responsable_bea",
                "プロジェクト",
                "責任者",
                Some("Béatrice"),
                20,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            )],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ))
        .unwrap();
    drop(store);

    let store = MemoryStore::open(&dir, CollectionOptions::default()).unwrap();
    let historical = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                temporal: BiTemporalQuery::as_of(15, 150),
                ..answer_request("historical", "プロジェクト 承認者", &[], 20, 0, None)
            },
        )
        .unwrap();
    let current = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                temporal: BiTemporalQuery::as_of(25, 250),
                ..answer_request("current", "プロジェクト 承認者", &[], 20, 0, None)
            },
        )
        .unwrap();
    assert_eq!(projected_value(&historical, "承認者"), Some("Ana"));
    assert_eq!(projected_value(&current, "承認者"), Some("Béatrice"));
    assert!(store
        .scan_dependency_traces(&scope, 20, Some(25))
        .unwrap()
        .iter()
        .any(|trace| !trace.parent_trace_ids.is_empty()));
    let historical_traces = store
        .scan_dependency_traces_bitemporal(&scope, 20, BiTemporalQuery::as_of(15, 250))
        .unwrap();
    let current_traces = store
        .scan_dependency_traces_bitemporal(&scope, 20, BiTemporalQuery::as_of(25, 250))
        .unwrap();
    assert!(historical_traces
        .iter()
        .any(|trace| trace.trigger_claim_id == "claim_responsable_ana"));
    assert!(current_traces
        .iter()
        .any(|trace| trace.trigger_claim_id == "claim_responsable_bea"));
    assert!(!current_traces
        .iter()
        .any(|trace| trace.trigger_claim_id == "claim_responsable_ana"));

    let historical_slots = store
        .scan_slots_bitemporal(&scope, 20, BiTemporalQuery::as_of(15, 250))
        .unwrap();
    let current_slots = store
        .scan_slots_bitemporal(&scope, 20, BiTemporalQuery::as_of(25, 250))
        .unwrap();
    assert!(historical_slots.iter().any(|slot| slot
        .source_claim_ids
        .iter()
        .any(|id| id == "claim_responsable_ana")));
    assert!(historical_slots.iter().any(|slot| slot
        .source_claim_ids
        .iter()
        .any(|id| id == "claim_responsable_bea")));
    assert!(current_slots.iter().any(|slot| slot
        .source_claim_ids
        .iter()
        .any(|id| id == "claim_responsable_bea")));
    let slots_before_update = store
        .scan_slots_bitemporal(&scope, 20, BiTemporalQuery::as_of(15, 150))
        .unwrap();
    assert!(slots_before_update.iter().any(|slot| slot
        .source_claim_ids
        .iter()
        .any(|id| id == "claim_responsable_ana")));
    assert!(!slots_before_update.iter().any(|slot| slot
        .source_claim_ids
        .iter()
        .any(|id| id == "claim_responsable_bea")));

    drop(store);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn late_arriving_history_does_not_replace_newer_current_state() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("state_protocol_late_history");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    store
        .apply_state_mutation_batch(mutation(
            &scope,
            200,
            vec![make_claim(
                &scope,
                "claim_owner_current",
                "proyecto",
                "responsable",
                Some("Ana"),
                20,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            )],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ))
        .unwrap();
    store
        .apply_state_mutation_batch(mutation(
            &scope,
            300,
            vec![make_claim(
                &scope,
                "claim_owner_late_history",
                "proyecto",
                "responsable",
                Some("Bea"),
                10,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            )],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ))
        .unwrap();

    let before_history_was_known = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                temporal: BiTemporalQuery::as_of(15, 250),
                ..answer_request(
                    "before_history_known",
                    "proyecto responsable",
                    &[],
                    10,
                    0,
                    None,
                )
            },
        )
        .unwrap();
    let historical = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                temporal: BiTemporalQuery::as_of(15, 350),
                ..answer_request("history_known", "proyecto responsable", &[], 10, 0, None)
            },
        )
        .unwrap();
    let current = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                temporal: BiTemporalQuery::as_of(25, 350),
                ..answer_request(
                    "current_after_history",
                    "proyecto responsable",
                    &[],
                    10,
                    0,
                    None,
                )
            },
        )
        .unwrap();

    assert_eq!(
        projected_value(&before_history_was_known, "responsable"),
        None
    );
    assert_eq!(projected_value(&historical, "responsable"), Some("Bea"));
    assert_eq!(projected_value(&current, "responsable"), Some("Ana"));

    store
        .apply_state_mutation_batch(mutation(
            &scope,
            400,
            vec![make_claim(
                &scope,
                "claim_owner_second_late_history",
                "proyecto",
                "responsable",
                Some("Cora"),
                15,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            )],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ))
        .unwrap();
    for (valid_at, expected) in [(12, "Bea"), (17, "Cora"), (25, "Ana")] {
        let projection = store
            .project_answer_ready_state(
                &scope,
                &AnswerReadyStateRequest {
                    temporal: BiTemporalQuery::as_of(valid_at, 450),
                    ..answer_request(
                        "second_late_history",
                        "proyecto responsable",
                        &[],
                        10,
                        0,
                        None,
                    )
                },
            )
            .unwrap();
        assert_eq!(projected_value(&projection, "responsable"), Some(expected));
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn one_batch_materializes_every_valid_time_boundary() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("state_protocol_batch_boundaries");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();

    store
        .apply_state_mutation_batch(mutation(
            &scope,
            100,
            vec![
                make_claim(
                    &scope,
                    "claim_owner_first",
                    "proyecto",
                    "responsable",
                    Some("Ana"),
                    10,
                    (ClaimKind::Fact, ClaimPolarity::Affirmative),
                ),
                make_claim(
                    &scope,
                    "claim_owner_second",
                    "proyecto",
                    "responsable",
                    Some("Bea"),
                    20,
                    (ClaimKind::Fact, ClaimPolarity::Affirmative),
                ),
            ],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ))
        .unwrap();

    for (valid_at, expected) in [(15, "Ana"), (25, "Bea")] {
        let projection = store
            .project_answer_ready_state(
                &scope,
                &AnswerReadyStateRequest {
                    temporal: BiTemporalQuery::as_of(valid_at, 150),
                    ..answer_request("batch_boundaries", "proyecto responsable", &[], 10, 0, None)
                },
            )
            .unwrap();
        assert_eq!(projected_value(&projection, "responsable"), Some(expected));
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn dependency_trace_reconciliation_preserves_other_active_member_lineage() {
    let scope = scope();
    let first = dependency_trace(&scope, "first", "derived_member_first", 10);
    let mut incoming = vec![dependency_trace(
        &scope,
        "second",
        "derived_member_second",
        20,
    )];
    let active_claim_ids = [
        "derived_member_first".to_string(),
        "derived_member_second".to_string(),
    ]
    .into_iter()
    .collect();

    let (historical, superseded) = reconcile_dependency_trace_intervals(
        &mut incoming,
        vec![first],
        &active_claim_ids,
        ProjectionWrite {
            time_ms: 200,
            key: "write_200",
        },
        Some(20),
    );
    assert!(historical.is_empty());
    assert!(superseded.is_empty());
}

#[test]
fn persisted_rule_binds_when_named_endpoint_appears_later() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("state_protocol_pending_rule_binding");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let mut rule = exact_rule(
        &scope,
        "rule_pending_owner_reviewer",
        "proyecto",
        "responsable",
        "revisor",
        RuleAction::DeriveValue,
    );
    rule.valid_from_ms = Some(10);
    store
        .apply_state_mutation_batch(mutation(
            &scope,
            100,
            Vec::new(),
            Vec::new(),
            vec![rule],
            Vec::new(),
        ))
        .unwrap();

    let result = store
        .apply_state_mutation_batch(mutation(
            &scope,
            200,
            vec![make_claim(
                &scope,
                "claim_owner_later",
                "proyecto",
                "responsable",
                Some("Ana"),
                20,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            )],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ))
        .unwrap();
    assert!(result.derived_claims.iter().any(|claim| {
        claim.predicate.as_deref() == Some("revisor")
            && claim.object_value.as_deref() == Some("Ana")
    }));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn persisted_rule_rebinds_through_a_later_evidence_backed_alias() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("state_protocol_pending_rule_alias_binding");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let rule = exact_rule(
        &scope,
        "rule_alias_owner_reviewer",
        "le projet",
        "responsable",
        "revisor",
        RuleAction::DeriveValue,
    );
    store
        .apply_state_mutation_batch(mutation(
            &scope,
            100,
            Vec::new(),
            Vec::new(),
            vec![rule],
            Vec::new(),
        ))
        .unwrap();

    let trigger = make_claim(
        &scope,
        "claim_canonical_owner",
        "projet",
        "responsable",
        Some("Ana"),
        20,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let result = store
        .apply_state_mutation_batch(mutation(
            &scope,
            200,
            vec![trigger.clone()],
            vec![EntityInput {
                id: Some("entity_project".to_string()),
                scope: scope.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                entity_type: "project".to_string(),
                canonical_name: "projet".to_string(),
                aliases: vec!["le projet".to_string()],
                source_claim_ids: vec![trigger.id],
                merge_parent_ids: Vec::new(),
                split_from_id: None,
                confidence: Some(1.0),
            }],
            Vec::new(),
            Vec::new(),
        ))
        .unwrap();
    let current = store.scan_current_claims(&scope, 10, Some(20)).unwrap();
    let canonical_trigger = current
        .iter()
        .find(|claim| claim.id == "claim_canonical_owner")
        .expect("trigger claim should be current");
    assert_eq!(
        canonical_trigger.subject_entity_id.as_deref(),
        Some("entity_project")
    );
    assert_eq!(
        store
            .rules_for_trigger_claim(&scope, canonical_trigger, 10, Some(20))
            .unwrap()
            .len(),
        1
    );
    assert!(result.derived_claims.iter().any(|claim| {
        claim.predicate.as_deref() == Some("revisor")
            && claim.object_value.as_deref() == Some("Ana")
            && claim.subject_entity_id.as_deref() == Some("entity_project")
    }));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn derived_state_uses_the_trigger_rule_interval_intersection() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("state_protocol_rule_interval");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let mut trigger = make_claim(
        &scope,
        "claim_owner_bounded",
        "proyecto",
        "responsable",
        Some("Ana"),
        16,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    trigger.valid_to_ms = Some(30);
    let mut rule = exact_rule(
        &scope,
        "rule_bounded",
        "proyecto",
        "responsable",
        "revisor",
        RuleAction::DeriveValue,
    );
    rule.valid_from_ms = Some(15);
    rule.valid_to_ms = Some(20);
    let result = store
        .apply_state_mutation_batch(mutation(
            &scope,
            100,
            vec![trigger],
            Vec::new(),
            vec![rule],
            Vec::new(),
        ))
        .unwrap();
    let derived = result
        .derived_claims
        .iter()
        .find(|claim| claim.predicate.as_deref() == Some("revisor"))
        .expect("overlapping trigger and rule should derive state");
    assert_eq!(derived.valid_from_ms, Some(16));
    assert_eq!(derived.valid_to_ms, Some(20));

    let during = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                temporal: BiTemporalQuery::as_of(17, 150),
                ..answer_request("rule_interval_during", "proyecto revisor", &[], 10, 0, None)
            },
        )
        .unwrap();
    let after = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                temporal: BiTemporalQuery::as_of(25, 150),
                ..answer_request("rule_interval_after", "proyecto revisor", &[], 10, 0, None)
            },
        )
        .unwrap();
    assert_eq!(projected_value(&during, "revisor"), Some("Ana"));
    assert_eq!(projected_value(&after, "revisor"), None);

    let _ = std::fs::remove_dir_all(&dir);
}

/// `on_change` rules apply to every proven transition, so their trigger slots stay repairable
/// after one has already fired. Only directly asserted state can be the comparison baseline.
#[test]
fn repairable_trigger_slots_track_every_transition_of_directly_asserted_state() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("state_protocol_repairable_triggers");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let claim = |id: &str, subject: &str, predicate: &str, value: &str, at_ms: i64| {
        make_claim(
            &scope,
            id,
            subject,
            predicate,
            Some(value),
            at_ms,
            (ClaimKind::Fact, ClaimPolarity::Affirmative),
        )
    };
    store
        .apply_state_mutation_batch(mutation(
            &scope,
            100,
            vec![
                claim("claim_responsable", "proyecto", "responsable", "Ana", 10),
                claim("claim_revisor", "proyecto", "revisor", "Morgan", 5),
                claim("claim_turno", "equipo", "turno", "mañana", 10),
                claim("claim_calendario", "equipo", "calendario", "estable", 5),
                claim("claim_aviso", "equipo", "aviso", "activo", 5),
                make_claim(
                    &scope,
                    "claim_disponibilidad_incierta",
                    "perfil",
                    "disponibilidad",
                    None,
                    10,
                    (ClaimKind::Fact, ClaimPolarity::Uncertain),
                ),
                claim("claim_agenda", "perfil", "agenda", "pendiente", 5),
            ],
            Vec::new(),
            vec![
                exact_rule(
                    &scope,
                    "rule_responsable_revisor",
                    "proyecto",
                    "responsable",
                    "revisor",
                    RuleAction::MarkUnsupported,
                ),
                exact_rule(
                    &scope,
                    "rule_turno_calendario",
                    "equipo",
                    "turno",
                    "calendario",
                    RuleAction::DeriveValue,
                ),
                exact_rule(
                    &scope,
                    "rule_calendario_aviso",
                    "equipo",
                    "calendario",
                    "aviso",
                    RuleAction::MarkUnsupported,
                ),
                exact_rule(
                    &scope,
                    "rule_disponibilidad_agenda",
                    "perfil",
                    "disponibilidad",
                    "agenda",
                    RuleAction::MarkUnsupported,
                ),
            ],
            Vec::new(),
        ))
        .unwrap();

    let repairable = store
        .repairable_trigger_slots(&scope, usize::MAX, Some(50))
        .unwrap();
    assert_eq!(
        repairable
            .iter()
            .map(|slot| (slot.predicate.as_deref(), slot.current_value.as_deref()))
            .collect::<Vec<_>>(),
        vec![(Some("responsable"), Some("Ana"))],
        "a continuous-projection trigger and a derived-only trigger are not repairable: {repairable:?}"
    );
    // The excluded derived-only slot really is a live `on_change` trigger.
    let derived_trigger = store
        .scan_slots(&scope, usize::MAX, Some(50))
        .unwrap()
        .into_iter()
        .find(|slot| slot.predicate_key == "calendario")
        .expect("the derive rule projects a canonical target slot");
    assert!(store
        .scan_rules(&scope, usize::MAX, Some(50))
        .unwrap()
        .iter()
        .any(|rule| matches!(rule.activation, RuleActivation::OnChange)
            && rule.trigger_slot_id.as_deref() == Some(derived_trigger.slot_key.as_str())));
    assert!(store
        .scan_current_claims_for_slot_ids(
            &scope,
            &BTreeSet::from([derived_trigger.slot_key.clone()]),
            usize::MAX,
            Some(50),
        )
        .unwrap()
        .iter()
        .all(|claim| !crate::is_direct_state_claim(claim)));

    // A proven transition moves the baseline instead of closing the trigger.
    store
        .apply_state_mutation_batch(mutation(
            &scope,
            200,
            vec![claim(
                "claim_responsable_2",
                "proyecto",
                "responsable",
                "Béa",
                20,
            )],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ))
        .unwrap();
    let repairable = store
        .repairable_trigger_slots(&scope, usize::MAX, Some(50))
        .unwrap();
    assert_eq!(repairable.len(), 1);
    assert_eq!(repairable[0].current_value.as_deref(), Some("Béa"));
    assert_eq!(repairable[0].predicate_key, "responsable");

    // Limits and scope-time filtering still bound the scan.
    assert!(store
        .repairable_trigger_slots(&scope, 0, Some(50))
        .unwrap()
        .is_empty());
    assert!(store
        .repairable_trigger_slots(&scope, usize::MAX, Some(-1))
        .unwrap()
        .is_empty());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn derive_dependency_becomes_unsupported_when_its_trigger_is_deleted() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("state_protocol_derive_trigger_deleted");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let trigger = make_claim(
        &scope,
        "claim_owner_before_delete",
        "proyecto",
        "responsable",
        Some("Ana"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    store
        .apply_state_mutation_batch(mutation(
            &scope,
            100,
            vec![
                trigger.clone(),
                make_claim(
                    &scope,
                    "claim_reviewer_before_delete",
                    "proyecto",
                    "revisor",
                    Some("Morgan"),
                    5,
                    (ClaimKind::Fact, ClaimPolarity::Affirmative),
                ),
            ],
            Vec::new(),
            vec![exact_rule(
                &scope,
                "rule_owner_reviewer_delete",
                "proyecto",
                "responsable",
                "revisor",
                RuleAction::DeriveValue,
            )],
            Vec::new(),
        ))
        .unwrap();
    let mut correction = crate::ingest::add_correction(
        CorrectionInput {
            id: Some("correction_owner_deleted".to_string()),
            scope: scope.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            operation: CorrectionOperation::Tombstone,
            target_type: "claim".to_string(),
            target_ids: vec![trigger.id],
            target_selector: None,
            new_value: None,
            reason: None,
            actor: ActorKind::User,
            authority: CorrectionAuthority::User,
            effective_at_ms: Some(20),
            applies_valid_from_ms: None,
            applies_valid_to_ms: None,
            cascade_policy: None,
            metadata_json: None,
        },
        200,
    );
    correction.source_span_ids = vec!["span_owner_deleted".to_string()];
    correction.source_episode_ids = vec!["episode_owner_deleted".to_string()];
    store
        .apply_state_mutation_batch(mutation(
            &scope,
            200,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![correction],
        ))
        .unwrap();

    let projection = store
        .project_answer_ready_state(
            &scope,
            &answer_request(
                "derive_trigger_deleted",
                "proyecto revisor",
                &[],
                10,
                0,
                Some(25),
            ),
        )
        .unwrap();
    assert!(projection.claims.iter().any(|claim| {
        claim.predicate.as_deref() == Some("revisor")
            && claim.polarity == ClaimPolarity::Uncertain
            && claim.object_value.is_none()
    }));
    assert_eq!(projected_value(&projection, "revisor"), None);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn expiring_derive_rule_does_not_resurrect_stale_direct_state() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("state_protocol_derive_rule_expiry");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let mut rule = exact_rule(
        &scope,
        "rule_owner_reviewer_expiring",
        "proyecto",
        "responsable",
        "revisor",
        RuleAction::DeriveValue,
    );
    rule.valid_to_ms = Some(20);
    store
        .apply_state_mutation_batch(mutation(
            &scope,
            100,
            vec![
                make_claim(
                    &scope,
                    "claim_reviewer_before_rule",
                    "proyecto",
                    "revisor",
                    Some("Morgan"),
                    5,
                    (ClaimKind::Fact, ClaimPolarity::Affirmative),
                ),
                make_claim(
                    &scope,
                    "claim_owner_for_expiring_rule",
                    "proyecto",
                    "responsable",
                    Some("Ana"),
                    10,
                    (ClaimKind::Fact, ClaimPolarity::Affirmative),
                ),
            ],
            Vec::new(),
            vec![rule],
            Vec::new(),
        ))
        .unwrap();

    let projection = store
        .project_answer_ready_state(
            &scope,
            &answer_request(
                "derive_rule_expired",
                "proyecto revisor",
                &[],
                10,
                0,
                Some(25),
            ),
        )
        .unwrap();
    assert!(projection.claims.iter().any(|claim| {
        claim.predicate.as_deref() == Some("revisor")
            && claim.polarity == ClaimPolarity::Uncertain
            && claim.object_value.is_none()
    }));
    assert_eq!(projected_value(&projection, "revisor"), None);
    assert!(!projection
        .rules
        .iter()
        .any(|rule| rule.id == "rule_owner_reviewer_expiring"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn propagation_uses_each_trigger_valid_time_instead_of_batch_maximum() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("state_protocol_boundary_local_rules");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let mut rule = exact_rule(
        &scope,
        "rule_temporal_owner_reviewer",
        "proyecto",
        "responsable",
        "revisor",
        RuleAction::DeriveValue,
    );
    rule.valid_from_ms = Some(9);
    rule.valid_to_ms = Some(15);
    let result = store
        .apply_state_mutation_batch(mutation(
            &scope,
            100,
            vec![
                make_claim(
                    &scope,
                    "claim_owner_inside_rule_window",
                    "proyecto",
                    "responsable",
                    Some("Ana"),
                    10,
                    (ClaimKind::Fact, ClaimPolarity::Affirmative),
                ),
                make_claim(
                    &scope,
                    "claim_owner_after_rule_window",
                    "proyecto",
                    "responsable",
                    Some("Bea"),
                    20,
                    (ClaimKind::Fact, ClaimPolarity::Affirmative),
                ),
            ],
            Vec::new(),
            vec![rule],
            Vec::new(),
        ))
        .unwrap();
    assert!(result
        .derived_claims
        .iter()
        .any(|claim| claim.object_value.as_deref() == Some("Ana")));
    assert!(!result
        .derived_claims
        .iter()
        .any(|claim| claim.object_value.as_deref() == Some("Bea")));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn answer_projection_scores_all_visible_state_before_applying_limit() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("state_protocol_complete_candidates");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let mut records = Vec::new();
    for i in 0..9 {
        let predicate = if i == 8 {
            "objetivo".to_string()
        } else {
            format!("irrelevante_{i}")
        };
        let claim = make_claim(
            &scope,
            &format!("claim_candidate_{i}"),
            "proyecto",
            &predicate,
            Some("valor"),
            i as i64,
            (ClaimKind::Fact, ClaimPolarity::Affirmative),
        );
        records.push(state_record_from_claim(claim, &BTreeMap::new(), 100));
    }
    upsert_many(
        &store.state_records,
        records
            .iter()
            .map(|record| state_record_doc(record).unwrap())
            .collect(),
    )
    .unwrap();

    let projection = store
        .project_answer_ready_state(
            &scope,
            &answer_request("complete", "proyecto objetivo", &[], 1, 0, Some(20)),
        )
        .unwrap();
    assert_eq!(projected_value(&projection, "objetivo"), Some("valor"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn state_selection_matches_unsegmented_unicode_queries_without_translation() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("state_protocol_unsegmented_unicode_query");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    store
        .apply_state_mutation_batch(mutation(
            &scope,
            100,
            vec![make_claim(
                &scope,
                "claim_approver_unicode",
                "プロジェクト",
                "承認者",
                Some("愛子"),
                10,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            )],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ))
        .unwrap();

    let projection = store
        .project_answer_ready_state(
            &scope,
            &answer_request(
                "unicode_query",
                "プロジェクトの承認者は誰ですか",
                &[],
                10,
                0,
                Some(20),
            ),
        )
        .unwrap();
    assert_eq!(projected_value(&projection, "承認者"), Some("愛子"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn historical_correction_projects_target_that_expired_before_batch_latest_boundary() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("state_protocol_historical_correction_target");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let mut historical = make_claim(
        &scope,
        "claim_historical_owner",
        "proyecto",
        "responsable",
        Some("Ana"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    historical.valid_to_ms = Some(15);
    store
        .apply_state_mutation_batch(mutation(
            &scope,
            100,
            vec![historical],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ))
        .unwrap();

    let mut correction = crate::ingest::add_correction(
        CorrectionInput {
            id: Some("correction_historical_owner".to_string()),
            scope: scope.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            operation: CorrectionOperation::Tombstone,
            target_type: "claim".to_string(),
            target_ids: vec!["claim_historical_owner".to_string()],
            target_selector: None,
            new_value: None,
            reason: None,
            actor: ActorKind::User,
            authority: CorrectionAuthority::User,
            effective_at_ms: Some(12),
            applies_valid_from_ms: None,
            applies_valid_to_ms: None,
            cascade_policy: None,
            metadata_json: None,
        },
        200,
    );
    correction.source_span_ids = vec!["span_correction".to_string()];
    correction.source_episode_ids = vec!["episode_correction".to_string()];
    store
        .apply_state_mutation_batch(mutation(
            &scope,
            200,
            vec![make_claim(
                &scope,
                "claim_unrelated_later",
                "otro",
                "estado",
                Some("activo"),
                20,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            )],
            Vec::new(),
            Vec::new(),
            vec![correction],
        ))
        .unwrap();

    let projection = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                temporal: BiTemporalQuery::as_of(13, 250),
                ..answer_request(
                    "historical_correction",
                    "proyecto responsable",
                    &[],
                    10,
                    0,
                    None,
                )
            },
        )
        .unwrap();
    assert!(projection
        .claims
        .iter()
        .any(|claim| claim.status == MemoryStatus::Tombstoned));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn alias_backed_set_state_is_projected_without_language_specific_routing() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("state_protocol_alias_set");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let member_a = make_claim(
        &scope,
        "claim_member_a",
        "決済会社",
        "メンバー",
        Some("アリス"),
        10,
        (ClaimKind::Relationship, ClaimPolarity::Affirmative),
    );
    let member_b = make_claim(
        &scope,
        "claim_member_b",
        "Northstar Pay",
        "メンバー",
        Some("ボブ"),
        11,
        (ClaimKind::Relationship, ClaimPolarity::Affirmative),
    );
    store
        .apply_state_mutation_batch(mutation(
            &scope,
            100,
            vec![member_a.clone(), member_b.clone()],
            vec![EntityInput {
                id: Some("entity_northstar".to_string()),
                scope: scope.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                entity_type: "provider".to_string(),
                canonical_name: "Northstar Pay".to_string(),
                aliases: vec!["決済会社".to_string(), "proveedor de pagos".to_string()],
                source_claim_ids: vec![member_a.id.clone(), member_b.id.clone()],
                merge_parent_ids: Vec::new(),
                split_from_id: None,
                confidence: Some(1.0),
            }],
            Vec::new(),
            Vec::new(),
        ))
        .unwrap();

    let projection = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                temporal: BiTemporalQuery::current(),
                ..answer_request("alias_set", "決済会社 メンバー", &[], 20, 10, None)
            },
        )
        .unwrap();
    assert_eq!(projection.set_states.len(), 1);
    assert_eq!(
        projection.set_states[0].members,
        vec!["アリス".to_string(), "ボブ".to_string()]
    );
    assert_eq!(
        projection.set_states[0].subject_entity_id.as_deref(),
        Some("entity_northstar")
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn late_arriving_open_set_member_updates_history_and_current_aggregate() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("state_protocol_late_set_member");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    store
        .apply_state_mutation_batch(mutation(
            &scope,
            200,
            vec![make_claim(
                &scope,
                "claim_member_current",
                "equipo",
                "miembros",
                Some("Ana"),
                20,
                (ClaimKind::Relationship, ClaimPolarity::Affirmative),
            )],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ))
        .unwrap();
    store
        .apply_state_mutation_batch(mutation(
            &scope,
            300,
            vec![make_claim(
                &scope,
                "claim_member_late",
                "equipo",
                "miembros",
                Some("Bea"),
                10,
                (ClaimKind::Relationship, ClaimPolarity::Affirmative),
            )],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ))
        .unwrap();

    let historical = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                temporal: BiTemporalQuery::as_of(15, 350),
                ..answer_request("late_set_history", "equipo miembros", &[], 10, 0, None)
            },
        )
        .unwrap();
    let current = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                temporal: BiTemporalQuery::as_of(25, 350),
                ..answer_request("late_set_current", "equipo miembros", &[], 10, 0, None)
            },
        )
        .unwrap();
    assert_eq!(historical.set_states[0].members, vec!["Bea".to_string()]);
    assert_eq!(
        current.set_states[0].members,
        vec!["Ana".to_string(), "Bea".to_string()]
    );
    let historical_slots = store
        .scan_slots_bitemporal(&scope, 10, BiTemporalQuery::as_of(15, 350))
        .unwrap();
    assert_eq!(historical_slots.len(), 1);
    assert_eq!(
        historical_slots[0].source_claim_ids,
        vec![
            "claim_member_current".to_string(),
            "claim_member_late".to_string()
        ]
    );
    assert!(historical_slots[0].valid_from_ms.is_none());
    assert!(historical_slots[0].valid_to_ms.is_none());
    let slots_before_late_arrival = store
        .scan_slots_bitemporal(&scope, 10, BiTemporalQuery::as_of(15, 250))
        .unwrap();
    assert_eq!(
        slots_before_late_arrival[0].source_claim_ids,
        vec!["claim_member_current".to_string()]
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn deletion_invalidates_dependent_state_and_direct_readd_restores_it() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("state_protocol_delete_readd");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let responsible = make_claim(
        &scope,
        "claim_responsable_initial",
        "servicio",
        "responsable",
        Some("Ana"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let contact = make_claim(
        &scope,
        "claim_contact_initial",
        "servicio",
        "contacto",
        Some("Béa"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let mut rule = exact_rule(
        &scope,
        "rule_responsable_contacto",
        "servicio",
        "responsable",
        "contacto",
        RuleAction::MarkUnsupported,
    );
    rule.value_template = None;
    store
        .apply_state_mutation_batch(mutation(
            &scope,
            100,
            vec![responsible.clone(), contact],
            Vec::new(),
            vec![rule],
            Vec::new(),
        ))
        .unwrap();

    let mut correction = crate::ingest::add_correction(
        CorrectionInput {
            id: Some("correction_responsable_deleted".to_string()),
            scope: scope.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            operation: CorrectionOperation::Tombstone,
            target_type: "claim".to_string(),
            target_ids: vec![responsible.id],
            target_selector: None,
            new_value: None,
            reason: None,
            actor: ActorKind::User,
            authority: CorrectionAuthority::User,
            effective_at_ms: Some(20),
            applies_valid_from_ms: None,
            applies_valid_to_ms: None,
            cascade_policy: None,
            metadata_json: None,
        },
        20,
    );
    correction.source_span_ids = vec!["span_correction_delete".to_string()];
    correction.source_episode_ids = vec!["episode_correction_delete".to_string()];
    let deleted = store
        .apply_state_mutation_batch(mutation(
            &scope,
            200,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![correction],
        ))
        .unwrap();
    assert!(deleted.state_records.iter().any(|record| {
        record.predicate.as_deref() == Some("contacto")
            && matches!(record.state_kind, StateRecordKind::Unsupported)
    }));

    store
        .apply_state_mutation_batch(mutation(
            &scope,
            300,
            vec![
                make_claim(
                    &scope,
                    "claim_responsable_readd",
                    "servicio",
                    "responsable",
                    Some("Carla"),
                    30,
                    (ClaimKind::Fact, ClaimPolarity::Affirmative),
                ),
                make_claim(
                    &scope,
                    "claim_contact_readd",
                    "servicio",
                    "contacto",
                    Some("Dora"),
                    31,
                    (ClaimKind::Fact, ClaimPolarity::Affirmative),
                ),
            ],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ))
        .unwrap();
    let current = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                temporal: BiTemporalQuery::as_of(40, 350),
                ..answer_request("readd", "servicio contacto", &[], 20, 0, None)
            },
        )
        .unwrap();
    assert_eq!(projected_value(&current, "contacto"), Some("Dora"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn unsupported_slot_state_keeps_partial_readd_fail_closed() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("state_protocol_set_unsupported_readd");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let responsible = make_claim(
        &scope,
        "claim_responsable_set_initial",
        "servicio",
        "responsable",
        Some("Ana"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let contact_bea = make_claim(
        &scope,
        "claim_contacto_bea_initial",
        "servicio",
        "contactos",
        Some("Béa"),
        10,
        (ClaimKind::Relationship, ClaimPolarity::Affirmative),
    );
    let contact_ines = make_claim(
        &scope,
        "claim_contacto_ines_initial",
        "servicio",
        "contactos",
        Some("Inés"),
        11,
        (ClaimKind::Relationship, ClaimPolarity::Affirmative),
    );
    let mut rule = exact_rule(
        &scope,
        "rule_responsable_contactos",
        "servicio",
        "responsable",
        "contactos",
        RuleAction::MarkUnsupported,
    );
    rule.value_template = None;
    store
        .apply_state_mutation_batch(mutation(
            &scope,
            100,
            vec![responsible, contact_bea.clone(), contact_ines],
            Vec::new(),
            vec![rule],
            Vec::new(),
        ))
        .unwrap();
    let contacts_slot_id = store.claims_by_ids(&scope, &[contact_bea.id]).unwrap()[0]
        .slot_id
        .clone()
        .unwrap();

    store
        .apply_state_mutation_batch(mutation(
            &scope,
            200,
            vec![make_claim(
                &scope,
                "claim_responsable_set_changed",
                "servicio",
                "responsable",
                Some("Carla"),
                20,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            )],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ))
        .unwrap();

    let unsupported = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                selections: &StateReadSelection::answer_targets(std::slice::from_ref(
                    &contacts_slot_id,
                )),
                ..answer_request(
                    "set-unsupported",
                    "contactos del servicio",
                    &[],
                    20,
                    0,
                    Some(21),
                )
            },
        )
        .unwrap();
    assert_eq!(unsupported.support.state, AnswerSupportState::Unsupported);
    assert_eq!(unsupported.support.slots.len(), 1);
    assert_eq!(unsupported.support.slots[0].slot_id, contacts_slot_id);
    assert_eq!(
        unsupported.support.slots[0].state,
        AnswerSupportState::Unsupported
    );
    assert!(unsupported.set_states.is_empty());
    assert!(unsupported.claims.iter().any(|claim| {
        claim.slot_id.as_ref() == Some(&contacts_slot_id)
            && matches!(claim.polarity, ClaimPolarity::Uncertain)
    }));

    let contact_dora = make_claim(
        &scope,
        "claim_contacto_dora_readd",
        "servicio",
        "contactos",
        Some("Dora"),
        30,
        (ClaimKind::Relationship, ClaimPolarity::Affirmative),
    );
    store
        .apply_state_mutation_batch(mutation(
            &scope,
            300,
            vec![contact_dora.clone()],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ))
        .unwrap();
    let partial = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                selections: &StateReadSelection::answer_targets(std::slice::from_ref(
                    &contacts_slot_id,
                )),
                ..answer_request(
                    "set-partial",
                    "contactos del servicio",
                    &[],
                    20,
                    0,
                    Some(31),
                )
            },
        )
        .unwrap();
    assert_eq!(partial.support.state, AnswerSupportState::Unsupported);
    assert_eq!(partial.support.slots.len(), 1);
    assert_eq!(
        partial.support.slots[0].state,
        AnswerSupportState::Unsupported
    );
    let set_state = partial
        .set_states
        .iter()
        .find(|state| state.slot_id.as_ref() == Some(&contacts_slot_id))
        .expect("post-boundary member remains visible as partial evidence");
    assert_eq!(set_state.members, vec!["Dora"]);
    assert_eq!(set_state.claim_ids, vec![contact_dora.id]);
    assert!(partial.claims.iter().any(|claim| {
        claim.slot_id.as_ref() == Some(&contacts_slot_id)
            && matches!(claim.polarity, ClaimPolarity::Uncertain)
    }));
    let resolved = partial.resolved_answer_slots();
    assert_eq!(resolved.len(), 1);
    assert_eq!(resolved[0].support_state, AnswerSupportState::Unsupported);
    assert!(resolved[0].members.is_empty());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn set_read_over_scalar_history_lists_every_value_the_slot_has_held() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("state_protocol_set_read_scalar_history");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let values = [
        ("claim_hobby_gitarre", "ギター", 10),
        ("claim_hobby_klettern", "クライミング", 20),
        ("claim_hobby_comedy", "コメディ", 30),
    ];
    for (id, value, at) in values {
        store
            .apply_state_mutation_batch(mutation(
                &scope,
                i64::from(at) * 10,
                vec![make_claim(
                    &scope,
                    id,
                    "利用者",
                    "趣味",
                    Some(value),
                    i64::from(at),
                    (ClaimKind::Fact, ClaimPolarity::Affirmative),
                )],
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ))
            .unwrap();
    }
    let slot_id = store
        .claims_by_ids(&scope, &["claim_hobby_gitarre".to_string()])
        .unwrap()[0]
        .slot_id
        .clone()
        .unwrap();

    let projection = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                selections: &[StateReadSelection {
                    slot_id: slot_id.clone(),
                    view: StateReadView::Set,
                    role: StateReadRole::AnswerTarget,
                }],
                ..answer_request(
                    "set_read_history",
                    "利用者の趣味を全部",
                    &[],
                    10,
                    0,
                    Some(40),
                )
            },
        )
        .unwrap();
    let resolved = projection
        .resolved_answer_slots()
        .into_iter()
        .find(|slot| slot.slot_id == slot_id)
        .expect("target slot resolved");
    assert_eq!(resolved.support_state, AnswerSupportState::Supported);
    assert_eq!(resolved.members, vec!["ギター", "クライミング", "コメディ"]);

    // A current read keeps its scalar shape: no synthesized members.
    let current = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                selections: &[StateReadSelection {
                    slot_id: slot_id.clone(),
                    view: StateReadView::Current,
                    role: StateReadRole::AnswerTarget,
                }],
                ..answer_request("current_read_history", "利用者の趣味", &[], 10, 0, Some(40))
            },
        )
        .unwrap();
    let current_resolved = current
        .resolved_answer_slots()
        .into_iter()
        .find(|slot| slot.slot_id == slot_id)
        .unwrap();
    assert!(current_resolved.members.is_empty());
    assert_eq!(current_resolved.current_value.as_deref(), Some("コメディ"));
    assert_eq!(
        store
            .slot_state_version_count(&scope, &slot_id, Some(40))
            .unwrap(),
        3
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn preference_values_accumulate_as_set_members() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("state_protocol_preference_set");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    for (id, value, at) in [
        ("claim_gusta_te", "té verde", 10),
        ("claim_gusta_cafe", "café", 20),
    ] {
        store
            .apply_state_mutation_batch(mutation(
                &scope,
                i64::from(at) * 10,
                vec![{
                    make_claim(
                        &scope,
                        id,
                        "usuario",
                        "bebida preferida",
                        Some(value),
                        i64::from(at),
                        (ClaimKind::Preference, ClaimPolarity::Affirmative),
                    )
                }],
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ))
            .unwrap();
    }
    let states = store
        .scan_state_records(&scope, state_scan(20, Some(30)))
        .unwrap();
    let set_state = states
        .iter()
        .find(|state| matches!(state.state_kind, StateRecordKind::Set))
        .expect("preference slot projects a set state");
    assert!(set_state.members.contains(&"té verde".to_string()));
    assert!(set_state.members.contains(&"café".to_string()));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn standing_dependency_governs_a_transition_stated_before_the_rule() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("state_protocol_retroactive_rule_window");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    // target asserted t=10; trigger holds v1 at t=5 and changes at t=20; rule stated t=30
    let batch = |t, claims, rules| mutation(&scope, t, claims, Vec::new(), rules, Vec::new());
    store
        .apply_state_mutation_batch(batch(
            50,
            vec![
                make_claim(
                    &scope,
                    "claim_ciudad_v1",
                    "usuario",
                    "ciudad",
                    Some("Lima"),
                    5,
                    (ClaimKind::Fact, ClaimPolarity::Affirmative),
                ),
                make_claim(
                    &scope,
                    "claim_gimnasio",
                    "usuario",
                    "gimnasio",
                    Some("Club Andino"),
                    10,
                    (ClaimKind::Fact, ClaimPolarity::Affirmative),
                ),
            ],
            Vec::new(),
        ))
        .unwrap();
    store
        .apply_state_mutation_batch(batch(
            200,
            vec![make_claim(
                &scope,
                "claim_ciudad_v2",
                "usuario",
                "ciudad",
                Some("Cusco"),
                20,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            )],
            Vec::new(),
        ))
        .unwrap();
    let mut rule = exact_rule(
        &scope,
        "rule_gimnasio_ciudad",
        "usuario",
        "ciudad",
        "gimnasio",
        RuleAction::MarkUnsupported,
    );
    rule.valid_from_ms = Some(30);
    store
        .apply_state_mutation_batch(batch(300, Vec::new(), vec![rule]))
        .unwrap();

    let projection = store
        .project_answer_ready_state(
            &scope,
            &answer_request("retro_rule", "usuario gimnasio", &[], 10, 0, Some(40)),
        )
        .unwrap();
    let resolved = projection
        .resolved_answer_slots()
        .into_iter()
        .find(|slot| slot.predicate.as_deref() == Some("gimnasio"))
        .expect("target slot resolved");
    assert_eq!(resolved.support_state, AnswerSupportState::Unsupported);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn rules_bound_to_an_alias_surface_reach_the_canonical_slot() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("state_protocol_alias_rule_reach");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let batch = |t, claims, rules| mutation(&scope, t, claims, Vec::new(), rules, Vec::new());
    store
        .apply_state_mutation_batch(batch(
            100,
            vec![
                make_claim(
                    &scope,
                    "claim_estado_salud",
                    "利用者",
                    "健康状態",
                    Some("良好"),
                    5,
                    (ClaimKind::Fact, ClaimPolarity::Affirmative),
                ),
                make_claim(
                    &scope,
                    "claim_medicacion",
                    "利用者",
                    "薬",
                    Some("A錠"),
                    10,
                    (ClaimKind::Fact, ClaimPolarity::Affirmative),
                ),
            ],
            Vec::new(),
        ))
        .unwrap();
    let med_slot = store
        .claims_by_ids(&scope, &["claim_medicacion".to_string()])
        .unwrap()[0]
        .slot_id
        .clone()
        .unwrap();
    store
        .add_slot_alias(
            SlotAliasInput {
                id: Some("alias_medicamento".to_string()),
                scope: scope.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                alias_subject: "利用者".to_string(),
                alias_predicate: "服用薬".to_string(),
                canonical_slot_id: Some(med_slot.clone()),
                target_claim_id: None,
                source_claim_ids: vec!["claim_medicacion".to_string()],
                valid_from_ms: Some(10),
                valid_to_ms: None,
            },
            10,
        )
        .unwrap();
    // rule targets the ALIAS surface, trigger changes after the rule
    let mut rule = exact_rule(
        &scope,
        "rule_salud_medicacion",
        "利用者",
        "健康状態",
        "服用薬",
        RuleAction::MarkUnsupported,
    );
    rule.valid_from_ms = Some(12);
    store
        .apply_state_mutation_batch(batch(200, Vec::new(), vec![rule]))
        .unwrap();
    store
        .apply_state_mutation_batch(batch(
            300,
            vec![make_claim(
                &scope,
                "claim_estado_salud_v2",
                "利用者",
                "健康状態",
                Some("高血圧"),
                20,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            )],
            Vec::new(),
        ))
        .unwrap();

    let projection = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                selections: &[StateReadSelection {
                    slot_id: med_slot.clone(),
                    view: StateReadView::Current,
                    role: StateReadRole::AnswerTarget,
                }],
                ..answer_request("alias_rule_reach", "利用者の薬", &[], 10, 0, Some(40))
            },
        )
        .unwrap();
    let resolved = projection
        .resolved_answer_slots()
        .into_iter()
        .find(|slot| slot.slot_id == med_slot)
        .expect("target slot resolved");
    assert_eq!(resolved.support_state, AnswerSupportState::Unsupported);
    let _ = std::fs::remove_dir_all(dir);
}

// One claim and one derived rule leave claims, state records and a dependency trace on slots.
fn store_with_derived_state(name: &str) -> (EvidencedStore, MemoryScope) {
    let store = EvidencedStore::create(&temp_dir(name), 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let claim = make_claim(
        &scope,
        "claim_decode",
        "servicio",
        "responsable",
        Some("Ana"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let rule = exact_rule(
        &scope,
        "rule_decode",
        "servicio",
        "responsable",
        "contacto",
        RuleAction::DeriveValue,
    );
    store
        .apply_state_mutation_batch(mutation(
            &scope,
            100,
            vec![claim],
            Vec::new(),
            vec![rule],
            Vec::new(),
        ))
        .unwrap();
    (store, scope)
}

fn corrupt_stored_row(collection: &dyn MemoryTable, pk: &str) {
    let stored = collection.fetch(vec![pk.to_string()]).unwrap()[pk].clone();
    let corrupt = (*stored).clone().set("status", "not_a_status");
    assert!(collection.upsert(vec![corrupt]).unwrap()[0].is_ok());
}

#[test]
fn state_record_scans_fail_on_undecodable_row() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let (store, scope) = store_with_derived_state("state_record_decode_error");
    let record = store
        .scan_state_records(&scope, state_scan(10, None))
        .unwrap()
        .into_iter()
        .find(|record| record.slot_id.is_some())
        .unwrap();
    let slot_id = record.slot_id.clone().unwrap();
    let slot_ids = BTreeSet::from([slot_id.clone()]);
    corrupt_stored_row(&store.state_records, &record.id);

    let temporal = BiTemporalQuery::default();
    assert!(store
        .scan_state_records_bitemporal_for_slot_ids(&scope, &slot_ids, 10, temporal)
        .is_err());
    assert!(store
        .slot_state_version_count(&scope, &slot_id, None)
        .is_err());
    assert!(store
        .current_state_records_for_slot_ids(&scope, &slot_ids)
        .is_err());

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn dependency_trace_scan_fails_on_undecodable_row() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let (store, scope) = store_with_derived_state("dependency_trace_decode_error");
    let trace = store
        .scan_dependency_traces(&scope, 10, None)
        .unwrap()
        .into_iter()
        .find(|trace| trace.target_slot_id.is_some())
        .unwrap();
    let slot_ids = BTreeSet::from([trace.target_slot_id.clone().unwrap()]);
    corrupt_stored_row(&store.dependency_traces, &trace.id);

    assert!(store
        .current_dependency_traces_for_slot_ids(&scope, &slot_ids)
        .is_err());

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn claim_scans_fail_on_undecodable_row() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let (store, scope) = store_with_derived_state("claim_decode_error");
    let claim = store
        .scan_claims(&scope, 10, None)
        .unwrap()
        .into_iter()
        .find(|claim| claim.id == "claim_decode")
        .unwrap();
    let slot_ids = BTreeSet::from([claim.slot_id.clone().unwrap()]);
    let subjects = BTreeSet::from([(claim.subject_entity_id.clone(), "servicio".to_string())]);
    corrupt_stored_row(&store.claims, &claim.id);

    assert!(store
        .scan_claim_versions_for_slot_ids(&scope, &slot_ids, 10, None)
        .is_err());
    assert!(store
        .scan_current_claims_for_subjects(&scope, &subjects, 10, None)
        .is_err());

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn scope_rebuild_does_not_retire_a_narrower_scopes_state() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("rebuild_keeps_tenant_state");
    let store = MemoryStore::create(&path, 3, CollectionOptions::default()).unwrap();
    let mut tenant = scope();
    tenant.tenant_id = Some("tenant_a".to_string());
    let mut claim = make_claim(
        &tenant,
        "tenant_claim",
        "project",
        "owner",
        Some("private"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    claim.source_span_ids.clear();
    claim.source_episode_ids.clear();
    store.append_claim(&claim, None).unwrap();

    // At valid time 5 the tenant claim does not hold yet, so its slot is outside the frontier.
    store.refresh_state_projection(&scope(), Some(5)).unwrap();
    let tenant_states = store
        .scan_state_records(&tenant, state_scan(10, Some(30)))
        .unwrap();

    drop(store);
    std::fs::remove_dir_all(path).unwrap();
    assert_eq!(tenant_states.len(), 1);
}

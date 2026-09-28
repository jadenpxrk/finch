use super::*;
use crate::store::lifecycle::materialized_rule_value;
use crate::store::state_records::state_record_from_claim;

#[test]
fn fixed_rule_value_takes_precedence_over_copy_template() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("fixed_rule_value_precedence");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let mut input = make_derived_rule_inputs();
    input.value = Some("resultado fijo".to_string());
    input.value_template = Some("{value}".to_string());
    let rule = store.add_rule(input).unwrap();
    let trigger = make_claim(
        &scope(),
        "claim_trigger_value",
        "proyecto",
        "responsable",
        Some("valor del disparador"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );

    let (value, polarity, _) = materialized_rule_value(&rule, &trigger, false)
        .unwrap()
        .unwrap();
    assert_eq!(value.as_deref(), Some("resultado fijo"));
    assert_eq!(polarity, ClaimPolarity::Affirmative);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn equal_time_lifecycle_state_has_one_authoritative_winner() {
    let scope = scope();
    let direct = make_claim(
        &scope,
        "z_direct",
        "servicio",
        "contacto",
        Some("Ana"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let mut derived = direct.clone();
    derived.id = "a_derived".to_string();
    derived.object_value = Some("Béa".to_string());
    derived.asserted_by = "extractor_derived".to_string();
    let mut unsupported = direct.clone();
    unsupported.id = "a_unsupported".to_string();
    unsupported.object_value = None;
    unsupported.polarity = ClaimPolarity::Uncertain;
    let mut tombstone = direct.clone();
    tombstone.id = "a_tombstone".to_string();
    tombstone.object_value = None;
    tombstone.status = MemoryStatus::Tombstoned;

    let current = resolve_current_claims(
        vec![direct, derived, unsupported, tombstone.clone()],
        usize::MAX,
    );
    assert_eq!(current.len(), 1);
    assert_eq!(current[0].id, tombstone.id);

    let later = make_claim(
        &scope,
        "later_direct",
        "servicio",
        "contacto",
        Some("Clara"),
        20,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let restored = resolve_current_claims(vec![tombstone, later.clone()], usize::MAX);
    assert_eq!(restored.len(), 1);
    assert_eq!(restored[0].id, later.id);
}

#[test]
fn retracted_claim_projects_closed_tombstone_state() {
    let scope = scope();
    let claim = make_claim(
        &scope,
        "claim_retracted",
        "service",
        "endpoint",
        Some("legacy"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let mut correction = crate::ingest::add_correction(
        CorrectionInput {
            id: Some("correction_retracted".to_string()),
            scope,
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            operation: CorrectionOperation::Retract,
            target_type: "claim".to_string(),
            target_ids: vec![claim.id.clone()],
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
    correction.source_span_ids = vec!["span_retraction".to_string()];
    correction.source_episode_ids = vec!["episode_retraction".to_string()];

    let corrected = apply_corrections_to_claims(vec![claim], &[correction], Some(20));
    assert_eq!(corrected[0].status, MemoryStatus::Retracted);
    assert_eq!(
        state_record_from_claim(corrected[0].clone(), &BTreeMap::new(), 20).state_kind,
        StateRecordKind::Tombstone
    );
    assert_eq!(
        support_contract_for_claims(&corrected, &[] as &[SetStateRecord]).state,
        AnswerSupportState::Deleted
    );
}

#[test]
fn canonical_slot_correction_suppresses_all_earlier_versions_and_allows_later_evidence() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("canonical_slot_correction_lifecycle");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let first = make_claim(
        &scope,
        "service_contact_first",
        "servicio",
        "contacto",
        Some("Ana"),
        5,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let second = make_claim(
        &scope,
        "service_contact_second",
        "servicio",
        "contacto",
        Some("Bea"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    store.append_claim(&first, None).unwrap();
    store.append_claim(&second, None).unwrap();
    let slot_id = store
        .scan_slots(&scope, usize::MAX, Some(10))
        .unwrap()
        .into_iter()
        .find(|slot| slot.predicate.as_deref() == Some("contacto"))
        .unwrap()
        .slot_key;
    let mut correction = crate::ingest::add_correction(
        CorrectionInput {
            id: Some("delete_service_contact_slot".to_string()),
            scope: scope.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            operation: CorrectionOperation::Tombstone,
            target_type: "claim".to_string(),
            target_ids: vec![second.id.clone()],
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
    correction.target_match = CorrectionTargetMatch::CanonicalSlot;
    correction.target_subject_key = Some(canonical_slot_part("servicio"));
    correction.target_predicate_key = Some(canonical_slot_part("contacto"));
    correction.target_slot_id = Some(slot_id.clone());
    store.add_correction(&correction).unwrap();

    let deleted = store
        .scan_state_records(&scope, state_scan(20, Some(20)))
        .unwrap();
    assert!(deleted.iter().any(|state| {
        state.slot_id.as_ref() == Some(&slot_id)
            && matches!(state.state_kind, StateRecordKind::Tombstone)
    }));
    assert!(!deleted.iter().any(|state| {
        state.slot_id.as_ref() == Some(&slot_id)
            && matches!(state.state_kind, StateRecordKind::Current)
    }));

    let current_read = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                selections: &[StateReadSelection {
                    slot_id: slot_id.clone(),
                    view: StateReadView::Current,
                    role: StateReadRole::AnswerTarget,
                }],
                ..answer_request("contact-current", "現在の連絡先", &[], 20, 20, Some(20))
            },
        )
        .unwrap();
    assert_eq!(current_read.support.slots[0].view, StateReadView::Current);
    assert_eq!(
        current_read.support.slots[0].state,
        AnswerSupportState::Deleted
    );
    let current_context =
        build_answer_ready_state_context(&[], &current_read, &[], &[], ContextOptions::default());
    assert!(current_context.body.contains("read_view: current"));
    assert!(current_context.body.contains("answer_disposition: refuse"));

    let timeline_read = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                selections: &[StateReadSelection {
                    slot_id: slot_id.clone(),
                    view: StateReadView::Timeline,
                    role: StateReadRole::AnswerTarget,
                }],
                ..answer_request("contact-timeline", "連絡先の履歴", &[], 20, 20, Some(20))
            },
        )
        .unwrap();
    assert_eq!(timeline_read.support.slots[0].view, StateReadView::Timeline);
    assert_eq!(
        timeline_read.support.slots[0].state,
        AnswerSupportState::Supported
    );
    let timeline_context =
        build_answer_ready_state_context(&[], &timeline_read, &[], &[], ContextOptions::default());
    assert!(timeline_context.body.contains("read_view: timeline"));
    assert!(timeline_context.body.contains("answer_disposition: answer"));
    assert!(timeline_context
        .body
        .contains("authoritative_current_value: false"));

    let restored = make_claim(
        &scope,
        "service_contact_restored",
        "servicio",
        "contacto",
        Some("Carla"),
        30,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    store.append_claim(&restored, None).unwrap();
    let current = store
        .scan_state_records(&scope, state_scan(20, Some(30)))
        .unwrap();
    assert!(current.iter().any(|state| {
        state.slot_id.as_ref() == Some(&slot_id)
            && matches!(state.state_kind, StateRecordKind::Current)
            && state.object_value.as_deref() == Some("Carla")
    }));
    assert!(!current.iter().any(|state| {
        state.slot_id.as_ref() == Some(&slot_id)
            && matches!(state.state_kind, StateRecordKind::Tombstone)
    }));

    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn structured_correction_binds_to_a_grounded_claim_in_the_same_batch() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("pending_claim_correction_target");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let pending = make_claim(
        &scope,
        "pending_sleep_schedule",
        "사용자",
        "수면 일정",
        Some("교대 근무에 따라 달라짐"),
        20,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let mut correction = crate::ingest::add_correction(
        CorrectionInput {
            id: Some("remove_sleep_schedule".to_string()),
            scope,
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            operation: CorrectionOperation::Tombstone,
            target_type: "claim".to_string(),
            target_ids: Vec::new(),
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
    correction.target_match = CorrectionTargetMatch::CanonicalSlot;
    correction.target_subject_key = Some(canonical_slot_part("사용자"));
    correction.target_predicate_key = Some(canonical_slot_part("수면 일정"));
    correction.source_span_ids = vec!["span_remove_sleep_schedule".to_string()];

    assert!(store
        .resolve_existing_correction_target(&correction)
        .is_err());
    let resolved = store
        .resolve_correction_target_with_pending_claims(&correction, &[pending])
        .unwrap();
    let expected_slot_id = canonical_slot_id(
        None,
        &canonical_slot_part("사용자"),
        &canonical_slot_part("수면 일정"),
    );
    assert_eq!(
        resolved.target_slot_id.as_deref(),
        Some(expected_slot_id.as_str())
    );

    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn answer_ready_projection_does_not_admit_unrelated_same_subject_slots() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("answer_ready_slot_selection");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let owner = make_claim(
        &scope,
        "claim_project_owner_selection",
        "project",
        "owner",
        Some("Mina"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let billing = make_claim(
        &scope,
        "claim_project_billing_selection",
        "project",
        "billing_contact",
        Some("Noah"),
        11,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    store.append_claim(&owner, None).unwrap();
    store.append_claim(&billing, None).unwrap();
    let hits = vec![SpanSearchHit {
        span: span_record(
            "span_claim_project_owner_selection",
            MemoryStatus::Active,
            Some(10),
        ),
        score: 1.0,
    }];

    let projection = store
        .project_answer_ready_state(
            &scope,
            &answer_request("selection", "project owner", &hits, 20, 0, Some(20)),
        )
        .unwrap();

    assert!(projection.claims.iter().any(|claim| claim.id == owner.id));
    assert!(!projection.claims.iter().any(|claim| claim.id == billing.id));

    let billing_slot_id = store
        .scan_state_records(&scope, state_scan(20, Some(20)))
        .unwrap()
        .into_iter()
        .find(|record| record.predicate.as_deref() == Some("billing_contact"))
        .and_then(|record| record.slot_id)
        .unwrap();
    let selected = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                selections: &StateReadSelection::answer_targets(&[billing_slot_id]),
                ..answer_request("selection", "project owner", &hits, 20, 0, Some(20))
            },
        )
        .unwrap();
    assert!(selected.claims.iter().any(|claim| claim.id == billing.id));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn retrieved_history_projects_current_state_for_the_same_canonical_slot() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("historical_evidence_current_state");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let previous = make_claim(
        &scope,
        "previous_owner",
        "proyecto",
        "propietaria",
        Some("Ana"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let current = make_claim(
        &scope,
        "current_owner",
        "proyecto",
        "propietaria",
        Some("Bea"),
        20,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    store.append_claim(&previous, None).unwrap();
    store.append_claim(&current, None).unwrap();

    let projection = store
        .project_answer_ready_state(
            &scope,
            &answer_request(
                "historical-evidence",
                "誰ですか？",
                &[SpanSearchHit {
                    span: {
                        let mut span = span_record("sibling_span", MemoryStatus::Active, Some(10));
                        span.source_id = "ep_previous_owner".to_string();
                        span
                    },
                    score: 1.0,
                }],
                20,
                0,
                Some(30),
            ),
        )
        .unwrap();

    assert!(projection.claims.iter().any(|claim| claim.id == current.id));
    assert!(!projection
        .claims
        .iter()
        .any(|claim| claim.id == previous.id));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn atomic_claim_embedding_projects_current_state_for_a_matched_historical_claim() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("atomic_claim_current_state");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let previous = make_claim(
        &scope,
        "previous_owner_vector",
        "proyecto",
        "propietaria",
        Some("Ana"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let current = make_claim(
        &scope,
        "current_owner_vector",
        "proyecto",
        "propietaria",
        Some("Bea"),
        20,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let claim_embeddings = BTreeMap::from([
        (previous.id.clone(), vec![1.0, 0.0, 0.0]),
        (current.id.clone(), vec![0.0, 1.0, 0.0]),
    ]);
    store
        .apply_state_mutation_batch_with_claim_embeddings(
            StateMutationBatch {
                scope: scope.clone(),
                transaction_time_ms: 30,
                propagation_seed: "atomic-claim".to_string(),
                max_rule_hops: 8,
                claims: vec![previous.clone(), current.clone()],
                entities: Vec::new(),
                rules: Vec::new(),
                slot_aliases: Vec::new(),
                corrections: Vec::new(),
            },
            claim_embeddings,
        )
        .unwrap();

    let selected_slot_ids = store
        .query_claim_slot_ids(&scope, vec![1.0, 0.0, 0.0], 1, Some(30))
        .unwrap();
    let projection = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                selections: &StateReadSelection::answer_targets(&selected_slot_ids),
                ..answer_request("atomic-claim", "誰ですか？", &[], 20, 0, Some(30))
            },
        )
        .unwrap();

    assert_eq!(selected_slot_ids.len(), 1);
    assert!(projection.claims.iter().any(|claim| claim.id == current.id));
    assert!(!projection
        .claims
        .iter()
        .any(|claim| claim.id == previous.id));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn grounded_claims_inherit_source_span_embeddings_for_slot_selection() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("grounded_claim_source_embedding");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let owner = make_claim(
        &scope,
        "source_embedded_owner",
        "proyecto",
        "responsable",
        Some("Ana"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let region = make_claim(
        &scope,
        "source_embedded_region",
        "proyecto",
        "región",
        Some("sur"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    for (claim, embedding) in [
        (&owner, vec![1.0, 0.0, 0.0]),
        (&region, vec![0.0, 1.0, 0.0]),
    ] {
        let mut span = span_record(&claim.source_span_ids[0], MemoryStatus::Active, Some(10));
        span.scope = scope.clone();
        span.source_id = claim.source_episode_ids[0].clone();
        store.append_span(&span, Some(&embedding)).unwrap();
    }
    store
        .apply_state_mutation_batch(StateMutationBatch {
            scope: scope.clone(),
            transaction_time_ms: 20,
            propagation_seed: "grounded-claim-source-embedding".to_string(),
            max_rule_hops: 8,
            claims: vec![owner.clone(), region],
            entities: Vec::new(),
            rules: Vec::new(),
            slot_aliases: Vec::new(),
            corrections: Vec::new(),
        })
        .unwrap();

    let selected = store
        .query_claim_slot_ids(&scope, vec![1.0, 0.0, 0.0], 1, Some(20))
        .unwrap();

    assert_eq!(selected.len(), 1);
    let selected_slot = store
        .scan_slots(&scope, usize::MAX, Some(20))
        .unwrap()
        .into_iter()
        .find(|slot| slot.slot_key == selected[0])
        .unwrap();
    assert_eq!(selected_slot.predicate.as_deref(), Some("responsable"));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn claim_vector_retrieval_applies_limit_after_canonical_slot_deduplication() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("claim_vector_canonical_slot_limit");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let mut claims = (0..8)
        .map(|version| {
            make_claim(
                &scope,
                &format!("owner_version_{version}"),
                "project",
                "owner",
                Some(&format!("owner {version}")),
                10 + version,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            )
        })
        .collect::<Vec<_>>();
    claims.push(make_claim(
        &scope,
        "project_region",
        "project",
        "region",
        Some("north"),
        20,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    ));
    let embeddings = claims
        .iter()
        .map(|claim| {
            let embedding = if claim.id == "project_region" {
                vec![0.8, 0.2, 0.0]
            } else {
                vec![1.0, 0.0, 0.0]
            };
            (claim.id.clone(), embedding)
        })
        .collect::<BTreeMap<_, _>>();
    store
        .apply_state_mutation_batch_with_claim_embeddings(
            StateMutationBatch {
                scope: scope.clone(),
                transaction_time_ms: 30,
                propagation_seed: "claim-vector-slot-limit".to_string(),
                max_rule_hops: 8,
                claims,
                entities: Vec::new(),
                rules: Vec::new(),
                slot_aliases: Vec::new(),
                corrections: Vec::new(),
            },
            embeddings,
        )
        .unwrap();

    let selected = store
        .query_claim_slot_ids(&scope, vec![1.0, 0.0, 0.0], 2, Some(30))
        .unwrap();

    assert_eq!(selected.len(), 2);
    let predicates = store
        .scan_slots(&scope, usize::MAX, None)
        .unwrap()
        .into_iter()
        .filter(|slot| selected.contains(&slot.slot_key))
        .map(|slot| slot.predicate_key)
        .collect::<BTreeSet<_>>();
    assert_eq!(
        predicates,
        BTreeSet::from(["owner".to_string(), "region".to_string()])
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn current_state_retrieval_admits_unembedded_derived_slots_by_slot_identity() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("current_state_lexical_derived_slot");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let owner = make_claim(
        &scope,
        "claim_responsable_vector",
        "proyecto",
        "responsable",
        Some("Ana"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    store.append_claim(&owner, Some(&[1.0, 0.0, 0.0])).unwrap();

    let mut standing = make_derived_rule_inputs();
    standing.id = Some("rule_revisor_sigue_responsable_retrieval".to_string());
    standing.trigger_subject_key = "proyecto".to_string();
    standing.trigger_predicate_key = "responsable".to_string();
    standing.target_subject_key = "proyecto".to_string();
    standing.target_predicate_key = "revisor".to_string();
    standing.trigger_subject = Some("proyecto".to_string());
    standing.trigger_predicate = Some("responsable".to_string());
    standing.target_subject = Some("proyecto".to_string());
    standing.target_predicate = Some("revisor".to_string());
    standing.activation = RuleActivation::ContinuousProjection;
    standing.valid_from_ms = Some(20);
    store.add_rule(standing).unwrap();

    let vector_only = store
        .query_claim_slot_ids(&scope, vec![1.0, 0.0, 0.0], 2, Some(20))
        .unwrap();
    let current = store
        .query_current_state_slot_ids(
            &scope,
            vec![1.0, 0.0, 0.0],
            "quién es el revisor",
            2,
            Some(20),
        )
        .unwrap();
    let slot_predicate = |slot_id: &str| {
        store
            .scan_slots(&scope, usize::MAX, Some(20))
            .unwrap()
            .into_iter()
            .find(|slot| slot.slot_key == slot_id)
            .and_then(|slot| slot.predicate)
    };
    assert!(vector_only
        .iter()
        .any(|slot_id| { slot_predicate(slot_id).as_deref() == Some("responsable") }));
    assert!(!vector_only
        .iter()
        .any(|slot_id| { slot_predicate(slot_id).as_deref() == Some("revisor") }));
    assert!(current
        .iter()
        .any(|slot_id| { slot_predicate(slot_id).as_deref() == Some("revisor") }));

    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn current_state_retrieval_ranks_paraphrased_value_over_vector_neighbor() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("current_state_paraphrase_overlap");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let neighbor = make_claim(
        &scope,
        "claim_auth_vecina",
        "cuenta",
        "autenticacion",
        Some("autenticación en dos pasos"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let target = make_claim(
        &scope,
        "claim_gestor_contrasenas",
        "cuenta",
        "almacen_secretos",
        Some("gestor de contraseñas"),
        20,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    store
        .append_claim(&neighbor, Some(&[1.0, 0.0, 0.0]))
        .unwrap();
    store.append_claim(&target, Some(&[0.0, 1.0, 0.0])).unwrap();

    let vector_only = store
        .query_claim_slot_ids(&scope, vec![1.0, 0.0, 0.0], 1, Some(20))
        .unwrap();
    let current = store
        .query_current_state_slot_ids(
            &scope,
            vec![1.0, 0.0, 0.0],
            "cómo mejoré la seguridad de las contraseñas",
            2,
            Some(20),
        )
        .unwrap();
    let slot_value = |slot_id: &str| {
        store
            .scan_state_records(&scope, state_scan(usize::MAX, Some(20)))
            .unwrap()
            .into_iter()
            .find(|record| record.slot_id.as_deref() == Some(slot_id))
            .and_then(|record| record.object_value)
    };
    assert_eq!(
        slot_value(&vector_only[0]).as_deref(),
        Some("autenticación en dos pasos")
    );
    assert!(current
        .iter()
        .any(|slot_id| { slot_value(slot_id).as_deref() == Some("gestor de contraseñas") }));
    assert_eq!(
        slot_value(&current[0]).as_deref(),
        Some("gestor de contraseñas")
    );

    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn projection_trace_separates_candidate_filtering_from_preserved_history() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("projection_trace_history");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    for (id, value, at) in [
        ("responsable_primero", "Ana", 10),
        ("responsable_segundo", "Bea", 20),
        ("responsable_tercero", "Carla", 30),
    ] {
        store
            .append_claim(
                &make_claim(
                    &scope,
                    id,
                    "proyecto",
                    "responsable",
                    Some(value),
                    at,
                    (ClaimKind::Fact, ClaimPolarity::Affirmative),
                ),
                None,
            )
            .unwrap();
    }
    store
        .append_claim(
            &make_claim(
                &scope,
                "proyecto_region",
                "proyecto",
                "región",
                Some("norte"),
                30,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            ),
            None,
        )
        .unwrap();
    let slots = store.scan_slots(&scope, usize::MAX, Some(30)).unwrap();
    let owner_slot = slots
        .iter()
        .find(|slot| slot.predicate.as_deref() == Some("responsable"))
        .unwrap()
        .slot_key
        .clone();
    let region_slot = slots
        .iter()
        .find(|slot| slot.predicate.as_deref() == Some("región"))
        .unwrap()
        .slot_key
        .clone();
    let current_projection = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                selections: &StateReadSelection::answer_targets(std::slice::from_ref(&owner_slot)),
                ..answer_request(
                    "estado-actual",
                    "¿Quién está a cargo?",
                    &[],
                    20,
                    20,
                    Some(30),
                )
            },
        )
        .unwrap();
    // A current read keeps its Current support view; the version history rides along as
    // non-authoritative context so supersession is visible rather than implied.
    assert_eq!(
        current_projection.support.slots[0].view,
        StateReadView::Current
    );
    assert!(current_projection
        .slot_histories
        .iter()
        .any(|history| history.slot_id == owner_slot && history.versions.len() == 3));

    let projection = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                selections: &[StateReadSelection {
                    slot_id: owner_slot.clone(),
                    view: StateReadView::Timeline,
                    role: StateReadRole::AnswerTarget,
                }],
                ..answer_request("traza", "¿Quién estuvo a cargo?", &[], 20, 20, Some(30))
            },
        )
        .unwrap();
    let projected_history = projection
        .slot_histories
        .iter()
        .find(|history| history.slot_id == owner_slot)
        .unwrap();
    assert_eq!(projection.support.slots[0].view, StateReadView::Timeline);
    assert_eq!(projected_history.versions.len(), 3);
    assert_eq!(
        projected_history
            .versions
            .iter()
            .filter_map(|version| version.object_value.as_deref())
            .collect::<Vec<_>>(),
        vec!["Ana", "Bea", "Carla"]
    );
    let context = build_answer_ready_state_context(
        &[],
        &projection,
        &[],
        &[],
        ContextOptions {
            token_budget: 1_000,
            include_provenance: true,
        },
    );
    let current_position = context.body.find("### ResolvedAnswerState").unwrap();
    let history_position = context.body.find("### SlotHistory").unwrap();
    assert!(current_position < history_position);
    let history_body = &context.body[history_position..];
    assert!(history_body.find("Ana").unwrap() < history_body.find("Bea").unwrap());
    assert!(history_body.find("Bea").unwrap() < history_body.find("Carla").unwrap());

    let historical_projection = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                selections: &[StateReadSelection {
                    slot_id: owner_slot.clone(),
                    view: StateReadView::Timeline,
                    role: StateReadRole::AnswerTarget,
                }],
                ..answer_request(
                    "traza-histórica",
                    "¿Quién estuvo a cargo?",
                    &[],
                    20,
                    20,
                    Some(20),
                )
            },
        )
        .unwrap();
    let historical_values = historical_projection.slot_histories[0]
        .versions
        .iter()
        .filter_map(|version| version.object_value.as_deref())
        .collect::<Vec<_>>();
    assert_eq!(historical_values, vec!["Ana", "Bea"]);
    let trace = store
        .explain_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                selections: &StateReadSelection::answer_targets(std::slice::from_ref(&owner_slot)),
                ..answer_request("", "¿Quién estuvo a cargo?", &[], 20, 20, Some(30))
            },
            &projection,
        )
        .unwrap();

    let history = trace
        .slot_histories
        .iter()
        .find(|history| history.slot_id == owner_slot)
        .unwrap();
    assert_eq!(history.version_count, 3);
    assert_eq!(history.distinct_value_count, 3);
    assert_eq!(history.first_valid_from_ms, Some(10));
    assert_eq!(history.last_valid_from_ms, Some(30));
    assert!(trace.candidates.iter().any(|candidate| {
        candidate.slot_id.as_ref() == Some(&owner_slot)
            && matches!(candidate.disposition, StateSelectionDisposition::Selected)
    }));
    assert!(trace.candidates.iter().any(|candidate| {
        candidate.slot_id.as_ref() == Some(&region_slot)
            && matches!(
                candidate.disposition,
                StateSelectionDisposition::CandidateFiltered
            )
    }));

    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn target_and_coverage_projection_keeps_coverage_without_equal_target_authority() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("target_and_coverage_projection");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    for (id, predicate, value) in [
        ("proyecto_responsable", "responsable", "Ana"),
        ("proyecto_region", "región", "norte"),
    ] {
        store
            .append_claim(
                &make_claim(
                    &scope,
                    id,
                    "proyecto",
                    predicate,
                    Some(value),
                    10,
                    (ClaimKind::Fact, ClaimPolarity::Affirmative),
                ),
                None,
            )
            .unwrap();
    }
    let slots = store.scan_slots(&scope, usize::MAX, Some(10)).unwrap();
    let owner_slot = slots
        .iter()
        .find(|slot| slot.predicate.as_deref() == Some("responsable"))
        .unwrap()
        .slot_key
        .clone();
    let region_slot = slots
        .iter()
        .find(|slot| slot.predicate.as_deref() == Some("región"))
        .unwrap()
        .slot_key
        .clone();
    let projection = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                selections: &[StateReadSelection {
                    slot_id: region_slot.clone(),
                    view: StateReadView::Current,
                    role: StateReadRole::Supporting,
                }],
                target_selections: Some(&[StateReadSelection {
                    slot_id: owner_slot.clone(),
                    view: StateReadView::Current,
                    role: StateReadRole::AnswerTarget,
                }]),
                ..answer_request(
                    "objetivo-cobertura",
                    "¿Quién está a cargo?",
                    &[],
                    20,
                    20,
                    Some(10),
                )
            },
        )
        .unwrap();

    assert!(projection
        .claims
        .iter()
        .any(|claim| claim.object_value.as_deref() == Some("Ana")));
    assert!(projection
        .claims
        .iter()
        .any(|claim| claim.object_value.as_deref() == Some("norte")));
    assert_eq!(projection.support.slots.len(), 1);
    assert_eq!(projection.support.slots[0].slot_id, owner_slot);

    let context = build_answer_ready_state_context(
        &[],
        &projection,
        &[],
        &[],
        ContextOptions {
            token_budget: 1_000,
            include_provenance: true,
        },
    );
    assert_eq!(context.body.matches("read_role: answer_target").count(), 1);
    assert!(!context.body.contains("\n\nread_role: answer_target"));
    assert!(context.body.find("Ana").unwrap() < context.body.find("norte").unwrap());

    let trace = store
        .explain_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                selections: &StateReadSelection::answer_targets(std::slice::from_ref(&region_slot)),
                target_selections: Some(&StateReadSelection::answer_targets(std::slice::from_ref(
                    &owner_slot,
                ))),
                ..answer_request("", "¿Quién está a cargo?", &[], 20, 20, Some(10))
            },
            &projection,
        )
        .unwrap();
    assert_eq!(trace.answer_target_slot_ids, vec![owner_slot]);

    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn explicit_state_selection_excludes_unrequested_evidence_slots() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("projection_evidence_slot_admission");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    for (id, predicate, value) in [
        ("servicio_responsable", "responsable", "Ana"),
        ("servicio_renovacion", "renovación", "septiembre"),
        ("servicio_region", "región", "sur"),
    ] {
        store
            .append_claim(
                &make_claim(
                    &scope,
                    id,
                    "servicio",
                    predicate,
                    Some(value),
                    10,
                    (ClaimKind::Fact, ClaimPolarity::Affirmative),
                ),
                None,
            )
            .unwrap();
    }
    let slots = store.scan_slots(&scope, usize::MAX, Some(10)).unwrap();
    let owner_slot = slots
        .iter()
        .find(|slot| slot.predicate.as_deref() == Some("responsable"))
        .unwrap()
        .slot_key
        .clone();
    let renewal_slot = slots
        .iter()
        .find(|slot| slot.predicate.as_deref() == Some("renovación"))
        .unwrap()
        .slot_key
        .clone();
    let region_slot = slots
        .iter()
        .find(|slot| slot.predicate.as_deref() == Some("región"))
        .unwrap()
        .slot_key
        .clone();
    let evidence = SpanSearchHit {
        span: span_record("span_servicio_renovacion", MemoryStatus::Active, Some(10)),
        score: 1.0,
    };

    let projection = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                selections: &StateReadSelection::answer_targets(std::slice::from_ref(&owner_slot)),
                ..answer_request("selección", "更新された情報", &[evidence], 20, 20, Some(10))
            },
        )
        .unwrap();
    let projected_slots = projection
        .claims
        .iter()
        .filter_map(|claim| claim.slot_id.as_ref())
        .collect::<BTreeSet<_>>();

    assert!(projected_slots.contains(&owner_slot));
    assert!(!projected_slots.contains(&renewal_slot));
    assert!(!projected_slots.contains(&region_slot));

    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn entity_alias_index_resolves_beyond_legacy_scan_limit() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("entity_alias_index_scale");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let entities = (0..1_100)
        .map(|index| EntityInput {
            id: Some(format!("entity_{index}")),
            scope: scope.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            entity_type: "service".to_string(),
            canonical_name: format!("service {index}"),
            aliases: vec![format!("別名 {index}")],
            source_claim_ids: vec![format!("claim_entity_{index}")],
            merge_parent_ids: Vec::new(),
            split_from_id: None,
            confidence: Some(1.0),
        })
        .collect();
    store
        .apply_state_mutation_batch(crate::state::StateMutationBatch {
            scope: scope.clone(),
            transaction_time_ms: 100,
            propagation_seed: "stream".to_string(),
            max_rule_hops: 3,
            claims: Vec::new(),
            entities,
            rules: Vec::new(),
            slot_aliases: Vec::new(),
            corrections: Vec::new(),
        })
        .unwrap();

    let result = store
        .apply_state_mutation_batch(crate::state::StateMutationBatch {
            scope: scope.clone(),
            transaction_time_ms: 200,
            propagation_seed: "stream".to_string(),
            max_rule_hops: 3,
            claims: vec![make_claim(
                &scope,
                "claim_last_service",
                "別名 1099",
                "状態",
                Some("稼働中"),
                200,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            )],
            entities: Vec::new(),
            rules: Vec::new(),
            slot_aliases: Vec::new(),
            corrections: Vec::new(),
        })
        .unwrap();

    assert_eq!(
        result.claims[0].subject_entity_id.as_deref(),
        Some("entity_1099")
    );
    assert!(result.claims[0].slot_id.is_some());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn entity_alias_index_fails_closed_on_ambiguous_unicode_alias() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("entity_alias_index_collision");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let entities = ["entity_alpha", "entity_beta"]
        .into_iter()
        .map(|id| EntityInput {
            id: Some(id.to_string()),
            scope: scope.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            entity_type: "service".to_string(),
            canonical_name: id.to_string(),
            aliases: vec!["共有名".to_string()],
            source_claim_ids: vec![format!("claim_{id}")],
            merge_parent_ids: Vec::new(),
            split_from_id: None,
            confidence: Some(1.0),
        })
        .collect();
    store
        .apply_state_mutation_batch(crate::state::StateMutationBatch {
            scope: scope.clone(),
            transaction_time_ms: 100,
            propagation_seed: "stream".to_string(),
            max_rule_hops: 3,
            claims: Vec::new(),
            entities,
            rules: Vec::new(),
            slot_aliases: Vec::new(),
            corrections: Vec::new(),
        })
        .unwrap();

    let result = store
        .apply_state_mutation_batch(crate::state::StateMutationBatch {
            scope: scope.clone(),
            transaction_time_ms: 200,
            propagation_seed: "stream".to_string(),
            max_rule_hops: 3,
            claims: vec![make_claim(
                &scope,
                "claim_ambiguous_service",
                "共有名",
                "状態",
                Some("稼働中"),
                200,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            )],
            entities: Vec::new(),
            rules: Vec::new(),
            slot_aliases: Vec::new(),
            corrections: Vec::new(),
        })
        .unwrap();

    assert!(result.claims[0].subject_entity_id.is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn structured_slot_deletion_invalidates_explicit_dependent_state() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("structured_slot_deletion_dependency");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let mut rule = make_derived_rule_inputs();
    rule.id = Some("rule_responsable_contact".to_string());
    rule.trigger_subject_key = "service".to_string();
    rule.trigger_predicate_key = "responsable".to_string();
    rule.target_subject_key = "service".to_string();
    rule.target_predicate_key = "contact".to_string();
    rule.trigger_subject = Some("service".to_string());
    rule.trigger_predicate = Some("responsable".to_string());
    rule.target_subject = Some("service".to_string());
    rule.target_predicate = Some("contact".to_string());
    rule.activation = RuleActivation::OnChange;
    rule.action = RuleAction::MarkUnsupported;
    rule.value_template = None;
    rule.source_span_ids = vec!["span_rule".to_string()];
    rule.source_episode_ids = vec!["episode_rule".to_string()];

    store
        .apply_state_mutation_batch(crate::state::StateMutationBatch {
            scope: scope.clone(),
            transaction_time_ms: 100,
            propagation_seed: "flux".to_string(),
            max_rule_hops: 3,
            claims: vec![
                make_claim(
                    &scope,
                    "claim_responsable",
                    "service",
                    "responsable",
                    Some("Ana"),
                    10,
                    (ClaimKind::Fact, ClaimPolarity::Affirmative),
                ),
                make_claim(
                    &scope,
                    "claim_contact",
                    "service",
                    "contact",
                    Some("Béa"),
                    10,
                    (ClaimKind::Fact, ClaimPolarity::Affirmative),
                ),
            ],
            entities: Vec::new(),
            rules: vec![rule],
            slot_aliases: Vec::new(),
            corrections: Vec::new(),
        })
        .unwrap();

    let mut correction = crate::ingest::add_correction(
        CorrectionInput {
            id: Some("correction_responsable_supprime".to_string()),
            scope: scope.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            operation: CorrectionOperation::Tombstone,
            target_type: "claim".to_string(),
            target_ids: Vec::new(),
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
    correction.target_subject_key = Some("service".to_string());
    correction.target_predicate_key = Some("responsable".to_string());
    correction.source_span_ids = vec!["span_correction".to_string()];
    correction.source_episode_ids = vec!["episode_correction".to_string()];

    let result = store
        .apply_state_mutation_batch(crate::state::StateMutationBatch {
            scope: scope.clone(),
            transaction_time_ms: 200,
            propagation_seed: "flux".to_string(),
            max_rule_hops: 3,
            claims: Vec::new(),
            entities: Vec::new(),
            rules: Vec::new(),
            slot_aliases: Vec::new(),
            corrections: vec![correction],
        })
        .unwrap();

    assert!(result.derived_claims.iter().any(|claim| {
        claim.predicate.as_deref() == Some("contact")
            && matches!(claim.polarity, ClaimPolarity::Uncertain)
    }));
    assert!(result.state_records.iter().any(|state| {
        state.predicate.as_deref() == Some("responsable")
            && matches!(state.state_kind, StateRecordKind::Tombstone)
            && state.valid_from_ms == Some(20)
    }));
    assert!(result.state_records.iter().any(|state| {
        state.predicate.as_deref() == Some("contact")
            && matches!(state.state_kind, StateRecordKind::Unsupported)
    }));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn memory_store_queries_valid_time_independently_from_transaction_time() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("bitemporal_state_projection");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();

    store
        .apply_state_mutation_batch(crate::state::StateMutationBatch {
            scope: scope.clone(),
            transaction_time_ms: 100,
            propagation_seed: "stream".to_string(),
            max_rule_hops: 3,
            claims: vec![
                make_claim(
                    &scope,
                    "claim_owner_old",
                    "project",
                    "owner",
                    Some("Eli"),
                    10,
                    (ClaimKind::Fact, ClaimPolarity::Affirmative),
                ),
                make_claim(
                    &scope,
                    "claim_status",
                    "project",
                    "status",
                    Some("active"),
                    10,
                    (ClaimKind::Fact, ClaimPolarity::Affirmative),
                ),
            ],
            entities: Vec::new(),
            rules: Vec::new(),
            slot_aliases: Vec::new(),
            corrections: Vec::new(),
        })
        .unwrap();
    store
        .apply_state_mutation_batch(crate::state::StateMutationBatch {
            scope: scope.clone(),
            transaction_time_ms: 200,
            propagation_seed: "stream".to_string(),
            max_rule_hops: 3,
            claims: vec![make_claim(
                &scope,
                "claim_owner_new",
                "project",
                "owner",
                Some("Dana"),
                20,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            )],
            entities: Vec::new(),
            rules: Vec::new(),
            slot_aliases: Vec::new(),
            corrections: Vec::new(),
        })
        .unwrap();

    let value_at = |valid_at_ms, transaction_at_ms| {
        store
            .scan_state_records(
                &scope,
                StateRecordScan {
                    limit: 20,
                    temporal: crate::state::BiTemporalQuery::as_of(valid_at_ms, transaction_at_ms),
                },
            )
            .unwrap()
            .into_iter()
            .find(|record| record.predicate.as_deref() == Some("owner"))
            .and_then(|record| record.object_value)
    };
    assert_eq!(value_at(15, 150).as_deref(), Some("Eli"));
    assert_eq!(value_at(25, 150).as_deref(), Some("Eli"));
    assert_eq!(value_at(15, 250).as_deref(), Some("Eli"));
    assert_eq!(value_at(25, 250).as_deref(), Some("Dana"));
    let current = store
        .scan_state_records(
            &scope,
            StateRecordScan {
                limit: 20,
                temporal: crate::state::BiTemporalQuery::as_of(25, 250),
            },
        )
        .unwrap();
    assert_eq!(
        current
            .iter()
            .find(|record| record.predicate.as_deref() == Some("status"))
            .map(|record| record.recorded_at_ms),
        Some(100),
    );

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn answer_ready_evidence_hydration_attaches_missing_target_proof() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("answer_target_proof_hydration");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let proof = span_record("span_target_proof", MemoryStatus::Active, Some(10));
    store.append_span(&proof, None).unwrap();

    let mut target = make_claim(
        &scope(),
        "claim_target_proof",
        "entrega",
        "revisor",
        Some("Dana"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    target.slot_id = Some("slot_entrega_revisor".to_string());
    target.source_span_ids = vec![proof.id.clone()];
    let projection = AnswerReadyStateProjection {
        claims: vec![target.clone()],
        set_states: Vec::new(),
        slot_histories: Vec::new(),
        rules: Vec::new(),
        rule_outcomes: Vec::new(),
        corrections: Vec::new(),
        entities: Vec::new(),
        proof_span_ids: vec![proof.id.clone()],
        support: AnswerSupportContract {
            state: AnswerSupportState::Supported,
            source_claim_ids: vec![target.id.clone()],
            source_span_ids: vec![proof.id.clone()],
            slots: vec![AnswerSlotSupport {
                slot_id: target.slot_id.clone().unwrap(),
                subject: target.subject.clone(),
                predicate: target.predicate.clone(),
                view: StateReadView::Current,
                state: AnswerSupportState::Supported,
                source_claim_ids: vec![target.id],
                source_span_ids: vec![proof.id.clone()],
            }],
        },
    };
    let retrieved = SpanSearchHit {
        span: span_record("span_retrieved", MemoryStatus::Active, Some(10)),
        score: 0.7,
    };

    let hydrated = store
        .hydrate_answer_ready_state_evidence(&scope(), &projection, &[retrieved])
        .unwrap();

    assert_eq!(hydrated[0].span.id, proof.id);
    assert_eq!(hydrated[1].span.id, "span_retrieved");
    assert_eq!(hydrated.len(), 2);

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn expired_answer_target_projects_no_evidence_until_direct_reassertion() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("expired_answer_target_no_evidence");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let mut expired = make_claim(
        &scope,
        "claim_delivery_approver_expired",
        "配送",
        "承認者",
        Some("愛子"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    expired.valid_to_ms = Some(20);
    store.append_claim(&expired, None).unwrap();

    let slot_id = store
        .scan_slots(&scope, usize::MAX, Some(30))
        .unwrap()
        .into_iter()
        .find(|slot| slot.predicate.as_deref() == Some("承認者"))
        .expect("expired state retains its canonical slot")
        .slot_key;
    let mut historical_span = span_record(
        "span_claim_delivery_approver_expired",
        MemoryStatus::Active,
        Some(10),
    );
    historical_span.text = "配送の承認者は愛子でした。".to_string();
    historical_span.lexical_text = historical_span.text.clone();
    historical_span.valid_to_ms = Some(20);
    let historical_hit = SpanSearchHit {
        span: historical_span,
        score: 1.0,
    };

    let missing = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                selections: &[StateReadSelection {
                    slot_id: slot_id.clone(),
                    view: StateReadView::Current,
                    role: StateReadRole::AnswerTarget,
                }],
                ..answer_request(
                    "expired-target",
                    "配送 承認者",
                    std::slice::from_ref(&historical_hit),
                    20,
                    0,
                    Some(30),
                )
            },
        )
        .unwrap();
    assert_eq!(missing.support.slots.len(), 1);
    assert_eq!(missing.support.slots[0].slot_id, slot_id);
    assert_eq!(
        missing.support.slots[0].state,
        AnswerSupportState::NoEvidence
    );
    let resolved = missing.resolved_answer_slots();
    assert_eq!(resolved.len(), 1);
    assert_eq!(resolved[0].subject.as_deref(), Some("配送"));
    assert_eq!(resolved[0].predicate.as_deref(), Some("承認者"));
    assert!(resolved[0].current_value.is_none());

    let context = build_answer_ready_state_context(
        &[],
        &missing,
        &[],
        std::slice::from_ref(&historical_hit),
        ContextOptions {
            token_budget: 512,
            include_provenance: true,
        },
    );
    assert!(context.body.starts_with("### ResolvedAnswerState"));
    assert!(context.body.contains("support_state: no_evidence"));
    assert!(context.body.contains("current_value: null"));
    assert!(!context.body.contains("配送の承認者は愛子でした。"));

    let historical = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                selections: &[StateReadSelection {
                    slot_id: slot_id.clone(),
                    view: StateReadView::Timeline,
                    role: StateReadRole::AnswerTarget,
                }],
                ..answer_request(
                    "expired-target-history",
                    "配送 承認者",
                    std::slice::from_ref(&historical_hit),
                    20,
                    0,
                    Some(30),
                )
            },
        )
        .unwrap();
    let historical_context = build_answer_ready_state_context(
        &[],
        &historical,
        &[],
        std::slice::from_ref(&historical_hit),
        ContextOptions {
            token_budget: 512,
            include_provenance: true,
        },
    );
    assert!(historical_context.body.contains("read_view: timeline"));
    assert!(historical_context
        .body
        .contains("配送の承認者は愛子でした。"));

    store
        .append_claim(
            &make_claim(
                &scope,
                "claim_delivery_approver_reasserted",
                "配送",
                "承認者",
                Some("Lucía"),
                40,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            ),
            None,
        )
        .unwrap();
    let restored = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                selections: &[StateReadSelection {
                    slot_id,
                    view: StateReadView::Current,
                    role: StateReadRole::AnswerTarget,
                }],
                ..answer_request("restored-target", "配送 承認者", &[], 20, 0, Some(41))
            },
        )
        .unwrap();
    assert_eq!(restored.support.slots.len(), 1);
    assert_eq!(
        restored.support.slots[0].state,
        AnswerSupportState::Supported
    );
    assert_eq!(
        restored.resolved_answer_slots()[0].current_value.as_deref(),
        Some("Lucía")
    );

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn mixed_answer_targets_have_one_authoritative_state_per_slot() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("mixed_answer_target_authority");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let current = make_claim(
        &scope,
        "claim_route_status_current",
        "ruta",
        "estado",
        Some("activa"),
        20,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let mut expired = make_claim(
        &scope,
        "claim_route_approver_expired",
        "ruta",
        "承認者",
        Some("愛子"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    expired.valid_to_ms = Some(15);
    store.append_claim(&current, None).unwrap();
    store.append_claim(&expired, None).unwrap();

    let slots = store.scan_slots(&scope, usize::MAX, Some(30)).unwrap();
    let current_slot = slots
        .iter()
        .find(|slot| slot.predicate.as_deref() == Some("estado"))
        .unwrap()
        .slot_key
        .clone();
    let expired_slot = slots
        .iter()
        .find(|slot| slot.predicate.as_deref() == Some("承認者"))
        .unwrap()
        .slot_key
        .clone();
    let hits = [
        SpanSearchHit {
            span: span_record(
                "span_claim_route_status_current",
                MemoryStatus::Active,
                Some(20),
            ),
            score: 1.0,
        },
        SpanSearchHit {
            span: span_record(
                "span_claim_route_approver_expired",
                MemoryStatus::Active,
                Some(10),
            ),
            score: 0.9,
        },
    ];
    let projection = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                selections: &[
                    StateReadSelection {
                        slot_id: current_slot.clone(),
                        view: StateReadView::Current,
                        role: StateReadRole::AnswerTarget,
                    },
                    StateReadSelection {
                        slot_id: expired_slot.clone(),
                        view: StateReadView::Current,
                        role: StateReadRole::AnswerTarget,
                    },
                ],
                ..answer_request(
                    "mixed-targets",
                    "ruta estado 承認者",
                    &hits,
                    20,
                    0,
                    Some(30),
                )
            },
        )
        .unwrap();

    assert_eq!(projection.support.slots.len(), 2);
    assert_eq!(
        projection
            .support
            .slots
            .iter()
            .filter(|slot| slot.slot_id == current_slot)
            .count(),
        1
    );
    assert_eq!(
        projection
            .support
            .slots
            .iter()
            .find(|slot| slot.slot_id == current_slot)
            .unwrap()
            .state,
        AnswerSupportState::Supported
    );
    assert_eq!(
        projection
            .support
            .slots
            .iter()
            .filter(|slot| slot.slot_id == expired_slot)
            .count(),
        1
    );
    assert_eq!(
        projection
            .support
            .slots
            .iter()
            .find(|slot| slot.slot_id == expired_slot)
            .unwrap()
            .state,
        AnswerSupportState::NoEvidence
    );
    let resolved = projection.resolved_answer_slots();
    assert_eq!(resolved.len(), 2);
    let context = build_answer_ready_state_context(
        &[],
        &projection,
        &[],
        &hits,
        ContextOptions {
            token_budget: 768,
            include_provenance: true,
        },
    );
    assert_eq!(context.body.matches("### ResolvedAnswerState").count(), 2);
    assert!(!context.body.contains("### CurrentState"));

    let _ = fs::remove_dir_all(dir);
}

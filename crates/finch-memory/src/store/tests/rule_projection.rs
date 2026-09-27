use super::*;

#[test]
fn memory_store_resolver_does_not_loop_on_cycle() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("rule_cycle");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();

    store
        .add_rule(RuleInput {
            id: Some("rule_a_to_b".to_string()),
            scope: scope(),
            status: MemoryStatus::Active,
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            trigger_subject_key: "a".to_string(),
            trigger_predicate_key: "status".to_string(),
            target_subject_key: "b".to_string(),
            target_predicate_key: "status".to_string(),
            trigger_slot_id: None,
            target_slot_id: None,
            trigger_subject: Some("a".to_string()),
            trigger_predicate: Some("status".to_string()),
            target_subject: Some("b".to_string()),
            target_predicate: Some("status".to_string()),
            target_match: RuleTargetMatch::ExactSlot,
            activation: RuleActivation::ContinuousProjection,
            action: RuleAction::DeriveValue,
            value_template: Some("{value}".to_string()),
            value: None,
            source_span_ids: vec!["rule_span".to_string()],
            source_episode_ids: vec!["ep_rule".to_string()],
            valid_from_ms: Some(0),
            valid_to_ms: None,
            confidence: Some(1.0),
        })
        .unwrap();
    store
        .add_rule(RuleInput {
            id: Some("rule_b_to_a".to_string()),
            scope: scope(),
            status: MemoryStatus::Active,
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            trigger_subject_key: "b".to_string(),
            trigger_predicate_key: "status".to_string(),
            target_subject_key: "a".to_string(),
            target_predicate_key: "status".to_string(),
            trigger_slot_id: None,
            target_slot_id: None,
            trigger_subject: Some("b".to_string()),
            trigger_predicate: Some("status".to_string()),
            target_subject: Some("a".to_string()),
            target_predicate: Some("status".to_string()),
            target_match: RuleTargetMatch::ExactSlot,
            activation: RuleActivation::ContinuousProjection,
            action: RuleAction::DeriveValue,
            value_template: Some("{value}".to_string()),
            value: None,
            source_span_ids: vec!["rule_span".to_string()],
            source_episode_ids: vec!["ep_rule".to_string()],
            valid_from_ms: Some(0),
            valid_to_ms: None,
            confidence: Some(1.0),
        })
        .unwrap();

    let scope = scope();
    let changed = make_claim(
        &scope,
        "claim_a",
        "a",
        "status",
        Some("green"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let applications = store
        .resolve_rules_for_changed_claims(&[changed], &BTreeSet::from(["claim_a".to_string()]), 3)
        .unwrap();

    assert_eq!(applications.len(), 1);
    assert_eq!(applications[0].claim.subject.as_deref(), Some("b"));

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn answer_target_dependency_closure_excludes_downstream_siblings() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("target_reverse_dependency_closure");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();

    for (id, subject, value) in [
        ("claim_origen", "origen", "uno"),
        ("claim_intermedio", "intermedio", "uno"),
        ("claim_objetivo", "objetivo", "uno"),
        ("claim_rama", "rama", "uno"),
    ] {
        store
            .append_claim(
                &make_claim(
                    &scope,
                    id,
                    subject,
                    "valor",
                    Some(value),
                    10,
                    (ClaimKind::Fact, ClaimPolarity::Affirmative),
                ),
                None,
            )
            .unwrap();
    }

    let add_rule = |id: &str, trigger: &str, target: &str| {
        store
            .add_rule(RuleInput {
                id: Some(id.to_string()),
                scope: scope.clone(),
                status: MemoryStatus::Active,
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                trigger_subject_key: trigger.to_string(),
                trigger_predicate_key: "valor".to_string(),
                target_subject_key: target.to_string(),
                target_predicate_key: "valor".to_string(),
                trigger_slot_id: None,
                target_slot_id: None,
                trigger_subject: Some(trigger.to_string()),
                trigger_predicate: Some("valor".to_string()),
                target_subject: Some(target.to_string()),
                target_predicate: Some("valor".to_string()),
                target_match: RuleTargetMatch::ExactSlot,
                activation: RuleActivation::ContinuousProjection,
                action: RuleAction::DeriveValue,
                value_template: Some("{value}".to_string()),
                value: None,
                source_span_ids: vec![format!("span_{id}")],
                source_episode_ids: vec![format!("episode_{id}")],
                valid_from_ms: Some(1),
                valid_to_ms: None,
                confidence: Some(1.0),
            })
            .unwrap();
    };
    add_rule("rule_origen_intermedio", "origen", "intermedio");
    add_rule("rule_intermedio_objetivo", "intermedio", "objetivo");
    add_rule("rule_origen_rama", "origen", "rama");

    let slots = store.scan_slots(&scope, usize::MAX, Some(20)).unwrap();
    let slot_id = |subject: &str| {
        slots
            .iter()
            .find(|slot| slot.subject.as_deref() == Some(subject))
            .map(|slot| slot.slot_key.clone())
            .unwrap()
    };
    let target_slot_id = slot_id("objetivo");
    let branch_slot_id = slot_id("rama");
    let selected_slot_ids = BTreeSet::from([target_slot_id]);
    let (rules, slot_ids) = store
        .dependency_closure(
            &scope,
            &selected_slot_ids,
            &selected_slot_ids,
            &BTreeSet::new(),
            Some(20),
            8,
        )
        .unwrap();
    let rule_ids = rules
        .iter()
        .map(|rule| rule.id.as_str())
        .collect::<BTreeSet<_>>();

    assert!(rule_ids.contains("rule_intermedio_objetivo"));
    assert!(rule_ids.contains("rule_origen_intermedio"));
    assert!(!rule_ids.contains("rule_origen_rama"));
    assert!(!slot_ids.contains(&branch_slot_id));

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn retrieved_derived_consequence_of_selected_trigger_becomes_answer_authority() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("retrieved_derived_consequence_authority");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();

    let contact = make_claim(
        &scope,
        "claim_contacto",
        "atelier",
        "contacto",
        Some("Noël"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let derive_rule = |id: &str, subject: &str, target_predicate: &str| RuleInput {
        id: Some(id.to_string()),
        scope: scope.clone(),
        status: MemoryStatus::Active,
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        trigger_subject_key: subject.to_string(),
        trigger_predicate_key: "responsable".to_string(),
        target_subject_key: subject.to_string(),
        target_predicate_key: target_predicate.to_string(),
        trigger_slot_id: None,
        target_slot_id: None,
        trigger_subject: Some(subject.to_string()),
        trigger_predicate: Some("responsable".to_string()),
        target_subject: Some(subject.to_string()),
        target_predicate: Some(target_predicate.to_string()),
        target_match: RuleTargetMatch::ExactSlot,
        activation: RuleActivation::ContinuousProjection,
        action: RuleAction::DeriveValue,
        value_template: Some("{value}".to_string()),
        value: None,
        source_span_ids: vec![format!("span_{id}")],
        source_episode_ids: vec![format!("episode_{id}")],
        valid_from_ms: Some(10),
        valid_to_ms: None,
        confidence: Some(1.0),
    };
    store
        .apply_state_mutation_batch(StateMutationBatch {
            scope: scope.clone(),
            transaction_time_ms: 20,
            propagation_seed: "read-set-lifecycle".to_string(),
            max_rule_hops: 8,
            claims: vec![
                make_claim(
                    &scope,
                    "claim_responsable",
                    "atelier",
                    "responsable",
                    Some("美咲"),
                    10,
                    (ClaimKind::Fact, ClaimPolarity::Affirmative),
                ),
                make_claim(
                    &scope,
                    "claim_horaire",
                    "atelier",
                    "horaire",
                    Some("08:00"),
                    10,
                    (ClaimKind::Fact, ClaimPolarity::Affirmative),
                ),
                make_claim(
                    &scope,
                    "claim_bureau_responsable",
                    "bureau",
                    "responsable",
                    Some("Noël"),
                    10,
                    (ClaimKind::Fact, ClaimPolarity::Affirmative),
                ),
                contact.clone(),
            ],
            entities: Vec::new(),
            rules: vec![
                derive_rule("rule_responsable_approver", "atelier", "承認者"),
                derive_rule("rule_responsable_suplente", "atelier", "suplente"),
                derive_rule("rule_bureau_backup", "bureau", "backup"),
            ],
            slot_aliases: Vec::new(),
            corrections: Vec::new(),
        })
        .unwrap();

    let slots = store.scan_slots(&scope, usize::MAX, Some(20)).unwrap();
    let slot_id = |predicate: &str| {
        slots
            .iter()
            .find(|slot| slot.predicate.as_deref() == Some(predicate))
            .map(|slot| slot.slot_key.clone())
            .unwrap()
    };
    let trigger_slot = slots
        .iter()
        .find(|slot| {
            slot.subject.as_deref() == Some("atelier")
                && slot.predicate.as_deref() == Some("responsable")
        })
        .map(|slot| slot.slot_key.clone())
        .unwrap();
    let derived_slot = slot_id("承認者");
    let sibling_slot = slot_id("suplente");
    let coverage_slot = slot_id("horaire");
    let unrelated_derived_slot = slot_id("backup");
    let deleted_slot = slot_id("contacto");

    let mut correction = crate::ingest::add_correction(
        CorrectionInput {
            id: Some("delete_atelier_contacto".to_string()),
            scope: scope.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            operation: CorrectionOperation::Tombstone,
            target_type: "claim".to_string(),
            target_ids: vec![contact.id.clone()],
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
    correction.target_subject_key = Some(canonical_slot_part("atelier"));
    correction.target_predicate_key = Some(canonical_slot_part("contacto"));
    correction.target_slot_id = Some(deleted_slot.clone());
    store.add_correction(&correction).unwrap();

    let states = store
        .scan_state_records(&scope, state_scan(usize::MAX, Some(20)))
        .unwrap();
    assert!(states.iter().any(|state| {
        state.slot_id.as_ref() == Some(&derived_slot)
            && matches!(state.state_kind, StateRecordKind::Derived)
    }));
    assert!(states.iter().any(|state| {
        state.slot_id.as_ref() == Some(&sibling_slot)
            && matches!(state.state_kind, StateRecordKind::Derived)
    }));
    assert!(states.iter().any(|state| {
        state.slot_id.as_ref() == Some(&unrelated_derived_slot)
            && matches!(state.state_kind, StateRecordKind::Derived)
    }));
    assert!(states.iter().any(|state| {
        state.slot_id.as_ref() == Some(&deleted_slot)
            && matches!(state.state_kind, StateRecordKind::Tombstone)
    }));

    let from_coverage = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                selections: &[
                    StateReadSelection {
                        slot_id: derived_slot.clone(),
                        view: StateReadView::Current,
                        role: StateReadRole::Supporting,
                    },
                    StateReadSelection {
                        slot_id: coverage_slot.clone(),
                        view: StateReadView::Current,
                        role: StateReadRole::Supporting,
                    },
                    StateReadSelection {
                        slot_id: unrelated_derived_slot.clone(),
                        view: StateReadView::Current,
                        role: StateReadRole::Supporting,
                    },
                ],
                target_selections: Some(&[StateReadSelection {
                    slot_id: trigger_slot.clone(),
                    view: StateReadView::Current,
                    role: StateReadRole::AnswerTarget,
                }]),
                ..answer_request(
                    "coverage-lifecycle-authority",
                    "atelier 承認者",
                    &[],
                    20,
                    0,
                    Some(20),
                )
            },
        )
        .unwrap();
    let from_coverage_ids = from_coverage
        .support
        .slots
        .iter()
        .map(|slot| slot.slot_id.as_str())
        .collect::<BTreeSet<_>>();
    assert!(from_coverage_ids.contains(trigger_slot.as_str()));
    assert!(from_coverage_ids.contains(derived_slot.as_str()));
    assert!(!from_coverage_ids.contains(coverage_slot.as_str()));
    assert!(!from_coverage_ids.contains(sibling_slot.as_str()));
    assert!(!from_coverage_ids.contains(unrelated_derived_slot.as_str()));
    assert!(!from_coverage_ids.contains(deleted_slot.as_str()));
    assert!(from_coverage.resolved_answer_slots().iter().any(|slot| {
        slot.predicate.as_deref() == Some("承認者") && slot.current_value.as_deref() == Some("美咲")
    }));

    let packed =
        build_answer_ready_state_context(&[], &from_coverage, &[], &[], ContextOptions::default());
    assert!(packed.body.contains("### ResolvedAnswerState"));
    assert!(packed.body.contains("承認者"));
    assert!(packed.body.contains("read_role: answer_target"));
    assert!(packed.body.contains("current_value: 美咲"));
    assert!(packed.body.contains("### CurrentState"));
    assert!(packed.body.contains("horaire"));

    let with_tombstone = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                selections: &[StateReadSelection {
                    slot_id: deleted_slot.clone(),
                    view: StateReadView::Current,
                    role: StateReadRole::Supporting,
                }],
                target_selections: Some(&[StateReadSelection {
                    slot_id: trigger_slot.clone(),
                    view: StateReadView::Current,
                    role: StateReadRole::AnswerTarget,
                }]),
                ..answer_request(
                    "coverage-tombstone-stays-coverage",
                    "atelier responsable",
                    &[],
                    20,
                    0,
                    Some(20),
                )
            },
        )
        .unwrap();
    assert!(!with_tombstone
        .support
        .slots
        .iter()
        .any(|slot| slot.slot_id == deleted_slot));
    assert_eq!(with_tombstone.support.slots.len(), 1);
    assert_eq!(with_tombstone.support.slots[0].slot_id, trigger_slot);
    assert!(!with_tombstone
        .support
        .slots
        .iter()
        .any(|slot| slot.slot_id == sibling_slot));

    let named_derived_without_trigger = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                selections: &[StateReadSelection {
                    slot_id: derived_slot.clone(),
                    view: StateReadView::Current,
                    role: StateReadRole::Supporting,
                }],
                target_selections: Some(&[StateReadSelection {
                    slot_id: coverage_slot.clone(),
                    view: StateReadView::Current,
                    role: StateReadRole::AnswerTarget,
                }]),
                ..answer_request(
                    "query-matched-derived-without-selected-trigger",
                    "atelier 承認者",
                    &[],
                    20,
                    0,
                    Some(20),
                )
            },
        )
        .unwrap();
    let named_derived_ids = named_derived_without_trigger
        .support
        .slots
        .iter()
        .map(|slot| slot.slot_id.as_str())
        .collect::<BTreeSet<_>>();
    assert!(named_derived_ids.contains(coverage_slot.as_str()));
    assert!(named_derived_ids.contains(derived_slot.as_str()));
    assert!(!named_derived_ids.contains(unrelated_derived_slot.as_str()));

    let named_tombstone = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                selections: &[StateReadSelection {
                    slot_id: deleted_slot.clone(),
                    view: StateReadView::Current,
                    role: StateReadRole::Supporting,
                }],
                target_selections: Some(&[StateReadSelection {
                    slot_id: trigger_slot.clone(),
                    view: StateReadView::Current,
                    role: StateReadRole::AnswerTarget,
                }]),
                ..answer_request(
                    "query-matched-tombstone-in-read-set",
                    "atelier contacto",
                    &[],
                    20,
                    0,
                    Some(20),
                )
            },
        )
        .unwrap();
    assert!(named_tombstone
        .support
        .slots
        .iter()
        .any(|slot| slot.slot_id == deleted_slot));

    let named_outside_read_set = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                selections: &[],
                target_selections: Some(&[StateReadSelection {
                    slot_id: trigger_slot.clone(),
                    view: StateReadView::Current,
                    role: StateReadRole::AnswerTarget,
                }]),
                ..answer_request(
                    "named-tombstone-outside-selector-read-set",
                    "atelier contacto",
                    &[],
                    20,
                    0,
                    Some(20),
                )
            },
        )
        .unwrap();
    assert!(named_outside_read_set
        .support
        .slots
        .iter()
        .any(|slot| slot.slot_id == deleted_slot));

    let bureau_trigger = slots
        .iter()
        .find(|slot| {
            slot.subject.as_deref() == Some("bureau")
                && slot.predicate.as_deref() == Some("responsable")
        })
        .map(|slot| slot.slot_key.clone())
        .unwrap();
    let other_subject = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                selections: &[],
                target_selections: Some(&[StateReadSelection {
                    slot_id: bureau_trigger,
                    view: StateReadView::Current,
                    role: StateReadRole::AnswerTarget,
                }]),
                ..answer_request(
                    "named-tombstone-other-subject-stays-out",
                    "atelier contacto",
                    &[],
                    20,
                    0,
                    Some(20),
                )
            },
        )
        .unwrap();
    assert!(!other_subject
        .support
        .slots
        .iter()
        .any(|slot| slot.slot_id == deleted_slot));

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn answer_ready_projection_fails_closed_for_an_unresolved_dependency_until_direct_reassertion() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("unresolved_dependency_closure");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let target = make_claim(
        &scope,
        "claim_release_reviewer",
        "lanzamiento",
        "revisión",
        Some("pendiente"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let rule = RuleInput {
        id: Some("rule_owner_release_reviewer".to_string()),
        scope: scope.clone(),
        status: MemoryStatus::Active,
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        trigger_subject_key: "proyecto".to_string(),
        trigger_predicate_key: "responsable".to_string(),
        target_subject_key: "lanzamiento".to_string(),
        target_predicate_key: "revisión".to_string(),
        trigger_slot_id: None,
        target_slot_id: None,
        trigger_subject: Some("proyecto".to_string()),
        trigger_predicate: Some("responsable".to_string()),
        target_subject: Some("lanzamiento".to_string()),
        target_predicate: Some("revisión".to_string()),
        target_match: RuleTargetMatch::ExactSlot,
        activation: RuleActivation::OnChange,
        action: RuleAction::DeriveValue,
        value_template: Some("{value}".to_string()),
        value: None,
        source_span_ids: vec!["span_dependency".to_string()],
        source_episode_ids: vec!["episode_dependency".to_string()],
        valid_from_ms: Some(15),
        valid_to_ms: None,
        confidence: Some(1.0),
    };
    store
        .apply_state_mutation_batch(StateMutationBatch {
            scope: scope.clone(),
            transaction_time_ms: 20,
            propagation_seed: "unresolved-dependency".to_string(),
            max_rule_hops: 8,
            claims: vec![target],
            entities: Vec::new(),
            rules: vec![rule],
            slot_aliases: Vec::new(),
            corrections: Vec::new(),
        })
        .unwrap();

    let target_slot_id = store
        .scan_slots(&scope, usize::MAX, Some(20))
        .unwrap()
        .into_iter()
        .find(|slot| slot.subject.as_deref() == Some("lanzamiento"))
        .unwrap()
        .slot_key;
    let projection = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                selections: &StateReadSelection::answer_targets(std::slice::from_ref(
                    &target_slot_id,
                )),
                ..answer_request(
                    "unresolved-dependency",
                    "¿Quién revisa el lanzamiento?",
                    &[],
                    20,
                    0,
                    Some(20),
                )
            },
        )
        .unwrap();

    assert!(projection
        .rules
        .iter()
        .any(|rule| rule.id == "rule_owner_release_reviewer"));
    assert!(projection.rule_outcomes.iter().any(|outcome| {
        outcome.rule_id == "rule_owner_release_reviewer"
            && matches!(outcome.status, RuleResolutionStatus::AppliedUnsupported)
    }));
    assert_eq!(projection.support.state, AnswerSupportState::Unsupported);
    assert!(projection.claims.iter().any(|claim| {
        claim.slot_id.as_ref() == Some(&target_slot_id)
            && matches!(claim.polarity, ClaimPolarity::Uncertain)
    }));
    let context =
        build_answer_ready_state_context(&[], &projection, &[], &[], ContextOptions::default());
    assert!(context
        .body
        .contains("dependency_rule_ids: rule_owner_release_reviewer"));
    assert!(context.body.contains("support_state: unsupported"));
    assert!(context.body.contains("answer_disposition: refuse"));

    store
        .append_claim(
            &make_claim(
                &scope,
                "claim_release_reviewer_reasserted",
                "lanzamiento",
                "revisión",
                Some("confirmada"),
                25,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            ),
            None,
        )
        .unwrap();
    let restored = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                selections: &StateReadSelection::answer_targets(&[target_slot_id]),
                ..answer_request(
                    "unresolved-dependency-restored",
                    "¿Quién revisa el lanzamiento?",
                    &[],
                    20,
                    0,
                    Some(30),
                )
            },
        )
        .unwrap();
    assert_eq!(restored.support.state, AnswerSupportState::Supported);
    assert!(restored.claims.iter().any(|claim| {
        claim.object_value.as_deref() == Some("confirmada")
            && matches!(claim.polarity, ClaimPolarity::Affirmative)
    }));
    assert!(restored.rule_outcomes.iter().any(|outcome| {
        outcome.rule_id == "rule_owner_release_reviewer"
            && matches!(outcome.status, RuleResolutionStatus::MissingTrigger)
    }));

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn unresolved_dependency_propagates_unsupported_state_through_multiple_hops() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("unresolved_dependency_multihop");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let intermediate = make_claim(
        &scope,
        "claim_intermediate_before_dependency",
        "工程",
        "段階",
        Some("旧状態"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let target = make_claim(
        &scope,
        "claim_target_before_dependency",
        "entrega",
        "estado",
        Some("anterior"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let rule = |id: &str,
                trigger_subject: &str,
                trigger_predicate: &str,
                target_subject: &str,
                target_predicate: &str,
                valid_from_ms: i64| RuleInput {
        id: Some(id.to_string()),
        scope: scope.clone(),
        status: MemoryStatus::Active,
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        trigger_subject_key: trigger_subject.to_string(),
        trigger_predicate_key: trigger_predicate.to_string(),
        target_subject_key: target_subject.to_string(),
        target_predicate_key: target_predicate.to_string(),
        trigger_slot_id: None,
        target_slot_id: None,
        trigger_subject: Some(trigger_subject.to_string()),
        trigger_predicate: Some(trigger_predicate.to_string()),
        target_subject: Some(target_subject.to_string()),
        target_predicate: Some(target_predicate.to_string()),
        target_match: RuleTargetMatch::ExactSlot,
        activation: RuleActivation::ContinuousProjection,
        action: RuleAction::DeriveValue,
        value_template: Some("{value}".to_string()),
        value: None,
        source_span_ids: vec![format!("span_{id}")],
        source_episode_ids: vec![format!("episode_{id}")],
        valid_from_ms: Some(valid_from_ms),
        valid_to_ms: None,
        confidence: Some(1.0),
    };
    store
        .apply_state_mutation_batch(StateMutationBatch {
            scope: scope.clone(),
            transaction_time_ms: 20,
            propagation_seed: "unresolved-multihop".to_string(),
            max_rule_hops: 8,
            claims: vec![intermediate, target],
            entities: Vec::new(),
            rules: vec![
                rule(
                    "rule_missing_to_intermediate",
                    "源",
                    "値",
                    "工程",
                    "段階",
                    15,
                ),
                rule(
                    "rule_intermediate_to_target",
                    "工程",
                    "段階",
                    "entrega",
                    "estado",
                    1,
                ),
            ],
            slot_aliases: Vec::new(),
            corrections: Vec::new(),
        })
        .unwrap();
    let target_slot_id = store
        .scan_slots(&scope, usize::MAX, Some(20))
        .unwrap()
        .into_iter()
        .find(|slot| slot.subject.as_deref() == Some("entrega"))
        .unwrap()
        .slot_key;
    let projection = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                selections: &StateReadSelection::answer_targets(&[target_slot_id]),
                ..answer_request(
                    "unresolved-multihop",
                    "entrega estado",
                    &[],
                    20,
                    0,
                    Some(20),
                )
            },
        )
        .unwrap();

    assert_eq!(projection.support.state, AnswerSupportState::Unsupported);
    let resolved = projection.resolved_answer_slots();
    assert_eq!(resolved.len(), 1);
    assert!(resolved[0]
        .dependency_rule_ids
        .contains(&"rule_missing_to_intermediate".to_string()));
    assert!(resolved[0]
        .dependency_rule_ids
        .contains(&"rule_intermediate_to_target".to_string()));

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn dependency_completion_validates_an_existing_materialized_claim() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("existing_materialized_dependency");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let trigger = make_claim(
        &scope,
        "claim_responsable",
        "project",
        "owner",
        Some("Ana"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let rule = store.add_rule(make_derived_rule_inputs()).unwrap();
    let materialized = materialize_rule_application(&trigger, &rule, false, &BTreeSet::new())
        .unwrap()
        .unwrap();

    let completion = store
        .complete_target_dependency_chains(
            std::slice::from_ref(&rule),
            &[trigger, materialized.clone()],
            &[],
            3,
        )
        .unwrap();

    assert!(completion.applications.is_empty());
    assert!(completion.projected_claims.is_empty());
    assert_eq!(completion.outcomes.len(), 1);
    assert_eq!(
        completion.outcomes[0].status,
        RuleResolutionStatus::AppliedDerived
    );
    assert_eq!(
        completion.outcomes[0].resolved_claim_id.as_deref(),
        Some(materialized.id.as_str())
    );

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn answer_context_only_emits_proof_for_the_projected_rule_result() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("projected_dependency_proof");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let trigger = make_claim(
        &scope,
        "claim_projected_responsable",
        "project",
        "owner",
        Some("Ana"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let rule = store.add_rule(make_derived_rule_inputs()).unwrap();
    let materialized = materialize_rule_application(&trigger, &rule, false, &BTreeSet::new())
        .unwrap()
        .unwrap();
    let outcome = RuleResolutionOutcome {
        rule_id: rule.id.clone(),
        trigger_slot_id: rule.trigger_slot_id.clone(),
        target_slot_id: materialized.slot_id.clone(),
        status: RuleResolutionStatus::AppliedDerived,
        resolved_claim_id: Some(materialized.id.clone()),
    };
    let projection = AnswerReadyStateProjection {
        claims: vec![materialized.clone()],
        set_states: Vec::new(),
        slot_histories: Vec::new(),
        rules: vec![rule.clone()],
        rule_outcomes: vec![outcome.clone()],
        corrections: Vec::new(),
        entities: Vec::new(),
        proof_span_ids: Vec::new(),
        support: AnswerSupportContract::default(),
    };

    let current =
        build_answer_ready_state_context(&[], &projection, &[], &[], ContextOptions::default());
    assert!(current.body.contains("### DependencyProof"));

    let mut lifecycle_winner = materialized;
    lifecycle_winner.id = "derived_later_lifecycle_winner".to_string();
    lifecycle_winner.object_value = Some("Bea".to_string());
    lifecycle_winner.claim_text = "project reviewer is Bea".to_string();
    let superseded_projection = AnswerReadyStateProjection {
        claims: vec![lifecycle_winner],
        set_states: Vec::new(),
        slot_histories: Vec::new(),
        rules: vec![rule],
        rule_outcomes: vec![outcome],
        corrections: Vec::new(),
        entities: Vec::new(),
        proof_span_ids: Vec::new(),
        support: AnswerSupportContract::default(),
    };
    let superseded = build_answer_ready_state_context(
        &[],
        &superseded_projection,
        &[],
        &[],
        ContextOptions::default(),
    );
    assert!(!superseded.body.contains("### DependencyProof"));
    assert_eq!(
        superseded.rule_outcomes,
        vec![superseded_projection.rule_outcomes[0].clone()]
    );

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn dependency_completion_validates_an_existing_unsupported_claim() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("existing_unsupported_dependency");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let trigger = make_claim(
        &scope,
        "claim_responsable_ausente",
        "project",
        "owner",
        Some("Ana"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let prior_trigger = make_claim(
        &scope,
        "claim_responsable_anterior",
        "project",
        "owner",
        Some("Bea"),
        5,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    store.append_claim(&prior_trigger, None).unwrap();
    store.append_claim(&trigger, None).unwrap();
    let mut input = make_derived_rule_inputs();
    input.id = Some("rule_owner_reviewer_unsupported".to_string());
    input.activation = RuleActivation::OnChange;
    input.action = RuleAction::MarkUnsupported;
    input.value_template = None;
    let rule = store.add_rule(input).unwrap();
    let unsupported = materialize_rule_application(&trigger, &rule, true, &BTreeSet::new())
        .unwrap()
        .unwrap();

    let completion = store
        .complete_target_dependency_chains(
            std::slice::from_ref(&rule),
            &[trigger, unsupported.clone()],
            &[],
            3,
        )
        .unwrap();

    assert!(completion.applications.is_empty());
    assert!(completion.projected_claims.is_empty());
    assert_eq!(completion.outcomes.len(), 1);
    assert_eq!(
        completion.outcomes[0].status,
        RuleResolutionStatus::AppliedUnsupported
    );
    assert_eq!(
        completion.outcomes[0].resolved_claim_id.as_deref(),
        Some(unsupported.id.as_str())
    );

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn late_slot_alias_binding_reconciles_dependency_state_without_another_trigger_write() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("late_slot_alias_dependency_reconciliation");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let prior_trigger = make_claim(
        &scope,
        "project_owner_before_alias",
        "proyecto",
        "responsable",
        Some("Ana"),
        5,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let target = make_claim(
        &scope,
        "release_reviewer_before_alias",
        "lanzamiento",
        "revisión",
        Some("valor anterior"),
        7,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let current_trigger = make_claim(
        &scope,
        "project_owner_after_alias",
        "proyecto",
        "responsable",
        Some("Bea"),
        20,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let rule = RuleInput {
        id: Some("rule_owner_to_release_approval".to_string()),
        scope: scope.clone(),
        status: MemoryStatus::Active,
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        trigger_subject_key: "proyecto".to_string(),
        trigger_predicate_key: "responsable".to_string(),
        target_subject_key: "lanzamiento".to_string(),
        target_predicate_key: "aprobación".to_string(),
        trigger_slot_id: None,
        target_slot_id: None,
        trigger_subject: Some("proyecto".to_string()),
        trigger_predicate: Some("responsable".to_string()),
        target_subject: Some("lanzamiento".to_string()),
        target_predicate: Some("aprobación".to_string()),
        target_match: RuleTargetMatch::ExactSlot,
        activation: RuleActivation::OnChange,
        action: RuleAction::DeriveValue,
        value_template: Some("{value}".to_string()),
        value: None,
        source_span_ids: vec!["span_alias_rule".to_string()],
        source_episode_ids: vec!["episode_alias_rule".to_string()],
        valid_from_ms: Some(1),
        valid_to_ms: None,
        confidence: Some(1.0),
    };
    store
        .apply_state_mutation_batch(StateMutationBatch {
            scope: scope.clone(),
            transaction_time_ms: 25,
            propagation_seed: "late-alias-before".to_string(),
            max_rule_hops: 8,
            claims: vec![prior_trigger, target.clone(), current_trigger],
            entities: Vec::new(),
            rules: vec![rule],
            slot_aliases: Vec::new(),
            corrections: Vec::new(),
        })
        .unwrap();
    let before = store
        .scan_rules(&scope, usize::MAX, Some(20))
        .unwrap()
        .into_iter()
        .find(|rule| rule.id == "rule_owner_to_release_approval")
        .unwrap();
    let target_slot_id = store
        .scan_slots(&scope, usize::MAX, Some(20))
        .unwrap()
        .into_iter()
        .find(|slot| slot.predicate.as_deref() == Some("revisión"))
        .unwrap()
        .slot_key;
    assert_eq!(before.target_binding_status, RuleBindingStatus::Bound);
    assert_ne!(before.target_slot_id.as_ref(), Some(&target_slot_id));

    store
        .apply_state_mutation_batch(StateMutationBatch {
            scope: scope.clone(),
            transaction_time_ms: 30,
            propagation_seed: "late-alias-after".to_string(),
            max_rule_hops: 8,
            claims: Vec::new(),
            entities: Vec::new(),
            rules: Vec::new(),
            slot_aliases: vec![SlotAliasInput {
                id: Some("slot_alias_release_approval".to_string()),
                scope: scope.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                alias_subject: "lanzamiento".to_string(),
                alias_predicate: "aprobación".to_string(),
                canonical_slot_id: None,
                target_claim_id: Some(target.id.clone()),
                source_claim_ids: vec![target.id.clone()],
                valid_from_ms: Some(1),
                valid_to_ms: None,
            }],
            corrections: Vec::new(),
        })
        .unwrap();

    let after = store
        .scan_rules(&scope, usize::MAX, Some(20))
        .unwrap()
        .into_iter()
        .find(|rule| rule.id == "rule_owner_to_release_approval")
        .unwrap();
    assert_eq!(after.target_binding_status, RuleBindingStatus::Bound);
    assert_eq!(after.target_slot_id.as_ref(), Some(&target_slot_id));
    let states = store
        .scan_state_records(&scope, state_scan(usize::MAX, Some(20)))
        .unwrap();
    assert!(states.iter().any(|state| {
        matches!(state.state_kind, StateRecordKind::Derived)
            && state.subject.as_deref() == Some("lanzamiento")
            && state.predicate.as_deref() == Some("revisión")
            && state.object_value.as_deref() == Some("Bea")
    }));
    assert!(!states.iter().any(|state| {
        matches!(state.state_kind, StateRecordKind::Current)
            && state.subject.as_deref() == Some("lanzamiento")
            && state.predicate.as_deref() == Some("revisión")
            && state.object_value.as_deref() == Some("valor anterior")
    }));

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn late_dependency_rule_persists_unsupported_state_until_direct_evidence_returns() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("late_rule_unsupported_reconciliation");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let prior_trigger = make_claim(
        &scope,
        "supplier_before_rule",
        "계정",
        "공급자",
        Some("이전 공급자"),
        5,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let dependent = make_claim(
        &scope,
        "contact_before_rule",
        "계정",
        "담당자",
        Some("민지"),
        7,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let current_trigger = make_claim(
        &scope,
        "supplier_after_rule",
        "계정",
        "공급자",
        Some("새 공급자"),
        20,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    store
        .apply_state_mutation_batch(StateMutationBatch {
            scope: scope.clone(),
            transaction_time_ms: 25,
            propagation_seed: "late-rule-before".to_string(),
            max_rule_hops: 8,
            claims: vec![prior_trigger, dependent, current_trigger],
            entities: Vec::new(),
            rules: Vec::new(),
            slot_aliases: Vec::new(),
            corrections: Vec::new(),
        })
        .unwrap();
    let before_states = store
        .scan_state_records(&scope, state_scan(usize::MAX, Some(20)))
        .unwrap();
    assert!(before_states.iter().any(|state| {
        matches!(state.state_kind, StateRecordKind::Current)
            && state.predicate.as_deref() == Some("담당자")
    }));

    store
        .apply_state_mutation_batch(StateMutationBatch {
            scope: scope.clone(),
            transaction_time_ms: 30,
            propagation_seed: "late-rule-after".to_string(),
            max_rule_hops: 8,
            claims: Vec::new(),
            entities: Vec::new(),
            rules: vec![RuleInput {
                id: Some("rule_supplier_invalidates_contact".to_string()),
                scope: scope.clone(),
                status: MemoryStatus::Active,
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                trigger_subject_key: "계정".to_string(),
                trigger_predicate_key: "공급자".to_string(),
                target_subject_key: "계정".to_string(),
                target_predicate_key: "담당자".to_string(),
                trigger_slot_id: None,
                target_slot_id: None,
                trigger_subject: Some("계정".to_string()),
                trigger_predicate: Some("공급자".to_string()),
                target_subject: Some("계정".to_string()),
                target_predicate: Some("담당자".to_string()),
                target_match: RuleTargetMatch::ExactSlot,
                activation: RuleActivation::OnChange,
                action: RuleAction::MarkUnsupported,
                value_template: None,
                value: None,
                source_span_ids: vec!["span_late_rule".to_string()],
                source_episode_ids: vec!["episode_late_rule".to_string()],
                valid_from_ms: Some(1),
                valid_to_ms: None,
                confidence: Some(1.0),
            }],
            slot_aliases: Vec::new(),
            corrections: Vec::new(),
        })
        .unwrap();

    let unsupported = store
        .scan_state_records(&scope, state_scan(usize::MAX, Some(20)))
        .unwrap();
    assert!(unsupported.iter().any(|state| {
        matches!(state.state_kind, StateRecordKind::Unsupported)
            && state.predicate.as_deref() == Some("담당자")
    }));
    assert!(!unsupported.iter().any(|state| {
        matches!(state.state_kind, StateRecordKind::Current)
            && state.predicate.as_deref() == Some("담당자")
    }));

    store
        .append_claim(
            &make_claim(
                &scope,
                "contact_reconfirmed",
                "계정",
                "담당자",
                Some("지수"),
                40,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            ),
            None,
        )
        .unwrap();
    let restored = store
        .scan_state_records(&scope, state_scan(usize::MAX, Some(40)))
        .unwrap();
    assert!(restored.iter().any(|state| {
        matches!(state.state_kind, StateRecordKind::Current)
            && state.predicate.as_deref() == Some("담당자")
            && state.object_value.as_deref() == Some("지수")
    }));
    assert!(!restored.iter().any(|state| {
        matches!(state.state_kind, StateRecordKind::Unsupported)
            && state.predicate.as_deref() == Some("담당자")
    }));

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn memory_store_context_packs_resolved_state_without_runner_plumbing() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("rule_context_pack");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();

    let mut unsupported_rule = make_derived_rule_inputs();
    unsupported_rule.id = Some("rule_owner_reviewer".to_string());
    unsupported_rule.target_predicate_key = "reviewer".to_string();
    unsupported_rule.target_predicate = Some("reviewer".to_string());
    unsupported_rule.action = RuleAction::DeriveValue;
    store.add_rule(unsupported_rule).unwrap();

    let scope = scope();
    let owner = make_claim(
        &scope,
        "claim_owner_pack",
        "project",
        "owner",
        Some("dana"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    store.append_claim(&owner, None).unwrap();

    let reviewer = make_claim(
        &scope,
        "claim_reviewer_pack",
        "project",
        "reviewer",
        Some("morgan"),
        5,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    store.append_claim(&reviewer, None).unwrap();

    let applications = store
        .resolve_rules_for_changed_claims(
            std::slice::from_ref(&owner),
            &BTreeSet::from(["claim_owner_pack".to_string()]),
            3,
        )
        .unwrap();
    assert!(!applications.is_empty());
    let persisted = store.scan_claims(&scope, 20, Some(10)).unwrap();
    assert!(applications.iter().all(|application| persisted
        .iter()
        .any(|claim| claim.id == application.claim.id)));

    let claims = store
        .scan_current_claims(&scope, 10, Some(owner.observed_at_ms))
        .unwrap();
    let context = build_query_state_context(
        &ContextInput {
            claims: &claims,
            ..Default::default()
        },
        ContextOptions::default(),
    );
    assert!(context.body.contains("### DerivedState"));

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn memory_store_projects_missing_read_time_rule_state() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("read_time_rule_projection");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();

    let mut rule = make_derived_rule_inputs();
    rule.id = Some("rule_owner_reviewer_read_time".to_string());
    rule.target_predicate_key = "reviewer".to_string();
    rule.target_predicate = Some("reviewer".to_string());
    rule.action = RuleAction::DeriveValue;
    store.add_rule(rule).unwrap();

    let scope = scope();
    let owner = make_claim(
        &scope,
        "claim_owner_read_time",
        "project",
        "owner",
        Some("Dana"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    // Missing target state is derived once, by the write-time evaluator, never at read time.
    store.append_claim(&owner, None).unwrap();
    let projected = store
        .scan_current_claims(&scope, 10, Some(10))
        .unwrap()
        .into_iter()
        .filter(|claim| claim.predicate.as_deref() == Some("reviewer"))
        .collect::<Vec<_>>();
    assert_eq!(projected.len(), 1);
    assert_eq!(projected[0].subject.as_deref(), Some("project"));
    assert_eq!(projected[0].predicate.as_deref(), Some("reviewer"));
    assert_eq!(projected[0].object_value.as_deref(), Some("Dana"));

    let existing_reviewer = make_claim(
        &scope,
        "claim_reviewer_read_time",
        "project",
        "reviewer",
        Some("Morgan"),
        11,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    store.append_claim(&existing_reviewer, None).unwrap();
    let reviewers = store
        .scan_current_claims(&scope, 10, Some(11))
        .unwrap()
        .into_iter()
        .filter(|claim| claim.predicate.as_deref() == Some("reviewer"))
        .filter_map(|claim| claim.object_value)
        .collect::<Vec<_>>();
    assert_eq!(reviewers, vec!["Morgan".to_string()]);

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn dependency_rule_projects_from_current_and_later_trigger_versions() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("rule_activation_boundary");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let mut input = make_derived_rule_inputs();
    input.id = Some("rule_activation_boundary".to_string());
    input.target_predicate_key = "reviewer".to_string();
    input.target_predicate = Some("reviewer".to_string());
    input.value_template = None;
    input.value = Some("Mina".to_string());
    input.valid_from_ms = Some(10);
    let rule = store.build_rule_record(input).unwrap();
    let baseline = make_claim(
        &scope,
        "claim_owner_baseline",
        "project",
        "owner",
        Some("Ana"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let changed = make_claim(
        &scope,
        "claim_owner_changed",
        "project",
        "owner",
        Some("Béa"),
        20,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );

    assert_eq!(
        materialize_rule_application(&baseline, &rule, false, &BTreeSet::new(),)
            .unwrap()
            .and_then(|claim| claim.object_value),
        Some("Mina".to_string())
    );
    assert_eq!(
        materialize_rule_application(&changed, &rule, false, &BTreeSet::new(),)
            .unwrap()
            .and_then(|claim| claim.object_value),
        Some("Mina".to_string())
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn memory_store_resolves_selected_target_from_indexed_trigger_at_read_time() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("read_time_target_rule_projection");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let reviewer = make_claim(
        &scope,
        "claim_reviewer_target_read_time",
        "project",
        "reviewer",
        Some("Morgan"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let prior_owner = make_claim(
        &scope,
        "claim_owner_prior_target_read_time",
        "project",
        "owner",
        Some("Ana"),
        5,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let owner = make_claim(
        &scope,
        "claim_owner_target_read_time",
        "project",
        "owner",
        Some("Dana"),
        20,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    store.append_claim(&prior_owner, None).unwrap();
    store.append_claim(&reviewer, None).unwrap();
    store.append_claim(&owner, None).unwrap();

    let mut input = make_derived_rule_inputs();
    input.id = Some("rule_owner_reviewer_target_read_time".to_string());
    input.target_predicate_key = "reviewer".to_string();
    input.target_predicate = Some("reviewer".to_string());
    input.activation = RuleActivation::OnChange;
    input.action = RuleAction::MarkUnsupported;
    input.value_template = None;
    store.add_rule(input).unwrap();

    let rules = store.scan_rules(&scope, usize::MAX, Some(30)).unwrap();
    let claims = store
        .scan_current_claims(&scope, usize::MAX, Some(30))
        .unwrap();
    let historical_trigger = store
        .scan_current_claims_for_slot_ids(
            &scope,
            &BTreeSet::from([rules[0].trigger_slot_id.clone().unwrap()]),
            usize::MAX,
            Some(19),
        )
        .unwrap();
    assert_eq!(
        historical_trigger
            .first()
            .and_then(|claim| claim.object_value.as_deref()),
        Some("Ana")
    );
    let completion = store
        .complete_target_dependency_chains(&rules, &claims, &[], 8)
        .unwrap();
    assert!(
        completion.applications.iter().any(|application| {
            application.claim.predicate.as_deref() == Some("reviewer")
                && matches!(application.claim.polarity, ClaimPolarity::Uncertain)
        }),
        "outcomes={:?}; rules={rules:?}; claims={claims:?}",
        completion.outcomes
    );

    let projection = store
        .project_answer_ready_state(
            &scope,
            &answer_request("target-read-time", "project reviewer", &[], 20, 0, Some(30)),
        )
        .unwrap();

    assert!(projection.claims.iter().any(|claim| {
        claim.predicate.as_deref() == Some("reviewer")
            && matches!(claim.polarity, ClaimPolarity::Uncertain)
    }));
    assert!(!projection.claims.iter().any(|claim| {
        claim.predicate.as_deref() == Some("reviewer")
            && claim.object_value.as_deref() == Some("Morgan")
            && matches!(claim.polarity, ClaimPolarity::Affirmative)
    }));
    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn memory_store_projects_answer_ready_state_from_core() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("answer_ready_state_projection");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();

    let mut rule = make_derived_rule_inputs();
    rule.id = Some("rule_owner_reviewer_answer_ready".to_string());
    rule.target_predicate_key = "reviewer".to_string();
    rule.target_predicate = Some("reviewer".to_string());
    rule.action = RuleAction::DeriveValue;
    store.add_rule(rule).unwrap();

    let owner = make_claim(
        &scope,
        "claim_project_owner_answer_ready",
        "project",
        "owner",
        Some("Dana"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    store.append_claim(&owner, None).unwrap();

    let unsupported = make_claim(
        &scope,
        "claim_billing_contact_unsupported_answer_ready",
        "account",
        "billing_contact",
        None,
        11,
        (ClaimKind::Fact, ClaimPolarity::Uncertain),
    );
    store.append_claim(&unsupported, None).unwrap();

    let deleted = make_claim(
        &scope,
        "claim_project_approver_deleted_answer_ready",
        "project",
        "approver",
        Some("Eli"),
        8,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    store.append_claim(&deleted, None).unwrap();
    let tombstone = crate::ingest::add_correction(
        CorrectionInput {
            id: Some("corr_project_approver_deleted_answer_ready".to_string()),
            scope: scope.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            operation: CorrectionOperation::Tombstone,
            target_type: "claim".to_string(),
            target_ids: vec![deleted.id.clone()],
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
        12,
    );
    store.add_correction(&tombstone).unwrap();

    let mut member = make_claim(
        &scope,
        "claim_project_member_answer_ready",
        "project",
        "member",
        Some("Morgan"),
        9,
        (ClaimKind::Relationship, ClaimPolarity::Affirmative),
    );
    member.claim_kind = ClaimKind::Relationship;
    store.append_claim(&member, None).unwrap();
    let second_member = make_claim(
        &scope,
        "claim_project_member_second_answer_ready",
        "project",
        "member",
        Some("Riley"),
        11,
        (ClaimKind::Relationship, ClaimPolarity::Affirmative),
    );
    store.append_claim(&second_member, None).unwrap();
    let removed_member = make_claim(
        &scope,
        "claim_project_member_removed_answer_ready",
        "project",
        "member",
        Some("Eli"),
        7,
        (ClaimKind::Relationship, ClaimPolarity::Affirmative),
    );
    store.append_claim(&removed_member, None).unwrap();
    let member_tombstone = crate::ingest::add_correction(
        CorrectionInput {
            id: Some("corr_project_member_removed_answer_ready".to_string()),
            scope: scope.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            operation: CorrectionOperation::Tombstone,
            target_type: "claim".to_string(),
            target_ids: vec![removed_member.id.clone()],
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
        12,
    );
    store.add_correction(&member_tombstone).unwrap();

    store
        .add_entity(
            EntityInput {
                id: Some("entity_project_answer_ready".to_string()),
                scope: scope.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                entity_type: "project".to_string(),
                canonical_name: "Project".to_string(),
                aliases: vec!["project".to_string()],
                source_claim_ids: vec![owner.id.clone()],
                merge_parent_ids: Vec::new(),
                split_from_id: None,
                confidence: Some(1.0),
            },
            None,
        )
        .unwrap();

    let hits = vec![
        SpanSearchHit {
            span: span_record(
                "span_claim_project_owner_answer_ready",
                MemoryStatus::Active,
                Some(10),
            ),
            score: 1.0,
        },
        SpanSearchHit {
            span: span_record(
                "span_claim_project_member_answer_ready",
                MemoryStatus::Active,
                Some(9),
            ),
            score: 0.9,
        },
    ];

    let projection = store
        .project_answer_ready_state(
            &scope,
            &answer_request(
                "case-answer-ready",
                "project owner reviewer approver member account billing contact",
                &hits,
                20,
                20,
                Some(13),
            ),
        )
        .unwrap();

    assert!(projection
        .claims
        .iter()
        .any(|claim| claim.id == "claim_project_owner_answer_ready"));
    assert!(projection
        .claims
        .iter()
        .any(|claim| claim.asserted_by == "extractor_derived"
            && claim.predicate.as_deref() == Some("reviewer")));
    assert!(projection.claims.iter().any(|claim| claim.id
        == "claim_billing_contact_unsupported_answer_ready"
        && matches!(claim.polarity, ClaimPolarity::Uncertain)));
    assert!(projection.claims.iter().any(|claim| claim.id
        == "claim_project_approver_deleted_answer_ready"
        && claim.status == MemoryStatus::Tombstoned));
    let set_state = projection
        .set_states
        .iter()
        .find(|state| {
            state.subject.as_deref() == Some("project")
                && state.predicate.as_deref() == Some("member")
        })
        .expect("project member set state");
    assert_eq!(set_state.members, vec!["Morgan", "Riley"]);
    assert_eq!(
        set_state.claim_ids,
        vec![
            "claim_project_member_answer_ready",
            "claim_project_member_second_answer_ready"
        ]
    );
    assert!(set_state
        .source_span_ids
        .iter()
        .any(|id| id == "span_claim_project_member_answer_ready"));
    assert!(!set_state.members.iter().any(|member| member == "Eli"));
    assert_eq!(set_state.valid_from_ms, Some(11));
    assert_eq!(set_state.valid_to_ms, None);
    let persisted_states = store
        .scan_state_records(&scope, state_scan(20, Some(13)))
        .unwrap();
    assert!(persisted_states.iter().any(|state| matches!(
        state.state_kind,
        StateRecordKind::Current
    ) && state.subject.as_deref() == Some("project")
        && state.predicate.as_deref() == Some("owner")
        && state.subject_entity_id.as_deref() == Some("entity_project_answer_ready")
        && state.slot_id.is_some()));
    assert!(persisted_states
        .iter()
        .any(|state| matches!(state.state_kind, StateRecordKind::Set)
            && state.members == vec!["Morgan", "Riley"]
            && state.subject_entity_id.as_deref() == Some("entity_project_answer_ready")
            && state.slot_id.is_some()));
    assert!(persisted_states.iter().any(|state| matches!(
        state.state_kind,
        StateRecordKind::Tombstone
    ) && state.claim_ids
        == vec!["claim_project_member_removed_answer_ready"]));
    let dependency_traces = store.scan_dependency_traces(&scope, 20, Some(13)).unwrap();
    let reviewer_trace = dependency_traces
        .iter()
        .find(|trace| trace.target_predicate_key == "reviewer")
        .expect("reviewer dependency trace");
    assert_eq!(reviewer_trace.rule_id, "rule_owner_reviewer_answer_ready");
    assert_eq!(
        reviewer_trace.trigger_claim_id,
        "claim_project_owner_answer_ready"
    );
    assert_eq!(reviewer_trace.target_subject_key, "project");
    assert_eq!(
        reviewer_trace.target_subject_entity_id.as_deref(),
        Some("entity_project_answer_ready")
    );
    assert!(reviewer_trace.target_slot_id.is_some());
    assert_eq!(reviewer_trace.hop, 1);
    assert!(matches!(
        reviewer_trace.state_kind,
        StateRecordKind::Derived
    ));
    assert!(reviewer_trace
        .source_span_ids
        .iter()
        .any(|id| id == "span_claim_project_owner_answer_ready"));
    assert!(reviewer_trace
        .source_span_ids
        .iter()
        .any(|id| id == "rule_span_owner"));
    assert!(projection
        .corrections
        .iter()
        .any(|correction| correction.id == "corr_project_approver_deleted_answer_ready"));
    assert!(projection
        .entities
        .iter()
        .any(|entity| entity.id == "entity_project_answer_ready"));
    assert!(projection
        .proof_span_ids
        .iter()
        .any(|id| id == "span_claim_project_owner_answer_ready"));
    let slots = store.scan_slots(&scope, 20, Some(13)).unwrap();
    assert!(slots.iter().any(|slot| {
        slot.subject_entity_id.as_deref() == Some("entity_project_answer_ready")
            && slot.predicate_key == "owner"
            && slot.source_claim_ids == vec!["claim_project_owner_answer_ready"]
    }));

    let context =
        build_answer_ready_state_context(&[], &projection, &[], &hits, ContextOptions::default());
    assert!(context.body.contains("### TombstoneState"));
    assert!(context.body.contains("support_state: unsupported"));
    assert!(
        context.body.contains("state_kind: derived") || context.body.contains("### DerivedState")
    );
    assert!(
        context.body.contains("state_kind: current") || context.body.contains("### CurrentState")
    );
    assert!(
        context.body.contains("state_kind: set") || context.body.contains("### SetState"),
        "{}",
        context.body
    );
    assert!(!context
        .body
        .contains("EntityStateCard entity_state:project"));

    let path = store.path.clone();
    drop(store);
    let reopened = MemoryStore::open(&dir, CollectionOptions::default()).unwrap();
    assert!(reopened
        .scan_state_records(&scope, state_scan(20, Some(13)))
        .unwrap()
        .iter()
        .any(|state| matches!(state.state_kind, StateRecordKind::Set)
            && state.members == vec!["Morgan", "Riley"]));
    assert!(reopened
        .scan_dependency_traces(&scope, 20, Some(13))
        .unwrap()
        .iter()
        .any(|trace| trace.rule_id == "rule_owner_reviewer_answer_ready"
            && trace.trigger_claim_id == "claim_project_owner_answer_ready"
            && trace.target_predicate_key == "reviewer"));
    drop(reopened);
    let _ = std::fs::remove_dir_all(&path);
}

#[test]
fn answer_ready_projection_follows_multilingual_reverse_dependency_chain() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("multilingual_reverse_dependency_chain");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();

    for (id, subject, predicate, value, observed_at) in [
        ("claim_source", "供給元", "状態", "更新済み", 10),
        ("claim_contract", "契約", "担当", "以前", 8),
        ("claim_billing", "請求", "連絡先", "以前", 8),
    ] {
        store
            .append_claim(
                &make_claim(
                    &scope,
                    id,
                    subject,
                    predicate,
                    Some(value),
                    observed_at,
                    (ClaimKind::Fact, ClaimPolarity::Affirmative),
                ),
                None,
            )
            .unwrap();
    }

    for (id, trigger_subject, trigger_predicate, target_subject, target_predicate) in [
        ("rule_source_contract", "供給元", "状態", "契約", "担当"),
        ("rule_contract_billing", "契約", "担当", "請求", "連絡先"),
    ] {
        store
            .add_rule(RuleInput {
                id: Some(id.to_string()),
                scope: scope.clone(),
                status: MemoryStatus::Active,
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                trigger_subject_key: trigger_subject.to_string(),
                trigger_predicate_key: trigger_predicate.to_string(),
                target_subject_key: target_subject.to_string(),
                target_predicate_key: target_predicate.to_string(),
                trigger_slot_id: None,
                target_slot_id: None,
                trigger_subject: Some(trigger_subject.to_string()),
                trigger_predicate: Some(trigger_predicate.to_string()),
                target_subject: Some(target_subject.to_string()),
                target_predicate: Some(target_predicate.to_string()),
                target_match: RuleTargetMatch::ExactSlot,
                activation: RuleActivation::ContinuousProjection,
                action: RuleAction::DeriveValue,
                value_template: Some("{value}".to_string()),
                value: None,
                source_span_ids: vec![format!("span_{id}")],
                source_episode_ids: vec![format!("episode_{id}")],
                valid_from_ms: Some(5),
                valid_to_ms: None,
                confidence: Some(1.0),
            })
            .unwrap();
    }

    let projection = store
        .project_answer_ready_state(
            &scope,
            &answer_request("multilingual-reverse", "請求の連絡先", &[], 1, 0, Some(12)),
        )
        .unwrap();

    assert_eq!(
        projection
            .rules
            .iter()
            .map(|rule| rule.id.as_str())
            .collect::<BTreeSet<_>>(),
        BTreeSet::from(["rule_source_contract", "rule_contract_billing"])
    );
    assert_eq!(projection.rule_outcomes.len(), 2);
    assert!(projection
        .rule_outcomes
        .iter()
        .all(|outcome| matches!(outcome.status, RuleResolutionStatus::AppliedDerived)));
    assert!(projection.claims.iter().any(|claim| {
        claim.subject.as_deref() == Some("供給元")
            && claim.predicate.as_deref() == Some("状態")
            && claim.object_value.as_deref() == Some("更新済み")
    }));
    assert!(!projection.claims.iter().any(|claim| {
        claim.subject.as_deref() == Some("請求")
            && claim.predicate.as_deref() == Some("連絡先")
            && claim.object_value.as_deref() == Some("以前")
    }));
    assert!(projection.claims.iter().any(|claim| {
        claim.subject.as_deref() == Some("請求")
            && claim.predicate.as_deref() == Some("連絡先")
            && claim.object_value.as_deref() == Some("更新済み")
    }));
    let context =
        build_answer_ready_state_context(&[], &projection, &[], &[], ContextOptions::default());
    assert!(context.body.contains("### ResolvedAnswerState"));
    assert!(context.body.contains("dependency_resolution: applied"));
    assert!(context.body.contains("rule_source_contract"));
    assert!(context.body.contains("rule_contract_billing"));
    assert!(!context.body.contains("### DependencyProof"));
    assert!(!context.body.contains("value: {value}"));
    assert!(!context.body.contains("unresolved_"));

    let source_slot_id = store
        .scan_state_records(&scope, state_scan(usize::MAX, Some(12)))
        .unwrap()
        .into_iter()
        .find(|state| {
            state.subject.as_deref() == Some("供給元") && state.predicate.as_deref() == Some("状態")
        })
        .and_then(|state| state.slot_id)
        .unwrap();
    let from_trigger = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                selections: &StateReadSelection::answer_targets(&[source_slot_id]),
                ..answer_request(
                    "multilingual-forward",
                    "¿Cuál es el resultado dependiente?",
                    &[],
                    1,
                    0,
                    Some(12),
                )
            },
        )
        .unwrap();
    assert_eq!(from_trigger.rules.len(), 2);
    assert!(from_trigger.claims.iter().any(|claim| {
        claim.subject.as_deref() == Some("請求")
            && claim.predicate.as_deref() == Some("連絡先")
            && claim.object_value.as_deref() == Some("更新済み")
    }));

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn memory_store_canonical_entity_alias_collision_fails_closed_for_slots() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("canonical_alias_collision");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();

    for (id, canonical_name) in [
        ("entity_project_a", "Project A"),
        ("entity_project_b", "Project B"),
    ] {
        store
            .add_entity(
                EntityInput {
                    id: Some(id.to_string()),
                    scope: scope.clone(),
                    visibility: Visibility::Private,
                    policy_tags: Vec::new(),
                    entity_type: "project".to_string(),
                    canonical_name: canonical_name.to_string(),
                    aliases: vec!["project".to_string()],
                    source_claim_ids: vec!["claim_project_owner_collision".to_string()],
                    merge_parent_ids: Vec::new(),
                    split_from_id: None,
                    confidence: Some(1.0),
                },
                None,
            )
            .unwrap();
    }

    let claim = store
        .add_manual_claim(
            ManualClaimInput {
                id: Some("claim_project_owner_collision".to_string()),
                scope: scope.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                claim_text: "project owner is Dana".to_string(),
                subject: Some("project".to_string()),
                predicate: Some("owner".to_string()),
                object_value: Some("Dana".to_string()),
                claim_kind: ClaimKind::Fact,
                polarity: ClaimPolarity::Affirmative,
                source_span_ids: vec!["span_claim_project_owner_collision".to_string()],
                source_episode_ids: vec!["ep_claim_project_owner_collision".to_string()],
                asserted_by: "user".to_string(),
                confidence: Some(1.0),
                observed_at_ms: 10,
                valid_from_ms: Some(10),
                valid_to_ms: None,
            },
            None,
        )
        .unwrap();
    assert_eq!(claim.subject_entity_id, None);
    assert!(claim.slot_id.is_some());

    let states = store
        .scan_state_records(&scope, state_scan(20, Some(10)))
        .unwrap();
    let owner = states
        .iter()
        .find(|state| state.predicate.as_deref() == Some("owner"))
        .expect("owner state");
    assert_eq!(owner.subject_entity_id, None);
    assert_eq!(owner.slot_id, claim.slot_id);

    let slots = store.scan_slots(&scope, 20, Some(10)).unwrap();
    let slot = slots
        .iter()
        .find(|slot| slot.predicate_key == "owner")
        .expect("owner slot");
    assert_eq!(slot.subject_entity_id, None);
    assert!(slot.source_entity_ids.is_empty());

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn memory_store_tombstone_suppresses_canonical_state_until_later_readd() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("canonical_tombstone_lifecycle");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();

    let owner_morgan = make_claim(
        &scope,
        "claim_project_owner_morgan_lifecycle",
        "project",
        "owner",
        Some("Morgan"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    store.append_claim(&owner_morgan, None).unwrap();
    let member_eli = make_claim(
        &scope,
        "claim_project_member_eli_lifecycle",
        "project",
        "member",
        Some("Eli"),
        10,
        (ClaimKind::Relationship, ClaimPolarity::Affirmative),
    );
    store.append_claim(&member_eli, None).unwrap();

    for (id, target_id) in [
        (
            "corr_project_owner_morgan_lifecycle_deleted",
            "claim_project_owner_morgan_lifecycle",
        ),
        (
            "corr_project_member_eli_lifecycle_deleted",
            "claim_project_member_eli_lifecycle",
        ),
    ] {
        let correction = crate::ingest::add_correction(
            CorrectionInput {
                id: Some(id.to_string()),
                scope: scope.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                operation: CorrectionOperation::Tombstone,
                target_type: "claim".to_string(),
                target_ids: vec![target_id.to_string()],
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
        store.add_correction(&correction).unwrap();
    }

    let after_delete = store
        .project_answer_ready_state(
            &scope,
            &answer_request(
                "case-lifecycle-delete",
                "project owner member",
                &[],
                20,
                0,
                Some(21),
            ),
        )
        .unwrap();
    assert!(after_delete.claims.iter().any(|claim| {
        claim.id == "claim_project_owner_morgan_lifecycle"
            && claim.status == MemoryStatus::Tombstoned
    }));
    assert!(!after_delete.claims.iter().any(|claim| {
        claim.predicate.as_deref() == Some("owner")
            && claim.object_value.as_deref() == Some("Morgan")
            && claim.status == MemoryStatus::Active
    }));
    assert!(after_delete.set_states.is_empty());
    assert!(store
        .scan_state_records(&scope, state_scan(20, Some(21)))
        .unwrap()
        .iter()
        .any(
            |state| matches!(state.state_kind, StateRecordKind::Tombstone)
                && state.claim_ids == vec!["claim_project_member_eli_lifecycle"]
        ));

    store
        .append_claim(
            &make_claim(
                &scope,
                "claim_project_owner_dana_lifecycle",
                "project",
                "owner",
                Some("Dana"),
                25,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            ),
            None,
        )
        .unwrap();
    store
        .append_claim(
            &make_claim(
                &scope,
                "claim_project_member_eli_readd_lifecycle",
                "project",
                "member",
                Some("Eli"),
                25,
                (ClaimKind::Relationship, ClaimPolarity::Affirmative),
            ),
            None,
        )
        .unwrap();

    let after_readd = store
        .project_answer_ready_state(
            &scope,
            &answer_request(
                "case-lifecycle-readd",
                "project owner member",
                &[],
                20,
                0,
                Some(26),
            ),
        )
        .unwrap();
    assert!(after_readd.claims.iter().any(|claim| {
        claim.id == "claim_project_owner_dana_lifecycle"
            && claim.object_value.as_deref() == Some("Dana")
            && claim.status == MemoryStatus::Active
    }));
    assert!(!after_readd.claims.iter().any(|claim| {
        claim.id == "claim_project_owner_morgan_lifecycle" && claim.status == MemoryStatus::Active
    }));
    let set_state = after_readd
        .set_states
        .iter()
        .find(|state| state.predicate.as_deref() == Some("member"))
        .expect("member set state after re-add");
    assert_eq!(set_state.members, vec!["Eli"]);
    assert_eq!(
        set_state.claim_ids,
        vec!["claim_project_member_eli_readd_lifecycle"]
    );

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn memory_store_unsupported_generic_target_suppresses_current_state_for_concrete_slot() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("generic_unsupported_suppresses_current_state");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();

    let mut rule = make_derived_rule_inputs();
    rule.id = Some("rule_billing_contact_depends_on_vendor_lifecycle".to_string());
    rule.trigger_subject_key = "account".to_string();
    rule.trigger_predicate_key = "payment_vendor".to_string();
    rule.target_subject_key = "account".to_string();
    rule.target_predicate_key = "vendor_dependent_slot".to_string();
    rule.trigger_subject = Some("account".to_string());
    rule.trigger_predicate = Some("payment_vendor".to_string());
    rule.target_subject = Some("account".to_string());
    rule.target_predicate = Some("vendor_dependent_slot".to_string());
    rule.target_match = RuleTargetMatch::AnyActiveSlotForSubject;
    rule.activation = RuleActivation::OnChange;
    rule.action = RuleAction::MarkUnsupported;
    rule.value_template = None;
    rule.value = None;
    store.add_rule(rule).unwrap();

    store
        .append_claim(
            &make_claim(
                &scope,
                "claim_payment_vendor_prior_lifecycle",
                "account",
                "payment_vendor",
                Some("Legacy Pay"),
                5,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            ),
            None,
        )
        .unwrap();
    store
        .append_claim(
            &make_claim(
                &scope,
                "claim_billing_contact_old_lifecycle",
                "account",
                "Billing Contact",
                Some("Morgan"),
                10,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            ),
            None,
        )
        .unwrap();
    store
        .append_claim(
            &make_claim(
                &scope,
                "claim_payment_vendor_new_lifecycle",
                "account",
                "payment_vendor",
                Some("Northstar Pay"),
                20,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            ),
            None,
        )
        .unwrap();

    let stored_rule = store
        .scan_rules(&scope, 10, Some(20))
        .unwrap()
        .into_iter()
        .find(|rule| rule.id == "rule_billing_contact_depends_on_vendor_lifecycle")
        .unwrap();
    assert_eq!(stored_rule.trigger_binding_status, RuleBindingStatus::Bound);
    assert_eq!(stored_rule.target_binding_status, RuleBindingStatus::Bound);
    let trigger = store
        .scan_current_claims(&scope, 10, Some(20))
        .unwrap()
        .into_iter()
        .find(|claim| claim.id == "claim_payment_vendor_new_lifecycle")
        .unwrap();
    assert_eq!(
        store
            .rules_for_trigger_claim(&scope, &trigger, 10, Some(20))
            .unwrap()
            .len(),
        1
    );
    let states = store
        .scan_state_records(&scope, state_scan(20, Some(20)))
        .unwrap();
    assert!(states.iter().any(|state| {
        matches!(state.state_kind, StateRecordKind::Unsupported)
            && state.predicate.as_deref() == Some("Billing Contact")
    }));
    assert!(!states.iter().any(|state| {
        matches!(state.state_kind, StateRecordKind::Current)
            && state.predicate.as_deref() == Some("Billing Contact")
            && state.object_value.as_deref() == Some("Morgan")
    }));
    let projection = store
        .project_answer_ready_state(
            &scope,
            &answer_request(
                "generic-unsupported-target",
                "account Billing Contact",
                &[],
                1,
                0,
                Some(20),
            ),
        )
        .unwrap();
    assert!(projection.claims.iter().any(|claim| {
        claim.predicate.as_deref() == Some("Billing Contact")
            && matches!(claim.polarity, ClaimPolarity::Uncertain)
    }));
    assert!(projection.rule_outcomes.iter().any(|outcome| {
        outcome.rule_id == "rule_billing_contact_depends_on_vendor_lifecycle"
            && matches!(outcome.status, RuleResolutionStatus::AppliedUnsupported)
    }));

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn read_time_subject_wide_invalidation_keeps_distinct_concrete_targets() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("read_time_subject_wide_targets");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();

    let mut rule = make_derived_rule_inputs();
    rule.id = Some("rule_dependencias_cuenta".to_string());
    rule.trigger_subject_key = "cuenta".to_string();
    rule.trigger_predicate_key = "proveedor".to_string();
    rule.target_subject_key = "cuenta".to_string();
    rule.target_predicate_key = "dependencia".to_string();
    rule.trigger_subject = Some("cuenta".to_string());
    rule.trigger_predicate = Some("proveedor".to_string());
    rule.target_subject = Some("cuenta".to_string());
    rule.target_predicate = Some("dependencia".to_string());
    rule.target_match = RuleTargetMatch::AnyActiveSlotForSubject;
    rule.activation = RuleActivation::OnChange;
    rule.action = RuleAction::MarkUnsupported;
    rule.value_template = None;
    rule.value = None;
    store.add_rule(rule).unwrap();

    for claim in [
        make_claim(
            &scope,
            "claim_proveedor_anterior",
            "cuenta",
            "proveedor",
            Some("Pago Uno"),
            5,
            (ClaimKind::Fact, ClaimPolarity::Affirmative),
        ),
        make_claim(
            &scope,
            "claim_contacto_anterior",
            "cuenta",
            "contacto",
            Some("Ana"),
            10,
            (ClaimKind::Fact, ClaimPolarity::Affirmative),
        ),
        make_claim(
            &scope,
            "claim_limite_anterior",
            "cuenta",
            "límite",
            Some("veinte"),
            10,
            (ClaimKind::Fact, ClaimPolarity::Affirmative),
        ),
        make_claim(
            &scope,
            "claim_idioma_perfil",
            "perfil",
            "idioma",
            Some("français"),
            10,
            (ClaimKind::Fact, ClaimPolarity::Affirmative),
        ),
        make_claim(
            &scope,
            "claim_proveedor_nuevo",
            "cuenta",
            "proveedor",
            Some("Pago Dos"),
            20,
            (ClaimKind::Fact, ClaimPolarity::Affirmative),
        ),
    ] {
        store.append_claim(&claim, None).unwrap();
    }

    let slots = store.scan_slots(&scope, usize::MAX, Some(20)).unwrap();
    let slot_id = |subject: &str, predicate: &str| {
        slots
            .iter()
            .find(|slot| {
                slot.subject.as_deref() == Some(subject)
                    && slot.predicate.as_deref() == Some(predicate)
            })
            .map(|slot| slot.slot_key.clone())
            .unwrap()
    };
    let contacto = slot_id("cuenta", "contacto");
    let limite = slot_id("cuenta", "límite");
    let proveedor = slot_id("cuenta", "proveedor");
    let idioma = slot_id("perfil", "idioma");
    let resolution_roots = BTreeSet::from([
        contacto.clone(),
        limite.clone(),
        proveedor.clone(),
        idioma.clone(),
    ]);
    let answer_targets = BTreeSet::from([contacto.clone(), limite.clone()]);

    let (rules, _) = store
        .dependency_closure(
            &scope,
            &resolution_roots,
            &answer_targets,
            &BTreeSet::new(),
            Some(20),
            8,
        )
        .unwrap();
    let concrete_targets = rules
        .iter()
        .filter(|rule| rule.id == "rule_dependencias_cuenta")
        .filter_map(|rule| rule.target_slot_id.clone())
        .collect::<BTreeSet<_>>();
    assert_eq!(concrete_targets, answer_targets);
    assert!(!concrete_targets.contains(&proveedor));
    assert!(!concrete_targets.contains(&idioma));

    let claims = store
        .scan_current_claims(&scope, usize::MAX, Some(20))
        .unwrap();
    let completion = store
        .complete_target_dependency_chains(&rules, &claims, &[], 8)
        .unwrap();
    let applied_targets = completion
        .outcomes
        .iter()
        .filter(|outcome| outcome.rule_id == "rule_dependencias_cuenta")
        .filter(|outcome| matches!(outcome.status, RuleResolutionStatus::AppliedUnsupported))
        .filter_map(|outcome| outcome.target_slot_id.clone())
        .collect::<BTreeSet<_>>();
    assert_eq!(applied_targets, answer_targets);

    let _ = std::fs::remove_dir_all(&store.path);
}

fn exact_dependency_rule(
    id: &str,
    trigger_subject: &str,
    trigger_predicate: &str,
    target_subject: &str,
    target_predicate: &str,
    action: RuleAction,
    value: Option<&str>,
) -> RuleInput {
    RuleInput {
        id: Some(id.to_string()),
        scope: scope(),
        status: MemoryStatus::Active,
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        trigger_subject_key: trigger_subject.to_string(),
        trigger_predicate_key: trigger_predicate.to_string(),
        target_subject_key: target_subject.to_string(),
        target_predicate_key: target_predicate.to_string(),
        trigger_slot_id: None,
        target_slot_id: None,
        trigger_subject: Some(trigger_subject.to_string()),
        trigger_predicate: Some(trigger_predicate.to_string()),
        target_subject: Some(target_subject.to_string()),
        target_predicate: Some(target_predicate.to_string()),
        target_match: RuleTargetMatch::ExactSlot,
        activation: if matches!(&action, RuleAction::MarkUnsupported) {
            RuleActivation::OnChange
        } else {
            RuleActivation::ContinuousProjection
        },
        action,
        value_template: None,
        value: value.map(str::to_string),
        source_span_ids: vec![format!("span_{id}")],
        source_episode_ids: vec![format!("episode_{id}")],
        valid_from_ms: Some(0),
        valid_to_ms: None,
        confidence: Some(1.0),
    }
}

#[test]
fn answer_ready_projection_prefers_a_concrete_derivation_over_same_transition_invalidation() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("same_transition_derivation_precedence");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();

    for claim in [
        make_claim(
            &scope,
            "claim_relationship_before",
            "利用者",
            "関係状態",
            Some("以前"),
            5,
            (ClaimKind::Fact, ClaimPolarity::Affirmative),
        ),
        make_claim(
            &scope,
            "claim_travel_before",
            "利用者",
            "旅行計画",
            Some("古い計画"),
            10,
            (ClaimKind::Fact, ClaimPolarity::Affirmative),
        ),
    ] {
        store.append_claim(&claim, None).unwrap();
    }
    store
        .add_rule(exact_dependency_rule(
            "rule_relationship_derives_travel",
            "利用者",
            "関係状態",
            "利用者",
            "旅行計画",
            RuleAction::DeriveValue,
            Some("旅行なし"),
        ))
        .unwrap();
    store
        .add_rule(exact_dependency_rule(
            "rule_relationship_invalidates_travel",
            "利用者",
            "関係状態",
            "利用者",
            "旅行計画",
            RuleAction::MarkUnsupported,
            None,
        ))
        .unwrap();
    store
        .append_claim(
            &make_claim(
                &scope,
                "claim_relationship_after",
                "利用者",
                "関係状態",
                Some("現在"),
                20,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            ),
            None,
        )
        .unwrap();

    let target_slot_id = store
        .scan_slots(&scope, usize::MAX, Some(20))
        .unwrap()
        .into_iter()
        .find(|slot| slot.predicate.as_deref() == Some("旅行計画"))
        .unwrap()
        .slot_key;
    let projection = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                selections: &StateReadSelection::answer_targets(&[target_slot_id]),
                ..answer_request(
                    "same-transition-derivation",
                    "利用者の旅行計画",
                    &[],
                    20,
                    0,
                    Some(20),
                )
            },
        )
        .unwrap();

    assert_eq!(projection.support.state, AnswerSupportState::Supported);
    assert!(projection.claims.iter().any(|claim| {
        claim.predicate.as_deref() == Some("旅行計画")
            && claim.object_value.as_deref() == Some("旅行なし")
            && matches!(claim.polarity, ClaimPolarity::Affirmative)
    }));
    assert!(!projection.claims.iter().any(|claim| {
        claim.predicate.as_deref() == Some("旅行計画")
            && matches!(claim.polarity, ClaimPolarity::Uncertain)
    }));
    assert!(projection.rule_outcomes.iter().any(|outcome| {
        outcome.rule_id == "rule_relationship_derives_travel"
            && matches!(outcome.status, RuleResolutionStatus::AppliedDerived)
    }));
    assert!(projection.rule_outcomes.iter().any(|outcome| {
        outcome.rule_id == "rule_relationship_invalidates_travel"
            && matches!(outcome.status, RuleResolutionStatus::NotApplicable)
    }));

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn write_time_resolution_does_not_invalidate_an_already_known_derivation() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("known_derivation_precedence");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let previous_trigger = make_claim(
        &scope,
        "claim_assignment_previous",
        "equipo",
        "asignación",
        Some("anterior"),
        5,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let stale_target = make_claim(
        &scope,
        "claim_contact_stale",
        "equipo",
        "contacto",
        Some("valor anterior"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let current_trigger = make_claim(
        &scope,
        "claim_assignment_current",
        "equipo",
        "asignación",
        Some("actual"),
        20,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    for claim in [&previous_trigger, &stale_target, &current_trigger] {
        store.append_claim(claim, None).unwrap();
    }
    let derive = store
        .build_rule_record(exact_dependency_rule(
            "rule_assignment_derives_contact",
            "equipo",
            "asignación",
            "equipo",
            "contacto",
            RuleAction::DeriveValue,
            Some("nuevo contacto"),
        ))
        .unwrap();
    let invalidate = store
        .build_rule_record(exact_dependency_rule(
            "rule_assignment_invalidates_contact",
            "equipo",
            "asignación",
            "equipo",
            "contacto",
            RuleAction::MarkUnsupported,
            None,
        ))
        .unwrap();
    insert_many(
        &store.rules,
        vec![rule_doc(&derive).unwrap(), rule_doc(&invalidate).unwrap()],
    )
    .unwrap();
    let known_derived =
        materialize_rule_application(&current_trigger, &derive, false, &BTreeSet::new())
            .unwrap()
            .unwrap();
    let existing_ids = BTreeSet::from([
        previous_trigger.id.clone(),
        stale_target.id.clone(),
        current_trigger.id.clone(),
        known_derived.id,
    ]);

    let applications = store
        .resolve_rules_for_changed_claims(&[current_trigger], &existing_ids, 3)
        .unwrap();

    assert!(applications.is_empty());

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn dependency_completion_keeps_invalidation_from_a_different_trigger() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("different_transition_invalidation");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let claims = vec![
        make_claim(
            &scope,
            "claim_owner_current",
            "proyecto",
            "responsable",
            Some("Ana"),
            20,
            (ClaimKind::Fact, ClaimPolarity::Affirmative),
        ),
        make_claim(
            &scope,
            "claim_budget_current",
            "proyecto",
            "presupuesto",
            Some("revisado"),
            21,
            (ClaimKind::Fact, ClaimPolarity::Affirmative),
        ),
        make_claim(
            &scope,
            "claim_schedule_old",
            "proyecto",
            "calendario",
            Some("anterior"),
            10,
            (ClaimKind::Fact, ClaimPolarity::Affirmative),
        ),
    ];
    for claim in [
        make_claim(
            &scope,
            "claim_owner_previous",
            "proyecto",
            "responsable",
            Some("Bea"),
            5,
            (ClaimKind::Fact, ClaimPolarity::Affirmative),
        ),
        make_claim(
            &scope,
            "claim_budget_previous",
            "proyecto",
            "presupuesto",
            Some("anterior"),
            5,
            (ClaimKind::Fact, ClaimPolarity::Affirmative),
        ),
    ] {
        store.append_claim(&claim, None).unwrap();
    }
    for claim in &claims {
        store.append_claim(claim, None).unwrap();
    }
    let derive = store
        .add_rule(exact_dependency_rule(
            "rule_owner_derives_schedule",
            "proyecto",
            "responsable",
            "proyecto",
            "calendario",
            RuleAction::DeriveValue,
            Some("nuevo"),
        ))
        .unwrap();
    let invalidate = store
        .add_rule(exact_dependency_rule(
            "rule_budget_invalidates_schedule",
            "proyecto",
            "presupuesto",
            "proyecto",
            "calendario",
            RuleAction::MarkUnsupported,
            None,
        ))
        .unwrap();

    let completion = store
        .complete_target_dependency_chains(&[derive, invalidate], &claims, &[], 3)
        .unwrap();

    assert!(completion.outcomes.iter().any(|outcome| {
        outcome.rule_id == "rule_owner_derives_schedule"
            && matches!(outcome.status, RuleResolutionStatus::AppliedDerived)
    }));
    assert!(completion.outcomes.iter().any(|outcome| {
        outcome.rule_id == "rule_budget_invalidates_schedule"
            && matches!(outcome.status, RuleResolutionStatus::AppliedUnsupported)
    }));

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn dependency_completion_keeps_invalidation_when_derivation_has_no_value() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("unmaterialized_derivation_invalidation");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let claims = vec![
        make_claim(
            &scope,
            "claim_owner_without_value",
            "équipe",
            "responsable",
            None,
            20,
            (ClaimKind::Fact, ClaimPolarity::Affirmative),
        ),
        make_claim(
            &scope,
            "claim_reviewer_old",
            "équipe",
            "réviseur",
            Some("ancien"),
            10,
            (ClaimKind::Fact, ClaimPolarity::Affirmative),
        ),
    ];
    store
        .append_claim(
            &make_claim(
                &scope,
                "claim_owner_previous_value",
                "équipe",
                "responsable",
                Some("ancien"),
                5,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            ),
            None,
        )
        .unwrap();
    for claim in &claims {
        store.append_claim(claim, None).unwrap();
    }
    let derive = store
        .add_rule(exact_dependency_rule(
            "rule_owner_derives_reviewer",
            "équipe",
            "responsable",
            "équipe",
            "réviseur",
            RuleAction::DeriveValue,
            None,
        ))
        .unwrap();
    let invalidate = store
        .add_rule(exact_dependency_rule(
            "rule_owner_invalidates_reviewer",
            "équipe",
            "responsable",
            "équipe",
            "réviseur",
            RuleAction::MarkUnsupported,
            None,
        ))
        .unwrap();

    let completion = store
        .complete_target_dependency_chains(&[derive, invalidate], &claims, &[], 3)
        .unwrap();

    assert!(!completion.outcomes.iter().any(|outcome| {
        outcome.rule_id == "rule_owner_derives_reviewer"
            && matches!(outcome.status, RuleResolutionStatus::AppliedDerived)
    }));
    assert!(completion.outcomes.iter().any(|outcome| {
        outcome.rule_id == "rule_owner_invalidates_reviewer"
            && matches!(outcome.status, RuleResolutionStatus::AppliedUnsupported)
    }));

    let _ = std::fs::remove_dir_all(&store.path);
}

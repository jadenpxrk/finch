use super::*;

#[test]
fn memory_store_persists_manual_claims_by_scope_and_time() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("claims");
    let store = crate::test_store(&dir, 3).unwrap();
    let mut user_1 = scope();
    user_1.user_id = Some("user_1".to_string());
    store
        .add_manual_claim(
            ManualClaimInput {
                id: Some("claim_user_1".to_string()),
                scope: user_1.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                claim_text: "User prefers compact Rust examples.".to_string(),
                subject: Some("user".to_string()),
                predicate: Some("prefers".to_string()),
                object_value: Some("compact Rust examples".to_string()),
                claim_kind: ClaimKind::Preference,
                polarity: ClaimPolarity::Affirmative,
                source_span_ids: Vec::new(),
                source_episode_ids: Vec::new(),
                asserted_by: "user".to_string(),
                confidence: Some(1.0),
                observed_at_ms: 10,
                valid_from_ms: Some(10),
                valid_to_ms: None,
            },
            Some(&[1.0, 0.0, 0.0]),
        )
        .unwrap();
    store
        .add_manual_claim(
            ManualClaimInput {
                id: Some("claim_user_2".to_string()),
                scope: {
                    let mut scope = scope();
                    scope.user_id = Some("user_2".to_string());
                    scope
                },
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                claim_text: "Other user prefers verbose examples.".to_string(),
                subject: Some("user".to_string()),
                predicate: Some("prefers".to_string()),
                object_value: Some("verbose examples".to_string()),
                claim_kind: ClaimKind::Preference,
                polarity: ClaimPolarity::Affirmative,
                source_span_ids: Vec::new(),
                source_episode_ids: Vec::new(),
                asserted_by: "user".to_string(),
                confidence: Some(1.0),
                observed_at_ms: 10,
                valid_from_ms: Some(10),
                valid_to_ms: Some(20),
            },
            Some(&[0.0, 1.0, 0.0]),
        )
        .unwrap();
    drop(store);

    let reopened = crate::reopen_test_store(&dir).unwrap();
    let current = reopened.scan_claims(&user_1, 100, Some(30)).unwrap();
    assert_eq!(current.len(), 1);
    assert_eq!(current[0].id, "claim_user_1");
    assert!(current[0].claim_text.contains("compact"));
}

#[test]
fn memory_store_resolves_current_claims_with_latest_and_corrections() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("current_claims");
    let store = EvidencedStore::create(&dir, 3).unwrap();
    let mut user = scope();
    user.user_id = Some("user_claim_current".to_string());

    for (id, city, at) in [
        ("claim_city_old", "Toronto", 10),
        ("claim_city_new", "Vancouver", 20),
    ] {
        store
            .add_manual_claim(
                ManualClaimInput {
                    id: Some(id.to_string()),
                    scope: user.clone(),
                    visibility: Visibility::Private,
                    policy_tags: Vec::new(),
                    claim_text: format!("User lives in {city}."),
                    subject: Some("user".to_string()),
                    predicate: Some("lives_in".to_string()),
                    object_value: Some(city.to_string()),
                    claim_kind: ClaimKind::Fact,
                    polarity: ClaimPolarity::Affirmative,
                    source_span_ids: Vec::new(),
                    source_episode_ids: Vec::new(),
                    asserted_by: "user".to_string(),
                    confidence: Some(1.0),
                    observed_at_ms: at,
                    valid_from_ms: Some(at),
                    valid_to_ms: None,
                },
                None,
            )
            .unwrap();
    }

    let historical = store.scan_current_claims(&user, 10, Some(15)).unwrap();
    assert_eq!(historical.len(), 1);
    assert_eq!(historical[0].object_value.as_deref(), Some("Toronto"));

    let current = store.scan_current_claims(&user, 10, Some(21)).unwrap();
    assert_eq!(current.len(), 1);
    assert_eq!(current[0].object_value.as_deref(), Some("Vancouver"));

    let correction = crate::ingest::add_correction(
        CorrectionInput {
            id: Some("corr_claim_city_new".to_string()),
            scope: user.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            operation: CorrectionOperation::Tombstone,
            target_type: "claim".to_string(),
            target_ids: Vec::new(),
            target_selector: Some("Vancouver".to_string()),
            new_value: None,
            reason: Some("outdated".to_string()),
            actor: ActorKind::User,
            authority: CorrectionAuthority::User,
            effective_at_ms: Some(25),
            applies_valid_from_ms: None,
            applies_valid_to_ms: None,
            cascade_policy: None,
            metadata_json: None,
        },
        25,
    );
    store.add_correction(&correction).unwrap();

    let after_correction = store.scan_current_claims(&user, 10, Some(30)).unwrap();
    assert_eq!(after_correction.len(), 1);
    assert_eq!(after_correction[0].status, MemoryStatus::Tombstoned);
    assert_eq!(
        after_correction[0].object_value.as_deref(),
        Some("Vancouver")
    );

    let mut replace_scope = scope();
    replace_scope.user_id = Some("user_claim_replace".to_string());
    let mut old_owner = make_claim(
        &replace_scope,
        "claim_project_owner_morgan",
        "project",
        "owner",
        Some("Morgan"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    old_owner.valid_to_ms = Some(30);
    store.append_claim(&old_owner, None).unwrap();
    let before_replace = store
        .scan_current_claims(&replace_scope, 10, Some(15))
        .unwrap();
    assert_eq!(before_replace.len(), 1);
    assert_eq!(before_replace[0].object_value.as_deref(), Some("Morgan"));

    let replacement = crate::ingest::add_correction(
        CorrectionInput {
            id: Some("corr_project_owner_replace".to_string()),
            scope: replace_scope.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            operation: CorrectionOperation::Replace,
            target_type: "claim".to_string(),
            target_ids: vec!["claim_project_owner_morgan".to_string()],
            target_selector: None,
            new_value: Some("Dana".to_string()),
            reason: Some("owner changed".to_string()),
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
    store.add_correction(&replacement).unwrap();

    let replaced = store
        .scan_current_claims(&replace_scope, 10, Some(25))
        .unwrap();
    assert_eq!(replaced.len(), 1);
    assert_eq!(replaced[0].status, MemoryStatus::Active);
    assert_eq!(replaced[0].object_value.as_deref(), Some("Dana"));
    assert_eq!(replaced[0].asserted_by, "correction_replace");
    assert!(replaced[0].source_span_ids.is_empty());
    assert!(replaced[0].source_episode_ids.is_empty());
    assert_eq!(replaced[0].valid_from_ms, Some(20));
    assert_eq!(replaced[0].valid_to_ms, Some(30));
    assert_eq!(
        replaced[0].correction_ids,
        vec!["corr_project_owner_replace".to_string()]
    );
    assert!(store
        .scan_current_claims(&replace_scope, 10, Some(31))
        .unwrap()
        .is_empty());

    for (id, owner, predicate, at) in [
        ("claim_owner_old", "Eli", "Owner", 40),
        ("claim_owner_new", "Dana", " owner ", 50),
    ] {
        store
            .add_manual_claim(
                ManualClaimInput {
                    id: Some(id.to_string()),
                    scope: user.clone(),
                    visibility: Visibility::Private,
                    policy_tags: Vec::new(),
                    claim_text: format!("Project owner is {owner}."),
                    subject: Some("project".to_string()),
                    predicate: Some(predicate.to_string()),
                    object_value: Some(owner.to_string()),
                    claim_kind: ClaimKind::Fact,
                    polarity: ClaimPolarity::Affirmative,
                    source_span_ids: Vec::new(),
                    source_episode_ids: Vec::new(),
                    asserted_by: "user".to_string(),
                    confidence: Some(1.0),
                    observed_at_ms: at,
                    valid_from_ms: Some(at),
                    valid_to_ms: None,
                },
                None,
            )
            .unwrap();
    }
    let owner = store.scan_current_claims(&user, 10, Some(60)).unwrap();
    let owner_values = owner
        .iter()
        .filter(|claim| {
            claim.subject.as_deref() == Some("project")
                && claim.predicate.as_deref().map(str::trim) == Some("owner")
        })
        .map(|claim| claim.object_value.as_deref())
        .collect::<Vec<_>>();
    assert_eq!(owner_values, vec![Some("Dana")]);
}

#[test]
fn memory_store_current_claim_limit_applies_after_filtering() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("current_claim_limit_after_filtering");
    let store = EvidencedStore::create(&dir, 3).unwrap();
    let mut target_scope = scope();
    target_scope.user_id = Some("target_user".to_string());
    let mut other_scope = scope();
    other_scope.user_id = Some("other_user".to_string());

    for i in 0..MAX_VECTOR_QUERY_TOPK {
        let id = format!("claim_other_{i}");
        store
            .append_claim(
                &make_claim(
                    &other_scope,
                    &id,
                    "user",
                    "timezone",
                    Some("UTC"),
                    10,
                    (ClaimKind::Fact, ClaimPolarity::Affirmative),
                ),
                None,
            )
            .unwrap();
    }
    store
        .append_claim(
            &make_claim(
                &target_scope,
                "claim_target_timezone",
                "user",
                "timezone",
                Some("America/Toronto"),
                20,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            ),
            None,
        )
        .unwrap();

    let current = store
        .scan_current_claims(&target_scope, 1, Some(30))
        .unwrap();
    assert_eq!(current.len(), 1);
    assert_eq!(current[0].object_value.as_deref(), Some("America/Toronto"));
}

#[test]
fn memory_store_claim_tombstone_survives_off_scope_correction_saturation() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("claim_tombstone_scope_saturation");
    let store = EvidencedStore::create(&dir, 3).unwrap();
    let mut target_scope = scope();
    target_scope.user_id = Some("target_user".to_string());
    let mut other_scope = scope();
    other_scope.user_id = Some("other_user".to_string());

    store
        .append_claim(
            &make_claim(
                &target_scope,
                "claim_target_owner",
                "project",
                "owner",
                Some("Morgan"),
                10,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            ),
            None,
        )
        .unwrap();

    for i in 0..SATURATING_CORRECTION_COUNT {
        let correction = crate::ingest::add_correction(
            CorrectionInput {
                id: Some(format!("corr_other_{i}")),
                scope: other_scope.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                operation: CorrectionOperation::Assert,
                target_type: "claim".to_string(),
                target_ids: vec![format!("other_claim_{i}")],
                target_selector: None,
                new_value: None,
                reason: None,
                actor: ActorKind::User,
                authority: CorrectionAuthority::User,
                effective_at_ms: None,
                applies_valid_from_ms: None,
                applies_valid_to_ms: None,
                cascade_policy: None,
                metadata_json: None,
            },
            20,
        );
        store.add_correction(&correction).unwrap();
    }

    let tombstone = crate::ingest::add_correction(
        CorrectionInput {
            id: Some("corr_target_owner_deleted".to_string()),
            scope: target_scope.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            operation: CorrectionOperation::Tombstone,
            target_type: "claim".to_string(),
            target_ids: vec!["claim_target_owner".to_string()],
            target_selector: None,
            new_value: None,
            reason: None,
            actor: ActorKind::User,
            authority: CorrectionAuthority::User,
            effective_at_ms: None,
            applies_valid_from_ms: None,
            applies_valid_to_ms: None,
            cascade_policy: None,
            metadata_json: None,
        },
        21,
    );
    store.add_correction(&tombstone).unwrap();

    let current = store
        .scan_current_claims(&target_scope, 10, Some(22))
        .unwrap();
    assert_eq!(current.len(), 1);
    assert_eq!(current[0].id, "claim_target_owner");
    assert_eq!(current[0].status, MemoryStatus::Tombstoned);
}

#[test]
fn memory_store_claim_tombstone_survives_future_correction_saturation() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("claim_tombstone_future_saturation");
    let store = EvidencedStore::create(&dir, 3).unwrap();
    let mut target_scope = scope();
    target_scope.user_id = Some("target_user".to_string());

    store
        .append_claim(
            &make_claim(
                &target_scope,
                "claim_target_owner",
                "project",
                "owner",
                Some("Morgan"),
                10,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            ),
            None,
        )
        .unwrap();

    for i in 0..SATURATING_CORRECTION_COUNT {
        let correction = crate::ingest::add_correction(
            CorrectionInput {
                id: Some(format!("corr_future_{i}")),
                scope: target_scope.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                operation: CorrectionOperation::Assert,
                target_type: "claim".to_string(),
                target_ids: vec!["claim_target_owner".to_string()],
                target_selector: None,
                new_value: None,
                reason: None,
                actor: ActorKind::User,
                authority: CorrectionAuthority::User,
                effective_at_ms: Some(100),
                applies_valid_from_ms: None,
                applies_valid_to_ms: None,
                cascade_policy: None,
                metadata_json: None,
            },
            100,
        );
        store.add_correction(&correction).unwrap();
    }

    let tombstone = crate::ingest::add_correction(
        CorrectionInput {
            id: Some("corr_target_owner_deleted".to_string()),
            scope: target_scope.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            operation: CorrectionOperation::Tombstone,
            target_type: "claim".to_string(),
            target_ids: vec!["claim_target_owner".to_string()],
            target_selector: None,
            new_value: None,
            reason: None,
            actor: ActorKind::User,
            authority: CorrectionAuthority::User,
            effective_at_ms: Some(21),
            applies_valid_from_ms: None,
            applies_valid_to_ms: None,
            cascade_policy: None,
            metadata_json: None,
        },
        21,
    );
    store.add_correction(&tombstone).unwrap();

    let current = store
        .scan_current_claims(&target_scope, 10, Some(22))
        .unwrap();
    assert_eq!(current.len(), 1);
    assert_eq!(current[0].id, "claim_target_owner");
    assert_eq!(current[0].status, MemoryStatus::Tombstoned);
}

#[test]
fn memory_store_claim_tombstone_survives_same_scope_unrelated_correction_saturation() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("claim_tombstone_unrelated_saturation");
    let store = EvidencedStore::create(&dir, 3).unwrap();
    let mut target_scope = scope();
    target_scope.user_id = Some("target_user".to_string());

    store
        .append_claim(
            &make_claim(
                &target_scope,
                "claim_target_owner",
                "project",
                "owner",
                Some("Morgan"),
                10,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            ),
            None,
        )
        .unwrap();

    for i in 0..SATURATING_CORRECTION_COUNT {
        let correction = crate::ingest::add_correction(
            CorrectionInput {
                id: Some(format!("corr_unrelated_{i}")),
                scope: target_scope.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                operation: CorrectionOperation::Assert,
                target_type: "claim".to_string(),
                target_ids: vec![format!("claim_unrelated_{i}")],
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

    let tombstone = crate::ingest::add_correction(
        CorrectionInput {
            id: Some("corr_target_owner_deleted".to_string()),
            scope: target_scope.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            operation: CorrectionOperation::Tombstone,
            target_type: "claim".to_string(),
            target_ids: vec!["claim_target_owner".to_string()],
            target_selector: None,
            new_value: None,
            reason: None,
            actor: ActorKind::User,
            authority: CorrectionAuthority::User,
            effective_at_ms: Some(21),
            applies_valid_from_ms: None,
            applies_valid_to_ms: None,
            cascade_policy: None,
            metadata_json: None,
        },
        21,
    );
    store.add_correction(&tombstone).unwrap();

    let current = store
        .scan_current_claims(&target_scope, 10, Some(22))
        .unwrap();
    assert_eq!(current.len(), 1);
    assert_eq!(current[0].id, "claim_target_owner");
    assert_eq!(current[0].status, MemoryStatus::Tombstoned);
}

#[test]
fn memory_store_claim_selector_tombstone_survives_unrelated_correction_saturation() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("claim_selector_tombstone_saturation");
    let store = EvidencedStore::create(&dir, 3).unwrap();
    let mut target_scope = scope();
    target_scope.user_id = Some("target_user".to_string());

    store
        .append_claim(
            &make_claim(
                &target_scope,
                "claim_target_owner",
                "project",
                "owner",
                Some("Morgan"),
                10,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            ),
            None,
        )
        .unwrap();

    for i in 0..SATURATING_CORRECTION_COUNT {
        let correction = crate::ingest::add_correction(
            CorrectionInput {
                id: Some(format!("corr_unrelated_selector_blocker_{i}")),
                scope: target_scope.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                operation: CorrectionOperation::Assert,
                target_type: "claim".to_string(),
                target_ids: vec![format!("claim_unrelated_{i}")],
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

    let tombstone = crate::ingest::add_correction(
        CorrectionInput {
            id: Some("corr_target_owner_selector_deleted".to_string()),
            scope: target_scope.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            operation: CorrectionOperation::Tombstone,
            target_type: "claim".to_string(),
            target_ids: Vec::new(),
            target_selector: Some("Morgan".to_string()),
            new_value: None,
            reason: None,
            actor: ActorKind::User,
            authority: CorrectionAuthority::User,
            effective_at_ms: Some(21),
            applies_valid_from_ms: None,
            applies_valid_to_ms: None,
            cascade_policy: None,
            metadata_json: None,
        },
        21,
    );
    store.add_correction(&tombstone).unwrap();

    let current = store
        .scan_current_claims(&target_scope, 10, Some(22))
        .unwrap();
    assert_eq!(current.len(), 1);
    assert_eq!(current[0].id, "claim_target_owner");
    assert_eq!(current[0].status, MemoryStatus::Tombstoned);
}

#[test]
fn memory_store_context_corrections_are_targeted_to_claims_and_spans() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("context_corrections_targeted");
    let store = EvidencedStore::create(&dir, 3).unwrap();
    let mut target_scope = scope();
    target_scope.user_id = Some("target_user".to_string());

    let claim = make_claim(
        &target_scope,
        "claim_target_owner",
        "project",
        "owner",
        Some("Morgan"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    store.append_claim(&claim, None).unwrap();
    let mut span = span_record("span_target_owner", MemoryStatus::Active, Some(10));
    span.scope = target_scope.clone();
    store.append_span(&span, Some(&[1.0, 0.0, 0.0])).unwrap();

    for i in 0..SATURATING_CORRECTION_COUNT {
        let correction = crate::ingest::add_correction(
            CorrectionInput {
                id: Some(format!("corr_unrelated_context_{i}")),
                scope: target_scope.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                operation: CorrectionOperation::Assert,
                target_type: if i % 2 == 0 {
                    "claim".to_string()
                } else {
                    "span".to_string()
                },
                target_ids: vec![format!("unrelated_target_{i}")],
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

    for (id, target_type, target_id) in [
        ("corr_claim_target_deleted", "claim", "claim_target_owner"),
        ("corr_span_target_deleted", "span", "span_target_owner"),
    ] {
        let correction = crate::ingest::add_correction(
            CorrectionInput {
                id: Some(id.to_string()),
                scope: target_scope.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                operation: CorrectionOperation::Tombstone,
                target_type: target_type.to_string(),
                target_ids: vec![target_id.to_string()],
                target_selector: None,
                new_value: None,
                reason: None,
                actor: ActorKind::User,
                authority: CorrectionAuthority::User,
                effective_at_ms: Some(21),
                applies_valid_from_ms: None,
                applies_valid_to_ms: None,
                cascade_policy: None,
                metadata_json: None,
            },
            21,
        );
        store.add_correction(&correction).unwrap();
    }

    let corrections = store
        .scan_context_corrections(
            &target_scope,
            std::slice::from_ref(&claim),
            &[SpanSearchHit {
                span: span.clone(),
                score: 1.0,
            }],
            Some(22),
        )
        .unwrap();
    let ids = corrections
        .iter()
        .map(|correction| correction.id.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        ids,
        vec!["corr_claim_target_deleted", "corr_span_target_deleted"]
    );
}

#[test]
fn memory_store_keeps_relationship_claims_as_set_members() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("relationship_set_claims");
    let store = crate::test_store(&dir, 3).unwrap();
    let mut user = scope();
    user.user_id = Some("user_claim_set".to_string());

    for (id, club, at) in [
        ("claim_club_salsa", "salsa club", 10),
        ("claim_club_chess", "chess club", 20),
    ] {
        store
            .add_manual_claim(
                ManualClaimInput {
                    id: Some(id.to_string()),
                    scope: user.clone(),
                    visibility: Visibility::Private,
                    policy_tags: Vec::new(),
                    claim_text: format!("User joined the {club}."),
                    subject: Some("user".to_string()),
                    predicate: Some("joined_club".to_string()),
                    object_value: Some(club.to_string()),
                    claim_kind: ClaimKind::Relationship,
                    polarity: ClaimPolarity::Affirmative,
                    source_span_ids: Vec::new(),
                    source_episode_ids: Vec::new(),
                    asserted_by: "user".to_string(),
                    confidence: Some(1.0),
                    observed_at_ms: at,
                    valid_from_ms: Some(at),
                    valid_to_ms: None,
                },
                None,
            )
            .unwrap();
    }

    let current = store.scan_current_claims(&user, 10, Some(25)).unwrap();
    assert_eq!(current.len(), 2);
    assert!(current
        .iter()
        .any(|claim| claim.object_value.as_deref() == Some("salsa club")));
    assert!(current
        .iter()
        .any(|claim| claim.object_value.as_deref() == Some("chess club")));

    let correction = crate::ingest::add_correction(
        CorrectionInput {
            id: Some("corr_salsa_removed".to_string()),
            scope: user.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            operation: CorrectionOperation::Tombstone,
            target_type: "claim".to_string(),
            target_ids: Vec::new(),
            target_selector: Some("salsa".to_string()),
            new_value: None,
            reason: Some("removed".to_string()),
            actor: ActorKind::User,
            authority: CorrectionAuthority::User,
            effective_at_ms: Some(30),
            applies_valid_from_ms: None,
            applies_valid_to_ms: None,
            cascade_policy: None,
            metadata_json: None,
        },
        30,
    );
    store.add_correction(&correction).unwrap();

    let after_correction = store.scan_current_claims(&user, 10, Some(31)).unwrap();
    assert_eq!(after_correction.len(), 2);
    assert!(after_correction.iter().any(|claim| {
        claim.object_value.as_deref() == Some("salsa club")
            && claim.status == MemoryStatus::Tombstoned
    }));
    assert!(after_correction.iter().any(|claim| {
        claim.object_value.as_deref() == Some("chess club") && claim.status == MemoryStatus::Active
    }));

    store
        .add_manual_claim(
            ManualClaimInput {
                id: Some("claim_club_salsa_readded".to_string()),
                scope: user.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                claim_text: "User rejoined the salsa club.".to_string(),
                subject: Some("user".to_string()),
                predicate: Some("joined_club".to_string()),
                object_value: Some("salsa club".to_string()),
                claim_kind: ClaimKind::Relationship,
                polarity: ClaimPolarity::Affirmative,
                source_span_ids: Vec::new(),
                source_episode_ids: Vec::new(),
                asserted_by: "user".to_string(),
                confidence: Some(1.0),
                observed_at_ms: 35,
                valid_from_ms: Some(35),
                valid_to_ms: None,
            },
            None,
        )
        .unwrap();
    let after_readd = store.scan_current_claims(&user, 10, Some(40)).unwrap();
    assert!(after_readd.iter().any(|claim| {
        claim.object_value.as_deref() == Some("salsa club") && claim.status == MemoryStatus::Active
    }));
    assert!(after_readd.iter().any(|claim| {
        claim.object_value.as_deref() == Some("chess club") && claim.status == MemoryStatus::Active
    }));

    let future_range_correction = crate::ingest::add_correction(
        CorrectionInput {
            id: Some("corr_salsa_future_removed".to_string()),
            scope: user.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            operation: CorrectionOperation::Tombstone,
            target_type: "claim".to_string(),
            target_ids: Vec::new(),
            target_selector: Some("salsa".to_string()),
            new_value: None,
            reason: Some("removed again".to_string()),
            actor: ActorKind::User,
            authority: CorrectionAuthority::User,
            effective_at_ms: Some(30),
            applies_valid_from_ms: Some(35),
            applies_valid_to_ms: None,
            cascade_policy: None,
            metadata_json: None,
        },
        45,
    );
    store.add_correction(&future_range_correction).unwrap();
    let after_future_range_correction = store.scan_current_claims(&user, 10, Some(46)).unwrap();
    assert!(after_future_range_correction.iter().any(|claim| {
        claim.object_value.as_deref() == Some("salsa club")
            && claim.status == MemoryStatus::Tombstoned
    }));
}

#[test]
fn memory_store_keeps_relationship_and_event_claims_separate_in_projection() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("relationship_event_projection");
    let store = crate::test_store(&dir, 3).unwrap();
    let mut user = scope();
    user.user_id = Some("user_relation_event".to_string());

    store
        .add_manual_claim(
            ManualClaimInput {
                id: Some("claim_relationship_member".to_string()),
                scope: user.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                claim_text: "Project has member Dana.".to_string(),
                subject: Some("project".to_string()),
                predicate: Some("member".to_string()),
                object_value: Some("Dana".to_string()),
                claim_kind: ClaimKind::Relationship,
                polarity: ClaimPolarity::Affirmative,
                source_span_ids: Vec::new(),
                source_episode_ids: Vec::new(),
                asserted_by: "user".to_string(),
                confidence: Some(1.0),
                observed_at_ms: 10,
                valid_from_ms: Some(10),
                valid_to_ms: None,
            },
            None,
        )
        .unwrap();

    store
        .add_manual_claim(
            ManualClaimInput {
                id: Some("claim_event_member".to_string()),
                scope: user.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                claim_text: "Project kickoff event included Dana.".to_string(),
                subject: Some("project".to_string()),
                predicate: Some("member".to_string()),
                object_value: Some("Dana".to_string()),
                claim_kind: ClaimKind::Event,
                polarity: ClaimPolarity::Affirmative,
                source_span_ids: Vec::new(),
                source_episode_ids: Vec::new(),
                asserted_by: "user".to_string(),
                confidence: Some(1.0),
                observed_at_ms: 20,
                valid_from_ms: Some(20),
                valid_to_ms: None,
            },
            None,
        )
        .unwrap();

    let current = store.scan_current_claims(&user, 10, Some(25)).unwrap();
    let kinds = current
        .iter()
        .filter(|claim| claim.subject.as_deref() == Some("project"))
        .map(|claim| claim.claim_kind)
        .collect::<Vec<_>>();
    assert_eq!(kinds.len(), 2);
    assert!(kinds.contains(&ClaimKind::Event));
    assert!(kinds.contains(&ClaimKind::Relationship));
}

#[test]
fn proposed_claim_slot_binding_is_admitted_only_with_subject_identity_and_becomes_an_alias() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("admit_bound_claim_slots");
    let store = EvidencedStore::create(&dir, 3).unwrap();
    let scope = scope();
    let responsable = make_claim(
        &scope,
        "claim_proyecto_responsable",
        "proyecto",
        "responsable",
        Some("Ana"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    store.append_claim(&responsable, None).unwrap();
    let slot_id = store
        .claims_by_ids(&scope, std::slice::from_ref(&responsable.id))
        .unwrap()[0]
        .slot_id
        .clone()
        .unwrap();

    // Same subject, different predicate surface, bound by a binder proposal: admitted as identity.
    let mut encargada = make_claim(
        &scope,
        "claim_proyecto_encargada",
        "proyecto",
        "encargada",
        Some("Bea"),
        20,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    encargada.slot_id = Some(slot_id.clone());
    // Different subject bound to the same slot: rejected, keeps its own canonical slot.
    let mut cliente = make_claim(
        &scope,
        "claim_cliente_encargada",
        "cliente",
        "encargada",
        Some("Zoe"),
        20,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    cliente.slot_id = Some(slot_id.clone());
    store
        .apply_state_mutation_batch(StateMutationBatch {
            scope: scope.clone(),
            transaction_time_ms: 20,
            propagation_seed: "admit-bindings".to_string(),
            max_rule_hops: 8,
            claims: vec![encargada, cliente],
            entities: Vec::new(),
            rules: Vec::new(),
            slot_aliases: Vec::new(),
            corrections: Vec::new(),
        })
        .unwrap();

    let stored = store
        .claims_by_ids(
            &scope,
            &[
                "claim_proyecto_encargada".to_string(),
                "claim_cliente_encargada".to_string(),
            ],
        )
        .unwrap();
    let by_id = stored
        .iter()
        .map(|claim| (claim.id.as_str(), claim))
        .collect::<std::collections::BTreeMap<_, _>>();
    assert_eq!(
        by_id["claim_proyecto_encargada"].slot_id.as_deref(),
        Some(slot_id.as_str())
    );
    assert!(by_id["claim_proyecto_encargada"].slot_facet.is_none());
    assert_ne!(
        by_id["claim_cliente_encargada"].slot_id.as_deref(),
        Some(slot_id.as_str())
    );
    let aliases = store
        .scan_slot_aliases(&scope, usize::MAX, Some(20))
        .unwrap();
    assert!(aliases.iter().any(|alias| {
        alias.canonical_slot_id == slot_id
            && alias.alias_predicate_key == "encargada"
            && alias.source_claim_ids == vec!["claim_proyecto_encargada".to_string()]
    }));
    let current = store
        .scan_current_claims(&scope, usize::MAX, Some(20))
        .unwrap();
    let current_value = current
        .iter()
        .find(|claim| claim.slot_id.as_deref() == Some(slot_id.as_str()))
        .and_then(|claim| claim.object_value.clone());
    assert_eq!(current_value.as_deref(), Some("Bea"));

    // A later claim on the alias surface resolves to the same slot without any proposal.
    store
        .append_claim(
            &make_claim(
                &scope,
                "claim_proyecto_encargada_2",
                "proyecto",
                "encargada",
                Some("Cris"),
                30,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            ),
            None,
        )
        .unwrap();
    let later = store
        .claims_by_ids(&scope, &["claim_proyecto_encargada_2".to_string()])
        .unwrap();
    assert_eq!(later[0].slot_id.as_deref(), Some(slot_id.as_str()));

    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn current_claim_limit_applies_after_newest_version_wins() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("current_claim_limit_versions");
    let store = EvidencedStore::create(&dir, 3).unwrap();
    let scope = scope();
    for version in 1..=5 {
        let claim = make_claim(
            &scope,
            &format!("claim_owner_v{version}"),
            "project",
            "owner",
            Some(&format!("owner {version}")),
            version * 10,
            (ClaimKind::Fact, ClaimPolarity::Affirmative),
        );
        store.append_claim(&claim, None).unwrap();
    }

    let current = store.scan_current_claims(&scope, 1, None).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(
        current
            .iter()
            .map(|claim| claim.id.as_str())
            .collect::<Vec<_>>(),
        vec!["claim_owner_v5"]
    );
}

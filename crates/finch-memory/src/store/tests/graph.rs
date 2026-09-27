use super::*;

#[test]
fn memory_store_persists_profile_views_by_scope_and_time() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("profiles");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let mut user_1 = scope();
    user_1.user_id = Some("user_1".to_string());
    store
        .add_profile(ProfileInput {
            id: Some("profile_user_1".to_string()),
            scope: user_1.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            subject_id: Some("user_1".to_string()),
            profile_key: "style".to_string(),
            profile_text: "Prefers compact Rust examples.".to_string(),
            value_json: Some(r#""compact""#.to_string()),
            evidence_claim_ids: vec!["claim_user_1".to_string()],
            source_span_ids: Vec::new(),
            generated_at_ms: 20,
            generator_version: "manual".to_string(),
            valid_from_ms: Some(20),
            valid_to_ms: None,
            correction_watermark: 20,
        })
        .unwrap();
    store
        .add_profile(ProfileInput {
            id: Some("profile_user_2".to_string()),
            scope: {
                let mut scope = scope();
                scope.user_id = Some("user_2".to_string());
                scope
            },
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            subject_id: Some("user_2".to_string()),
            profile_key: "style".to_string(),
            profile_text: "Prefers verbose examples.".to_string(),
            value_json: Some(r#""verbose""#.to_string()),
            evidence_claim_ids: Vec::new(),
            source_span_ids: Vec::new(),
            generated_at_ms: 20,
            generator_version: "manual".to_string(),
            valid_from_ms: Some(20),
            valid_to_ms: Some(30),
            correction_watermark: 20,
        })
        .unwrap();
    drop(store);

    let reopened = MemoryStore::open(&dir, CollectionOptions::default()).unwrap();
    let profiles = reopened.scan_profiles(&user_1, 100, Some(40)).unwrap();
    assert_eq!(profiles.len(), 1);
    assert_eq!(profiles[0].id, "profile_user_1");
    assert!(profiles[0].profile_text.contains("compact"));
    let _ = std::fs::remove_dir_all(&reopened.path);
}

#[test]
fn memory_store_profile_limit_applies_after_scope_filtering() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("profile_scope_limit");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let mut target_scope = scope();
    target_scope.user_id = Some("target_user".to_string());
    let mut other_scope = scope();
    other_scope.user_id = Some("other_user".to_string());

    for id in ["profile_other_a", "profile_other_b"] {
        store
            .add_profile(ProfileInput {
                id: Some(id.to_string()),
                scope: other_scope.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                subject_id: Some("other_user".to_string()),
                profile_key: "style".to_string(),
                profile_text: "Prefers verbose examples.".to_string(),
                value_json: Some(r#""verbose""#.to_string()),
                evidence_claim_ids: Vec::new(),
                source_span_ids: Vec::new(),
                generated_at_ms: 10,
                generator_version: "manual".to_string(),
                valid_from_ms: Some(10),
                valid_to_ms: None,
                correction_watermark: 10,
            })
            .unwrap();
    }

    store
        .add_profile(ProfileInput {
            id: Some("profile_target_style".to_string()),
            scope: target_scope.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            subject_id: Some("target_user".to_string()),
            profile_key: "style".to_string(),
            profile_text: "Prefers concise state summaries.".to_string(),
            value_json: Some(r#""concise""#.to_string()),
            evidence_claim_ids: vec!["claim_target_style".to_string()],
            source_span_ids: Vec::new(),
            generated_at_ms: 20,
            generator_version: "manual".to_string(),
            valid_from_ms: Some(20),
            valid_to_ms: None,
            correction_watermark: 20,
        })
        .unwrap();

    let profiles = store.scan_profiles(&target_scope, 1, Some(30)).unwrap();
    assert_eq!(profiles.len(), 1);
    assert_eq!(profiles[0].id, "profile_target_style");
    assert!(profiles[0].profile_text.contains("concise"));

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn memory_store_persists_entities_and_expands_edges_one_hop() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("graph");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let mut user_1 = scope();
    user_1.user_id = Some("user_1".to_string());
    store
        .add_entity(
            EntityInput {
                id: Some("entity_finch".to_string()),
                scope: user_1.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                entity_type: "project".to_string(),
                canonical_name: "Finch".to_string(),
                aliases: vec!["finch-db".to_string()],
                source_claim_ids: Vec::new(),
                merge_parent_ids: Vec::new(),
                split_from_id: None,
                confidence: Some(1.0),
            },
            Some(&[1.0, 0.0, 0.0]),
        )
        .unwrap();
    store
        .add_entity(
            EntityInput {
                id: Some("entity_memory".to_string()),
                scope: user_1.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                entity_type: "feature".to_string(),
                canonical_name: "finch-memory".to_string(),
                aliases: Vec::new(),
                source_claim_ids: Vec::new(),
                merge_parent_ids: Vec::new(),
                split_from_id: None,
                confidence: Some(1.0),
            },
            Some(&[0.0, 1.0, 0.0]),
        )
        .unwrap();
    store
        .add_entity(
            EntityInput {
                id: Some("entity_context".to_string()),
                scope: user_1.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                entity_type: "concept".to_string(),
                canonical_name: "context packing".to_string(),
                aliases: Vec::new(),
                source_claim_ids: Vec::new(),
                merge_parent_ids: Vec::new(),
                split_from_id: None,
                confidence: Some(1.0),
            },
            Some(&[0.0, 0.0, 1.0]),
        )
        .unwrap();
    store
        .add_edge(EdgeInput {
            id: Some("edge_project_feature".to_string()),
            scope: user_1.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            src_entity_id: "entity_finch".to_string(),
            dst_entity_id: "entity_memory".to_string(),
            relation_type: "has_feature".to_string(),
            claim_id: None,
            source_span_ids: Vec::new(),
            valid_from_ms: Some(10),
            valid_to_ms: None,
            confidence: Some(0.9),
        })
        .unwrap();
    store
        .add_edge(EdgeInput {
            id: Some("edge_feature_concept".to_string()),
            scope: user_1.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            src_entity_id: "entity_memory".to_string(),
            dst_entity_id: "entity_context".to_string(),
            relation_type: "uses".to_string(),
            claim_id: None,
            source_span_ids: Vec::new(),
            valid_from_ms: Some(10),
            valid_to_ms: None,
            confidence: Some(0.8),
        })
        .unwrap();
    drop(store);

    let reopened = MemoryStore::open(&dir, CollectionOptions::default()).unwrap();
    let entities = reopened.scan_entities(&user_1, 100).unwrap();
    assert_eq!(entities.len(), 3);
    let edges = reopened
        .expand_edges_one_hop(&user_1, &["entity_finch".to_string()], 100, Some(20))
        .unwrap();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].dst_entity_id, "entity_memory");
    let ranked = reopened
        .expand_edges_ranked(&user_1, &["entity_finch".to_string()], 2, 100, Some(20))
        .unwrap();
    assert_eq!(ranked.len(), 2);
    assert_eq!(ranked[0].edge.id, "edge_project_feature");
    assert_eq!(ranked[0].depth, 1);
    assert_eq!(ranked[1].edge.id, "edge_feature_concept");
    assert_eq!(ranked[1].depth, 2);
    let _ = std::fs::remove_dir_all(&reopened.path);
}

#[test]
fn memory_store_edge_limit_applies_after_scope_filtering() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("edge_scope_limit");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let mut target_scope = scope();
    target_scope.user_id = Some("target_user".to_string());
    let mut other_scope = scope();
    other_scope.user_id = Some("other_user".to_string());

    for i in 0..MAX_VECTOR_QUERY_TOPK {
        store
            .add_edge(EdgeInput {
                id: Some(format!("edge_other_{i}")),
                scope: other_scope.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                src_entity_id: "entity_seed".to_string(),
                dst_entity_id: format!("entity_other_{i}"),
                relation_type: "mentions".to_string(),
                claim_id: None,
                source_span_ids: Vec::new(),
                valid_from_ms: Some(10),
                valid_to_ms: None,
                confidence: Some(0.5),
            })
            .unwrap();
    }
    store
        .add_edge(EdgeInput {
            id: Some("edge_target_relation".to_string()),
            scope: target_scope.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            src_entity_id: "entity_seed".to_string(),
            dst_entity_id: "entity_target".to_string(),
            relation_type: "owns".to_string(),
            claim_id: None,
            source_span_ids: Vec::new(),
            valid_from_ms: Some(20),
            valid_to_ms: None,
            confidence: Some(1.0),
        })
        .unwrap();

    let edges = store
        .expand_edges_one_hop(&target_scope, &["entity_seed".to_string()], 1, Some(30))
        .unwrap();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].id, "edge_target_relation");
    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn memory_store_entity_limit_applies_after_scope_filtering() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("entity_scope_limit");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let mut target_scope = scope();
    target_scope.user_id = Some("target_user".to_string());
    let mut other_scope = scope();
    other_scope.user_id = Some("other_user".to_string());

    for id in ["entity_other_a", "entity_other_b"] {
        store
            .add_entity(
                EntityInput {
                    id: Some(id.to_string()),
                    scope: other_scope.clone(),
                    visibility: Visibility::Private,
                    policy_tags: Vec::new(),
                    entity_type: "project".to_string(),
                    canonical_name: id.to_string(),
                    aliases: Vec::new(),
                    source_claim_ids: vec![format!("claim_{id}")],
                    merge_parent_ids: Vec::new(),
                    split_from_id: None,
                    confidence: Some(1.0),
                },
                None,
            )
            .unwrap();
    }

    store
        .add_entity(
            EntityInput {
                id: Some("entity_finch_memory".to_string()),
                scope: target_scope.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                entity_type: "project".to_string(),
                canonical_name: "Finch Memory".to_string(),
                aliases: vec!["memory substrate".to_string()],
                source_claim_ids: vec!["claim_memory_alias".to_string()],
                merge_parent_ids: Vec::new(),
                split_from_id: None,
                confidence: Some(1.0),
            },
            None,
        )
        .unwrap();

    let entities = store.scan_entities(&target_scope, 1).unwrap();
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0].id, "entity_finch_memory");
    assert_eq!(entities[0].aliases, vec!["memory substrate".to_string()]);

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn entity_upsert_does_not_overwrite_another_tenant() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("tenant_entity");
    let store = MemoryStore::create(&path, 4, CollectionOptions::default()).unwrap();
    let mut scope_a = MemoryScope::new("shared-space");
    scope_a.tenant_id = Some("tenant-a".into());
    let mut scope_b = scope_a.clone();
    scope_b.tenant_id = Some("tenant-b".into());
    for scope in [&scope_a, &scope_b] {
        let claim = store
            .add_manual_claim(
                ManualClaimInput {
                    id: Some(format!("evidence-{}", scope.tenant_id.as_ref().unwrap())),
                    scope: scope.clone(),
                    visibility: Visibility::Private,
                    policy_tags: vec![],
                    claim_text: "Alex is an employee".into(),
                    subject: None,
                    predicate: None,
                    object_value: None,
                    claim_kind: ClaimKind::Fact,
                    polarity: ClaimPolarity::Affirmative,
                    source_span_ids: vec![],
                    source_episode_ids: vec![],
                    asserted_by: "review".into(),
                    confidence: None,
                    observed_at_ms: 1,
                    valid_from_ms: None,
                    valid_to_ms: None,
                },
                None,
            )
            .unwrap();
        store
            .apply_state_mutation_batch(StateMutationBatch {
                scope: scope.clone(),
                transaction_time_ms: 1,
                propagation_seed: "review".into(),
                max_rule_hops: 1,
                claims: vec![],
                rules: vec![],
                slot_aliases: vec![],
                corrections: vec![],
                entities: vec![EntityInput {
                    id: None,
                    scope: scope.clone(),
                    visibility: Visibility::Private,
                    policy_tags: vec![],
                    entity_type: "person".into(),
                    canonical_name: "Alex".into(),
                    aliases: vec![],
                    source_claim_ids: vec![claim.id],
                    merge_parent_ids: vec![],
                    split_from_id: None,
                    confidence: None,
                }],
            })
            .unwrap();
        assert_eq!(store.scan_entities(scope, 10).unwrap().len(), 1);
    }
    let remaining_a = store.scan_entities(&scope_a, 10).unwrap().len();
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
    assert_eq!(
        remaining_a, 1,
        "tenant B silently replaced tenant A's entity"
    );
}

#[test]
fn explicit_entity_id_cannot_take_over_another_scope() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("entity_id_scope");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let entity = |tenant: &str| {
        let mut scope = scope();
        scope.tenant_id = Some(tenant.to_string());
        EntityInput {
            id: Some("entity_alex".to_string()),
            scope,
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            entity_type: "person".to_string(),
            canonical_name: "Alex".to_string(),
            aliases: Vec::new(),
            source_claim_ids: Vec::new(),
            merge_parent_ids: Vec::new(),
            split_from_id: None,
            confidence: None,
        }
    };
    let owner = store.add_entity(entity("tenant-a"), None).unwrap();

    let err = store.add_entity(entity("tenant-b"), None).unwrap_err();
    assert!(err.is_already_exists(), "{}", err.message);
    assert_eq!(store.scan_entities(&owner.scope, 10).unwrap().len(), 1);

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn explicit_slot_alias_id_cannot_take_over_another_scope() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("slot_alias_id_scope");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let add_alias = |tenant: &str| {
        let mut scope = scope();
        scope.tenant_id = Some(tenant.to_string());
        let target = make_claim(
            &scope,
            &format!("claim_target_{tenant}"),
            "project",
            "billing contact",
            Some("Alice"),
            10,
            (ClaimKind::Fact, ClaimPolarity::Affirmative),
        );
        let evidence = make_claim(
            &scope,
            &format!("claim_evidence_{tenant}"),
            "schema",
            "equivalence",
            Some("billing owner = billing contact"),
            11,
            (ClaimKind::Fact, ClaimPolarity::Affirmative),
        );
        store.append_claim(&target, None).unwrap();
        store.append_claim(&evidence, None).unwrap();
        let alias = store.add_slot_alias(
            SlotAliasInput {
                id: Some("slot_alias_shared".to_string()),
                scope: scope.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                alias_subject: "project".to_string(),
                alias_predicate: "billing owner".to_string(),
                canonical_slot_id: None,
                target_claim_id: Some(target.id),
                source_claim_ids: vec![evidence.id],
                valid_from_ms: Some(11),
                valid_to_ms: None,
            },
            11,
        );
        (scope, alias)
    };
    let (scope_a, owner) = add_alias("tenant-a");
    owner.unwrap();

    let (_, err) = add_alias("tenant-b");
    let err = err.unwrap_err();
    assert!(err.is_already_exists(), "{}", err.message);
    assert_eq!(
        store.scan_slot_aliases(&scope_a, 10, None).unwrap().len(),
        1,
        "tenant A's alias was overwritten"
    );

    let _ = std::fs::remove_dir_all(&store.path);
}

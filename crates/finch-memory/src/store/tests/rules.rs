use super::*;

#[test]
fn memory_store_rejects_rules_without_source_evidence() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("rule_source_evidence");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();

    let mut input = make_derived_rule_inputs();
    input.source_span_ids = Vec::new();
    assert!(store.add_rule(input).is_err());

    let mut second = make_derived_rule_inputs();
    second.id = Some("rule_no_episode".to_string());
    second.source_episode_ids = Vec::new();
    assert!(store.add_rule(second).is_err());

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn memory_store_stores_explicit_rules_with_provenance() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("rule_with_provenance");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();

    let input = make_derived_rule_inputs();
    let rule = store.add_rule(input.clone()).unwrap();
    let rules = store.scan_rules(&scope(), 10, None).unwrap();

    assert_eq!(rules.len(), 1);
    assert_eq!(rules[0].id, rule.id);
    assert_eq!(
        rules[0].source_span_ids,
        vec!["rule_span_owner".to_string()]
    );
    assert_eq!(rules[0].source_episode_ids, vec!["ep_owner".to_string()]);

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn memory_store_scan_rules_fails_on_undecodable_row() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("rule_decode_error");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();

    let rule = store.add_rule(make_derived_rule_inputs()).unwrap();
    let corrupt = rule_doc(&rule).unwrap().set("status", "not_a_status");
    let statuses = store.rules.upsert(vec![corrupt]).unwrap();
    assert!(statuses.iter().all(|status| status.is_ok()));

    assert!(store.scan_rules(&scope(), 10, None).is_err());

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn memory_store_persists_surface_bound_rule_before_registry_resolution() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("rule_surface_binding");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();

    let mut input = make_derived_rule_inputs();
    input.trigger_subject_key = "proyecto alfa".to_string();
    input.trigger_predicate_key = "responsable".to_string();
    input.target_subject_key = "proyecto alfa".to_string();
    input.target_predicate_key = "revisor".to_string();
    input.trigger_subject = Some("Proyecto Alfa".to_string());
    input.trigger_predicate = Some("responsable".to_string());
    input.target_subject = Some("Proyecto Alfa".to_string());
    input.target_predicate = Some("revisor".to_string());

    let rule = store.add_rule(input).unwrap();
    assert_eq!(rule.trigger_binding_status, RuleBindingStatus::Bound);
    assert_eq!(rule.target_binding_status, RuleBindingStatus::Bound);
    assert!(rule.trigger_slot_id.is_some());
    assert!(rule.target_slot_id.is_some());
    assert_eq!(rule.source_span_ids, vec!["rule_span_owner"]);
    let slots = store.scan_slots(&scope(), 10, None).unwrap();
    assert_eq!(slots.len(), 2);
    assert!(slots.iter().all(|slot| {
        slot.source_rule_ids == vec![rule.id.clone()]
            && slot.valid_from_ms.is_none()
            && slot.valid_to_ms.is_none()
    }));

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn memory_store_rebinds_rule_when_entity_alias_becomes_available() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("rule_deferred_entity_binding");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();

    let mut input = make_derived_rule_inputs();
    input.trigger_subject_key = "equipo norte".to_string();
    input.trigger_predicate_key = "responsable".to_string();
    input.target_subject_key = "equipo norte".to_string();
    input.target_predicate_key = "revisor".to_string();
    input.trigger_subject = Some("equipo norte".to_string());
    input.trigger_predicate = Some("responsable".to_string());
    input.target_subject = Some("equipo norte".to_string());
    input.target_predicate = Some("revisor".to_string());
    let before = store.add_rule(input).unwrap();
    assert_eq!(before.trigger_subject_entity_id, None);

    store
        .add_entity(
            EntityInput {
                id: Some("entity_equipo_norte".to_string()),
                scope: scope(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                entity_type: "team".to_string(),
                canonical_name: "Équipe Nord".to_string(),
                aliases: vec!["equipo norte".to_string()],
                source_claim_ids: vec!["claim_alias_equipo_norte".to_string()],
                merge_parent_ids: Vec::new(),
                split_from_id: None,
                confidence: Some(1.0),
            },
            None,
        )
        .unwrap();

    let after = store
        .scan_rules(&scope(), 10, None)
        .unwrap()
        .into_iter()
        .find(|rule| rule.id == before.id)
        .unwrap();
    assert_eq!(
        after.trigger_subject_entity_id.as_deref(),
        Some("entity_equipo_norte")
    );
    assert_eq!(
        after.target_subject_entity_id.as_deref(),
        Some("entity_equipo_norte")
    );
    assert_eq!(after.trigger_binding_status, RuleBindingStatus::Bound);
    assert_eq!(after.target_binding_status, RuleBindingStatus::Bound);
    assert_ne!(after.trigger_slot_id, before.trigger_slot_id);
    assert_eq!(after.source_span_ids, before.source_span_ids);

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn memory_store_resolver_applies_derive_rule_on_changed_trigger() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("rule_derive");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();

    store.add_rule(make_derived_rule_inputs()).unwrap();
    let scope = scope();
    let claim = make_claim(
        &scope,
        "claim_owner",
        "project",
        "owner",
        Some("dana"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );

    let applications = store
        .resolve_rules_for_changed_claims(&[claim], &BTreeSet::from(["claim_owner".to_string()]), 3)
        .unwrap();
    assert_eq!(applications.len(), 1);
    let derived = &applications[0].claim;
    assert_eq!(derived.subject.as_deref(), Some("project"));
    assert_eq!(derived.predicate.as_deref(), Some("reviewer"));
    assert_eq!(derived.object_value.as_deref(), Some("dana"));

    let _ = std::fs::remove_dir_all(&store.path);
}

fn ingest_sequenced_rule_evidence(
    store: &MemoryStore,
    id: &str,
    text: &str,
    sequence_no: i64,
    at_ms: i64,
) -> crate::ingest::IngestedEpisode {
    store
        .ingest_episode(
            crate::ingest::EpisodeInput {
                id: Some(id.to_string()),
                scope: scope(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                source_kind: SourceKind::UserMessage,
                actor: ActorKind::User,
                sequence_no,
                event_time_ms: Some(at_ms),
                valid_from_ms: Some(at_ms),
                valid_to_ms: None,
                raw_text: text.to_string(),
                blob_ref: None,
                mime_type: Some("text/plain".to_string()),
                causal_parent_ids: Vec::new(),
                metadata_json: None,
            },
            at_ms,
            &crate::ingest::ChunkOptions::default(),
        )
        .unwrap()
}

#[test]
fn on_change_rule_uses_episode_sequence_when_timestamps_are_equal() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("rule_equal_timestamp_sequence");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let at_ms = 100;
    let prior_evidence =
        ingest_sequenced_rule_evidence(&store, "episode_prior", "La responsable es Ana.", 0, at_ms);
    let rule_evidence = ingest_sequenced_rule_evidence(
        &store,
        "episode_rule",
        "Cuando cambie la responsable, la revisora seguirá ese valor.",
        1,
        at_ms,
    );
    let changed_evidence = ingest_sequenced_rule_evidence(
        &store,
        "episode_changed",
        "La responsable es Bea.",
        2,
        at_ms,
    );

    let mut prior = make_claim(
        &scope(),
        "claim_prior_owner",
        "proyecto",
        "responsable",
        Some("Ana"),
        at_ms,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    prior.source_span_ids = vec![prior_evidence.spans[0].id.clone()];
    prior.source_episode_ids = vec![prior_evidence.episode.id.clone()];
    store.append_claim(&prior, None).unwrap();

    let mut rule = make_derived_rule_inputs();
    rule.id = Some("rule_responsable_revisora".to_string());
    rule.trigger_subject_key = "proyecto".to_string();
    rule.trigger_predicate_key = "responsable".to_string();
    rule.target_subject_key = "proyecto".to_string();
    rule.target_predicate_key = "revisora".to_string();
    rule.trigger_subject = Some("proyecto".to_string());
    rule.trigger_predicate = Some("responsable".to_string());
    rule.target_subject = Some("proyecto".to_string());
    rule.target_predicate = Some("revisora".to_string());
    rule.activation = RuleActivation::OnChange;
    rule.valid_from_ms = Some(at_ms);
    rule.source_span_ids = vec![rule_evidence.spans[0].id.clone()];
    rule.source_episode_ids = vec![rule_evidence.episode.id.clone()];
    let stored_rule = store.add_rule(rule).unwrap();
    assert_eq!(stored_rule.source_sequence_no, Some(1));

    let mut changed = make_claim(
        &scope(),
        "claim_changed_owner",
        "proyecto",
        "responsable",
        Some("Bea"),
        at_ms,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    changed.source_span_ids = vec![changed_evidence.spans[0].id.clone()];
    changed.source_episode_ids = vec![changed_evidence.episode.id.clone()];
    store.append_claim(&changed, None).unwrap();

    let current = store
        .scan_current_claims(&scope(), usize::MAX, Some(at_ms))
        .unwrap();
    let reviewer = current
        .iter()
        .find(|claim| claim.predicate.as_deref() == Some("revisora"))
        .unwrap();
    assert_eq!(reviewer.object_value.as_deref(), Some("Bea"));
    assert_eq!(reviewer.source_sequence_no, Some(2));
    let owner = current
        .iter()
        .find(|claim| claim.predicate.as_deref() == Some("responsable"))
        .unwrap();
    assert_eq!(owner.object_value.as_deref(), Some("Bea"));

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn unsupported_rule_uses_episode_sequence_when_timestamps_are_equal() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("unsupported_equal_timestamp_sequence");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let at_ms = 200;
    let prior_evidence = ingest_sequenced_rule_evidence(
        &store,
        "episode_initial",
        "担当者は葵で、連絡先は凛です。",
        0,
        at_ms,
    );
    let rule_evidence = ingest_sequenced_rule_evidence(
        &store,
        "episode_dependency",
        "担当者が変わった場合、連絡先は再確認が必要です。",
        1,
        at_ms,
    );
    let changed_evidence = ingest_sequenced_rule_evidence(
        &store,
        "episode_update",
        "担当者は海になりました。",
        2,
        at_ms,
    );

    let mut owner = make_claim(
        &scope(),
        "claim_initial_owner",
        "案件",
        "担当者",
        Some("葵"),
        at_ms,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    owner.source_span_ids = vec![prior_evidence.spans[0].id.clone()];
    owner.source_episode_ids = vec![prior_evidence.episode.id.clone()];
    let mut contact = make_claim(
        &scope(),
        "claim_initial_contact",
        "案件",
        "連絡先",
        Some("凛"),
        at_ms,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    contact.source_span_ids = vec![prior_evidence.spans[0].id.clone()];
    contact.source_episode_ids = vec![prior_evidence.episode.id.clone()];
    store.append_claim(&owner, None).unwrap();
    store.append_claim(&contact, None).unwrap();

    let mut rule = make_derived_rule_inputs();
    rule.id = Some("rule_owner_contact".to_string());
    rule.trigger_subject_key = "案件".to_string();
    rule.trigger_predicate_key = "担当者".to_string();
    rule.target_subject_key = "案件".to_string();
    rule.target_predicate_key = "連絡先".to_string();
    rule.trigger_subject = Some("案件".to_string());
    rule.trigger_predicate = Some("担当者".to_string());
    rule.target_subject = Some("案件".to_string());
    rule.target_predicate = Some("連絡先".to_string());
    rule.activation = RuleActivation::OnChange;
    rule.action = RuleAction::MarkUnsupported;
    rule.value_template = None;
    rule.valid_from_ms = Some(at_ms);
    rule.source_span_ids = vec![rule_evidence.spans[0].id.clone()];
    rule.source_episode_ids = vec![rule_evidence.episode.id.clone()];
    store.add_rule(rule).unwrap();

    let mut changed = make_claim(
        &scope(),
        "claim_changed_owner",
        "案件",
        "担当者",
        Some("海"),
        at_ms,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    changed.source_span_ids = vec![changed_evidence.spans[0].id.clone()];
    changed.source_episode_ids = vec![changed_evidence.episode.id.clone()];
    store.append_claim(&changed, None).unwrap();

    let current = store
        .scan_current_claims(&scope(), usize::MAX, Some(at_ms))
        .unwrap();
    let contact = current
        .iter()
        .find(|claim| claim.predicate.as_deref() == Some("連絡先"))
        .unwrap();
    assert_eq!(contact.polarity, ClaimPolarity::Uncertain);
    assert_eq!(contact.object_value, None);
    assert_eq!(contact.source_sequence_no, Some(2));

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn on_change_rule_requires_a_proven_value_transition() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("rule_requires_transition");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let prior = make_claim(
        &scope,
        "claim_owner_prior",
        "proyecto",
        "responsable",
        Some("Ana"),
        5,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    store.append_claim(&prior, None).unwrap();

    let mut input = make_derived_rule_inputs();
    input.id = Some("rule_transition_only".to_string());
    input.trigger_subject_key = "proyecto".to_string();
    input.trigger_predicate_key = "responsable".to_string();
    input.target_subject_key = "proyecto".to_string();
    input.target_predicate_key = "revisor".to_string();
    input.trigger_subject = Some("proyecto".to_string());
    input.trigger_predicate = Some("responsable".to_string());
    input.target_subject = Some("proyecto".to_string());
    input.target_predicate = Some("revisor".to_string());
    input.activation = RuleActivation::OnChange;
    store.add_rule(input).unwrap();

    let unchanged = make_claim(
        &scope,
        "claim_owner_unchanged",
        "proyecto",
        "responsable",
        Some("Ana"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    assert!(store
        .resolve_rules_for_changed_claims(
            std::slice::from_ref(&unchanged),
            &BTreeSet::from([unchanged.id.clone()]),
            3,
        )
        .unwrap()
        .is_empty());
    store.append_claim(&unchanged, None).unwrap();

    let changed = make_claim(
        &scope,
        "claim_owner_changed",
        "proyecto",
        "responsable",
        Some("Béa"),
        20,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let applications = store
        .resolve_rules_for_changed_claims(
            std::slice::from_ref(&changed),
            &BTreeSet::from([changed.id.clone()]),
            3,
        )
        .unwrap();
    assert_eq!(applications.len(), 1);
    assert_eq!(
        applications[0].prior_trigger_claim_id.as_deref(),
        Some("claim_owner_unchanged")
    );
    assert_eq!(applications[0].claim.object_value.as_deref(), Some("Béa"));
    for proof in [
        "span_claim_owner_unchanged",
        "span_claim_owner_changed",
        "rule_span_owner",
    ] {
        assert!(applications[0]
            .claim
            .source_span_ids
            .iter()
            .any(|id| id == proof));
    }

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn standing_derive_rule_applies_to_current_trigger_from_the_rule_boundary() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("standing_derive_current_trigger");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    store
        .append_claim(
            &make_claim(
                &scope,
                "claim_responsable_actual",
                "proyecto",
                "responsable",
                Some("Ana"),
                10,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            ),
            None,
        )
        .unwrap();

    let mut standing = make_derived_rule_inputs();
    standing.id = Some("rule_revisor_sigue_responsable".to_string());
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

    let before_rule = store
        .scan_current_claims(&scope, usize::MAX, Some(15))
        .unwrap();
    assert!(!before_rule
        .iter()
        .any(|claim| claim.predicate.as_deref() == Some("revisor")));

    let after_rule = store
        .scan_current_claims(&scope, usize::MAX, Some(20))
        .unwrap();
    let reviewer = after_rule
        .iter()
        .find(|claim| claim.predicate.as_deref() == Some("revisor"))
        .expect("standing derive should copy the current trigger from the rule start");
    assert_eq!(reviewer.object_value.as_deref(), Some("Ana"));
    assert_eq!(reviewer.valid_from_ms, Some(20));

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn standing_derive_rule_in_a_later_batch_applies_to_existing_trigger() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("standing_derive_later_batch");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    store
        .apply_state_mutation_batch(StateMutationBatch {
            scope: scope.clone(),
            transaction_time_ms: 10,
            propagation_seed: "trigger-first".to_string(),
            max_rule_hops: 8,
            claims: vec![make_claim(
                &scope,
                "claim_owner_before_rule",
                "proyecto",
                "responsable",
                Some("Ana"),
                10,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            )],
            entities: Vec::new(),
            rules: Vec::new(),
            slot_aliases: Vec::new(),
            corrections: Vec::new(),
        })
        .unwrap();

    let mut standing = make_derived_rule_inputs();
    standing.id = Some("rule_revisor_desde_ahora".to_string());
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
    store
        .apply_state_mutation_batch(StateMutationBatch {
            scope: scope.clone(),
            transaction_time_ms: 20,
            propagation_seed: "rule-later".to_string(),
            max_rule_hops: 8,
            claims: Vec::new(),
            entities: Vec::new(),
            rules: vec![standing],
            slot_aliases: Vec::new(),
            corrections: Vec::new(),
        })
        .unwrap();

    let current = store
        .scan_current_claims(&scope, usize::MAX, Some(20))
        .unwrap();
    let reviewer = current
        .iter()
        .find(|claim| claim.predicate.as_deref() == Some("revisor"))
        .expect("later standing rule should derive from the already-current trigger");
    assert_eq!(reviewer.object_value.as_deref(), Some("Ana"));
    assert_eq!(reviewer.valid_from_ms, Some(20));

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn on_change_derive_does_not_copy_a_pre_rule_trigger_without_a_later_change() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("on_change_no_pre_rule_copy");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    store
        .append_claim(
            &make_claim(
                &scope,
                "claim_responsable_antes",
                "proyecto",
                "responsable",
                Some("Ana"),
                10,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            ),
            None,
        )
        .unwrap();

    let mut on_change = make_derived_rule_inputs();
    on_change.id = Some("rule_revisor_cuando_cambie".to_string());
    on_change.trigger_subject_key = "proyecto".to_string();
    on_change.trigger_predicate_key = "responsable".to_string();
    on_change.target_subject_key = "proyecto".to_string();
    on_change.target_predicate_key = "revisor".to_string();
    on_change.trigger_subject = Some("proyecto".to_string());
    on_change.trigger_predicate = Some("responsable".to_string());
    on_change.target_subject = Some("proyecto".to_string());
    on_change.target_predicate = Some("revisor".to_string());
    on_change.activation = RuleActivation::OnChange;
    on_change.valid_from_ms = Some(20);
    store.add_rule(on_change).unwrap();

    let after_rule = store
        .scan_current_claims(&scope, usize::MAX, Some(20))
        .unwrap();
    assert!(!after_rule
        .iter()
        .any(|claim| claim.predicate.as_deref() == Some("revisor")));

    store
        .append_claim(
            &make_claim(
                &scope,
                "claim_responsable_despues",
                "proyecto",
                "responsable",
                Some("Bea"),
                30,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            ),
            None,
        )
        .unwrap();
    let after_change = store
        .scan_current_claims(&scope, usize::MAX, Some(30))
        .unwrap();
    let reviewer = after_change
        .iter()
        .find(|claim| claim.predicate.as_deref() == Some("revisor"))
        .expect("on_change derive should fire on a later trigger change");
    assert_eq!(reviewer.object_value.as_deref(), Some("Bea"));

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn on_change_rule_finds_prior_value_after_slot_id_rebinding() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("rule_transition_rebound_slot");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let prior = make_claim(
        &scope,
        "claim_owner_before_rebinding",
        "proyecto",
        "responsable",
        Some("Ana"),
        5,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let target = make_claim(
        &scope,
        "claim_reviewer_before_rebinding",
        "proyecto",
        "revisor",
        Some("Ivo"),
        5,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    store.append_claim(&prior, None).unwrap();
    store.append_claim(&target, None).unwrap();

    let mut input = make_derived_rule_inputs();
    input.id = Some("rule_rebound_trigger_slot".to_string());
    input.trigger_subject_key = "proyecto".to_string();
    input.trigger_predicate_key = "responsable".to_string();
    input.target_subject_key = "proyecto".to_string();
    input.target_predicate_key = "revisor".to_string();
    input.trigger_subject = Some("proyecto".to_string());
    input.trigger_predicate = Some("responsable".to_string());
    input.target_subject = Some("proyecto".to_string());
    input.target_predicate = Some("revisor".to_string());
    input.activation = RuleActivation::OnChange;
    let mut rule = store.build_rule_record(input).unwrap();
    rule.trigger_slot_id = Some("slot_after_registry_rebinding".to_string());
    rule.trigger_binding_status = RuleBindingStatus::Bound;
    store.append_rule(&rule).unwrap();

    let mut changed = make_claim(
        &scope,
        "claim_owner_after_rebinding",
        "proyecto",
        "responsable",
        Some("Béa"),
        20,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    changed.slot_id = Some("slot_after_registry_rebinding".to_string());
    let applications = store
        .resolve_rules_for_changed_claims(
            std::slice::from_ref(&changed),
            &BTreeSet::from([changed.id.clone()]),
            3,
        )
        .unwrap();

    assert_eq!(applications.len(), 1);
    assert_eq!(
        applications[0].prior_trigger_claim_id.as_deref(),
        Some("claim_owner_before_rebinding")
    );
    assert_eq!(applications[0].claim.object_value.as_deref(), Some("Béa"));

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn on_change_unsupported_rule_finds_prior_value_through_later_entity_alias() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("rule_transition_entity_alias");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let prior = make_claim(
        &scope,
        "claim_owner_before_entity_alias",
        "equipo anterior",
        "responsable",
        Some("Ana"),
        5,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    store.append_claim(&prior, None).unwrap();
    store
        .append_entity(
            &EntityRecord {
                id: "entity_equipo".to_string(),
                scope: scope.clone(),
                status: MemoryStatus::Active,
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                entity_type: "team".to_string(),
                canonical_name: "equipo".to_string(),
                aliases: vec!["equipo anterior".to_string()],
                source_claim_ids: vec![prior.id.clone()],
                merge_parent_ids: Vec::new(),
                split_from_id: None,
                confidence: Some(1.0),
            },
            None,
        )
        .unwrap();
    let target = make_claim(
        &scope,
        "claim_contact_before_entity_change",
        "equipo",
        "contacto",
        Some("Ivo"),
        7,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    store.append_claim(&target, None).unwrap();

    let mut input = make_derived_rule_inputs();
    input.id = Some("rule_entity_alias_transition".to_string());
    input.trigger_subject_key = "equipo".to_string();
    input.trigger_predicate_key = "responsable".to_string();
    input.target_subject_key = "equipo".to_string();
    input.target_predicate_key = "contacto".to_string();
    input.trigger_subject = Some("equipo".to_string());
    input.trigger_predicate = Some("responsable".to_string());
    input.target_subject = Some("equipo".to_string());
    input.target_predicate = Some("contacto".to_string());
    input.activation = RuleActivation::OnChange;
    input.action = RuleAction::MarkUnsupported;
    input.value_template = None;
    input.value = None;
    store.add_rule(input).unwrap();

    let changed = make_claim(
        &scope,
        "claim_owner_after_entity_alias",
        "equipo",
        "responsable",
        Some("Béa"),
        20,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let registry = store
        .canonical_registry_for_names(&scope, changed.subject.iter())
        .unwrap();
    let changed = canonicalize_claim(changed, &registry);
    let applications = store
        .resolve_rules_for_changed_claims(
            std::slice::from_ref(&changed),
            &BTreeSet::from([changed.id.clone()]),
            3,
        )
        .unwrap();

    assert_eq!(applications.len(), 1);
    assert_eq!(
        applications[0].prior_trigger_claim_id.as_deref(),
        Some("claim_owner_before_entity_alias")
    );
    assert_eq!(applications[0].claim.subject.as_deref(), Some("equipo"));
    assert_eq!(applications[0].claim.predicate.as_deref(), Some("contacto"));
    assert_eq!(applications[0].claim.object_value, None);
    assert_eq!(applications[0].claim.polarity, ClaimPolarity::Uncertain);

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn memory_store_resolver_marks_dependent_target_unsupported() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("rule_unsupported");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();

    let mut unsupported_rule = make_derived_rule_inputs();
    let scope = scope();
    unsupported_rule.id = Some("rule_unsupported".to_string());
    unsupported_rule.trigger_subject_key = "account".to_string();
    unsupported_rule.trigger_predicate_key = "payment_vendor".to_string();
    unsupported_rule.target_subject_key = "account".to_string();
    unsupported_rule.target_predicate_key = "billing_contact".to_string();
    unsupported_rule.target_subject = Some("account".to_string());
    unsupported_rule.target_predicate = Some("billing_contact".to_string());
    unsupported_rule.activation = RuleActivation::OnChange;
    unsupported_rule.action = RuleAction::MarkUnsupported;
    unsupported_rule.value_template = None;
    unsupported_rule.value = None;
    store
        .append_claim(
            &make_claim(
                &scope,
                "claim_vendor_prior",
                "account",
                "payment_vendor",
                Some("legacy"),
                1,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            ),
            None,
        )
        .unwrap();
    store.add_rule(unsupported_rule).unwrap();

    let target = make_claim(
        &scope,
        "claim_billing",
        "account",
        "billing_contact",
        Some("morgan"),
        5,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let trigger = make_claim(
        &scope,
        "claim_vendor",
        "account",
        "payment_vendor",
        Some("northstar"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let applications = store
        .resolve_rules_for_changed_claims(
            &[target, trigger],
            &BTreeSet::from(["claim_billing".to_string(), "claim_vendor".to_string()]),
            3,
        )
        .unwrap();

    assert_eq!(applications.len(), 1);
    assert_eq!(
        applications[0].prior_trigger_claim_id.as_deref(),
        Some("claim_vendor_prior")
    );
    let unsupported = &applications[0].claim;
    assert_eq!(unsupported.subject.as_deref(), Some("account"));
    assert_eq!(unsupported.predicate.as_deref(), Some("billing_contact"));
    assert_eq!(unsupported.object_value, None);
    assert!(unsupported
        .claim_text
        .contains(r#""support_state":"unsupported""#));
    for proof in [
        "span_claim_vendor_prior",
        "span_claim_vendor",
        "rule_span_owner",
    ] {
        assert!(unsupported.source_span_ids.iter().any(|id| id == proof));
    }

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn memory_store_resolver_marks_persisted_current_target_unsupported() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("rule_unsupported_persisted_target");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();

    let mut unsupported_rule = make_derived_rule_inputs();
    let scope = scope();
    unsupported_rule.id = Some("rule_unsupported_persisted".to_string());
    unsupported_rule.trigger_subject_key = "account".to_string();
    unsupported_rule.trigger_predicate_key = "payment_vendor".to_string();
    unsupported_rule.target_subject_key = "account".to_string();
    unsupported_rule.target_predicate_key = "billing_contact".to_string();
    unsupported_rule.target_subject = Some("account".to_string());
    unsupported_rule.target_predicate = Some("billing_contact".to_string());
    unsupported_rule.activation = RuleActivation::OnChange;
    unsupported_rule.action = RuleAction::MarkUnsupported;
    unsupported_rule.value_template = None;
    unsupported_rule.value = None;
    store
        .append_claim(
            &make_claim(
                &scope,
                "claim_vendor_prior_persisted",
                "account",
                "payment_vendor",
                Some("legacy"),
                1,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            ),
            None,
        )
        .unwrap();
    store.add_rule(unsupported_rule).unwrap();

    let target = make_claim(
        &scope,
        "claim_billing_persisted",
        "account",
        "billing_contact",
        Some("morgan"),
        5,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    store.append_claim(&target, None).unwrap();

    let trigger = make_claim(
        &scope,
        "claim_vendor_changed",
        "account",
        "payment_vendor",
        Some("northstar"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    store.append_claim(&trigger, None).unwrap();

    let current = store.scan_current_claims(&scope, 20, Some(10)).unwrap();
    let unsupported = current
        .iter()
        .find(|claim| {
            claim.predicate.as_deref() == Some("billing_contact")
                && matches!(claim.polarity, ClaimPolarity::Uncertain)
        })
        .expect("unsupported billing contact");
    assert_eq!(unsupported.subject.as_deref(), Some("account"));
    assert_eq!(unsupported.predicate.as_deref(), Some("billing_contact"));
    assert_eq!(unsupported.object_value, None);
    assert!(unsupported
        .claim_text
        .contains(r#""support_state":"unsupported""#));

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn memory_store_resolver_matches_structured_generic_target_for_mark_unsupported() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("rule_structured_generic_target");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();

    let mut unsupported_rule = make_derived_rule_inputs();
    let scope = scope();
    unsupported_rule.id = Some("rule_vendor_billing_generic".to_string());
    unsupported_rule.trigger_subject_key = "account".to_string();
    unsupported_rule.trigger_predicate_key = "payment_vendor".to_string();
    unsupported_rule.target_subject_key = "billing contact".to_string();
    unsupported_rule.target_predicate_key = "vendor_dependent_slot".to_string();
    unsupported_rule.target_subject = Some("billing contact".to_string());
    unsupported_rule.target_predicate = Some("vendor_dependent_slot".to_string());
    unsupported_rule.target_match = RuleTargetMatch::AnyActiveSlotForSubject;
    unsupported_rule.activation = RuleActivation::OnChange;
    unsupported_rule.action = RuleAction::MarkUnsupported;
    unsupported_rule.value_template = None;
    unsupported_rule.value = None;
    store
        .append_claim(
            &make_claim(
                &scope,
                "claim_payment_vendor_prior",
                "account",
                "payment_vendor",
                Some("legacy"),
                1,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            ),
            None,
        )
        .unwrap();
    store.add_rule(unsupported_rule).unwrap();

    let target = make_claim(
        &scope,
        "claim_billing_contact_old",
        "billing_contact",
        "current_vendor_handler",
        Some("Morgan"),
        5,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let trigger = make_claim(
        &scope,
        "claim_payment_vendor_new",
        "account",
        "payment_vendor",
        Some("Northstar Pay"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let applications = store
        .resolve_rules_for_changed_claims(
            &[target, trigger],
            &BTreeSet::from([
                "claim_billing_contact_old".to_string(),
                "claim_payment_vendor_new".to_string(),
            ]),
            3,
        )
        .unwrap();

    assert_eq!(applications.len(), 1);
    let unsupported = &applications[0].claim;
    assert_eq!(unsupported.subject.as_deref(), Some("billing_contact"));
    assert_eq!(
        unsupported.predicate.as_deref(),
        Some("current_vendor_handler")
    );
    assert_eq!(unsupported.object_value, None);
    assert!(unsupported
        .claim_text
        .contains(r#""support_state":"unsupported""#));

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn memory_store_resolver_generic_fallback_does_not_invalidate_exact_signal_target() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("rule_generic_target_with_specific_signal");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();

    let mut generic_rule = make_derived_rule_inputs();
    generic_rule.id = Some("rule_vendor_generic_unsupported".to_string());
    generic_rule.trigger_subject_key = "account".to_string();
    generic_rule.trigger_predicate_key = "payment_vendor".to_string();
    generic_rule.target_subject_key = "account".to_string();
    generic_rule.target_predicate_key = "vendor_dependent_slot".to_string();
    generic_rule.target_subject = Some("account".to_string());
    generic_rule.target_predicate = Some("vendor_dependent_slot".to_string());
    generic_rule.target_match = RuleTargetMatch::AnyActiveSlotForSubject;
    generic_rule.activation = RuleActivation::OnChange;
    generic_rule.action = RuleAction::MarkUnsupported;
    generic_rule.value_template = None;
    generic_rule.value = None;

    let mut specific_rule = make_derived_rule_inputs();
    specific_rule.id = Some("rule_owner_is_vendor".to_string());
    specific_rule.trigger_subject_key = "account".to_string();
    specific_rule.trigger_predicate_key = "payment_vendor".to_string();
    specific_rule.target_subject_key = "account".to_string();
    specific_rule.target_predicate_key = "owner".to_string();
    specific_rule.target_subject = Some("account".to_string());
    specific_rule.target_predicate = Some("owner".to_string());
    specific_rule.action = RuleAction::DeriveValue;
    specific_rule.value_template = Some("owner-from-{value}".to_string());
    specific_rule.value = None;

    store
        .append_claim(
            &make_claim(
                &scope,
                "claim_vendor_prior_specific",
                "account",
                "payment_vendor",
                Some("legacy"),
                1,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            ),
            None,
        )
        .unwrap();
    store.add_rule(generic_rule).unwrap();
    store.add_rule(specific_rule).unwrap();

    let billing_contact = make_claim(
        &scope,
        "claim_billing_contact_old",
        "account",
        "billing_contact",
        Some("Morgan"),
        5,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let owner = make_claim(
        &scope,
        "claim_owner",
        "account",
        "owner",
        Some("Eli"),
        8,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let trigger = make_claim(
        &scope,
        "claim_vendor_new",
        "account",
        "payment_vendor",
        Some("Northstar Pay"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let applications = store
        .resolve_rules_for_changed_claims(
            &[billing_contact, owner, trigger],
            &BTreeSet::from([
                "claim_billing_contact_old".to_string(),
                "claim_owner".to_string(),
                "claim_vendor_new".to_string(),
            ]),
            3,
        )
        .unwrap();

    assert_eq!(applications.len(), 2);
    let unsupported_billing = applications
        .iter()
        .find(|application| {
            application.claim.subject.as_deref() == Some("account")
                && application.claim.predicate.as_deref() == Some("billing_contact")
                && application.claim.object_value.is_none()
        })
        .expect("billing contact should be unsupported");
    assert!(unsupported_billing
        .claim
        .claim_text
        .contains(r#""support_state":"unsupported""#));
    assert!(!applications.iter().any(|application| {
        application.claim.predicate.as_deref() == Some("owner")
            && application.claim.object_value.is_none()
    }));
    assert!(applications
        .iter()
        .any(
            |application| application.claim.subject.as_deref() == Some("account")
                && application.claim.predicate.as_deref() == Some("owner")
                && application.claim.object_value.as_deref() == Some("owner-from-Northstar Pay")
        ));

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn memory_store_resolver_derive_value_uses_exact_target_slot() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("rule_exact_target_derive_value");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();

    let mut derive_rule = make_derived_rule_inputs();
    derive_rule.id = Some("rule_owner_from_vendor".to_string());
    derive_rule.trigger_subject_key = "account".to_string();
    derive_rule.trigger_predicate_key = "payment_vendor".to_string();
    derive_rule.target_subject_key = "account".to_string();
    derive_rule.target_predicate_key = "owner".to_string();
    derive_rule.target_subject = Some("account".to_string());
    derive_rule.target_predicate = Some("owner".to_string());
    derive_rule.action = RuleAction::DeriveValue;
    derive_rule.value_template = Some("{value}".to_string());
    derive_rule.value = None;

    store.add_rule(derive_rule).unwrap();

    let owner_slot = make_claim(
        &scope,
        "claim_owner_old",
        "account",
        "owner",
        Some("Eli"),
        5,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let trigger = make_claim(
        &scope,
        "claim_vendor_new",
        "account",
        "payment_vendor",
        Some("Northstar Pay"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let applications = store
        .resolve_rules_for_changed_claims(
            &[owner_slot, trigger],
            &BTreeSet::from([
                "claim_owner_old".to_string(),
                "claim_vendor_new".to_string(),
            ]),
            3,
        )
        .unwrap();

    assert_eq!(applications.len(), 1);
    assert_eq!(applications[0].claim.predicate.as_deref(), Some("owner"));
    assert_eq!(
        applications[0].claim.object_value.as_deref(),
        Some("Northstar Pay")
    );

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn memory_store_resolver_supports_two_hop_derivation() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("rule_two_hop");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();

    store
        .add_rule(RuleInput {
            id: Some("rule_status_mirror".to_string()),
            scope: scope(),
            status: MemoryStatus::Active,
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            trigger_subject_key: "source".to_string(),
            trigger_predicate_key: "status".to_string(),
            target_subject_key: "mirror".to_string(),
            target_predicate_key: "status".to_string(),
            trigger_slot_id: None,
            target_slot_id: None,
            trigger_subject: Some("source".to_string()),
            trigger_predicate: Some("status".to_string()),
            target_subject: Some("mirror".to_string()),
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
            id: Some("rule_mirror_downstream".to_string()),
            scope: scope(),
            status: MemoryStatus::Active,
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            trigger_subject_key: "mirror".to_string(),
            trigger_predicate_key: "status".to_string(),
            target_predicate_key: "status".to_string(),
            target_subject_key: "downstream".to_string(),
            trigger_slot_id: None,
            target_slot_id: None,
            trigger_subject: Some("mirror".to_string()),
            trigger_predicate: Some("status".to_string()),
            target_subject: Some("downstream".to_string()),
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
        "claim_source_status",
        "source",
        "status",
        Some("green"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let applications = store
        .resolve_rules_for_changed_claims(
            &[changed],
            &BTreeSet::from(["claim_source_status".to_string()]),
            3,
        )
        .unwrap();

    assert_eq!(applications.len(), 2);
    let target_subjects = applications
        .iter()
        .map(|application| {
            application
                .claim
                .subject
                .as_deref()
                .unwrap_or_default()
                .to_string()
        })
        .collect::<Vec<_>>();
    assert!(target_subjects.iter().any(|value| value == "mirror"));
    assert!(target_subjects.iter().any(|value| value == "downstream"));
    let mirror = applications
        .iter()
        .find(|application| application.claim.subject.as_deref() == Some("mirror"))
        .expect("mirror application");
    let downstream = applications
        .iter()
        .find(|application| application.claim.subject.as_deref() == Some("downstream"))
        .expect("downstream application");
    assert_eq!(mirror.hop, 1);
    assert!(mirror.parent_trace_ids.is_empty());
    assert_eq!(downstream.hop, 2);
    assert_eq!(downstream.parent_trace_ids, vec![mirror.trace_id.clone()]);
    assert_eq!(downstream.trigger_claim_id, mirror.claim.id);

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn memory_store_resolver_orders_branching_dependency_applications() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("rule_branch_order");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();

    for (id, trigger_subject, target_subject) in [
        ("rule_source_alpha", "source", "alpha"),
        ("rule_source_beta", "source", "beta"),
        ("rule_alpha_leaf", "alpha", "alpha_leaf"),
        ("rule_beta_leaf", "beta", "beta_leaf"),
    ] {
        store
            .add_rule(RuleInput {
                id: Some(id.to_string()),
                scope: scope(),
                status: MemoryStatus::Active,
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                trigger_subject_key: trigger_subject.to_string(),
                trigger_predicate_key: "status".to_string(),
                target_subject_key: target_subject.to_string(),
                target_predicate_key: "status".to_string(),
                trigger_slot_id: None,
                target_slot_id: None,
                trigger_subject: Some(trigger_subject.to_string()),
                trigger_predicate: Some("status".to_string()),
                target_subject: Some(target_subject.to_string()),
                target_predicate: Some("status".to_string()),
                target_match: RuleTargetMatch::ExactSlot,
                activation: RuleActivation::ContinuousProjection,
                action: RuleAction::DeriveValue,
                value_template: Some("{value}".to_string()),
                value: None,
                source_span_ids: vec![format!("span_{id}")],
                source_episode_ids: vec![format!("episode_{id}")],
                valid_from_ms: Some(0),
                valid_to_ms: None,
                confidence: Some(1.0),
            })
            .unwrap();
    }

    let changed = make_claim(
        &scope(),
        "claim_source_status",
        "source",
        "status",
        Some("green"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let applications = store
        .resolve_rules_for_changed_claims(
            &[changed],
            &BTreeSet::from(["claim_source_status".to_string()]),
            3,
        )
        .unwrap();

    assert_eq!(
        applications
            .iter()
            .map(|application| application.rule_id.as_str())
            .collect::<Vec<_>>(),
        vec![
            "rule_source_alpha",
            "rule_source_beta",
            "rule_alpha_leaf",
            "rule_beta_leaf",
        ]
    );
    assert_eq!(
        applications[2].parent_trace_ids,
        vec![applications[0].trace_id.clone()]
    );
    assert_eq!(
        applications[3].parent_trace_ids,
        vec![applications[1].trace_id.clone()]
    );

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn memory_store_rules_for_trigger_slot_and_scope_time_filtering() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("rule_trigger_slot_scope_time");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let mut base_scope = scope();
    base_scope.tenant_id = Some("tenant-A".to_string());

    let mut by_trigger = make_derived_rule_inputs();
    by_trigger.id = Some("rule_trigger_exact".to_string());
    by_trigger.trigger_subject_key = "Project Alpha".to_string();
    by_trigger.trigger_predicate_key = "owner".to_string();
    by_trigger.valid_from_ms = Some(10);
    by_trigger.valid_to_ms = None;
    by_trigger.scope = base_scope.clone();

    let mut off_scope = scope();
    off_scope.tenant_id = Some("tenant-B".to_string());

    let mut off_scope_rule = make_derived_rule_inputs();
    off_scope_rule.id = Some("rule_off_scope".to_string());
    off_scope_rule.scope = off_scope.clone();
    off_scope_rule.trigger_subject_key = "project alpha".to_string();
    off_scope_rule.trigger_predicate_key = "owner".to_string();
    off_scope_rule.valid_from_ms = Some(5);

    let mut stale_rule = make_derived_rule_inputs();
    stale_rule.id = Some("rule_inactive_or_expired".to_string());
    stale_rule.trigger_subject_key = "project alpha".to_string();
    stale_rule.trigger_predicate_key = "status".to_string();
    stale_rule.valid_from_ms = Some(1);
    stale_rule.valid_to_ms = Some(5);
    stale_rule.scope = base_scope.clone();

    let mut future_rule = make_derived_rule_inputs();
    future_rule.id = Some("rule_future".to_string());
    future_rule.target_predicate_key = "reviewer".to_string();
    future_rule.trigger_subject_key = "project alpha".to_string();
    future_rule.trigger_predicate_key = "owner".to_string();
    future_rule.valid_from_ms = Some(100);
    future_rule.scope = base_scope.clone();

    store.add_rule(by_trigger).unwrap();
    store.add_rule(off_scope_rule).unwrap();
    store.add_rule(stale_rule).unwrap();
    store.add_rule(future_rule).unwrap();

    let filtered = store
        .rules_for_trigger_slot(&base_scope, " Project Alpha ", "owner", 10, Some(50))
        .unwrap();
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].id, "rule_trigger_exact".to_string());

    let by_target = store
        .rules_for_target_slot(
            &base_scope,
            &CanonicalSlot::new("project", "reviewer"),
            10,
            Some(50),
        )
        .unwrap();
    assert_eq!(by_target.len(), 1);
    assert_eq!(by_target[0].id, "rule_trigger_exact");

    let scoped_all = store.scan_rules(&base_scope, 10, Some(200)).unwrap();
    assert_eq!(scoped_all.len(), 2);
    assert!(scoped_all
        .iter()
        .any(|rule| rule.id == "rule_trigger_exact"));
    assert!(scoped_all.iter().any(|rule| rule.id == "rule_future"));
    assert!(!scoped_all
        .iter()
        .any(|rule| rule.id == "rule_inactive_or_expired"));

    let full = store.scan_rules(&base_scope, 10, None).unwrap();
    assert_eq!(full.len(), 3);

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn memory_store_rule_limit_applies_after_filtering() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("rule_limit_after_filtering");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let mut base_scope = scope();
    base_scope.tenant_id = Some("tenant-A".to_string());
    let mut other_scope = scope();
    other_scope.tenant_id = Some("tenant-B".to_string());

    for i in 0..MAX_VECTOR_QUERY_TOPK {
        let mut rule = make_derived_rule_inputs();
        rule.id = Some(format!("rule_off_scope_{i}"));
        rule.scope = other_scope.clone();
        store.add_rule(rule).unwrap();
    }

    let mut valid = make_derived_rule_inputs();
    valid.id = Some("rule_valid_owner".to_string());
    valid.scope = base_scope.clone();
    valid.trigger_subject_key = "project alpha".to_string();
    valid.trigger_predicate_key = "owner".to_string();
    store.add_rule(valid).unwrap();

    let rules = store
        .rules_for_trigger_slot(&base_scope, "project alpha", "owner", 1, None)
        .unwrap();
    assert_eq!(rules.len(), 1);
    assert_eq!(rules[0].id, "rule_valid_owner");

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn memory_store_rule_scan_does_not_truncate_at_vector_query_limit() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("rule_scan_without_vector_cap");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let mut docs = Vec::new();
    for i in 0..=MAX_VECTOR_QUERY_TOPK {
        let mut input = make_derived_rule_inputs();
        input.id = Some(format!("rule_uncapped_{i}"));
        input.scope = scope.clone();
        let record = store.build_rule_record(input).unwrap();
        docs.push(rule_doc(&record).unwrap());
    }
    insert_many(&store.rules, docs).unwrap();

    let rules = store
        .rules_for_trigger_slot(&scope, "project", "owner", MAX_VECTOR_QUERY_TOPK + 1, None)
        .unwrap();
    assert_eq!(rules.len(), MAX_VECTOR_QUERY_TOPK + 1);

    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn memory_store_preserves_explicit_distinct_rule_slot_bindings() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("explicit_distinct_rule_slot_bindings");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let trigger = store
        .add_manual_claim(
            ManualClaimInput {
                id: Some("claim_owner".to_string()),
                scope: scope.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                claim_text: "La propriétaire est Mina.".to_string(),
                subject: Some("projet".to_string()),
                predicate: Some("propriétaire".to_string()),
                object_value: Some("Mina".to_string()),
                claim_kind: ClaimKind::Fact,
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
    let target = store
        .add_manual_claim(
            ManualClaimInput {
                id: Some("claim_reviewer".to_string()),
                scope: scope.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                claim_text: "La réviseuse est Mina.".to_string(),
                subject: Some("projet".to_string()),
                predicate: Some("réviseuse".to_string()),
                object_value: Some("Mina".to_string()),
                claim_kind: ClaimKind::Fact,
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
    let trigger_slot_id = trigger.slot_id.unwrap();
    let target_slot_id = target.slot_id.unwrap();

    let rule = store
        .add_rule(RuleInput {
            id: Some("rule_owner_reviewer".to_string()),
            scope,
            status: MemoryStatus::Active,
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            trigger_subject_key: "surface ambiguë".to_string(),
            trigger_predicate_key: "champ".to_string(),
            target_subject_key: "surface ambiguë".to_string(),
            target_predicate_key: "champ".to_string(),
            trigger_slot_id: Some(trigger_slot_id.clone()),
            target_slot_id: Some(target_slot_id.clone()),
            trigger_subject: Some("surface ambiguë".to_string()),
            trigger_predicate: Some("champ".to_string()),
            target_subject: Some("surface ambiguë".to_string()),
            target_predicate: Some("champ".to_string()),
            target_match: RuleTargetMatch::ExactSlot,
            activation: RuleActivation::ContinuousProjection,
            action: RuleAction::DeriveValue,
            value_template: Some("{value}".to_string()),
            value: None,
            source_span_ids: vec!["span_rule".to_string()],
            source_episode_ids: vec!["episode_rule".to_string()],
            valid_from_ms: Some(10),
            valid_to_ms: None,
            confidence: Some(1.0),
        })
        .unwrap();

    assert_eq!(
        rule.trigger_slot_id.as_deref(),
        Some(trigger_slot_id.as_str())
    );
    assert_eq!(
        rule.target_slot_id.as_deref(),
        Some(target_slot_id.as_str())
    );
    assert_ne!(rule.trigger_slot_id, rule.target_slot_id);
    assert_eq!(rule.trigger_binding_status, RuleBindingStatus::Bound);
    assert_eq!(rule.target_binding_status, RuleBindingStatus::Bound);
    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn subject_wide_invalidation_with_a_concrete_target_slot_only_retires_that_slot() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("wildcard_with_concrete_target");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let mk = |id: &str, predicate: &str, value: &str, at: i64| {
        make_claim(
            &scope,
            id,
            "usuaria",
            predicate,
            Some(value),
            at,
            (ClaimKind::Fact, ClaimPolarity::Affirmative),
        )
    };
    let mut rule = make_derived_rule_inputs();
    rule.id = Some("rule_zona_de_reforma_si_cambia_deseo".to_string());
    rule.trigger_subject_key = "usuaria".to_string();
    rule.trigger_predicate_key = "deseo".to_string();
    rule.target_subject_key = "usuaria".to_string();
    rule.target_predicate_key = "zona de reforma".to_string();
    rule.trigger_subject = Some("usuaria".to_string());
    rule.trigger_predicate = Some("deseo".to_string());
    rule.target_subject = Some("usuaria".to_string());
    rule.target_predicate = Some("zona de reforma".to_string());
    rule.target_match = RuleTargetMatch::AnyActiveSlotForSubject;
    rule.activation = RuleActivation::OnChange;
    rule.action = RuleAction::MarkUnsupported;
    rule.value_template = None;
    rule.valid_from_ms = Some(10);
    let result = store
        .apply_state_mutation_batch(StateMutationBatch {
            scope: scope.clone(),
            transaction_time_ms: 10,
            propagation_seed: "wildcard-concrete".to_string(),
            max_rule_hops: 8,
            claims: vec![
                mk("c_deseo_1", "deseo", "reformar la casa", 10),
                mk("c_zona", "zona de reforma", "cocina", 10),
                mk("c_lenguaje", "lenguaje preferido", "Python", 10),
            ],
            entities: Vec::new(),
            rules: vec![rule],
            slot_aliases: Vec::new(),
            corrections: Vec::new(),
        })
        .unwrap();
    assert!(matches!(
        result.rules[0].target_match,
        RuleTargetMatch::ExactSlot
    ));
    store
        .append_claim(&mk("c_deseo_2", "deseo", "aprender cerámica", 20), None)
        .unwrap();
    let current = store
        .scan_current_claims(&scope, usize::MAX, Some(20))
        .unwrap();
    let state = |predicate: &str| {
        current
            .iter()
            .find(|claim| claim.predicate.as_deref() == Some(predicate))
            .map(|claim| (claim.polarity, claim.object_value.clone()))
    };
    assert!(matches!(
        state("zona de reforma"),
        Some((ClaimPolarity::Uncertain, _))
    ));
    assert_eq!(
        state("lenguaje preferido"),
        Some((ClaimPolarity::Affirmative, Some("Python".to_string())))
    );

    let _ = std::fs::remove_dir_all(&store.path);
}

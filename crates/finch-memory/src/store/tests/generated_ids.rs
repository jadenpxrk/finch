use super::*;
use crate::ingest::{create_edge, create_entity, create_profile, ingest_episode, EpisodeInput};
use crate::types::{EdgeInput, ProfileInput, RuleAction, RuleActivation, RuleTargetMatch};

fn tenant_scopes() -> [MemoryScope; 2] {
    ["tenant_a", "tenant_b"].map(|tenant| {
        let mut scope = scope();
        scope.tenant_id = Some(tenant.to_string());
        scope
    })
}

fn correction_input(scope: MemoryScope, selector: Option<&str>) -> CorrectionInput {
    CorrectionInput {
        id: None,
        scope,
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        operation: CorrectionOperation::Forget,
        target_type: "span".to_string(),
        target_ids: vec!["span_note".to_string()],
        target_selector: selector.map(str::to_string),
        new_value: None,
        reason: None,
        actor: ActorKind::User,
        authority: CorrectionAuthority::User,
        effective_at_ms: Some(20),
        applies_valid_from_ms: None,
        applies_valid_to_ms: None,
        cascade_policy: None,
        metadata_json: None,
    }
}

fn rule_input(scope: MemoryScope) -> RuleInput {
    RuleInput {
        id: None,
        scope,
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
    }
}

/// `rule_input` citing evidence of its own tenant; evidence rows belong to one scope.
fn tenant_rule_input(scope: &MemoryScope) -> RuleInput {
    let tenant = scope.tenant_id.clone().unwrap_or_default();
    RuleInput {
        source_span_ids: vec![format!("rule_span_{tenant}")],
        source_episode_ids: vec![format!("ep_rule_{tenant}")],
        ..rule_input(scope.clone())
    }
}

#[test]
fn generated_correction_ids_collide_between_tenants() {
    // Corrections created independently in different tenant scopes must coexist.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("generated_correction_scope");
    let store = MemoryStore::create(&path, 3, CollectionOptions::default()).unwrap();
    let scopes = tenant_scopes();
    let corrections = scopes.each_ref().map(|scope| {
        crate::ingest::add_correction(correction_input(scope.clone(), Some("note")), 20)
    });
    let results = corrections
        .each_ref()
        .map(|correction| store.add_correction(correction));
    drop(store);
    let reopened = MemoryStore::open(&path, CollectionOptions::default()).unwrap();
    let stored = scopes
        .each_ref()
        .map(|scope| reopened.scan_corrections(scope, 10).unwrap());
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);
    assert!(results[0].is_ok());
    assert!(
        results[1].is_ok(),
        "independent tenant correction failed: {:?}; reopened correction counts: {:?}",
        results[1],
        stored.each_ref().map(|corrections| corrections.len())
    );
    for (scope, corrections) in scopes.iter().zip(&stored) {
        assert_eq!(corrections.len(), 1);
        assert_eq!(&corrections[0].scope, scope);
    }
    assert_ne!(stored[0][0].id, stored[1][0].id);
}

#[test]
fn generated_rule_ids_collide_between_tenants() {
    // Rules created independently in different tenant scopes must coexist.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("generated_rule_scope");
    let store = EvidencedStore::create(&path, 3, CollectionOptions::default()).unwrap();
    let scopes = tenant_scopes();
    // Identical inputs in two tenants still generate two ids.
    let built_ids = scopes.each_ref().map(|scope| {
        store
            .build_rule_record(rule_input(scope.clone()))
            .unwrap()
            .id
    });
    let results = scopes
        .each_ref()
        .map(|scope| store.add_rule(tenant_rule_input(scope)));
    drop(store);
    let reopened = MemoryStore::open(&path, CollectionOptions::default()).unwrap();
    let rules = scopes
        .each_ref()
        .map(|scope| reopened.scan_rules(scope, 10, Some(20)).unwrap());
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);
    assert!(results[0].is_ok());
    assert!(
        results[1].is_ok(),
        "independent tenant rule failed: {:?}; reopened rule counts: {:?}",
        results[1],
        rules.each_ref().map(|rules| rules.len())
    );
    for (scope, rules) in scopes.iter().zip(&rules) {
        assert_eq!(rules.len(), 1);
        assert_eq!(&rules[0].scope, scope);
    }
    assert_ne!(rules[0][0].id, rules[1][0].id);
    assert_ne!(built_ids[0], built_ids[1]);
}

#[test]
fn space_only_generated_ids_keep_their_stored_values() {
    // Existing stores hold these ids; a change here orphans every record written before it.
    let episode = ingest_episode(
        EpisodeInput {
            id: None,
            scope: scope(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            source_kind: SourceKind::UserMessage,
            actor: ActorKind::User,
            sequence_no: 7,
            event_time_ms: Some(100),
            valid_from_ms: None,
            valid_to_ms: None,
            raw_text: "abcdef".to_string(),
            blob_ref: None,
            mime_type: None,
            causal_parent_ids: Vec::new(),
            metadata_json: None,
        },
        101,
        &ChunkOptions::default(),
    );
    let correction = crate::ingest::add_correction(correction_input(scope(), None), 20);
    let selector_correction =
        crate::ingest::add_correction(correction_input(scope(), Some("note")), 20);
    let claim = create_manual_claim(ManualClaimInput {
        id: None,
        scope: scope(),
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        claim_text: "User prefers Rust.".to_string(),
        subject: Some("user".to_string()),
        predicate: Some("language".to_string()),
        object_value: Some("Rust".to_string()),
        claim_kind: ClaimKind::Fact,
        polarity: ClaimPolarity::Affirmative,
        source_span_ids: vec!["span_a".to_string()],
        source_episode_ids: vec!["ep_a".to_string()],
        asserted_by: "user".to_string(),
        confidence: None,
        observed_at_ms: 10,
        valid_from_ms: Some(10),
        valid_to_ms: None,
    });
    let profile = create_profile(ProfileInput {
        id: None,
        scope: scope(),
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        subject_id: None,
        profile_key: "style".to_string(),
        profile_text: "prefers concise answers".to_string(),
        value_json: None,
        evidence_claim_ids: vec!["claim_a".to_string()],
        source_span_ids: Vec::new(),
        generated_at_ms: 10,
        generator_version: "manual".to_string(),
        valid_from_ms: Some(10),
        valid_to_ms: None,
        correction_watermark: 10,
    });
    let entity = create_entity(EntityInput {
        id: None,
        scope: scope(),
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        entity_type: "person".to_string(),
        canonical_name: "Ana".to_string(),
        aliases: Vec::new(),
        source_claim_ids: Vec::new(),
        merge_parent_ids: Vec::new(),
        split_from_id: None,
        confidence: None,
    });
    let edge = create_edge(EdgeInput {
        id: None,
        scope: scope(),
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        src_entity_id: "entity_a".to_string(),
        dst_entity_id: "entity_b".to_string(),
        relation_type: "works_with".to_string(),
        claim_id: None,
        source_span_ids: Vec::new(),
        valid_from_ms: None,
        valid_to_ms: None,
        confidence: None,
    });
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("space_only_generated_ids");
    let store = EvidencedStore::create(&path, 3, CollectionOptions::default()).unwrap();
    let rule = store.add_rule(rule_input(scope())).unwrap();
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);
    let ids = [
        episode.episode.id.as_str(),
        episode.spans[0].id.as_str(),
        correction.id.as_str(),
        selector_correction.id.as_str(),
        claim.id.as_str(),
        profile.id.as_str(),
        entity.id.as_str(),
        edge.id.as_str(),
        rule.id.as_str(),
    ];
    assert_eq!(
        ids,
        [
            "ep_2b797ec4fa1c434c",
            "span_1f20e247c1aef4b2",
            "corr_26ecfa608b3c35a2",
            "corr_a7ad98910baf52a3",
            "claim_9cb645f397d4799c",
            "profile_1cf011d949b53815",
            "entity_7ad1e73089e00fec",
            "edge_238d02f7fd118b40",
            "rule_079f036c7e83d1ce",
        ]
    );
}

#[test]
fn identical_tenant_writes_project_distinct_derived_ids() {
    // Ids derived from generated ids (state, trace, and derived claim versions) inherit their scope.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("generated_derived_scope");
    let store = EvidencedStore::create(&path, 3, CollectionOptions::default()).unwrap();
    let scopes = tenant_scopes();
    for scope in &scopes {
        store.add_rule(tenant_rule_input(scope)).unwrap();
        store
            .add_manual_claim(
                ManualClaimInput {
                    id: None,
                    scope: scope.clone(),
                    visibility: Visibility::Private,
                    policy_tags: Vec::new(),
                    claim_text: "source status is green".to_string(),
                    subject: Some("source".to_string()),
                    predicate: Some("status".to_string()),
                    object_value: Some("green".to_string()),
                    claim_kind: ClaimKind::Fact,
                    polarity: ClaimPolarity::Affirmative,
                    source_span_ids: Vec::new(),
                    source_episode_ids: Vec::new(),
                    asserted_by: "user".to_string(),
                    confidence: None,
                    observed_at_ms: 10,
                    valid_from_ms: Some(10),
                    valid_to_ms: None,
                },
                None,
            )
            .unwrap();
    }
    let ids = scopes.each_ref().map(|scope| {
        let states = store
            .scan_state_records(scope, state_scan(50, None))
            .unwrap();
        let traces = store.scan_dependency_traces(scope, 50, None).unwrap();
        let claims = store.scan_claims(scope, 50, None).unwrap();
        assert!(states.iter().all(|record| &record.scope == scope));
        assert!(traces.iter().all(|record| &record.scope == scope));
        assert!(claims.iter().all(|record| &record.scope == scope));
        assert_eq!((traces.len(), claims.len()), (1, 2));
        assert!(states.len() >= 2);
        states
            .into_iter()
            .map(|record| record.id)
            .chain(traces.into_iter().map(|record| record.id))
            .chain(claims.into_iter().map(|record| record.id))
            .collect::<BTreeSet<_>>()
    });
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);
    assert!(ids[0].is_disjoint(&ids[1]));
}

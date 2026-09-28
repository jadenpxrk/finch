use super::*;
use crate::retrieval::apply_corrections_to_span_hits;
use crate::types::CorrectionTargetMatch;

fn tenant(id: &str) -> MemoryScope {
    let mut scope = scope();
    scope.tenant_id = Some(id.to_string());
    scope
}

fn fact(
    scope: &MemoryScope,
    id: &str,
    (subject, predicate, value): (&str, &str, &str),
    at: i64,
    polarity: ClaimPolarity,
) -> ClaimRecord {
    let mut claim = make_claim(
        scope,
        id,
        subject,
        predicate,
        Some(value),
        at,
        (ClaimKind::Fact, polarity),
    );
    claim.source_span_ids.clear();
    claim.source_episode_ids.clear();
    claim
}

fn owner(scope: &MemoryScope, id: &str, value: &str, at: i64) -> ClaimRecord {
    fact(
        scope,
        id,
        ("project", "owner", value),
        at,
        ClaimPolarity::Affirmative,
    )
}

/// Both tenants' owner claims, then the space's: all valid from 10, the tenants' observed later
/// so each tenant claim is newer than the space claim.
fn write_owners_then_space(store: &MemoryStore) -> [MemoryScope; 3] {
    let scopes = [tenant("tenant_a"), tenant("tenant_b"), scope()];
    for (scope, id, value, observed_at_ms) in [
        (&scopes[0], "owner_a", "Ada", 30),
        (&scopes[1], "owner_b", "Bo", 40),
        (&scopes[2], "owner_space", "Sam", 10),
    ] {
        let mut claim = owner(scope, id, value, 10);
        claim.observed_at_ms = observed_at_ms;
        store.append_claim(&claim, None).unwrap();
    }
    scopes
}

fn open_store(name: &str) -> EvidencedStore {
    EvidencedStore::create(&temp_dir(name), 3, CollectionOptions::default()).unwrap()
}

fn close_store(store: EvidencedStore) {
    let path = store.path.clone();
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
}

/// Claim ids of each active state `scope` itself owns.
fn own_state_claim_ids(store: &MemoryStore, scope: &MemoryScope) -> Vec<Vec<MemoryId>> {
    store
        .scan_state_records(scope, state_scan(usize::MAX, None))
        .unwrap()
        .into_iter()
        .filter(|record| record.scope == *scope)
        .map(|record| record.claim_ids)
        .collect()
}

fn claim_status(claims: &[ClaimRecord], id: &str) -> Option<MemoryStatus> {
    claims
        .iter()
        .find(|claim| claim.id == id)
        .map(|claim| claim.status)
}

fn owner_to_reviewer_rule(scope: &MemoryScope, id: &str, valid_from_ms: i64) -> RuleInput {
    RuleInput {
        id: Some(id.to_string()),
        scope: scope.clone(),
        valid_from_ms: Some(valid_from_ms),
        // Evidence rows belong to one scope, so each rule cites its own.
        source_span_ids: vec![format!("span_{id}")],
        source_episode_ids: vec![format!("ep_{id}")],
        ..make_derived_rule_inputs()
    }
}

fn slot_correction(
    scope: &MemoryScope,
    id: &str,
    authority: CorrectionAuthority,
) -> CorrectionRecord {
    let mut correction = crate::ingest::add_correction(
        CorrectionInput {
            id: Some(id.to_string()),
            scope: scope.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            operation: CorrectionOperation::Retract,
            target_type: "claim".to_string(),
            target_ids: Vec::new(),
            target_selector: None,
            new_value: None,
            reason: None,
            actor: ActorKind::User,
            authority,
            effective_at_ms: Some(50),
            applies_valid_from_ms: None,
            applies_valid_to_ms: None,
            cascade_policy: None,
            metadata_json: None,
        },
        50,
    );
    correction.target_match = CorrectionTargetMatch::CanonicalSlot;
    correction.target_subject_key = Some("project".to_string());
    correction.target_predicate_key = Some("owner".to_string());
    correction
}

fn reviewer_values(claims: &[ClaimRecord]) -> Vec<(MemoryScope, Option<String>)> {
    claims
        .iter()
        .filter(|claim| claim.predicate.as_deref() == Some("reviewer"))
        .filter(|claim| matches!(claim.polarity, ClaimPolarity::Affirmative))
        .map(|claim| (claim.scope.clone(), claim.object_value.clone()))
        .collect()
}

#[test]
fn space_claim_projection_keeps_each_scope_on_its_own_state() {
    // The space write projects only its own claim; the tenants' newer claims are not its state.
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let store = open_store("scope_keys_projection");
    let [a, b, space] = write_owners_then_space(&store);
    let states = [&a, &b, &space].map(|scope| own_state_claim_ids(&store, scope));
    close_store(store);
    assert_eq!(
        states,
        [
            vec![vec!["owner_a".to_string()]],
            vec![vec!["owner_b".to_string()]],
            vec![vec!["owner_space".to_string()]],
        ]
    );
}

#[test]
fn space_claim_does_not_absorb_tenant_slot_identities() {
    // A space write rewrites only the space's slot identity, not the tenants' identities.
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let store = open_store("scope_keys_slot_write");
    let [a, b, space] = write_owners_then_space(&store);
    let sources = [&a, &b, &space].map(|scope| {
        store
            .scan_slots(scope, usize::MAX, None)
            .unwrap()
            .into_iter()
            .filter(|slot| slot.scope == *scope)
            .map(|slot| slot.source_claim_ids)
            .collect::<Vec<_>>()
    });
    close_store(store);
    assert_eq!(
        sources,
        [
            vec![vec!["owner_a".to_string()]],
            vec![vec!["owner_b".to_string()]],
            vec![vec!["owner_space".to_string()]],
        ]
    );
}

#[test]
fn space_slot_scan_keeps_each_tenant_slot_identity() {
    // Two tenants on one slot key are two identities in a space-wide slot read.
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let store = open_store("scope_keys_slot_scan");
    let (a, b) = (tenant("tenant_a"), tenant("tenant_b"));
    store
        .append_claim(&owner(&a, "owner_a", "Ada", 20), None)
        .unwrap();
    store
        .append_claim(&owner(&b, "owner_b", "Bo", 30), None)
        .unwrap();
    let slots = store
        .scan_slots(&scope(), usize::MAX, None)
        .unwrap()
        .into_iter()
        .map(|slot| (slot.scope, slot.source_claim_ids))
        .collect::<Vec<_>>();
    close_store(store);
    assert_eq!(
        slots,
        vec![
            (a, vec!["owner_a".to_string()]),
            (b, vec!["owner_b".to_string()]),
        ]
    );
}

#[test]
fn tenant_correction_does_not_retract_another_tenants_claim() {
    // Tenant A's slot correction must not reach tenant B's claim in a space-wide read.
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let store = open_store("scope_keys_tenant_correction");
    let (a, b) = (tenant("tenant_a"), tenant("tenant_b"));
    store
        .append_claim(&owner(&a, "owner_a", "Ada", 20), None)
        .unwrap();
    store
        .append_claim(&owner(&b, "owner_b", "Bo", 30), None)
        .unwrap();
    store
        .add_correction(&slot_correction(&a, "retract_a", CorrectionAuthority::User))
        .unwrap();
    let space_view = store.scan_current_claims(&scope(), 10, Some(60)).unwrap();
    let b_view = store.scan_current_claims(&b, 10, Some(60)).unwrap();
    close_store(store);
    assert_eq!(
        claim_status(&space_view, "owner_a"),
        Some(MemoryStatus::Retracted)
    );
    assert_eq!(
        claim_status(&space_view, "owner_b"),
        claim_status(&b_view, "owner_b")
    );
    assert_eq!(claim_status(&b_view, "owner_b"), Some(MemoryStatus::Active));
}

#[test]
fn space_admin_correction_retracts_only_space_claims() {
    // A correction reaches claims of its own scope only, whatever its authority: tenant reads
    // never load a parent's corrections, so a wider reach would give two answers for one claim.
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let store = open_store("scope_keys_admin_correction");
    let (a, space) = (tenant("tenant_a"), scope());
    store
        .append_claim(&owner(&a, "owner_a", "Ada", 20), None)
        .unwrap();
    store
        .append_claim(&owner(&space, "owner_space", "Sam", 10), None)
        .unwrap();
    store
        .add_correction(&slot_correction(
            &space,
            "retract_space",
            CorrectionAuthority::Admin,
        ))
        .unwrap();
    let space_view = store.scan_current_claims(&space, 10, Some(60)).unwrap();
    let a_view = store.scan_current_claims(&a, 10, Some(60)).unwrap();
    let a_states = own_state_claim_ids(&store, &a);
    close_store(store);
    assert_eq!(
        claim_status(&space_view, "owner_space"),
        Some(MemoryStatus::Retracted)
    );
    assert_eq!(
        claim_status(&space_view, "owner_a"),
        Some(MemoryStatus::Active)
    );
    assert_eq!(claim_status(&a_view, "owner_a"), Some(MemoryStatus::Active));
    assert_eq!(a_states, vec![vec!["owner_a".to_string()]]);
}

#[test]
fn space_selector_correction_binds_only_space_claims() {
    // Resolving a space correction's selector must not collect tenant claims as its targets.
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let store = open_store("scope_keys_correction_binding");
    let (a, space) = (tenant("tenant_a"), scope());
    store
        .append_claim(
            &fact(
                &a,
                "design_owner_a",
                ("design", "owner", "Sam"),
                20,
                ClaimPolarity::Affirmative,
            ),
            None,
        )
        .unwrap();
    store
        .append_claim(&owner(&space, "owner_space", "Sam", 10), None)
        .unwrap();
    let mut correction = slot_correction(&space, "retract_sam", CorrectionAuthority::User);
    correction.target_match = CorrectionTargetMatch::ClaimVersions;
    correction.target_subject_key = None;
    correction.target_predicate_key = None;
    correction.target_selector = Some("sam".to_string());
    let added = store.add_correction(&correction);
    let space_view = store.scan_current_claims(&space, 10, Some(60)).unwrap();
    close_store(store);
    assert!(added.is_ok(), "space correction failed to bind: {added:?}");
    assert_eq!(
        claim_status(&space_view, "owner_space"),
        Some(MemoryStatus::Retracted)
    );
    assert_eq!(
        claim_status(&space_view, "design_owner_a"),
        Some(MemoryStatus::Active)
    );
}

#[test]
fn space_span_selector_correction_does_not_hide_tenant_span() {
    // A space read loads space corrections; their selector must not sweep a tenant's spans.
    let mut span = span_record("tenant_span", MemoryStatus::Active, Some(10));
    span.scope = tenant("tenant_a");
    span.text = "Morgan owns the project".to_string();
    span.lexical_text = span.text.clone();
    let correction = crate::ingest::add_correction(
        CorrectionInput {
            id: Some("forget_morgan".to_string()),
            scope: scope(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            operation: CorrectionOperation::Tombstone,
            target_type: "span".to_string(),
            target_ids: Vec::new(),
            target_selector: Some("morgan".to_string()),
            new_value: None,
            reason: None,
            actor: ActorKind::User,
            authority: CorrectionAuthority::Admin,
            effective_at_ms: Some(20),
            applies_valid_from_ms: None,
            applies_valid_to_ms: None,
            cascade_policy: None,
            metadata_json: None,
        },
        20,
    );
    let hits = vec![SpanSearchHit { span, score: 1.0 }];
    let kept = apply_corrections_to_span_hits(&hits, &[correction]);
    assert_eq!(kept, hits);
}

#[test]
fn parent_rule_does_not_fire_on_tenant_claim() {
    // A space rule derives from space claims only; tenant A keeps just its own claim and state.
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let store = open_store("scope_keys_parent_rule");
    let (a, space) = (tenant("tenant_a"), scope());
    store
        .append_claim(&owner(&space, "a_owner_space", "Sam", 10), None)
        .unwrap();
    store
        .append_claim(&owner(&a, "b_owner_a", "Ada", 10), None)
        .unwrap();
    store
        .add_rule(owner_to_reviewer_rule(&space, "rule_space", 20))
        .unwrap();
    let a_claims = store.scan_current_claims(&a, 10, Some(30)).unwrap();
    let space_claims = store
        .scan_current_claims(&space, 10, Some(30))
        .unwrap()
        .into_iter()
        .filter(|claim| claim.scope == space)
        .collect::<Vec<_>>();
    let a_states = own_state_claim_ids(&store, &a);
    close_store(store);
    assert_eq!(reviewer_values(&a_claims), Vec::new());
    assert_eq!(a_states, vec![vec!["b_owner_a".to_string()]]);
    assert_eq!(
        reviewer_values(&space_claims),
        vec![(space, Some("Sam".to_string()))]
    );
}

#[test]
fn tenant_rule_does_not_fire_on_another_tenants_claim() {
    // Rebuilding the space must not fire tenant A's rule on tenant B's or the space's claims.
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let store = open_store("scope_keys_tenant_rule");
    let (a, b, space) = (tenant("tenant_a"), tenant("tenant_b"), scope());
    store
        .add_rule(owner_to_reviewer_rule(&a, "rule_tenant_a", 0))
        .unwrap();
    store
        .append_claim(&owner(&space, "a_owner_space", "Sam", 10), None)
        .unwrap();
    store
        .append_claim(&owner(&b, "b_owner_b", "Bo", 10), None)
        .unwrap();
    store.refresh_state_projection(&space, None).unwrap();
    let states = [&b, &space].map(|scope| own_state_claim_ids(&store, scope));
    let b_claims = store.scan_current_claims(&b, 10, Some(30)).unwrap();
    close_store(store);
    assert_eq!(reviewer_values(&b_claims), Vec::new());
    assert_eq!(
        states,
        [
            vec![vec!["b_owner_b".to_string()]],
            vec![vec!["a_owner_space".to_string()]],
        ]
    );
}

#[test]
fn space_read_does_not_fire_tenant_rule_on_another_tenants_claim() {
    // Read-time rule evaluation over a space pairs each rule with triggers of its own scope.
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let store = open_store("scope_keys_read_rule");
    let (a, b, space) = (tenant("tenant_a"), tenant("tenant_b"), scope());
    let rule = store
        .add_rule(owner_to_reviewer_rule(&a, "rule_tenant_a", 0))
        .unwrap();
    store
        .append_claim(&owner(&space, "a_owner_space", "Sam", 10), None)
        .unwrap();
    store
        .append_claim(&owner(&b, "b_owner_b", "Bo", 10), None)
        .unwrap();
    let selections = [
        StateReadSelection {
            slot_id: rule.trigger_slot_id.clone().unwrap(),
            view: StateReadView::Current,
            role: StateReadRole::AnswerTarget,
        },
        StateReadSelection {
            slot_id: rule.target_slot_id.clone().unwrap(),
            view: StateReadView::Current,
            role: StateReadRole::Supporting,
        },
    ];
    let projection = store
        .project_answer_ready_state(
            &space,
            &AnswerReadyStateRequest {
                selections: &selections,
                ..answer_request("scope_keys", "", &[], 10, 0, Some(30))
            },
        )
        .unwrap();
    close_store(store);
    assert_eq!(reviewer_values(&projection.claims), Vec::new());
}

#[test]
fn tenant_claims_do_not_make_space_rule_binding_ambiguous() {
    // A space rule binds against space claims; two tenants' different slots are not its concern.
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let store = open_store("scope_keys_rule_binding");
    let (a, b, space) = (tenant("tenant_a"), tenant("tenant_b"), scope());
    add_project_entity(&store, &a);
    add_owner_claim(&store, &a, "Ada");
    add_owner_claim(&store, &b, "Bo");
    let rule = store
        .add_rule(owner_to_reviewer_rule(&space, "rule_space", 0))
        .unwrap();
    close_store(store);
    assert_eq!(
        (rule.trigger_binding_status, rule.trigger_subject_entity_id),
        (RuleBindingStatus::Bound, None)
    );
}

#[test]
fn space_claim_does_not_adopt_a_tenant_entity() {
    // Canonicalising a space claim uses the space's aliases, not a tenant's entity.
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let store = open_store("scope_keys_registry");
    let (a, space) = (tenant("tenant_a"), scope());
    add_project_entity(&store, &a);
    let tenant_claim = add_owner_claim(&store, &a, "Ada");
    let space_claim = add_owner_claim(&store, &space, "Sam");
    close_store(store);
    assert!(tenant_claim.subject_entity_id.is_some());
    assert_eq!(space_claim.subject_entity_id, None);
}

#[test]
fn space_read_keeps_a_tenant_set_that_another_tenant_invalidated() {
    // Tenant A's unsupported slot must not drop tenant B's set state from a space read.
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let store = open_store("scope_keys_set_reconcile");
    let (a, b) = (tenant("tenant_a"), tenant("tenant_b"));
    store
        .append_claim(
            &fact(
                &a,
                "friend_a",
                ("alice", "friend", "Carol"),
                30,
                ClaimPolarity::Negative,
            ),
            None,
        )
        .unwrap();
    let mut friend_b = make_claim(
        &b,
        "friend_b",
        "alice",
        "friend",
        Some("Bob"),
        10,
        (ClaimKind::Relationship, ClaimPolarity::Affirmative),
    );
    friend_b.source_span_ids.clear();
    friend_b.source_episode_ids.clear();
    store.append_claim(&friend_b, None).unwrap();
    let projection = store
        .project_answer_ready_state(
            &scope(),
            &answer_request("scope_keys", "alice friend", &[], 10, 0, Some(40)),
        )
        .unwrap();
    close_store(store);
    assert_eq!(
        projection
            .set_states
            .iter()
            .map(|state| (state.scope.clone(), state.members.clone()))
            .collect::<Vec<_>>(),
        vec![(b, vec!["Bob".to_string()])]
    );
}

#[test]
fn space_read_expands_a_tenants_derived_target_that_another_tenant_marked_unsupported() {
    // One tenant's unsupported state on a slot must not hide another tenant's derived state there.
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let store = open_store("scope_keys_state_kinds");
    let (a, b) = (tenant("tenant_a"), tenant("tenant_b"));
    let rule = store
        .add_rule(owner_to_reviewer_rule(&b, "rule_tenant_b", 0))
        .unwrap();
    store
        .append_claim(&owner(&b, "owner_b", "Bo", 10), None)
        .unwrap();
    store
        .append_claim(
            &fact(
                &a,
                "reviewer_a",
                ("project", "reviewer", "Ada"),
                10,
                ClaimPolarity::Negative,
            ),
            None,
        )
        .unwrap();
    let trigger_slot_id = rule.trigger_slot_id.clone().unwrap();
    let target_slot_id = rule.target_slot_id.clone().unwrap();
    let selections = [
        StateReadSelection {
            slot_id: trigger_slot_id.clone(),
            view: StateReadView::Current,
            role: StateReadRole::AnswerTarget,
        },
        StateReadSelection {
            slot_id: target_slot_id.clone(),
            view: StateReadView::Current,
            role: StateReadRole::Supporting,
        },
    ];
    let projection = store
        .project_answer_ready_state(
            &scope(),
            &AnswerReadyStateRequest {
                selections: &selections,
                ..answer_request("scope_keys", "", &[], 10, 0, Some(30))
            },
        )
        .unwrap();
    close_store(store);
    let mut expected = vec![trigger_slot_id, target_slot_id];
    expected.sort();
    assert_eq!(
        projection
            .support
            .slots
            .iter()
            .map(|slot| slot.slot_id.clone())
            .collect::<Vec<_>>(),
        expected
    );
}

/// Tenant A's and tenant B's owner claims on one slot, A's observed first.
fn write_tenant_owners(store: &MemoryStore) -> [MemoryScope; 2] {
    let scopes = [tenant("tenant_a"), tenant("tenant_b")];
    store
        .append_claim(&owner(&scopes[0], "owner_a", "Ada", 10), None)
        .unwrap();
    store
        .append_claim(&owner(&scopes[1], "owner_b", "Bo", 20), None)
        .unwrap();
    scopes
}

#[test]
fn space_binding_contexts_keep_each_tenants_states() {
    // Each tenant's slot context carries its own recent states, not both tenants'.
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let store = open_store("scope_keys_binding_contexts");
    write_tenant_owners(&store);
    let contexts = store
        .canonical_slot_binding_contexts(&scope(), 10, None)
        .unwrap();
    close_store(store);
    assert_eq!(
        contexts
            .iter()
            .map(|context| {
                context
                    .temporal_state
                    .iter()
                    .map(|state| state.state_text.clone())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
        vec![
            vec!["project owner is Ada".to_string()],
            vec!["project owner is Bo".to_string()],
        ]
    );
}

#[test]
fn space_repairable_trigger_slots_keep_each_tenants_value() {
    // Tenant B's newer trigger value must not stand in for tenant A's.
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let store = open_store("scope_keys_repairable");
    let [a, b] = [tenant("tenant_a"), tenant("tenant_b")];
    for (scope, id) in [(&a, "rule_tenant_a"), (&b, "rule_tenant_b")] {
        store
            .add_rule(RuleInput {
                activation: RuleActivation::OnChange,
                ..owner_to_reviewer_rule(scope, id, 0)
            })
            .unwrap();
    }
    write_tenant_owners(&store);
    let repairable = store.repairable_trigger_slots(&scope(), 10, None).unwrap();
    close_store(store);
    assert_eq!(
        repairable
            .iter()
            .map(|slot| slot.current_value.clone())
            .collect::<Vec<_>>(),
        vec![Some("Ada".to_string()), Some("Bo".to_string())]
    );
}

#[test]
fn space_selection_trace_does_not_merge_tenant_histories() {
    // One claim per tenant on a slot is no revision history.
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let store = open_store("scope_keys_trace_histories");
    write_tenant_owners(&store);
    let request = answer_request("scope_keys", "project owner", &[], 10, 0, None);
    let projection = store
        .project_answer_ready_state(&scope(), &request)
        .unwrap();
    let trace = store
        .explain_answer_ready_state(&scope(), &request, &projection)
        .unwrap();
    close_store(store);
    assert_eq!(trace.slot_histories.len(), 0);
}

#[test]
fn space_slot_alias_cannot_cite_a_tenant_claim() {
    // An alias binds within its own scope; a tenant's claim is not evidence for a space alias.
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let store = open_store("scope_keys_slot_alias");
    write_tenant_owners(&store);
    let added = store.add_slot_alias(
        SlotAliasInput {
            id: None,
            scope: scope(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            alias_subject: "project".to_string(),
            alias_predicate: "lead".to_string(),
            canonical_slot_id: None,
            target_claim_id: Some("owner_a".to_string()),
            source_claim_ids: vec!["owner_a".to_string()],
            valid_from_ms: Some(10),
            valid_to_ms: None,
        },
        20,
    );
    close_store(store);
    assert!(
        added.is_err(),
        "space alias bound a tenant claim: {added:?}"
    );
}

fn add_project_entity(store: &MemoryStore, scope: &MemoryScope) {
    store
        .add_entity(
            EntityInput {
                id: None,
                scope: scope.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                entity_type: "project".to_string(),
                canonical_name: "Project".to_string(),
                aliases: Vec::new(),
                source_claim_ids: Vec::new(),
                merge_parent_ids: Vec::new(),
                split_from_id: None,
                confidence: None,
            },
            None,
        )
        .unwrap();
}

fn add_owner_claim(store: &MemoryStore, scope: &MemoryScope, value: &str) -> ClaimRecord {
    store
        .add_manual_claim(
            ManualClaimInput {
                id: None,
                scope: scope.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                claim_text: format!("The project owner is {value}."),
                subject: Some("project".to_string()),
                predicate: Some("owner".to_string()),
                object_value: Some(value.to_string()),
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
        .unwrap()
}

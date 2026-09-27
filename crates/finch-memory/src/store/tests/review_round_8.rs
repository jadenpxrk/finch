use super::*;

#[test]
fn generated_manual_claim_ids_collide_between_tenants() {
    // Identical manual facts in independent tenants must not contend for one generated ID.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("round_8_manual_claim_ids");
    let store = MemoryStore::create(&path, 3, CollectionOptions::default()).unwrap();
    let mut tenant_a = scope();
    tenant_a.tenant_id = Some("tenant_a".to_string());
    let mut tenant_b = scope();
    tenant_b.tenant_id = Some("tenant_b".to_string());
    let input = ManualClaimInput {
        id: None,
        scope: tenant_a.clone(),
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        claim_text: "User prefers Rust.".to_string(),
        subject: Some("user".to_string()),
        predicate: Some("language".to_string()),
        object_value: Some("Rust".to_string()),
        claim_kind: ClaimKind::Fact,
        polarity: ClaimPolarity::Affirmative,
        source_span_ids: Vec::new(),
        source_episode_ids: Vec::new(),
        asserted_by: "user".to_string(),
        confidence: None,
        observed_at_ms: 10,
        valid_from_ms: Some(10),
        valid_to_ms: None,
    };
    store.add_manual_claim(input.clone(), None).unwrap();
    let second = store.add_manual_claim(
        ManualClaimInput {
            scope: tenant_b.clone(),
            ..input
        },
        None,
    );
    drop(store);
    let reopened = MemoryStore::open(&path, CollectionOptions::default()).unwrap();
    let a = reopened
        .scan_current_claims(&tenant_a, 10, Some(20))
        .unwrap();
    let b = reopened
        .scan_current_claims(&tenant_b, 10, Some(20))
        .unwrap();
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);
    assert!(
        second.is_ok(),
        "tenant B's independent claim failed: {second:?}"
    );
    assert_eq!((a.len(), b.len()), (1, 1));
}

#[test]
fn space_wide_current_claim_read_collapses_independent_tenant_slots() {
    // Newest-wins resolution must not discard another tenant's current claim.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("round_8_current_claim_scope");
    let store = MemoryStore::create(&path, 3, CollectionOptions::default()).unwrap();
    for (tenant, id, value, at) in [
        ("tenant_a", "claim_a", "Rust", 10),
        ("tenant_b", "claim_b", "Python", 20),
    ] {
        let mut tenant_scope = scope();
        tenant_scope.tenant_id = Some(tenant.to_string());
        let mut claim = make_claim(
            &tenant_scope,
            id,
            "user",
            "language",
            Some(value),
            at,
            (ClaimKind::Fact, ClaimPolarity::Affirmative),
        );
        claim.source_span_ids.clear();
        claim.source_episode_ids.clear();
        store.append_claim(&claim, None).unwrap();
        let current = store
            .scan_current_claims(&tenant_scope, 10, Some(30))
            .unwrap();
        assert_eq!(current.len(), 1);
        assert_eq!(current[0].id, id);
    }
    drop(store);
    let reopened = MemoryStore::open(&path, CollectionOptions::default()).unwrap();
    let raw = reopened.scan_claims(&scope(), 10, Some(30)).unwrap();
    let current = reopened
        .scan_current_claims(&scope(), 10, Some(30))
        .unwrap();
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);
    assert_eq!(raw.len(), 2);
    assert_eq!(
        current
            .iter()
            .map(|claim| claim.id.as_str())
            .collect::<BTreeSet<_>>(),
        BTreeSet::from(["claim_a", "claim_b"])
    );
}

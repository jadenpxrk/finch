use super::*;

fn tenant(id: &str) -> MemoryScope {
    let mut scope = scope();
    scope.tenant_id = Some(id.to_string());
    scope
}

fn open_store(name: &str) -> EvidencedStore {
    EvidencedStore::create(&temp_dir(name), 3).unwrap()
}

fn entity(scope: &MemoryScope, id: &str) -> EntityInput {
    EntityInput {
        id: Some(id.to_string()),
        scope: scope.clone(),
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        entity_type: "project".to_string(),
        canonical_name: id.to_string(),
        aliases: Vec::new(),
        source_claim_ids: Vec::new(),
        merge_parent_ids: Vec::new(),
        split_from_id: None,
        confidence: Some(1.0),
    }
}

#[test]
fn claims_by_ids_returns_only_the_callers_scope_in_id_order() {
    let store = open_store("claims_by_ids_scope");
    let (a, b) = (tenant("a"), tenant("b"));
    for (scope, id) in [
        (&a, "claim_3"),
        (&b, "claim_2"),
        (&a, "claim_1"),
        (&b, "claim_4"),
    ] {
        let claim = make_claim(
            scope,
            id,
            "project",
            "owner",
            Some(id),
            10,
            (ClaimKind::Fact, ClaimPolarity::Affirmative),
        );
        store.append_claim(&claim, None).unwrap();
    }
    let ask = [
        "claim_4", "claim_3", "missing", "claim_2", "claim_1", "claim_3",
    ]
    .map(String::from);

    let ids = |scope: &MemoryScope| {
        store
            .claims_by_ids(scope, &ask)
            .unwrap()
            .into_iter()
            .map(|claim| claim.id)
            .collect::<Vec<_>>()
    };
    assert_eq!(ids(&a), ["claim_1", "claim_3"]);
    assert_eq!(ids(&b), ["claim_2", "claim_4"]);
}

#[test]
fn entities_by_ids_returns_only_the_callers_active_scope_in_id_order() {
    let store = open_store("entities_by_ids_scope");
    let (a, b) = (tenant("a"), tenant("b"));
    for (scope, id) in [
        (&a, "entity_3"),
        (&b, "entity_2"),
        (&a, "entity_1"),
        (&b, "entity_4"),
    ] {
        store.add_entity(entity(scope, id), None).unwrap();
    }
    let mut merged = store.add_entity(entity(&a, "entity_0"), None).unwrap();
    merged.status = MemoryStatus::Superseded;
    let statuses = store
        .entities
        .upsert(vec![entity_doc(&merged, None).unwrap()])
        .unwrap();
    assert!(statuses.iter().all(Status::ok));
    let ask = [
        "entity_4", "entity_3", "entity_0", "missing", "entity_2", "entity_1",
    ]
    .map(String::from);

    let ids = |scope: &MemoryScope| {
        store
            .entities_by_ids(scope, &ask)
            .unwrap()
            .into_iter()
            .map(|entity| entity.id)
            .collect::<Vec<_>>()
    };
    assert_eq!(ids(&a), ["entity_1", "entity_3"]);
    assert_eq!(ids(&b), ["entity_2", "entity_4"]);
}

#[test]
fn fetch_spans_by_ids_returns_each_span_once_in_id_order() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let store = open_store("fetch_spans_order");
    let scope = scope();
    let ids = (0..40).map(|n| format!("span_{n:02}")).collect::<Vec<_>>();
    store.seed_evidence(&scope, &ids, &["ep_spans".to_string()]);
    let mut ask = ids.iter().rev().cloned().collect::<Vec<_>>();
    ask.push(ids[0].clone());

    let fetched = store
        .fetch_spans_by_ids(&scope, &ask, None)
        .unwrap()
        .into_iter()
        .map(|span| span.id)
        .collect::<Vec<_>>();

    assert_eq!(fetched, ids);
}

#[test]
fn get_claim_finds_a_claim_after_any_number_of_earlier_claims() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let store = open_store("get_claim_by_id");
    let (a, b) = (tenant("a"), tenant("b"));
    for (scope, id) in [(&a, "claim_first"), (&a, "claim_last"), (&b, "claim_other")] {
        let claim = make_claim(
            scope,
            id,
            "project",
            id,
            Some("value"),
            10,
            (ClaimKind::Fact, ClaimPolarity::Affirmative),
        );
        store.append_claim(&claim, None).unwrap();
    }
    let found = |scope: &MemoryScope, id: &str, at_ms| {
        store
            .get_claim(scope, id, at_ms)
            .unwrap()
            .map(|claim| claim.id)
    };

    assert_eq!(found(&a, "claim_last", None).as_deref(), Some("claim_last"));
    assert_eq!(found(&a, "claim_other", None), None);
    assert_eq!(found(&a, "claim_last", Some(5)), None);
    assert_eq!(found(&a, "missing", None), None);
}

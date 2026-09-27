use super::*;

#[test]
fn tenant_selector_correction_hides_another_tenants_span() {
    // A tenant's selector correction must not affect another tenant in a space-wide read.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("round_2_correction_scope");
    let store = MemoryStore::create(&path, 3, CollectionOptions::default()).unwrap();
    let mut a = span_record("span_a", MemoryStatus::Active, Some(10));
    a.scope.tenant_id = Some("tenant_a".to_string());
    a.text = "shared note".to_string();
    let mut b = a.clone();
    b.id = "span_b".to_string();
    b.scope.tenant_id = Some("tenant_b".to_string());
    store
        .append_vector_spans(&[
            (a.clone(), vec![1.0, 0.0, 0.0]),
            (b.clone(), vec![1.0, 0.0, 0.0]),
        ])
        .unwrap();
    let correction = crate::ingest::add_correction(
        CorrectionInput {
            id: Some("forget_a".to_string()),
            scope: a.scope.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            operation: CorrectionOperation::Forget,
            target_type: "span".to_string(),
            target_ids: Vec::new(),
            target_selector: Some("shared note".to_string()),
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
    drop(store);
    let store = MemoryStore::open(&path, CollectionOptions::default()).unwrap();
    let a_hits = store
        .query_spans(&a.scope, vec![1.0, 0.0, 0.0], 10, Some(30))
        .unwrap();
    let b_hits = store
        .query_spans(&b.scope, vec![1.0, 0.0, 0.0], 10, Some(30))
        .unwrap();
    let all_hits = store
        .query_spans(&scope(), vec![1.0, 0.0, 0.0], 10, Some(30))
        .unwrap();
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);
    assert!(a_hits.is_empty());
    assert_eq!(b_hits.len(), 1);
    assert_eq!(b_hits[0].span.id, b.id);
    assert_eq!(
        all_hits.len(),
        1,
        "tenant A's correction suppressed tenant B's span"
    );
    assert_eq!(all_hits[0].span.id, b.id);
}

#[test]
fn generated_episode_ids_collide_between_tenants() {
    // Independent tenants must be able to ingest identical events without an ID conflict.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("round_2_episode_scope");
    let store = MemoryStore::create(&path, 3, CollectionOptions::default()).unwrap();
    let mut tenant_a = scope();
    tenant_a.tenant_id = Some("tenant_a".to_string());
    let mut tenant_b = scope();
    tenant_b.tenant_id = Some("tenant_b".to_string());
    let input = EpisodeInput {
        id: None,
        scope: tenant_a.clone(),
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        source_kind: SourceKind::UserMessage,
        actor: ActorKind::User,
        sequence_no: 1,
        event_time_ms: Some(10),
        valid_from_ms: Some(10),
        valid_to_ms: None,
        raw_text: "Hello".to_string(),
        blob_ref: None,
        mime_type: None,
        causal_parent_ids: Vec::new(),
        metadata_json: None,
    };
    let first = store
        .ingest_episode(input.clone(), 10, &ChunkOptions::default())
        .unwrap();
    let second = store.ingest_episode(
        EpisodeInput {
            scope: tenant_b.clone(),
            ..input
        },
        10,
        &ChunkOptions::default(),
    );
    drop(store);
    let reopened = MemoryStore::open(&path, CollectionOptions::default()).unwrap();
    let a = reopened.scan_spans(&tenant_a, 10, Some(20)).unwrap();
    let b = reopened.scan_spans(&tenant_b, 10, Some(20)).unwrap();
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);
    assert_eq!(a.len(), 1);
    let second = second.expect("automatic episode IDs must include the tenant scope");
    assert_ne!(first.episode.id, second.episode.id);
    assert_eq!(b.len(), 1);
}

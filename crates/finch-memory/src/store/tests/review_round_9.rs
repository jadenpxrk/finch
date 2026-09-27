use super::*;

#[test]
fn trailing_backslash_in_project_scope_makes_stored_spans_unsearchable() {
    // A project path accepted on write must remain usable as a scope filter.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("round_9_backslash_scope");
    let store = MemoryStore::create(&path, 3, CollectionOptions::default()).unwrap();
    let mut span = span_record("project_span", MemoryStatus::Active, Some(10));
    span.scope.project_id = Some("C:\\notes\\".to_string());
    store
        .append_vector_spans(&[(span.clone(), vec![1.0, 0.0, 0.0])])
        .unwrap();
    drop(store);
    let reopened = MemoryStore::open(&path, CollectionOptions::default()).unwrap();
    let fetched = reopened
        .fetch_spans_by_ids(&span.scope, std::slice::from_ref(&span.id), Some(20))
        .unwrap();
    let searched = reopened.query_spans(&span.scope, vec![1.0, 0.0, 0.0], 1, Some(20));
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);
    assert_eq!(fetched.len(), 1);
    let searched = searched.expect("a project scope ending in a backslash must be queryable");
    assert_eq!(searched.len(), 1);
    assert_eq!(searched[0].span.id, span.id);
}

#[test]
fn distinct_selector_corrections_created_together_collide_on_generated_id() {
    // Independent selector corrections in one millisecond must retain distinct identities.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("round_9_selector_correction_ids");
    let store = MemoryStore::create(&path, 3, CollectionOptions::default()).unwrap();
    let a = span_record("alpha_note", MemoryStatus::Active, Some(10));
    let b = span_record("beta_note", MemoryStatus::Active, Some(10));
    store
        .append_vector_spans(&[(a, vec![1.0, 0.0, 0.0]), (b, vec![0.0, 1.0, 0.0])])
        .unwrap();
    let corrections = ["alpha_note", "beta_note"].map(|selector| {
        crate::ingest::add_correction(
            CorrectionInput {
                id: None,
                scope: scope(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                operation: CorrectionOperation::Forget,
                target_type: "span".to_string(),
                target_ids: Vec::new(),
                target_selector: Some(selector.to_string()),
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
        )
    });
    store.add_correction(&corrections[0]).unwrap();
    let second = store.add_correction(&corrections[1]);
    drop(store);
    let reopened = MemoryStore::open(&path, CollectionOptions::default()).unwrap();
    let stored = reopened.scan_corrections(&scope(), 10).unwrap();
    let visible = reopened
        .query_spans(&scope(), vec![1.0, 0.0, 0.0], 10, Some(30))
        .unwrap();
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);
    assert!(second.is_ok(), "second selector correction failed: {second:?}; stored corrections: {}; visible spans: {:?}", stored.len(), visible.iter().map(|hit| &hit.span.id).collect::<Vec<_>>());
    assert_ne!(corrections[0].id, corrections[1].id);
    assert_eq!(stored.len(), 2);
    assert!(visible.is_empty());
}

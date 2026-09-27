use super::*;

#[test]
fn expired_profile_consumes_limit_and_hides_active_profile() {
    // Expired profiles must not consume the caller's limit on active profiles.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("round_3_profile_limit");
    let store = MemoryStore::create(&path, 3, CollectionOptions::default()).unwrap();
    for (id, valid_to_ms) in [("expired", Some(20)), ("active", None)] {
        store
            .add_profile(ProfileInput {
                id: Some(id.to_string()),
                scope: scope(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                subject_id: None,
                profile_key: "style".to_string(),
                profile_text: id.to_string(),
                value_json: None,
                evidence_claim_ids: Vec::new(),
                source_span_ids: Vec::new(),
                generated_at_ms: 10,
                generator_version: "manual".to_string(),
                valid_from_ms: Some(10),
                valid_to_ms,
                correction_watermark: 10,
            })
            .unwrap();
    }
    drop(store);
    let reopened = MemoryStore::open(&path, CollectionOptions::default()).unwrap();
    let all = reopened.scan_profiles(&scope(), 10, Some(30)).unwrap();
    let limited = reopened.scan_profiles(&scope(), 1, Some(30)).unwrap();
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].id, "active");
    assert_eq!(
        limited
            .iter()
            .map(|profile| profile.id.as_str())
            .collect::<Vec<_>>(),
        vec!["active"]
    );
}

#[test]
fn selected_source_hydration_returns_forgotten_span() {
    // Source hydration must honor Forget corrections just as direct vector retrieval does.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("round_3_forgotten_source");
    let index_path = temp_dir("round_3_forgotten_source_index");
    let store = MemoryStore::create(&path, 3, CollectionOptions::default()).unwrap();
    let span = span_record("forgotten", MemoryStatus::Active, Some(10));
    store
        .append_vector_spans(&[(span.clone(), vec![1.0, 0.0, 0.0])])
        .unwrap();
    let mut builder = crate::source_index::SourceLexicalIndexBuilder::new();
    builder
        .add_source(&span.source_id, "", "", &span.text, vec![span.id.clone()])
        .unwrap();
    builder.write(&index_path).unwrap();
    let correction = crate::ingest::add_correction(
        CorrectionInput {
            id: Some("forget_source_span".to_string()),
            scope: scope(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            operation: CorrectionOperation::Forget,
            target_type: "span".to_string(),
            target_ids: vec![span.id.clone()],
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
    drop(store);
    let reopened = MemoryStore::open(&path, CollectionOptions::default()).unwrap();
    let direct = reopened
        .query_spans(&scope(), vec![1.0, 0.0, 0.0], 10, Some(30))
        .unwrap();
    let index = crate::source_index::SourceLexicalIndex::open(&index_path).unwrap();
    let hydrated = reopened
        .hydrate_selected_sources(
            &scope(),
            &[SourceSpanSelection {
                source_id: span.source_id,
                max_spans: 1,
            }],
            &[],
            &index,
            Some(30),
        )
        .unwrap();
    drop(index);
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
    std::fs::remove_dir_all(index_path).unwrap();
    drop(guard);
    assert!(direct.is_empty());
    assert!(
        hydrated.is_empty(),
        "source hydration returned forgotten text: {hydrated:?}"
    );
}

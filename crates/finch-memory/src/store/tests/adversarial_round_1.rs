use super::*;

#[test]
fn apostrophe_in_tenant_id_makes_stored_spans_unsearchable() {
    // Scope values accepted at ingestion must round-trip through the generated filter dialect.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("round_1_quoted_scope");
    let store = MemoryStore::create(&path, 3, CollectionOptions::default()).unwrap();
    let mut span = span_record("quoted_tenant_span", MemoryStatus::Active, Some(10));
    span.scope.tenant_id = Some("o'brien".to_string());
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
    let searched =
        searched.expect("a persisted tenant ID containing an apostrophe must be queryable");
    assert_eq!(searched.len(), 1);
    assert_eq!(searched[0].span.id, span.id);
}

#[test]
fn evidence_completion_resurrects_forgotten_neighbor_text() {
    // Evidence expansion must honor the same Forget correction as direct retrieval.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("round_1_forgotten_neighbor");
    let store = MemoryStore::create(&path, 3, CollectionOptions::default()).unwrap();
    let ingested = store
        .ingest_episode(
            EpisodeInput {
                id: Some("episode".to_string()),
                scope: scope(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                source_kind: SourceKind::UserMessage,
                actor: ActorKind::User,
                sequence_no: 1,
                event_time_ms: Some(10),
                valid_from_ms: Some(10),
                valid_to_ms: None,
                raw_text: "current obsolete".to_string(),
                blob_ref: None,
                mime_type: None,
                causal_parent_ids: Vec::new(),
                metadata_json: None,
            },
            10,
            &ChunkOptions {
                max_chars: 8,
                overlap_chars: 0,
                chunker_version: "test".to_string(),
            },
        )
        .unwrap();
    assert_eq!(ingested.spans.len(), 2);
    let forgotten_id = ingested.spans[1].id.clone();
    let correction = crate::ingest::add_correction(
        CorrectionInput {
            id: Some("forget_neighbor".to_string()),
            scope: scope(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            operation: CorrectionOperation::Forget,
            target_type: "span".to_string(),
            target_ids: vec![forgotten_id.clone()],
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
        .keyword_search_spans(&scope(), "obsolete", 10, 100, Some(30))
        .unwrap();
    let anchors = reopened
        .keyword_search_spans(&scope(), "current", 10, 100, Some(30))
        .unwrap();
    let expanded = reopened
        .complete_span_evidence(&scope(), &anchors, 0, 1, 100, Some(30))
        .unwrap();
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);
    assert!(direct.is_empty());
    assert_eq!(anchors.len(), 1);
    assert!(
        expanded.iter().all(|hit| hit.span.id != forgotten_id),
        "the forgotten span reappeared in completed evidence"
    );
}

#[test]
fn expired_edge_consumes_limit_and_hides_live_graph_neighbor() {
    // Expired edges must not consume the requested count of currently valid neighbors.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("round_1_expired_edge_limit");
    let store = MemoryStore::create(&path, 3, CollectionOptions::default()).unwrap();
    for (id, valid_to_ms) in [("expired", Some(20)), ("live", None)] {
        store
            .add_edge(EdgeInput {
                id: Some(id.to_string()),
                scope: scope(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                src_entity_id: "seed".to_string(),
                dst_entity_id: format!("neighbor_{id}"),
                relation_type: "mentions".to_string(),
                claim_id: None,
                source_span_ids: Vec::new(),
                valid_from_ms: Some(10),
                valid_to_ms,
                confidence: Some(1.0),
            })
            .unwrap();
    }
    drop(store);
    let reopened = MemoryStore::open(&path, CollectionOptions::default()).unwrap();
    let all = reopened
        .expand_edges_ranked(&scope(), &["seed".to_string()], 1, 2, Some(30))
        .unwrap();
    let limited = reopened
        .expand_edges_ranked(&scope(), &["seed".to_string()], 1, 1, Some(30))
        .unwrap();
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].edge.id, "live");
    assert_eq!(
        limited.len(),
        1,
        "the live edge must survive a limit of one"
    );
    assert_eq!(limited[0].edge.id, "live");
}

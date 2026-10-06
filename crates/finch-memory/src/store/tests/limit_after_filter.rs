use super::*;

// A span or claim vector search with k = 1 fetches this many candidates before filtering.
const VECTOR_FETCH_K: usize = 8;

fn correction(
    id: &str,
    operation: CorrectionOperation,
    target_type: &str,
    target_ids: Vec<String>,
    target_selector: Option<&str>,
    effective_at_ms: i64,
) -> CorrectionRecord {
    crate::ingest::add_correction(
        CorrectionInput {
            id: Some(id.to_string()),
            scope: scope(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            operation,
            target_type: target_type.to_string(),
            target_ids,
            target_selector: target_selector.map(str::to_string),
            new_value: None,
            reason: None,
            actor: ActorKind::User,
            authority: CorrectionAuthority::User,
            effective_at_ms: Some(effective_at_ms),
            applies_valid_from_ms: None,
            applies_valid_to_ms: None,
            cascade_policy: None,
            metadata_json: None,
        },
        effective_at_ms,
    )
}

fn owner_claim(id: &str, subject: &str, observed_at_ms: i64) -> ClaimRecord {
    make_claim(
        &scope(),
        id,
        subject,
        "owner",
        Some("Morgan"),
        observed_at_ms,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    )
}

fn stored_slot_id(store: &MemoryStore, claim_id: &str) -> MemoryId {
    store
        .claims_by_ids(&scope(), &[claim_id.to_string()])
        .unwrap()
        .remove(0)
        .slot_id
        .unwrap()
}

fn expired_span(id: &str) -> SpanRecord {
    let mut span = span_record(id, MemoryStatus::Active, Some(10));
    span.valid_to_ms = Some(20);
    span
}

#[test]
fn expired_spans_consume_vector_fetch_and_hide_active_span() {
    // Expired spans nearer the query must not fill the vector fetch ahead of a live span.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("limit_after_filter_span_vector");
    let store = crate::test_store(&path, 3).unwrap();
    let mut records = (0..VECTOR_FETCH_K)
        .map(|i| (expired_span(&format!("expired_{i}")), vec![1.0, 0.0, 0.0]))
        .collect::<Vec<_>>();
    records.push((
        span_record("active", MemoryStatus::Active, Some(10)),
        vec![0.8, 0.6, 0.0],
    ));
    store.append_vector_spans(&records).unwrap();
    let hits = store
        .query_spans(&scope(), vec![1.0, 0.0, 0.0], 1, Some(30))
        .unwrap();
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);
    assert_eq!(
        hits.iter()
            .map(|hit| hit.span.id.as_str())
            .collect::<Vec<_>>(),
        vec!["active"]
    );
}

#[test]
fn future_claims_consume_vector_fetch_and_hide_valid_slot() {
    // Claims not yet valid must not fill the claim vector fetch ahead of a valid claim.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("limit_after_filter_claim_vector");
    let store = EvidencedStore::create(&path, 3).unwrap();
    for i in 0..VECTOR_FETCH_K {
        let future = owner_claim(&format!("future_{i}"), &format!("future_{i}"), 100);
        store.append_claim(&future, Some(&[1.0, 0.0, 0.0])).unwrap();
    }
    let valid = owner_claim("valid", "project", 10);
    store.append_claim(&valid, Some(&[0.8, 0.6, 0.0])).unwrap();
    let rankings = store
        .query_current_state_slot_rankings(&scope(), vec![1.0, 0.0, 0.0], "", 1, Some(50))
        .unwrap();
    let valid_slot_id = stored_slot_id(&store, &valid.id);
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);
    assert_eq!(rankings.vector, vec![valid_slot_id]);
}

#[test]
fn expired_span_postings_consume_scan_limit_and_hide_active_span() {
    // Postings of expired spans must not fill the keyword scan limit ahead of a live span.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("limit_after_filter_keyword");
    let store = crate::test_store(&path, 3).unwrap();
    // A keyword search for one hit scans at least eight postings.
    let expired = (0..8).map(|i| expired_span(&format!("expired_{i}")));
    let active = span_record("active", MemoryStatus::Active, Some(10));
    for mut span in expired.chain(std::iter::once(active)) {
        span.text = "beacon".to_string();
        span.lexical_text = span.text.clone();
        store.append_span(&span, None).unwrap();
    }
    let hits = store
        .keyword_search_spans(&scope(), "beacon", 1, 1, Some(30))
        .unwrap();
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);
    assert_eq!(
        hits.iter()
            .map(|hit| hit.span.id.as_str())
            .collect::<Vec<_>>(),
        vec!["active"]
    );
}

#[test]
fn keyword_search_reads_postings_past_a_page_of_expired_spans() {
    // Postings are read in doubling pages; 200 expired spans fill the first pages.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("limit_after_filter_keyword_pages");
    let store = crate::test_store(&path, 3).unwrap();
    let expired = (0..200).map(|i| expired_span(&format!("expired_{i}")));
    let active = span_record("active", MemoryStatus::Active, Some(10));
    for mut span in expired.chain(std::iter::once(active)) {
        span.text = "beacon".to_string();
        span.lexical_text = span.text.clone();
        store.append_span(&span, None).unwrap();
    }
    let hits = store
        .keyword_search_spans(&scope(), "beacon", 1, 1, Some(30))
        .unwrap();
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);
    assert_eq!(
        hits.iter()
            .map(|hit| hit.span.id.as_str())
            .collect::<Vec<_>>(),
        vec!["active"]
    );
}

#[test]
fn context_corrections_keep_more_than_1024_claim_corrections() {
    // Context corrections must not drop the latest correction once a claim has more than 1024.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("limit_after_filter_context_corrections");
    let store = EvidencedStore::create(&path, 3).unwrap();
    let claim = owner_claim("claim", "project", 10);
    store.append_claim(&claim, None).unwrap();
    for i in 0..SATURATING_CORRECTION_COUNT {
        let assert = correction(
            &format!("assert_{i}"),
            CorrectionOperation::Assert,
            "claim",
            vec![claim.id.clone()],
            None,
            20,
        );
        store.add_correction(&assert).unwrap();
    }
    let latest = correction(
        "latest",
        CorrectionOperation::Assert,
        "claim",
        vec![claim.id.clone()],
        None,
        21,
    );
    store.add_correction(&latest).unwrap();
    let corrections = store
        .scan_context_corrections(&scope(), std::slice::from_ref(&claim), &[], Some(22))
        .unwrap();
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);
    assert_eq!(corrections.len(), SATURATING_CORRECTION_COUNT + 1);
    assert_eq!(corrections.last().unwrap().id, "latest");
}

#[test]
fn claim_selector_correction_after_1024_selector_corrections_is_scanned() {
    // A claim selector correction must be found however many selector corrections precede it.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("limit_after_filter_claim_selector");
    let store = EvidencedStore::create(&path, 3).unwrap();
    let claim = owner_claim("claim", "project", 10);
    store.append_claim(&claim, None).unwrap();
    for i in 0..SATURATING_CORRECTION_COUNT {
        let assert = correction(
            &format!("assert_{i}"),
            CorrectionOperation::Assert,
            "claim",
            Vec::new(),
            Some(&format!("unrelated_{i}")),
            20,
        );
        store.add_correction(&assert).unwrap();
    }
    let latest = correction(
        "latest",
        CorrectionOperation::Assert,
        "claim",
        Vec::new(),
        Some("Morgan"),
        20,
    );
    store.add_correction(&latest).unwrap();
    let corrections = store
        .scan_context_corrections(&scope(), std::slice::from_ref(&claim), &[], Some(22))
        .unwrap();
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);
    assert!(corrections.iter().any(|record| record.id == "latest"));
}

#[test]
fn claim_slot_tombstone_after_1024_slot_corrections_applies() {
    // A slot tombstone must retract the claim however many slot corrections precede it.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("limit_after_filter_claim_slot");
    let store = EvidencedStore::create(&path, 3).unwrap();
    let claim = owner_claim("claim", "project", 10);
    store.append_claim(&claim, None).unwrap();
    let slot_id = stored_slot_id(&store, &claim.id);
    for i in 0..SATURATING_CORRECTION_COUNT {
        let mut assert = correction(
            &format!("assert_{i}"),
            CorrectionOperation::Assert,
            "claim",
            Vec::new(),
            None,
            20,
        );
        assert.target_slot_id = Some(slot_id.clone());
        store.add_correction(&assert).unwrap();
    }
    let mut tombstone = correction(
        "tombstone",
        CorrectionOperation::Tombstone,
        "claim",
        Vec::new(),
        None,
        21,
    );
    tombstone.target_match = crate::CorrectionTargetMatch::CanonicalSlot;
    tombstone.target_slot_id = Some(slot_id);
    store.add_correction(&tombstone).unwrap();
    let current = store.scan_current_claims(&scope(), 1, Some(22)).unwrap();
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);
    assert_eq!(current.len(), 1);
    assert_eq!(current[0].status, MemoryStatus::Tombstoned);
}

#[test]
fn span_tombstone_after_1024_span_corrections_hides_span() {
    // A span tombstone must hide the span however many corrections name it first.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("limit_after_filter_span_ids");
    let store = crate::test_store(&path, 3).unwrap();
    let span = span_record("span", MemoryStatus::Active, Some(10));
    store.append_span(&span, Some(&[1.0, 0.0, 0.0])).unwrap();
    for i in 0..SATURATING_CORRECTION_COUNT {
        let assert = correction(
            &format!("assert_{i}"),
            CorrectionOperation::Assert,
            "span",
            vec![span.id.clone()],
            None,
            20,
        );
        store.add_correction(&assert).unwrap();
    }
    let tombstone = correction(
        "tombstone",
        CorrectionOperation::Tombstone,
        "span",
        vec![span.id.clone()],
        None,
        21,
    );
    store.add_correction(&tombstone).unwrap();
    let hits = store
        .query_spans(&scope(), vec![1.0, 0.0, 0.0], 1, Some(22))
        .unwrap();
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);
    assert!(hits.is_empty());
}

#[test]
fn span_selector_tombstone_after_1024_selector_corrections_hides_span() {
    // A span selector tombstone must hide the span however many selector corrections precede it.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("limit_after_filter_span_selector");
    let store = crate::test_store(&path, 3).unwrap();
    let mut span = span_record("span", MemoryStatus::Active, Some(10));
    span.text = "Runbook token SELECTOR-TOMBSTONE.".to_string();
    span.lexical_text = span.text.clone();
    store.append_span(&span, Some(&[1.0, 0.0, 0.0])).unwrap();
    for i in 0..SATURATING_CORRECTION_COUNT {
        let assert = correction(
            &format!("assert_{i}"),
            CorrectionOperation::Assert,
            "span",
            Vec::new(),
            Some(&format!("unrelated_{i}")),
            20,
        );
        store.add_correction(&assert).unwrap();
    }
    let tombstone = correction(
        "tombstone",
        CorrectionOperation::Tombstone,
        "span",
        Vec::new(),
        Some("SELECTOR-TOMBSTONE"),
        21,
    );
    store.add_correction(&tombstone).unwrap();
    let hits = store
        .query_spans(&scope(), vec![1.0, 0.0, 0.0], 1, Some(22))
        .unwrap();
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);
    assert!(hits.is_empty());
}

#[test]
fn expired_edge_does_not_use_up_the_limit_of_live_graph_neighbors() {
    // Expired edges must not consume the requested count of currently valid neighbors.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("expired_edge_limit");
    let store = crate::test_store(&path, 3).unwrap();
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
    let reopened = crate::reopen_test_store(&path).unwrap();
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

#[test]
fn expired_artifact_does_not_use_up_the_limit_of_active_artifacts() {
    // Expired artifacts must not consume the caller's limit on active artifacts.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("artifact_limit");
    let store = crate::test_store(&path, 3).unwrap();
    for (id, valid_to_ms) in [("expired", Some(20)), ("active", None)] {
        store
            .ingest_artifact_text(
                ArtifactRecord {
                    id: id.to_string(),
                    scope: scope(),
                    status: MemoryStatus::Active,
                    visibility: Visibility::Private,
                    policy_tags: Vec::new(),
                    artifact_kind: ArtifactKind::File,
                    title: id.to_string(),
                    uri: None,
                    blob_ref: None,
                    mime_type: Some("text/plain".to_string()),
                    content_hash: id.to_string(),
                    source_created_at_ms: None,
                    source_modified_at_ms: None,
                    created_at_ms: 10,
                    ingested_at_ms: 10,
                    valid_from_ms: Some(10),
                    valid_to_ms,
                    extracted_text_ref: None,
                    metadata_json: None,
                },
                id,
                &ChunkOptions::default(),
            )
            .unwrap();
    }
    drop(store);
    let reopened = crate::reopen_test_store(&path).unwrap();
    let all = reopened.scan_artifacts(&scope(), 10, Some(30)).unwrap();
    let limited = reopened.scan_artifacts(&scope(), 1, Some(30)).unwrap();
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);
    assert_eq!(
        all.iter()
            .map(|record| record.id.as_str())
            .collect::<Vec<_>>(),
        vec!["active"]
    );
    assert_eq!(
        limited
            .iter()
            .map(|record| record.id.as_str())
            .collect::<Vec<_>>(),
        vec!["active"]
    );
}

#[test]
fn expired_profile_does_not_use_up_the_limit_of_active_profiles() {
    // Expired profiles must not consume the caller's limit on active profiles.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("profile_limit");
    let store = crate::test_store(&path, 3).unwrap();
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
    let reopened = crate::reopen_test_store(&path).unwrap();
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
fn entity_scan_returns_at_most_its_limit() {
    // Entity scans must enforce their public result limit after scope and status filtering.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("entity_limit");
    let store = crate::test_store(&path, 3).unwrap();
    for id in ["first", "second"] {
        store
            .add_entity(
                EntityInput {
                    id: Some(id.to_string()),
                    scope: scope(),
                    visibility: Visibility::Private,
                    policy_tags: Vec::new(),
                    entity_type: "project".to_string(),
                    canonical_name: id.to_string(),
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
    drop(store);
    let reopened = crate::reopen_test_store(&path).unwrap();
    let all = reopened.scan_entities(&scope(), 10).unwrap();
    let limited = reopened.scan_entities(&scope(), 1).unwrap();
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);
    assert_eq!(all.len(), 2);
    assert_eq!(limited.len(), 1);
}

#[test]
fn span_scan_returns_at_most_its_limit() {
    // A scan must enforce its public result limit after scope and validity filtering.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("span_scan_limit");
    let store = crate::test_store(&path, 3).unwrap();
    let records = ["first", "second"]
        .into_iter()
        .map(|id| {
            (
                span_record(id, MemoryStatus::Active, Some(10)),
                vec![1.0, 0.0, 0.0],
            )
        })
        .collect::<Vec<_>>();
    store.append_vector_spans(&records).unwrap();
    drop(store);
    let reopened = crate::reopen_test_store(&path).unwrap();
    let all = reopened.scan_spans(&scope(), 10, Some(20)).unwrap();
    let limited = reopened.scan_spans(&scope(), 1, Some(20)).unwrap();
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);
    assert_eq!(all.len(), 2);
    assert_eq!(limited.len(), 1);
}

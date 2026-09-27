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
    let store = MemoryStore::create(&path, 3, CollectionOptions::default()).unwrap();
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
    let store = MemoryStore::create(&path, 3, CollectionOptions::default()).unwrap();
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
    let store = MemoryStore::create(&path, 3, CollectionOptions::default()).unwrap();
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
fn context_corrections_keep_more_than_1024_claim_corrections() {
    // Context corrections must not drop the latest correction once a claim has more than 1024.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("limit_after_filter_context_corrections");
    let store = MemoryStore::create(&path, 3, CollectionOptions::default()).unwrap();
    let claim = owner_claim("claim", "project", 10);
    store.append_claim(&claim, None).unwrap();
    for i in 0..DEFAULT_CORRECTION_SCAN_LIMIT {
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
    assert_eq!(corrections.len(), DEFAULT_CORRECTION_SCAN_LIMIT + 1);
    assert_eq!(corrections.last().unwrap().id, "latest");
}

#[test]
fn claim_selector_correction_after_1024_selector_corrections_is_scanned() {
    // A claim selector correction must be found however many selector corrections precede it.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("limit_after_filter_claim_selector");
    let store = MemoryStore::create(&path, 3, CollectionOptions::default()).unwrap();
    let claim = owner_claim("claim", "project", 10);
    store.append_claim(&claim, None).unwrap();
    for i in 0..DEFAULT_CORRECTION_SCAN_LIMIT {
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
    let store = MemoryStore::create(&path, 3, CollectionOptions::default()).unwrap();
    let claim = owner_claim("claim", "project", 10);
    store.append_claim(&claim, None).unwrap();
    let slot_id = stored_slot_id(&store, &claim.id);
    for i in 0..DEFAULT_CORRECTION_SCAN_LIMIT {
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
    let store = MemoryStore::create(&path, 3, CollectionOptions::default()).unwrap();
    let span = span_record("span", MemoryStatus::Active, Some(10));
    store.append_span(&span, Some(&[1.0, 0.0, 0.0])).unwrap();
    for i in 0..DEFAULT_CORRECTION_SCAN_LIMIT {
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
    let store = MemoryStore::create(&path, 3, CollectionOptions::default()).unwrap();
    let mut span = span_record("span", MemoryStatus::Active, Some(10));
    span.text = "Runbook token SELECTOR-TOMBSTONE.".to_string();
    span.lexical_text = span.text.clone();
    store.append_span(&span, Some(&[1.0, 0.0, 0.0])).unwrap();
    for i in 0..DEFAULT_CORRECTION_SCAN_LIMIT {
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

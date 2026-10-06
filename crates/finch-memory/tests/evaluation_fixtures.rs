use finch_memory::{
    add_correction, evaluate_retrieval_baseline, evaluate_span_hits_with_admission, ActorKind,
    ChunkOptions, ClaimKind, ClaimPolarity, CorrectionAuthority, CorrectionInput,
    CorrectionOperation, EpisodeInput, HybridSpanSearch, ManualClaimInput, MemoryScope,
    MemoryStore, RetrievalBaselineHits, SourceKind, Visibility,
};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

static STORE_MUTEX: Mutex<()> = Mutex::new(());

fn temp_dir(name: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "finch_memory_eval_{}_{}_{}",
        name,
        std::process::id(),
        n
    ));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// A new, empty store named after `dir` on the server in FINCH_MEMORY_TEST_POSTGRES_URL.
fn new_store(dir: &std::path::Path, embedding_dim: usize) -> finch_types::ZResult<MemoryStore> {
    let url = std::env::var("FINCH_MEMORY_TEST_POSTGRES_URL")
        .expect("set FINCH_MEMORY_TEST_POSTGRES_URL to a Postgres server with pgvector");
    let name = format!(
        "t_{}",
        finch_memory::stable_hash_hex(&[&dir.to_string_lossy()])
    );
    finch_memory::drop_store(&url, &name)?;
    MemoryStore::create(&url, &name, embedding_dim, false)
}

fn scope(user_id: &str) -> MemoryScope {
    let mut scope = MemoryScope::new("eval");
    scope.user_id = Some(user_id.to_string());
    scope
}

fn remember(
    store: &MemoryStore,
    id: &str,
    scope: MemoryScope,
    text: &str,
    event_time_ms: i64,
    valid_to_ms: Option<i64>,
) -> String {
    let ingested = store
        .ingest_episode(
            EpisodeInput {
                id: Some(id.to_string()),
                scope,
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                source_kind: SourceKind::UserMessage,
                actor: ActorKind::User,
                sequence_no: event_time_ms,
                event_time_ms: Some(event_time_ms),
                valid_from_ms: Some(event_time_ms),
                valid_to_ms,
                raw_text: text.to_string(),
                blob_ref: None,
                mime_type: Some("text/plain".to_string()),
                causal_parent_ids: Vec::new(),
                metadata_json: None,
            },
            event_time_ms,
            &ChunkOptions {
                max_chars: 500,
                overlap_chars: 0,
                chunker_version: "eval".to_string(),
            },
        )
        .unwrap();
    ingested.spans[0].id.clone()
}

#[test]
fn production_writes_reject_fabricated_evidence_references() {
    let _guard = STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("provenance_guard");
    let store = new_store(&dir, 3).unwrap();
    let scope = scope("provenance");
    let base = ManualClaimInput {
        id: Some("claim_provenance".to_string()),
        scope,
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        claim_text: "La configuración está activa.".to_string(),
        subject: Some("configuración".to_string()),
        predicate: Some("estado".to_string()),
        object_value: Some("activa".to_string()),
        claim_kind: ClaimKind::Fact,
        polarity: ClaimPolarity::Affirmative,
        source_span_ids: vec!["missing_span".to_string()],
        source_episode_ids: vec!["missing_episode".to_string()],
        asserted_by: "extractor".to_string(),
        confidence: Some(1.0),
        observed_at_ms: 10,
        valid_from_ms: Some(10),
        valid_to_ms: None,
    };
    assert!(store.add_manual_claim(base.clone(), None).is_err());

    let direct = ManualClaimInput {
        id: Some("claim_direct_assertion".to_string()),
        source_span_ids: Vec::new(),
        source_episode_ids: Vec::new(),
        asserted_by: "user".to_string(),
        ..base
    };
    assert!(store.add_manual_claim(direct, None).is_ok());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn exact_literal_recall_is_scoped() {
    let _guard = STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("exact_literal");
    let store = new_store(&dir, 3).unwrap();
    let user_1 = scope("user_1");
    remember(
        &store,
        "ep_exact_user_1",
        user_1.clone(),
        "The failing build is in src/vector_index.rs with error E0425 and ticket X-123.",
        100,
        None,
    );
    remember(
        &store,
        "ep_exact_user_2",
        scope("user_2"),
        "Another user also mentioned ticket X-123, but this must not leak.",
        101,
        None,
    );

    let hits = store
        .keyword_search_spans(&user_1, "X-123 src/vector_index.rs", 10, 100, Some(200))
        .unwrap();
    assert_eq!(hits.len(), 1);
    let metrics = evaluate_span_hits_with_admission(
        &hits,
        &[hits[0].span.id.clone()],
        &["X-123".to_string(), "src/vector_index.rs".to_string()],
        &[],
        10,
    );
    assert!(hits[0].span.text.contains("src/vector_index.rs"));
    assert_eq!(hits[0].span.scope.user_id.as_deref(), Some("user_1"));
    assert_eq!(metrics.exact_token_miss_rate, 0.0);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn tombstoned_memory_is_not_returned_as_current() {
    let _guard = STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("tombstone");
    let store = new_store(&dir, 3).unwrap();
    let user = scope("user_1");
    let span_id = remember(
        &store,
        "ep_tombstone",
        user.clone(),
        "Temporary API key is sk-test-old. Forget this after rotation.",
        100,
        None,
    );
    let before = store
        .keyword_search_spans(&user, "sk-test-old", 10, 100, Some(101))
        .unwrap();
    assert_eq!(before.len(), 1);

    store
        .add_correction(&add_correction(
            CorrectionInput {
                id: Some("corr_tombstone".to_string()),
                scope: user.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                operation: CorrectionOperation::Tombstone,
                target_type: "span".to_string(),
                target_ids: Vec::new(),
                target_selector: Some("sk-test-old".to_string()),
                new_value: None,
                reason: Some("rotated".to_string()),
                actor: ActorKind::User,
                authority: CorrectionAuthority::User,
                effective_at_ms: Some(102),
                applies_valid_from_ms: None,
                applies_valid_to_ms: None,
                cascade_policy: None,
                metadata_json: None,
            },
            102,
        ))
        .unwrap();

    let after = store
        .keyword_search_spans(&user, "sk-test-old", 10, 100, Some(103))
        .unwrap();
    let metrics = evaluate_span_hits_with_admission(
        &after,
        &["nonexistent-current-gold".to_string()],
        &[],
        &[span_id],
        10,
    );
    assert!(after.is_empty());
    assert_eq!(metrics.stale_false_positive_rate, 0.0);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn validity_windows_support_historical_and_current_lookup() {
    let _guard = STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("validity");
    let store = new_store(&dir, 3).unwrap();
    let user = scope("user_1");
    remember(
        &store,
        "ep_old_city",
        user.clone(),
        "The user's city is Toronto.",
        100,
        Some(200),
    );
    remember(
        &store,
        "ep_new_city",
        user.clone(),
        "The user's city is Vancouver.",
        200,
        None,
    );

    let historical = store
        .keyword_search_spans(&user, "city", 10, 100, Some(150))
        .unwrap();
    assert_eq!(historical.len(), 1);
    assert!(historical[0].span.text.contains("Toronto"));

    let current = store
        .keyword_search_spans(&user, "city", 10, 100, Some(250))
        .unwrap();
    assert_eq!(current.len(), 1);
    assert!(current[0].span.text.contains("Vancouver"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn retrieval_metrics_measure_hybrid_baseline() {
    let _guard = STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("baseline");
    let store = new_store(&dir, 3).unwrap();
    let user = scope("user_1");
    let gold_span = remember(
        &store,
        "ep_gold",
        user.clone(),
        "The incident code is VECTOR-77 and the owner is the indexing team.",
        100,
        None,
    );
    remember(
        &store,
        "ep_distractor",
        user.clone(),
        "The incident owner is unknown but vector search is mentioned often.",
        101,
        None,
    );

    let vector_hits = store
        .query_spans(&user, vec![1.0, 0.0, 0.0], 10, Some(200))
        .unwrap();
    let keyword_hits = store
        .keyword_search_spans(&user, "VECTOR-77 owner", 10, 100, Some(200))
        .unwrap();
    let hybrid_hits = store
        .hybrid_search_spans(
            &user,
            HybridSpanSearch {
                query_embedding: vec![1.0, 0.0, 0.0],
                query_text: "VECTOR-77 owner",
                k: 10,
                scan_limit: 100,
                at_ms: Some(200),
            },
        )
        .unwrap()
        .fused_hits;
    let gold = [gold_span];
    let exact = ["VECTOR-77".to_string()];
    let report = evaluate_retrieval_baseline(
        &RetrievalBaselineHits {
            vector: &vector_hits,
            keyword: &keyword_hits,
            hybrid: &hybrid_hits,
        },
        &gold,
        &exact,
        &[],
        10,
    );
    assert_eq!(report.vector.recall_at_k, 0.0);
    assert!(report.keyword.recall_at_k >= 1.0);
    assert_eq!(report.keyword.exact_token_miss_rate, 0.0);
    assert!(report.hybrid.recall_at_k >= report.keyword.recall_at_k);
    assert!(report.hybrid_vs_keyword.meets_gate(0.0, 0.0));

    let _ = std::fs::remove_dir_all(&dir);
}

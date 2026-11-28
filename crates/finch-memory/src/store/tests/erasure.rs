use super::*;
use crate::ingest::EpisodeInput;
use std::collections::BTreeMap;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

fn wait_for_advisory_waiter(transaction: &mut postgres::Transaction<'_>, key: i64) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let waiting: bool = transaction
            .query_one(
                "SELECT EXISTS (SELECT 1 FROM pg_locks \
                 WHERE locktype = 'advisory' AND NOT granted \
                 AND classid::bigint = $1 AND objid::bigint = $2 AND objsubid = 1 \
                 AND pg_backend_pid() = ANY(pg_blocking_pids(pid)))",
                &[
                    &((key as u64 >> 32) as i64),
                    &((key as u64 & 0xffff_ffff) as i64),
                ],
            )
            .unwrap()
            .get(0);
        if waiting {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "write did not wait for the advisory lock"
        );
        thread::sleep(Duration::from_millis(10));
    }
}

fn user_scope(user: &str) -> MemoryScope {
    let mut scope = scope();
    scope.user_id = Some(user.to_string());
    scope
}

fn ingest_input(scope: &MemoryScope, id: &str, text: &str, at_ms: i64) -> EpisodeInput {
    EpisodeInput {
        id: Some(id.to_string()),
        scope: scope.clone(),
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        source_kind: SourceKind::UserMessage,
        actor: ActorKind::User,
        sequence_no: at_ms,
        event_time_ms: Some(at_ms),
        valid_from_ms: None,
        valid_to_ms: None,
        raw_text: text.to_string(),
        blob_ref: None,
        mime_type: None,
        causal_parent_ids: Vec::new(),
        metadata_json: None,
    }
}

fn ingest(
    store: &MemoryStore,
    scope: &MemoryScope,
    id: &str,
    text: &str,
    at_ms: i64,
) -> IngestedEpisode {
    let input = ingest_input(scope, id, text, at_ms);
    store
        .ingest_episode(input, at_ms, &ChunkOptions::default())
        .unwrap()
}

fn cited_claim(
    scope: &MemoryScope,
    id: &str,
    (subject, predicate, value): (&str, &str, &str),
    at_ms: i64,
    evidence: &IngestedEpisode,
) -> ClaimRecord {
    let mut claim = make_claim(
        scope,
        id,
        subject,
        predicate,
        Some(value),
        at_ms,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    claim.source_span_ids = evidence.spans.iter().map(|span| span.id.clone()).collect();
    claim.source_episode_ids = vec![evidence.episode.id.clone()];
    claim
}

fn city_rule(scope: &MemoryScope, id: &str, evidence: &IngestedEpisode) -> RuleInput {
    RuleInput {
        id: Some(id.to_string()),
        scope: scope.clone(),
        trigger_subject_key: "user".to_string(),
        trigger_predicate_key: "city".to_string(),
        target_subject_key: "user".to_string(),
        target_predicate_key: "region".to_string(),
        trigger_subject: Some("user".to_string()),
        trigger_predicate: Some("city".to_string()),
        target_subject: Some("user".to_string()),
        target_predicate: Some("region".to_string()),
        value_template: Some("{value} region".to_string()),
        source_span_ids: evidence.spans.iter().map(|span| span.id.clone()).collect(),
        source_episode_ids: vec![evidence.episode.id.clone()],
        ..make_derived_rule_inputs()
    }
}

fn entity(scope: &MemoryScope, name: &str, claim_id: &str) -> EntityInput {
    EntityInput {
        id: None,
        scope: scope.clone(),
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        entity_type: "place".to_string(),
        canonical_name: name.to_string(),
        aliases: Vec::new(),
        source_claim_ids: vec![claim_id.to_string()],
        merge_parent_ids: Vec::new(),
        split_from_id: None,
        confidence: None,
    }
}

fn batch(scope: &MemoryScope, at_ms: i64) -> StateMutationBatch {
    StateMutationBatch {
        extraction_job: None,
        scope: scope.clone(),
        transaction_time_ms: at_ms,
        propagation_seed: format!("erase-{at_ms}"),
        max_rule_hops: 8,
        claims: Vec::new(),
        entities: Vec::new(),
        rules: Vec::new(),
        slot_aliases: Vec::new(),
        corrections: Vec::new(),
    }
}

fn forget(scope: &MemoryScope, id: &str, selector: &str, at_ms: i64) -> CorrectionRecord {
    crate::ingest::add_correction(
        CorrectionInput {
            id: Some(id.to_string()),
            scope: scope.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            operation: CorrectionOperation::Forget,
            target_type: "claim".to_string(),
            target_ids: Vec::new(),
            target_selector: Some(selector.to_string()),
            new_value: None,
            reason: Some(format!("forget {selector}")),
            actor: ActorKind::User,
            authority: CorrectionAuthority::User,
            effective_at_ms: Some(at_ms),
            applies_valid_from_ms: None,
            applies_valid_to_ms: None,
            cascade_policy: None,
            metadata_json: None,
        },
        at_ms,
    )
}

/// Rows per table of the store's schema that `condition` matches, with `$1` bound to `value`.
fn raw_rows(path: &Path, condition: &str, value: &str) -> BTreeMap<String, i64> {
    let mut client =
        postgres::Client::connect(&crate::test_postgres_url(), postgres::NoTls).unwrap();
    let schema = crate::test_store_name(path);
    let tables = client
        .query(
            "SELECT table_name::text FROM information_schema.tables \
             WHERE table_schema = $1 AND table_name LIKE 'memory\\_%'",
            &[&schema],
        )
        .unwrap()
        .iter()
        .map(|row| row.get::<_, String>(0))
        .collect::<Vec<_>>();
    assert!(!tables.is_empty());
    tables
        .into_iter()
        .map(|table| {
            let sql = format!("SELECT count(*) FROM \"{schema}\".\"{table}\" t WHERE {condition}");
            let count = client.query_one(&sql, &[&value]).unwrap().get::<_, i64>(0);
            (table, count)
        })
        .filter(|(_, count)| *count > 0)
        .collect()
}

fn rows_with_text(path: &Path, text: &str) -> BTreeMap<String, i64> {
    raw_rows(path, "strpos(t::text, $1) > 0", text)
}

fn user_reads(store: &MemoryStore, scope: &MemoryScope) -> String {
    serde_json::to_string(&(
        store.scan_claims(scope, usize::MAX, None).unwrap(),
        store
            .scan_state_records(scope, state_scan(usize::MAX, None))
            .unwrap(),
        store.scan_entities(scope, usize::MAX).unwrap(),
        store.scan_rules(scope, usize::MAX, None).unwrap(),
        store.scan_spans(scope, usize::MAX, None).unwrap(),
    ))
    .unwrap()
}

/// Writes one user's episode, claim, rule, entity, and retraction.
fn write_user(store: &MemoryStore, scope: &MemoryScope, user: &str) {
    let evidence = ingest(
        store,
        scope,
        &format!("ep_{user}"),
        &format!("{user} lives in Porto"),
        10,
    );
    let claim = cited_claim(
        scope,
        &format!("city_{user}"),
        ("user", "city", "Porto"),
        10,
        &evidence,
    );
    let old = cited_claim(
        scope,
        &format!("pet_{user}"),
        ("user", "pet", "cat"),
        10,
        &evidence,
    );
    let mut write = batch(scope, 10);
    write.entities = vec![entity(scope, "Porto", &claim.id)];
    write.claims = vec![claim, old.clone()];
    write.rules = vec![city_rule(scope, &format!("rule_{user}"), &evidence)];
    store.apply_state_mutation_batch(write).unwrap();
    let mut retract = forget(scope, &format!("retract_{user}"), "cat", 20);
    retract.operation = CorrectionOperation::Retract;
    retract.target_selector = None;
    retract.target_ids = vec![old.id];
    store.add_correction(&retract).unwrap();
}

#[test]
fn ingestion_waits_for_an_erase_of_its_scope() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("ingestion_waits_for_erase");
    let store = crate::test_store(&path, 3).unwrap();
    let a = user_scope("user_a");
    let mut child = a.clone();
    child.thread_id = Some("thread_a".to_string());
    let key = store.session.advisory_key(scope_lock_keys(&a, true).1);

    let ingested = thread::scope(|threads| {
        let mut client =
            postgres::Client::connect(&crate::test_postgres_url(), postgres::NoTls).unwrap();
        let mut transaction = client.transaction().unwrap();
        transaction
            .execute("SELECT pg_advisory_xact_lock($1)", &[&key])
            .unwrap();
        let writer = threads.spawn(|| ingest(&store, &child, "ep_a", "A lives in Porto", 10));
        wait_for_advisory_waiter(&mut transaction, key);
        assert!(!writer.is_finished());
        assert!(raw_rows(&path, "user_id = $1", "user_a").is_empty());
        transaction.commit().unwrap();
        writer.join().unwrap()
    });

    assert_eq!(
        raw_rows(&path, "user_id = $1", "user_a"),
        BTreeMap::from([
            (EPISODES_COLLECTION.to_string(), 1),
            (SPANS_COLLECTION.to_string(), ingested.spans.len() as i64),
            (EXTRACTION_JOBS_COLLECTION.to_string(), 1),
        ])
    );
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn erase_waits_for_an_ingestion_in_flight() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("erase_waits_for_ingestion");
    let store = crate::test_store(&path, 3).unwrap();
    let a = user_scope("user_a");
    write_user(&store, &a, "user_a");
    let before = raw_rows(&path, "user_id = $1", "user_a");
    let key = store.session.advisory_key(scope_lock_keys(&a, true).1);

    thread::scope(|threads| {
        let mut client =
            postgres::Client::connect(&crate::test_postgres_url(), postgres::NoTls).unwrap();
        let mut transaction = client.transaction().unwrap();
        transaction
            .execute("SELECT pg_advisory_xact_lock_shared($1)", &[&key])
            .unwrap();
        let eraser = threads.spawn(|| store.erase_scope(&a));
        wait_for_advisory_waiter(&mut transaction, key);
        assert!(!eraser.is_finished());
        assert_eq!(raw_rows(&path, "user_id = $1", "user_a"), before);
        transaction.commit().unwrap();
        eraser.join().unwrap().unwrap();
    });

    assert!(raw_rows(&path, "user_id = $1", "user_a").is_empty());
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn bulk_write_locks_every_record_scope() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("bulk_erase_locks");
    let store = crate::test_store(&path, 3).unwrap();
    let (a, b) = (user_scope("user_a"), user_scope("user_b"));
    let mut span_a = span_record("span_a", MemoryStatus::Active, None);
    span_a.scope = a.clone();
    let mut span_b = span_record("span_b", MemoryStatus::Active, None);
    span_b.scope = b;
    let records = vec![(span_b, vec![0.0, 1.0, 0.0]), (span_a, vec![1.0, 0.0, 0.0])];
    let key = store.session.advisory_key(scope_lock_keys(&a, true).1);

    thread::scope(|threads| {
        let mut client =
            postgres::Client::connect(&crate::test_postgres_url(), postgres::NoTls).unwrap();
        let mut transaction = client.transaction().unwrap();
        transaction
            .execute("SELECT pg_advisory_xact_lock($1)", &[&key])
            .unwrap();
        let writer = threads.spawn(|| store.append_vector_spans(&records));
        wait_for_advisory_waiter(&mut transaction, key);
        assert!(!writer.is_finished());
        assert!(raw_rows(&path, "space_id = $1", "default").is_empty());
        transaction.commit().unwrap();
        writer.join().unwrap().unwrap();
    });

    assert_eq!(
        raw_rows(&path, "space_id = $1", "default"),
        BTreeMap::from([(SPANS_COLLECTION.to_string(), 2),])
    );
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn sibling_and_state_writes_do_not_wait_for_ingestion_locks() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("independent_ingestion_locks");
    let store = crate::test_store(&path, 3).unwrap();
    let (a, b) = (user_scope("user_a"), user_scope("user_b"));
    let mut claim = make_claim(
        &a,
        "claim_a",
        "user",
        "city",
        Some("Porto"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    claim.source_span_ids.clear();
    claim.source_episode_ids.clear();
    let key = store.session.advisory_key(scope_lock_keys(&a, true).1);

    thread::scope(|threads| {
        let mut client =
            postgres::Client::connect(&crate::test_postgres_url(), postgres::NoTls).unwrap();
        let mut transaction = client.transaction().unwrap();
        transaction
            .execute("SELECT pg_advisory_xact_lock($1)", &[&key])
            .unwrap();
        let blocked = threads.spawn(|| ingest(&store, &a, "ep_a", "A lives in Porto", 10));
        wait_for_advisory_waiter(&mut transaction, key);
        let (sent, done) = mpsc::channel();
        let sibling_done = sent.clone();
        let store = &store;
        threads.spawn(move || {
            ingest(store, &b, "ep_b", "B lives in Lisbon", 10);
            sibling_done.send(()).unwrap();
        });
        threads.spawn(move || {
            store.append_claim(&claim, None).unwrap();
            sent.send(()).unwrap();
        });
        for _ in 0..2 {
            done.recv_timeout(Duration::from_secs(10))
                .expect("sibling and state writes must finish while the erase lock is held");
        }
        assert!(!blocked.is_finished());
        transaction.commit().unwrap();
        blocked.join().unwrap();
    });

    assert_eq!(
        store
            .claims_by_ids(&a, &["claim_a".to_string()])
            .unwrap()
            .len(),
        1
    );
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn batch_rejects_evidence_erased_before_it_takes_the_lock() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("batch_evidence_under_lock");
    let store = crate::test_store(&path, 3).unwrap();
    let a = user_scope("user_a");
    let evidence = ingest(&store, &a, "ep_a", "A lives in Porto", 10);
    let mut write = batch(&a, 20);
    write.claims = vec![cited_claim(
        &a,
        "claim_a",
        ("user", "city", "Porto"),
        10,
        &evidence,
    )];
    let key = store.session.advisory_key(scope_lock_keys(&a, false).1);
    let schema = crate::test_store_name(&path);

    let result = thread::scope(|threads| {
        let mut client =
            postgres::Client::connect(&crate::test_postgres_url(), postgres::NoTls).unwrap();
        let mut transaction = client.transaction().unwrap();
        transaction
            .execute("SELECT pg_advisory_xact_lock($1)", &[&key])
            .unwrap();
        let store = &store;
        let writer = threads.spawn(move || store.apply_state_mutation_batch(write));
        wait_for_advisory_waiter(&mut transaction, key);
        transaction
            .execute(
                &format!("DELETE FROM \"{schema}\".memory_spans WHERE source_id = $1"),
                &[&evidence.episode.id],
            )
            .unwrap();
        transaction
            .execute(
                &format!("DELETE FROM \"{schema}\".memory_episodes WHERE id = $1"),
                &[&evidence.episode.id],
            )
            .unwrap();
        transaction.commit().unwrap();
        writer.join().unwrap()
    });

    let error = result.expect_err("batch accepted evidence deleted before its write lock");
    assert_eq!(error.code, finch_types::StatusCode::InvalidArgument);
    assert_eq!(
        error.message,
        "evidence spans must exist in the requested scope"
    );
    assert_eq!(
        raw_rows(&path, "user_id = $1", "user_a"),
        BTreeMap::from([(EXTRACTION_JOBS_COLLECTION.to_string(), 1),])
    );
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn erase_scope_deletes_every_row_of_the_user_and_keeps_other_scopes() {
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("erase_scope");
    let store = crate::test_store(&path, 3).unwrap();
    let (a, b) = (user_scope("user_a"), user_scope("user_b"));
    write_user(&store, &a, "user_a");
    write_user(&store, &b, "user_b");
    let mut thread = a.clone();
    thread.thread_id = Some("thread_1".to_string());
    write_user(&store, &thread, "user_a_thread");
    let space_claim = make_claim(
        &scope(),
        "space_claim",
        "office",
        "city",
        Some("Porto"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    let mut space_claim = space_claim;
    space_claim.source_span_ids.clear();
    space_claim.source_episode_ids.clear();
    store.append_claim(&space_claim, None).unwrap();
    let b_before = user_reads(&store, &b);
    let b_rows = raw_rows(&path, "user_id = $1", "user_b");
    let a_rows = raw_rows(&path, "user_id = $1", "user_a");

    let report = store.erase_scope(&a).unwrap();
    drop(store);
    let store = crate::reopen_test_store(&path).unwrap();
    let a_after = raw_rows(&path, "user_id = $1", "user_a");
    let b_after = user_reads(&store, &b);
    let b_rows_after = raw_rows(&path, "user_id = $1", "user_b");
    let space_left = store
        .claims_by_ids(&scope(), &["space_claim".to_string()])
        .unwrap();
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);

    assert!(a_rows.len() > 5, "user A wrote too few tables: {a_rows:?}");
    assert!(a_after.is_empty(), "rows of user A remain: {a_after:?}");
    assert_eq!(
        report.deleted_rows.values().sum::<usize>() as i64,
        a_rows.values().sum::<i64>()
    );
    assert_eq!(b_after, b_before);
    assert_eq!(b_rows_after, b_rows);
    assert_eq!(space_left.len(), 1);
}

#[test]
fn erase_episodes_deletes_their_text_and_derived_state_and_keeps_other_support() {
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("erase_episodes");
    let store = crate::test_store(&path, 3).unwrap();
    let scope = user_scope("user_e");
    let erased_text = "I moved to Zanzibarville and Lisbon stays my favourite city.";
    let erased = ingest(&store, &scope, "ep_erased", erased_text, 10);
    let kept = ingest(
        &store,
        &scope,
        "ep_kept",
        "Lisbon is still my favourite city.",
        20,
    );
    let city = cited_claim(
        &scope,
        "city",
        ("user", "city", "Zanzibarville"),
        10,
        &erased,
    );
    let first_favourite = cited_claim(
        &scope,
        "favourite_1",
        ("user", "favourite city", "Lisbon"),
        10,
        &erased,
    );
    let mut write = batch(&scope, 10);
    write.entities = vec![entity(&scope, "Zanzibarville", &city.id)];
    write.claims = vec![city, first_favourite];
    write.rules = vec![city_rule(&scope, "rule_region", &kept)];
    store.apply_state_mutation_batch(write).unwrap();
    let mut write = batch(&scope, 20);
    write.claims = vec![cited_claim(
        &scope,
        "favourite_2",
        ("user", "favourite city", "Lisbon"),
        20,
        &kept,
    )];
    store.apply_state_mutation_batch(write).unwrap();
    let before = rows_with_text(&path, "Zanzibarville");

    let report = store
        .erase_episodes(&scope, &["ep_erased".to_string()])
        .unwrap();
    drop(store);
    let store = crate::reopen_test_store(&path).unwrap();
    let after = rows_with_text(&path, "Zanzibarville");
    let text_after = rows_with_text(&path, erased_text);
    let current = store.scan_current_claims(&scope, usize::MAX, None).unwrap();
    let states = store
        .scan_state_records(&scope, state_scan(usize::MAX, None))
        .unwrap();
    let projection = store
        .project_answer_ready_state(
            &scope,
            &answer_request(
                "erase",
                "Which city does the user live in?",
                &[],
                20,
                20,
                None,
            ),
        )
        .unwrap();
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);

    assert!(
        before.contains_key("memory_state_records") && before.contains_key("memory_entities"),
        "the value never reached derived state: {before:?}"
    );
    assert!(after.is_empty(), "the erased value remains: {after:?}");
    assert!(
        text_after.is_empty(),
        "the erased text remains: {text_after:?}"
    );
    assert_eq!(report.deleted_rows["memory_episodes"], 1);
    assert!(report.deleted_rows["memory_claims"] >= 2);
    let current_values = current
        .iter()
        .map(|claim| (claim.id.as_str(), claim.object_value.as_deref()))
        .collect::<Vec<_>>();
    assert_eq!(current_values, vec![("favourite_2", Some("Lisbon"))]);
    assert!(states
        .iter()
        .any(|state| state.claim_ids == vec!["favourite_2".to_string()]));
    assert!(!serde_json::to_string(&projection)
        .unwrap()
        .contains("Zanzibarville"));
}

#[test]
fn purge_deletes_the_forgotten_value_and_keeps_the_correction() {
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("forget_deletes");
    let store = crate::test_store(&path, 3).unwrap();
    let scope = user_scope("user_f");
    let evidence = ingest(&store, &scope, "ep_rule", "Regions follow the city.", 5);
    let request = ingest(
        &store,
        &scope,
        "ep_request",
        "Please forget where I live.",
        30,
    );
    let mut write = batch(&scope, 5);
    write.rules = vec![city_rule(&scope, "rule_region", &evidence)];
    store.apply_state_mutation_batch(write).unwrap();
    for (id, subject, value, at) in [
        ("city", "user", "Quixadaville", 10),
        ("phone", "user", "555-0199-private", 10),
    ] {
        let predicate = if id == "city" { "city" } else { "phone" };
        let mut claim = make_claim(
            &scope,
            id,
            subject,
            predicate,
            Some(value),
            at,
            (ClaimKind::Fact, ClaimPolarity::Affirmative),
        );
        claim.source_span_ids.clear();
        claim.source_episode_ids.clear();
        store.append_claim(&claim, None).unwrap();
    }
    let before = rows_with_text(&path, "Quixadaville");

    // A selector would also match the derived region claim, so the forget names its target.
    let mut by_id = forget(&scope, "forget_city", "Quixadaville", 20);
    by_id.target_selector = None;
    by_id.target_ids = vec!["city".to_string()];
    store.add_correction(&by_id).unwrap();
    let mut correction = forget(&scope, "forget_phone", "555-0199-private", 30);
    correction.source_span_ids = request.spans.iter().map(|span| span.id.clone()).collect();
    correction.source_episode_ids = vec![request.episode.id.clone()];
    let mut write = batch(&scope, 30);
    write.corrections = vec![correction];
    store.apply_state_mutation_batch(write).unwrap();
    store.purge_forgotten(&scope, i64::MAX).unwrap();
    drop(store);
    let store = crate::reopen_test_store(&path).unwrap();
    let city_after = rows_with_text(&path, "Quixadaville");
    let phone_after = rows_with_text(&path, "555-0199-private");
    let corrections = store.scan_corrections(&scope, usize::MAX).unwrap();
    let current = store.scan_current_claims(&scope, usize::MAX, None).unwrap();
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);

    assert!(
        before.contains_key("memory_state_records"),
        "the value never reached state: {before:?}"
    );
    assert!(
        city_after.is_empty(),
        "the forgotten city remains: {city_after:?}"
    );
    assert!(
        phone_after.is_empty(),
        "the forgotten phone remains: {phone_after:?}"
    );
    // Each forget keeps its id and targets only the forget id of its slot.
    let mut kept = corrections
        .iter()
        .map(|correction| {
            let forget_ids = correction
                .target_ids
                .iter()
                .all(|id| id.starts_with("forget_"));
            (
                correction.id.as_str(),
                correction.target_ids.len(),
                forget_ids,
            )
        })
        .collect::<Vec<_>>();
    kept.sort();
    assert_eq!(
        kept,
        vec![("forget_city", 1, true), ("forget_phone", 1, true)]
    );
    assert!(corrections
        .iter()
        .all(|correction| correction.target_selector.is_none()));
    assert!(
        current.iter().all(|claim| claim.object_value.is_none()),
        "claims with a value remain: {current:?}"
    );
}

#[test]
fn forget_leaves_a_tombstone_until_a_new_claim() {
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("forget_tombstone");
    let store = crate::test_store(&path, 3).unwrap();
    let scope = user_scope("user_t");
    let city = |id: &str, value: &str, at: i64| {
        let mut claim = make_claim(
            &scope,
            id,
            "user",
            "city",
            Some(value),
            at,
            (ClaimKind::Fact, ClaimPolarity::Affirmative),
        );
        claim.source_span_ids.clear();
        claim.source_episode_ids.clear();
        claim
    };
    store
        .append_claim(&city("boston", "Boston", 10), None)
        .unwrap();
    store
        .append_claim(&city("denver", "Denver", 20), None)
        .unwrap();
    store
        .add_correction(&forget(&scope, "forget_denver", "Denver", 30))
        .unwrap();
    store.purge_forgotten(&scope, i64::MAX).unwrap();
    let states_31 = store
        .scan_state_records(&scope, state_scan(usize::MAX, Some(31)))
        .unwrap();
    let current_31 = store
        .scan_current_claims(&scope, usize::MAX, Some(31))
        .unwrap();
    let denver_rows = rows_with_text(&path, "Denver");
    let boston_rows = rows_with_text(&path, "Boston");
    store
        .append_claim(&city("seattle", "Seattle", 40), None)
        .unwrap();
    let states_41 = store
        .scan_state_records(&scope, state_scan(usize::MAX, Some(41)))
        .unwrap();
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);

    let kinds = |states: &[StateRecord]| {
        states
            .iter()
            .map(|state| (state.state_kind, state.object_value.clone()))
            .collect::<Vec<_>>()
    };
    assert_eq!(kinds(&states_31), vec![(StateRecordKind::Tombstone, None)]);
    assert!(
        denver_rows.is_empty(),
        "the forgotten value remains: {denver_rows:?}"
    );
    assert!(boston_rows.contains_key("memory_claims"));
    assert!(!current_31.iter().any(|claim| {
        claim.status == MemoryStatus::Active && claim.object_value.as_deref() == Some("Boston")
    }));
    assert_eq!(
        kinds(&states_41),
        vec![(StateRecordKind::Current, Some("Seattle".to_string()))]
    );
}

#[test]
fn forgetting_one_set_member_keeps_the_other_members() {
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("forget_set_member");
    let store = crate::test_store(&path, 3).unwrap();
    let scope = user_scope("user_s");
    let allergy = |id: &str, value: &str, at: i64| {
        let mut claim = make_claim(
            &scope,
            id,
            "user",
            "allergy",
            Some(value),
            at,
            (ClaimKind::Preference, ClaimPolarity::Affirmative),
        );
        claim.source_span_ids.clear();
        claim.source_episode_ids.clear();
        claim
    };
    store
        .append_claim(&allergy("peanuts", "peanuts", 10), None)
        .unwrap();
    store
        .append_claim(&allergy("shellfish", "shellfish", 20), None)
        .unwrap();
    store
        .add_correction(&forget(&scope, "forget_peanuts", "peanuts", 30))
        .unwrap();
    store.purge_forgotten(&scope, i64::MAX).unwrap();
    let current = store
        .scan_current_claims(&scope, usize::MAX, Some(31))
        .unwrap();
    let peanut_rows = rows_with_text(&path, "peanuts");
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);

    assert!(
        peanut_rows.is_empty(),
        "the forgotten value remains: {peanut_rows:?}"
    );
    let values = current
        .iter()
        .filter(|claim| claim.status == MemoryStatus::Active)
        .filter_map(|claim| claim.object_value.clone())
        .collect::<Vec<_>>();
    assert_eq!(values, vec!["shellfish".to_string()]);
}

#[test]
fn forget_masks_the_value_in_search_results_and_purge_redacts_the_span() {
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("forget_masks_raw_text");
    let store = crate::test_store(&path, 3).unwrap();
    let scope = user_scope("user_m");
    let said = ingest(
        &store,
        &scope,
        "ep_said",
        "I go to Marlo's Kitchen on Fridays.",
        10,
    );
    let mut claim = make_claim(
        &scope,
        "claim_restaurant",
        "user",
        "favorite restaurant",
        Some("Marlo's Kitchen"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    claim.source_span_ids = said.spans.iter().map(|span| span.id.clone()).collect();
    claim.source_episode_ids = vec![said.episode.id.clone()];
    store.append_claim(&claim, None).unwrap();
    store
        .add_correction(&forget(&scope, "forget_restaurant", "Marlo's Kitchen", 20))
        .unwrap();

    let hidden = store
        .keyword_search_spans(&scope, "Fridays", 10, 1000, None)
        .unwrap();
    let stored_before = rows_with_text(&path, "Marlo");
    let report = store.purge_forgotten(&scope, i64::MAX).unwrap();
    let stored_after = rows_with_text(&path, "Marlo");
    let found_after = store
        .keyword_search_spans(&scope, "Marlo", 10, 1000, None)
        .unwrap();
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);

    assert_eq!(hidden.len(), 1, "{hidden:?}");
    assert_eq!(hidden[0].span.text, "I go to [forgotten] on Fridays.");
    assert!(
        stored_before.contains_key("memory_spans"),
        "{stored_before:?}"
    );
    assert!(
        stored_after.is_empty(),
        "the value remains: {stored_after:?}"
    );
    assert_eq!(
        report.redacted_span_ids,
        said.spans
            .iter()
            .map(|span| span.id.clone())
            .collect::<Vec<_>>()
    );
    assert!(found_after.is_empty(), "{found_after:?}");
}

#[test]
fn forget_bound_to_a_slot_masks_the_slot_values_in_search_results() {
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("forget_slot_masks_raw_text");
    let store = crate::test_store(&path, 3).unwrap();
    let scope = user_scope("user_s");
    let said = ingest(
        &store,
        &scope,
        "ep_said",
        "I go to Marlo's Kitchen on Fridays.",
        10,
    );
    let mut claim = make_claim(
        &scope,
        "claim_restaurant",
        "user",
        "favorite restaurant",
        Some("Marlo's Kitchen"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    claim.source_span_ids = said.spans.iter().map(|span| span.id.clone()).collect();
    claim.source_episode_ids = vec![said.episode.id.clone()];
    store.append_claim(&claim, None).unwrap();
    let slot_id = store
        .claims_by_ids(&scope, &["claim_restaurant".to_string()])
        .unwrap()[0]
        .slot_id
        .clone()
        .unwrap();
    // The selector names the topic, not the stored value, so the forget resolves no claim ids.
    let mut correction = forget(&scope, "forget_restaurant", "where I eat on Fridays", 20);
    correction.target_match = CorrectionTargetMatch::CanonicalSlot;
    correction.target_slot_id = Some(slot_id);
    store.add_correction(&correction).unwrap();

    let hidden = store
        .keyword_search_spans(&scope, "Fridays", 10, 1000, None)
        .unwrap();
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);

    assert_eq!(hidden.len(), 1, "{hidden:?}");
    assert_eq!(hidden[0].span.text, "I go to [forgotten] on Fridays.");
}

#[test]
fn forget_bound_to_a_slot_with_several_values_masks_only_the_named_value() {
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("forget_slot_masks_named_value");
    let store = crate::test_store(&path, 3).unwrap();
    let scope = user_scope("user_n");
    let said = ingest(
        &store,
        &scope,
        "ep_said",
        "I like Marlo's Kitchen and Marrow House.",
        10,
    );
    for (id, value) in [
        ("claim_marlo", "Marlo's Kitchen"),
        ("claim_marrow", "Marrow House"),
    ] {
        let mut claim = make_claim(
            &scope,
            id,
            "user",
            "liked restaurant",
            Some(value),
            10,
            (ClaimKind::Preference, ClaimPolarity::Affirmative),
        );
        claim.source_span_ids = said.spans.iter().map(|span| span.id.clone()).collect();
        claim.source_episode_ids = vec![said.episode.id.clone()];
        store.append_claim(&claim, None).unwrap();
    }
    let slot_id = store
        .claims_by_ids(&scope, &["claim_marlo".to_string()])
        .unwrap()[0]
        .slot_id
        .clone()
        .unwrap();
    let mut correction = forget(&scope, "forget_marlo", "Marlo's Kitchen", 20);
    correction.target_match = CorrectionTargetMatch::CanonicalSlot;
    correction.target_slot_id = Some(slot_id);
    correction.target_ids = Vec::new();
    store.add_correction(&correction).unwrap();

    let hidden = store
        .keyword_search_spans(&scope, "like", 10, 1000, None)
        .unwrap();
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);

    assert_eq!(hidden.len(), 1, "{hidden:?}");
    assert_eq!(hidden[0].span.text, "I like [forgotten] and Marrow House.");
}

#[test]
fn masking_replaces_whole_words_in_any_case() {
    let values = BTreeSet::from(["keto".to_string(), "marlo's kitchen".to_string()]);
    assert_eq!(
        crate::store::erasure::mask_values("Keto, not ketones, at MARLO'S KITCHEN.", &values),
        "[forgotten], not ketones, at [forgotten]."
    );
}

/// Ingests the text of `said` (episode id, text) and returns a claim (id, predicate, value)
/// that cites its spans and episode.
fn stated_claim(
    store: &MemoryStore,
    scope: &MemoryScope,
    said: (&str, &str),
    claim: (&str, &str, &str),
    at_ms: i64,
) -> ClaimRecord {
    let ingested = ingest(store, scope, said.0, said.1, at_ms);
    let mut record = make_claim(
        scope,
        claim.0,
        "user",
        claim.1,
        Some(claim.2),
        at_ms,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    record.source_span_ids = ingested.spans.iter().map(|span| span.id.clone()).collect();
    record.source_episode_ids = vec![ingested.episode.id.clone()];
    store.append_claim(&record, None).unwrap();
    record
}

/// The texts of the spans that a keyword search for each of `queries` returns, in order.
fn searched_texts(store: &MemoryStore, scope: &MemoryScope, queries: &[&str]) -> Vec<String> {
    let mut texts = BTreeSet::new();
    for query in queries {
        let hits = store
            .keyword_search_spans(scope, query, 10, 1000, None)
            .unwrap();
        texts.extend(hits.into_iter().map(|hit| hit.span.text));
    }
    texts.into_iter().collect()
}

#[test]
fn forgotten_value_stays_out_of_the_context_packed_with_another_slot() {
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("forget_context_history");
    let store = crate::test_store(&path, 3).unwrap();
    let scope = user_scope("user_h");
    let home = ("claim_home", "home city", "Quixadaville");
    stated_claim(
        &store,
        &scope,
        ("ep_home", "I live in Quixadaville."),
        home,
        10,
    );
    let color = ("claim_color", "favorite color", "teal");
    stated_claim(
        &store,
        &scope,
        ("ep_color", "My favorite color is teal."),
        color,
        12,
    );
    store
        .add_correction(&forget(&scope, "forget_home", "Quixadaville", 20))
        .unwrap();
    let selections = store
        .claims_by_ids(
            &scope,
            &["claim_home".to_string(), "claim_color".to_string()],
        )
        .unwrap()
        .iter()
        .filter_map(|claim| claim.slot_id.clone())
        .map(|slot_id| serde_json::json!({"slot_id": slot_id, "view": "current"}))
        .collect::<Vec<_>>();
    let request = serde_json::json!({
        "scope": scope,
        "query_text": "what do you know about me?",
        "selections": selections,
        "profile_limit": 0,
        "claim_limit": 50,
        "artifact_limit": 0,
        "token_budget": 4096,
        "include_provenance": true,
    });
    let packed = store
        .build_answer_ready_context_json(&request.to_string())
        .unwrap();
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);

    assert!(packed.contains("teal"), "{packed}");
    assert!(
        !packed.to_lowercase().contains("quixadaville"),
        "the forgotten value is in the context: {packed}"
    );
}

#[test]
fn forget_masks_a_short_value_only_in_the_text_that_states_it() {
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("forget_short_value");
    let store = crate::test_store(&path, 3).unwrap();
    let scope = user_scope("user_short");
    let locker = ("claim_locker", "locker number", "42");
    stated_claim(
        &store,
        &scope,
        ("ep_locker", "My locker number is 42."),
        locker,
        10,
    );
    ingest(&store, &scope, "ep_bus", "Bus 42 leaves at noon.", 11);
    store
        .add_correction(&forget(&scope, "forget_locker", "42", 20))
        .unwrap();
    let masked = searched_texts(&store, &scope, &["locker", "Bus"]);
    let report = store.purge_forgotten(&scope, i64::MAX).unwrap();
    let purged = searched_texts(&store, &scope, &["locker", "Bus"]);
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);

    let want = vec![
        "Bus 42 leaves at noon.".to_string(),
        "My locker number is [forgotten].".to_string(),
    ];
    assert_eq!(masked, want);
    assert_eq!(purged, want);
    assert_eq!(report.redacted_span_ids.len(), 1, "{report:?}");
}

#[test]
fn forget_bound_to_a_slot_that_names_no_value_masks_every_value_of_the_slot() {
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("forget_slot_masks_history");
    let store = crate::test_store(&path, 3).unwrap();
    let scope = user_scope("user_two");
    let first = ("claim_first", "home city", "Quixadaville");
    stated_claim(
        &store,
        &scope,
        ("ep_first", "I live in Quixadaville."),
        first,
        10,
    );
    let second = ("claim_second", "home city", "Marrowport");
    let moved = stated_claim(
        &store,
        &scope,
        ("ep_second", "I moved to Marrowport."),
        second,
        15,
    );
    let slot_id = store
        .claims_by_ids(&scope, &[moved.id])
        .unwrap()
        .remove(0)
        .slot_id;
    let mut correction = forget(&scope, "forget_where", "where I live", 20);
    correction.target_match = CorrectionTargetMatch::CanonicalSlot;
    correction.target_slot_id = slot_id;
    store.add_correction(&correction).unwrap();
    let masked = searched_texts(&store, &scope, &["live", "moved"]);
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);

    assert_eq!(
        masked,
        vec![
            "I live in [forgotten].".to_string(),
            "I moved to [forgotten].".to_string(),
        ]
    );
}

#[test]
fn purge_redacts_every_copy_of_the_value_and_a_second_purge_changes_nothing() {
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("purge_uncited_copy");
    let store = crate::test_store(&path, 3).unwrap();
    let scope = user_scope("user_copy");
    let home = ("claim_home", "home city", "Quixadaville");
    stated_claim(
        &store,
        &scope,
        ("ep_home", "I live in Quixadaville."),
        home,
        10,
    );
    // No claim cites the reply, so only a redaction of the whole scope reaches it.
    ingest(
        &store,
        &scope,
        "ep_reply",
        "Got it, Quixadaville is noted.",
        11,
    );
    store
        .add_correction(&forget(&scope, "forget_home", "Quixadaville", 20))
        .unwrap();
    let masked = searched_texts(&store, &scope, &["live", "noted"]);
    let first = store.purge_forgotten(&scope, i64::MAX).unwrap();
    let purged = searched_texts(&store, &scope, &["live", "noted"]);
    let stored = rows_with_text(&path, "Quixadaville");
    let second = store.purge_forgotten(&scope, i64::MAX).unwrap();
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);

    let want = vec![
        "Got it, [forgotten] is noted.".to_string(),
        "I live in [forgotten].".to_string(),
    ];
    assert_eq!(masked, want);
    assert_eq!(purged, want);
    assert!(stored.is_empty(), "the value remains: {stored:?}");
    assert_eq!(first.redacted_span_ids.len(), 2, "{first:?}");
    assert_eq!(second, crate::store::erasure::PurgeReport::default());
}

#[test]
fn erase_scope_reads_a_backslash_before_a_double_quote_in_an_id_literally() {
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("erase_backslash_quote");
    let store = crate::test_store(&path, 3).unwrap();
    let mut erased = scope();
    erased.thread_id = Some("x\\\"y".to_string());
    let mut kept = scope();
    kept.thread_id = Some("x\"y".to_string());
    ingest(&store, &erased, "ep_erased", "alpha note", 10);
    ingest(&store, &kept, "ep_kept", "bravo note", 10);
    let own_before = searched_texts(&store, &erased, &["note"]);
    store.erase_scope(&erased).unwrap();
    let left = searched_texts(&store, &scope(), &["note"]);
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);

    assert_eq!(own_before, vec!["alpha note".to_string()]);
    assert_eq!(left, vec!["bravo note".to_string()]);
}

#[test]
fn source_text_masks_a_forgotten_value_after_it_joins_overlapping_spans() {
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("forget_source_text");
    let index_dir = temp_dir("forget_source_text_index");
    let store = crate::test_store(&path, 3).unwrap();
    let scope = user_scope("user_doc");
    let text = (0..400)
        .map(|n| format!("Line {n} names Quixadaville as stop {n}."))
        .collect::<Vec<_>>()
        .join("\n");
    let ingested = ingest(&store, &scope, "ep_doc", &text, 10);
    let span_ids = ingested
        .spans
        .iter()
        .map(|span| span.id.clone())
        .collect::<Vec<_>>();
    let mut claim = make_claim(
        &scope,
        "claim_stop",
        "user",
        "stop",
        Some("Quixadaville"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    claim.source_span_ids = span_ids.clone();
    claim.source_episode_ids = vec![ingested.episode.id.clone()];
    store.append_claim(&claim, None).unwrap();
    store
        .add_correction(&forget(&scope, "forget_stop", "Quixadaville", 20))
        .unwrap();
    let mut builder = crate::source_index::SourceLexicalIndexBuilder::new();
    builder
        .add_source(crate::source_index::IndexedSource {
            source_id: "ep_doc",
            title: "Stops",
            directory: "docs",
            body: &text,
            date_ms: None,
            span_ids: span_ids.clone(),
        })
        .unwrap();
    builder.write(&index_dir).unwrap();
    let index = crate::source_index::SourceLexicalIndex::open(&index_dir).unwrap();
    let texts = store
        .source_texts(&scope, &index, &["ep_doc".to_string()], None)
        .unwrap();
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
    std::fs::remove_dir_all(index_dir).unwrap();
    drop(guard);

    assert!(span_ids.len() > 1, "the text must span several chunks");
    assert_eq!(texts["ep_doc"], text.replace("Quixadaville", "[forgotten]"));
}

#[test]
fn forgetting_one_preference_keeps_the_slot_supported_by_its_other_members() {
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("forget_preference_member");
    let store = EvidencedStore::create(&path, 3).unwrap();
    let scope = user_scope("user_pref");
    for (id, value, at) in [
        ("claim_peanuts", "peanuts", 10),
        ("claim_shellfish", "shellfish", 20),
    ] {
        let claim = make_claim(
            &scope,
            id,
            "user",
            "allergy",
            Some(value),
            at,
            (ClaimKind::Preference, ClaimPolarity::Affirmative),
        );
        store.append_claim(&claim, None).unwrap();
    }
    store
        .add_correction(&forget(&scope, "forget_peanuts", "peanuts", 30))
        .unwrap();
    let slot_id = store
        .claims_by_ids(&scope, &["claim_shellfish".to_string()])
        .unwrap()
        .remove(0)
        .slot_id
        .unwrap();
    let projection = store
        .project_answer_ready_state(
            &scope,
            &AnswerReadyStateRequest {
                selections: &StateReadSelection::answer_targets(&[slot_id]),
                ..answer_request("forget-preference", "allergies", &[], 20, 0, Some(40))
            },
        )
        .unwrap();
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);

    assert_eq!(projection.support.state, AnswerSupportState::Supported);
    let members = projection
        .set_states
        .iter()
        .flat_map(|state| state.members.clone())
        .collect::<Vec<_>>();
    assert_eq!(members, ["shellfish"]);
}

/// Runs `statement` on the stored span, with `{spans}` as its table and `$1` as its id.
fn span_sql(path: &Path, statement: &str, span_id: &str) -> Vec<postgres::Row> {
    let mut client =
        postgres::Client::connect(&crate::test_postgres_url(), postgres::NoTls).unwrap();
    let table = format!("\"{}\".memory_spans", crate::test_store_name(path));
    client
        .query(&statement.replace("{spans}", &table), &[&span_id])
        .unwrap()
}

#[test]
fn purge_at_a_parent_scope_redacts_only_the_scope_of_each_forget() {
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("purge_parent_scope");
    let store = crate::test_store(&path, 3).unwrap();
    let (a, b) = (user_scope("user_a"), user_scope("user_b"));
    let home = ("claim_home", "home city", "Quixadaville");
    stated_claim(&store, &a, ("ep_a", "I live in Quixadaville."), home, 10);
    let cousin = "My cousin lives in Quixadaville.";
    let other = ingest(&store, &b, "ep_b", cousin, 11);
    let b_span = other.spans[0].id.as_str();
    span_sql(
        &path,
        "UPDATE {spans} SET embedding = '[0,1,0]' WHERE pk = $1",
        b_span,
    );
    store
        .add_correction(&forget(&a, "forget_home", "Quixadaville", 20))
        .unwrap();
    let report = store.purge_forgotten(&scope(), i64::MAX).unwrap();
    let a_texts = searched_texts(&store, &a, &["live"]);
    let b_texts = searched_texts(&store, &b, &["cousin"]);
    let b_rows = rows_with_text(&path, cousin);
    let b_embedded = span_sql(
        &path,
        "SELECT embedding IS NOT NULL FROM {spans} WHERE pk = $1",
        b_span,
    )[0]
    .get::<_, bool>(0);
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);

    assert_eq!(a_texts, vec!["I live in [forgotten].".to_string()]);
    assert_eq!(b_texts, vec![cousin.to_string()]);
    assert_eq!(
        b_rows,
        BTreeMap::from([
            (EPISODES_COLLECTION.to_string(), 1),
            (SPANS_COLLECTION.to_string(), 1),
        ])
    );
    assert!(b_embedded, "the purge cleared the embedding of user B");
    assert_eq!(report.redacted_span_ids.len(), 1, "{report:?}");
}

/// The start time and the values of each version of the slot's history at `at_ms` that holds a
/// value or a set member, oldest first.
fn history_versions(
    store: &MemoryStore,
    scope: &MemoryScope,
    slot_id: &MemoryId,
    at_ms: i64,
) -> Vec<(Option<i64>, Vec<String>)> {
    let selections = [StateReadSelection {
        slot_id: slot_id.clone(),
        view: StateReadView::Timeline,
        role: StateReadRole::AnswerTarget,
    }];
    let projection = store
        .project_answer_ready_state(
            scope,
            &AnswerReadyStateRequest {
                selections: &selections,
                ..answer_request("history", "history", &[], 20, 0, Some(at_ms))
            },
        )
        .unwrap();
    projection
        .slot_histories
        .iter()
        .flat_map(|history| &history.versions)
        .map(|version| {
            let values = version.object_value.iter().chain(&version.members);
            (version.valid_from_ms, values.cloned().collect::<Vec<_>>())
        })
        .filter(|(_, values)| !values.is_empty())
        .collect()
}

/// Writes one preference claim per (id, predicate, value) and returns the slot id of each.
fn preferences(
    store: &EvidencedStore,
    scope: &MemoryScope,
    claims: &[(&str, &str, &str)],
) -> Vec<MemoryId> {
    for (at_ms, (id, predicate, value)) in claims.iter().enumerate() {
        let kind = (ClaimKind::Preference, ClaimPolarity::Affirmative);
        let at_ms = 10 + at_ms as i64;
        let claim = make_claim(scope, id, "user", predicate, Some(value), at_ms, kind);
        store.append_claim(&claim, None).unwrap();
    }
    let ids = claims
        .iter()
        .map(|(id, _, _)| id.to_string())
        .collect::<Vec<_>>();
    let stored = store.claims_by_ids(scope, &ids).unwrap();
    ids.iter()
        .map(|id| {
            let claim = stored.iter().find(|claim| &claim.id == id).unwrap();
            claim.slot_id.clone().unwrap()
        })
        .collect()
}

#[test]
fn forget_bound_to_a_slot_hides_only_the_member_its_selector_names_in_the_history() {
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("forget_slot_history_member");
    let store = EvidencedStore::create(&path, 3).unwrap();
    let scope = user_scope("user_hist");
    let slot_ids = preferences(
        &store,
        &scope,
        &[
            ("claim_marlo", "liked restaurant", "Marlo's Kitchen"),
            ("claim_marrow", "liked restaurant", "Marrow House"),
        ],
    );
    let before = history_versions(&store, &scope, &slot_ids[0], 40);
    let mut correction = forget(&scope, "forget_marlo", "Marlo's Kitchen", 20);
    correction.target_match = CorrectionTargetMatch::CanonicalSlot;
    correction.target_slot_id = Some(slot_ids[0].clone());
    store.add_correction(&correction).unwrap();
    let after = history_versions(&store, &scope, &slot_ids[0], 40);
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);

    let versions = |versions: &[(i64, &[&str])]| {
        versions
            .iter()
            .map(|(at_ms, values)| (Some(*at_ms), values.iter().map(|v| v.to_string()).collect()))
            .collect::<Vec<(Option<i64>, Vec<String>)>>()
    };
    assert_eq!(
        before,
        versions(&[
            (10, &["Marlo's Kitchen"]),
            (11, &["Marlo's Kitchen", "Marrow House"])
        ])
    );
    assert_eq!(after, versions(&[(11, &["Marrow House"])]));
}

#[test]
fn forgetting_a_value_in_one_slot_keeps_that_value_in_the_history_of_another_set_slot() {
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("forget_history_other_slot");
    let store = EvidencedStore::create(&path, 3).unwrap();
    let scope = user_scope("user_slots");
    let slot_ids = preferences(
        &store,
        &scope,
        &[
            ("claim_liked_dunmere", "liked city", "Dunmere"),
            ("claim_visited_ostrel", "visited city", "Ostrel"),
            ("claim_visited_dunmere", "visited city", "Dunmere"),
        ],
    );
    let mut liked = forget(&scope, "forget_liked", "", 20);
    liked.target_selector = None;
    liked.reason = None;
    liked.target_ids = vec!["claim_liked_dunmere".to_string()];
    store.add_correction(&liked).unwrap();
    // A forget bound to the slot keeps the versions from before it, so the history must hide
    // "Ostrel" in them and nothing else.
    let mut ostrel = forget(&scope, "forget_ostrel", "Ostrel", 22);
    ostrel.target_match = CorrectionTargetMatch::CanonicalSlot;
    ostrel.target_slot_id = Some(slot_ids[1].clone());
    store.add_correction(&ostrel).unwrap();
    let visited = history_versions(&store, &scope, &slot_ids[1], 40);
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);

    assert_eq!(visited, [(Some(12), vec!["Dunmere".to_string()])]);
}

#[test]
fn source_text_joins_overlapping_spans_that_a_purge_made_shorter() {
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("purge_source_text");
    let index_dir = temp_dir("purge_source_text_index");
    let store = crate::test_store(&path, 3).unwrap();
    let scope = user_scope("user_join");
    // Each line is 50 characters, so no span starts or ends inside the forgotten value.
    let lines = (0..100)
        .map(|n| format!("Línea {n:03} names Quixadaville as the new stop {n:03}."))
        .collect::<Vec<_>>();
    assert!(lines.iter().all(|line| line.chars().count() == 49));
    let text = lines.join("\n");
    let ingested = ingest(&store, &scope, "ep_doc", &text, 10);
    let span_ids = ingested
        .spans
        .iter()
        .map(|span| span.id.clone())
        .collect::<Vec<_>>();
    let mut claim = make_claim(
        &scope,
        "claim_stop",
        "user",
        "stop",
        Some("Quixadaville"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    claim.source_span_ids = span_ids.clone();
    claim.source_episode_ids = vec![ingested.episode.id.clone()];
    store.append_claim(&claim, None).unwrap();
    store
        .add_correction(&forget(&scope, "forget_stop", "Quixadaville", 20))
        .unwrap();
    store.purge_forgotten(&scope, i64::MAX).unwrap();
    let mut builder = crate::source_index::SourceLexicalIndexBuilder::new();
    builder
        .add_source(crate::source_index::IndexedSource {
            source_id: "ep_doc",
            title: "Stops",
            directory: "docs",
            body: &text,
            date_ms: None,
            span_ids: span_ids.clone(),
        })
        .unwrap();
    builder.write(&index_dir).unwrap();
    let index = crate::source_index::SourceLexicalIndex::open(&index_dir).unwrap();
    let texts = store
        .source_texts(&scope, &index, &["ep_doc".to_string()], None)
        .unwrap();
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
    std::fs::remove_dir_all(index_dir).unwrap();
    drop(guard);

    assert!(span_ids.len() > 1, "the text must span several chunks");
    assert_eq!(texts["ep_doc"], text.replace("Quixadaville", "[forgotten]"));
}

#[test]
fn erase_episodes_deletes_the_spans_of_a_document_stored_without_its_episode() {
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("erase_document_spans");
    let store = crate::test_store(&path, 3).unwrap();
    let scope = user_scope("user_docs");
    let chunks = |id: &str, text: &str| {
        let options = ChunkOptions {
            max_chars: 20,
            overlap_chars: 5,
            ..ChunkOptions::default()
        };
        let mut input = ingest_input(&scope, id, text, 10);
        input.source_kind = SourceKind::Import;
        crate::ingest::ingest_episode(input, 10, &options)
            .spans
            .into_iter()
            .map(|span| (span, vec![1.0, 0.0, 0.0]))
            .collect::<Vec<_>>()
    };
    let long = chunks("doc", "alpha bravo charlie delta echo foxtrot golf hotel");
    let other = chunks("other", "kilo lima");
    store.append_vector_spans(&long).unwrap();
    store.append_vector_spans(&other).unwrap();
    // The document is now shorter, so every chunk of its first version must go.
    store.erase_episodes(&scope, &["doc".to_string()]).unwrap();
    store
        .append_vector_spans(&chunks("doc", "alpha bravo"))
        .unwrap();
    let mut texts = store
        .scan_spans(&scope, usize::MAX, None)
        .unwrap()
        .into_iter()
        .map(|span| span.text)
        .collect::<Vec<_>>();
    texts.sort();
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);

    assert!(long.len() > 1, "the first version must have several chunks");
    assert_eq!(texts, ["alpha bravo", "kilo lima"]);
}

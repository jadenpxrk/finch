use super::*;
use crate::ingest::EpisodeInput;
use std::collections::BTreeMap;

fn user_scope(user: &str) -> MemoryScope {
    let mut scope = scope();
    scope.user_id = Some(user.to_string());
    scope
}

fn ingest(
    store: &MemoryStore,
    scope: &MemoryScope,
    id: &str,
    text: &str,
    at_ms: i64,
) -> IngestedEpisode {
    store
        .ingest_episode(
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
            },
            at_ms,
            &ChunkOptions::default(),
        )
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
            "SELECT table_name::text FROM information_schema.tables WHERE table_schema = $1",
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
fn forget_deletes_the_forgotten_value_and_keeps_the_correction() {
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
    let result = store.apply_state_mutation_batch(write).unwrap();
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
    assert_eq!(result.corrections[0].target_selector, None);
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

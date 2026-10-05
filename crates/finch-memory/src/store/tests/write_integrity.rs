use super::*;
use crate::store::mutation_journal::{
    FAIL_NEXT_ROLLBACK, JOURNAL_APPENDS, STATE_WRITES_BEFORE_CRASH,
};
use std::io::Write;
use std::panic::{catch_unwind, AssertUnwindSafe};

type Rows = BTreeMap<&'static str, BTreeMap<String, BTreeMap<String, String>>>;

/// Every row of every journaled collection, by collection, pk, and field.
fn state_rows(store: &MemoryStore) -> Rows {
    [
        (ENTITIES_COLLECTION, &store.entities),
        (ENTITY_ALIASES_COLLECTION, &store.entity_aliases),
        (CLAIMS_COLLECTION, &store.claims),
        (RULES_COLLECTION, &store.rules),
        (CORRECTIONS_COLLECTION, &store.corrections),
        (STATE_RECORDS_COLLECTION, &store.state_records),
        (DEPENDENCY_TRACES_COLLECTION, &store.dependency_traces),
        (SLOTS_COLLECTION, &store.slots),
        (SLOT_ALIASES_COLLECTION, &store.slot_aliases),
    ]
    .into_iter()
    .map(|(name, collection)| {
        let rows = collection
            .scan_filter_only(VectorQuery::new("", Vec::new(), usize::MAX))
            .unwrap()
            .into_iter()
            .map(|doc| {
                let fields = doc
                    .fields
                    .iter()
                    .map(|(field, value)| (field.clone(), format!("{value:?}")))
                    .collect();
                (doc.pk.clone(), fields)
            })
            .collect();
        (name, rows)
    })
    .collect()
}

/// Runs `write` until its second journaled write panics like a killed process, then asserts the
/// first write landed and that reopening the store restores every prior row.
fn assert_crash_after_first_write_is_restored(
    name: &str,
    setup: impl FnOnce(&EvidencedStore),
    write: impl FnOnce(&MemoryStore) -> ZResult<()>,
) {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir(name);
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    setup(&store);
    let before = state_rows(&store);

    STATE_WRITES_BEFORE_CRASH.with(|writes| writes.set(Some(1)));
    let crashed = catch_unwind(AssertUnwindSafe(|| write(&store))).is_err();
    STATE_WRITES_BEFORE_CRASH.with(|writes| writes.set(None));
    let at_crash = state_rows(&store);
    drop(store);
    let reopened = MemoryStore::open(&dir, CollectionOptions::default()).unwrap();
    let after_reopen = state_rows(&reopened);
    drop(reopened);
    let _ = std::fs::remove_dir_all(&dir);

    assert!(crashed, "{name} finished without a second journaled write");
    assert_ne!(at_crash, before, "{name} crashed before its first write");
    assert_eq!(after_reopen, before, "{name} left rows after reopen");
}

fn bare_claim(id: &str, (subject, predicate, value): (&str, &str, &str), at: i64) -> ClaimRecord {
    let mut claim = make_claim(
        &scope(),
        id,
        subject,
        predicate,
        Some(value),
        at,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    claim.source_span_ids.clear();
    claim.source_episode_ids.clear();
    claim
}

fn owner_ana() -> ClaimRecord {
    bare_claim("claim_owner_ana", ("project", "owner", "Ana"), 10)
}

fn user_correction(target_id: &str, actor: ActorKind) -> CorrectionRecord {
    crate::ingest::add_correction(
        CorrectionInput {
            id: Some(format!("correction_{target_id}")),
            scope: scope(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            operation: CorrectionOperation::Retract,
            target_type: "claim".to_string(),
            target_ids: vec![target_id.to_string()],
            target_selector: None,
            new_value: None,
            reason: None,
            actor,
            authority: CorrectionAuthority::User,
            effective_at_ms: Some(20),
            applies_valid_from_ms: None,
            applies_valid_to_ms: None,
            cascade_policy: None,
            metadata_json: None,
        },
        20,
    )
}

fn entity_input(source_claim_ids: Vec<MemoryId>, aliases: Vec<String>) -> EntityInput {
    EntityInput {
        id: Some("entity_project".to_string()),
        scope: scope(),
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        entity_type: "project".to_string(),
        canonical_name: "Project".to_string(),
        aliases,
        source_claim_ids,
        merge_parent_ids: Vec::new(),
        split_from_id: None,
        confidence: Some(1.0),
    }
}

#[test]
fn crash_during_append_claim_is_restored_on_reopen() {
    assert_crash_after_first_write_is_restored(
        "crash_append_claim",
        |store| store.append_claim(&owner_ana(), None).unwrap(),
        |store| {
            store.append_claim(
                &bare_claim("claim_owner_bea", ("project", "owner", "Bea"), 20),
                None,
            )
        },
    );
}

#[test]
fn crash_during_add_correction_is_restored_on_reopen() {
    assert_crash_after_first_write_is_restored(
        "crash_add_correction",
        |store| store.append_claim(&owner_ana(), None).unwrap(),
        |store| store.add_correction(&user_correction("claim_owner_ana", ActorKind::User)),
    );
}

#[test]
fn crash_during_add_rule_is_restored_on_reopen() {
    assert_crash_after_first_write_is_restored(
        "crash_add_rule",
        |store| {
            store.append_claim(&owner_ana(), None).unwrap();
            let rule = make_derived_rule_inputs();
            store.seed_evidence(&scope(), &rule.source_span_ids, &rule.source_episode_ids);
        },
        |store| store.add_rule(make_derived_rule_inputs()).map(|_| ()),
    );
}

#[test]
fn crash_during_add_entity_is_restored_on_reopen() {
    assert_crash_after_first_write_is_restored(
        "crash_add_entity",
        |store| store.append_claim(&owner_ana(), None).unwrap(),
        |store| {
            store
                .add_entity(
                    entity_input(
                        vec!["claim_owner_ana".to_string()],
                        vec!["project".to_string()],
                    ),
                    None,
                )
                .map(|_| ())
        },
    );
}

#[test]
fn crash_during_add_slot_alias_is_restored_on_reopen() {
    assert_crash_after_first_write_is_restored(
        "crash_add_slot_alias",
        |store| store.append_claim(&owner_ana(), None).unwrap(),
        |store| {
            let slot_id = store.claims_by_ids(&scope(), &["claim_owner_ana".to_string()])?[0]
                .slot_id
                .clone();
            store
                .add_slot_alias(
                    SlotAliasInput {
                        id: Some("slot_alias_project_lead".to_string()),
                        scope: scope(),
                        visibility: Visibility::Private,
                        policy_tags: Vec::new(),
                        alias_subject: "project".to_string(),
                        alias_predicate: "lead".to_string(),
                        canonical_slot_id: slot_id,
                        target_claim_id: None,
                        source_claim_ids: vec!["claim_owner_ana".to_string()],
                        valid_from_ms: Some(10),
                        valid_to_ms: None,
                    },
                    30,
                )
                .map(|_| ())
        },
    );
}

#[test]
fn crash_during_refresh_state_projection_is_restored_on_reopen() {
    // At valid time 15 the reviewer claim does not hold yet, so the rebuild retires its state.
    assert_crash_after_first_write_is_restored(
        "crash_refresh_state_projection",
        |store| {
            store.append_claim(&owner_ana(), None).unwrap();
            store
                .append_claim(
                    &bare_claim("claim_reviewer_bo", ("project", "reviewer", "Bo"), 20),
                    None,
                )
                .unwrap();
        },
        |store| store.refresh_state_projection(&scope(), Some(15)),
    );
}

#[test]
fn a_write_on_one_store_does_not_skip_the_read_lock_of_another() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dirs = [
        temp_dir("reentrancy_store_a"),
        temp_dir("reentrancy_store_b"),
    ];
    let [a, b] = dirs
        .each_ref()
        .map(|dir| MemoryStore::create(dir, 3, CollectionOptions::default()).unwrap());

    let write = a.lock_state_mutation();
    let a_reads_locked = a.lock_state_read().unwrap().is_some();
    let b_reads_locked = b.lock_state_read().unwrap().is_some();
    drop(write);
    drop([a, b]);
    for dir in dirs {
        let _ = std::fs::remove_dir_all(dir);
    }

    assert!(!a_reads_locked, "a write must not relock its own store");
    assert!(
        b_reads_locked,
        "another store's write must not skip this store's lock"
    );
}

/// Runs `write` on a store holding one claim and asserts it left every row unchanged.
fn rejected_write<T>(name: &str, write: impl FnOnce(&MemoryStore) -> ZResult<T>) -> ZResult<T> {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir(name);
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    store.append_claim(&owner_ana(), None).unwrap();
    let before = state_rows(&store);
    let result = write(&store);
    let after = state_rows(&store);
    drop(store);
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(after, before, "{name} left rows behind");
    result
}

fn assert_invalid_argument<T: std::fmt::Debug>(result: ZResult<T>, message: &str) {
    let error = result.expect_err(message);
    assert_eq!(
        error.code(),
        finch_types::StatusCode::InvalidArgument,
        "{error:?}"
    );
    assert!(error.message().contains(message), "{error:?}");
}

#[test]
fn append_claim_rejects_evidence_that_does_not_exist() {
    let claim = make_claim(
        &scope(),
        "claim_unproven",
        "project",
        "owner",
        Some("Eve"),
        20,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    assert_invalid_argument(
        rejected_write("reject_claim_evidence", |store| {
            store.append_claim(&claim, None)
        }),
        "evidence spans must exist in the requested scope",
    );
}

#[test]
fn add_entity_rejects_aliases_without_source_claims() {
    assert_invalid_argument(
        rejected_write("reject_entity_alias_evidence", |store| {
            store.add_entity(entity_input(Vec::new(), vec!["project".to_string()]), None)
        }),
        "entity aliases require source claim evidence",
    );
}

#[test]
fn add_entity_rejects_source_claims_outside_its_scope() {
    assert_invalid_argument(
        rejected_write("reject_entity_source_claims", |store| {
            store.add_entity(
                entity_input(vec!["claim_missing".to_string()], Vec::new()),
                None,
            )
        }),
        "entity source claims must exist in the entity scope",
    );
}

#[test]
fn add_correction_rejects_missing_or_unresolved_evidence() {
    assert_invalid_argument(
        rejected_write("reject_correction_actor", |store| {
            store.add_correction(&user_correction("claim_owner_ana", ActorKind::Assistant))
        }),
        "non-user corrections require source evidence",
    );
    let mut correction = user_correction("claim_owner_ana", ActorKind::User);
    correction.source_span_ids = vec!["span_missing".to_string()];
    correction.source_episode_ids = vec!["ep_missing".to_string()];
    assert_invalid_argument(
        rejected_write("reject_correction_evidence", |store| {
            store.add_correction(&correction)
        }),
        "evidence spans must exist in the requested scope",
    );
}

#[test]
fn add_rule_rejects_evidence_that_does_not_exist() {
    assert_invalid_argument(
        rejected_write("reject_rule_evidence", |store| {
            store.add_rule(make_derived_rule_inputs())
        }),
        "evidence spans must exist in the requested scope",
    );
}

#[test]
fn state_mutation_batch_rejects_evidence_that_does_not_exist() {
    let claim = make_claim(
        &scope(),
        "claim_unproven",
        "project",
        "owner",
        Some("Eve"),
        20,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    assert_invalid_argument(
        rejected_write("reject_batch_evidence", |store| {
            store.apply_state_mutation_batch(StateMutationBatch {
                scope: scope(),
                transaction_time_ms: 30,
                propagation_seed: "reject_batch_evidence".to_string(),
                max_rule_hops: 8,
                claims: vec![claim],
                entities: Vec::new(),
                rules: Vec::new(),
                slot_aliases: Vec::new(),
                corrections: Vec::new(),
            })
        }),
        "evidence spans must exist in the requested scope",
    );
}

#[test]
fn correction_reevaluates_every_current_claim_of_its_slot() {
    // More current claims than one vector query returns; each surviving trigger keeps its trace.
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("correction_reevaluates_every_claim");
    let store = EvidencedStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    store
        .add_rule(RuleInput {
            id: Some("rule_drink_suggestion".to_string()),
            trigger_subject_key: "user".to_string(),
            trigger_predicate_key: "drink".to_string(),
            target_subject_key: "user".to_string(),
            target_predicate_key: "suggestion".to_string(),
            trigger_subject: Some("user".to_string()),
            trigger_predicate: Some("drink".to_string()),
            target_subject: Some("user".to_string()),
            target_predicate: Some("suggestion".to_string()),
            ..make_derived_rule_inputs()
        })
        .unwrap();
    let claim_ids = (0..MAX_VECTOR_QUERY_TOPK + 2)
        .map(|index| format!("claim_drink_{index:04}"))
        .collect::<Vec<_>>();
    let claims = claim_ids
        .iter()
        .map(|id| {
            make_claim(
                &scope,
                id,
                "user",
                "drink",
                Some(id),
                10,
                (ClaimKind::Preference, ClaimPolarity::Affirmative),
            )
        })
        .collect();
    store
        .apply_state_mutation_batch(StateMutationBatch {
            scope: scope.clone(),
            transaction_time_ms: 15,
            propagation_seed: "drinks".to_string(),
            max_rule_hops: 8,
            claims,
            entities: Vec::new(),
            rules: Vec::new(),
            slot_aliases: Vec::new(),
            corrections: Vec::new(),
        })
        .unwrap();

    store
        .add_correction(&user_correction(&claim_ids[0], ActorKind::User))
        .unwrap();
    let traced = store
        .scan_dependency_traces(&scope, usize::MAX, None)
        .unwrap()
        .into_iter()
        .map(|trace| trace.trigger_claim_id)
        .collect::<BTreeSet<_>>();
    drop(store);
    let _ = std::fs::remove_dir_all(&dir);

    let untraced = claim_ids[1..]
        .iter()
        .filter(|id| !traced.contains(*id))
        .collect::<Vec<_>>();
    assert_eq!(untraced, Vec::<&String>::new());
}

#[test]
fn scan_slots_applies_each_slot_valid_interval() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("scan_slots_valid_interval");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    store.append_claim(&owner_ana(), None).unwrap();
    // No writer bounds a slot's validity yet, so the stored versions get one directly.
    let bounded = store
        .slots
        .scan_filter_only(VectorQuery::new("", Vec::new(), usize::MAX))
        .unwrap()
        .into_iter()
        .map(|doc| {
            let mut slot = slot_from_doc(&doc).unwrap();
            slot.valid_from_ms = Some(10);
            slot.valid_to_ms = Some(20);
            slot_doc(&slot).unwrap()
        })
        .collect();
    upsert_many(&store.slots, bounded).unwrap();

    let slot_count_at = |at_ms| store.scan_slots(&scope(), 10, Some(at_ms)).unwrap().len();
    let counts = [5, 10, 19, 20].map(slot_count_at);
    drop(store);
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(counts, [0, 1, 1, 0]);
}

#[test]
fn a_mutation_of_2000_claims_appends_to_the_journal_without_rewriting_it() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("journal_appends");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    JOURNAL_APPENDS.with(|appends| appends.borrow_mut().clear());
    let captured = store.with_state_mutation(&scope(), || {
        for i in 0..2_000 {
            let project = format!("project {i}");
            let claim = bare_claim(&format!("claim_owner_{i}"), (&project, "owner", "Ana"), i);
            let docs = [claim_doc(&claim, None).map_err(json_error)?];
            store.capture_state_mutation_docs(CLAIMS_COLLECTION, &store.claims, &docs)?;
        }
        Ok(())
    });
    let appends = JOURNAL_APPENDS.with(|appends| appends.take());
    drop(store);
    let _ = std::fs::remove_dir_all(&dir);

    captured.unwrap();
    assert_eq!(appends.len(), 2_000);
    for pair in appends.windows(2) {
        let [(_, previous_length), (written, length)] = pair else {
            unreachable!()
        };
        assert_eq!(
            *length,
            previous_length + written,
            "a capture rewrote the journal"
        );
    }
}

#[test]
fn reopen_recovers_a_torn_journal_tail_and_a_whole_file_journal() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let whole_file = serde_json::json!({
        "scope": scope(),
        "collections": [{
            "name": CLAIMS_COLLECTION,
            "documents": [{"pk": "claim_owner_bea", "doc": null}],
        }],
    });
    for (name, torn_tail) in [("journal_torn_tail", true), ("journal_whole_file", false)] {
        let dir = temp_dir(name);
        let journal = dir.join(".state-mutation-journal.json");
        let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
        store.append_claim(&owner_ana(), None).unwrap();
        let before = state_rows(&store);
        if torn_tail {
            store.begin_state_mutation_journal(&scope()).unwrap();
            store
                .capture_state_mutation_documents(
                    CLAIMS_COLLECTION,
                    &store.claims,
                    ["claim_owner_bea".to_string()],
                )
                .unwrap();
            let mut file = std::fs::OpenOptions::new()
                .append(true)
                .open(&journal)
                .unwrap();
            file.write_all(b"0000004c{\"name\":\"claims\"").unwrap();
        } else {
            std::fs::write(&journal, whole_file.to_string()).unwrap();
        }
        let bea = bare_claim("claim_owner_bea", ("project", "owner", "Bea"), 20);
        insert_one(&store.claims, claim_doc(&bea, None).unwrap()).unwrap();
        drop(store);

        let reopened = MemoryStore::open(&dir, CollectionOptions::default()).unwrap();
        let after = state_rows(&reopened);
        drop(reopened);
        let journal_left = journal.exists();
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(after, before, "{name} left rows after reopen");
        assert!(!journal_left, "{name} left its journal");
    }
}

#[test]
fn a_failed_rollback_refuses_reads_and_writes_until_reopen() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("failed_rollback");
    let store = MemoryStore::create(&dir, 3, CollectionOptions::default()).unwrap();
    store.append_claim(&owner_ana(), None).unwrap();
    let before = state_rows(&store);

    FAIL_NEXT_ROLLBACK.with(|fail| fail.set(true));
    let failed = store.with_state_mutation(&scope(), || {
        let bea = bare_claim("claim_owner_bea", ("project", "owner", "Bea"), 20);
        let docs = vec![claim_doc(&bea, None).map_err(json_error)?];
        store.capture_state_mutation_docs(CLAIMS_COLLECTION, &store.claims, &docs)?;
        insert_many(&store.claims, docs)?;
        Err::<(), _>(Status::internal("injected mutation failure"))
    });
    FAIL_NEXT_ROLLBACK.with(|fail| fail.set(false));
    let read = store.scan_claims(&scope(), 10, None).map(|_| ());
    let cy = bare_claim("claim_owner_cy", ("project", "owner", "Cy"), 30);
    let write = store.append_claim(&cy, None).map(|_| ());
    drop(store);
    let reopened = MemoryStore::open(&dir, CollectionOptions::default()).unwrap();
    let after = state_rows(&reopened);
    let reopened_claims = reopened
        .scan_claims(&scope(), 10, None)
        .map(|claims| claims.len());
    drop(reopened);
    let _ = std::fs::remove_dir_all(&dir);

    for (what, result) in [("mutation", failed), ("read", read), ("write", write)] {
        let error = result.expect_err(what);
        assert!(error.message().contains("reopen"), "{what}: {error:?}");
    }
    assert_eq!(after, before, "reopen left the failed mutation's rows");
    assert_eq!(reopened_claims, Ok(1));
}

#[test]
fn concurrent_claim_scan_sees_a_mutation_batch_whole_or_not_at_all() {
    // Readers must see all claims in a committed state mutation batch or none of them.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("batch_visibility");
    let store = MemoryStore::create(&path, 3, CollectionOptions::default()).unwrap();
    let scope = scope();
    let evidence = store
        .ingest_episode(
            EpisodeInput {
                id: Some("evidence".to_string()),
                scope: scope.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                source_kind: SourceKind::UserMessage,
                actor: ActorKind::User,
                sequence_no: 1,
                event_time_ms: Some(10),
                valid_from_ms: Some(10),
                valid_to_ms: None,
                raw_text: "The inventory contains 64 available items.".to_string(),
                blob_ref: None,
                mime_type: None,
                causal_parent_ids: Vec::new(),
                metadata_json: None,
            },
            10,
            &ChunkOptions::default(),
        )
        .unwrap();
    let barrier = std::sync::Barrier::new(2);
    let done = std::sync::atomic::AtomicBool::new(false);
    let partial = std::thread::scope(|threads| {
        let writer = threads.spawn(|| {
            barrier.wait();
            for batch_index in 0..4 {
                let claims = (0..64)
                    .map(|index| {
                        let mut claim = make_claim(
                            &scope,
                            &format!("claim_{batch_index}_{index}"),
                            &format!("item_{batch_index}_{index}"),
                            "availability",
                            Some("available"),
                            10,
                            (ClaimKind::Fact, ClaimPolarity::Affirmative),
                        );
                        claim.source_span_ids = vec![evidence.spans[0].id.clone()];
                        claim.source_episode_ids = vec![evidence.episode.id.clone()];
                        claim
                    })
                    .collect();
                store
                    .apply_state_mutation_batch(StateMutationBatch {
                        scope: scope.clone(),
                        transaction_time_ms: 100 + batch_index,
                        propagation_seed: format!("batch_{batch_index}"),
                        max_rule_hops: 8,
                        claims,
                        entities: Vec::new(),
                        rules: Vec::new(),
                        slot_aliases: Vec::new(),
                        corrections: Vec::new(),
                    })
                    .unwrap();
            }
            done.store(true, Ordering::Release);
        });
        barrier.wait();
        let mut partial = None;
        for iteration in 0..20_000 {
            let count = store
                .scan_claims(&scope, usize::MAX, Some(20))
                .unwrap()
                .len();
            if !count.is_multiple_of(64) {
                partial = Some((iteration, count));
                break;
            }
            if done.load(Ordering::Acquire) {
                break;
            }
            std::thread::yield_now();
        }
        writer.join().unwrap();
        partial
    });
    let committed_count = store
        .scan_claims(&scope, usize::MAX, Some(20))
        .unwrap()
        .len();
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);
    assert_eq!(committed_count, 256);
    assert_eq!(partial, None, "a reader observed an incomplete batch");
}

#[test]
fn concurrent_entity_updates_keep_every_acknowledged_alias_and_source() {
    // Concurrent additions to one entity must preserve both writers' aliases and evidence.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("entity_update_race");
    let store = MemoryStore::create(&path, 3, CollectionOptions::default()).unwrap();
    for id in ["left_claim", "right_claim"] {
        let mut claim = make_claim(
            &scope(),
            id,
            "Alex",
            "role",
            Some("engineer"),
            10,
            (ClaimKind::Fact, ClaimPolarity::Affirmative),
        );
        claim.source_span_ids.clear();
        claim.source_episode_ids.clear();
        store.append_claim(&claim, None).unwrap();
    }
    let mut lost = None;
    for attempt in 0..32 {
        let entity_id = format!("entity_{attempt}");
        let inputs = ["left", "right"].map(|side| EntityInput {
            id: Some(entity_id.clone()),
            scope: scope(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            entity_type: "person".to_string(),
            canonical_name: format!("Alex {attempt}"),
            aliases: vec![format!("{side} {attempt}")],
            source_claim_ids: vec![format!("{side}_claim")],
            merge_parent_ids: Vec::new(),
            split_from_id: None,
            confidence: None,
        });
        let barrier = std::sync::Barrier::new(2);
        std::thread::scope(|threads| {
            let workers = inputs
                .into_iter()
                .map(|input| {
                    let store = &store;
                    let barrier = &barrier;
                    threads.spawn(move || {
                        barrier.wait();
                        store.add_entity(input, None)
                    })
                })
                .collect::<Vec<_>>();
            for worker in workers {
                worker.join().unwrap().unwrap();
            }
        });
        let entity = store
            .scan_entities(&scope(), 100)
            .unwrap()
            .into_iter()
            .find(|entity| entity.id == entity_id)
            .unwrap();
        if entity.aliases.len() != 2 || entity.source_claim_ids.len() != 2 {
            lost = Some(entity_id);
            break;
        }
    }
    drop(store);
    let reopened = MemoryStore::open(&path, CollectionOptions::default()).unwrap();
    let persisted = reopened.scan_entities(&scope(), 100).unwrap();
    let lost = lost.map(|id| {
        persisted
            .into_iter()
            .find(|entity| entity.id == id)
            .unwrap()
    });
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);
    assert!(
        lost.is_none(),
        "an acknowledged entity update was lost after reopen: {lost:?}"
    );
}

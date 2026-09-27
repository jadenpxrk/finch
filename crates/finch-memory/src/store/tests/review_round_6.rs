use super::*;

#[test]
fn space_level_claim_supersedes_another_tenants_state() {
    // Writing a space-level claim must not retire state owned by a narrower tenant scope.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("round_6_scope_projection");
    let store = MemoryStore::create(&path, 3, CollectionOptions::default()).unwrap();
    let mut tenant = scope();
    tenant.tenant_id = Some("tenant_a".to_string());
    for (claim_scope, id, value, at) in [
        (&tenant, "tenant_claim", "private", 10),
        (&scope(), "space_claim", "shared", 20),
    ] {
        let mut claim = make_claim(
            claim_scope,
            id,
            "project",
            "owner",
            Some(value),
            at,
            (ClaimKind::Fact, ClaimPolarity::Affirmative),
        );
        claim.source_span_ids.clear();
        claim.source_episode_ids.clear();
        store.append_claim(&claim, None).unwrap();
        if claim_scope == &tenant {
            assert_eq!(
                store
                    .scan_state_records(&tenant, state_scan(10, Some(30)))
                    .unwrap()
                    .len(),
                1
            );
        }
    }
    drop(store);
    let reopened = MemoryStore::open(&path, CollectionOptions::default()).unwrap();
    let tenant_states = reopened
        .scan_state_records(&tenant, state_scan(10, Some(30)))
        .unwrap();
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);
    assert_eq!(
        tenant_states.len(),
        1,
        "the space-level write retired tenant A's state"
    );
}

#[test]
fn concurrent_claim_scan_observes_part_of_an_atomic_mutation_batch() {
    // Readers must see all claims in a committed state mutation batch or none of them.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("round_6_batch_visibility");
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

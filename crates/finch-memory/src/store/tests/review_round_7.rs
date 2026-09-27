use super::*;

#[test]
fn concurrent_entity_updates_lose_acknowledged_aliases_and_provenance() {
    // Concurrent additions to one entity must preserve both writers' aliases and evidence.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("round_7_entity_update_race");
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

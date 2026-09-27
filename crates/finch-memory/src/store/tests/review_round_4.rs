use super::*;

#[test]
fn ranked_graph_limit_discards_the_highest_confidence_edge() {
    // A ranked result limit must keep the strongest edge regardless of insertion order.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("round_4_graph_ranking");
    let store = MemoryStore::create(&path, 3, CollectionOptions::default()).unwrap();
    for (id, confidence) in [("weak", 0.1), ("strong", 0.9)] {
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
                valid_to_ms: None,
                confidence: Some(confidence),
            })
            .unwrap();
    }
    drop(store);
    let reopened = MemoryStore::open(&path, CollectionOptions::default()).unwrap();
    let all = reopened
        .expand_edges_ranked(&scope(), &["seed".to_string()], 1, 2, Some(20))
        .unwrap();
    let limited = reopened
        .expand_edges_ranked(&scope(), &["seed".to_string()], 1, 1, Some(20))
        .unwrap();
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);
    assert_eq!(all.len(), 2);
    assert_eq!(all[0].edge.id, "strong");
    assert_eq!(limited.len(), 1);
    assert_eq!(limited[0].edge.id, all[0].edge.id);
}

#[test]
fn span_scan_returns_more_records_than_its_limit() {
    // A scan must enforce its public result limit after scope and validity filtering.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("round_4_span_scan_limit");
    let store = MemoryStore::create(&path, 3, CollectionOptions::default()).unwrap();
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
    let reopened = MemoryStore::open(&path, CollectionOptions::default()).unwrap();
    let all = reopened.scan_spans(&scope(), 10, Some(20)).unwrap();
    let limited = reopened.scan_spans(&scope(), 1, Some(20)).unwrap();
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);
    assert_eq!(all.len(), 2);
    assert_eq!(limited.len(), 1);
}

use super::*;

#[test]
fn expired_artifact_consumes_limit_and_hides_active_artifact() {
    // Expired artifacts must not consume the caller's limit on active artifacts.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("round_5_artifact_limit");
    let store = MemoryStore::create(&path, 3, CollectionOptions::default()).unwrap();
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
    let reopened = MemoryStore::open(&path, CollectionOptions::default()).unwrap();
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
fn entity_scan_returns_more_records_than_its_limit() {
    // Entity scans must enforce their public result limit after scope and status filtering.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("round_5_entity_limit");
    let store = MemoryStore::create(&path, 3, CollectionOptions::default()).unwrap();
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
    let reopened = MemoryStore::open(&path, CollectionOptions::default()).unwrap();
    let all = reopened.scan_entities(&scope(), 10).unwrap();
    let limited = reopened.scan_entities(&scope(), 1).unwrap();
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);
    assert_eq!(all.len(), 2);
    assert_eq!(limited.len(), 1);
}

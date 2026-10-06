use super::*;

#[test]
fn memory_store_persists_active_records() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("persist");
    let store = crate::test_store(&dir, 3).unwrap();
    let ingested = store
        .ingest_episode(
            EpisodeInput {
                id: Some("ep_store".to_string()),
                scope: scope(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                source_kind: SourceKind::UserMessage,
                actor: ActorKind::User,
                sequence_no: 1,
                event_time_ms: Some(10),
                valid_from_ms: None,
                valid_to_ms: None,
                raw_text: "alpha beta gamma".to_string(),
                blob_ref: None,
                mime_type: Some("text/plain".to_string()),
                causal_parent_ids: Vec::new(),
                metadata_json: None,
            },
            11,
            &ChunkOptions {
                max_chars: 8,
                overlap_chars: 0,
                chunker_version: "test".to_string(),
            },
        )
        .unwrap();
    assert_eq!(ingested.spans.len(), 2);

    let correction = crate::ingest::add_correction(
        CorrectionInput {
            id: Some("corr_store".to_string()),
            scope: scope(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            operation: CorrectionOperation::Tombstone,
            target_type: "span".to_string(),
            target_ids: vec![ingested.spans[0].id.clone()],
            target_selector: None,
            new_value: None,
            reason: None,
            actor: ActorKind::User,
            authority: CorrectionAuthority::User,
            effective_at_ms: None,
            applies_valid_from_ms: None,
            applies_valid_to_ms: None,
            cascade_policy: None,
            metadata_json: None,
        },
        12,
    );
    store.add_correction(&correction).unwrap();
    drop(store);

    let reopened = crate::reopen_test_store(&dir).unwrap();
    assert!(reopened
        .episodes
        .fetch(vec!["ep_store".to_string()])
        .unwrap()
        .contains_key("ep_store"));
}

#[test]
fn memory_store_preserves_episode_actor_in_retrieved_provenance() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("retrieved_actor_provenance");
    let store = crate::test_store(&dir, 3).unwrap();
    store
        .ingest_episode(
            EpisodeInput {
                id: Some("ep_assistant".to_string()),
                scope: scope(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                source_kind: SourceKind::AssistantMessage,
                actor: ActorKind::Assistant,
                sequence_no: 1,
                event_time_ms: Some(10),
                valid_from_ms: None,
                valid_to_ms: None,
                raw_text: "El proyecto sigue activo.".to_string(),
                blob_ref: None,
                mime_type: Some("text/plain".to_string()),
                causal_parent_ids: Vec::new(),
                metadata_json: None,
            },
            11,
            &ChunkOptions::default(),
        )
        .unwrap();

    let hits = store
        .keyword_search_spans(&scope(), "proyecto", 1, 10, Some(20))
        .unwrap();

    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].span.provenance[0].actor, Some(ActorKind::Assistant));
    let context = crate::build_context(
        &hits,
        crate::ContextOptions {
            token_budget: 128,
            include_provenance: true,
        },
    );
    assert!(context.body.contains("source_actor: assistant"));
}

#[test]
fn memory_store_ingests_artifact_text_as_keyword_searchable_spans() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("artifact_text");
    let store = crate::test_store(&dir, 3).unwrap();
    let artifact = ArtifactRecord {
        id: "artifact_text".to_string(),
        scope: scope(),
        status: MemoryStatus::Active,
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        artifact_kind: ArtifactKind::File,
        title: "notes.md".to_string(),
        uri: Some("file:///notes.md".to_string()),
        blob_ref: None,
        mime_type: Some("text/markdown".to_string()),
        content_hash: "artifact_hash".to_string(),
        source_created_at_ms: None,
        source_modified_at_ms: None,
        created_at_ms: 20,
        ingested_at_ms: 20,
        valid_from_ms: Some(20),
        valid_to_ms: None,
        extracted_text_ref: None,
        metadata_json: None,
    };
    let ingested = store
        .ingest_artifact_text(
            artifact,
            "Artifact text contains RARE-ARTIFACT-KEY.",
            &ChunkOptions {
                max_chars: 200,
                overlap_chars: 0,
                chunker_version: "test".to_string(),
            },
        )
        .unwrap();
    assert_eq!(ingested.spans.len(), 1);
    assert_eq!(ingested.spans[0].source_id, "artifact_text");
    let hits = store
        .keyword_search_spans(&scope(), "RARE-ARTIFACT-KEY", 10, 100, Some(20))
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].span.source_id, "artifact_text");
}

#[test]
fn memory_store_appends_persistent_vector_only_spans() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("vector_only_spans");
    let store = crate::test_store(&dir, 3).unwrap();
    let ingested = ingest_episode(
        EpisodeInput {
            id: Some("external_document".to_string()),
            scope: scope(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            source_kind: SourceKind::Import,
            actor: ActorKind::External,
            sequence_no: 1,
            event_time_ms: None,
            valid_from_ms: None,
            valid_to_ms: None,
            raw_text: "external vector document".to_string(),
            blob_ref: None,
            mime_type: Some("text/plain".to_string()),
            causal_parent_ids: Vec::new(),
            metadata_json: None,
        },
        1,
        &ChunkOptions::default(),
    );
    store
        .append_vector_spans(&[(ingested.spans[0].clone(), vec![1.0, 0.0, 0.0])])
        .unwrap();
    drop(store);

    let reopened = crate::reopen_test_store(&dir).unwrap();
    let vector_hits = reopened
        .query_spans(&scope(), vec![1.0, 0.0, 0.0], 1, None)
        .unwrap();
    assert_eq!(vector_hits[0].span.source_id, "external_document");
    assert!(reopened
        .keyword_search_spans(&scope(), "external", 10, 100, None)
        .unwrap()
        .is_empty());
}

#[test]
fn memory_store_artifact_limit_applies_after_scope_filtering() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("artifact_limit_after_filtering");
    let store = crate::test_store(&dir, 3).unwrap();
    let mut target_scope = scope();
    target_scope.user_id = Some("target_user".to_string());
    let mut other_scope = scope();
    other_scope.user_id = Some("other_user".to_string());

    for i in 0..MAX_VECTOR_QUERY_TOPK {
        store
            .append_artifact(&ArtifactRecord {
                id: format!("artifact_other_{i}"),
                scope: other_scope.clone(),
                status: MemoryStatus::Active,
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                artifact_kind: ArtifactKind::File,
                title: format!("other-{i}.md"),
                uri: None,
                blob_ref: None,
                mime_type: Some("text/markdown".to_string()),
                content_hash: format!("artifact_other_hash_{i}"),
                source_created_at_ms: None,
                source_modified_at_ms: None,
                created_at_ms: 10,
                ingested_at_ms: 10,
                valid_from_ms: Some(10),
                valid_to_ms: None,
                extracted_text_ref: None,
                metadata_json: None,
            })
            .unwrap();
    }
    store
        .append_artifact(&ArtifactRecord {
            id: "artifact_target_runbook".to_string(),
            scope: target_scope.clone(),
            status: MemoryStatus::Active,
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            artifact_kind: ArtifactKind::File,
            title: "target-runbook.md".to_string(),
            uri: None,
            blob_ref: None,
            mime_type: Some("text/markdown".to_string()),
            content_hash: "artifact_target_hash".to_string(),
            source_created_at_ms: None,
            source_modified_at_ms: None,
            created_at_ms: 20,
            ingested_at_ms: 20,
            valid_from_ms: Some(20),
            valid_to_ms: None,
            extracted_text_ref: None,
            metadata_json: None,
        })
        .unwrap();

    let artifacts = store.scan_artifacts(&target_scope, 1, Some(30)).unwrap();
    assert_eq!(artifacts.len(), 1);
    assert_eq!(artifacts[0].id, "artifact_target_runbook");
}

#[test]
fn memory_store_keyword_search_chunks_large_term_queries() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("keyword_many_terms");
    let store = crate::test_store(&dir, 3).unwrap();

    let mut span = span_record("many_terms_span", MemoryStatus::Active, Some(1));
    span.text = "term39 should be found".to_string();
    span.lexical_text = span.text.clone();
    store.append_span(&span, Some(&[1.0, 0.0, 0.0])).unwrap();

    let query = (0..40)
        .map(|i| format!("term{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    let hits = store
        .keyword_search_spans(&scope(), &query, 10, 100, Some(1))
        .unwrap();

    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].span.id, "many_terms_span");
}

#[test]
fn memory_store_keyword_search_chunks_large_span_id_fetches() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("keyword_many_ids");
    let store = crate::test_store(&dir, 3).unwrap();

    for i in 0..40 {
        let mut span = span_record(
            &format!("many_ids_span_{i:02}"),
            MemoryStatus::Active,
            Some(1),
        );
        span.text = format!("alpha shared memory {i}");
        span.lexical_text = span.text.clone();
        store.append_span(&span, Some(&[1.0, 0.0, 0.0])).unwrap();
    }

    let hits = store
        .keyword_search_spans(&scope(), "alpha", 40, 100, Some(1))
        .unwrap();

    assert_eq!(hits.len(), 40);
}

#[test]
fn memory_store_completes_span_evidence_with_neighbors() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("span_neighbors");
    let store = crate::test_store(&dir, 3).unwrap();

    let mut before = span_record("span_before", MemoryStatus::Active, Some(1));
    before.span_index = 0;
    let mut middle = span_record("span_middle", MemoryStatus::Active, Some(1));
    middle.span_index = 1;
    let mut after = span_record("span_after", MemoryStatus::Active, Some(1));
    after.span_index = 2;
    for span in [&before, &middle, &after] {
        store.append_span(span, Some(&[1.0, 0.0, 0.0])).unwrap();
    }

    let completed = store
        .complete_span_evidence(
            &scope(),
            &[SpanSearchHit {
                span: middle,
                score: 1.0,
            }],
            1,
            1,
            100,
            Some(2),
        )
        .unwrap();
    let ids = completed
        .iter()
        .map(|hit| hit.span.id.as_str())
        .collect::<Vec<_>>();
    assert_eq!(ids, vec!["span_before", "span_middle", "span_after"]);
}

#[test]
fn memory_store_source_diverse_search_limits_repeated_documents() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("source_diverse_search");
    let store = crate::test_store(&dir, 3).unwrap();

    for (id, source_id, embedding) in [
        ("alpha_0", "document_alpha", [1.0, 0.0, 0.0]),
        ("alpha_1", "document_alpha", [0.99, 0.01, 0.0]),
        ("beta_0", "document_beta", [0.98, 0.02, 0.0]),
        ("gamma_0", "document_gamma", [0.97, 0.03, 0.0]),
    ] {
        let mut span = span_record(id, MemoryStatus::Active, Some(1));
        span.source_id = source_id.to_string();
        store.append_span(&span, Some(&embedding)).unwrap();
    }

    let hits = store
        .query_source_diverse_spans(&scope(), vec![1.0, 0.0, 0.0], 4, 2, 1, Some(2))
        .unwrap();
    assert_eq!(hits.len(), 2);
    assert_ne!(hits[0].span.source_id, hits[1].span.source_id);
}

#[test]
fn memory_store_hybrid_source_diverse_search_fuses_lexical_sources_and_hydrates_spans() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("hybrid_source_diverse_search");
    let store = crate::test_store(&dir, 3).unwrap();

    // document_alpha is close in vector space; document_beta only matches lexically.
    for (id, source_id, span_index, text, embedding) in [
        (
            "alpha_0",
            "document_alpha",
            0,
            "general overview",
            [1.0, 0.0, 0.0],
        ),
        (
            "alpha_1",
            "document_alpha",
            1,
            "more detail",
            [0.99, 0.01, 0.0],
        ),
        (
            "beta_0",
            "document_beta",
            0,
            "kestrel failover runbook",
            [0.0, 1.0, 0.0],
        ),
        (
            "beta_1",
            "document_beta",
            1,
            "kestrel steps",
            [0.0, 0.99, 0.01],
        ),
    ] {
        let mut span = span_record(id, MemoryStatus::Active, Some(1));
        span.source_id = source_id.to_string();
        span.span_index = span_index;
        span.text = text.to_string();
        span.lexical_text = text.to_string();
        store.append_span(&span, Some(&embedding)).unwrap();
    }

    let index_dir = temp_dir("hybrid_source_diverse_index");
    let mut builder = crate::source_index::SourceLexicalIndexBuilder::new();
    builder
        .add_source(
            "document_alpha",
            "",
            "",
            "general overview more detail",
            vec!["alpha_0".to_string(), "alpha_1".to_string()],
        )
        .unwrap();
    builder
        .add_source(
            "document_beta",
            "",
            "",
            "kestrel failover runbook kestrel steps",
            vec!["beta_0".to_string(), "beta_1".to_string()],
        )
        .unwrap();
    builder.write(&index_dir).unwrap();
    let index = crate::source_index::SourceLexicalIndex::open(&index_dir).unwrap();

    let hits = store
        .query_hybrid_source_diverse_spans(
            &scope(),
            HybridSourceDiverseQuery {
                query_embedding: vec![1.0, 0.0, 0.0],
                query_text: "kestrel failover",
                expansions: &[],
                supplemental_queries: &[],
                index: &index,
                at_ms: Some(2),
            },
            HybridSourceDiverseOptions {
                candidate_k: 4,
                lexical_source_k: 4,
                max_sources: 2,
                max_spans_per_source: 2,
                neighbor_radius: 1,
                lexical_max_df_ratio: 1.0,
                expansion_weight: 0.5,
                supplemental_sources: 0,
                supplemental_spans_per_source: 0,
                supplemental_candidate_k: 0,
            },
        )
        .unwrap();

    let sources = hits
        .iter()
        .map(|hit| hit.span.source_id.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    // The lexical-only source is present even though its vectors are far away,
    // and its spans were hydrated from the registry.
    assert!(sources.contains("document_beta"));
    assert!(sources.contains("document_alpha"));
    assert!(hits.iter().any(|hit| hit.span.id == "beta_0"));
    // Round-robin interleave: first spans of every source come first.
    assert_ne!(hits[0].span.source_id, hits[1].span.source_id);
    let _ = std::fs::remove_dir_all(&index_dir);
}

#[test]
fn memory_store_hybrid_source_diverse_search_fuses_expansion_lexical_sources() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("hybrid_source_diverse_expansion_search");
    let store = crate::test_store(&dir, 3).unwrap();

    for (id, source_id, text, embedding) in [
        (
            "overview_0",
            "document_overview",
            "general orientation",
            [1.0, 0.0, 0.0],
        ),
        (
            "checklist_0",
            "document_checklist",
            "avionics checklist",
            [0.0, 1.0, 0.0],
        ),
        (
            "release_0",
            "document_release",
            "deployment playbook",
            [0.0, 0.0, 1.0],
        ),
    ] {
        let mut span = span_record(id, MemoryStatus::Active, Some(1));
        span.source_id = source_id.to_string();
        span.text = text.to_string();
        span.lexical_text = text.to_string();
        store.append_span(&span, Some(&embedding)).unwrap();
    }

    let index_dir = temp_dir("hybrid_source_diverse_expansion_index");
    let mut builder = crate::source_index::SourceLexicalIndexBuilder::new();
    builder
        .add_source(
            "document_overview",
            "",
            "",
            "general orientation",
            vec!["overview_0".to_string()],
        )
        .unwrap();
    builder
        .add_source(
            "document_checklist",
            "",
            "",
            "avionics checklist",
            vec!["checklist_0".to_string()],
        )
        .unwrap();
    builder
        .add_source(
            "document_release",
            "",
            "",
            "deployment playbook",
            vec!["release_0".to_string()],
        )
        .unwrap();
    builder.write(&index_dir).unwrap();
    let index = crate::source_index::SourceLexicalIndex::open(&index_dir).unwrap();
    let options = HybridSourceDiverseOptions {
        candidate_k: 1,
        lexical_source_k: 3,
        max_sources: 3,
        max_spans_per_source: 1,
        neighbor_radius: 0,
        lexical_max_df_ratio: 1.0,
        expansion_weight: 0.5,
        supplemental_sources: 0,
        supplemental_spans_per_source: 0,
        supplemental_candidate_k: 0,
    };

    let without_expansion = store
        .query_hybrid_source_diverse_spans(
            &scope(),
            HybridSourceDiverseQuery {
                query_embedding: vec![1.0, 0.0, 0.0],
                query_text: "avionics checklist",
                expansions: &[],
                supplemental_queries: &[],
                index: &index,
                at_ms: Some(2),
            },
            options,
        )
        .unwrap();
    assert!(!without_expansion
        .iter()
        .any(|hit| hit.span.source_id == "document_release"));

    let with_expansion = store
        .query_hybrid_source_diverse_spans(
            &scope(),
            HybridSourceDiverseQuery {
                query_embedding: vec![1.0, 0.0, 0.0],
                query_text: "avionics checklist",
                expansions: &[ExpansionQuery {
                    text: "deployment playbook".to_string(),
                    embedding: None,
                }],
                supplemental_queries: &[],
                index: &index,
                at_ms: Some(2),
            },
            options,
        )
        .unwrap();
    assert!(with_expansion
        .iter()
        .any(|hit| hit.span.source_id == "document_release"));
    assert!(with_expansion.iter().any(|hit| hit.span.id == "release_0"));

    let _ = std::fs::remove_dir_all(&index_dir);
}

#[test]
fn memory_store_hybrid_source_diverse_search_appends_supplemental_sources() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("hybrid_source_diverse_supplemental_search");
    let store = crate::test_store(&dir, 3).unwrap();

    for (id, source_id, text, embedding) in [
        (
            "handbook_0",
            "document_handbook",
            "incident response handbook",
            [1.0, 0.0, 0.0],
        ),
        (
            "runbook_0",
            "document_runbook",
            "incident response runbook",
            [0.99, 0.01, 0.0],
        ),
        (
            "release_0",
            "document_release",
            "quarterly release schedule",
            [0.0, 1.0, 0.0],
        ),
    ] {
        let mut span = span_record(id, MemoryStatus::Active, Some(1));
        span.source_id = source_id.to_string();
        span.text = text.to_string();
        span.lexical_text = text.to_string();
        store.append_span(&span, Some(&embedding)).unwrap();
    }

    let index_dir = temp_dir("hybrid_source_diverse_supplemental_index");
    let mut builder = crate::source_index::SourceLexicalIndexBuilder::new();
    builder
        .add_source(
            "document_handbook",
            "",
            "",
            "incident response handbook",
            vec!["handbook_0".to_string()],
        )
        .unwrap();
    builder
        .add_source(
            "document_runbook",
            "",
            "",
            "incident response runbook",
            vec!["runbook_0".to_string()],
        )
        .unwrap();
    builder
        .add_source(
            "document_release",
            "",
            "",
            "quarterly release schedule",
            vec!["release_0".to_string()],
        )
        .unwrap();
    builder.write(&index_dir).unwrap();
    let index = crate::source_index::SourceLexicalIndex::open(&index_dir).unwrap();
    let options = HybridSourceDiverseOptions {
        candidate_k: 2,
        lexical_source_k: 3,
        max_sources: 2,
        max_spans_per_source: 1,
        neighbor_radius: 0,
        lexical_max_df_ratio: 1.0,
        expansion_weight: 0.5,
        supplemental_sources: 1,
        supplemental_spans_per_source: 1,
        supplemental_candidate_k: 0,
    };

    let primary_only = store
        .query_hybrid_source_diverse_spans(
            &scope(),
            HybridSourceDiverseQuery {
                query_embedding: vec![1.0, 0.0, 0.0],
                query_text: "incident response",
                expansions: &[],
                supplemental_queries: &[],
                index: &index,
                at_ms: Some(2),
            },
            options,
        )
        .unwrap();
    assert_eq!(primary_only.len(), 2);
    assert!(!primary_only
        .iter()
        .any(|hit| hit.span.source_id == "document_release"));

    let supplemental_disabled = store
        .query_hybrid_source_diverse_spans(
            &scope(),
            HybridSourceDiverseQuery {
                query_embedding: vec![1.0, 0.0, 0.0],
                query_text: "incident response",
                expansions: &[],
                supplemental_queries: &[ExpansionQuery {
                    text: "quarterly release schedule".to_string(),
                    embedding: None,
                }],
                index: &index,
                at_ms: Some(2),
            },
            HybridSourceDiverseOptions {
                supplemental_sources: 0,
                ..options
            },
        )
        .unwrap();
    assert_eq!(
        supplemental_disabled
            .iter()
            .map(|hit| hit.span.id.as_str())
            .collect::<Vec<_>>(),
        primary_only
            .iter()
            .map(|hit| hit.span.id.as_str())
            .collect::<Vec<_>>()
    );

    let with_supplemental = store
        .query_hybrid_source_diverse_spans(
            &scope(),
            HybridSourceDiverseQuery {
                query_embedding: vec![1.0, 0.0, 0.0],
                query_text: "incident response",
                expansions: &[],
                supplemental_queries: &[ExpansionQuery {
                    text: "quarterly release schedule".to_string(),
                    embedding: None,
                }],
                index: &index,
                at_ms: Some(2),
            },
            options,
        )
        .unwrap();
    assert_eq!(with_supplemental.len(), 3);
    assert_eq!(
        with_supplemental[..primary_only.len()]
            .iter()
            .map(|hit| hit.span.source_id.as_str())
            .collect::<Vec<_>>(),
        primary_only
            .iter()
            .map(|hit| hit.span.source_id.as_str())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        with_supplemental.last().unwrap().span.source_id,
        "document_release"
    );

    let _ = std::fs::remove_dir_all(&index_dir);
}

#[test]
fn memory_store_searches_matching_passages_inside_selected_sources() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("query_matched_passages");
    let store = crate::test_store(&dir, 3).unwrap();
    let index_dir = temp_dir("query_matched_passages_index");
    let mut builder = crate::source_index::SourceLexicalIndexBuilder::new();
    let mut dense_hits = Vec::new();
    for (source_id, texts) in [
        (
            "handbook",
            vec![
                "Office hours.",
                "Visitor sign-in.",
                "Travel reimbursements close Friday.",
            ],
        ),
        (
            "calendar",
            vec![
                "Team meetings.",
                "Office deliveries.",
                "Travel reimbursements open Monday.",
            ],
        ),
    ] {
        let mut ids = Vec::new();
        for (position, text) in texts.iter().enumerate() {
            let id = format!("{source_id}_{position}");
            let mut span = span_record(&id, MemoryStatus::Active, Some(1));
            span.source_id = source_id.to_string();
            span.span_index = position as i64;
            span.text = (*text).to_string();
            span.lexical_text = span.text.clone();
            store.append_span(&span, Some(&[1.0, 0.0, 0.0])).unwrap();
            if position == texts.len() - 1 {
                dense_hits.push(SpanSearchHit { span, score: 1.0 });
            }
            ids.push(id);
        }
        builder
            .add_source(source_id, "", "", &texts.join(" "), ids)
            .unwrap();
    }
    builder.write(&index_dir).unwrap();
    let index = crate::source_index::SourceLexicalIndex::open(&index_dir).unwrap();
    let selections = [
        SourceSpanSelection {
            source_id: "handbook".to_string(),
            max_spans: 1,
        },
        SourceSpanSelection {
            source_id: "calendar".to_string(),
            max_spans: 1,
        },
    ];
    let prefixes = store
        .hydrate_selected_sources(&scope(), &selections, &[], &index, Some(2))
        .unwrap();
    assert!(prefixes
        .iter()
        .all(|hit| !hit.span.text.contains("reimbursements")));

    for (query, prior) in [
        ("travel reimbursements", &[][..]),
        ("expense repayment deadlines", dense_hits.as_slice()),
        ("travel reimbursements", dense_hits.as_slice()),
    ] {
        let hits = store
            .search_selected_sources(&scope(), query, &selections, prior, &index, Some(2))
            .unwrap();
        assert_eq!(hits.len(), prefixes.len());
        assert_eq!(
            hits.iter()
                .map(|hit| hit.span.id.as_str())
                .collect::<Vec<_>>(),
            vec!["handbook_2", "calendar_2"]
        );
    }
    let other_scope = MemoryScope::new("other");
    assert!(store
        .search_selected_sources(
            &other_scope,
            "travel reimbursements",
            &selections,
            &dense_hits,
            &index,
            Some(2)
        )
        .unwrap()
        .is_empty());
    assert!(store
        .search_selected_sources(
            &scope(),
            "travel reimbursements",
            &selections,
            &dense_hits,
            &index,
            Some(0)
        )
        .unwrap()
        .is_empty());
    assert!(store
        .search_selected_sources(
            &scope(),
            "travel reimbursements",
            &[],
            &dense_hits,
            &index,
            Some(2)
        )
        .unwrap()
        .is_empty());
    let zero_budget = [SourceSpanSelection {
        source_id: "handbook".to_string(),
        max_spans: 0,
    }];
    assert!(store
        .search_selected_sources(
            &scope(),
            "travel reimbursements",
            &zero_budget,
            &dense_hits,
            &index,
            Some(2)
        )
        .unwrap()
        .is_empty());
    let _ = std::fs::remove_dir_all(&index_dir);
}

#[test]
fn memory_store_hydrates_explicit_source_selections_in_round_robin_order() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("explicit_source_hydration");
    let store = crate::test_store(&dir, 3).unwrap();

    for (source_id, span_count) in [
        ("document_alpha", 4),
        ("document_beta", 3),
        ("document_gamma", 2),
    ] {
        for span_index in 0..span_count {
            let id = format!("{source_id}_{span_index}");
            let mut span = span_record(&id, MemoryStatus::Active, Some(1));
            span.source_id = source_id.to_string();
            span.span_index = span_index;
            span.text = format!("{source_id} detail {span_index}");
            span.lexical_text = span.text.clone();
            store.append_span(&span, Some(&[1.0, 0.0, 0.0])).unwrap();
        }
    }

    let index_dir = temp_dir("explicit_source_hydration_index");
    let mut builder = crate::source_index::SourceLexicalIndexBuilder::new();
    for (source_id, span_count) in [
        ("document_alpha", 4),
        ("document_beta", 3),
        ("document_gamma", 2),
    ] {
        builder
            .add_source(
                source_id,
                "",
                "",
                source_id,
                (0..span_count)
                    .map(|span_index| format!("{source_id}_{span_index}"))
                    .collect(),
            )
            .unwrap();
    }
    builder.write(&index_dir).unwrap();
    let index = crate::source_index::SourceLexicalIndex::open(&index_dir).unwrap();

    let fetched = store
        .fetch_spans_by_ids(
            &scope(),
            &[
                "document_alpha_2".to_string(),
                "document_beta_1".to_string(),
                "document_alpha_0".to_string(),
            ],
            Some(2),
        )
        .unwrap()
        .into_iter()
        .map(|span| (span.id.clone(), span))
        .collect::<std::collections::BTreeMap<_, _>>();
    let prior_hits = [
        ("document_alpha_2", 0.9),
        ("document_beta_1", 0.8),
        ("document_alpha_2", 0.7),
        ("document_alpha_0", 0.6),
    ]
    .into_iter()
    .map(|(span_id, score)| SpanSearchHit {
        span: fetched[span_id].clone(),
        score,
    })
    .collect::<Vec<_>>();

    let hits = store
        .hydrate_selected_sources(
            &scope(),
            &[
                SourceSpanSelection {
                    source_id: "document_alpha".to_string(),
                    max_spans: 3,
                },
                SourceSpanSelection {
                    source_id: "document_gamma".to_string(),
                    max_spans: 2,
                },
                SourceSpanSelection {
                    source_id: "document_beta".to_string(),
                    max_spans: 2,
                },
            ],
            &prior_hits,
            &index,
            Some(2),
        )
        .unwrap();

    assert_eq!(
        hits.iter()
            .map(|hit| hit.span.id.as_str())
            .collect::<Vec<_>>(),
        vec![
            "document_alpha_2",
            "document_gamma_0",
            "document_beta_1",
            "document_alpha_0",
            "document_gamma_1",
            "document_beta_2",
            "document_alpha_3",
        ]
    );

    let _ = std::fs::remove_dir_all(&index_dir);
}

#[test]
fn memory_store_scan_span_limit_applies_after_scope_filtering() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("span_scan_limit_after_filtering");
    let store = crate::test_store(&dir, 3).unwrap();
    let mut target_scope = scope();
    target_scope.user_id = Some("target_user".to_string());
    let mut other_scope = scope();
    other_scope.user_id = Some("other_user".to_string());

    for i in 0..MAX_VECTOR_QUERY_TOPK {
        let mut span = span_record(&format!("span_other_{i}"), MemoryStatus::Active, Some(1));
        span.scope = other_scope.clone();
        store.append_span(&span, Some(&[1.0, 0.0, 0.0])).unwrap();
    }
    let mut target = span_record("span_target_proof", MemoryStatus::Active, Some(1));
    target.scope = target_scope.clone();
    store.append_span(&target, Some(&[1.0, 0.0, 0.0])).unwrap();

    let spans = store.scan_spans(&target_scope, 1, Some(2)).unwrap();
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].id, "span_target_proof");
}

#[test]
fn memory_store_queries_persisted_spans_and_applies_corrections() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("query_spans");
    let store = crate::test_store(&dir, 3).unwrap();
    let mut query_scope = scope();
    query_scope.user_id = Some("user_1".to_string());
    let mut other_scope = scope();
    other_scope.user_id = Some("user_2".to_string());
    let mut active = span_record("active", MemoryStatus::Active, Some(1));
    active.scope = query_scope.clone();
    let mut future = span_record("future", MemoryStatus::Active, Some(100));
    future.scope = query_scope.clone();
    let mut tombstoned = span_record("gone", MemoryStatus::Tombstoned, Some(1));
    tombstoned.scope = query_scope.clone();
    let mut other_user = span_record("other", MemoryStatus::Active, Some(1));
    other_user.scope = other_scope.clone();
    store.append_span(&active, Some(&[1.0, 0.0, 0.0])).unwrap();
    store.append_span(&future, Some(&[1.0, 0.0, 0.0])).unwrap();
    store
        .append_span(&tombstoned, Some(&[1.0, 0.0, 0.0]))
        .unwrap();
    store
        .append_span(&other_user, Some(&[1.0, 0.0, 0.0]))
        .unwrap();
    drop(store);

    let reopened = crate::reopen_test_store(&dir).unwrap();
    let hits = reopened
        .query_spans(&query_scope, vec![1.0, 0.0, 0.0], 10, Some(10))
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].span.id, "active");

    let keyword_hits = reopened
        .keyword_search_spans(&query_scope, "active", 10, 100, Some(10))
        .unwrap();
    assert_eq!(keyword_hits.len(), 1);
    assert_eq!(keyword_hits[0].span.id, "active");

    let hybrid_hits = reopened
        .hybrid_search_spans(
            &query_scope,
            HybridSpanSearch {
                query_embedding: vec![1.0, 0.0, 0.0],
                query_text: "active",
                k: 10,
                scan_limit: 100,
                at_ms: Some(10),
            },
        )
        .unwrap()
        .fused_hits;
    assert_eq!(hybrid_hits.len(), 1);
    assert_eq!(hybrid_hits[0].span.id, "active");

    let recall_20_hits = reopened
        .query_spans(&query_scope, vec![1.0, 0.0, 0.0], 20, Some(10))
        .unwrap();
    assert_eq!(recall_20_hits.len(), 1);
    assert_eq!(recall_20_hits[0].span.id, "active");

    let correction = crate::ingest::add_correction(
        CorrectionInput {
            id: Some("corr_query".to_string()),
            scope: query_scope.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            operation: CorrectionOperation::Tombstone,
            target_type: "span".to_string(),
            target_ids: vec!["active".to_string()],
            target_selector: None,
            new_value: None,
            reason: None,
            actor: ActorKind::User,
            authority: CorrectionAuthority::User,
            effective_at_ms: None,
            applies_valid_from_ms: None,
            applies_valid_to_ms: None,
            cascade_policy: None,
            metadata_json: None,
        },
        12,
    );
    for i in 0..SATURATING_CORRECTION_COUNT {
        let irrelevant = crate::ingest::add_correction(
            CorrectionInput {
                id: Some(format!("corr_irrelevant_{i}")),
                scope: other_scope.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                operation: CorrectionOperation::Tombstone,
                target_type: "span".to_string(),
                target_ids: vec!["active".to_string()],
                target_selector: None,
                new_value: None,
                reason: None,
                actor: ActorKind::User,
                authority: CorrectionAuthority::User,
                effective_at_ms: None,
                applies_valid_from_ms: None,
                applies_valid_to_ms: None,
                cascade_policy: None,
                metadata_json: None,
            },
            12,
        );
        reopened.add_correction(&irrelevant).unwrap();
    }
    reopened.add_correction(&correction).unwrap();
    let historical_hits = reopened
        .query_spans(&query_scope, vec![1.0, 0.0, 0.0], 10, Some(10))
        .unwrap();
    assert_eq!(historical_hits.len(), 1);

    let hits = reopened
        .query_spans(&query_scope, vec![1.0, 0.0, 0.0], 10, Some(12))
        .unwrap();
    assert!(hits.is_empty());

    let hits = reopened
        .query_spans_with_corrections(
            &query_scope,
            vec![1.0, 0.0, 0.0],
            10,
            Some(12),
            &[correction],
        )
        .unwrap();
    assert!(hits.is_empty());
}

#[test]
fn memory_store_vector_span_limit_applies_after_scope_filtering() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("span_vector_limit_after_filtering");
    let store = crate::test_store(&dir, 3).unwrap();
    let mut target_scope = scope();
    target_scope.user_id = Some("target_user".to_string());
    let mut other_scope = scope();
    other_scope.user_id = Some("other_user".to_string());

    for i in 0..MAX_VECTOR_QUERY_TOPK {
        let mut span = span_record(&format!("span_other_{i}"), MemoryStatus::Active, Some(1));
        span.scope = other_scope.clone();
        store.append_span(&span, Some(&[1.0, 0.0, 0.0])).unwrap();
    }
    let mut target = span_record("span_target_vector", MemoryStatus::Active, Some(1));
    target.scope = target_scope.clone();
    store.append_span(&target, Some(&[1.0, 0.0, 0.0])).unwrap();

    let hits = store
        .query_spans(&target_scope, vec![1.0, 0.0, 0.0], 1, Some(2))
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].span.id, "span_target_vector");
}

#[test]
fn memory_store_span_selector_tombstone_survives_future_correction_saturation() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let dir = temp_dir("span_selector_future_saturation");
    let store = crate::test_store(&dir, 3).unwrap();
    let mut target_scope = scope();
    target_scope.user_id = Some("target_user".to_string());
    let mut target = span_record("span_target_selector", MemoryStatus::Active, Some(1));
    target.scope = target_scope.clone();
    target.text = "Runbook token SELECTOR-TOMBSTONE.".to_string();
    target.lexical_text = target.text.clone();
    store.append_span(&target, Some(&[1.0, 0.0, 0.0])).unwrap();

    for i in 0..SATURATING_CORRECTION_COUNT {
        let correction = crate::ingest::add_correction(
            CorrectionInput {
                id: Some(format!("corr_future_selector_{i}")),
                scope: target_scope.clone(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                operation: CorrectionOperation::Tombstone,
                target_type: "span".to_string(),
                target_ids: Vec::new(),
                target_selector: Some("SELECTOR-TOMBSTONE".to_string()),
                new_value: None,
                reason: None,
                actor: ActorKind::User,
                authority: CorrectionAuthority::User,
                effective_at_ms: Some(100),
                applies_valid_from_ms: None,
                applies_valid_to_ms: None,
                cascade_policy: None,
                metadata_json: None,
            },
            100,
        );
        store.add_correction(&correction).unwrap();
    }
    let tombstone = crate::ingest::add_correction(
        CorrectionInput {
            id: Some("corr_target_selector_deleted".to_string()),
            scope: target_scope.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            operation: CorrectionOperation::Tombstone,
            target_type: "span".to_string(),
            target_ids: Vec::new(),
            target_selector: Some("SELECTOR-TOMBSTONE".to_string()),
            new_value: None,
            reason: None,
            actor: ActorKind::User,
            authority: CorrectionAuthority::User,
            effective_at_ms: Some(12),
            applies_valid_from_ms: None,
            applies_valid_to_ms: None,
            cascade_policy: None,
            metadata_json: None,
        },
        12,
    );
    store.add_correction(&tombstone).unwrap();

    let historical = store
        .query_spans(&target_scope, vec![1.0, 0.0, 0.0], 10, Some(10))
        .unwrap();
    assert_eq!(historical.len(), 1);
    let current = store
        .query_spans(&target_scope, vec![1.0, 0.0, 0.0], 10, Some(12))
        .unwrap();
    assert!(current.is_empty());
}

#[test]
fn evidence_completion_skips_forgotten_neighbor_text() {
    // Evidence expansion must honor the same Forget correction as direct retrieval.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("forgotten_neighbor");
    let store = crate::test_store(&path, 3).unwrap();
    let ingested = store
        .ingest_episode(
            EpisodeInput {
                id: Some("episode".to_string()),
                scope: scope(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                source_kind: SourceKind::UserMessage,
                actor: ActorKind::User,
                sequence_no: 1,
                event_time_ms: Some(10),
                valid_from_ms: Some(10),
                valid_to_ms: None,
                raw_text: "current obsolete".to_string(),
                blob_ref: None,
                mime_type: None,
                causal_parent_ids: Vec::new(),
                metadata_json: None,
            },
            10,
            &ChunkOptions {
                max_chars: 8,
                overlap_chars: 0,
                chunker_version: "test".to_string(),
            },
        )
        .unwrap();
    assert_eq!(ingested.spans.len(), 2);
    let forgotten_id = ingested.spans[1].id.clone();
    let correction = crate::ingest::add_correction(
        CorrectionInput {
            id: Some("forget_neighbor".to_string()),
            scope: scope(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            operation: CorrectionOperation::Forget,
            target_type: "span".to_string(),
            target_ids: vec![forgotten_id.clone()],
            target_selector: None,
            new_value: None,
            reason: None,
            actor: ActorKind::User,
            authority: CorrectionAuthority::User,
            effective_at_ms: Some(20),
            applies_valid_from_ms: None,
            applies_valid_to_ms: None,
            cascade_policy: None,
            metadata_json: None,
        },
        20,
    );
    store.add_correction(&correction).unwrap();
    drop(store);
    let reopened = crate::reopen_test_store(&path).unwrap();
    let direct = reopened
        .keyword_search_spans(&scope(), "obsolete", 10, 100, Some(30))
        .unwrap();
    let anchors = reopened
        .keyword_search_spans(&scope(), "current", 10, 100, Some(30))
        .unwrap();
    let expanded = reopened
        .complete_span_evidence(&scope(), &anchors, 0, 1, 100, Some(30))
        .unwrap();
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);
    assert!(direct.is_empty());
    assert_eq!(anchors.len(), 1);
    assert!(
        expanded.iter().all(|hit| hit.span.id != forgotten_id),
        "the forgotten span reappeared in completed evidence"
    );
}

#[test]
fn selected_source_hydration_skips_forgotten_span() {
    // Source hydration must honor Forget corrections just as direct vector retrieval does.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("forgotten_source");
    let index_path = temp_dir("forgotten_source_index");
    let store = crate::test_store(&path, 3).unwrap();
    let span = span_record("forgotten", MemoryStatus::Active, Some(10));
    store
        .append_vector_spans(&[(span.clone(), vec![1.0, 0.0, 0.0])])
        .unwrap();
    let mut builder = crate::source_index::SourceLexicalIndexBuilder::new();
    builder
        .add_source(&span.source_id, "", "", &span.text, vec![span.id.clone()])
        .unwrap();
    builder.write(&index_path).unwrap();
    let correction = crate::ingest::add_correction(
        CorrectionInput {
            id: Some("forget_source_span".to_string()),
            scope: scope(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            operation: CorrectionOperation::Forget,
            target_type: "span".to_string(),
            target_ids: vec![span.id.clone()],
            target_selector: None,
            new_value: None,
            reason: None,
            actor: ActorKind::User,
            authority: CorrectionAuthority::User,
            effective_at_ms: Some(20),
            applies_valid_from_ms: None,
            applies_valid_to_ms: None,
            cascade_policy: None,
            metadata_json: None,
        },
        20,
    );
    store.add_correction(&correction).unwrap();
    drop(store);
    let reopened = crate::reopen_test_store(&path).unwrap();
    let direct = reopened
        .query_spans(&scope(), vec![1.0, 0.0, 0.0], 10, Some(30))
        .unwrap();
    let index = crate::source_index::SourceLexicalIndex::open(&index_path).unwrap();
    let hydrated = reopened
        .hydrate_selected_sources(
            &scope(),
            &[SourceSpanSelection {
                source_id: span.source_id,
                max_spans: 1,
            }],
            &[],
            &index,
            Some(30),
        )
        .unwrap();
    drop(index);
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
    std::fs::remove_dir_all(index_path).unwrap();
    drop(guard);
    assert!(direct.is_empty());
    assert!(
        hydrated.is_empty(),
        "source hydration returned forgotten text: {hydrated:?}"
    );
}

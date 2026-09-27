use super::*;

#[test]
fn build_context_dedupes_and_respects_budget() {
    let hits = vec![
        hit("a", 1.0, "alpha beta gamma"),
        hit("a", 0.9, "duplicate"),
        hit("b", 0.8, "delta epsilon zeta"),
    ];
    let compiled = build_context(
        &hits,
        ContextOptions {
            token_budget: 45,
            include_provenance: false,
        },
    );
    assert_eq!(compiled.included.len(), 2);
    assert_eq!(compiled.included[0].id, "a");
    assert_eq!(compiled.included[0].kind, "span");
    assert_eq!(compiled.included[1].id, "b");
    assert_eq!(compiled.included[1].kind, "span");
    assert!(!compiled.body.contains("duplicate"));
}

#[test]
fn build_context_clips_first_item_when_budget_is_tiny() {
    let hits = vec![hit("a", 1.0, "one two three four five six")];
    let compiled = build_context(
        &hits,
        ContextOptions {
            token_budget: 3,
            include_provenance: false,
        },
    );
    assert!(compiled.truncated);
    assert_eq!(compiled.included.len(), 1);
    assert_eq!(compiled.included[0].kind, "span");
    assert!(compiled.included[0].source_text.is_none());
    assert_eq!(
        compiled.support.state,
        crate::AnswerSupportState::NoEvidence
    );
    assert!(compiled.estimated_tokens <= 3);
}

#[test]
fn build_context_preserves_rank_order_when_all_hits_fit() {
    let hits = vec![
        hit(
            "long",
            1.0,
            "one two three four five six seven eight nine ten",
        ),
        hit("short", 0.8, "needle"),
    ];
    let compiled = build_context(
        &hits,
        ContextOptions {
            token_budget: 100,
            include_provenance: false,
        },
    );
    assert_eq!(compiled.included[0].id, "long");
    assert_eq!(compiled.included[1].id, "short");
    assert_eq!(compiled.included[1].source_text.as_deref(), Some("needle"));
    assert_eq!(compiled.support.state, crate::AnswerSupportState::Supported);
    assert!(!compiled.truncated);
}

#[test]
fn supplemental_slot_history_does_not_reorder_raw_evidence() {
    let history = SlotHistoryRecord {
        id: "history_project_owner".to_string(),
        slot_id: "slot_project_owner".to_string(),
        subject_entity_id: None,
        subject: Some("proyecto".to_string()),
        predicate: Some("responsable".to_string()),
        versions: vec![
            SlotHistoryVersion {
                state_id: "state_owner_ana".to_string(),
                state_kind: StateRecordKind::Current,
                object_value: Some("Ana".to_string()),
                members: Vec::new(),
                claim_ids: vec!["claim_owner_ana".to_string()],
                correction_ids: Vec::new(),
                rule_ids: Vec::new(),
                source_span_ids: vec!["span_not_retrieved".to_string()],
                source_episode_ids: vec!["episode_old".to_string()],
                observed_at_ms: 10,
                valid_from_ms: Some(10),
                valid_to_ms: Some(20),
            },
            SlotHistoryVersion {
                state_id: "state_owner_bea".to_string(),
                state_kind: StateRecordKind::Current,
                object_value: Some("Bea".to_string()),
                members: Vec::new(),
                claim_ids: vec!["claim_owner_bea".to_string()],
                correction_ids: Vec::new(),
                rule_ids: Vec::new(),
                source_span_ids: vec!["span_later".to_string()],
                source_episode_ids: vec!["episode_current".to_string()],
                observed_at_ms: 20,
                valid_from_ms: Some(20),
                valid_to_ms: None,
            },
        ],
    };
    let hits = vec![
        hit_at("span_earlier", 0.8, "Evidencia anterior.", 10),
        hit_at("span_later", 0.9, "Evidencia posterior.", 20),
    ];

    let compiled = build_query_state_context(
        &ContextInput {
            slot_histories: &[history],
            hits: &hits,
            ..Default::default()
        },
        ContextOptions {
            token_budget: 500,
            include_provenance: false,
        },
    );

    assert_eq!(
        compiled
            .included
            .iter()
            .map(|item| item.id.as_str())
            .collect::<Vec<_>>(),
        vec!["span_earlier", "span_later", "history_project_owner"]
    );
}

#[test]
fn build_typed_context_packs_state_before_evidence() {
    let profile = ProfileRecord {
        id: "profile_1".to_string(),
        scope: MemoryScope::new("default"),
        status: MemoryStatus::Active,
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        subject_id: Some("user".to_string()),
        profile_key: "style".to_string(),
        profile_text: "Use exact IDs.".to_string(),
        value_json: None,
        evidence_claim_ids: Vec::new(),
        source_span_ids: Vec::new(),
        generated_at_ms: 10,
        generator_version: "manual".to_string(),
        valid_from_ms: Some(10),
        valid_to_ms: None,
        correction_watermark: 0,
    };
    let claim = ClaimRecord {
        id: "claim_1".to_string(),
        scope: MemoryScope::new("default"),
        status: MemoryStatus::Active,
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        claim_text: "Ticket ABC-123 matters.".to_string(),
        subject: Some("ticket".to_string()),
        predicate: Some("matters".to_string()),
        object_value: Some("ABC-123".to_string()),
        subject_entity_id: None,
        slot_id: None,
        slot_facet: None,
        claim_kind: ClaimKind::Fact,
        polarity: ClaimPolarity::Affirmative,
        source_span_ids: Vec::new(),
        source_episode_ids: Vec::new(),
        source_sequence_no: None,
        asserted_by: "user".to_string(),
        extractor_version: None,
        confidence: Some(1.0),
        observed_at_ms: 11,
        valid_from_ms: Some(11),
        valid_to_ms: None,
        correction_ids: Vec::new(),
    };
    let artifact = ArtifactRecord {
        id: "artifact_1".to_string(),
        scope: MemoryScope::new("default"),
        status: MemoryStatus::Active,
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        artifact_kind: ArtifactKind::File,
        title: "runbook.md".to_string(),
        uri: Some("file:///runbook.md".to_string()),
        blob_ref: None,
        mime_type: Some("text/markdown".to_string()),
        content_hash: "hash".to_string(),
        source_created_at_ms: None,
        source_modified_at_ms: None,
        created_at_ms: 12,
        ingested_at_ms: 12,
        valid_from_ms: Some(12),
        valid_to_ms: None,
        extracted_text_ref: Some("span_1".to_string()),
        metadata_json: None,
    };
    let compiled = build_state_context(
        &ContextInput {
            profiles: &[profile],
            claims: &[claim],
            artifacts: &[artifact],
            hits: &[hit("span_1", 0.9, "Raw evidence.")],
            ..Default::default()
        },
        ContextOptions {
            token_budget: 200,
            include_provenance: false,
        },
    );
    assert!(
        compiled.body.find("value: ABC-123").unwrap()
            < compiled.body.find("Use exact IDs").unwrap()
    );
    assert!(
        compiled.body.find("runbook.md").unwrap() < compiled.body.find("Raw evidence").unwrap()
    );
    assert_eq!(compiled.included.len(), 4);
    assert_eq!(compiled.included[0].kind, "state_current");
    assert_eq!(compiled.included[1].kind, "profile");
    assert_eq!(compiled.included[2].kind, "artifact");
    assert_eq!(compiled.included[3].kind, "span");
}

#[test]
fn sourced_state_without_retrieved_evidence_does_not_crowd_out_hits() {
    let relevant = ClaimRecord {
        id: "claim_relevant".to_string(),
        scope: MemoryScope::new("default"),
        status: MemoryStatus::Active,
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        claim_text: "Project status is green.".to_string(),
        subject: Some("project".to_string()),
        predicate: Some("status".to_string()),
        object_value: Some("green".to_string()),
        subject_entity_id: None,
        slot_id: None,
        slot_facet: None,
        claim_kind: ClaimKind::Fact,
        polarity: ClaimPolarity::Affirmative,
        source_span_ids: vec!["span_hit".to_string()],
        source_episode_ids: vec!["episode_project".to_string()],
        source_sequence_no: None,
        asserted_by: "extractor".to_string(),
        extractor_version: None,
        confidence: Some(1.0),
        observed_at_ms: 10,
        valid_from_ms: Some(10),
        valid_to_ms: None,
        correction_ids: Vec::new(),
    };
    let mut unrelated = relevant.clone();
    unrelated.id = "claim_unrelated".to_string();
    unrelated.claim_text =
        "Unrelated preferences about office snacks, meeting rooms, and parking passes.".to_string();
    unrelated.subject = Some("office".to_string());
    unrelated.predicate = Some("preferences".to_string());
    unrelated.object_value = Some("snacks rooms parking".to_string());
    unrelated.source_span_ids = vec!["span_other".to_string()];
    unrelated.source_episode_ids = vec!["episode_office".to_string()];

    let compiled = build_state_context(
        &ContextInput {
            claims: &[relevant, unrelated],
            hits: &[hit("span_hit", 0.9, "Raw project evidence.")],
            ..Default::default()
        },
        ContextOptions {
            token_budget: 32,
            include_provenance: false,
        },
    );

    let ids = compiled
        .included
        .iter()
        .map(|item| item.id.as_str())
        .collect::<Vec<_>>();
    assert_eq!(ids, vec!["claim_relevant", "span_hit"]);
    assert!(!compiled.truncated);
    assert!(!compiled.body.contains("office snacks"));
}

#[test]
fn query_state_context_admits_query_selected_sourced_state() {
    let claim = ClaimRecord {
        id: "claim_project_status".to_string(),
        scope: MemoryScope::new("default"),
        status: MemoryStatus::Active,
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        claim_text: "Project status is green.".to_string(),
        subject: Some("project".to_string()),
        predicate: Some("status".to_string()),
        object_value: Some("green".to_string()),
        subject_entity_id: None,
        slot_id: None,
        slot_facet: None,
        claim_kind: ClaimKind::Fact,
        polarity: ClaimPolarity::Affirmative,
        source_span_ids: vec!["span_source".to_string()],
        source_episode_ids: vec!["episode_project".to_string()],
        source_sequence_no: None,
        asserted_by: "extractor".to_string(),
        extractor_version: None,
        confidence: Some(1.0),
        observed_at_ms: 10,
        valid_from_ms: Some(10),
        valid_to_ms: None,
        correction_ids: Vec::new(),
    };

    let compiled = build_query_state_context(
        &ContextInput {
            claims: &[claim],
            hits: &[hit("span_hit", 0.9, "Raw project planning notes.")],
            ..Default::default()
        },
        ContextOptions {
            token_budget: 100,
            include_provenance: false,
        },
    );

    let ids = compiled
        .included
        .iter()
        .map(|item| item.id.as_str())
        .collect::<Vec<_>>();
    assert_eq!(ids, vec!["claim_project_status", "span_hit"]);
    assert!(compiled.body.contains("slot: project / status"));
    assert!(compiled.body.contains("value: green"));
    assert!(!compiled
        .body
        .contains("EntityStateCard entity_state:project"));
}

#[test]
fn compact_specific_state_avoids_entity_card_fallback() {
    let small = claim(
        "claim_project_status",
        "project",
        "status",
        Some("green"),
        "Project status is green.",
        10,
    );
    let huge_text = "Dashboard notes. ".repeat(400);
    let huge = claim(
        "claim_dashboard_owner",
        "dashboard",
        "owner",
        Some("Dana"),
        &huge_text,
        20,
    );

    let compiled = build_query_state_context(
        &ContextInput {
            claims: &[small, huge],
            ..Default::default()
        },
        ContextOptions {
            token_budget: 140,
            include_provenance: false,
        },
    );

    let ids = compiled
        .included
        .iter()
        .map(|item| item.id.as_str())
        .collect::<Vec<_>>();
    assert!(ids.contains(&"claim_project_status"));
    assert!(!ids.contains(&"entity_state:project"));
    assert!(ids.contains(&"claim_dashboard_owner"));
    assert!(!ids.contains(&"entity_state:dashboard"));
}

#[test]
fn ambiguous_entity_alias_collision_fails_closed() {
    let aliases = entity_alias_map(&[
        entity("entity_alpha", "Alpha project", &["shared project"]),
        entity("entity_beta", "Beta project", &["shared project"]),
    ]);

    assert!(!aliases.contains_key("shared project"));
    assert_eq!(
        aliases.get("alpha project").map(|(key, _)| key.as_str()),
        Some("alpha project")
    );
    assert_eq!(
        aliases.get("beta project").map(|(key, _)| key.as_str()),
        Some("beta project")
    );
}

#[test]
fn coverage_state_leaves_half_of_the_budget_to_retrieved_evidence() {
    let mut claims = Vec::new();
    let mut hits = Vec::new();
    for index in 0..12 {
        let span_id = format!("span_estado_{index}");
        let mut record = claim(
            &format!("claim_estado_{index}"),
            "equipo",
            &format!("atributo_{index}"),
            Some("valor largo con muchas palabras descriptivas adicionales para ocupar presupuesto"),
            "El equipo mantiene un atributo con un valor largo y descriptivo que ocupa bastante presupuesto de tokens.",
            10 + index,
        );
        record.source_span_ids = vec![span_id.clone()];
        record.source_episode_ids = vec!["episodio_equipo".to_string()];
        claims.push(record);
        hits.push(hit(
            &span_id,
            1.0 - index as f32 * 0.01,
            "Evidencia cruda del equipo con suficiente texto para contar como un fragmento real.",
        ));
    }
    for index in 0..12 {
        hits.push(hit(
            &format!("span_extra_{index}"),
            0.5 - index as f32 * 0.01,
            "Otro fragmento recuperado que podría contener el dato buscado por la pregunta.",
        ));
    }
    let budget = 600;
    let projection = AnswerReadyStateProjection {
        claims,
        set_states: Vec::new(),
        slot_histories: Vec::new(),
        rules: Vec::new(),
        rule_outcomes: Vec::new(),
        corrections: Vec::new(),
        entities: Vec::new(),
        proof_span_ids: Vec::new(),
        support: AnswerSupportContract::default(),
    };
    let compiled = build_answer_ready_state_context(
        &[],
        &projection,
        &[],
        &hits,
        ContextOptions {
            token_budget: budget,
            include_provenance: false,
        },
    );
    let span_tokens = compiled
        .included
        .iter()
        .filter(|item| item.kind == "span")
        .map(|item| item.estimated_tokens)
        .sum::<usize>();
    let state_tokens = compiled
        .included
        .iter()
        .filter(|item| item.kind != "span")
        .map(|item| item.estimated_tokens)
        .sum::<usize>();
    assert!(compiled.truncated);
    assert!(state_tokens > 0);
    assert!(
        span_tokens * 10 >= budget * 4,
        "evidence got {span_tokens} of {budget} tokens (state {state_tokens})"
    );
}

#[test]
fn coverage_after_an_answer_target_keeps_half_of_the_remaining_budget_for_evidence() {
    let mut target = claim(
        "claim_objetivo",
        "proyecto",
        "responsable",
        Some("Ana"),
        "La responsable del proyecto es Ana.",
        10,
    );
    target.slot_id = Some("slot_proyecto_responsable".to_string());
    target.source_span_ids = vec!["span_objetivo".to_string()];
    let mut claims = vec![target];
    let mut hits = vec![hit(
        "span_objetivo",
        1.0,
        "Ana es la responsable del proyecto.",
    )];
    for index in 0..12 {
        let span_id = format!("span_cobertura_{index}");
        let mut record = claim(
            &format!("claim_cobertura_{index}"),
            "equipo",
            &format!("atributo_{index}"),
            Some("valor largo con muchas palabras descriptivas adicionales para ocupar presupuesto"),
            "El equipo mantiene un atributo con un valor largo y descriptivo que ocupa bastante presupuesto de tokens.",
            10 + index,
        );
        record.slot_id = Some(format!("slot_cobertura_{index}"));
        record.source_span_ids = vec![span_id.clone()];
        claims.push(record);
        hits.push(hit(
            &span_id,
            0.9 - index as f32 * 0.01,
            "Evidencia cruda del equipo con suficiente texto para contar como un fragmento real.",
        ));
    }
    for index in 0..12 {
        hits.push(hit(
            &format!("span_extra_{index}"),
            0.5 - index as f32 * 0.01,
            "Otro fragmento recuperado que podría contener el dato buscado por la pregunta.",
        ));
    }
    let support = AnswerSupportContract {
        state: crate::AnswerSupportState::Supported,
        source_claim_ids: vec!["claim_objetivo".to_string()],
        source_span_ids: vec!["span_objetivo".to_string()],
        slots: vec![crate::AnswerSlotSupport {
            slot_id: "slot_proyecto_responsable".to_string(),
            subject: Some("proyecto".to_string()),
            predicate: Some("responsable".to_string()),
            view: crate::StateReadView::Current,
            state: crate::AnswerSupportState::Supported,
            source_claim_ids: vec!["claim_objetivo".to_string()],
            source_span_ids: vec!["span_objetivo".to_string()],
        }],
    };
    let budget = 700;
    let projection = AnswerReadyStateProjection {
        claims,
        set_states: Vec::new(),
        slot_histories: Vec::new(),
        rules: Vec::new(),
        rule_outcomes: Vec::new(),
        corrections: Vec::new(),
        entities: Vec::new(),
        proof_span_ids: vec!["span_objetivo".to_string()],
        support,
    };
    let compiled = build_answer_ready_state_context(
        &[],
        &projection,
        &[],
        &hits,
        ContextOptions {
            token_budget: budget,
            include_provenance: false,
        },
    );
    assert!(compiled
        .body
        .contains("### ResolvedAnswerState slot_proyecto_responsable"));
    let span_tokens = compiled
        .included
        .iter()
        .filter(|item| item.kind == "span")
        .map(|item| item.estimated_tokens)
        .sum::<usize>();
    let coverage_tokens = compiled
        .included
        .iter()
        .filter(|item| item.kind == "state_current")
        .map(|item| item.estimated_tokens)
        .sum::<usize>();
    assert!(coverage_tokens > 0);
    assert!(
        span_tokens >= coverage_tokens,
        "evidence {span_tokens} vs coverage {coverage_tokens} of {budget}"
    );
}

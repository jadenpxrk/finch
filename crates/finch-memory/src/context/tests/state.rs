use super::*;

#[test]
fn current_question_packs_current_state_before_stale_raw_evidence() {
    let current_owner = claim(
        "claim_current_owner",
        "project",
        "owner",
        Some("Dana"),
        "Project owner is Dana.",
        20,
    );

    let compiled = build_query_state_context(
        &ContextInput {
            claims: &[current_owner],
            hits: &[
                hit_at("span_old_owner", 0.9, "Project owner used to be Eli.", 10),
                hit_at("span_new_owner", 0.8, "Project owner is Dana.", 20),
            ],
            ..Default::default()
        },
        ContextOptions {
            token_budget: 200,
            include_provenance: false,
        },
    );

    let ids = compiled
        .included
        .iter()
        .map(|item| item.id.as_str())
        .collect::<Vec<_>>();
    assert_eq!(ids[0], "claim_current_owner");
    assert_eq!(ids[1], "span_old_owner");
    assert!(!compiled
        .body
        .contains("EntityStateCard entity_state:project"));
    assert!(
        compiled.body.find("value: Dana").unwrap()
            < compiled.body.find("Project owner used to be Eli.").unwrap()
    );
}

#[test]
fn historical_raw_owner_evidence_is_preserved_chronologically_with_state() {
    let current_owner = claim(
        "claim_current_owner",
        "project",
        "owner",
        Some("Dana"),
        "Project owner is Dana.",
        20,
    );

    let compiled = build_query_state_context(
        &ContextInput {
            claims: &[current_owner],
            hits: &[
                hit_at("span_new_owner", 0.9, "Project owner changed to Dana.", 20),
                hit_at(
                    "span_old_owner",
                    0.8,
                    "Project owner was Eli before launch.",
                    10,
                ),
            ],
            ..Default::default()
        },
        ContextOptions {
            token_budget: 200,
            include_provenance: false,
        },
    );

    let ids = compiled
        .included
        .iter()
        .map(|item| item.id.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        ids,
        vec!["claim_current_owner", "span_old_owner", "span_new_owner"]
    );
    assert!(
        compiled.body.find("Eli before launch").unwrap()
            < compiled.body.find("changed to Dana").unwrap()
    );
}

#[test]
fn current_deletion_question_packs_tombstone_before_stale_raw_evidence() {
    let mut deleted = claim(
        "claim_deleted_owner",
        "project",
        "owner",
        Some("Eli"),
        "Project owner was deleted.",
        20,
    );
    deleted.status = MemoryStatus::Tombstoned;

    let compiled = build_query_state_context(
        &ContextInput {
            claims: &[deleted],
            hits: &[hit_at("span_stale_owner", 0.9, "Project owner is Eli.", 10)],
            ..Default::default()
        },
        ContextOptions {
            token_budget: 200,
            include_provenance: true,
        },
    );

    let ids = compiled
        .included
        .iter()
        .map(|item| item.id.as_str())
        .collect::<Vec<_>>();
    assert_eq!(ids[0], "claim_deleted_owner");
    assert_eq!(ids[1], "span_stale_owner");
    assert!(compiled
        .body
        .contains("### TombstoneState claim_deleted_owner"));
    assert!(compiled.body.contains("support_state: deleted"));
    assert!(compiled.body.contains("current_value: null"));
    assert!(compiled
        .body
        .contains("historical_evidence_is_current: false"));
    assert!(!compiled
        .body
        .contains("EntityStateCard entity_state:project"));
    assert!(compiled.body.find("TombstoneState").unwrap() < compiled.body.find("Eli").unwrap());
}

#[test]
fn retracted_state_renders_as_deleted_current_state() {
    let mut retracted = claim(
        "claim_retracted_owner",
        "project",
        "owner",
        Some("Eli"),
        "Project owner was retracted.",
        20,
    );
    retracted.status = MemoryStatus::Retracted;

    let compiled = build_query_state_context(
        &ContextInput {
            claims: &[retracted],
            ..Default::default()
        },
        ContextOptions {
            token_budget: 100,
            include_provenance: false,
        },
    );

    assert!(compiled
        .body
        .contains("### TombstoneState claim_retracted_owner"));
    assert!(compiled.body.contains("support_state: deleted"));
    assert!(compiled.body.contains("current_value: null"));
    assert!(!compiled.body.contains("value: Eli"));
}

#[test]
fn list_question_preserves_membership_evidence_with_set_state() {
    let mut membership = claim(
        "claim_project_member",
        "project",
        "member",
        Some("Dana"),
        "Dana is an active project member.",
        20,
    );
    membership.claim_kind = ClaimKind::Relationship;

    let compiled = build_query_state_context(
        &ContextInput {
            claims: &[membership],
            hits: &[
                hit_at("span_member_dana", 0.8, "Dana joined the project.", 20),
                hit_at("span_member_eli", 0.9, "Eli joined the project.", 10),
            ],
            ..Default::default()
        },
        ContextOptions {
            token_budget: 200,
            include_provenance: false,
        },
    );

    let ids = compiled
        .included
        .iter()
        .map(|item| item.id.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        ids,
        vec![
            "set_state:Relationship:project:member",
            "span_member_eli",
            "span_member_dana",
        ]
    );
    assert!(compiled.body.find("Eli joined").unwrap() < compiled.body.find("Dana joined").unwrap());
    assert!(compiled
        .body
        .contains("### SetState set_state:Relationship:project:member"));
    assert!(!compiled
        .body
        .contains("EntityStateCard entity_state:project"));
}

#[test]
fn set_state_groups_active_members_and_preserves_raw_support() {
    let mut dana = claim(
        "claim_member_dana",
        "project",
        "member",
        Some("Dana"),
        "Dana joined the project.",
        10,
    );
    dana.claim_kind = ClaimKind::Relationship;
    dana.source_span_ids = vec!["span_dana".to_string()];
    let mut eli = claim(
        "claim_member_eli",
        "project",
        "member",
        Some("Eli"),
        "Eli joined the project.",
        20,
    );
    eli.claim_kind = ClaimKind::Relationship;
    eli.source_span_ids = vec!["span_eli".to_string()];

    let compiled = build_query_state_context(
        &ContextInput {
            claims: &[dana, eli],
            hits: &[
                hit_at("span_eli", 0.8, "Eli joined the project.", 20),
                hit_at("span_dana", 0.9, "Dana joined the project.", 10),
            ],
            ..Default::default()
        },
        ContextOptions {
            token_budget: 240,
            include_provenance: true,
        },
    );

    assert!(compiled
        .body
        .contains("### SetState set_state:Relationship:project:member"));
    assert!(compiled.body.contains("members: Dana, Eli"));
    assert!(compiled.body.contains("span_dana"));
    assert!(compiled.body.contains("span_eli"));
    assert!(compiled.body.find("SetState").unwrap() < compiled.body.find("Dana joined").unwrap());
}

#[test]
fn set_state_excludes_tombstoned_member_but_keeps_historical_raw_evidence() {
    let mut dana = claim(
        "claim_member_dana",
        "project",
        "member",
        Some("Dana"),
        "Dana is an active project member.",
        20,
    );
    dana.claim_kind = ClaimKind::Relationship;
    let mut eli_removed = claim(
        "claim_member_eli_removed",
        "project",
        "member",
        Some("Eli"),
        "Eli used to be a project member.",
        10,
    );
    eli_removed.claim_kind = ClaimKind::Relationship;
    eli_removed.status = MemoryStatus::Tombstoned;

    let compiled = build_query_state_context(
        &ContextInput {
            claims: &[dana, eli_removed],
            hits: &[
                hit_at("span_eli_old", 0.9, "Eli joined the project earlier.", 10),
                hit_at("span_dana", 0.8, "Dana joined the project.", 20),
            ],
            ..Default::default()
        },
        ContextOptions {
            token_budget: 240,
            include_provenance: false,
        },
    );

    assert!(compiled.body.contains("members: Dana"));
    assert!(!compiled.body.contains("members: Dana, Eli"));
    assert!(compiled.body.contains("Eli joined the project earlier."));
}

#[test]
fn set_state_canonicalizes_subject_predicate_for_grouping_and_ids() {
    let mut first = claim(
        "claim_member_canonical_a",
        "Project",
        " Member ",
        Some("Dana"),
        "Dana is an active project member.",
        10,
    );
    first.claim_kind = ClaimKind::Relationship;
    let mut second = claim(
        "claim_member_canonical_b",
        " project ",
        "member",
        Some("Eli"),
        "Eli is an active project member.",
        20,
    );
    second.claim_kind = ClaimKind::Relationship;

    let compiled = build_query_state_context(
        &ContextInput {
            claims: &[first, second],
            hits: &[
                hit_at("span_member_eli", 0.9, "Eli joined the project.", 20),
                hit_at("span_member_dana", 0.8, "Dana joined the project.", 10),
            ],
            ..Default::default()
        },
        ContextOptions {
            token_budget: 220,
            include_provenance: false,
        },
    );

    let ids = compiled
        .included
        .iter()
        .map(|item| item.id.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        ids.iter()
            .filter(|id| id.contains("set_state:Relationship:project:member"))
            .count(),
        1
    );
    assert!(compiled
        .body
        .contains("### SetState set_state:Relationship:project:member"));
    assert!(compiled.body.contains("members: Dana, Eli"));
}

#[test]
fn set_state_keeps_distinct_subject_or_predicate_sets_separate() {
    let mut first = claim(
        "claim_member_a",
        "project",
        "member",
        Some("Dana"),
        "Dana is an active project member.",
        10,
    );
    first.claim_kind = ClaimKind::Relationship;
    let mut second = claim(
        "claim_member_b",
        "project",
        "member list",
        Some("Eli"),
        "Eli is on the member list.",
        20,
    );
    second.claim_kind = ClaimKind::Relationship;
    let mut third = claim(
        "claim_member_c",
        "project beta",
        "member",
        Some("Rae"),
        "Rae is an active project beta member.",
        30,
    );
    third.claim_kind = ClaimKind::Relationship;

    let compiled = build_query_state_context(
        &ContextInput {
            claims: &[first, second, third],
            hits: &[
                hit_at("span_member_dana", 0.8, "Dana joined the project.", 10),
                hit_at(
                    "span_member_eli",
                    0.7,
                    "Eli was added to the member list.",
                    20,
                ),
                hit_at("span_member_rae", 0.9, "Rae joined project beta.", 30),
            ],
            ..Default::default()
        },
        ContextOptions {
            token_budget: 280,
            include_provenance: false,
        },
    );

    let ids = compiled
        .included
        .iter()
        .map(|item| item.id.as_str())
        .collect::<Vec<_>>();
    assert!(ids.contains(&"set_state:Relationship:project:member"));
    assert!(ids.contains(&"set_state:Relationship:project:member list"));
    assert!(ids.contains(&"set_state:Relationship:project beta:member"));
    assert_eq!(
        ids.iter()
            .filter(|id| id.starts_with("set_state:Relationship:"))
            .count(),
        3
    );
}

#[test]
fn set_state_does_not_merge_relationship_and_event_claims() {
    let mut relationship = claim(
        "claim_member_dana",
        "project",
        "member",
        Some("Dana"),
        "Dana is a project member.",
        10,
    );
    relationship.claim_kind = ClaimKind::Relationship;
    let mut event = claim(
        "claim_member_launch",
        "project",
        "member",
        Some("launch meeting"),
        "The launch meeting included project members.",
        20,
    );
    event.claim_kind = ClaimKind::Event;

    let compiled = build_query_state_context(
        &ContextInput {
            claims: &[relationship, event],
            ..Default::default()
        },
        ContextOptions {
            token_budget: 200,
            include_provenance: false,
        },
    );

    let ids = compiled
        .included
        .iter()
        .map(|item| item.id.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        ids,
        vec![
            "set_state:Relationship:project:member",
            "set_state:Event:project:member",
        ]
    );
    assert!(!compiled
        .body
        .contains("EntityStateCard entity_state:project"));
}

#[test]
fn query_state_context_uses_fixed_critical_state_order() {
    let current = claim(
        "claim_current",
        "project",
        "owner",
        Some("Dana"),
        "Project owner is Dana.",
        30,
    );
    let mut set = claim(
        "claim_set",
        "project",
        "member",
        Some("Eli"),
        "Eli is an active project member.",
        40,
    );
    set.claim_kind = ClaimKind::Relationship;
    let mut unsupported = claim(
        "claim_unsupported",
        "project",
        "billing contact",
        None,
        "Project billing contact is unsupported after vendor changed.",
        20,
    );
    unsupported.polarity = ClaimPolarity::Uncertain;
    let mut derived = claim(
        "claim_derived",
        "project",
        "reviewer",
        Some("Dana"),
        "Project reviewer follows the current owner Dana.",
        25,
    );
    derived.asserted_by = "extractor_derived".to_string();
    let mut tombstone = claim(
        "claim_tombstone",
        "project",
        "old owner",
        Some("Eli"),
        "Project old owner was deleted.",
        10,
    );
    tombstone.status = MemoryStatus::Tombstoned;

    let compiled = build_query_state_context(
        &ContextInput {
            claims: &[current, set, unsupported, derived, tombstone],
            hits: &[hit_at(
                "span_raw",
                0.9,
                "Raw project evidence remains available.",
                50,
            )],
            ..Default::default()
        },
        ContextOptions {
            token_budget: 300,
            include_provenance: false,
        },
    );

    let ids = compiled
        .included
        .iter()
        .map(|item| item.id.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        ids,
        vec![
            "claim_tombstone",
            "claim_unsupported",
            "claim_derived",
            "claim_current",
            "set_state:Relationship:project:member",
            "span_raw",
        ]
    );
    assert!(
        compiled.body.find("TombstoneState").unwrap()
            < compiled.body.find("UnsupportedState").unwrap()
    );
    assert!(
        compiled.body.find("UnsupportedState").unwrap()
            < compiled.body.find("DerivedState").unwrap()
    );
    assert!(
        compiled.body.find("DerivedState").unwrap() < compiled.body.find("CurrentState").unwrap()
    );
    assert!(compiled.body.find("CurrentState").unwrap() < compiled.body.find("SetState").unwrap());
    assert!(!compiled
        .body
        .contains("EntityStateCard entity_state:project"));
    assert!(compiled
        .body
        .contains("Raw project evidence remains available."));
}

#[test]
fn unsupported_state_refusal_text_precedes_stale_raw_evidence() {
    let mut unsupported = claim(
        "claim_billing_contact_unsupported",
        "project",
        "billing contact",
        None,
        "Project billing contact is unsupported after vendor changed.",
        20,
    );
    unsupported.polarity = ClaimPolarity::Uncertain;

    let compiled = build_query_state_context(
        &ContextInput {
            claims: &[unsupported],
            hits: &[hit_at(
                "span_old_billing_contact",
                0.9,
                "Before the vendor changed, Eli handled billing.",
                10,
            )],
            ..Default::default()
        },
        ContextOptions {
            token_budget: 200,
            include_provenance: false,
        },
    );

    assert!(compiled
        .body
        .contains("### UnsupportedState claim_billing_contact_unsupported"));
    assert!(compiled.body.contains("support_state: unsupported"));
    assert!(
        compiled.body.find("UnsupportedState").unwrap()
            < compiled.body.find("Before the vendor changed").unwrap()
    );
    assert!(compiled.body.contains("Before the vendor changed"));
}

#[test]
fn inactive_claim_context_hides_stale_value() {
    let claim = ClaimRecord {
        id: "claim_deleted".to_string(),
        scope: MemoryScope::new("default"),
        status: MemoryStatus::Tombstoned,
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        claim_text: "User's deleted code is SECRET-123.".to_string(),
        subject: Some("user".to_string()),
        predicate: Some("deleted_code".to_string()),
        object_value: Some("SECRET-123".to_string()),
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
    let compiled = build_state_context(
        &ContextInput {
            claims: &[claim],
            ..Default::default()
        },
        ContextOptions {
            token_budget: 100,
            include_provenance: true,
        },
    );
    assert!(compiled.body.contains("### TombstoneState claim_deleted"));
    assert!(compiled.body.contains("deleted_code"));
    assert!(!compiled.body.contains("SECRET-123"));
    assert_eq!(compiled.included[0].kind, "state_tombstone");
}

#[test]
fn derived_claim_context_is_labeled() {
    let claim = ClaimRecord {
        id: "claim_derived".to_string(),
        scope: MemoryScope::new("default"),
        status: MemoryStatus::Active,
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        claim_text: "Dependent project status is green.".to_string(),
        subject: Some("project".to_string()),
        predicate: Some("status".to_string()),
        object_value: Some("green".to_string()),
        subject_entity_id: None,
        slot_id: None,
        slot_facet: None,
        claim_kind: ClaimKind::Fact,
        polarity: ClaimPolarity::Affirmative,
        source_span_ids: Vec::new(),
        source_episode_ids: vec!["rule_event".to_string(), "change_event".to_string()],
        source_sequence_no: None,
        asserted_by: "extractor_derived".to_string(),
        extractor_version: None,
        confidence: Some(1.0),
        observed_at_ms: 20,
        valid_from_ms: Some(20),
        valid_to_ms: None,
        correction_ids: Vec::new(),
    };
    let compiled = build_state_context(
        &ContextInput {
            claims: &[claim],
            ..Default::default()
        },
        ContextOptions {
            token_budget: 100,
            include_provenance: true,
        },
    );
    assert!(compiled.body.contains("### DerivedState claim_derived"));
    assert_eq!(compiled.included[0].kind, "state_derived");
}

#[test]
fn answer_ready_context_emits_one_resolved_target_with_its_raw_proof() {
    let mut target = claim(
        "claim_revision",
        "entrega",
        "revisor",
        Some("Dana"),
        "La persona revisora de la entrega es Dana.",
        20,
    );
    target.slot_id = Some("slot_entrega_revisor".to_string());
    target.source_span_ids = vec!["span_revision".to_string()];
    target.asserted_by = "extractor_derived".to_string();
    let mut coverage = claim(
        "claim_formato",
        "entrega",
        "formato",
        Some("PDF"),
        "El formato de la entrega es PDF.",
        10,
    );
    coverage.slot_id = Some("slot_entrega_formato".to_string());
    coverage.source_span_ids = vec!["span_formato".to_string()];
    let projection = AnswerReadyStateProjection {
        claims: vec![target.clone(), coverage],
        set_states: Vec::new(),
        slot_histories: Vec::new(),
        rules: Vec::new(),
        rule_outcomes: vec![RuleResolutionOutcome {
            rule_id: "rule_revision".to_string(),
            trigger_slot_id: Some("slot_entrega_responsable".to_string()),
            target_slot_id: target.slot_id.clone(),
            status: RuleResolutionStatus::AppliedDerived,
            resolved_claim_id: Some(target.id.clone()),
        }],
        corrections: Vec::new(),
        entities: Vec::new(),
        proof_span_ids: vec!["span_revision".to_string()],
        support: crate::AnswerSupportContract {
            state: crate::AnswerSupportState::Supported,
            source_claim_ids: vec![target.id.clone()],
            source_span_ids: vec!["span_revision".to_string()],
            slots: vec![crate::AnswerSlotSupport {
                slot_id: "slot_entrega_revisor".to_string(),
                subject: Some("entrega".to_string()),
                predicate: Some("revisor".to_string()),
                view: crate::StateReadView::Current,
                state: crate::AnswerSupportState::Supported,
                source_claim_ids: vec![target.id.clone()],
                source_span_ids: vec!["span_revision".to_string()],
            }],
        },
    };
    let proof = hit_at(
        "span_revision",
        1.0,
        "La persona revisora de la entrega es Dana.",
        20,
    );
    let historical = hit_at(
        "span_revision_anterior",
        0.9,
        "La persona revisora anterior era Eli.",
        10,
    );
    let coverage_proof = hit_at("span_formato", 0.8, "El formato es PDF.", 10);

    let compiled = build_answer_ready_state_context(
        &[],
        &projection,
        &[],
        &[historical, proof, coverage_proof],
        ContextOptions::default(),
    );

    assert!(compiled
        .body
        .starts_with("### ResolvedAnswerState slot_entrega_revisor"));
    assert!(compiled.body.contains("support_state: supported"));
    assert!(compiled.body.contains("state_kind: derived"));
    assert!(compiled.body.contains("current_value: Dana"));
    assert!(compiled.body.contains("dependency_resolution: applied"));
    assert!(!compiled.body.contains("### DerivedState claim_revision"));
    assert!(compiled.body.contains("### CurrentState claim_formato"));
    assert!(compiled.body.contains("### Memory span_revision"));
    assert!(compiled.body.contains("score: 1.0000"));
    assert!(compiled.body.contains("score: 0.8000"));
    assert!(compiled.body.contains("### Memory span_revision_anterior"));
    assert!(compiled.body.contains("score: 0.9000"));
    assert!(!compiled.body.contains("\nrole: "));
    let target_position = compiled.body.find("### Memory span_revision\n").unwrap();
    let state_position = compiled.body.find("### Memory span_formato\n").unwrap();
    let history_position = compiled
        .body
        .find("### Memory span_revision_anterior\n")
        .unwrap();
    assert!(target_position < state_position);
    assert!(state_position < history_position);
    assert_eq!(compiled.body.matches("read_role: answer_target").count(), 1);
    assert_eq!(compiled.support.state, crate::AnswerSupportState::Supported);
}

#[test]
fn resolved_unsupported_current_target_fails_closed_without_raw_history() {
    let mut unsupported = claim(
        "claim_contacto_sin_apoyo",
        "cuenta",
        "contacto",
        None,
        "El contacto anterior ya no tiene apoyo vigente.",
        30,
    );
    unsupported.slot_id = Some("slot_cuenta_contacto".to_string());
    unsupported.polarity = ClaimPolarity::Uncertain;
    unsupported.source_span_ids = vec!["span_cambio".to_string()];
    let projection = AnswerReadyStateProjection {
        claims: vec![unsupported.clone()],
        set_states: Vec::new(),
        slot_histories: Vec::new(),
        rules: Vec::new(),
        rule_outcomes: Vec::new(),
        corrections: Vec::new(),
        entities: Vec::new(),
        proof_span_ids: vec!["span_cambio".to_string()],
        support: crate::AnswerSupportContract {
            state: crate::AnswerSupportState::Unsupported,
            source_claim_ids: vec![unsupported.id.clone()],
            source_span_ids: vec!["span_cambio".to_string()],
            slots: vec![crate::AnswerSlotSupport {
                slot_id: "slot_cuenta_contacto".to_string(),
                subject: Some("cuenta".to_string()),
                predicate: Some("contacto".to_string()),
                view: crate::StateReadView::Current,
                state: crate::AnswerSupportState::Unsupported,
                source_claim_ids: vec![unsupported.id],
                source_span_ids: vec!["span_cambio".to_string()],
            }],
        },
    };

    let evidence = hit_at(
        "span_cambio",
        1.0,
        "El proveedor cambió; el contacto anterior requiere confirmación.",
        30,
    );
    let compiled = build_answer_ready_state_context(
        &[],
        &projection,
        &[],
        std::slice::from_ref(&evidence),
        ContextOptions::default(),
    );

    assert!(compiled.body.contains("support_state: unsupported"));
    assert!(compiled.body.contains("authoritative_current_value: false"));
    assert!(compiled.body.contains("current_value: null"));
    assert!(!compiled
        .body
        .contains("### UnsupportedState claim_contacto_sin_apoyo"));
    assert!(!compiled.body.contains("### Memory span_cambio"));
    assert_eq!(compiled.included.len(), 1);
    assert_eq!(compiled.included[0].kind, "state_resolved");
    assert_eq!(
        compiled.support.state,
        crate::AnswerSupportState::Unsupported
    );

    let mut timeline_projection = projection;
    timeline_projection.support.slots[0].view = crate::StateReadView::Timeline;
    let timeline = build_answer_ready_state_context(
        &[],
        &timeline_projection,
        &[],
        &[evidence],
        ContextOptions::default(),
    );
    assert!(timeline.body.contains("read_view: timeline"));
    assert!(timeline.body.contains("### Memory span_cambio"));
}

#[test]
fn mixed_current_targets_withhold_raw_only_for_non_supported_slots() {
    let mut hours = claim(
        "claim_horario",
        "atelier",
        "horaire",
        Some("08:00"),
        "L'atelier ouvre à 08:00.",
        20,
    );
    hours.slot_id = Some("slot_atelier_horaire".to_string());
    hours.source_span_ids = vec!["span_horaire".to_string()];

    let mut contact = claim(
        "claim_contacto",
        "atelier",
        "contacto",
        Some("Noël"),
        "El contacto anterior era Noël.",
        10,
    );
    contact.slot_id = Some("slot_atelier_contacto".to_string());
    contact.status = MemoryStatus::Tombstoned;
    contact.source_span_ids = vec!["span_contacto".to_string()];

    let mut coverage = claim(
        "claim_turno",
        "atelier",
        "turno",
        Some("noche"),
        "El turno vigente es noche.",
        15,
    );
    coverage.slot_id = Some("slot_atelier_turno".to_string());
    coverage.source_span_ids = vec!["span_turno".to_string()];

    let projection = AnswerReadyStateProjection {
        claims: vec![hours.clone(), contact.clone(), coverage],
        set_states: Vec::new(),
        slot_histories: Vec::new(),
        rules: Vec::new(),
        rule_outcomes: Vec::new(),
        corrections: Vec::new(),
        entities: Vec::new(),
        proof_span_ids: vec!["span_horaire".to_string(), "span_contacto".to_string()],
        support: crate::AnswerSupportContract {
            state: crate::AnswerSupportState::Deleted,
            source_claim_ids: vec![hours.id.clone(), contact.id.clone()],
            source_span_ids: vec!["span_horaire".to_string(), "span_contacto".to_string()],
            slots: vec![
                crate::AnswerSlotSupport {
                    slot_id: "slot_atelier_horaire".to_string(),
                    subject: Some("atelier".to_string()),
                    predicate: Some("horaire".to_string()),
                    view: crate::StateReadView::Current,
                    state: crate::AnswerSupportState::Supported,
                    source_claim_ids: vec![hours.id.clone()],
                    source_span_ids: vec!["span_horaire".to_string()],
                },
                crate::AnswerSlotSupport {
                    slot_id: "slot_atelier_contacto".to_string(),
                    subject: Some("atelier".to_string()),
                    predicate: Some("contacto".to_string()),
                    view: crate::StateReadView::Current,
                    state: crate::AnswerSupportState::Deleted,
                    source_claim_ids: vec![contact.id.clone()],
                    source_span_ids: vec!["span_contacto".to_string()],
                },
            ],
        },
    };

    let compiled = build_answer_ready_state_context(
        &[],
        &projection,
        &[],
        &[
            hit_at("span_horaire", 1.0, "L'atelier ouvre à 08:00.", 20),
            hit_at("span_contacto", 0.9, "El contacto es Noël.", 10),
            hit_at("span_turno", 0.8, "El turno vigente es noche.", 15),
        ],
        ContextOptions::default(),
    );

    assert_eq!(compiled.body.matches("### ResolvedAnswerState").count(), 2);
    assert!(compiled.body.contains("current_value: 08:00"));
    assert!(compiled.body.contains("support_state: deleted"));
    assert!(compiled.body.contains("current_value: null"));
    assert!(compiled.body.contains("### CurrentState claim_turno"));
    assert!(compiled.body.contains("### Memory span_horaire"));
    assert!(compiled.body.contains("### Memory span_turno"));
    assert!(!compiled.body.contains("### Memory span_contacto"));
    assert!(!compiled.body.contains("El contacto es Noël."));
}

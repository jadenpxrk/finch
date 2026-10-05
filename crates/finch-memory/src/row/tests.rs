use super::*;
use crate::schema::{
    artifact_schema, claim_schema, correction_schema, dependency_trace_schema, edge_schema,
    entity_alias_schema, entity_schema, episode_schema, profile_schema, rule_schema, slot_schema,
    span_schema, state_record_schema,
};
use crate::types::{
    ActorKind, ArtifactKind, ClaimKind, ClaimPolarity, CorrectionAuthority, CorrectionOperation,
    MemoryStatus, SourceKind, SourceType, StateRecordKind, TemporalFields,
};

fn scope() -> MemoryScope {
    let mut scope = MemoryScope::new("default");
    scope.user_id = Some("user_1".to_string());
    scope.thread_id = Some("thread_1".to_string());
    scope
}

fn temporal() -> TemporalFields {
    TemporalFields {
        created_at_ms: 1,
        ingested_at_ms: 2,
        event_time_ms: Some(1),
        valid_from_ms: Some(1),
        valid_to_ms: None,
    }
}

#[test]
fn episode_doc_validates_against_schema() {
    let record = EpisodeRecord {
        id: "ep_1".to_string(),
        scope: scope(),
        status: MemoryStatus::Active,
        visibility: Visibility::Private,
        policy_tags: vec!["local".to_string()],
        source_kind: SourceKind::UserMessage,
        actor: ActorKind::User,
        sequence_no: 1,
        temporal: temporal(),
        raw_text: "Remember that the repo uses Rust.".to_string(),
        blob_ref: None,
        mime_type: Some("text/plain".to_string()),
        content_hash: "hash_1".to_string(),
        causal_parent_ids: Vec::new(),
        metadata_json: None,
    };
    let doc = episode_doc(&record).unwrap();
    doc.validate(&episode_schema(), false).unwrap();
}

#[test]
fn span_doc_validates_with_embedding() {
    let record = SpanRecord {
        id: "span_1".to_string(),
        scope: scope(),
        status: MemoryStatus::Active,
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        source_type: SourceType::Episode,
        source_id: "ep_1".to_string(),
        byte_start: 0,
        byte_end: 12,
        char_start: 0,
        char_end: 12,
        token_start: None,
        token_end: None,
        span_index: 0,
        text: "Rust repo".to_string(),
        text_hash: "hash_2".to_string(),
        chunker_version: "v1".to_string(),
        embedding_model: Some("test".to_string()),
        embedding_version: Some("1".to_string()),
        lexical_text: "rust repo".to_string(),
        created_at_ms: 3,
        valid_from_ms: Some(3),
        valid_to_ms: None,
        provenance: Vec::new(),
    };
    let doc = span_doc(&record, Some(&[0.0, 1.0, 0.0])).unwrap();
    doc.validate(&span_schema(3), false).unwrap();
}

#[test]
fn span_from_doc_round_trips_scalar_fields() {
    let record = SpanRecord {
        id: "span_1".to_string(),
        scope: scope(),
        status: MemoryStatus::Active,
        visibility: Visibility::Private,
        policy_tags: vec!["local".to_string()],
        source_type: SourceType::Episode,
        source_id: "ep_1".to_string(),
        byte_start: 0,
        byte_end: 12,
        char_start: 0,
        char_end: 12,
        token_start: Some(0),
        token_end: Some(3),
        span_index: 0,
        text: "Rust repo".to_string(),
        text_hash: "hash_2".to_string(),
        chunker_version: "v1".to_string(),
        embedding_model: Some("test".to_string()),
        embedding_version: Some("1".to_string()),
        lexical_text: "rust repo".to_string(),
        created_at_ms: 3,
        valid_from_ms: Some(3),
        valid_to_ms: None,
        provenance: vec![ProvenanceRef {
            source_type: SourceType::Episode,
            source_id: "ep_1".to_string(),
            span_id: Some("span_1".to_string()),
            actor: Some(ActorKind::Assistant),
        }],
    };
    let doc = span_doc(&record, Some(&[0.0, 1.0, 0.0])).unwrap();
    assert_eq!(span_from_doc(&doc).unwrap(), record);
}

#[test]
fn provenance_without_actor_decodes() {
    let provenance: ProvenanceRef =
        serde_json::from_str(r#"{"source_type":"episode","source_id":"ep_1","span_id":null}"#)
            .unwrap();

    assert_eq!(provenance.actor, None);
}

#[test]
fn artifact_doc_validates_against_schema() {
    let record = ArtifactRecord {
        id: "art_1".to_string(),
        scope: scope(),
        status: MemoryStatus::Active,
        visibility: Visibility::Tool,
        policy_tags: vec!["repo".to_string()],
        artifact_kind: ArtifactKind::File,
        title: "README.md".to_string(),
        uri: Some("file:///repo/README.md".to_string()),
        blob_ref: None,
        mime_type: Some("text/markdown".to_string()),
        content_hash: "hash_3".to_string(),
        source_created_at_ms: None,
        source_modified_at_ms: Some(4),
        created_at_ms: 5,
        ingested_at_ms: 5,
        valid_from_ms: Some(5),
        valid_to_ms: None,
        extracted_text_ref: None,
        metadata_json: None,
    };
    let doc = artifact_doc(&record).unwrap();
    doc.validate(&artifact_schema(), false).unwrap();
    assert_eq!(artifact_from_doc(&doc).unwrap(), record);
}

#[test]
fn correction_doc_validates_against_schema() {
    let record = CorrectionRecord {
        id: "corr_1".to_string(),
        scope: scope(),
        status: MemoryStatus::Active,
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        operation: CorrectionOperation::Tombstone,
        target_type: "span".to_string(),
        target_match: CorrectionTargetMatch::ClaimVersions,
        target_ids: vec!["span_1".to_string()],
        target_selector: None,
        target_subject_key: None,
        target_predicate_key: None,
        target_subject_entity_id: None,
        target_slot_id: None,
        new_value: None,
        reason: Some("user requested forgetting".to_string()),
        actor: ActorKind::User,
        authority: CorrectionAuthority::User,
        created_at_ms: 10,
        effective_at_ms: 10,
        applies_valid_from_ms: None,
        applies_valid_to_ms: None,
        cascade_policy: None,
        source_span_ids: vec!["span_1".to_string()],
        source_episode_ids: vec!["ep_1".to_string()],
        source_sequence_no: Some(2),
        metadata_json: None,
    };
    let doc = correction_doc(&record).unwrap();
    doc.validate(&correction_schema(), false).unwrap();
    assert_eq!(correction_from_doc(&doc).unwrap(), record);
}

#[test]
fn claim_doc_validates_and_round_trips() {
    let record = ClaimRecord {
        id: "claim_1".to_string(),
        scope: scope(),
        status: MemoryStatus::Active,
        visibility: Visibility::Private,
        policy_tags: vec!["manual".to_string()],
        claim_text: "User prefers concise Rust explanations.".to_string(),
        subject: Some("user".to_string()),
        predicate: Some("prefers".to_string()),
        object_value: Some("concise Rust explanations".to_string()),
        subject_entity_id: Some("entity_user".to_string()),
        slot_id: Some("slot_user_prefers".to_string()),
        slot_facet: None,
        claim_kind: ClaimKind::Preference,
        polarity: ClaimPolarity::Affirmative,
        source_span_ids: vec!["span_1".to_string()],
        source_episode_ids: vec!["ep_1".to_string()],
        source_sequence_no: Some(2),
        asserted_by: "user".to_string(),
        extractor_version: None,
        confidence: Some(1.0),
        observed_at_ms: 10,
        valid_from_ms: Some(10),
        valid_to_ms: None,
        correction_ids: Vec::new(),
    };
    let doc = claim_doc(&record, Some(&[1.0, 0.0, 0.0])).unwrap();
    doc.validate(&claim_schema(3), false).unwrap();
    assert_eq!(claim_from_doc(&doc).unwrap(), record);
}

#[test]
fn state_record_doc_validates_and_round_trips() {
    let record = StateRecord {
        id: "state_current_owner".to_string(),
        state_key: "state_current_owner".to_string(),
        scope: scope(),
        status: MemoryStatus::Active,
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        state_kind: StateRecordKind::Current,
        claim_kind: ClaimKind::Fact,
        subject: Some("project".to_string()),
        predicate: Some("owner".to_string()),
        object_value: Some("Dana".to_string()),
        subject_entity_id: Some("entity_project".to_string()),
        slot_id: Some("slot_project_owner".to_string()),
        slot_facet: None,
        state_text: "project owner Dana".to_string(),
        members: Vec::new(),
        claim_ids: vec!["claim_1".to_string()],
        correction_ids: Vec::new(),
        rule_ids: vec!["rule_1".to_string()],
        source_span_ids: vec!["span_1".to_string()],
        source_episode_ids: vec!["ep_1".to_string()],
        source_sequence_no: Some(2),
        observed_at_ms: 10,
        valid_from_ms: Some(10),
        valid_to_ms: None,
        projected_at_ms: 11,
        recorded_at_ms: 11,
        superseded_at_ms: None,
    };
    let doc = state_record_doc(&record).unwrap();
    doc.validate(&state_record_schema(), false).unwrap();
    assert_eq!(state_record_from_doc(&doc).unwrap(), record);
}

#[test]
fn dependency_trace_doc_validates_and_round_trips() {
    let record = DependencyTraceRecord {
        id: "dependency_trace_1".to_string(),
        trace_key: "dependency_trace_1".to_string(),
        scope: scope(),
        status: MemoryStatus::Active,
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        rule_id: "rule_1".to_string(),
        trigger_claim_id: "claim_trigger".to_string(),
        prior_trigger_claim_id: Some("claim_prior_trigger".to_string()),
        derived_claim_id: "claim_derived".to_string(),
        target_subject_key: "project".to_string(),
        target_predicate_key: "reviewer".to_string(),
        target_subject_entity_id: Some("entity_project".to_string()),
        target_slot_id: Some("slot_project_reviewer".to_string()),
        target_subject: Some("project".to_string()),
        target_predicate: Some("reviewer".to_string()),
        state_kind: StateRecordKind::Derived,
        hop: 1,
        parent_trace_ids: vec!["dependency_trace_parent".to_string()],
        source_span_ids: vec!["span_1".to_string(), "span_rule".to_string()],
        source_episode_ids: vec!["ep_1".to_string(), "ep_rule".to_string()],
        observed_at_ms: 10,
        valid_from_ms: Some(10),
        valid_to_ms: None,
        projected_at_ms: 11,
        recorded_at_ms: 11,
        superseded_at_ms: None,
    };
    let doc = dependency_trace_doc(&record).unwrap();
    doc.validate(&dependency_trace_schema(), false).unwrap();
    assert_eq!(dependency_trace_from_doc(&doc).unwrap(), record);
}

#[test]
fn rule_doc_validates_and_round_trips_binding_status() {
    let record = RuleRecord {
        id: "rule_1".to_string(),
        scope: scope(),
        status: MemoryStatus::Active,
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        trigger_subject_key: "project".to_string(),
        trigger_predicate_key: "owner".to_string(),
        target_subject_key: "project".to_string(),
        target_predicate_key: "reviewer".to_string(),
        trigger_subject_entity_id: Some("entity_project".to_string()),
        target_subject_entity_id: Some("entity_project".to_string()),
        trigger_slot_id: Some("slot_project_owner".to_string()),
        target_slot_id: Some("slot_project_reviewer".to_string()),
        trigger_binding_status: RuleBindingStatus::Bound,
        target_binding_status: RuleBindingStatus::Bound,
        trigger_subject: Some("project".to_string()),
        trigger_predicate: Some("owner".to_string()),
        target_subject: Some("project".to_string()),
        target_predicate: Some("reviewer".to_string()),
        target_match: RuleTargetMatch::ExactSlot,
        activation: RuleActivation::ContinuousProjection,
        action: RuleAction::DeriveValue,
        value_template: Some("{value}".to_string()),
        value: None,
        source_span_ids: vec!["span_rule".to_string()],
        source_episode_ids: vec!["episode_rule".to_string()],
        source_sequence_no: Some(1),
        valid_from_ms: Some(10),
        valid_to_ms: None,
        confidence: Some(1.0),
    };
    let doc = rule_doc(&record).unwrap();
    doc.validate(&rule_schema(), false).unwrap();
    assert_eq!(rule_from_doc(&doc).unwrap(), record);
}

#[test]
fn slot_doc_validates_and_round_trips() {
    let record = CanonicalSlotRecord {
        id: "slot_project_owner".to_string(),
        slot_key: "slot_project_owner".to_string(),
        scope: scope(),
        status: MemoryStatus::Active,
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        subject_key: "project".to_string(),
        predicate_key: "owner".to_string(),
        subject_entity_id: Some("entity_project".to_string()),
        subject: Some("project".to_string()),
        predicate: Some("owner".to_string()),
        source_claim_ids: vec!["claim_1".to_string()],
        source_rule_ids: vec!["rule_1".to_string()],
        source_entity_ids: vec!["entity_project".to_string()],
        observed_at_ms: 10,
        valid_from_ms: Some(10),
        valid_to_ms: None,
        projected_at_ms: 11,
        recorded_at_ms: 11,
        superseded_at_ms: None,
    };
    let doc = slot_doc(&record).unwrap();
    doc.validate(&slot_schema(), false).unwrap();
    assert_eq!(slot_from_doc(&doc).unwrap(), record);
}

#[test]
fn profile_doc_validates_and_round_trips() {
    let record = ProfileRecord {
        id: "profile_1".to_string(),
        scope: scope(),
        status: MemoryStatus::Active,
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        subject_id: Some("user_1".to_string()),
        profile_key: "style".to_string(),
        profile_text: "Prefers compact Rust examples.".to_string(),
        value_json: Some(r#""compact""#.to_string()),
        evidence_claim_ids: vec!["claim_1".to_string()],
        source_span_ids: vec!["span_1".to_string()],
        generated_at_ms: 20,
        generator_version: "manual".to_string(),
        valid_from_ms: Some(20),
        valid_to_ms: None,
        correction_watermark: 20,
    };
    let doc = profile_doc(&record).unwrap();
    doc.validate(&profile_schema(), false).unwrap();
    assert_eq!(profile_from_doc(&doc).unwrap(), record);
}

#[test]
fn entity_and_edge_docs_validate_and_round_trip() {
    let entity = EntityRecord {
        id: "entity_1".to_string(),
        scope: scope(),
        status: MemoryStatus::Active,
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        entity_type: "project".to_string(),
        canonical_name: "Finch".to_string(),
        aliases: vec!["finch-db".to_string()],
        source_claim_ids: vec!["claim_1".to_string()],
        merge_parent_ids: Vec::new(),
        split_from_id: None,
        confidence: Some(0.9),
    };
    let doc = entity_doc(&entity, Some(&[1.0, 0.0, 0.0])).unwrap();
    doc.validate(&entity_schema(3), false).unwrap();
    assert_eq!(entity_from_doc(&doc).unwrap(), entity);

    let alias = EntityAliasRecord {
        id: "entity_alias_1".to_string(),
        scope: scope(),
        status: MemoryStatus::Active,
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        alias_key: "finch db".to_string(),
        entity_id: "entity_1".to_string(),
        entity_type: "project".to_string(),
        source_claim_ids: vec!["claim_1".to_string()],
        recorded_at_ms: 10,
        superseded_at_ms: None,
    };
    let doc = entity_alias_doc(&alias).unwrap();
    doc.validate(&entity_alias_schema(), false).unwrap();
    assert_eq!(entity_alias_from_doc(&doc).unwrap(), alias);

    let edge = EdgeRecord {
        id: "edge_1".to_string(),
        scope: scope(),
        status: MemoryStatus::Active,
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        src_entity_id: "entity_1".to_string(),
        dst_entity_id: "entity_2".to_string(),
        relation_type: "mentions".to_string(),
        claim_id: Some("claim_1".to_string()),
        source_span_ids: vec!["span_1".to_string()],
        valid_from_ms: Some(10),
        valid_to_ms: None,
        confidence: Some(0.8),
    };
    let doc = edge_doc(&edge).unwrap();
    doc.validate(&edge_schema(), false).unwrap();
    assert_eq!(edge_from_doc(&doc).unwrap(), edge);
}

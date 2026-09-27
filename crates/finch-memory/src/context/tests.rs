use super::*;
use crate::retrieval::SpanSearchHit;
use crate::types::{
    ActorKind, ArtifactKind, ArtifactRecord, ClaimKind, ClaimPolarity, ClaimRecord, MemoryScope,
    MemoryStatus, ProfileRecord, SlotHistoryRecord, SlotHistoryVersion, SourceType, SpanRecord,
    StateRecordKind, Visibility,
};

fn hit(id: &str, score: f32, text: &str) -> SpanSearchHit {
    hit_at(id, score, text, 0)
}

fn hit_at(id: &str, score: f32, text: &str, valid_from_ms: i64) -> SpanSearchHit {
    SpanSearchHit {
        score,
        span: SpanRecord {
            id: id.to_string(),
            scope: MemoryScope::new("default"),
            status: MemoryStatus::Active,
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            source_type: SourceType::Episode,
            source_id: "ep_1".to_string(),
            byte_start: 0,
            byte_end: text.len() as i64,
            char_start: 0,
            char_end: text.chars().count() as i64,
            token_start: None,
            token_end: None,
            span_index: 0,
            text: text.to_string(),
            text_hash: id.to_string(),
            chunker_version: "test".to_string(),
            embedding_model: None,
            embedding_version: None,
            lexical_text: text.to_lowercase(),
            created_at_ms: 0,
            valid_from_ms: Some(valid_from_ms),
            valid_to_ms: None,
            provenance: Vec::new(),
        },
    }
}

#[test]
fn raw_memory_renders_structured_source_actor() {
    let mut evidence = hit("span_assistant", 1.0, "El proyecto sigue activo.");
    evidence.span.provenance = vec![crate::ProvenanceRef {
        source_type: SourceType::Episode,
        source_id: evidence.span.source_id.clone(),
        span_id: Some(evidence.span.id.clone()),
        actor: Some(ActorKind::Assistant),
    }];

    let compiled = build_context(
        &[evidence],
        ContextOptions {
            token_budget: 128,
            include_provenance: true,
        },
    );

    assert!(compiled.body.contains("source_actor: assistant"));
}

fn claim(
    id: &str,
    subject: &str,
    predicate: &str,
    object_value: Option<&str>,
    text: &str,
    observed_at_ms: i64,
) -> ClaimRecord {
    ClaimRecord {
        id: id.to_string(),
        scope: MemoryScope::new("default"),
        status: MemoryStatus::Active,
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        claim_text: text.to_string(),
        subject: Some(subject.to_string()),
        predicate: Some(predicate.to_string()),
        object_value: object_value.map(str::to_string),
        subject_entity_id: None,
        slot_id: None,
        slot_facet: None,
        claim_kind: ClaimKind::Fact,
        polarity: ClaimPolarity::Affirmative,
        source_span_ids: Vec::new(),
        source_episode_ids: Vec::new(),
        source_sequence_no: None,
        asserted_by: "extractor".to_string(),
        extractor_version: None,
        confidence: Some(1.0),
        observed_at_ms,
        valid_from_ms: Some(observed_at_ms),
        valid_to_ms: None,
        correction_ids: Vec::new(),
    }
}

fn entity(id: &str, canonical_name: &str, aliases: &[&str]) -> EntityRecord {
    EntityRecord {
        id: id.to_string(),
        scope: MemoryScope::new("default"),
        status: MemoryStatus::Active,
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        entity_type: "thing".to_string(),
        canonical_name: canonical_name.to_string(),
        aliases: aliases.iter().map(|alias| alias.to_string()).collect(),
        source_claim_ids: vec!["claim_source".to_string()],
        merge_parent_ids: Vec::new(),
        split_from_id: None,
        confidence: Some(1.0),
    }
}

mod packing;
mod state;

use crate::types::{
    canonical_slot_part, ActorKind, ArtifactKind, ArtifactRecord, CanonicalSlotRecord, ClaimKind,
    ClaimPolarity, ClaimRecord, CorrectionAuthority, CorrectionOperation, CorrectionRecord,
    CorrectionTargetMatch, DependencyTraceRecord, EdgeRecord, EntityAliasRecord, EntityRecord,
    EpisodeRecord, MemoryScope, MemoryStatus, ProfileRecord, ProvenanceRef, RuleAction,
    RuleActivation, RuleBindingStatus, RuleRecord, RuleTargetMatch, SlotAliasRecord, SourceType,
    SpanRecord, StateRecord, StateRecordKind, Visibility,
};
use finch_types::{Doc, Status, Value, ZResult};
use serde::Serialize;

mod helpers;

use helpers::*;

pub fn state_record_doc(record: &StateRecord) -> Result<Doc, serde_json::Error> {
    let doc = set_scope(
        Doc::new(record.id.clone()).set("id", Value::String(record.id.clone())),
        &record.scope,
        record.visibility,
        &record.policy_tags,
    )?;
    Ok(set_status(doc, record.status)
        .set("state_key", Value::String(record.state_key.clone()))
        .set(
            "state_kind",
            Value::String(record.state_kind.as_str().to_string()),
        )
        .set("claim_kind", Value::String(enum_name(record.claim_kind)?))
        .set("subject", opt_string(&record.subject))
        .set("predicate", opt_string(&record.predicate))
        .set("object_value", opt_string(&record.object_value))
        .set("subject_entity_id", opt_string(&record.subject_entity_id))
        .set("slot_id", opt_string(&record.slot_id))
        .set("slot_facet", opt_string(&record.slot_facet))
        .set("state_text", Value::String(record.state_text.clone()))
        .set("members_json", json_string(&record.members)?)
        .set("claim_ids_json", json_string(&record.claim_ids)?)
        .set("correction_ids_json", json_string(&record.correction_ids)?)
        .set("rule_ids_json", json_string(&record.rule_ids)?)
        .set("source_spans_json", json_string(&record.source_span_ids)?)
        .set("source_eps_json", json_string(&record.source_episode_ids)?)
        .set("source_sequence_no", opt_i64(record.source_sequence_no))
        .set("observed_at_ms", Value::I64(record.observed_at_ms))
        .set("valid_from_ms", opt_i64(record.valid_from_ms))
        .set("valid_to_ms", opt_i64(record.valid_to_ms))
        .set("projected_at_ms", Value::I64(record.projected_at_ms))
        .set("recorded_at_ms", Value::I64(record.recorded_at_ms))
        .set("superseded_at_ms", opt_i64(record.superseded_at_ms)))
}

pub fn state_record_from_doc(doc: &Doc) -> ZResult<StateRecord> {
    let scope = MemoryScope {
        space_id: required_string(doc, "space_id")?,
        tenant_id: optional_string(doc, "tenant_id")?,
        user_id: optional_string(doc, "user_id")?,
        agent_id: optional_string(doc, "agent_id")?,
        project_id: optional_string(doc, "project_id")?,
        thread_id: optional_string(doc, "thread_id")?,
    };
    let claim_kind = required_string(doc, "claim_kind")?;
    let id = required_string(doc, "id")?;
    let projected_at_ms = required_i64(doc, "projected_at_ms")?;
    Ok(StateRecord {
        state_key: required_string(doc, "state_key")?,
        id,
        scope,
        status: parse_status(&required_string(doc, "status")?)?,
        visibility: parse_visibility(&required_string(doc, "visibility")?)?,
        policy_tags: json_vec(doc, "policy_tags_json")?,
        state_kind: parse_state_record_kind(&required_string(doc, "state_kind")?)?,
        claim_kind: parse_enum::<ClaimKind>("claim_kind", &claim_kind)?,
        subject: optional_string(doc, "subject")?,
        predicate: optional_string(doc, "predicate")?,
        object_value: optional_string(doc, "object_value")?,
        subject_entity_id: optional_string(doc, "subject_entity_id")?,
        slot_id: optional_string(doc, "slot_id")?,
        slot_facet: optional_string(doc, "slot_facet")?,
        state_text: required_string(doc, "state_text")?,
        members: json_vec(doc, "members_json")?,
        claim_ids: json_vec(doc, "claim_ids_json")?,
        correction_ids: json_vec(doc, "correction_ids_json")?,
        rule_ids: json_vec(doc, "rule_ids_json")?,
        source_span_ids: json_vec(doc, "source_spans_json")?,
        source_episode_ids: json_vec(doc, "source_eps_json")?,
        source_sequence_no: optional_i64(doc, "source_sequence_no")?,
        observed_at_ms: required_i64(doc, "observed_at_ms")?,
        valid_from_ms: optional_i64(doc, "valid_from_ms")?,
        valid_to_ms: optional_i64(doc, "valid_to_ms")?,
        projected_at_ms,
        recorded_at_ms: required_i64(doc, "recorded_at_ms")?,
        superseded_at_ms: optional_i64(doc, "superseded_at_ms")?,
    })
}

pub fn dependency_trace_doc(record: &DependencyTraceRecord) -> Result<Doc, serde_json::Error> {
    let doc = set_scope(
        Doc::new(record.id.clone()).set("id", Value::String(record.id.clone())),
        &record.scope,
        record.visibility,
        &record.policy_tags,
    )?;
    Ok(set_status(doc, record.status)
        .set("trace_key", Value::String(record.trace_key.clone()))
        .set("rule_id", Value::String(record.rule_id.clone()))
        .set(
            "trigger_claim_id",
            Value::String(record.trigger_claim_id.clone()),
        )
        .set(
            "prior_trigger_claim_id",
            opt_string(&record.prior_trigger_claim_id),
        )
        .set(
            "derived_claim_id",
            Value::String(record.derived_claim_id.clone()),
        )
        .set(
            "target_subject_key",
            Value::String(record.target_subject_key.clone()),
        )
        .set(
            "target_predicate_key",
            Value::String(record.target_predicate_key.clone()),
        )
        .set(
            "target_subject_entity_id",
            opt_string(&record.target_subject_entity_id),
        )
        .set("target_slot_id", opt_string(&record.target_slot_id))
        .set("target_subject", opt_string(&record.target_subject))
        .set("target_predicate", opt_string(&record.target_predicate))
        .set(
            "state_kind",
            Value::String(record.state_kind.as_str().to_string()),
        )
        .set("hop", Value::I64(record.hop as i64))
        .set(
            "parent_trace_ids_json",
            json_string(&record.parent_trace_ids)?,
        )
        .set("source_spans_json", json_string(&record.source_span_ids)?)
        .set("source_eps_json", json_string(&record.source_episode_ids)?)
        .set("observed_at_ms", Value::I64(record.observed_at_ms))
        .set("valid_from_ms", opt_i64(record.valid_from_ms))
        .set("valid_to_ms", opt_i64(record.valid_to_ms))
        .set("projected_at_ms", Value::I64(record.projected_at_ms))
        .set("recorded_at_ms", Value::I64(record.recorded_at_ms))
        .set("superseded_at_ms", opt_i64(record.superseded_at_ms)))
}

pub fn dependency_trace_from_doc(doc: &Doc) -> ZResult<DependencyTraceRecord> {
    let scope = MemoryScope {
        space_id: required_string(doc, "space_id")?,
        tenant_id: optional_string(doc, "tenant_id")?,
        user_id: optional_string(doc, "user_id")?,
        agent_id: optional_string(doc, "agent_id")?,
        project_id: optional_string(doc, "project_id")?,
        thread_id: optional_string(doc, "thread_id")?,
    };
    let hop = required_i64(doc, "hop")?;
    if hop < 0 {
        return Err(row_error("dependency trace hop must be non-negative"));
    }
    let projected_at_ms = required_i64(doc, "projected_at_ms")?;
    let id = required_string(doc, "id")?;
    Ok(DependencyTraceRecord {
        trace_key: required_string(doc, "trace_key")?,
        id,
        scope,
        status: parse_status(&required_string(doc, "status")?)?,
        visibility: parse_visibility(&required_string(doc, "visibility")?)?,
        policy_tags: json_vec(doc, "policy_tags_json")?,
        rule_id: required_string(doc, "rule_id")?,
        trigger_claim_id: required_string(doc, "trigger_claim_id")?,
        prior_trigger_claim_id: optional_string(doc, "prior_trigger_claim_id")?,
        derived_claim_id: required_string(doc, "derived_claim_id")?,
        target_subject_key: required_string(doc, "target_subject_key")?,
        target_predicate_key: required_string(doc, "target_predicate_key")?,
        target_subject_entity_id: optional_string(doc, "target_subject_entity_id")?,
        target_slot_id: optional_string(doc, "target_slot_id")?,
        target_subject: optional_string(doc, "target_subject")?,
        target_predicate: optional_string(doc, "target_predicate")?,
        state_kind: parse_state_record_kind(&required_string(doc, "state_kind")?)?,
        hop: hop as usize,
        parent_trace_ids: json_vec(doc, "parent_trace_ids_json")?,
        source_span_ids: json_vec(doc, "source_spans_json")?,
        source_episode_ids: json_vec(doc, "source_eps_json")?,
        observed_at_ms: required_i64(doc, "observed_at_ms")?,
        valid_from_ms: optional_i64(doc, "valid_from_ms")?,
        valid_to_ms: optional_i64(doc, "valid_to_ms")?,
        projected_at_ms,
        recorded_at_ms: required_i64(doc, "recorded_at_ms")?,
        superseded_at_ms: optional_i64(doc, "superseded_at_ms")?,
    })
}

pub fn slot_doc(record: &CanonicalSlotRecord) -> Result<Doc, serde_json::Error> {
    let doc = set_scope(
        Doc::new(record.id.clone()).set("id", Value::String(record.id.clone())),
        &record.scope,
        record.visibility,
        &record.policy_tags,
    )?;
    Ok(set_status(doc, record.status)
        .set("slot_key", Value::String(record.slot_key.clone()))
        .set("subject_key", Value::String(record.subject_key.clone()))
        .set("predicate_key", Value::String(record.predicate_key.clone()))
        .set("subject_entity_id", opt_string(&record.subject_entity_id))
        .set("subject", opt_string(&record.subject))
        .set("predicate", opt_string(&record.predicate))
        .set(
            "source_claim_ids_json",
            json_string(&record.source_claim_ids)?,
        )
        .set(
            "source_rule_ids_json",
            json_string(&record.source_rule_ids)?,
        )
        .set(
            "source_entity_ids_json",
            json_string(&record.source_entity_ids)?,
        )
        .set("observed_at_ms", Value::I64(record.observed_at_ms))
        .set("valid_from_ms", opt_i64(record.valid_from_ms))
        .set("valid_to_ms", opt_i64(record.valid_to_ms))
        .set("projected_at_ms", Value::I64(record.projected_at_ms))
        .set("recorded_at_ms", Value::I64(record.recorded_at_ms))
        .set("superseded_at_ms", opt_i64(record.superseded_at_ms)))
}

pub fn slot_from_doc(doc: &Doc) -> ZResult<CanonicalSlotRecord> {
    let scope = MemoryScope {
        space_id: required_string(doc, "space_id")?,
        tenant_id: optional_string(doc, "tenant_id")?,
        user_id: optional_string(doc, "user_id")?,
        agent_id: optional_string(doc, "agent_id")?,
        project_id: optional_string(doc, "project_id")?,
        thread_id: optional_string(doc, "thread_id")?,
    };
    let projected_at_ms = required_i64(doc, "projected_at_ms")?;
    let id = required_string(doc, "id")?;
    Ok(CanonicalSlotRecord {
        slot_key: required_string(doc, "slot_key")?,
        id,
        scope,
        status: parse_status(&required_string(doc, "status")?)?,
        visibility: parse_visibility(&required_string(doc, "visibility")?)?,
        policy_tags: json_vec(doc, "policy_tags_json")?,
        subject_key: required_string(doc, "subject_key")?,
        predicate_key: required_string(doc, "predicate_key")?,
        subject_entity_id: optional_string(doc, "subject_entity_id")?,
        subject: optional_string(doc, "subject")?,
        predicate: optional_string(doc, "predicate")?,
        source_claim_ids: json_vec(doc, "source_claim_ids_json")?,
        source_rule_ids: json_vec(doc, "source_rule_ids_json")?,
        source_entity_ids: json_vec(doc, "source_entity_ids_json")?,
        observed_at_ms: required_i64(doc, "observed_at_ms")?,
        valid_from_ms: optional_i64(doc, "valid_from_ms")?,
        valid_to_ms: optional_i64(doc, "valid_to_ms")?,
        projected_at_ms,
        recorded_at_ms: required_i64(doc, "recorded_at_ms")?,
        superseded_at_ms: optional_i64(doc, "superseded_at_ms")?,
    })
}

pub fn episode_doc(record: &EpisodeRecord) -> Result<Doc, serde_json::Error> {
    let doc = set_scope(
        Doc::new(record.id.clone()).set("id", Value::String(record.id.clone())),
        &record.scope,
        record.visibility,
        &record.policy_tags,
    )?;
    Ok(set_status(doc, record.status)
        .set("source_kind", Value::String(enum_name(record.source_kind)?))
        .set("actor_kind", Value::String(enum_name(record.actor)?))
        .set("sequence_no", Value::I64(record.sequence_no))
        .set("created_at_ms", Value::I64(record.temporal.created_at_ms))
        .set("ingested_at_ms", Value::I64(record.temporal.ingested_at_ms))
        .set("event_time_ms", opt_i64(record.temporal.event_time_ms))
        .set("valid_from_ms", opt_i64(record.temporal.valid_from_ms))
        .set("valid_to_ms", opt_i64(record.temporal.valid_to_ms))
        .set("raw_text", Value::String(record.raw_text.clone()))
        .set("blob_ref", opt_string(&record.blob_ref))
        .set("mime_type", opt_string(&record.mime_type))
        .set("content_hash", Value::String(record.content_hash.clone()))
        .set("parent_ids_json", json_string(&record.causal_parent_ids)?)
        .set("metadata_json", opt_string(&record.metadata_json)))
}

pub fn artifact_doc(record: &ArtifactRecord) -> Result<Doc, serde_json::Error> {
    let doc = set_scope(
        Doc::new(record.id.clone()).set("id", Value::String(record.id.clone())),
        &record.scope,
        record.visibility,
        &record.policy_tags,
    )?;
    Ok(set_status(doc, record.status)
        .set(
            "artifact_kind",
            Value::String(enum_name(record.artifact_kind)?),
        )
        .set("title", Value::String(record.title.clone()))
        .set("uri", opt_string(&record.uri))
        .set("blob_ref", opt_string(&record.blob_ref))
        .set("mime_type", opt_string(&record.mime_type))
        .set("content_hash", Value::String(record.content_hash.clone()))
        .set("source_created_ms", opt_i64(record.source_created_at_ms))
        .set("source_modified_ms", opt_i64(record.source_modified_at_ms))
        .set("created_at_ms", Value::I64(record.created_at_ms))
        .set("ingested_at_ms", Value::I64(record.ingested_at_ms))
        .set("valid_from_ms", opt_i64(record.valid_from_ms))
        .set("valid_to_ms", opt_i64(record.valid_to_ms))
        .set("extracted_text_ref", opt_string(&record.extracted_text_ref))
        .set("metadata_json", opt_string(&record.metadata_json)))
}

pub fn artifact_from_doc(doc: &Doc) -> ZResult<ArtifactRecord> {
    let scope = MemoryScope {
        space_id: required_string(doc, "space_id")?,
        tenant_id: optional_string(doc, "tenant_id")?,
        user_id: optional_string(doc, "user_id")?,
        agent_id: optional_string(doc, "agent_id")?,
        project_id: optional_string(doc, "project_id")?,
        thread_id: optional_string(doc, "thread_id")?,
    };
    let artifact_kind = required_string(doc, "artifact_kind")?;
    Ok(ArtifactRecord {
        id: required_string(doc, "id")?,
        scope,
        status: parse_status(&required_string(doc, "status")?)?,
        visibility: parse_visibility(&required_string(doc, "visibility")?)?,
        policy_tags: json_vec(doc, "policy_tags_json")?,
        artifact_kind: parse_enum::<ArtifactKind>("artifact_kind", &artifact_kind)?,
        title: required_string(doc, "title")?,
        uri: optional_string(doc, "uri")?,
        blob_ref: optional_string(doc, "blob_ref")?,
        mime_type: optional_string(doc, "mime_type")?,
        content_hash: required_string(doc, "content_hash")?,
        source_created_at_ms: optional_i64(doc, "source_created_ms")?,
        source_modified_at_ms: optional_i64(doc, "source_modified_ms")?,
        created_at_ms: required_i64(doc, "created_at_ms")?,
        ingested_at_ms: required_i64(doc, "ingested_at_ms")?,
        valid_from_ms: optional_i64(doc, "valid_from_ms")?,
        valid_to_ms: optional_i64(doc, "valid_to_ms")?,
        extracted_text_ref: optional_string(doc, "extracted_text_ref")?,
        metadata_json: optional_string(doc, "metadata_json")?,
    })
}

pub fn span_doc(record: &SpanRecord, embedding: Option<&[f32]>) -> Result<Doc, serde_json::Error> {
    let doc = set_scope(
        Doc::new(record.id.clone()).set("id", Value::String(record.id.clone())),
        &record.scope,
        record.visibility,
        &record.policy_tags,
    )?;
    let embedding = embedding
        .map(|items| Value::VecF32(items.to_vec()))
        .unwrap_or(Value::Null);
    Ok(set_status(doc, record.status)
        .set("source_type", Value::String(enum_name(record.source_type)?))
        .set("source_id", Value::String(record.source_id.clone()))
        .set("byte_start", Value::I64(record.byte_start))
        .set("byte_end", Value::I64(record.byte_end))
        .set("char_start", Value::I64(record.char_start))
        .set("char_end", Value::I64(record.char_end))
        .set("token_start", opt_i64(record.token_start))
        .set("token_end", opt_i64(record.token_end))
        .set("span_index", Value::I64(record.span_index))
        .set("text", Value::String(record.text.clone()))
        .set("text_hash", Value::String(record.text_hash.clone()))
        .set(
            "chunker_version",
            Value::String(record.chunker_version.clone()),
        )
        .set("embedding_model", opt_string(&record.embedding_model))
        .set("embedding_version", opt_string(&record.embedding_version))
        .set("lexical_text", Value::String(record.lexical_text.clone()))
        .set("created_at_ms", Value::I64(record.created_at_ms))
        .set("valid_from_ms", opt_i64(record.valid_from_ms))
        .set("valid_to_ms", opt_i64(record.valid_to_ms))
        .set("provenance_json", json_string(&record.provenance)?)
        .set("embedding", embedding))
}

pub fn span_from_doc(doc: &Doc) -> ZResult<SpanRecord> {
    let scope = MemoryScope {
        space_id: required_string(doc, "space_id")?,
        tenant_id: optional_string(doc, "tenant_id")?,
        user_id: optional_string(doc, "user_id")?,
        agent_id: optional_string(doc, "agent_id")?,
        project_id: optional_string(doc, "project_id")?,
        thread_id: optional_string(doc, "thread_id")?,
    };
    Ok(SpanRecord {
        id: required_string(doc, "id")?,
        scope,
        status: parse_status(&required_string(doc, "status")?)?,
        visibility: parse_visibility(&required_string(doc, "visibility")?)?,
        policy_tags: json_vec(doc, "policy_tags_json")?,
        source_type: parse_source_type(&required_string(doc, "source_type")?)?,
        source_id: required_string(doc, "source_id")?,
        byte_start: required_i64(doc, "byte_start")?,
        byte_end: required_i64(doc, "byte_end")?,
        char_start: required_i64(doc, "char_start")?,
        char_end: required_i64(doc, "char_end")?,
        token_start: optional_i64(doc, "token_start")?,
        token_end: optional_i64(doc, "token_end")?,
        span_index: required_i64(doc, "span_index")?,
        text: required_string(doc, "text")?,
        text_hash: required_string(doc, "text_hash")?,
        chunker_version: required_string(doc, "chunker_version")?,
        embedding_model: optional_string(doc, "embedding_model")?,
        embedding_version: optional_string(doc, "embedding_version")?,
        lexical_text: required_string(doc, "lexical_text")?,
        created_at_ms: required_i64(doc, "created_at_ms")?,
        valid_from_ms: optional_i64(doc, "valid_from_ms")?,
        valid_to_ms: optional_i64(doc, "valid_to_ms")?,
        provenance: json_vec::<ProvenanceRef>(doc, "provenance_json")?,
    })
}

pub fn correction_doc(record: &CorrectionRecord) -> Result<Doc, serde_json::Error> {
    let doc = set_scope(
        Doc::new(record.id.clone()).set("id", Value::String(record.id.clone())),
        &record.scope,
        record.visibility,
        &record.policy_tags,
    )?;
    Ok(set_status(doc, record.status)
        .set("operation", Value::String(enum_name(record.operation)?))
        .set("target_type", Value::String(record.target_type.clone()))
        .set(
            "target_match",
            Value::String(record.target_match.as_str().to_string()),
        )
        .set("target_ids", Value::ArrayString(record.target_ids.clone()))
        .set("target_ids_json", json_string(&record.target_ids)?)
        .set("target_selector", opt_string(&record.target_selector))
        .set("target_subject_key", opt_string(&record.target_subject_key))
        .set(
            "target_predicate_key",
            opt_string(&record.target_predicate_key),
        )
        .set(
            "target_subject_entity_id",
            opt_string(&record.target_subject_entity_id),
        )
        .set("target_slot_id", opt_string(&record.target_slot_id))
        .set("new_value", opt_string(&record.new_value))
        .set("reason", opt_string(&record.reason))
        .set("actor_kind", Value::String(enum_name(record.actor)?))
        .set("authority", Value::String(enum_name(record.authority)?))
        .set("created_at_ms", Value::I64(record.created_at_ms))
        .set("effective_at_ms", Value::I64(record.effective_at_ms))
        .set("applies_from_ms", opt_i64(record.applies_valid_from_ms))
        .set("applies_to_ms", opt_i64(record.applies_valid_to_ms))
        .set("cascade_policy", opt_string(&record.cascade_policy))
        .set("source_spans_json", json_string(&record.source_span_ids)?)
        .set("source_eps_json", json_string(&record.source_episode_ids)?)
        .set("source_sequence_no", opt_i64(record.source_sequence_no))
        .set("metadata_json", opt_string(&record.metadata_json)))
}

pub fn claim_doc(
    record: &ClaimRecord,
    embedding: Option<&[f32]>,
) -> Result<Doc, serde_json::Error> {
    let doc = set_scope(
        Doc::new(record.id.clone()).set("id", Value::String(record.id.clone())),
        &record.scope,
        record.visibility,
        &record.policy_tags,
    )?;
    let embedding = embedding
        .map(|items| Value::VecF32(items.to_vec()))
        .unwrap_or(Value::Null);
    Ok(set_status(doc, record.status)
        .set("claim_text", Value::String(record.claim_text.clone()))
        .set("subject", opt_string(&record.subject))
        .set(
            "subject_key",
            record
                .subject
                .as_deref()
                .map(canonical_slot_part)
                .map(Value::String)
                .unwrap_or(Value::Null),
        )
        .set("predicate", opt_string(&record.predicate))
        .set("object_value", opt_string(&record.object_value))
        .set("subject_entity_id", opt_string(&record.subject_entity_id))
        .set("slot_id", opt_string(&record.slot_id))
        .set("slot_facet", opt_string(&record.slot_facet))
        .set("claim_kind", Value::String(enum_name(record.claim_kind)?))
        .set("polarity", Value::String(enum_name(record.polarity)?))
        .set("source_spans_json", json_string(&record.source_span_ids)?)
        .set("source_eps_json", json_string(&record.source_episode_ids)?)
        .set("source_sequence_no", opt_i64(record.source_sequence_no))
        .set("asserted_by", Value::String(record.asserted_by.clone()))
        .set("extractor_version", opt_string(&record.extractor_version))
        .set("confidence", opt_f32(record.confidence))
        .set("observed_at_ms", Value::I64(record.observed_at_ms))
        .set("valid_from_ms", opt_i64(record.valid_from_ms))
        .set("valid_to_ms", opt_i64(record.valid_to_ms))
        .set("correction_ids_json", json_string(&record.correction_ids)?)
        .set("embedding", embedding))
}

pub fn claim_from_doc(doc: &Doc) -> ZResult<ClaimRecord> {
    let scope = MemoryScope {
        space_id: required_string(doc, "space_id")?,
        tenant_id: optional_string(doc, "tenant_id")?,
        user_id: optional_string(doc, "user_id")?,
        agent_id: optional_string(doc, "agent_id")?,
        project_id: optional_string(doc, "project_id")?,
        thread_id: optional_string(doc, "thread_id")?,
    };
    let claim_kind = required_string(doc, "claim_kind")?;
    let polarity = required_string(doc, "polarity")?;
    Ok(ClaimRecord {
        id: required_string(doc, "id")?,
        scope,
        status: parse_status(&required_string(doc, "status")?)?,
        visibility: parse_visibility(&required_string(doc, "visibility")?)?,
        policy_tags: json_vec(doc, "policy_tags_json")?,
        claim_text: required_string(doc, "claim_text")?,
        subject: optional_string(doc, "subject")?,
        predicate: optional_string(doc, "predicate")?,
        object_value: optional_string(doc, "object_value")?,
        subject_entity_id: optional_string(doc, "subject_entity_id")?,
        slot_id: optional_string(doc, "slot_id")?,
        slot_facet: optional_string(doc, "slot_facet")?,
        claim_kind: parse_enum::<ClaimKind>("claim_kind", &claim_kind)?,
        polarity: parse_enum::<ClaimPolarity>("polarity", &polarity)?,
        source_span_ids: json_vec(doc, "source_spans_json")?,
        source_episode_ids: json_vec(doc, "source_eps_json")?,
        source_sequence_no: optional_i64(doc, "source_sequence_no")?,
        asserted_by: required_string(doc, "asserted_by")?,
        extractor_version: optional_string(doc, "extractor_version")?,
        confidence: optional_f32(doc, "confidence")?,
        observed_at_ms: required_i64(doc, "observed_at_ms")?,
        valid_from_ms: optional_i64(doc, "valid_from_ms")?,
        valid_to_ms: optional_i64(doc, "valid_to_ms")?,
        correction_ids: json_vec(doc, "correction_ids_json")?,
    })
}

pub fn rule_doc(record: &RuleRecord) -> Result<Doc, serde_json::Error> {
    let doc = set_scope(
        Doc::new(record.id.clone()).set("id", Value::String(record.id.clone())),
        &record.scope,
        record.visibility,
        &record.policy_tags,
    )?;
    Ok(set_status(doc, record.status)
        .set(
            "trigger_subject_key",
            Value::String(record.trigger_subject_key.clone()),
        )
        .set(
            "trigger_predicate_key",
            Value::String(record.trigger_predicate_key.clone()),
        )
        .set(
            "target_subject_key",
            Value::String(record.target_subject_key.clone()),
        )
        .set(
            "target_predicate_key",
            Value::String(record.target_predicate_key.clone()),
        )
        .set(
            "trigger_subject_entity_id",
            opt_string(&record.trigger_subject_entity_id),
        )
        .set(
            "target_subject_entity_id",
            opt_string(&record.target_subject_entity_id),
        )
        .set("trigger_slot_id", opt_string(&record.trigger_slot_id))
        .set("target_slot_id", opt_string(&record.target_slot_id))
        .set(
            "trigger_binding_status",
            Value::String(record.trigger_binding_status.as_str().to_string()),
        )
        .set(
            "target_binding_status",
            Value::String(record.target_binding_status.as_str().to_string()),
        )
        .set("trigger_subject", opt_string(&record.trigger_subject))
        .set("trigger_predicate", opt_string(&record.trigger_predicate))
        .set("target_subject", opt_string(&record.target_subject))
        .set("target_predicate", opt_string(&record.target_predicate))
        .set(
            "target_match",
            Value::String(record.target_match.as_str().to_string()),
        )
        .set(
            "activation",
            Value::String(record.activation.as_str().to_string()),
        )
        .set("action", Value::String(record.action.as_str().to_string()))
        .set("value_template", opt_string(&record.value_template))
        .set("value", opt_string(&record.value))
        .set(
            "source_span_ids_json",
            json_string(&record.source_span_ids)?,
        )
        .set("source_eps_json", json_string(&record.source_episode_ids)?)
        .set("source_sequence_no", opt_i64(record.source_sequence_no))
        .set("valid_from_ms", opt_i64(record.valid_from_ms))
        .set("valid_to_ms", opt_i64(record.valid_to_ms))
        .set("confidence", opt_f32(record.confidence)))
}

pub fn rule_from_doc(doc: &Doc) -> ZResult<RuleRecord> {
    let scope = MemoryScope {
        space_id: required_string(doc, "space_id")?,
        tenant_id: optional_string(doc, "tenant_id")?,
        user_id: optional_string(doc, "user_id")?,
        agent_id: optional_string(doc, "agent_id")?,
        project_id: optional_string(doc, "project_id")?,
        thread_id: optional_string(doc, "thread_id")?,
    };
    let action = parse_rule_action(&required_string(doc, "action")?)?;
    Ok(RuleRecord {
        id: required_string(doc, "id")?,
        scope,
        status: parse_status(&required_string(doc, "status")?)?,
        visibility: parse_visibility(&required_string(doc, "visibility")?)?,
        policy_tags: json_vec(doc, "policy_tags_json")?,
        trigger_subject_key: required_string(doc, "trigger_subject_key")?,
        trigger_predicate_key: required_string(doc, "trigger_predicate_key")?,
        target_subject_key: required_string(doc, "target_subject_key")?,
        target_predicate_key: required_string(doc, "target_predicate_key")?,
        trigger_subject_entity_id: optional_string(doc, "trigger_subject_entity_id")?,
        target_subject_entity_id: optional_string(doc, "target_subject_entity_id")?,
        trigger_slot_id: optional_string(doc, "trigger_slot_id")?,
        target_slot_id: optional_string(doc, "target_slot_id")?,
        trigger_binding_status: parse_rule_binding_status(&required_string(
            doc,
            "trigger_binding_status",
        )?)?,
        target_binding_status: parse_rule_binding_status(&required_string(
            doc,
            "target_binding_status",
        )?)?,
        trigger_subject: optional_string(doc, "trigger_subject")?,
        trigger_predicate: optional_string(doc, "trigger_predicate")?,
        target_subject: optional_string(doc, "target_subject")?,
        target_predicate: optional_string(doc, "target_predicate")?,
        target_match: parse_rule_target_match(&required_string(doc, "target_match")?)?,
        activation: parse_rule_activation(&required_string(doc, "activation")?)?,
        action,
        value_template: optional_string(doc, "value_template")?,
        value: optional_string(doc, "value")?,
        source_span_ids: json_vec(doc, "source_span_ids_json")?,
        source_episode_ids: json_vec(doc, "source_eps_json")?,
        source_sequence_no: optional_i64(doc, "source_sequence_no")?,
        valid_from_ms: optional_i64(doc, "valid_from_ms")?,
        valid_to_ms: optional_i64(doc, "valid_to_ms")?,
        confidence: optional_f32(doc, "confidence")?,
    })
}

pub fn profile_doc(record: &ProfileRecord) -> Result<Doc, serde_json::Error> {
    let doc = set_scope(
        Doc::new(record.id.clone()).set("id", Value::String(record.id.clone())),
        &record.scope,
        record.visibility,
        &record.policy_tags,
    )?;
    Ok(set_status(doc, record.status)
        .set("subject_id", opt_string(&record.subject_id))
        .set("profile_key", Value::String(record.profile_key.clone()))
        .set("profile_text", Value::String(record.profile_text.clone()))
        .set("value_json", opt_string(&record.value_json))
        .set(
            "evidence_claims_json",
            json_string(&record.evidence_claim_ids)?,
        )
        .set("source_spans_json", json_string(&record.source_span_ids)?)
        .set("generated_at_ms", Value::I64(record.generated_at_ms))
        .set(
            "generator_version",
            Value::String(record.generator_version.clone()),
        )
        .set("valid_from_ms", opt_i64(record.valid_from_ms))
        .set("valid_to_ms", opt_i64(record.valid_to_ms))
        .set(
            "correction_watermark",
            Value::I64(record.correction_watermark),
        ))
}

pub fn profile_from_doc(doc: &Doc) -> ZResult<ProfileRecord> {
    let scope = MemoryScope {
        space_id: required_string(doc, "space_id")?,
        tenant_id: optional_string(doc, "tenant_id")?,
        user_id: optional_string(doc, "user_id")?,
        agent_id: optional_string(doc, "agent_id")?,
        project_id: optional_string(doc, "project_id")?,
        thread_id: optional_string(doc, "thread_id")?,
    };
    Ok(ProfileRecord {
        id: required_string(doc, "id")?,
        scope,
        status: parse_status(&required_string(doc, "status")?)?,
        visibility: parse_visibility(&required_string(doc, "visibility")?)?,
        policy_tags: json_vec(doc, "policy_tags_json")?,
        subject_id: optional_string(doc, "subject_id")?,
        profile_key: required_string(doc, "profile_key")?,
        profile_text: required_string(doc, "profile_text")?,
        value_json: optional_string(doc, "value_json")?,
        evidence_claim_ids: json_vec(doc, "evidence_claims_json")?,
        source_span_ids: json_vec(doc, "source_spans_json")?,
        generated_at_ms: required_i64(doc, "generated_at_ms")?,
        generator_version: required_string(doc, "generator_version")?,
        valid_from_ms: optional_i64(doc, "valid_from_ms")?,
        valid_to_ms: optional_i64(doc, "valid_to_ms")?,
        correction_watermark: required_i64(doc, "correction_watermark")?,
    })
}

pub fn entity_doc(
    record: &EntityRecord,
    embedding: Option<&[f32]>,
) -> Result<Doc, serde_json::Error> {
    let doc = set_scope(
        Doc::new(record.id.clone()).set("id", Value::String(record.id.clone())),
        &record.scope,
        record.visibility,
        &record.policy_tags,
    )?;
    let embedding = embedding
        .map(|items| Value::VecF32(items.to_vec()))
        .unwrap_or(Value::Null);
    Ok(set_status(doc, record.status)
        .set("entity_type", Value::String(record.entity_type.clone()))
        .set(
            "canonical_name",
            Value::String(record.canonical_name.clone()),
        )
        .set("aliases_json", json_string(&record.aliases)?)
        .set("source_claims_json", json_string(&record.source_claim_ids)?)
        .set("merge_parents_json", json_string(&record.merge_parent_ids)?)
        .set("split_from_id", opt_string(&record.split_from_id))
        .set("confidence", opt_f32(record.confidence))
        .set("embedding", embedding))
}

pub fn entity_from_doc(doc: &Doc) -> ZResult<EntityRecord> {
    let scope = MemoryScope {
        space_id: required_string(doc, "space_id")?,
        tenant_id: optional_string(doc, "tenant_id")?,
        user_id: optional_string(doc, "user_id")?,
        agent_id: optional_string(doc, "agent_id")?,
        project_id: optional_string(doc, "project_id")?,
        thread_id: optional_string(doc, "thread_id")?,
    };
    Ok(EntityRecord {
        id: required_string(doc, "id")?,
        scope,
        status: parse_status(&required_string(doc, "status")?)?,
        visibility: parse_visibility(&required_string(doc, "visibility")?)?,
        policy_tags: json_vec(doc, "policy_tags_json")?,
        entity_type: required_string(doc, "entity_type")?,
        canonical_name: required_string(doc, "canonical_name")?,
        aliases: json_vec(doc, "aliases_json")?,
        source_claim_ids: json_vec(doc, "source_claims_json")?,
        merge_parent_ids: json_vec(doc, "merge_parents_json")?,
        split_from_id: optional_string(doc, "split_from_id")?,
        confidence: optional_f32(doc, "confidence")?,
    })
}

pub fn entity_alias_doc(record: &EntityAliasRecord) -> Result<Doc, serde_json::Error> {
    let doc = set_scope(
        Doc::new(record.id.clone()).set("id", Value::String(record.id.clone())),
        &record.scope,
        record.visibility,
        &record.policy_tags,
    )?;
    Ok(set_status(doc, record.status)
        .set("alias_key", Value::String(record.alias_key.clone()))
        .set("entity_id", Value::String(record.entity_id.clone()))
        .set("entity_type", Value::String(record.entity_type.clone()))
        .set("source_claims_json", json_string(&record.source_claim_ids)?)
        .set("recorded_at_ms", Value::I64(record.recorded_at_ms))
        .set("superseded_at_ms", opt_i64(record.superseded_at_ms)))
}

pub fn entity_alias_from_doc(doc: &Doc) -> ZResult<EntityAliasRecord> {
    Ok(EntityAliasRecord {
        id: required_string(doc, "id")?,
        scope: MemoryScope {
            space_id: required_string(doc, "space_id")?,
            tenant_id: optional_string(doc, "tenant_id")?,
            user_id: optional_string(doc, "user_id")?,
            agent_id: optional_string(doc, "agent_id")?,
            project_id: optional_string(doc, "project_id")?,
            thread_id: optional_string(doc, "thread_id")?,
        },
        status: parse_status(&required_string(doc, "status")?)?,
        visibility: parse_visibility(&required_string(doc, "visibility")?)?,
        policy_tags: json_vec(doc, "policy_tags_json")?,
        alias_key: required_string(doc, "alias_key")?,
        entity_id: required_string(doc, "entity_id")?,
        entity_type: required_string(doc, "entity_type")?,
        source_claim_ids: json_vec(doc, "source_claims_json")?,
        recorded_at_ms: required_i64(doc, "recorded_at_ms")?,
        superseded_at_ms: optional_i64(doc, "superseded_at_ms")?,
    })
}

pub fn slot_alias_doc(record: &SlotAliasRecord) -> Result<Doc, serde_json::Error> {
    let doc = set_scope(
        Doc::new(record.id.clone()).set("id", Value::String(record.id.clone())),
        &record.scope,
        record.visibility,
        &record.policy_tags,
    )?;
    Ok(set_status(doc, record.status)
        .set(
            "alias_subject_key",
            Value::String(record.alias_subject_key.clone()),
        )
        .set(
            "alias_predicate_key",
            Value::String(record.alias_predicate_key.clone()),
        )
        .set(
            "alias_subject_entity_id",
            opt_string(&record.alias_subject_entity_id),
        )
        .set(
            "canonical_slot_id",
            Value::String(record.canonical_slot_id.clone()),
        )
        .set("source_claims_json", json_string(&record.source_claim_ids)?)
        .set("valid_from_ms", opt_i64(record.valid_from_ms))
        .set("valid_to_ms", opt_i64(record.valid_to_ms))
        .set("recorded_at_ms", Value::I64(record.recorded_at_ms))
        .set("superseded_at_ms", opt_i64(record.superseded_at_ms)))
}

pub fn slot_alias_from_doc(doc: &Doc) -> ZResult<SlotAliasRecord> {
    Ok(SlotAliasRecord {
        id: required_string(doc, "id")?,
        scope: MemoryScope {
            space_id: required_string(doc, "space_id")?,
            tenant_id: optional_string(doc, "tenant_id")?,
            user_id: optional_string(doc, "user_id")?,
            agent_id: optional_string(doc, "agent_id")?,
            project_id: optional_string(doc, "project_id")?,
            thread_id: optional_string(doc, "thread_id")?,
        },
        status: parse_status(&required_string(doc, "status")?)?,
        visibility: parse_visibility(&required_string(doc, "visibility")?)?,
        policy_tags: json_vec(doc, "policy_tags_json")?,
        alias_subject_key: required_string(doc, "alias_subject_key")?,
        alias_predicate_key: required_string(doc, "alias_predicate_key")?,
        alias_subject_entity_id: optional_string(doc, "alias_subject_entity_id")?,
        canonical_slot_id: required_string(doc, "canonical_slot_id")?,
        source_claim_ids: json_vec(doc, "source_claims_json")?,
        valid_from_ms: optional_i64(doc, "valid_from_ms")?,
        valid_to_ms: optional_i64(doc, "valid_to_ms")?,
        recorded_at_ms: required_i64(doc, "recorded_at_ms")?,
        superseded_at_ms: optional_i64(doc, "superseded_at_ms")?,
    })
}

pub fn edge_doc(record: &EdgeRecord) -> Result<Doc, serde_json::Error> {
    let doc = set_scope(
        Doc::new(record.id.clone()).set("id", Value::String(record.id.clone())),
        &record.scope,
        record.visibility,
        &record.policy_tags,
    )?;
    Ok(set_status(doc, record.status)
        .set("src_entity_id", Value::String(record.src_entity_id.clone()))
        .set("dst_entity_id", Value::String(record.dst_entity_id.clone()))
        .set("relation_type", Value::String(record.relation_type.clone()))
        .set("claim_id", opt_string(&record.claim_id))
        .set("source_spans_json", json_string(&record.source_span_ids)?)
        .set("valid_from_ms", opt_i64(record.valid_from_ms))
        .set("valid_to_ms", opt_i64(record.valid_to_ms))
        .set("confidence", opt_f32(record.confidence)))
}

pub fn edge_from_doc(doc: &Doc) -> ZResult<EdgeRecord> {
    let scope = MemoryScope {
        space_id: required_string(doc, "space_id")?,
        tenant_id: optional_string(doc, "tenant_id")?,
        user_id: optional_string(doc, "user_id")?,
        agent_id: optional_string(doc, "agent_id")?,
        project_id: optional_string(doc, "project_id")?,
        thread_id: optional_string(doc, "thread_id")?,
    };
    Ok(EdgeRecord {
        id: required_string(doc, "id")?,
        scope,
        status: parse_status(&required_string(doc, "status")?)?,
        visibility: parse_visibility(&required_string(doc, "visibility")?)?,
        policy_tags: json_vec(doc, "policy_tags_json")?,
        src_entity_id: required_string(doc, "src_entity_id")?,
        dst_entity_id: required_string(doc, "dst_entity_id")?,
        relation_type: required_string(doc, "relation_type")?,
        claim_id: optional_string(doc, "claim_id")?,
        source_span_ids: json_vec(doc, "source_spans_json")?,
        valid_from_ms: optional_i64(doc, "valid_from_ms")?,
        valid_to_ms: optional_i64(doc, "valid_to_ms")?,
        confidence: optional_f32(doc, "confidence")?,
    })
}

pub fn correction_from_doc(doc: &Doc) -> ZResult<CorrectionRecord> {
    let scope = MemoryScope {
        space_id: required_string(doc, "space_id")?,
        tenant_id: optional_string(doc, "tenant_id")?,
        user_id: optional_string(doc, "user_id")?,
        agent_id: optional_string(doc, "agent_id")?,
        project_id: optional_string(doc, "project_id")?,
        thread_id: optional_string(doc, "thread_id")?,
    };
    let operation = required_string(doc, "operation")?;
    let target_match = required_string(doc, "target_match")?;
    let actor = required_string(doc, "actor_kind")?;
    let authority = required_string(doc, "authority")?;
    Ok(CorrectionRecord {
        id: required_string(doc, "id")?,
        scope,
        status: parse_status(&required_string(doc, "status")?)?,
        visibility: parse_visibility(&required_string(doc, "visibility")?)?,
        policy_tags: json_vec(doc, "policy_tags_json")?,
        operation: parse_enum::<CorrectionOperation>("operation", &operation)?,
        target_type: required_string(doc, "target_type")?,
        target_match: parse_enum::<CorrectionTargetMatch>("target_match", &target_match)?,
        target_ids: json_vec(doc, "target_ids_json")?,
        target_selector: optional_string(doc, "target_selector")?,
        target_subject_key: optional_string(doc, "target_subject_key")?,
        target_predicate_key: optional_string(doc, "target_predicate_key")?,
        target_subject_entity_id: optional_string(doc, "target_subject_entity_id")?,
        target_slot_id: optional_string(doc, "target_slot_id")?,
        new_value: optional_string(doc, "new_value")?,
        reason: optional_string(doc, "reason")?,
        actor: parse_enum::<ActorKind>("actor_kind", &actor)?,
        authority: parse_enum::<CorrectionAuthority>("authority", &authority)?,
        created_at_ms: required_i64(doc, "created_at_ms")?,
        effective_at_ms: required_i64(doc, "effective_at_ms")?,
        applies_valid_from_ms: optional_i64(doc, "applies_from_ms")?,
        applies_valid_to_ms: optional_i64(doc, "applies_to_ms")?,
        cascade_policy: optional_string(doc, "cascade_policy")?,
        source_span_ids: json_vec(doc, "source_spans_json")?,
        source_episode_ids: json_vec(doc, "source_eps_json")?,
        source_sequence_no: optional_i64(doc, "source_sequence_no")?,
        metadata_json: optional_string(doc, "metadata_json")?,
    })
}

#[cfg(test)]
mod tests;

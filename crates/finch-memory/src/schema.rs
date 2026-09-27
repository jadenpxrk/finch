use finch_types::{
    CollectionSchema, DataType, FieldSchema, HnswIndexParams, IndexParams, MetricType,
};

pub const EPISODES_COLLECTION: &str = "memory_episodes";
pub const SPANS_COLLECTION: &str = "memory_spans";
pub const ARTIFACTS_COLLECTION: &str = "memory_artifacts";
pub const CORRECTIONS_COLLECTION: &str = "memory_corrections";
pub const TERMS_COLLECTION: &str = "memory_terms";

pub const CLAIMS_COLLECTION: &str = "memory_claims";
pub const PROFILES_COLLECTION: &str = "memory_profiles";
pub const ENTITIES_COLLECTION: &str = "memory_entities";
pub const ENTITY_ALIASES_COLLECTION: &str = "memory_entity_aliases";
pub const EDGES_COLLECTION: &str = "memory_edges";
pub const RULES_COLLECTION: &str = "memory_rules";
pub const STATE_RECORDS_COLLECTION: &str = "memory_state_records";
pub const DEPENDENCY_TRACES_COLLECTION: &str = "memory_dependency_traces";
pub const SLOTS_COLLECTION: &str = "memory_slots";
pub const SLOT_ALIASES_COLLECTION: &str = "memory_slot_aliases";

fn string_field(name: &str) -> FieldSchema {
    FieldSchema::new(name, DataType::String)
}

fn i64_field(name: &str) -> FieldSchema {
    FieldSchema::new(name, DataType::Int64)
}

fn f32_field(name: &str) -> FieldSchema {
    FieldSchema::new(name, DataType::Float32)
}

fn array_string_field(name: &str) -> FieldSchema {
    FieldSchema::new(name, DataType::ArrayString)
}

fn vector_field(name: &str, dim: usize) -> FieldSchema {
    vector_field_with_hnsw(name, dim, HnswIndexParams::new(MetricType::Cosine))
}

fn vector_field_with_hnsw(name: &str, dim: usize, params: HnswIndexParams) -> FieldSchema {
    FieldSchema::new(name, DataType::VectorFp32)
        .with_dimension(dim)
        .with_index(IndexParams::Hnsw(params))
}

fn add_scope_fields(schema: CollectionSchema) -> CollectionSchema {
    schema
        .with_field(string_field("space_id"))
        .with_field(string_field("tenant_id"))
        .with_field(string_field("user_id"))
        .with_field(string_field("agent_id"))
        .with_field(string_field("project_id"))
        .with_field(string_field("thread_id"))
        .with_field(string_field("visibility"))
        .with_field(string_field("policy_tags_json"))
}

pub fn episode_schema() -> CollectionSchema {
    add_scope_fields(CollectionSchema::new(EPISODES_COLLECTION))
        .with_field(string_field("id"))
        .with_field(string_field("status"))
        .with_field(string_field("source_kind"))
        .with_field(string_field("actor_kind"))
        .with_field(i64_field("sequence_no"))
        .with_field(i64_field("created_at_ms"))
        .with_field(i64_field("ingested_at_ms"))
        .with_field(i64_field("event_time_ms"))
        .with_field(i64_field("valid_from_ms"))
        .with_field(i64_field("valid_to_ms"))
        .with_field(string_field("raw_text"))
        .with_field(string_field("blob_ref"))
        .with_field(string_field("mime_type"))
        .with_field(string_field("content_hash"))
        .with_field(string_field("parent_ids_json"))
        .with_field(string_field("metadata_json"))
}

pub fn artifact_schema() -> CollectionSchema {
    add_scope_fields(CollectionSchema::new(ARTIFACTS_COLLECTION))
        .with_field(string_field("id"))
        .with_field(string_field("status"))
        .with_field(string_field("artifact_kind"))
        .with_field(string_field("title"))
        .with_field(string_field("uri"))
        .with_field(string_field("blob_ref"))
        .with_field(string_field("mime_type"))
        .with_field(string_field("content_hash"))
        .with_field(i64_field("source_created_ms"))
        .with_field(i64_field("source_modified_ms"))
        .with_field(i64_field("created_at_ms"))
        .with_field(i64_field("ingested_at_ms"))
        .with_field(i64_field("valid_from_ms"))
        .with_field(i64_field("valid_to_ms"))
        .with_field(string_field("extracted_text_ref"))
        .with_field(string_field("metadata_json"))
}

pub fn span_schema(embedding_dim: usize) -> CollectionSchema {
    span_schema_with_hnsw(embedding_dim, HnswIndexParams::new(MetricType::Cosine))
}

pub fn span_schema_with_hnsw(
    embedding_dim: usize,
    hnsw_params: HnswIndexParams,
) -> CollectionSchema {
    add_scope_fields(CollectionSchema::new(SPANS_COLLECTION))
        .with_field(string_field("id"))
        .with_field(string_field("status"))
        .with_field(string_field("source_type"))
        .with_field(string_field("source_id"))
        .with_field(i64_field("byte_start"))
        .with_field(i64_field("byte_end"))
        .with_field(i64_field("char_start"))
        .with_field(i64_field("char_end"))
        .with_field(i64_field("token_start"))
        .with_field(i64_field("token_end"))
        .with_field(i64_field("span_index"))
        .with_field(string_field("text"))
        .with_field(string_field("text_hash"))
        .with_field(string_field("chunker_version"))
        .with_field(string_field("embedding_model"))
        .with_field(string_field("embedding_version"))
        .with_field(string_field("lexical_text"))
        .with_field(i64_field("created_at_ms"))
        .with_field(i64_field("valid_from_ms"))
        .with_field(i64_field("valid_to_ms"))
        .with_field(string_field("provenance_json"))
        .with_field(vector_field_with_hnsw(
            "embedding",
            embedding_dim,
            hnsw_params,
        ))
}

pub fn correction_schema() -> CollectionSchema {
    add_scope_fields(CollectionSchema::new(CORRECTIONS_COLLECTION))
        .with_field(string_field("id"))
        .with_field(string_field("status"))
        .with_field(string_field("operation"))
        .with_field(string_field("target_type"))
        .with_field(string_field("target_match"))
        .with_field(array_string_field("target_ids"))
        .with_field(string_field("target_ids_json"))
        .with_field(string_field("target_selector"))
        .with_field(string_field("target_subject_key"))
        .with_field(string_field("target_predicate_key"))
        .with_field(string_field("target_subject_entity_id"))
        .with_field(string_field("target_slot_id"))
        .with_field(string_field("new_value"))
        .with_field(string_field("reason"))
        .with_field(string_field("actor_kind"))
        .with_field(string_field("authority"))
        .with_field(i64_field("created_at_ms"))
        .with_field(i64_field("effective_at_ms"))
        .with_field(i64_field("applies_from_ms"))
        .with_field(i64_field("applies_to_ms"))
        .with_field(string_field("cascade_policy"))
        .with_field(string_field("source_spans_json"))
        .with_field(string_field("source_eps_json"))
        .with_field(i64_field("source_sequence_no"))
        .with_field(string_field("metadata_json"))
}

pub fn term_schema() -> CollectionSchema {
    add_scope_fields(CollectionSchema::new(TERMS_COLLECTION))
        .with_field(string_field("id"))
        .with_field(string_field("term"))
        .with_field(string_field("span_id"))
        .with_field(i64_field("tf"))
        .with_field(i64_field("doc_len"))
}

pub fn claim_schema(embedding_dim: usize) -> CollectionSchema {
    add_scope_fields(CollectionSchema::new(CLAIMS_COLLECTION))
        .with_field(string_field("id"))
        .with_field(string_field("status"))
        .with_field(string_field("claim_text"))
        .with_field(string_field("subject"))
        .with_field(string_field("subject_key"))
        .with_field(string_field("predicate"))
        .with_field(string_field("object_value"))
        .with_field(string_field("subject_entity_id"))
        .with_field(string_field("slot_id"))
        .with_field(string_field("slot_facet"))
        .with_field(string_field("claim_kind"))
        .with_field(string_field("polarity"))
        .with_field(string_field("source_spans_json"))
        .with_field(string_field("source_eps_json"))
        .with_field(i64_field("source_sequence_no"))
        .with_field(string_field("asserted_by"))
        .with_field(string_field("extractor_version"))
        .with_field(f32_field("confidence"))
        .with_field(i64_field("observed_at_ms"))
        .with_field(i64_field("valid_from_ms"))
        .with_field(i64_field("valid_to_ms"))
        .with_field(string_field("correction_ids_json"))
        .with_field(vector_field("embedding", embedding_dim))
}

pub fn profile_schema() -> CollectionSchema {
    add_scope_fields(CollectionSchema::new(PROFILES_COLLECTION))
        .with_field(string_field("id"))
        .with_field(string_field("status"))
        .with_field(string_field("subject_id"))
        .with_field(string_field("profile_key"))
        .with_field(string_field("profile_text"))
        .with_field(string_field("value_json"))
        .with_field(string_field("evidence_claims_json"))
        .with_field(string_field("source_spans_json"))
        .with_field(i64_field("generated_at_ms"))
        .with_field(string_field("generator_version"))
        .with_field(i64_field("valid_from_ms"))
        .with_field(i64_field("valid_to_ms"))
        .with_field(i64_field("correction_watermark"))
}

pub fn entity_schema(embedding_dim: usize) -> CollectionSchema {
    add_scope_fields(CollectionSchema::new(ENTITIES_COLLECTION))
        .with_field(string_field("id"))
        .with_field(string_field("status"))
        .with_field(string_field("entity_type"))
        .with_field(string_field("canonical_name"))
        .with_field(string_field("aliases_json"))
        .with_field(string_field("source_claims_json"))
        .with_field(string_field("merge_parents_json"))
        .with_field(string_field("split_from_id"))
        .with_field(f32_field("confidence"))
        .with_field(vector_field("embedding", embedding_dim))
}

pub fn entity_alias_schema() -> CollectionSchema {
    add_scope_fields(CollectionSchema::new(ENTITY_ALIASES_COLLECTION))
        .with_field(string_field("id"))
        .with_field(string_field("status"))
        .with_field(string_field("alias_key"))
        .with_field(string_field("entity_id"))
        .with_field(string_field("entity_type"))
        .with_field(string_field("source_claims_json"))
        .with_field(i64_field("recorded_at_ms"))
        .with_field(i64_field("superseded_at_ms"))
}

pub fn slot_alias_schema() -> CollectionSchema {
    add_scope_fields(CollectionSchema::new(SLOT_ALIASES_COLLECTION))
        .with_field(string_field("id"))
        .with_field(string_field("status"))
        .with_field(string_field("alias_subject_key"))
        .with_field(string_field("alias_predicate_key"))
        .with_field(string_field("alias_subject_entity_id"))
        .with_field(string_field("canonical_slot_id"))
        .with_field(string_field("source_claims_json"))
        .with_field(i64_field("valid_from_ms"))
        .with_field(i64_field("valid_to_ms"))
        .with_field(i64_field("recorded_at_ms"))
        .with_field(i64_field("superseded_at_ms"))
}

pub fn edge_schema() -> CollectionSchema {
    add_scope_fields(CollectionSchema::new(EDGES_COLLECTION))
        .with_field(string_field("id"))
        .with_field(string_field("status"))
        .with_field(string_field("src_entity_id"))
        .with_field(string_field("dst_entity_id"))
        .with_field(string_field("relation_type"))
        .with_field(string_field("claim_id"))
        .with_field(string_field("source_spans_json"))
        .with_field(i64_field("valid_from_ms"))
        .with_field(i64_field("valid_to_ms"))
        .with_field(f32_field("confidence"))
}

pub fn rule_schema() -> CollectionSchema {
    add_scope_fields(CollectionSchema::new(RULES_COLLECTION))
        .with_field(string_field("id"))
        .with_field(string_field("status"))
        .with_field(string_field("trigger_subject_key"))
        .with_field(string_field("trigger_predicate_key"))
        .with_field(string_field("target_subject_key"))
        .with_field(string_field("target_predicate_key"))
        .with_field(string_field("trigger_subject_entity_id"))
        .with_field(string_field("target_subject_entity_id"))
        .with_field(string_field("trigger_slot_id"))
        .with_field(string_field("target_slot_id"))
        .with_field(string_field("trigger_binding_status"))
        .with_field(string_field("target_binding_status"))
        .with_field(string_field("trigger_subject"))
        .with_field(string_field("trigger_predicate"))
        .with_field(string_field("target_subject"))
        .with_field(string_field("target_predicate"))
        .with_field(string_field("target_match"))
        .with_field(string_field("activation"))
        .with_field(string_field("action"))
        .with_field(string_field("value_template"))
        .with_field(string_field("value"))
        .with_field(string_field("source_span_ids_json"))
        .with_field(string_field("source_eps_json"))
        .with_field(i64_field("source_sequence_no"))
        .with_field(i64_field("valid_from_ms"))
        .with_field(i64_field("valid_to_ms"))
        .with_field(f32_field("confidence"))
}

pub fn state_record_schema() -> CollectionSchema {
    add_scope_fields(CollectionSchema::new(STATE_RECORDS_COLLECTION))
        .with_field(string_field("id"))
        .with_field(string_field("state_key"))
        .with_field(string_field("status"))
        .with_field(string_field("state_kind"))
        .with_field(string_field("claim_kind"))
        .with_field(string_field("subject"))
        .with_field(string_field("predicate"))
        .with_field(string_field("object_value"))
        .with_field(string_field("subject_entity_id"))
        .with_field(string_field("slot_id"))
        .with_field(string_field("slot_facet"))
        .with_field(string_field("state_text"))
        .with_field(string_field("members_json"))
        .with_field(string_field("claim_ids_json"))
        .with_field(string_field("correction_ids_json"))
        .with_field(string_field("rule_ids_json"))
        .with_field(string_field("source_spans_json"))
        .with_field(string_field("source_eps_json"))
        .with_field(i64_field("source_sequence_no"))
        .with_field(i64_field("observed_at_ms"))
        .with_field(i64_field("valid_from_ms"))
        .with_field(i64_field("valid_to_ms"))
        .with_field(i64_field("projected_at_ms"))
        .with_field(i64_field("recorded_at_ms"))
        .with_field(i64_field("superseded_at_ms"))
}

pub fn dependency_trace_schema() -> CollectionSchema {
    add_scope_fields(CollectionSchema::new(DEPENDENCY_TRACES_COLLECTION))
        .with_field(string_field("id"))
        .with_field(string_field("trace_key"))
        .with_field(string_field("status"))
        .with_field(string_field("rule_id"))
        .with_field(string_field("trigger_claim_id"))
        .with_field(string_field("prior_trigger_claim_id"))
        .with_field(string_field("derived_claim_id"))
        .with_field(string_field("target_subject_key"))
        .with_field(string_field("target_predicate_key"))
        .with_field(string_field("target_subject_entity_id"))
        .with_field(string_field("target_slot_id"))
        .with_field(string_field("target_subject"))
        .with_field(string_field("target_predicate"))
        .with_field(string_field("state_kind"))
        .with_field(i64_field("hop"))
        .with_field(string_field("parent_trace_ids_json"))
        .with_field(string_field("source_spans_json"))
        .with_field(string_field("source_eps_json"))
        .with_field(i64_field("observed_at_ms"))
        .with_field(i64_field("valid_from_ms"))
        .with_field(i64_field("valid_to_ms"))
        .with_field(i64_field("projected_at_ms"))
        .with_field(i64_field("recorded_at_ms"))
        .with_field(i64_field("superseded_at_ms"))
}

pub fn slot_schema() -> CollectionSchema {
    add_scope_fields(CollectionSchema::new(SLOTS_COLLECTION))
        .with_field(string_field("id"))
        .with_field(string_field("slot_key"))
        .with_field(string_field("status"))
        .with_field(string_field("subject_key"))
        .with_field(string_field("predicate_key"))
        .with_field(string_field("subject_entity_id"))
        .with_field(string_field("subject"))
        .with_field(string_field("predicate"))
        .with_field(string_field("source_claim_ids_json"))
        .with_field(string_field("source_rule_ids_json"))
        .with_field(string_field("source_entity_ids_json"))
        .with_field(i64_field("observed_at_ms"))
        .with_field(i64_field("valid_from_ms"))
        .with_field(i64_field("valid_to_ms"))
        .with_field(i64_field("projected_at_ms"))
        .with_field(i64_field("recorded_at_ms"))
        .with_field(i64_field("superseded_at_ms"))
}

pub fn active_collection_schemas(embedding_dim: usize) -> Vec<CollectionSchema> {
    vec![
        episode_schema(),
        artifact_schema(),
        span_schema(embedding_dim),
        correction_schema(),
        term_schema(),
        claim_schema(embedding_dim),
        profile_schema(),
        entity_schema(embedding_dim),
        entity_alias_schema(),
        edge_schema(),
        rule_schema(),
        state_record_schema(),
        dependency_trace_schema(),
        slot_schema(),
        slot_alias_schema(),
    ]
}

pub fn future_collection_schemas(embedding_dim: usize) -> Vec<CollectionSchema> {
    let _ = embedding_dim;
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_memory_schemas_validate() {
        for schema in active_collection_schemas(1536) {
            schema.validate().unwrap();
        }
    }

    #[test]
    fn future_memory_schemas_validate() {
        for schema in future_collection_schemas(1536) {
            schema.validate().unwrap();
        }
    }

    #[test]
    fn span_schema_contains_vector_embedding() {
        let schema = span_schema(384);
        let embedding = schema.get_field("embedding").unwrap();
        assert!(embedding.is_vector());
        assert_eq!(embedding.dimension, Some(384));
        assert!(embedding.is_indexed());
    }

    #[test]
    fn all_schemas_include_scope_fields() {
        for schema in active_collection_schemas(128)
            .into_iter()
            .chain(future_collection_schemas(128))
        {
            assert!(schema.has_field("space_id"));
            assert!(schema.has_field("visibility"));
            assert!(schema.has_field("policy_tags_json"));
        }
    }
}

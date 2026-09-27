use crate::types::{
    ActorKind, ArtifactRecord, ClaimKind, ClaimPolarity, ClaimRecord, CorrectionAuthority,
    CorrectionOperation, CorrectionRecord, EdgeRecord, EntityRecord, EpisodeRecord, MemoryId,
    MemoryScope, MemoryStatus, ProfileRecord, ProvenanceRef, SourceKind, SourceType, SpanRecord,
    TemporalFields, Visibility,
};
pub use crate::types::{EdgeInput, EntityInput, ProfileInput};
use serde::{Deserialize, Serialize};

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkOptions {
    pub max_chars: usize,
    pub overlap_chars: usize,
    pub chunker_version: String,
}

impl Default for ChunkOptions {
    fn default() -> Self {
        Self {
            max_chars: 1_200,
            overlap_chars: 200,
            chunker_version: "finch-memory-v1".to_string(),
        }
    }
}

impl ChunkOptions {
    fn normalized(&self) -> Self {
        let max_chars = self.max_chars.max(1);
        Self {
            max_chars,
            overlap_chars: self.overlap_chars.min(max_chars.saturating_sub(1)),
            chunker_version: self.chunker_version.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeInput {
    pub id: Option<MemoryId>,
    pub scope: MemoryScope,
    pub visibility: Visibility,
    pub policy_tags: Vec<String>,
    pub source_kind: SourceKind,
    pub actor: ActorKind,
    pub sequence_no: i64,
    pub event_time_ms: Option<i64>,
    pub valid_from_ms: Option<i64>,
    pub valid_to_ms: Option<i64>,
    pub raw_text: String,
    pub blob_ref: Option<String>,
    pub mime_type: Option<String>,
    pub causal_parent_ids: Vec<MemoryId>,
    pub metadata_json: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestedEpisode {
    pub episode: EpisodeRecord,
    pub spans: Vec<SpanRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestedArtifact {
    pub artifact: ArtifactRecord,
    pub spans: Vec<SpanRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionInput {
    pub id: Option<MemoryId>,
    pub scope: MemoryScope,
    pub visibility: Visibility,
    pub policy_tags: Vec<String>,
    pub operation: CorrectionOperation,
    pub target_type: String,
    pub target_ids: Vec<MemoryId>,
    pub target_selector: Option<String>,
    pub new_value: Option<String>,
    pub reason: Option<String>,
    pub actor: ActorKind,
    pub authority: CorrectionAuthority,
    pub effective_at_ms: Option<i64>,
    pub applies_valid_from_ms: Option<i64>,
    pub applies_valid_to_ms: Option<i64>,
    pub cascade_policy: Option<String>,
    pub metadata_json: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManualClaimInput {
    pub id: Option<MemoryId>,
    pub scope: MemoryScope,
    pub visibility: Visibility,
    pub policy_tags: Vec<String>,
    pub claim_text: String,
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object_value: Option<String>,
    pub claim_kind: ClaimKind,
    pub polarity: ClaimPolarity,
    pub source_span_ids: Vec<MemoryId>,
    pub source_episode_ids: Vec<MemoryId>,
    pub asserted_by: String,
    pub confidence: Option<f32>,
    pub observed_at_ms: i64,
    pub valid_from_ms: Option<i64>,
    pub valid_to_ms: Option<i64>,
}

pub fn stable_hash_hex(parts: &[&str]) -> String {
    let mut hash = FNV_OFFSET;
    for part in parts {
        for byte in part.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash ^= 0xff;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{hash:016x}")
}

pub fn ingest_episode(
    input: EpisodeInput,
    ingested_at_ms: i64,
    chunk_options: &ChunkOptions,
) -> IngestedEpisode {
    let content_hash = stable_hash_hex(&[&input.raw_text]);
    let id = input.id.unwrap_or_else(|| {
        let sequence_no = input.sequence_no.to_string();
        let event_time_ms = input.event_time_ms.unwrap_or(ingested_at_ms).to_string();
        let scope_hash = narrower_scope_hash(&input.scope);
        let mut parts = vec![
            input.scope.space_id.as_str(),
            &sequence_no,
            &event_time_ms,
            &content_hash,
        ];
        parts.extend(scope_hash.as_deref());
        stable_memory_id("ep", &parts)
    });
    let temporal = TemporalFields {
        created_at_ms: ingested_at_ms,
        ingested_at_ms,
        event_time_ms: input.event_time_ms,
        valid_from_ms: input.valid_from_ms.or(input.event_time_ms),
        valid_to_ms: input.valid_to_ms,
    };
    let episode = EpisodeRecord {
        id: id.clone(),
        scope: input.scope.clone(),
        status: MemoryStatus::Active,
        visibility: input.visibility,
        policy_tags: input.policy_tags.clone(),
        source_kind: input.source_kind,
        actor: input.actor,
        sequence_no: input.sequence_no,
        temporal: temporal.clone(),
        raw_text: input.raw_text,
        blob_ref: input.blob_ref,
        mime_type: input.mime_type,
        content_hash,
        causal_parent_ids: input.causal_parent_ids,
        metadata_json: input.metadata_json,
    };
    let spans = chunk_text(
        &ChunkSource {
            source_type: SourceType::Episode,
            source_id: &id,
            source_actor: Some(episode.actor),
            scope: &input.scope,
            visibility: episode.visibility,
            policy_tags: &episode.policy_tags,
            valid_from_ms: temporal.valid_from_ms,
            valid_to_ms: temporal.valid_to_ms,
            created_at_ms: ingested_at_ms,
        },
        &episode.raw_text,
        chunk_options,
    );
    IngestedEpisode { episode, spans }
}

pub fn chunk_artifact_text(
    artifact: &ArtifactRecord,
    text: &str,
    chunk_options: &ChunkOptions,
) -> Vec<SpanRecord> {
    chunk_text(
        &ChunkSource {
            source_type: SourceType::Artifact,
            source_id: &artifact.id,
            source_actor: None,
            scope: &artifact.scope,
            visibility: artifact.visibility,
            policy_tags: &artifact.policy_tags,
            valid_from_ms: artifact.valid_from_ms,
            valid_to_ms: artifact.valid_to_ms,
            created_at_ms: artifact.ingested_at_ms,
        },
        text,
        chunk_options,
    )
}

pub fn chunk_episode(episode: &EpisodeRecord, chunk_options: &ChunkOptions) -> Vec<SpanRecord> {
    chunk_text(
        &ChunkSource {
            source_type: SourceType::Episode,
            source_id: &episode.id,
            source_actor: Some(episode.actor),
            scope: &episode.scope,
            visibility: episode.visibility,
            policy_tags: &episode.policy_tags,
            valid_from_ms: episode.temporal.valid_from_ms,
            valid_to_ms: episode.temporal.valid_to_ms,
            created_at_ms: episode.temporal.ingested_at_ms,
        },
        &episode.raw_text,
        chunk_options,
    )
}

pub fn add_correction(input: CorrectionInput, created_at_ms: i64) -> CorrectionRecord {
    let target_hash = stable_hash_hex(
        &input
            .target_ids
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
    );
    let id = input.id.unwrap_or_else(|| {
        stable_memory_id(
            "corr",
            &[
                input.scope.space_id.as_str(),
                input.target_type.as_str(),
                &target_hash,
                &created_at_ms.to_string(),
            ],
        )
    });
    CorrectionRecord {
        id,
        scope: input.scope,
        status: MemoryStatus::Active,
        visibility: input.visibility,
        policy_tags: input.policy_tags,
        operation: input.operation,
        target_type: input.target_type,
        target_match: crate::CorrectionTargetMatch::ClaimVersions,
        target_ids: input.target_ids,
        target_selector: input.target_selector,
        target_subject_key: None,
        target_predicate_key: None,
        target_subject_entity_id: None,
        target_slot_id: None,
        new_value: input.new_value,
        reason: input.reason,
        actor: input.actor,
        authority: input.authority,
        created_at_ms,
        effective_at_ms: input.effective_at_ms.unwrap_or(created_at_ms),
        applies_valid_from_ms: input.applies_valid_from_ms,
        applies_valid_to_ms: input.applies_valid_to_ms,
        cascade_policy: input.cascade_policy,
        source_span_ids: Vec::new(),
        source_episode_ids: Vec::new(),
        source_sequence_no: None,
        metadata_json: input.metadata_json,
    }
}

pub fn create_manual_claim(input: ManualClaimInput) -> ClaimRecord {
    let source_hash = stable_hash_hex(
        &input
            .source_span_ids
            .iter()
            .chain(input.source_episode_ids.iter())
            .map(String::as_str)
            .collect::<Vec<_>>(),
    );
    let id = input.id.unwrap_or_else(|| {
        stable_memory_id(
            "claim",
            &[
                input.scope.space_id.as_str(),
                input.claim_text.as_str(),
                &input.observed_at_ms.to_string(),
                &source_hash,
            ],
        )
    });
    ClaimRecord {
        id,
        scope: input.scope,
        status: MemoryStatus::Active,
        visibility: input.visibility,
        policy_tags: input.policy_tags,
        claim_text: input.claim_text,
        subject: input.subject,
        predicate: input.predicate,
        object_value: input.object_value,
        subject_entity_id: None,
        slot_id: None,
        slot_facet: None,
        claim_kind: input.claim_kind,
        polarity: input.polarity,
        source_span_ids: input.source_span_ids,
        source_episode_ids: input.source_episode_ids,
        source_sequence_no: None,
        asserted_by: input.asserted_by,
        extractor_version: None,
        confidence: input.confidence,
        observed_at_ms: input.observed_at_ms,
        valid_from_ms: input.valid_from_ms,
        valid_to_ms: input.valid_to_ms,
        correction_ids: Vec::new(),
    }
}

pub fn create_profile(mut input: ProfileInput) -> ProfileRecord {
    let id = input.id.take().unwrap_or_else(|| {
        let evidence_ids = input
            .evidence_claim_ids
            .iter()
            .chain(input.source_span_ids.iter())
            .map(String::as_str)
            .collect::<Vec<_>>();
        stable_memory_id(
            "profile",
            &[
                input.scope.space_id.as_str(),
                input.profile_key.as_str(),
                &input.generated_at_ms.to_string(),
                &stable_hash_hex(&evidence_ids),
            ],
        )
    });
    input.into_record(id)
}

pub fn create_entity(mut input: EntityInput) -> EntityRecord {
    let id = input.id.take().unwrap_or_else(|| {
        let scope_hash = narrower_scope_hash(&input.scope);
        let mut parts = vec![
            input.scope.space_id.as_str(),
            input.entity_type.as_str(),
            input.canonical_name.as_str(),
        ];
        parts.extend(scope_hash.as_deref());
        stable_memory_id("entity", &parts)
    });
    input.into_record(id)
}

pub fn create_edge(mut input: EdgeInput) -> EdgeRecord {
    let id = input.id.take().unwrap_or_else(|| {
        stable_memory_id(
            "edge",
            &[
                input.scope.space_id.as_str(),
                input.src_entity_id.as_str(),
                input.dst_entity_id.as_str(),
                input.relation_type.as_str(),
            ],
        )
    });
    input.into_record(id)
}

/// The record a text is chunked from; every span inherits these fields.
struct ChunkSource<'a> {
    source_type: SourceType,
    source_id: &'a str,
    source_actor: Option<ActorKind>,
    scope: &'a MemoryScope,
    visibility: Visibility,
    policy_tags: &'a [String],
    valid_from_ms: Option<i64>,
    valid_to_ms: Option<i64>,
    created_at_ms: i64,
}

fn chunk_text(source: &ChunkSource<'_>, text: &str, options: &ChunkOptions) -> Vec<SpanRecord> {
    if text.is_empty() {
        return Vec::new();
    }
    let options = options.normalized();
    let char_starts: Vec<usize> = text.char_indices().map(|(idx, _)| idx).collect();
    let total_chars = char_starts.len();
    if total_chars == 0 {
        return Vec::new();
    }
    let mut spans = Vec::new();
    let mut char_start = 0usize;
    while char_start < total_chars {
        let char_end = (char_start + options.max_chars).min(total_chars);
        let bounds = ChunkBounds {
            span_index: spans.len(),
            char_start,
            char_end,
            byte_start: char_starts[char_start],
            byte_end: char_starts.get(char_end).copied().unwrap_or(text.len()),
        };
        spans.push(chunk_span(source, &options, &bounds, text));
        if char_end == total_chars {
            break;
        }
        let next_start = char_end.saturating_sub(options.overlap_chars);
        char_start = if next_start <= char_start {
            char_end
        } else {
            next_start
        };
    }
    spans
}

/// Where one chunk sits in its source text.
struct ChunkBounds {
    span_index: usize,
    char_start: usize,
    char_end: usize,
    byte_start: usize,
    byte_end: usize,
}

fn chunk_span(
    source: &ChunkSource<'_>,
    options: &ChunkOptions,
    bounds: &ChunkBounds,
    text: &str,
) -> SpanRecord {
    let chunk = &text[bounds.byte_start..bounds.byte_end];
    let text_hash = stable_hash_hex(&[chunk]);
    let span_id = stable_memory_id(
        "span",
        &[
            source.source_id,
            &bounds.byte_start.to_string(),
            &bounds.byte_end.to_string(),
            &options.chunker_version,
            &text_hash,
        ],
    );
    SpanRecord {
        id: span_id,
        scope: source.scope.clone(),
        status: MemoryStatus::Active,
        visibility: source.visibility,
        policy_tags: source.policy_tags.to_vec(),
        source_type: source.source_type,
        source_id: source.source_id.to_string(),
        byte_start: bounds.byte_start as i64,
        byte_end: bounds.byte_end as i64,
        char_start: bounds.char_start as i64,
        char_end: bounds.char_end as i64,
        token_start: None,
        token_end: None,
        span_index: bounds.span_index as i64,
        text: chunk.to_string(),
        text_hash,
        chunker_version: options.chunker_version.clone(),
        embedding_model: None,
        embedding_version: None,
        lexical_text: chunk.to_lowercase(),
        created_at_ms: source.created_at_ms,
        valid_from_ms: source.valid_from_ms,
        valid_to_ms: source.valid_to_ms,
        provenance: vec![ProvenanceRef {
            source_type: source.source_type,
            source_id: source.source_id.to_string(),
            span_id: None,
            actor: source.source_actor,
        }],
    }
}

// Space-only scopes keep their existing ids; any narrower scope gets its own id.
fn narrower_scope_hash(scope: &MemoryScope) -> Option<String> {
    let narrower = [
        &scope.tenant_id,
        &scope.user_id,
        &scope.agent_id,
        &scope.project_id,
        &scope.thread_id,
    ];
    narrower.iter().any(|value| value.is_some()).then(|| {
        let encoded = narrower
            .iter()
            .map(|value| value.as_ref().map_or(String::new(), |v| format!("={v}")))
            .collect::<Vec<_>>();
        stable_hash_hex(&encoded.iter().map(String::as_str).collect::<Vec<_>>())
    })
}

fn stable_memory_id(prefix: &str, parts: &[&str]) -> String {
    format!("{prefix}_{}", stable_hash_hex(parts))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope() -> MemoryScope {
        MemoryScope::new("default")
    }

    #[test]
    fn ingest_episode_builds_stable_episode_and_spans() {
        let input = EpisodeInput {
            id: None,
            scope: scope(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            source_kind: SourceKind::UserMessage,
            actor: ActorKind::User,
            sequence_no: 7,
            event_time_ms: Some(100),
            valid_from_ms: None,
            valid_to_ms: None,
            raw_text: "abcdef".to_string(),
            blob_ref: None,
            mime_type: None,
            causal_parent_ids: Vec::new(),
            metadata_json: None,
        };
        let options = ChunkOptions {
            max_chars: 3,
            overlap_chars: 1,
            chunker_version: "test-v1".to_string(),
        };
        let first = ingest_episode(input.clone(), 101, &options);
        let second = ingest_episode(input, 101, &options);
        assert_eq!(first.episode.id, second.episode.id);
        assert_eq!(first.spans.len(), 3);
        assert_eq!(first.spans[0].text, "abc");
        assert_eq!(first.spans[1].text, "cde");
        assert_eq!(first.spans[2].text, "ef");
        assert_eq!(first.spans[0].valid_from_ms, Some(100));
        assert_eq!(first.spans[0].provenance[0].actor, Some(ActorKind::User));
        assert_eq!(
            chunk_episode(&first.episode, &options),
            first.spans,
            "explicit chunk_episode helper should match ingest output"
        );
    }

    #[test]
    fn chunker_handles_unicode_boundaries() {
        let input = EpisodeInput {
            id: Some("ep_unicode".to_string()),
            scope: scope(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            source_kind: SourceKind::UserMessage,
            actor: ActorKind::User,
            sequence_no: 1,
            event_time_ms: None,
            valid_from_ms: None,
            valid_to_ms: None,
            raw_text: "aé日b".to_string(),
            blob_ref: None,
            mime_type: None,
            causal_parent_ids: Vec::new(),
            metadata_json: None,
        };
        let options = ChunkOptions {
            max_chars: 2,
            overlap_chars: 0,
            chunker_version: "test-v1".to_string(),
        };
        let ingested = ingest_episode(input, 1, &options);
        assert_eq!(ingested.spans[0].text, "aé");
        assert_eq!(ingested.spans[1].text, "日b");
    }

    #[test]
    fn add_correction_defaults_effective_time() {
        let correction = add_correction(
            CorrectionInput {
                id: None,
                scope: scope(),
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                operation: CorrectionOperation::Tombstone,
                target_type: "span".to_string(),
                target_ids: vec!["span_1".to_string()],
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
            42,
        );
        assert_eq!(correction.effective_at_ms, 42);
        assert_eq!(correction.target_ids, vec!["span_1"]);
    }

    #[test]
    fn request_types_are_json_round_trippable_for_bindings() {
        let episode = EpisodeInput {
            id: Some("ep_json".to_string()),
            scope: scope(),
            visibility: Visibility::Private,
            policy_tags: vec!["local".to_string()],
            source_kind: SourceKind::UserMessage,
            actor: ActorKind::User,
            sequence_no: 1,
            event_time_ms: Some(10),
            valid_from_ms: Some(10),
            valid_to_ms: None,
            raw_text: "remember this".to_string(),
            blob_ref: None,
            mime_type: Some("text/plain".to_string()),
            causal_parent_ids: Vec::new(),
            metadata_json: None,
        };
        let decoded: EpisodeInput =
            serde_json::from_str(&serde_json::to_string(&episode).unwrap()).unwrap();
        assert_eq!(decoded.raw_text, episode.raw_text);

        let claim = ManualClaimInput {
            id: Some("claim_json".to_string()),
            scope: scope(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            claim_text: "User likes compact output.".to_string(),
            subject: Some("user".to_string()),
            predicate: Some("likes".to_string()),
            object_value: Some("compact output".to_string()),
            claim_kind: ClaimKind::Preference,
            polarity: ClaimPolarity::Affirmative,
            source_span_ids: Vec::new(),
            source_episode_ids: Vec::new(),
            asserted_by: "user".to_string(),
            confidence: Some(1.0),
            observed_at_ms: 10,
            valid_from_ms: Some(10),
            valid_to_ms: None,
        };
        let decoded: ManualClaimInput =
            serde_json::from_str(&serde_json::to_string(&claim).unwrap()).unwrap();
        assert_eq!(decoded.claim_text, claim.claim_text);
    }
}

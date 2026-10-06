pub use crate::types::EntityInput;
use crate::types::{
    ActorKind, ArtifactRecord, ClaimKind, ClaimPolarity, ClaimRecord, CorrectionAuthority,
    CorrectionOperation, CorrectionRecord, EdgeRecord, EntityRecord, EpisodeRecord, MemoryId,
    MemoryScope, MemoryStatus, ProfileRecord, ProvenanceRef, SourceKind, SourceType, SpanRecord,
    TemporalFields, Visibility,
};
pub(crate) use crate::types::{EdgeInput, ProfileInput};
use serde::{Deserialize, Serialize};

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

/// Settings that cut text into spans.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkOptions {
    /// Maximum characters in one span.
    pub max_chars: usize,
    /// Characters that two adjacent spans share.
    pub overlap_chars: usize,
    /// Version label of the chunker that cut the spans.
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

/// Input for one episode to ingest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeInput {
    /// Id for the episode. `None` makes an id from the scope, sequence, time, and content.
    pub id: Option<MemoryId>,
    /// Memory scope that owns the record.
    pub scope: MemoryScope,
    /// Visibility label of the record.
    pub visibility: Visibility,
    /// Caller-defined policy labels that stay with the record.
    pub policy_tags: Vec<String>,
    /// Kind of event the episode records.
    pub source_kind: SourceKind,
    /// Who produced the content.
    pub actor: ActorKind,
    /// Position of the episode in its conversation.
    pub sequence_no: i64,
    /// Time the event occurred, in Unix ms. `None` means unknown.
    pub event_time_ms: Option<i64>,
    /// Start of the valid-time interval in Unix ms. `None` uses `event_time_ms`.
    pub valid_from_ms: Option<i64>,
    /// End of the valid-time interval in Unix ms, exclusive. `None` means no upper bound.
    pub valid_to_ms: Option<i64>,
    /// Full text of the episode.
    pub raw_text: String,
    /// Reference to the raw content in external storage. `None` means the text holds all content.
    pub blob_ref: Option<String>,
    /// MIME type of the raw content.
    pub mime_type: Option<String>,
    /// Ids of the episodes that caused this episode.
    pub causal_parent_ids: Vec<MemoryId>,
    /// Caller metadata as a JSON string.
    pub metadata_json: Option<String>,
}

/// An episode and the spans cut from its text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestedEpisode {
    /// Stored episode.
    pub episode: EpisodeRecord,
    /// Spans cut from the episode text, in order.
    pub spans: Vec<SpanRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct IngestedArtifact {
    pub artifact: ArtifactRecord,
    pub spans: Vec<SpanRecord>,
}

/// Input for one correction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionInput {
    /// Id for the correction. `None` makes the store generate one.
    pub id: Option<MemoryId>,
    /// Memory scope that owns the record.
    pub scope: MemoryScope,
    /// Visibility label of the record.
    pub visibility: Visibility,
    /// Caller-defined policy labels that stay with the record.
    pub policy_tags: Vec<String>,
    /// Change the correction makes.
    pub operation: CorrectionOperation,
    /// Kind of record the correction targets, such as `claim`.
    pub target_type: String,
    /// Ids of the records the correction targets.
    pub target_ids: Vec<MemoryId>,
    /// Text that selects target records. `None` means the target ids alone select them.
    pub target_selector: Option<String>,
    /// Replacement value. `None` means the operation sets no value.
    pub new_value: Option<String>,
    /// Reason for the correction, in free text.
    pub reason: Option<String>,
    /// Who produced the content.
    pub actor: ActorKind,
    /// Who made the correction.
    pub authority: CorrectionAuthority,
    /// Time the correction takes effect, in Unix ms. `None` means the creation time.
    pub effective_at_ms: Option<i64>,
    /// Start of the valid time the correction changes, in Unix ms. `None` means no lower bound.
    pub applies_valid_from_ms: Option<i64>,
    /// End of the valid time the correction changes, in Unix ms. `None` means no upper bound.
    pub applies_valid_to_ms: Option<i64>,
    /// How the correction spreads to dependent records. `None` means the default policy.
    pub cascade_policy: Option<String>,
    /// Caller metadata as a JSON string.
    pub metadata_json: Option<String>,
}

/// Input for a claim that a caller states directly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManualClaimInput {
    /// Id for the claim. `None` makes the store generate one.
    pub id: Option<MemoryId>,
    /// Memory scope that owns the record.
    pub scope: MemoryScope,
    /// Visibility label of the record.
    pub visibility: Visibility,
    /// Caller-defined policy labels that stay with the record.
    pub policy_tags: Vec<String>,
    /// Statement of the claim in natural language.
    pub claim_text: String,
    /// Subject text as the evidence states it.
    pub subject: Option<String>,
    /// Predicate text as the evidence states it.
    pub predicate: Option<String>,
    /// Value the record states for its slot.
    pub object_value: Option<String>,
    /// Category of the claim.
    pub claim_kind: ClaimKind,
    /// Whether the claim asserts, denies, or is not certain.
    pub polarity: ClaimPolarity,
    /// Ids of the evidence spans that support the record.
    pub source_span_ids: Vec<MemoryId>,
    /// Ids of the episodes that support the record.
    pub source_episode_ids: Vec<MemoryId>,
    /// Name of the extractor or person that made the claim.
    pub asserted_by: String,
    /// Confidence score. `None` means the source gave no score.
    pub confidence: Option<f32>,
    /// Time the evidence was observed, in Unix ms.
    pub observed_at_ms: i64,
    /// Start of the valid-time interval in Unix ms, inclusive. `None` means no lower bound.
    pub valid_from_ms: Option<i64>,
    /// End of the valid-time interval in Unix ms, exclusive. `None` means no upper bound.
    pub valid_to_ms: Option<i64>,
}

/// Returns a 64-bit FNV-1a hash of the parts as 16 hex digits.
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

/// Makes an episode record and cuts its text into spans. It writes nothing.
pub fn ingest_episode(
    input: EpisodeInput,
    ingested_at_ms: i64,
    chunk_options: &ChunkOptions,
) -> IngestedEpisode {
    let content_hash = stable_hash_hex(&[&input.raw_text]);
    let id = input.id.unwrap_or_else(|| {
        let sequence_no = input.sequence_no.to_string();
        let event_time_ms = input.event_time_ms.unwrap_or(ingested_at_ms).to_string();
        generated_id(
            "ep",
            &input.scope,
            &[&sequence_no, &event_time_ms, &content_hash],
        )
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

pub(crate) fn chunk_artifact_text(
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

#[cfg(test)]
pub(crate) fn chunk_episode(
    episode: &EpisodeRecord,
    chunk_options: &ChunkOptions,
) -> Vec<SpanRecord> {
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

/// Makes a correction record from the input. It writes nothing.
pub fn add_correction(input: CorrectionInput, created_at_ms: i64) -> CorrectionRecord {
    let target_hash = stable_hash_hex(
        &input
            .target_ids
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
    );
    let id = input.id.unwrap_or_else(|| {
        let created_at = created_at_ms.to_string();
        let mut parts = vec![input.target_type.as_str(), &target_hash, &created_at];
        // Appended only when set so corrections without a selector keep their existing IDs.
        parts.extend(input.target_selector.as_deref());
        generated_id("corr", &input.scope, &parts)
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

/// Makes a claim record from the input. It writes nothing.
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
        let observed_at_ms = input.observed_at_ms.to_string();
        generated_id(
            "claim",
            &input.scope,
            &[&input.claim_text, &observed_at_ms, &source_hash],
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

pub(crate) fn create_profile(mut input: ProfileInput) -> ProfileRecord {
    let id = input.id.take().unwrap_or_else(|| {
        let evidence_ids = input
            .evidence_claim_ids
            .iter()
            .chain(input.source_span_ids.iter())
            .map(String::as_str)
            .collect::<Vec<_>>();
        let generated_at = input.generated_at_ms.to_string();
        let evidence_hash = stable_hash_hex(&evidence_ids);
        generated_id(
            "profile",
            &input.scope,
            &[&input.profile_key, &generated_at, &evidence_hash],
        )
    });
    input.into_record(id)
}

pub(crate) fn create_entity(mut input: EntityInput) -> EntityRecord {
    let id = input.id.take().unwrap_or_else(|| {
        generated_id(
            "entity",
            &input.scope,
            &[&input.entity_type, &input.canonical_name],
        )
    });
    input.into_record(id)
}

pub(crate) fn create_edge(mut input: EdgeInput) -> EdgeRecord {
    let id = input.id.take().unwrap_or_else(|| {
        generated_id(
            "edge",
            &input.scope,
            &[
                &input.src_entity_id,
                &input.dst_entity_id,
                &input.relation_type,
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
pub(crate) fn generated_id(kind: &str, scope: &MemoryScope, parts: &[&str]) -> MemoryId {
    let scope_hash = narrower_scope_hash(scope);
    let mut all = Vec::with_capacity(parts.len() + 2);
    all.push(scope.space_id.as_str());
    all.extend_from_slice(parts);
    all.extend(scope_hash.as_deref());
    stable_memory_id(kind, &all)
}

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

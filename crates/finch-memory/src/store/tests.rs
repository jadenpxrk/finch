use super::*;
use crate::context::{
    build_answer_ready_state_context, build_query_state_context, ContextInput, ContextOptions,
};
use crate::ingest::{create_manual_claim, CorrectionInput, EntityInput, ManualClaimInput};
use crate::types::{
    ActorKind, ArtifactKind, ClaimKind, ClaimPolarity, CorrectionAuthority, CorrectionOperation,
    MemoryScope, MemoryStatus, SourceKind, SourceType, Visibility,
};
use crate::StateMutationBatch;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, Ordering};

// The correction scan cap that used to drop corrections; saturation tests insert this many.
const DEFAULT_CORRECTION_SCAN_LIMIT: usize = 1024;

fn temp_dir(name: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "finch_memory_test_{}_{}_{}",
        name,
        std::process::id(),
        n
    ));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn scope() -> MemoryScope {
    MemoryScope::new("default")
}

fn answer_request<'a>(
    projection_seed: &'a str,
    query_text: &'a str,
    evidence_hits: &'a [SpanSearchHit],
    claim_limit: usize,
    entity_limit: usize,
    at_ms: Option<i64>,
) -> AnswerReadyStateRequest<'a> {
    AnswerReadyStateRequest {
        projection_seed,
        query_text,
        evidence_hits,
        claim_limit,
        entity_limit,
        temporal: BiTemporalQuery {
            valid_at_ms: at_ms,
            transaction_at_ms: None,
        },
        ..Default::default()
    }
}

fn state_scan(limit: usize, at_ms: Option<i64>) -> StateRecordScan {
    StateRecordScan {
        limit,
        temporal: BiTemporalQuery {
            valid_at_ms: at_ms,
            transaction_at_ms: None,
        },
    }
}

#[test]
fn sql_escape_round_trips_backslashes_and_quotes_through_the_filter_parser() {
    for value in [
        r"C:\notes\",
        r"a\b",
        r"a\\b",
        r"it's",
        r"a\'b",
        r"a\\'b",
        r"\",
        r"\\",
    ] {
        let sql = format!("project_id = '{}'", sql_escape(value));
        let expr = finch_db::sqlengine::parse_filter(&sql).unwrap();
        assert!(
            matches!(&expr, finch_db::sqlengine::FilterExpr::Compare { value: Value::String(s), .. } if s == value),
            "{sql} parsed as {expr:?}"
        );
    }
}

#[test]
fn shared_state_requires_a_non_assistant_source() {
    assert_eq!(
        crate::shared_state_evidence_role([ActorKind::Assistant]),
        crate::SharedStateEvidenceRole::AdvisoryOnly
    );
    assert_eq!(
        crate::shared_state_evidence_role([ActorKind::Assistant, ActorKind::User]),
        crate::SharedStateEvidenceRole::StateBearing
    );
    assert_eq!(
        crate::shared_state_evidence_role([ActorKind::Tool]),
        crate::SharedStateEvidenceRole::StateBearing
    );
    assert_eq!(
        crate::shared_state_evidence_role(std::iter::empty()),
        crate::SharedStateEvidenceRole::AdvisoryOnly
    );
}

fn span_record(id: &str, status: MemoryStatus, valid_from_ms: Option<i64>) -> SpanRecord {
    SpanRecord {
        id: id.to_string(),
        scope: scope(),
        status,
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        source_type: SourceType::Episode,
        source_id: "ep_1".to_string(),
        byte_start: 0,
        byte_end: id.len() as i64,
        char_start: 0,
        char_end: id.len() as i64,
        token_start: None,
        token_end: None,
        span_index: 0,
        text: id.to_string(),
        text_hash: id.to_string(),
        chunker_version: "test".to_string(),
        embedding_model: Some("test".to_string()),
        embedding_version: Some("1".to_string()),
        lexical_text: id.to_string(),
        created_at_ms: 1,
        valid_from_ms,
        valid_to_ms: None,
        provenance: Vec::new(),
    }
}

fn make_derived_rule_inputs() -> RuleInput {
    RuleInput {
        id: Some("rule_owner_reviewer".to_string()),
        scope: scope(),
        status: MemoryStatus::Active,
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        trigger_subject_key: "project".to_string(),
        trigger_predicate_key: "owner".to_string(),
        target_subject_key: "project".to_string(),
        target_predicate_key: "reviewer".to_string(),
        trigger_slot_id: None,
        target_slot_id: None,
        trigger_subject: Some("project".to_string()),
        trigger_predicate: Some("owner".to_string()),
        target_subject: Some("project".to_string()),
        target_predicate: Some("reviewer".to_string()),
        target_match: RuleTargetMatch::ExactSlot,
        activation: RuleActivation::ContinuousProjection,
        action: RuleAction::DeriveValue,
        value_template: Some("{value}".to_string()),
        value: None,
        source_span_ids: vec!["rule_span_owner".to_string()],
        source_episode_ids: vec!["ep_owner".to_string()],
        valid_from_ms: Some(0),
        valid_to_ms: None,
        confidence: Some(1.0),
    }
}

fn make_claim(
    scope: &MemoryScope,
    id: &str,
    subject: &str,
    predicate: &str,
    object_value: Option<&str>,
    observed_at_ms: i64,
    (claim_kind, polarity): (ClaimKind, ClaimPolarity),
) -> ClaimRecord {
    create_manual_claim(ManualClaimInput {
        id: Some(id.to_string()),
        scope: scope.clone(),
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        claim_text: format!("{subject} {predicate} is {}", object_value.unwrap_or("")),
        subject: Some(subject.to_string()),
        predicate: Some(predicate.to_string()),
        object_value: object_value.map(std::string::ToString::to_string),
        claim_kind,
        polarity,
        source_span_ids: vec![format!("span_{id}")],
        source_episode_ids: vec![format!("ep_{id}")],
        asserted_by: "user".to_string(),
        confidence: Some(1.0),
        observed_at_ms,
        valid_from_ms: Some(observed_at_ms),
        valid_to_ms: None,
    })
}

/// A store that first writes the episodes and spans each record cites, because writes reject
/// evidence ids that do not resolve in the record's scope.
struct EvidencedStore(MemoryStore);

impl std::ops::Deref for EvidencedStore {
    type Target = MemoryStore;

    fn deref(&self) -> &MemoryStore {
        &self.0
    }
}

impl EvidencedStore {
    fn create(path: &Path, embedding_dim: usize, options: CollectionOptions) -> ZResult<Self> {
        MemoryStore::create(path, embedding_dim, options).map(Self)
    }

    /// Writes the cited episodes and spans that do not exist yet; each span quotes the first
    /// cited episode. Sequence 0 on every seeded episode keeps claim ordering ties unchanged.
    fn seed_evidence(&self, scope: &MemoryScope, span_ids: &[MemoryId], episode_ids: &[MemoryId]) {
        let Some(source_episode_id) = episode_ids.first() else {
            return;
        };
        let existing_episodes = self.episodes.fetch(episode_ids.to_vec()).unwrap();
        for id in episode_ids
            .iter()
            .filter(|id| !existing_episodes.contains_key(*id))
        {
            self.append_episode(&EpisodeRecord {
                id: id.clone(),
                scope: scope.clone(),
                status: MemoryStatus::Active,
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                source_kind: SourceKind::UserMessage,
                actor: ActorKind::User,
                sequence_no: 0,
                temporal: crate::types::TemporalFields {
                    created_at_ms: 0,
                    ingested_at_ms: 0,
                    event_time_ms: None,
                    valid_from_ms: None,
                    valid_to_ms: None,
                },
                raw_text: id.clone(),
                blob_ref: None,
                mime_type: None,
                content_hash: id.clone(),
                causal_parent_ids: Vec::new(),
                metadata_json: None,
            })
            .unwrap();
        }
        let existing_spans = self.spans.fetch(span_ids.to_vec()).unwrap();
        for id in span_ids
            .iter()
            .filter(|id| !existing_spans.contains_key(*id))
        {
            let mut span = span_record(id, MemoryStatus::Active, None);
            span.scope = scope.clone();
            span.source_id = source_episode_id.clone();
            // Stored without lexical terms or an embedding, so searches never return it.
            insert_one(&self.spans, span_doc(&span, None).unwrap()).unwrap();
        }
    }

    /// Writes the cited claims that do not exist yet as superseded rows, so they prove the
    /// reference without becoming current state.
    fn seed_source_claims(&self, scope: &MemoryScope, claim_ids: &[MemoryId]) {
        let existing = self.claims.fetch(claim_ids.to_vec()).unwrap();
        for id in claim_ids.iter().filter(|id| !existing.contains_key(*id)) {
            let mut claim = make_claim(
                scope,
                id,
                "",
                "",
                None,
                0,
                (ClaimKind::Fact, ClaimPolarity::Affirmative),
            );
            claim.status = MemoryStatus::Superseded;
            claim.source_span_ids.clear();
            claim.source_episode_ids.clear();
            insert_one(&self.claims, claim_doc(&claim, None).unwrap()).unwrap();
        }
    }

    fn add_entity(&self, input: EntityInput, embedding: Option<&[f32]>) -> ZResult<EntityRecord> {
        self.seed_source_claims(&input.scope, &input.source_claim_ids);
        self.0.add_entity(input, embedding)
    }

    fn append_claim(&self, record: &ClaimRecord, embedding: Option<&[f32]>) -> ZResult<()> {
        self.seed_evidence(
            &record.scope,
            &record.source_span_ids,
            &record.source_episode_ids,
        );
        self.0.append_claim(record, embedding)
    }

    fn add_manual_claim(
        &self,
        input: ManualClaimInput,
        embedding: Option<&[f32]>,
    ) -> ZResult<ClaimRecord> {
        self.seed_evidence(
            &input.scope,
            &input.source_span_ids,
            &input.source_episode_ids,
        );
        self.0.add_manual_claim(input, embedding)
    }

    fn add_rule(&self, input: RuleInput) -> ZResult<RuleRecord> {
        self.seed_evidence(
            &input.scope,
            &input.source_span_ids,
            &input.source_episode_ids,
        );
        self.0.add_rule(input)
    }

    fn append_rule(&self, record: &RuleRecord) -> ZResult<()> {
        self.seed_evidence(
            &record.scope,
            &record.source_span_ids,
            &record.source_episode_ids,
        );
        self.0.append_rule(record)
    }

    fn add_correction(&self, record: &CorrectionRecord) -> ZResult<()> {
        self.seed_evidence(
            &record.scope,
            &record.source_span_ids,
            &record.source_episode_ids,
        );
        self.0.add_correction(record)
    }

    fn apply_state_mutation_batch(
        &self,
        batch: StateMutationBatch,
    ) -> ZResult<crate::StateMutationResult> {
        self.apply_state_mutation_batch_with_claim_embeddings(batch, BTreeMap::new())
    }

    fn apply_state_mutation_batch_with_claim_embeddings(
        &self,
        batch: StateMutationBatch,
        claim_embeddings: BTreeMap<MemoryId, Vec<f32>>,
    ) -> ZResult<crate::StateMutationResult> {
        for (span_ids, episode_ids) in batch
            .claims
            .iter()
            .map(|record| (&record.source_span_ids, &record.source_episode_ids))
            .chain(
                batch
                    .rules
                    .iter()
                    .map(|record| (&record.source_span_ids, &record.source_episode_ids)),
            )
            .chain(
                batch
                    .corrections
                    .iter()
                    .map(|record| (&record.source_span_ids, &record.source_episode_ids)),
            )
        {
            self.seed_evidence(&batch.scope, span_ids, episode_ids);
        }
        let incoming_claim_ids = batch
            .claims
            .iter()
            .map(|claim| claim.id.clone())
            .collect::<BTreeSet<_>>();
        let entity_claim_ids = batch
            .entities
            .iter()
            .flat_map(|entity| entity.source_claim_ids.iter())
            .filter(|id| !incoming_claim_ids.contains(*id))
            .cloned()
            .collect::<Vec<_>>();
        self.seed_source_claims(&batch.scope, &entity_claim_ids);
        self.0
            .apply_state_mutation_batch_with_claim_embeddings(batch, claim_embeddings)
    }
}

mod adversarial_round_1;
mod claims;
mod fetch_by_id;
mod generated_ids;
mod graph;
mod lifecycle;
mod limit_after_filter;
mod retrieval;
mod review_round_10;
mod review_round_2;
mod review_round_3;
mod review_round_4;
mod review_round_5;
mod review_round_6;
mod review_round_7;
mod review_round_8;
mod review_round_9;
mod rule_projection;
mod rule_time;
mod rules;
mod scope_keys;
mod state_protocol;
mod string_indexes;
mod write_integrity;

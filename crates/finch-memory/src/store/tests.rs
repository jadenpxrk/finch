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

mod adversarial_round_1;
mod claims;
mod graph;
mod lifecycle;
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
mod rules;
mod state_protocol;

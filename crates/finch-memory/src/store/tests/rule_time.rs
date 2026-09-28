use super::*;

const RULE_ID: &str = "rule_owner_reviewer_window";

/// A store whose owner-to-reviewer rule is in force from 10 to 20, with an owner claim valid
/// from `trigger_from_ms`.
fn windowed_rule_store(name: &str, trigger_from_ms: i64) -> EvidencedStore {
    let store = EvidencedStore::create(&temp_dir(name), 3, CollectionOptions::default()).unwrap();
    let mut rule = make_derived_rule_inputs();
    rule.id = Some(RULE_ID.to_string());
    rule.valid_from_ms = Some(10);
    rule.valid_to_ms = Some(20);
    store.add_rule(rule).unwrap();
    let owner = make_claim(
        &scope(),
        "claim_owner_window",
        "project",
        "owner",
        Some("Dana"),
        trigger_from_ms,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    store.append_claim(&owner, None).unwrap();
    store
}

/// Whether an answer-ready read at `at_ms` carries the rule's derived reviewer.
fn read_shows_derived_reviewer(store: &MemoryStore, at_ms: i64) -> bool {
    let projection = store
        .project_answer_ready_state(
            &scope(),
            &answer_request(
                "rule-window",
                "project owner reviewer",
                &[],
                20,
                0,
                Some(at_ms),
            ),
        )
        .unwrap();
    let derived_claim = projection.claims.iter().any(|claim| {
        claim.predicate.as_deref() == Some("reviewer")
            && claim.object_value.as_deref() == Some("Dana")
    });
    let rule_applied = projection
        .rule_outcomes
        .iter()
        .any(|outcome| outcome.rule_id == RULE_ID && outcome.status.is_applied());
    assert_eq!(
        derived_claim, rule_applied,
        "at {at_ms}: claims={:?} outcomes={:?}",
        projection.claims, projection.rule_outcomes
    );
    derived_claim
}

#[test]
fn read_time_rules_are_evaluated_at_the_read_time() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let store = windowed_rule_store("rule_time_read_time", 5);

    assert!(!read_shows_derived_reviewer(&store, 30));
    assert!(read_shows_derived_reviewer(&store, 15));
    let _ = std::fs::remove_dir_all(&store.path);
}

#[test]
fn read_after_a_rule_ends_does_not_show_its_derivation() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let store = windowed_rule_store("rule_time_rule_ended", 12);

    assert!(!read_shows_derived_reviewer(&store, 30));
    assert!(read_shows_derived_reviewer(&store, 15));
    let _ = std::fs::remove_dir_all(&store.path);
}

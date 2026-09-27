use super::*;

#[test]
fn generated_profile_ids_collide_between_tenants() {
    // Profiles generated independently in different tenant scopes must coexist.
    let guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let path = temp_dir("round_10_profile_scope");
    let store = MemoryStore::create(&path, 3, CollectionOptions::default()).unwrap();
    let scopes = ["tenant_a", "tenant_b"].map(|tenant| {
        let mut scope = scope();
        scope.tenant_id = Some(tenant.to_string());
        scope
    });
    let results = scopes.each_ref().map(|scope| {
        store.add_profile(ProfileInput {
            id: None,
            scope: scope.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            subject_id: None,
            profile_key: "style".to_string(),
            profile_text: format!(
                "{} prefers concise answers",
                scope.tenant_id.as_ref().unwrap()
            ),
            value_json: None,
            evidence_claim_ids: Vec::new(),
            source_span_ids: Vec::new(),
            generated_at_ms: 10,
            generator_version: "manual".to_string(),
            valid_from_ms: Some(10),
            valid_to_ms: None,
            correction_watermark: 10,
        })
    });
    drop(store);
    let reopened = MemoryStore::open(&path, CollectionOptions::default()).unwrap();
    let profiles = scopes
        .each_ref()
        .map(|scope| reopened.scan_profiles(scope, 10, Some(20)).unwrap());
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
    drop(guard);
    assert!(results[0].is_ok());
    assert!(
        results[1].is_ok(),
        "independent tenant profile failed: {:?}; reopened profile counts: {:?}",
        results[1],
        profiles.each_ref().map(|profiles| profiles.len())
    );
    for (scope, profiles) in scopes.iter().zip(&profiles) {
        assert_eq!(profiles.len(), 1);
        assert_eq!(&profiles[0].scope, scope);
    }
    assert_ne!(profiles[0][0].id, profiles[1][0].id);
}

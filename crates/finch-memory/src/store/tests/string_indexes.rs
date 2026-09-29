use super::*;
use crate::schema::active_collection_schemas;

const DIM: usize = 3;

fn space(id: &str) -> MemoryScope {
    MemoryScope::new(id)
}

fn owner(scope: &MemoryScope, id: &str, value: &str) -> ClaimRecord {
    let mut claim = make_claim(
        scope,
        id,
        "project",
        "owner",
        Some(value),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    claim.source_span_ids.clear();
    claim.source_episode_ids.clear();
    claim
}

fn read_only() -> CollectionOptions {
    CollectionOptions {
        read_only: true,
        ..CollectionOptions::default()
    }
}

/// Writes `claims` through a current store, then copies every row into a store at a new path
/// whose collections have the schema from before the string indexes.
fn store_without_string_indexes(name: &str, claims: &[ClaimRecord]) -> PathBuf {
    let current = temp_dir(&format!("{name}_current"));
    let store = MemoryStore::create(&current, DIM, CollectionOptions::default()).unwrap();
    for claim in claims {
        store.append_claim(claim, None).unwrap();
    }
    drop(store);
    let old = temp_dir(name);
    std::fs::create_dir_all(&old).unwrap();
    for mut schema in active_collection_schemas(DIM) {
        for field in &mut schema.fields {
            if matches!(field.index_params, Some(IndexParams::Invert(_))) {
                field.index_params = None;
            }
        }
        let name = schema.name.clone();
        let source = Collection::open(&current.join(&name), CollectionOptions::default()).unwrap();
        let docs = source
            .scan_filter_only(VectorQuery::new("", Vec::new(), usize::MAX))
            .unwrap()
            .iter()
            .map(|doc| (**doc).clone())
            .collect::<Vec<_>>();
        let target =
            Collection::create_and_open(&old.join(&name), schema, CollectionOptions::default())
                .unwrap();
        insert_many(&target, docs).unwrap();
        target.flush().unwrap();
    }
    std::fs::remove_dir_all(current).unwrap();
    old
}

/// Each collection's `(name, field)` pairs among the indexed string fields that have an index.
fn indexed_string_fields(path: &Path) -> Vec<(String, String)> {
    active_collection_schemas(DIM)
        .into_iter()
        .flat_map(|schema| {
            let stored = Collection::open(&path.join(&schema.name), CollectionOptions::default())
                .unwrap()
                .schema_info();
            INDEXED_STRING_FIELDS
                .into_iter()
                .filter(|field| {
                    stored
                        .get_field(field)
                        .is_some_and(|f| matches!(f.index_params, Some(IndexParams::Invert(_))))
                })
                .map(|field| (schema.name.clone(), field.to_string()))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn state_claim_ids(store: &MemoryStore, scope: &MemoryScope) -> Vec<Vec<MemoryId>> {
    store
        .scan_state_records(scope, state_scan(usize::MAX, None))
        .unwrap()
        .into_iter()
        .map(|record| record.claim_ids)
        .collect()
}

#[test]
fn new_store_indexes_the_string_fields_wherever_they_exist() {
    let path = temp_dir("string_indexes_new");
    drop(MemoryStore::create(&path, DIM, CollectionOptions::default()).unwrap());

    let indexed = indexed_string_fields(&path);

    assert_eq!(indexed.len(), 15 + 5);
    for collection in [CLAIMS_COLLECTION, STATE_RECORDS_COLLECTION] {
        for field in INDEXED_STRING_FIELDS {
            assert!(indexed.contains(&(collection.to_string(), field.to_string())));
        }
    }
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn read_write_open_of_an_older_store_adds_the_string_indexes_once() {
    let (a, b) = (space("space_a"), space("space_b"));
    let path = store_without_string_indexes(
        "string_indexes_upgrade",
        &[owner(&a, "owner_a", "Ada"), owner(&b, "owner_b", "Bo")],
    );
    assert!(indexed_string_fields(&path).is_empty());

    let store = MemoryStore::open(&path, CollectionOptions::default()).unwrap();
    assert_eq!(
        state_claim_ids(&store, &a),
        vec![vec!["owner_a".to_string()]]
    );
    assert_eq!(
        state_claim_ids(&store, &b),
        vec![vec!["owner_b".to_string()]]
    );
    drop(store);
    assert_eq!(indexed_string_fields(&path).len(), 15 + 5);

    let store = MemoryStore::open(&path, CollectionOptions::default()).unwrap();
    assert_eq!(
        state_claim_ids(&store, &a),
        vec![vec!["owner_a".to_string()]]
    );
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn nul_in_a_scope_value_blocks_the_upgrade_and_names_the_column() {
    let a = space("space_a");
    let path = store_without_string_indexes("string_indexes_nul", &[owner(&a, "owner_a", "Ada")]);
    let claims =
        Collection::open(&path.join(CLAIMS_COLLECTION), CollectionOptions::default()).unwrap();
    let nul_claim = owner(&space("space\0b"), "owner_nul", "Nul");
    insert_one(&claims, claim_doc(&nul_claim, None).unwrap()).unwrap();
    claims.flush().unwrap();
    drop(claims);

    let err = MemoryStore::open(&path, CollectionOptions::default())
        .err()
        .expect("a NUL in an indexed column must block the upgrade");

    assert!(err.message().contains("`memory_claims`"), "{err:?}");
    assert!(err.message().contains("`space_id`"), "{err:?}");
    assert!(indexed_string_fields(&path).is_empty());
    let store = MemoryStore::open(&path, read_only()).unwrap();
    assert_eq!(
        state_claim_ids(&store, &a),
        vec![vec!["owner_a".to_string()]]
    );
    drop(store);
    std::fs::remove_dir_all(path).unwrap();
}

#[test]
fn read_only_open_of_an_older_store_answers_without_the_indexes() {
    let a = space("space_a");
    let path =
        store_without_string_indexes("string_indexes_read_only", &[owner(&a, "owner_a", "Ada")]);

    let store = MemoryStore::open(&path, read_only()).unwrap();

    assert_eq!(
        state_claim_ids(&store, &a),
        vec![vec!["owner_a".to_string()]]
    );
    drop(store);
    assert!(indexed_string_fields(&path).is_empty());
    std::fs::remove_dir_all(path).unwrap();
}

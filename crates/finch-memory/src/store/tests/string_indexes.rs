use super::*;
use crate::schema::{active_collection_schemas, INDEXED_STRING_FIELDS};

const DIM: usize = 3;

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

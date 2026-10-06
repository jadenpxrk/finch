use std::collections::HashMap;
use std::sync::Arc;

use finch_types::{Doc, ZResult};

use crate::doc_fetch;

use super::Collection;

impl Collection {
    /// Returns the current document of each primary key, by key; a missing key has no entry.
    pub fn fetch(&self, pks: Vec<String>) -> ZResult<HashMap<String, Arc<Doc>>> {
        // Writers publish a key and its doc under the delete store's write lock.
        let _published = self.delete_store.read();
        let pk_refs: Vec<&str> = pks.iter().map(|s| s.as_str()).collect();
        let doc_ids = self.id_map.multi_get(&pk_refs)?;

        let id_pk_pairs: Vec<(u64, String)> = pks
            .iter()
            .zip(doc_ids.iter())
            .filter_map(|(pk, maybe_id)| maybe_id.map(|id| (id, pk.clone())))
            .collect();

        let ids: Vec<u64> = id_pk_pairs.iter().map(|(id, _)| *id).collect();
        let mut fetched = self.fetch_by_ids(&ids)?;

        let mut result = HashMap::new();
        for (id, pk) in id_pk_pairs {
            if let Some(doc) = fetched.remove(&id) {
                result.insert(pk, Arc::new(doc));
            }
        }

        Ok(result)
    }

    pub(super) fn fetch_by_ids(&self, doc_ids: &[u64]) -> ZResult<HashMap<u64, Doc>> {
        let mut docs =
            doc_fetch::fetch_docs_by_ids(&self.writing_segment, &self.persisted_segments, doc_ids)?;
        // Persisted segments keep a dropped column's data until compaction rewrites them.
        let version = self.cur_version();
        for doc in docs.values_mut() {
            doc.fields.retain(|name, _| version.schema.has_field(name));
        }
        Ok(docs)
    }

    pub(super) fn fetch_pks_by_ids(&self, doc_ids: &[u64]) -> ZResult<HashMap<u64, String>> {
        doc_fetch::fetch_pks_by_ids(&self.writing_segment, &self.persisted_segments, doc_ids)
    }

    pub(super) fn fetch_i64_pks_by_ids_ordered(&self, doc_ids: &[u64]) -> ZResult<Vec<i64>> {
        doc_fetch::fetch_i64_pks_by_ids_ordered(
            &self.writing_segment,
            &self.persisted_segments,
            doc_ids,
        )
    }
}

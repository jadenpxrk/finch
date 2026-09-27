use std::collections::HashMap;
use std::fs;

use finch_types::{Doc, Operator, Status, ZResult};
use parking_lot::RwLockWriteGuard;
use roaring::RoaringTreemap;

use crate::delete_store::DeleteStore;
use crate::query_filter::prepare_required_filter_expr;
use crate::sqlengine::parser::FilterExpr;
use crate::wal::{WalEntry, WalOp};
use crate::write_normalization::{
    normalize_binary_fields_for_write, normalize_vector_fields_for_write,
};

use super::Collection;

impl Collection {
    pub fn insert(&self, docs: Vec<Doc>) -> ZResult<Vec<Status>> {
        self.check_not_readonly()?;
        if docs.len() > 1024 {
            return Err(Status::invalid_argument("Too many docs"));
        }
        let _guard = self.write_lock.lock();
        let version = self.cur_version();

        let mut results = Vec::with_capacity(docs.len());

        for mut doc in docs {
            normalize_binary_fields_for_write(&version.schema, &mut doc);
            if let Err(s) = doc.validate(&version.schema, false) {
                results.push(s);
                continue;
            }
            if let Err(s) = normalize_vector_fields_for_write(&version.schema, &mut doc) {
                results.push(s);
                continue;
            }
            if let Ok(Some(_)) = self.id_map.get(&doc.pk) {
                results.push(Status::already_exists(format!(
                    "pk '{}' already exists",
                    doc.pk
                )));
                continue;
            }

            let doc_id = self.allocate_doc_id();
            doc.doc_id = doc_id;
            doc.op = Operator::Insert;

            // Write-ahead log before mutating state
            let wal_entry = WalEntry {
                op: WalOp::Insert,
                doc_id,
                prev_doc_id: None,
                pk: doc.pk.clone(),
                doc: Some(doc.clone()),
            };
            if let Err(s) = self.wal_append(&wal_entry) {
                results.push(s);
                continue;
            }

            self.id_map.insert(&doc.pk, doc_id)?;

            let mut writing = self.writing_segment.write();
            writing.insert(doc_id, doc)?;

            if self.should_rotate_writing(&writing, version.schema.max_doc_count_per_segment) {
                drop(writing);
                self.rotate_segment()?;
            }

            results.push(Status::default());
        }

        Ok(results)
    }

    pub fn upsert(&self, docs: Vec<Doc>) -> ZResult<Vec<Status>> {
        self.check_not_readonly()?;
        if docs.len() > 1024 {
            return Err(Status::invalid_argument("Too many docs"));
        }
        let _guard = self.write_lock.lock();
        let version = self.cur_version();
        let max_docs_per_segment = version.schema.max_doc_count_per_segment;

        let mut results = Vec::with_capacity(docs.len());

        for mut doc in docs {
            normalize_binary_fields_for_write(&version.schema, &mut doc);
            if let Err(s) = doc.validate(&version.schema, false) {
                results.push(s);
                continue;
            }
            if let Err(s) = normalize_vector_fields_for_write(&version.schema, &mut doc) {
                results.push(s);
                continue;
            }
            let existing_id = self.id_map.get(&doc.pk)?;
            let (doc_id, prev_doc_id) = if let Some(old_id) = existing_id {
                (self.allocate_doc_id(), Some(old_id))
            } else {
                (self.allocate_doc_id(), None)
            };
            doc.doc_id = doc_id;
            doc.op = Operator::Upsert;

            let wal_entry = WalEntry {
                op: WalOp::Upsert,
                doc_id,
                prev_doc_id,
                pk: doc.pk.clone(),
                doc: Some(doc.clone()),
            };
            if let Err(s) = self.wal_append(&wal_entry) {
                results.push(s);
                continue;
            }

            // Readers hold the delete store across their scan, so tombstone and insert land together.
            let mut delete_store = self.delete_store.write();
            if let Some(old_id) = prev_doc_id {
                delete_store.mark_deleted(old_id);
            }
            self.id_map.insert(&doc.pk, doc_id)?;

            let mut writing = self.writing_segment.write();
            writing.insert(doc_id, doc)?;
            drop(delete_store);
            if self.should_rotate_writing(&writing, max_docs_per_segment) {
                drop(writing);
                self.rotate_segment()?;
            }

            results.push(Status::default());
        }

        Ok(results)
    }

    pub fn update(&self, docs: Vec<Doc>) -> ZResult<Vec<Status>> {
        self.check_not_readonly()?;
        if docs.len() > 1024 {
            return Err(Status::invalid_argument("Too many docs"));
        }
        let _guard = self.write_lock.lock();
        let version = self.cur_version();
        let max_docs_per_segment = version.schema.max_doc_count_per_segment;

        let mut results = Vec::with_capacity(docs.len());

        for mut doc in docs {
            normalize_binary_fields_for_write(&version.schema, &mut doc);
            if let Err(s) = doc.validate(&version.schema, true) {
                results.push(s);
                continue;
            }
            let old_doc_id = match self.id_map.get(&doc.pk)? {
                None => {
                    results.push(Status::not_found(format!("pk '{}' not found", doc.pk)));
                    continue;
                }
                Some(id) => id,
            };
            let delete_bitmap = self.delete_store.read().bitmap();
            if delete_bitmap.contains(old_doc_id) {
                results.push(Status::not_found(format!("pk '{}' not found", doc.pk)));
                continue;
            }

            // Fetch existing doc and merge patch fields.
            let old_doc = self
                .fetch_by_ids(&[old_doc_id])?
                .remove(&old_doc_id)
                .ok_or_else(|| Status::internal("update failed: existing doc not found"))?;

            let mut merged = old_doc;
            for (k, v) in doc.fields.iter() {
                merged.fields.insert(k.clone(), v.clone());
            }

            let new_doc_id = self.allocate_doc_id();
            merged.doc_id = new_doc_id;
            merged.op = Operator::Update;
            if let Err(s) = normalize_vector_fields_for_write(&version.schema, &mut merged) {
                results.push(s);
                continue;
            }

            let wal_entry = WalEntry {
                op: WalOp::Update,
                doc_id: new_doc_id,
                prev_doc_id: Some(old_doc_id),
                pk: doc.pk.clone(),
                doc: Some(merged.clone()),
            };
            if let Err(s) = self.wal_append(&wal_entry) {
                results.push(s);
                continue;
            }

            // Apply tombstone + PK remap before writing the new doc, keeping a
            // consistent lock order with query paths (delete_store -> writing_segment).
            let mut delete_store = self.delete_store.write();
            delete_store.mark_deleted(old_doc_id);
            self.id_map.insert(&doc.pk, new_doc_id)?;
            let mut writing = self.writing_segment.write();
            writing.insert(new_doc_id, merged)?;
            drop(delete_store);
            if self.should_rotate_writing(&writing, max_docs_per_segment) {
                drop(writing);
                self.rotate_segment()?;
            }

            results.push(Status::default());
        }

        Ok(results)
    }

    pub fn delete(&self, pks: Vec<String>) -> ZResult<Vec<Status>> {
        self.check_not_readonly()?;
        let _guard = self.write_lock.lock();

        let pk_refs: Vec<&str> = pks.iter().map(|s| s.as_str()).collect();
        let doc_ids = self.id_map.multi_get(&pk_refs)?;

        let mut results = Vec::with_capacity(pks.len());
        let mut delete_store = self.delete_store.write();

        for (pk, maybe_id) in pks.iter().zip(doc_ids.iter()) {
            let Some(doc_id) = maybe_id else {
                results.push(Status::not_found(format!("pk '{}' not found", pk)));
                continue;
            };
            let wal_entry = WalEntry {
                op: WalOp::Delete,
                doc_id: *doc_id,
                prev_doc_id: None,
                pk: pk.clone(),
                doc: None,
            };
            if let Err(s) = self.wal_append(&wal_entry) {
                results.push(s);
                continue;
            }
            delete_store.mark_deleted(*doc_id);
            self.id_map.delete(pk)?;
            results.push(Status::default());
        }

        self.persist_delete_snapshot(delete_store)?;
        Ok(results)
    }

    // Persist the updated delete bitmap and advance version delete_suffix.
    fn persist_delete_snapshot(
        &self,
        delete_store: RwLockWriteGuard<'_, DeleteStore>,
    ) -> ZResult<()> {
        let new_suffix = delete_store.snapshot()?;
        drop(delete_store);
        let version = self.cur_version();
        let old_suffix = version.delete_suffix;
        let mut new_version = (*version).clone();
        new_version.delete_suffix = new_suffix;
        self.version_manager.flush(&new_version)?;
        self.delete_store.write().commit_snapshot(new_suffix);
        if old_suffix != new_suffix {
            let _ = std::fs::remove_file(self.path.join(format!("delete_{}.bitmap", old_suffix)));
            let _ = fs::File::open(&self.path).and_then(|d| d.sync_all());
        }
        Ok(())
    }

    // Live docs matching `filter`, sorted by doc_id; writing-segment versions win over
    // persisted ones for filter evaluation.
    fn collect_filter_deletions(
        &self,
        filter: &FilterExpr,
        delete_bitmap: &RoaringTreemap,
    ) -> ZResult<Vec<(u64, String)>> {
        let mut to_delete: HashMap<u64, String> = HashMap::new();
        let mut writing_doc_ids = roaring::RoaringTreemap::new();

        // Scan writing segment.
        {
            let writing = self.writing_segment.read();
            let matched = writing.scan_filter(filter, delete_bitmap);
            for (doc_id, doc) in matched {
                writing_doc_ids.insert(doc_id);
                to_delete.insert(doc_id, doc.pk.clone());
            }
        }

        // Scan persisted segments.
        let segs = self.persisted_segments.read();
        for seg in segs.iter() {
            let ids = seg.scan_filter_ids_limit(Some(filter), delete_bitmap, usize::MAX)?;
            if ids.is_empty() {
                continue;
            }
            let pks = seg.forward_store.read().get_pks_by_doc_ids(&ids)?;
            for (doc_id, maybe_pk) in ids.iter().zip(pks.iter()) {
                if delete_bitmap.contains(*doc_id) {
                    continue;
                }
                // If a doc_id exists in the writing segment, that version
                // is considered authoritative for filter evaluation.
                if writing_doc_ids.contains(*doc_id) {
                    continue;
                }
                let Some(pk) = maybe_pk else {
                    continue;
                };
                to_delete.entry(*doc_id).or_insert_with(|| pk.clone());
            }
        }
        drop(segs);

        let mut to_delete: Vec<(u64, String)> = to_delete.into_iter().collect();
        to_delete.sort_by_key(|(doc_id, _)| *doc_id);
        Ok(to_delete)
    }

    pub fn delete_by_filter(&self, filter_str: &str) -> ZResult<Status> {
        self.check_not_readonly()?;
        let version = self.cur_version();
        let filter = prepare_required_filter_expr(&version.schema, filter_str)?;
        let _guard = self.write_lock.lock();
        let mut delete_store = self.delete_store.write();
        let delete_bitmap = delete_store.bitmap();

        // Compute deletions first, then append WAL, then mutate state. This
        // avoids applying deletes that can't be made durable.
        let to_delete = self.collect_filter_deletions(&filter, delete_bitmap.as_ref())?;

        // Append WAL for all deletions.
        for (doc_id, pk) in &to_delete {
            let wal_entry = WalEntry {
                op: WalOp::Delete,
                doc_id: *doc_id,
                prev_doc_id: None,
                pk: pk.clone(),
                doc: None,
            };
            self.wal_append(&wal_entry)?;
        }

        // Apply deletions.
        for (doc_id, pk) in &to_delete {
            delete_store.mark_deleted(*doc_id);
            self.id_map.delete(pk)?;
        }

        self.persist_delete_snapshot(delete_store)?;
        Ok(Status::default())
    }
}

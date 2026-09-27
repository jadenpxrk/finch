use super::*;

impl WritingSegment {
    /// Fails where `insert` would reject a value of `doc`, so a caller can refuse the doc
    /// before logging it.
    pub fn check_insert(&self, doc: &Doc) -> ZResult<()> {
        // Sorted so the reported field does not depend on hash order.
        let mut indexed: Vec<_> = self.invert_indexes.iter().collect();
        indexed.sort_by_key(|(field_name, _)| *field_name);
        for (field_name, idx) in indexed {
            if let Some(val) = doc.fields.get(field_name).filter(|v| !v.is_null()) {
                idx.check_value(val)?;
            }
        }
        Ok(())
    }

    pub fn insert(&mut self, doc_id: u64, mut doc: Doc) -> ZResult<()> {
        doc.doc_id = doc_id;

        // WAL replay can repeat a doc_id; drop the earlier copy's index entries first.
        if let Some(old) = self.get_doc(doc_id) {
            self.remove_invert_entries(&old)?;
        }

        if let Err(e) = self.index_doc(&doc) {
            self.remove_vectors(doc_id);
            self.remove_invert_entries(&doc)?;
            return Err(e);
        }

        // The row goes in after the indexes so a failed insert never leaves one behind.
        self.forward_store.insert(doc_id, &doc)?;

        // Track range
        self.min_doc_id.fetch_min(doc_id, Ordering::Relaxed);
        self.max_doc_id.fetch_max(doc_id, Ordering::Relaxed);

        self.store_doc(doc_id, doc);
        Ok(())
    }

    fn index_doc(&mut self, doc: &Doc) -> ZResult<()> {
        for (field_name, idx) in &self.invert_indexes {
            match doc.fields.get(field_name) {
                Some(val) if !val.is_null() => {
                    idx.insert_nonnull_marker(doc.doc_id)?;
                    idx.insert(doc.doc_id, val)?;
                }
                _ => {
                    idx.insert_null_marker(doc.doc_id)?;
                }
            };
        }
        self.add_vectors(doc.doc_id, doc)
    }

    fn remove_invert_entries(&self, old: &Doc) -> ZResult<()> {
        for (field_name, idx) in &self.invert_indexes {
            match old.fields.get(field_name) {
                Some(v) if !v.is_null() => {
                    idx.delete(old.doc_id, v)?;
                    idx.delete_nonnull_marker(old.doc_id)?;
                }
                _ => {
                    idx.delete_null_marker(old.doc_id)?;
                }
            }
        }
        Ok(())
    }

    // Writing segments keep flat in-memory vector stores.
    fn add_vectors(&mut self, doc_id: u64, doc: &Doc) -> ZResult<()> {
        for (field, idx) in self.dense_indexes.iter_mut() {
            if let Some(v) = doc.get_vec_f32(field) {
                idx.add(doc_id, v)?;
            }
        }
        for (field, idx) in self.binary32_indexes.iter_mut() {
            if let Some(v) = doc.get_vec_u32(field) {
                idx.add(doc_id, v)?;
            }
        }
        for (field, idx) in self.binary64_indexes.iter_mut() {
            if let Some(v) = doc.get_vec_u64(field) {
                idx.add(doc_id, v)?;
            }
        }
        for (field, idx) in self.sparse_indexes.iter_mut() {
            if let Some((indices, values)) =
                doc.get_sparse_f32(field).filter(|(i, _)| !i.is_empty())
            {
                let sv = SparseVector::new(indices.to_vec(), values.to_vec());
                idx.add(doc_id, &sv)?;
            }
        }
        Ok(())
    }

    fn remove_vectors(&mut self, doc_id: u64) {
        for idx in self.dense_indexes.values_mut() {
            idx.remove(doc_id);
        }
        for idx in self.binary32_indexes.values_mut() {
            idx.remove(doc_id);
        }
        for idx in self.binary64_indexes.values_mut() {
            idx.remove(doc_id);
        }
        for idx in self.sparse_indexes.values_mut() {
            idx.remove(doc_id);
        }
    }

    // Replace existing entry (upsert/update) rather than appending a duplicate
    fn store_doc(&self, doc_id: u64, doc: Doc) {
        let mut docs = self.docs.write();
        let mut pos_map = self.doc_positions.write();
        if let Some(pos) = pos_map.get(&doc_id).copied() {
            if let Some(slot) = docs.get_mut(pos) {
                *slot = (doc_id, doc);
            } else {
                // Map out of sync (shouldn't happen); fallback to append.
                pos_map.insert(doc_id, docs.len());
                docs.push((doc_id, doc));
            }
            // doc_count unchanged: same doc replaced
        } else {
            self.doc_count.fetch_add(1, Ordering::Relaxed);
            pos_map.insert(doc_id, docs.len());
            docs.push((doc_id, doc));
        }
    }

    pub fn upsert(&mut self, doc_id: u64, doc: Doc, old_doc: Option<Doc>) -> ZResult<()> {
        // Remove old invert index entries
        if let Some(old) = old_doc {
            self.remove_invert_entries(&old)?;
            self.remove_vectors(doc_id);
        }
        self.insert(doc_id, doc)
    }

    pub fn update(&mut self, doc_id: u64, old_doc: Doc, new_doc: Doc) -> ZResult<()> {
        self.upsert(doc_id, new_doc, Some(old_doc))
    }

    pub fn doc_count(&self) -> u64 {
        self.doc_count.load(Ordering::Relaxed)
    }

    /// Scan docs in writing segment that match filter
    pub fn scan_filter(
        &self,
        filter: &FilterExpr,
        deleted: &roaring::RoaringTreemap,
    ) -> Vec<(u64, Arc<Doc>)> {
        let row_id_base = self.min_doc_id.load(Ordering::Relaxed);
        self.docs
            .read()
            .iter()
            .filter_map(|(doc_id, doc)| {
                if deleted.contains(*doc_id) {
                    return None;
                }
                let row_id = Some(doc_id.saturating_sub(row_id_base));
                if DocFilterEvaluator::passes(filter, doc, Some(deleted), row_id) {
                    Some((*doc_id, Arc::new(doc.clone())))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Scan doc_ids in the writing segment that match `filter` (or all docs if `filter` is None),
    /// returning at most `limit` doc_ids in doc_id order.
    pub fn scan_filter_ids_limit(
        &self,
        filter: Option<&FilterExpr>,
        deleted: &roaring::RoaringTreemap,
        limit: usize,
    ) -> Vec<u64> {
        if limit == 0 {
            return Vec::new();
        }

        let Some(expr) = filter else {
            let mut out: Vec<u64> = Vec::new();
            let docs = self.docs.read();
            for (doc_id, _) in docs.iter() {
                if deleted.contains(*doc_id) {
                    continue;
                }
                out.push(*doc_id);
                if out.len() >= limit {
                    break;
                }
            }
            return out;
        };

        let plan = self
            .invert_prefilter_plan(expr)
            .unwrap_or_else(|_| InvertPrefilterPlan::none());
        let row_id_base = self.min_doc_id.load(Ordering::Relaxed);

        let docs_guard = self.docs.read();
        let pos_guard = self.doc_positions.read();

        let Some(al) = plan.allowlist.as_ref() else {
            // No allowlist: fall back to full scan.
            return full_scan_ids(&docs_guard, expr, deleted, row_id_base, limit);
        };
        if plan.exact {
            return take_live_ids(al.iter(), deleted, limit);
        }

        let mut out: Vec<u64> = Vec::new();
        for doc_id in al.iter() {
            if deleted.contains(doc_id) {
                continue;
            }
            let Some(pos) = pos_guard.get(&doc_id).copied() else {
                continue;
            };
            let Some((_, doc)) = docs_guard.get(pos) else {
                continue;
            };
            let row_id = Some(doc_id.saturating_sub(row_id_base));
            if DocFilterEvaluator::passes(expr, doc, Some(deleted), row_id) {
                out.push(doc_id);
                if out.len() >= limit {
                    break;
                }
            }
        }
        out
    }

    /// Get a doc by doc_id
    pub fn get_doc(&self, doc_id: u64) -> Option<Doc> {
        let pos = self.doc_positions.read().get(&doc_id).copied()?;
        self.docs.read().get(pos).map(|(_, doc)| doc.clone())
    }

    /// All (doc_id, doc) pairs for vector index building
    pub fn all_docs(&self) -> Vec<(u64, Doc)> {
        self.docs.read().clone()
    }

    /// Direct access to flat dense vectors for a field, avoiding the all_docs() clone.
    /// Returns (keys, flat_vectors, dim) or None if the field has no dense index.
    pub fn dense_vectors(&self, field: &str) -> Option<(&[u64], &[f32], usize)> {
        let idx = self.dense_indexes.get(field)?;
        if idx.keys.is_empty() {
            return None;
        }
        Some((&idx.keys, &idx.vectors, idx.dim))
    }
}

fn full_scan_ids(
    docs: &[(u64, Doc)],
    expr: &FilterExpr,
    deleted: &roaring::RoaringTreemap,
    row_id_base: u64,
    limit: usize,
) -> Vec<u64> {
    let mut out: Vec<u64> = Vec::new();
    for (doc_id, doc) in docs.iter() {
        if deleted.contains(*doc_id) {
            continue;
        }
        let row_id = Some(doc_id.saturating_sub(row_id_base));
        if DocFilterEvaluator::passes(expr, doc, Some(deleted), row_id) {
            out.push(*doc_id);
            if out.len() >= limit {
                break;
            }
        }
    }
    out
}

use super::*;

const FETCH_BATCH: usize = 512;

/// Fetches candidate docs in batches and keeps the ones that pass the filter.
struct BatchedIdScan<'a> {
    segment: &'a PersistedSegment,
    filter: Option<&'a FilterExpr>,
    deleted: &'a roaring::RoaringTreemap,
    limit: usize,
    forward_for_row_id: Option<Arc<MmapForwardStore>>,
    batch_ids: Vec<u64>,
    out: Vec<u64>,
}

impl BatchedIdScan<'_> {
    fn feed(&mut self, ids: impl IntoIterator<Item = u64>) -> ZResult<()> {
        for doc_id in ids {
            if self.deleted.contains(doc_id) {
                continue;
            }
            self.batch_ids.push(doc_id);
            if self.batch_ids.len() >= FETCH_BATCH {
                self.flush()?;
                if self.out.len() >= self.limit {
                    break;
                }
            }
        }
        Ok(())
    }

    fn flush(&mut self) -> ZResult<()> {
        if self.batch_ids.is_empty() || self.out.len() >= self.limit {
            self.batch_ids.clear();
            return Ok(());
        }
        let docs = self.segment.fetch_docs(&self.batch_ids)?;
        for (doc_id, maybe_doc) in self.batch_ids.iter().zip(docs.iter()) {
            if self.out.len() >= self.limit {
                break;
            }
            if self.deleted.contains(*doc_id) {
                continue;
            }
            let Some(doc) = maybe_doc else {
                continue;
            };
            if let Some(expr) = self.filter {
                let row_id = self
                    .forward_for_row_id
                    .as_ref()
                    .and_then(|f| f.row_index_of_doc_id(*doc_id));
                if !DocFilterEvaluator::passes(expr, doc, Some(self.deleted), row_id) {
                    continue;
                }
            }
            self.out.push(*doc_id);
        }
        self.batch_ids.clear();
        Ok(())
    }
}

impl PersistedSegment {
    /// Return up to `limit` doc_ids (ascending) that match `filter` (or all docs if `filter` is None).
    ///
    /// This uses the inverted-index prefilter allowlist when available to avoid full forward scans.
    pub fn scan_filter_ids_limit(
        &self,
        filter: Option<&FilterExpr>,
        deleted: &roaring::RoaringTreemap,
        limit: usize,
    ) -> ZResult<Vec<u64>> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let (allowlist, exact) = match filter {
            None => (None, true),
            Some(expr) => {
                let plan = self.invert_prefilter_plan(expr)?;
                (plan.allowlist, plan.exact)
            }
        };

        // Exact allowlist: no forward fetch needed.
        if exact {
            if let Some(al) = allowlist.as_ref() {
                return Ok(take_live_ids(al.iter(), deleted, limit));
            }
            // No allowlist: walk forward store doc_ids (do not assume contiguous ranges).
            let ids = self.forward_store.read().all_doc_ids();
            return Ok(take_live_ids(ids, deleted, limit));
        }

        // Non-exact allowlist: fetch + evaluate only those candidates.
        let needs_row_id = filter.is_some_and(filter_expr_uses_local_row_id);
        let mut scan = BatchedIdScan {
            segment: self,
            filter,
            deleted,
            limit,
            forward_for_row_id: needs_row_id.then(|| self.forward_store.read().clone()),
            batch_ids: Vec::with_capacity(FETCH_BATCH),
            out: Vec::new(),
        };
        if let Some(al) = allowlist.as_ref() {
            scan.feed(al.iter())?;
        } else {
            // No allowlist: scan forward store doc_ids in batches (do not assume contiguous ranges).
            scan.feed(self.forward_store.read().all_doc_ids())?;
        }
        scan.flush()?;
        Ok(scan.out)
    }

    pub(super) fn invert_prefilter_plan(&self, expr: &FilterExpr) -> ZResult<InvertPrefilterPlan> {
        // use the segment's logical row count, not the doc_id span
        // (doc_ids may be sparse after deletes / compaction).
        let max_range_hits = max_range_hits(self.doc_count);
        plan_for(expr, &|field, lookup| {
            let invert = self.invert_indexes.read();
            let Some(idx) = invert.get(field) else {
                return Ok(None);
            };
            lookup.run(idx, max_range_hits)
        })
    }
}

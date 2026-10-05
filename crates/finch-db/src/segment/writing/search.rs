use super::*;

impl WritingSegment {
    pub fn search_dense_vectors(
        &self,
        field: &str,
        query: &[f32],
        topk: usize,
        delete_bitmap: &roaring::RoaringTreemap,
        filter_expr: Option<&FilterExpr>,
    ) -> ZResult<Vec<(u64, f32)>> {
        let Some(idx) = self.dense_indexes.get(field) else {
            return Ok(Vec::new());
        };

        let plan = if let Some(expr) = filter_expr {
            self.invert_prefilter_plan(expr)?
        } else {
            InvertPrefilterPlan::none()
        };
        let allowlist = plan.allowlist;
        let exact = plan.exact;
        let eval_expr = filter_expr.filter(|_| !exact);

        let docs_guard = self.docs.read();
        let pos_guard = self.doc_positions.read();

        let cfg = crate::config::global_config();
        let bf_threshold = (self.doc_count() as f64 * cfg.brute_force_by_keys_ratio as f64) as u64;
        if let Some(al) = allowlist
            .as_ref()
            .filter(|al| exact && al.len() <= bf_threshold)
        {
            return Ok(idx.search_by_keys(query, topk, al.iter(), |doc_id| {
                !delete_bitmap.contains(doc_id)
            }));
        }

        let row_id_base = self.min_doc_id.load(Ordering::Relaxed);
        Ok(idx.search(query, topk, |doc_id| {
            if delete_bitmap.contains(doc_id) {
                return false;
            }
            if let Some(al) = &allowlist {
                if !al.contains(doc_id) {
                    return false;
                }
            }
            let Some(expr) = eval_expr else {
                return true;
            };
            let Some(pos) = pos_guard.get(&doc_id).copied() else {
                return false;
            };
            let Some((_, doc)) = docs_guard.get(pos) else {
                return false;
            };
            let row_id = Some(doc_id.saturating_sub(row_id_base));
            DocFilterEvaluator::passes(expr, doc, Some(delete_bitmap), row_id)
        }))
    }

    pub fn search_binary_u32(
        &self,
        field: &str,
        query: &[u32],
        topk: usize,
        delete_bitmap: &roaring::RoaringTreemap,
        filter_expr: Option<&FilterExpr>,
    ) -> ZResult<Vec<(u64, f32)>> {
        let Some(idx) = self.binary32_indexes.get(field) else {
            return Ok(Vec::new());
        };

        let plan = if let Some(expr) = filter_expr {
            self.invert_prefilter_plan(expr)?
        } else {
            InvertPrefilterPlan::none()
        };
        let allowlist = plan.allowlist;
        let exact = plan.exact;
        let eval_expr = filter_expr.filter(|_| !exact);

        let docs_guard = self.docs.read();
        let pos_guard = self.doc_positions.read();

        let cfg = crate::config::global_config();
        let bf_threshold = (self.doc_count() as f64 * cfg.brute_force_by_keys_ratio as f64) as u64;
        if let Some(al) = allowlist
            .as_ref()
            .filter(|al| exact && al.len() <= bf_threshold)
        {
            return Ok(idx.search_by_keys(query, topk, al.iter(), |doc_id| {
                !delete_bitmap.contains(doc_id)
            }));
        }

        let row_id_base = self.min_doc_id.load(Ordering::Relaxed);
        Ok(idx.search(query, topk, |doc_id| {
            if delete_bitmap.contains(doc_id) {
                return false;
            }
            if let Some(al) = &allowlist {
                if !al.contains(doc_id) {
                    return false;
                }
            }
            let Some(expr) = eval_expr else {
                return true;
            };
            let Some(pos) = pos_guard.get(&doc_id).copied() else {
                return false;
            };
            let Some((_, doc)) = docs_guard.get(pos) else {
                return false;
            };
            let row_id = Some(doc_id.saturating_sub(row_id_base));
            DocFilterEvaluator::passes(expr, doc, Some(delete_bitmap), row_id)
        }))
    }

    pub fn search_binary_u64(
        &self,
        field: &str,
        query: &[u64],
        topk: usize,
        delete_bitmap: &roaring::RoaringTreemap,
        filter_expr: Option<&FilterExpr>,
    ) -> ZResult<Vec<(u64, f32)>> {
        let Some(idx) = self.binary64_indexes.get(field) else {
            return Ok(Vec::new());
        };

        let plan = if let Some(expr) = filter_expr {
            self.invert_prefilter_plan(expr)?
        } else {
            InvertPrefilterPlan::none()
        };
        let allowlist = plan.allowlist;
        let exact = plan.exact;
        let eval_expr = filter_expr.filter(|_| !exact);

        let docs_guard = self.docs.read();
        let pos_guard = self.doc_positions.read();

        let cfg = crate::config::global_config();
        let bf_threshold = (self.doc_count() as f64 * cfg.brute_force_by_keys_ratio as f64) as u64;
        if let Some(al) = allowlist
            .as_ref()
            .filter(|al| exact && al.len() <= bf_threshold)
        {
            return Ok(idx.search_by_keys(query, topk, al.iter(), |doc_id| {
                !delete_bitmap.contains(doc_id)
            }));
        }

        let row_id_base = self.min_doc_id.load(Ordering::Relaxed);
        Ok(idx.search(query, topk, |doc_id| {
            if delete_bitmap.contains(doc_id) {
                return false;
            }
            if let Some(al) = &allowlist {
                if !al.contains(doc_id) {
                    return false;
                }
            }
            let Some(expr) = eval_expr else {
                return true;
            };
            let Some(pos) = pos_guard.get(&doc_id).copied() else {
                return false;
            };
            let Some((_, doc)) = docs_guard.get(pos) else {
                return false;
            };
            let row_id = Some(doc_id.saturating_sub(row_id_base));
            DocFilterEvaluator::passes(expr, doc, Some(delete_bitmap), row_id)
        }))
    }

    pub fn search_sparse_vectors(
        &self,
        field: &str,
        query: &SparseVector,
        topk: usize,
        delete_bitmap: &roaring::RoaringTreemap,
        filter_expr: Option<&FilterExpr>,
    ) -> ZResult<Vec<(u64, f32)>> {
        let Some(idx) = self.sparse_indexes.get(field) else {
            return Ok(Vec::new());
        };

        let plan = if let Some(expr) = filter_expr {
            self.invert_prefilter_plan(expr)?
        } else {
            InvertPrefilterPlan::none()
        };
        let allowlist = plan.allowlist;
        let exact = plan.exact;
        let eval_expr = filter_expr.filter(|_| !exact);

        let docs_guard = self.docs.read();
        let pos_guard = self.doc_positions.read();

        let cfg = crate::config::global_config();
        let bf_threshold = (self.doc_count() as f64 * cfg.brute_force_by_keys_ratio as f64) as u64;
        if let Some(al) = allowlist
            .as_ref()
            .filter(|al| exact && al.len() <= bf_threshold)
        {
            return Ok(idx.search_by_keys(query, topk, al.iter(), |doc_id| {
                !delete_bitmap.contains(doc_id)
            }));
        }

        let row_id_base = self.min_doc_id.load(Ordering::Relaxed);
        Ok(idx.search(query, topk, |doc_id| {
            if delete_bitmap.contains(doc_id) {
                return false;
            }
            if let Some(al) = &allowlist {
                if !al.contains(doc_id) {
                    return false;
                }
            }
            let Some(expr) = eval_expr else {
                return true;
            };
            let Some(pos) = pos_guard.get(&doc_id).copied() else {
                return false;
            };
            let Some((_, doc)) = docs_guard.get(pos) else {
                return false;
            };
            let row_id = Some(doc_id.saturating_sub(row_id_base));
            DocFilterEvaluator::passes(expr, doc, Some(delete_bitmap), row_id)
        }))
    }
}

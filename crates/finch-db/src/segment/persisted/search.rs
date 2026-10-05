use super::*;

type DocFetchFn = Arc<dyn Fn(u64) -> ZResult<Option<Doc>> + Send + Sync>;

/// Index knobs a query sets through `QueryParams`; `None` keeps each index's default.
#[derive(Clone, Copy, Debug, Default)]
pub struct IndexQueryParams {
    pub ef_or_nprobe: Option<u32>,
    pub hnsw_upper_ef: Option<u32>,
    pub hnsw_l0_seeds: Option<u32>,
}

impl From<&QueryParams> for IndexQueryParams {
    fn from(params: &QueryParams) -> Self {
        IndexQueryParams {
            ef_or_nprobe: params.ef.or(params.n_probe),
            hnsw_upper_ef: params.hnsw_upper_ef,
            hnsw_l0_seeds: params.hnsw_l0_seeds,
        }
    }
}

/// Per-segment ANN search settings.
pub struct AnnSearch<'a> {
    pub topk: usize,
    pub index_params: IndexQueryParams,
    pub force_linear: bool,
    pub delete_bitmap: Arc<roaring::RoaringTreemap>,
    pub filter_expr: Option<&'a FilterExpr>,
}

/// How an indexed search restricts its candidates.
enum IndexCandidates {
    /// Non-empty, non-deleted ids of an exact allowlist small enough to score directly.
    SmallExact(Vec<u64>),
    /// Search the index with this filter (None when nothing needs filtering).
    Index(Option<Box<dyn DocFilter>>),
}

/// Forward-store access for filters that evaluate the expression per doc.
struct ForwardDocAccess {
    doc_fetch: DocFetchFn,
    row_id_for_doc_id: Option<RowIdFn>,
}

/// Rows for a brute-force scan over the forward store.
struct BruteForceRows<'a> {
    ids: Vec<u64>,
    docs: Vec<Option<Doc>>,
    /// None when the exact allowlist already answered the filter.
    eval_expr: Option<&'a FilterExpr>,
    needs_row_id: bool,
    forward: Arc<MmapForwardStore>,
}

fn sort_topk(mut candidates: Vec<(u64, f32)>, topk: usize) -> Vec<(u64, f32)> {
    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(topk);
    candidates
}

impl PersistedSegment {
    pub fn compute_dense_distances_by_doc_ids(
        &self,
        field: &str,
        query: &[f32],
        metric: MetricType,
        doc_ids: &[u64],
    ) -> ZResult<Vec<Option<f32>>> {
        self.forward_store
            .read()
            .compute_dense_distance_by_doc_ids(field, query, metric, doc_ids)
    }

    // Uses the logical row count (doc_count), not the doc_id span.
    fn brute_force_threshold(&self) -> u64 {
        let cfg = crate::config::global_config();
        (self.doc_count as f64 * cfg.brute_force_by_keys_ratio as f64) as u64
    }

    fn index_candidates(
        &self,
        filter_expr: Option<&FilterExpr>,
        delete_bitmap: &Arc<roaring::RoaringTreemap>,
    ) -> ZResult<IndexCandidates> {
        let Some(expr) = filter_expr else {
            if delete_bitmap.is_empty() {
                return Ok(IndexCandidates::Index(None));
            }
            let filter = DeletedDocFilter::new(delete_bitmap.clone(), None);
            return Ok(IndexCandidates::Index(Some(Box::new(filter))));
        };
        let plan = self.invert_prefilter_plan(expr)?;
        // A very small inverted allowlist bypasses the ANN index.
        if let Some(ids) = self.small_exact_ids(&plan, delete_bitmap) {
            return Ok(IndexCandidates::SmallExact(ids));
        }
        let filter = self.prefilter_doc_filter(expr, plan, delete_bitmap);
        Ok(IndexCandidates::Index(Some(filter)))
    }

    fn small_exact_ids(
        &self,
        plan: &InvertPrefilterPlan,
        delete_bitmap: &roaring::RoaringTreemap,
    ) -> Option<Vec<u64>> {
        let bf_threshold = self.brute_force_threshold();
        let al = plan
            .allowlist
            .as_ref()
            .filter(|al| plan.exact && al.len() <= bf_threshold)?;
        let ids: Vec<u64> = al
            .iter()
            .filter(|id| !delete_bitmap.contains(*id))
            .collect();
        (!ids.is_empty()).then_some(ids)
    }

    fn prefilter_doc_filter(
        &self,
        expr: &FilterExpr,
        plan: InvertPrefilterPlan,
        delete_bitmap: &Arc<roaring::RoaringTreemap>,
    ) -> Box<dyn DocFilter> {
        let delete_clone = delete_bitmap.clone();
        let Some(bm) = plan.allowlist else {
            let access = self.forward_doc_access(expr);
            return Box::new(ExprDocFilter::new(
                expr.clone(),
                access.doc_fetch,
                delete_clone,
                access.row_id_for_doc_id,
            ));
        };
        let bm = Arc::new(bm);
        if plan.exact {
            return Box::new(BitmapDocFilter::new(bm, delete_clone));
        }
        let access = self.forward_doc_access(expr);
        Box::new(BitmapExprDocFilter::new(
            bm,
            expr.clone(),
            access.doc_fetch,
            delete_clone,
            access.row_id_for_doc_id,
        ))
    }

    fn forward_doc_access(&self, expr: &FilterExpr) -> ForwardDocAccess {
        let forward = self.forward_store.read().clone();
        let row_id_for_doc_id: Option<RowIdFn> = if filter_expr_uses_local_row_id(expr) {
            let forward_for_row_id = forward.clone();
            Some(Arc::new(move |id: u64| {
                forward_for_row_id.row_index_of_doc_id(id)
            }))
        } else {
            None
        };
        let doc_fetch: DocFetchFn = Arc::new(move |id: u64| {
            Ok(forward.get_by_doc_ids(&[id])?.into_iter().next().flatten())
        });
        ForwardDocAccess {
            doc_fetch,
            row_id_for_doc_id,
        }
    }

    /// Search dense vector field with optional delete + expr filter.
    /// Falls back to brute-force scan when no vector index exists.
    pub fn search_vectors(
        &self,
        field: &str,
        query: &[f32],
        search: AnnSearch<'_>,
        metric: MetricType,
    ) -> ZResult<Vec<(u64, f32)>> {
        let AnnSearch {
            topk,
            index_params,
            force_linear,
            delete_bitmap,
            filter_expr,
        } = search;
        let index = self.vector_indexes.read().get(field).cloned();
        let Some(index) = index.filter(|_| !force_linear) else {
            return self.search_brute_force(field, query, topk, delete_bitmap, filter_expr, metric);
        };
        match self.index_candidates(filter_expr, &delete_bitmap)? {
            IndexCandidates::SmallExact(ids) => {
                self.score_dense_ids(field, query, metric, ids, topk)
            }
            IndexCandidates::Index(filter) => {
                index.search(query, topk, index_params, filter.as_deref())
            }
        }
    }

    /// Search binary vector field (u32 words) using Hamming distance.
    pub fn search_binary_vectors_u32(
        &self,
        field: &str,
        query: &[u32],
        topk: usize,
        force_linear: bool,
        delete_bitmap: Arc<roaring::RoaringTreemap>,
        filter_expr: Option<&FilterExpr>,
    ) -> ZResult<Vec<(u64, f32)>> {
        let index = self.vector_indexes.read().get(field).cloned();
        let Some(index) = index.filter(|_| !force_linear) else {
            return self.search_binary_u32_brute_force(
                field,
                query,
                topk,
                delete_bitmap,
                filter_expr,
            );
        };
        let VectorIndex::FlatBinary32(index) = index.as_ref() else {
            return Err(Status::invalid_argument(
                "binary32 field indexed with non-binary index",
            ));
        };
        match self.index_candidates(filter_expr, &delete_bitmap)? {
            IndexCandidates::SmallExact(ids) => self.score_binary_u32_ids(field, query, &ids, topk),
            IndexCandidates::Index(filter) => index.search(query, topk, filter.as_deref()),
        }
    }

    /// Search binary vector field (u64 words) using Hamming distance.
    pub fn search_binary_vectors_u64(
        &self,
        field: &str,
        query: &[u64],
        topk: usize,
        force_linear: bool,
        delete_bitmap: Arc<roaring::RoaringTreemap>,
        filter_expr: Option<&FilterExpr>,
    ) -> ZResult<Vec<(u64, f32)>> {
        let index = self.vector_indexes.read().get(field).cloned();
        let Some(index) = index.filter(|_| !force_linear) else {
            return self.search_binary_u64_brute_force(
                field,
                query,
                topk,
                delete_bitmap,
                filter_expr,
            );
        };
        let VectorIndex::FlatBinary64(index) = index.as_ref() else {
            return Err(Status::invalid_argument(
                "binary64 field indexed with non-binary index",
            ));
        };
        match self.index_candidates(filter_expr, &delete_bitmap)? {
            IndexCandidates::SmallExact(ids) => self.score_binary_u64_ids(field, query, &ids, topk),
            IndexCandidates::Index(filter) => index.search(query, topk, filter.as_deref()),
        }
    }

    /// Forward-store rows to scan; an exact allowlist under the threshold replaces the full scan.
    fn brute_force_rows<'a>(
        &self,
        filter_expr: Option<&'a FilterExpr>,
    ) -> ZResult<BruteForceRows<'a>> {
        let bf_threshold = self.brute_force_threshold();
        let mut ids = self.forward_store.read().all_doc_ids();
        let mut eval_expr = filter_expr;
        if let Some(expr) = filter_expr {
            let plan = self.invert_prefilter_plan(expr)?;
            let exact = plan.exact;
            if let Some(al) = plan
                .allowlist
                .filter(|al| exact && al.len() <= bf_threshold)
            {
                ids = al.iter().collect::<Vec<u64>>();
                eval_expr = None;
            }
        }

        let docs = self.forward_store.read().get_by_doc_ids(&ids)?;
        let needs_row_id = filter_expr.is_some_and(filter_expr_uses_local_row_id);
        let forward = self.forward_store.read().clone();
        Ok(BruteForceRows {
            ids,
            docs,
            eval_expr,
            needs_row_id,
            forward,
        })
    }

    fn search_binary_u32_brute_force(
        &self,
        field: &str,
        query: &[u32],
        topk: usize,
        delete_bitmap: Arc<roaring::RoaringTreemap>,
        filter_expr: Option<&FilterExpr>,
    ) -> ZResult<Vec<(u64, f32)>> {
        let rows = self.brute_force_rows(filter_expr)?;
        let mut candidates: Vec<(u64, f32)> = Vec::new();
        for (id, maybe_doc) in rows.ids.iter().zip(rows.docs.iter()) {
            if delete_bitmap.contains(*id) {
                continue;
            }
            let Some(doc) = maybe_doc else {
                continue;
            };
            if let Some(expr) = rows.eval_expr {
                let row_id = if rows.needs_row_id {
                    rows.forward.row_index_of_doc_id(*id)
                } else {
                    None
                };
                if !DocFilterEvaluator::passes(expr, doc, None, row_id) {
                    continue;
                }
            }
            if let Some(vec) = doc.get_vec_u32(field) {
                let dist = compute_hamming_u32(vec, query);
                candidates.push((*id, dist));
            }
        }
        Ok(sort_topk(candidates, topk))
    }

    fn search_binary_u64_brute_force(
        &self,
        field: &str,
        query: &[u64],
        topk: usize,
        delete_bitmap: Arc<roaring::RoaringTreemap>,
        filter_expr: Option<&FilterExpr>,
    ) -> ZResult<Vec<(u64, f32)>> {
        let rows = self.brute_force_rows(filter_expr)?;
        let mut candidates: Vec<(u64, f32)> = Vec::new();
        for (id, maybe_doc) in rows.ids.iter().zip(rows.docs.iter()) {
            if delete_bitmap.contains(*id) {
                continue;
            }
            let Some(doc) = maybe_doc else {
                continue;
            };
            if let Some(expr) = rows.eval_expr {
                let row_id = if rows.needs_row_id {
                    rows.forward.row_index_of_doc_id(*id)
                } else {
                    None
                };
                if !DocFilterEvaluator::passes(expr, doc, None, row_id) {
                    continue;
                }
            }
            if let Some(vec) = doc.get_vec_u64(field) {
                let dist = compute_hamming_u64(vec, query);
                candidates.push((*id, dist));
            }
        }
        Ok(sort_topk(candidates, topk))
    }

    /// Search sparse vector field; falls back to brute-force scan when no index.
    pub fn search_sparse_vectors(
        &self,
        field: &str,
        query: &SparseVector,
        search: AnnSearch<'_>,
    ) -> ZResult<Vec<(u64, f32)>> {
        let AnnSearch {
            topk,
            index_params,
            force_linear,
            delete_bitmap,
            filter_expr,
        } = search;
        let index = self.vector_indexes.read().get(field).cloned();
        let Some(index) = index.filter(|_| !force_linear) else {
            return self.search_sparse_brute_force(field, query, topk, delete_bitmap, filter_expr);
        };
        match self.index_candidates(filter_expr, &delete_bitmap)? {
            IndexCandidates::SmallExact(ids) => self.score_sparse_ids(field, query, &ids, topk),
            IndexCandidates::Index(filter) => {
                index.search_sparse(query, topk, index_params.ef_or_nprobe, filter.as_deref())
            }
        }
    }

    /// Brute-force dense scan over the forward store using the field's metric.
    fn search_brute_force(
        &self,
        field: &str,
        query: &[f32],
        topk: usize,
        delete_bitmap: Arc<roaring::RoaringTreemap>,
        filter_expr: Option<&FilterExpr>,
        metric: MetricType,
    ) -> ZResult<Vec<(u64, f32)>> {
        let rows = self.brute_force_rows(filter_expr)?;
        let mut candidates: Vec<(u64, f32)> = Vec::new();
        for (id, maybe_doc) in rows.ids.iter().zip(rows.docs.iter()) {
            if delete_bitmap.contains(*id) {
                continue;
            }
            let Some(doc) = maybe_doc else {
                continue;
            };
            if let Some(expr) = rows.eval_expr {
                let row_id = if rows.needs_row_id {
                    rows.forward.row_index_of_doc_id(*id)
                } else {
                    None
                };
                if !DocFilterEvaluator::passes(expr, doc, None, row_id) {
                    continue;
                }
            }
            if let Some(vec) = doc.get_vec_f32(field) {
                let dist = compute_distance(vec, query, metric);
                candidates.push((*id, dist));
            }
        }
        Ok(sort_topk(candidates, topk))
    }

    /// Brute-force sparse scan over the forward store (neg inner product).
    fn search_sparse_brute_force(
        &self,
        field: &str,
        query: &SparseVector,
        topk: usize,
        delete_bitmap: Arc<roaring::RoaringTreemap>,
        filter_expr: Option<&FilterExpr>,
    ) -> ZResult<Vec<(u64, f32)>> {
        let rows = self.brute_force_rows(filter_expr)?;
        let mut candidates: Vec<(u64, f32)> = Vec::new();
        for (id, maybe_doc) in rows.ids.iter().zip(rows.docs.iter()) {
            if delete_bitmap.contains(*id) {
                continue;
            }
            let Some(doc) = maybe_doc else {
                continue;
            };
            if let Some(expr) = rows.eval_expr {
                let row_id = if rows.needs_row_id {
                    rows.forward.row_index_of_doc_id(*id)
                } else {
                    None
                };
                if !DocFilterEvaluator::passes(expr, doc, None, row_id) {
                    continue;
                }
            }
            if let Some((indices, values)) = doc.get_sparse_f32(field) {
                let sv = SparseVector::new(indices.to_vec(), values.to_vec());
                // Neg inner product: smaller = more similar
                let score = -query.dot(&sv);
                candidates.push((*id, score));
            }
        }
        Ok(sort_topk(candidates, topk))
    }

    fn score_dense_ids(
        &self,
        field: &str,
        query: &[f32],
        metric: MetricType,
        ids: Vec<u64>,
        topk: usize,
    ) -> ZResult<Vec<(u64, f32)>> {
        let dists = self.compute_dense_distances_by_doc_ids(field, query, metric, &ids)?;
        let mut out: Vec<(u64, f32)> = Vec::new();
        for (id, d) in ids.into_iter().zip(dists.into_iter()) {
            if let Some(d) = d {
                out.push((id, d));
            }
        }
        Ok(sort_topk(out, topk))
    }

    fn score_binary_u32_ids(
        &self,
        field: &str,
        query: &[u32],
        ids: &[u64],
        topk: usize,
    ) -> ZResult<Vec<(u64, f32)>> {
        let docs = self.forward_store.read().get_by_doc_ids(ids)?;
        let mut candidates: Vec<(u64, f32)> = Vec::new();
        for (id, maybe_doc) in ids.iter().zip(docs.iter()) {
            let Some(doc) = maybe_doc else {
                continue;
            };
            if let Some(vec) = doc.get_vec_u32(field) {
                let dist = compute_hamming_u32(vec, query);
                candidates.push((*id, dist));
            }
        }
        Ok(sort_topk(candidates, topk))
    }

    fn score_binary_u64_ids(
        &self,
        field: &str,
        query: &[u64],
        ids: &[u64],
        topk: usize,
    ) -> ZResult<Vec<(u64, f32)>> {
        let docs = self.forward_store.read().get_by_doc_ids(ids)?;
        let mut candidates: Vec<(u64, f32)> = Vec::new();
        for (id, maybe_doc) in ids.iter().zip(docs.iter()) {
            let Some(doc) = maybe_doc else {
                continue;
            };
            if let Some(vec) = doc.get_vec_u64(field) {
                let dist = compute_hamming_u64(vec, query);
                candidates.push((*id, dist));
            }
        }
        Ok(sort_topk(candidates, topk))
    }

    fn score_sparse_ids(
        &self,
        field: &str,
        query: &SparseVector,
        ids: &[u64],
        topk: usize,
    ) -> ZResult<Vec<(u64, f32)>> {
        let docs = self.forward_store.read().get_by_doc_ids(ids)?;
        let mut candidates: Vec<(u64, f32)> = Vec::new();
        for (id, maybe_doc) in ids.iter().zip(docs.iter()) {
            let Some(doc) = maybe_doc else {
                continue;
            };
            if let Some((indices, values)) = doc.get_sparse_f32(field) {
                let sv = SparseVector::new(indices.to_vec(), values.to_vec());
                let score = -query.dot(&sv);
                candidates.push((*id, score));
            }
        }
        Ok(sort_topk(candidates, topk))
    }

    /// Fetch docs by doc_ids
    pub fn fetch_docs(&self, doc_ids: &[u64]) -> ZResult<Vec<Option<Doc>>> {
        self.forward_store.read().get_by_doc_ids(doc_ids)
    }
}

/// Compute distance between two dense vectors according to the given metric.
/// Lower is better (more similar) in all cases.
pub fn compute_distance(a: &[f32], b: &[f32], metric: MetricType) -> f32 {
    match metric {
        MetricType::InnerProduct => {
            // Neg inner product so smaller = more similar
            -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
        }
        MetricType::MipsL2 => {
            // localized spherical injection (e2=0.0)
            //   dist = 2 - 2 * ip(a,b) / max(||a||^2, ||b||^2)
            let mut ip = 0.0f32;
            let mut u2 = 0.0f32;
            let mut v2 = 0.0f32;
            for (x, y) in a.iter().zip(b.iter()) {
                ip += x * y;
                u2 += x * x;
                v2 += y * y;
            }
            let denom = u2.max(v2);
            2.0 - 2.0 * ip / denom
        }
        MetricType::Cosine => {
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            let denom = na * nb;
            if denom == 0.0 {
                0.0
            } else {
                1.0 - dot / denom
            }
        }
        MetricType::Hamming => {
            // Bit differences over the raw f32 payloads, as finch-core computes them.
            let acc = a.iter().zip(b).fold(0u32, |acc, (x, y)| {
                acc.wrapping_add((x.to_bits() ^ y.to_bits()).count_ones())
            });
            acc as f32
        }
        // L2 (default) and Undefined fall back to squared Euclidean.
        _ => a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum(),
    }
}

/// Hamming distance between two u32-word binary vectors (smaller is more similar).
pub fn compute_hamming_u32(a: &[u32], b: &[u32]) -> f32 {
    let n = a.len().min(b.len());
    let mut acc: u32 = 0;
    for i in 0..n {
        acc = acc.wrapping_add((a[i] ^ b[i]).count_ones());
    }
    acc as f32
}

/// Hamming distance between two u64-word binary vectors (smaller is more similar).
pub fn compute_hamming_u64(a: &[u64], b: &[u64]) -> f32 {
    let n = a.len().min(b.len());
    let mut acc: u32 = 0;
    for i in 0..n {
        acc = acc.wrapping_add((a[i] ^ b[i]).count_ones());
    }
    acc as f32
}

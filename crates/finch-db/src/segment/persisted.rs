//! Persisted (read-only, memory-mapped) segment

use super::prefilter::{max_range_hits, plan_for, take_live_ids, InvertPrefilterPlan};
use super::writing::WrittenSegmentMeta;
use crate::invert::InvertIndex;
use crate::sqlengine::executor::DocFilterEvaluator;
use crate::sqlengine::parser::FilterExpr;
use finch_core::algorithm::flat::{
    FlatBinary32Searcher, FlatBinary64Searcher, FlatSearcher, SegmentBytes,
    StorageReader as CoreStorageReader,
};
use finch_core::algorithm::flat_sparse::{FlatSparseSearcher, SparseVector};
use finch_core::algorithm::hnsw::HnswSearcher;
use finch_core::algorithm::hnsw_sparse::HnswSparseSearcher;
use finch_core::algorithm::ivf::IvfSearcher;
use finch_core::algorithm::DocFilter;
use finch_storage::{ForwardStoreOpenOptions, MmapForwardStore};
use finch_types::{Doc, MetricType, Status, ZResult, SYS_LOCAL_ROW_ID};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

type RowIdFn = Arc<dyn Fn(u64) -> Option<u64> + Send + Sync>;

fn filter_expr_uses_local_row_id(expr: &FilterExpr) -> bool {
    match expr {
        FilterExpr::AlwaysTrue | FilterExpr::AlwaysFalse => false,
        FilterExpr::And(a, b) | FilterExpr::Or(a, b) => {
            filter_expr_uses_local_row_id(a) || filter_expr_uses_local_row_id(b)
        }
        FilterExpr::Not(inner) => filter_expr_uses_local_row_id(inner),
        FilterExpr::Compare { field, .. }
        | FilterExpr::IsNull(field)
        | FilterExpr::IsNotNull(field)
        | FilterExpr::ArrayLengthCompare { field, .. }
        | FilterExpr::InList { field, .. }
        | FilterExpr::HasPrefix { field, .. }
        | FilterExpr::HasSuffix { field, .. }
        | FilterExpr::LikePattern { field, .. }
        | FilterExpr::ContainAll { field, .. }
        | FilterExpr::ContainAny { field, .. } => field == SYS_LOCAL_ROW_ID,
    }
}

/// A concrete vector index loaded from disk
pub enum VectorIndex {
    Flat(FlatSearcher),
    FlatBinary32(FlatBinary32Searcher),
    FlatBinary64(FlatBinary64Searcher),
    FlatSparse(FlatSparseSearcher),
    Hnsw(HnswSearcher),
    HnswSparse(HnswSparseSearcher),
    Ivf(IvfSearcher),
}

impl VectorIndex {
    pub fn search(
        &self,
        query: &[f32],
        topk: usize,
        ef_or_nprobe: Option<u32>,
        filter: Option<&dyn DocFilter>,
    ) -> ZResult<Vec<(u64, f32)>> {
        match self {
            VectorIndex::Flat(s) => s.search(query, topk, filter),
            VectorIndex::FlatBinary32(_) | VectorIndex::FlatBinary64(_) => Err(
                Status::invalid_argument("use search_binary for binary flat indexes"),
            ),
            VectorIndex::Hnsw(s) => {
                let ef = ef_or_nprobe.unwrap_or(s.default_ef as u32) as usize;
                s.search(query, topk, ef, filter)
            }
            VectorIndex::Ivf(s) => {
                // default nprobe is 10.
                let n_probe = ef_or_nprobe.unwrap_or(10) as usize;
                s.search(query, topk, n_probe, filter)
            }
            VectorIndex::FlatSparse(_) | VectorIndex::HnswSparse(_) => Err(
                Status::invalid_argument("use search_sparse for sparse indexes"),
            ),
        }
    }

    pub fn search_sparse(
        &self,
        query: &SparseVector,
        topk: usize,
        ef_or_nprobe: Option<u32>,
        filter: Option<&dyn DocFilter>,
    ) -> ZResult<Vec<(u64, f32)>> {
        match self {
            VectorIndex::FlatSparse(s) => s.search(query, topk, filter),
            VectorIndex::HnswSparse(s) => {
                // default ef_search is 300.
                let ef = ef_or_nprobe.unwrap_or(300) as usize;
                s.search(query, topk, ef, filter)
            }
            _ => Err(Status::invalid_argument("use search for dense indexes")),
        }
    }
}

/// File-backed StorageReader wrapping a directory of segment files
pub struct FileStorageReader {
    base_path: PathBuf,
    enable_mmap: bool,
    cache: Mutex<HashMap<String, SegmentBytes>>,
}

impl FileStorageReader {
    pub fn new(path: impl Into<PathBuf>, enable_mmap: bool) -> Self {
        FileStorageReader {
            base_path: path.into(),
            enable_mmap,
            cache: Mutex::new(HashMap::new()),
        }
    }
}

impl CoreStorageReader for FileStorageReader {
    fn read_segment(&self, name: &str) -> ZResult<SegmentBytes> {
        let path = self.base_path.join(name);
        if let Some(b) = self.cache.lock().get(name).cloned() {
            return Ok(b);
        }

        let bytes = if self.enable_mmap {
            let file = std::fs::File::open(&path).map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    Status::not_found(format!("segment file '{}' not found", name))
                } else {
                    Status::io_error(e.to_string())
                }
            })?;
            // SAFETY: segment files are written once and not modified while mapped.
            let mmap = unsafe { memmap2::Mmap::map(&file) }
                .map_err(|e| Status::io_error(e.to_string()))?;
            SegmentBytes::Mmap(Arc::new(mmap))
        } else {
            let data = std::fs::read(&path).map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    Status::not_found(format!("segment file '{}' not found", name))
                } else {
                    Status::io_error(e.to_string())
                }
            })?;
            SegmentBytes::Shared(Arc::new(data))
        };
        self.cache.lock().insert(name.to_string(), bytes.clone());
        Ok(bytes)
    }

    fn exists(&self, name: &str) -> bool {
        self.base_path.join(name).exists()
    }
}

/// Filter that rejects deleted doc IDs
pub struct DeletedDocFilter {
    bitmap: Arc<roaring::RoaringTreemap>,
    inner: Option<Arc<dyn DocFilter>>,
}

impl DeletedDocFilter {
    pub fn new(bitmap: Arc<roaring::RoaringTreemap>, inner: Option<Arc<dyn DocFilter>>) -> Self {
        DeletedDocFilter { bitmap, inner }
    }
}

impl DocFilter for DeletedDocFilter {
    fn is_valid(&self, doc_id: u64) -> ZResult<bool> {
        if self.bitmap.contains(doc_id) {
            return Ok(false);
        }
        if let Some(inner) = &self.inner {
            inner.is_valid(doc_id)
        } else {
            Ok(true)
        }
    }
}

/// Filter that uses a forward-scan predicate expression
pub struct ExprDocFilter {
    expr: FilterExpr,
    doc_fetch: Arc<dyn Fn(u64) -> ZResult<Option<Doc>> + Send + Sync>,
    delete_bitmap: Arc<roaring::RoaringTreemap>,
    row_id_for_doc_id: Option<RowIdFn>,
}

impl ExprDocFilter {
    pub fn new(
        expr: FilterExpr,
        doc_fetch: Arc<dyn Fn(u64) -> ZResult<Option<Doc>> + Send + Sync>,
        delete_bitmap: Arc<roaring::RoaringTreemap>,
        row_id_for_doc_id: Option<RowIdFn>,
    ) -> Self {
        ExprDocFilter {
            expr,
            doc_fetch,
            delete_bitmap,
            row_id_for_doc_id,
        }
    }
}

impl DocFilter for ExprDocFilter {
    fn is_valid(&self, doc_id: u64) -> ZResult<bool> {
        if self.delete_bitmap.contains(doc_id) {
            return Ok(false);
        }
        if let Some(doc) = (self.doc_fetch)(doc_id)? {
            let row_id = self.row_id_for_doc_id.as_ref().and_then(|f| f(doc_id));
            Ok(DocFilterEvaluator::passes(&self.expr, &doc, None, row_id))
        } else {
            Ok(false)
        }
    }
}

/// Filter that restricts candidates to an allowlist bitmap (and rejects deletes).
pub struct BitmapDocFilter {
    allowlist: Arc<roaring::RoaringTreemap>,
    delete_bitmap: Arc<roaring::RoaringTreemap>,
}

impl BitmapDocFilter {
    pub fn new(
        allowlist: Arc<roaring::RoaringTreemap>,
        delete_bitmap: Arc<roaring::RoaringTreemap>,
    ) -> Self {
        BitmapDocFilter {
            allowlist,
            delete_bitmap,
        }
    }
}

impl DocFilter for BitmapDocFilter {
    fn is_valid(&self, doc_id: u64) -> ZResult<bool> {
        if self.delete_bitmap.contains(doc_id) {
            return Ok(false);
        }
        Ok(self.allowlist.contains(doc_id))
    }
}

/// Allowlist + expression evaluation. The allowlist must be a superset of the
/// expression's matches; it's used only to short-circuit obvious non-matches.
pub struct BitmapExprDocFilter {
    allowlist: Arc<roaring::RoaringTreemap>,
    expr: FilterExpr,
    doc_fetch: Arc<dyn Fn(u64) -> ZResult<Option<Doc>> + Send + Sync>,
    delete_bitmap: Arc<roaring::RoaringTreemap>,
    row_id_for_doc_id: Option<RowIdFn>,
}

impl BitmapExprDocFilter {
    pub fn new(
        allowlist: Arc<roaring::RoaringTreemap>,
        expr: FilterExpr,
        doc_fetch: Arc<dyn Fn(u64) -> ZResult<Option<Doc>> + Send + Sync>,
        delete_bitmap: Arc<roaring::RoaringTreemap>,
        row_id_for_doc_id: Option<RowIdFn>,
    ) -> Self {
        BitmapExprDocFilter {
            allowlist,
            expr,
            doc_fetch,
            delete_bitmap,
            row_id_for_doc_id,
        }
    }
}

impl DocFilter for BitmapExprDocFilter {
    fn is_valid(&self, doc_id: u64) -> ZResult<bool> {
        if self.delete_bitmap.contains(doc_id) {
            return Ok(false);
        }
        if !self.allowlist.contains(doc_id) {
            return Ok(false);
        }
        if let Some(doc) = (self.doc_fetch)(doc_id)? {
            let row_id = self.row_id_for_doc_id.as_ref().and_then(|f| f(doc_id));
            Ok(DocFilterEvaluator::passes(&self.expr, &doc, None, row_id))
        } else {
            Ok(false)
        }
    }
}

/// A persisted segment (immutable, memory-mapped or file-backed)
pub struct PersistedSegment {
    pub id: u32,
    pub min_doc_id: u64,
    pub max_doc_id: u64,
    pub doc_count: u64,
    pub forward_store: parking_lot::RwLock<Arc<MmapForwardStore>>,
    pub(crate) invert_indexes: parking_lot::RwLock<HashMap<String, Arc<InvertIndex>>>,
    /// Interior-mutable so create_index can load indexes after initial open
    pub vector_indexes: parking_lot::RwLock<HashMap<String, Arc<VectorIndex>>>,
}

mod filter;
mod open;
mod search;

pub use search::{compute_distance, compute_hamming_u32, compute_hamming_u64, AnnSearch};

#[cfg(test)]
mod tests;

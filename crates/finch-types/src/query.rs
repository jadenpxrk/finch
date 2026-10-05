use crate::doc::Value;
use crate::types::FileFormat;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Runtime query parameters
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryParams {
    /// HNSW search ef parameter (overrides default)
    pub ef: Option<u32>,
    /// IVF n_probe parameter (number of cells to search)
    pub n_probe: Option<u32>,
    /// Per-query concurrency override (number of worker threads).
    ///
    /// If not set, Finch uses the process-wide `GlobalConfigData.query_thread_count`
    /// to size Finch's dedicated query thread pool.
    pub concurrency: Option<usize>,
    /// Restrict a brute-force scan to a provided list of primary keys.
    ///
    /// When set, finch will resolve PKs -> doc_ids and compute exact distances
    /// only over that candidate set.
    pub bf_pks: Option<Vec<String>>,
    /// Search radius threshold. `None` or `<= 0` disables radius filtering.
    ///
    /// Note: Finch currently applies this as a post-filter on the distance/score
    /// used for ranking (lower is better).
    pub radius: Option<f32>,
    /// Force linear (brute-force) search even if an index exists.
    pub is_linear: Option<bool>,
    /// If true, recompute exact distances on a candidate set fetched from storage and rerank.
    ///
    /// This is primarily useful when the underlying index uses quantized vectors.
    #[serde(default)]
    pub use_refiner: bool,
    /// Global candidate limit to refine (defaults to a small multiple of `topk` when refiner is enabled).
    pub refiner_k: Option<u32>,
    /// Refiner scale factor. When `use_refiner=true` and `refiner_k` is not set,
    /// finch may derive `refiner_k = ceil(topk * refiner_scale_factor)`.
    pub refiner_scale_factor: Option<f32>,
    /// HNSW upper-level beam width; `None` descends greedily.
    #[serde(default)]
    pub hnsw_upper_ef: Option<u32>,
    /// Level-1 routing points that seed the HNSW level-0 beam; `None` uses 24, `1` one entry.
    #[serde(default)]
    pub hnsw_l0_seeds: Option<u32>,
}

impl QueryParams {
    pub fn effective_is_linear(&self) -> bool {
        self.is_linear.unwrap_or(false)
    }

    pub fn effective_radius(&self) -> Option<f32> {
        let r = self.radius?;
        if !r.is_finite() || r <= 0.0 {
            None
        } else {
            Some(r)
        }
    }

    pub fn effective_refiner_k(&self, topk: usize) -> u32 {
        if !self.use_refiner {
            return topk as u32;
        }

        if let Some(k) = self.refiner_k {
            return k.max(topk as u32).min(10_000);
        }

        if let Some(sf) = self.refiner_scale_factor {
            if sf.is_finite() && sf > 0.0 {
                let k = ((topk as f32) * sf).ceil() as u32;
                return k.max(topk as u32).min(10_000);
            }
        }

        let base = (topk as u32).saturating_mul(10).max(topk as u32);
        base.min(10_000)
    }
}

/// A vector similarity search query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorQuery {
    /// Number of nearest neighbors to return
    pub topk: usize,
    /// Name of the vector field to search
    pub field_name: String,
    /// Use an existing document (by primary key) as the query vector source.
    #[serde(default)]
    pub id: Option<String>,
    /// Dense query vector (raw f32 bytes or f32 slice)
    pub query_vector: Vec<f32>,
    /// Binary query vector for `VectorBinary32` fields (word-aligned).
    #[serde(default)]
    pub query_vector_u32: Vec<u32>,
    /// Binary query vector for `VectorBinary64` fields (word-aligned).
    #[serde(default)]
    pub query_vector_u64: Vec<u64>,
    /// Sparse query: indices (for sparse vector fields)
    pub sparse_indices: Vec<u32>,
    /// Sparse query: values
    pub sparse_values: Vec<f32>,
    /// SQL WHERE filter expression
    pub filter: Option<String>,
    /// Whether to include vectors in results
    pub include_vector: bool,
    /// Whether to include internal doc IDs in results
    pub include_doc_id: bool,
    /// Which scalar fields to return:
    /// - `None` => all fields
    /// - `Some(vec![])` => no fields
    /// - `Some([...])` => selected fields
    pub output_fields: Option<Vec<String>>,
    /// Runtime search parameters
    pub query_params: QueryParams,
}

impl VectorQuery {
    pub fn new(field_name: impl Into<String>, query_vector: Vec<f32>, topk: usize) -> Self {
        VectorQuery {
            topk,
            field_name: field_name.into(),
            id: None,
            query_vector,
            query_vector_u32: Vec::new(),
            query_vector_u64: Vec::new(),
            sparse_indices: Vec::new(),
            sparse_values: Vec::new(),
            filter: None,
            include_vector: false,
            include_doc_id: false,
            output_fields: None,
            query_params: QueryParams::default(),
        }
    }

    pub fn new_binary32(
        field_name: impl Into<String>,
        query_vector: Vec<u32>,
        topk: usize,
    ) -> Self {
        VectorQuery {
            topk,
            field_name: field_name.into(),
            id: None,
            query_vector: Vec::new(),
            query_vector_u32: query_vector,
            query_vector_u64: Vec::new(),
            sparse_indices: Vec::new(),
            sparse_values: Vec::new(),
            filter: None,
            include_vector: false,
            include_doc_id: false,
            output_fields: None,
            query_params: QueryParams::default(),
        }
    }

    pub fn new_binary64(
        field_name: impl Into<String>,
        query_vector: Vec<u64>,
        topk: usize,
    ) -> Self {
        VectorQuery {
            topk,
            field_name: field_name.into(),
            id: None,
            query_vector: Vec::new(),
            query_vector_u32: Vec::new(),
            query_vector_u64: query_vector,
            sparse_indices: Vec::new(),
            sparse_values: Vec::new(),
            filter: None,
            include_vector: false,
            include_doc_id: false,
            output_fields: None,
            query_params: QueryParams::default(),
        }
    }

    pub fn with_filter(mut self, filter: impl Into<String>) -> Self {
        self.filter = Some(filter.into());
        self
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn with_output_fields(mut self, fields: Vec<String>) -> Self {
        self.output_fields = Some(fields);
        self
    }

    pub fn with_params(mut self, params: QueryParams) -> Self {
        self.query_params = params;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effective_radius() {
        let mut q = QueryParams::default();
        assert_eq!(q.effective_radius(), None);
        q.radius = Some(0.0);
        assert_eq!(q.effective_radius(), None);
        q.radius = Some(-1.0);
        assert_eq!(q.effective_radius(), None);
        q.radius = Some(1.25);
        assert_eq!(q.effective_radius(), Some(1.25));
    }

    #[test]
    fn test_effective_refiner_k_priority() {
        let topk = 10usize;

        // Default: no refiner => topk
        let q = QueryParams::default();
        assert_eq!(q.effective_refiner_k(topk), 10);

        // Refiner enabled with explicit refiner_k
        let mut q = QueryParams {
            use_refiner: true,
            refiner_k: Some(3),
            ..Default::default()
        };
        assert_eq!(q.effective_refiner_k(topk), 10); // never below topk
        q.refiner_k = Some(1234);
        assert_eq!(q.effective_refiner_k(topk), 1234);

        // Refiner scale factor derives refiner_k when not provided
        let q = QueryParams {
            use_refiner: true,
            refiner_scale_factor: Some(2.0),
            ..Default::default()
        };
        assert_eq!(q.effective_refiner_k(topk), 20);

        // Bad scale factor falls back to default multiplier
        let q = QueryParams {
            use_refiner: true,
            refiner_scale_factor: Some(0.0),
            ..Default::default()
        };
        assert_eq!(q.effective_refiner_k(topk), 100);
    }
}

/// A group-by vector search query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupByVectorQuery {
    pub base: VectorQuery,
    /// Field to group results by
    pub group_by_field: String,
    /// Maximum number of docs to return *per group*.
    pub group_count: usize,
    /// Maximum number of distinct groups to return ("top groups").
    pub group_topk: usize,
}

/// Result from a group-by query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupResult {
    pub group_value: Value,
    pub docs: Vec<crate::doc::Doc>,
}

/// Collection open options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionOptions {
    #[serde(default)]
    pub read_only: bool,
    #[serde(default = "CollectionOptions::default_enable_mmap")]
    pub enable_mmap: bool,
    /// Optional storage mode for vector index files.
    ///
    /// - `None`: `enable_mmap` decides
    /// - `Some(Mmap|Memory)`: overrides `enable_mmap` for index loading
    /// - `Some(BufferPool)`: currently unsupported for vector indexes
    /// - `Some(None)`: treated like unset
    #[serde(default)]
    pub index_storage: Option<crate::types::StorageType>,
    /// Optional storage mode for the forward store.
    ///
    /// - Arrow IPC: `Mmap` uses mmap, `Memory/BufferPool` uses heap-backed reads
    /// - Parquet: `Memory` eagerly loads the entire file on open; anything else uses
    ///   a bounded row-group cache (bufferpool-like)
    #[serde(default)]
    pub forward_storage: Option<crate::types::StorageType>,
    /// Maximum in-memory write buffer size in bytes.
    #[serde(default = "CollectionOptions::default_max_buffer_size")]
    pub max_buffer_size: u32,
    /// Persisted forward-store file format (finch-only).
    ///
    /// `None` means "use the default" (currently Arrow IPC).
    #[serde(default)]
    pub forward_file_format: Option<FileFormat>,
}

impl Default for CollectionOptions {
    fn default() -> Self {
        Self::read_write()
    }
}

impl CollectionOptions {
    pub const DEFAULT_MAX_BUFFER_SIZE: u32 = 64 * 1024 * 1024;

    const fn default_enable_mmap() -> bool {
        true
    }

    const fn default_max_buffer_size() -> u32 {
        Self::DEFAULT_MAX_BUFFER_SIZE
    }

    pub fn read_write() -> Self {
        CollectionOptions {
            read_only: false,
            enable_mmap: true,
            index_storage: None,
            forward_storage: None,
            max_buffer_size: Self::DEFAULT_MAX_BUFFER_SIZE,
            forward_file_format: None,
        }
    }

    pub fn read_only() -> Self {
        CollectionOptions {
            read_only: true,
            enable_mmap: true,
            index_storage: None,
            forward_storage: None,
            max_buffer_size: Self::DEFAULT_MAX_BUFFER_SIZE,
            forward_file_format: None,
        }
    }

    pub fn with_mmap(mut self) -> Self {
        self.enable_mmap = true;
        self
    }
}

/// Statistics for a single vector field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorFieldStats {
    pub field_name: String,
    pub doc_count: u64,
    pub indexed: bool,
}

/// Collection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionStats {
    pub doc_count: u64,
    pub segment_count: usize,
    pub vector_field_stats: Vec<VectorFieldStats>,
    /// per-vector-field completeness ratio in [0,1].
    /// 1.0 means all live docs are covered by an index for that field.
    #[serde(default)]
    pub index_completeness: HashMap<String, f32>,
}

impl CollectionStats {
    pub fn to_string_formatted(&self, indent_level: usize) -> String {
        let indent = " ".repeat(indent_level);
        let mut out = String::new();
        out.push_str(&format!("{indent}doc_count: {}\n", self.doc_count));
        out.push_str(&format!("{indent}segment_count: {}\n", self.segment_count));
        if !self.index_completeness.is_empty() {
            out.push_str(&format!("{indent}index_completeness:\n"));
            let mut items: Vec<_> = self.index_completeness.iter().collect();
            items.sort_by(|a, b| a.0.cmp(b.0));
            for (k, v) in items {
                out.push_str(&format!("{indent}  {k}: {v:.3}\n"));
            }
        }
        out
    }
}

/// Options for create_index DDL
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CreateIndexOptions {
    /// Rebuild index even if one exists
    pub rebuild: bool,
    /// Concurrency limit for index building
    pub concurrency: Option<usize>,
}

/// Options for add_column DDL
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AddColumnOptions {
    pub rebuild_index: bool,
    pub concurrency: Option<usize>,
}

/// Options for alter_column DDL
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AlterColumnOptions {
    pub rebuild_index: bool,
    pub concurrency: Option<usize>,
}

/// Options for optimize (compaction)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OptimizeOptions {
    /// Maximum number of segments to merge
    pub max_segments: Option<usize>,
    /// Target segment size in bytes
    pub target_segment_size: Option<u64>,
    /// Concurrency limit for compaction/index rebuild
    pub concurrency: Option<usize>,
}

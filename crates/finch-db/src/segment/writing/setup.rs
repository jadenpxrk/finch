use super::*;

type VectorIndexMaps = (
    HashMap<String, DenseMemIndex>,
    HashMap<String, Binary32MemIndex>,
    HashMap<String, Binary64MemIndex>,
    HashMap<String, SparseMemIndex>,
);

impl WritingSegment {
    pub(super) fn init_vector_indexes(schema: &CollectionSchema) -> ZResult<VectorIndexMaps> {
        let mut dense = HashMap::new();
        let mut b32 = HashMap::new();
        let mut b64 = HashMap::new();
        let mut sparse = HashMap::new();

        for f in schema.vector_fields() {
            let dim = f.dimension.unwrap_or(0);
            if !f.data_type.is_sparse()
                && dim == 0
                && !matches!(f.data_type, DataType::SparseFp16 | DataType::SparseFp32)
            {
                // Dense vectors must have dim; schema validation enforces this.
            }

            let (metric, quantize) = match f.index_params.as_ref() {
                Some(p) => (
                    p.metric().unwrap_or(MetricType::InnerProduct),
                    p.quantize().unwrap_or(QuantizeType::Undefined),
                ),
                None => (MetricType::InnerProduct, QuantizeType::Undefined),
            };

            match f.data_type {
                DataType::VectorBinary32 => {
                    b32.insert(f.name.clone(), Binary32MemIndex::new(dim));
                }
                DataType::VectorBinary64 => {
                    b64.insert(f.name.clone(), Binary64MemIndex::new(dim));
                }
                DataType::SparseFp16 | DataType::SparseFp32 => {
                    sparse.insert(f.name.clone(), SparseMemIndex::new(quantize));
                }
                _ => {
                    // All non-binary dense vectors are normalized to VecF32 on write.
                    dense.insert(f.name.clone(), DenseMemIndex::new(dim, metric, quantize));
                }
            }
        }

        Ok((dense, b32, b64, sparse))
    }

    pub fn new(id: u32, schema: CollectionSchema) -> ZResult<Self> {
        let forward_store = MemoryForwardStore::new(schema.clone());
        let (dense_indexes, binary32_indexes, binary64_indexes, sparse_indexes) =
            Self::init_vector_indexes(&schema)?;
        Ok(WritingSegment {
            id,
            schema: schema.clone(),
            forward_store,
            invert_indexes: HashMap::new(),
            dense_indexes,
            binary32_indexes,
            binary64_indexes,
            sparse_indexes,
            doc_count: AtomicU64::new(0),
            min_doc_id: AtomicU64::new(u64::MAX),
            max_doc_id: AtomicU64::new(0),
            docs: parking_lot::RwLock::new(Vec::new()),
            doc_positions: parking_lot::RwLock::new(HashMap::new()),
        })
    }

    pub fn new_with_invert(
        id: u32,
        schema: CollectionSchema,
        invert_base_path: &Path,
    ) -> ZResult<Self> {
        let forward_store = MemoryForwardStore::new(schema.clone());
        let mut invert_indexes = HashMap::new();
        let (dense_indexes, binary32_indexes, binary64_indexes, sparse_indexes) =
            Self::init_vector_indexes(&schema)?;

        // Persist invert indexes under the writing segment directory, matching
        // `dump()`'s `invert_paths` layout: `seg_{id}/{field}_invert`.
        let seg_path = invert_base_path.join(format!("seg_{}", id));
        fs::create_dir_all(&seg_path).map_err(|e| Status::io_error(e.to_string()))?;
        // Best-effort crash safety: persist the segment directory entry.
        sync_dir_best_effort(&seg_path);
        sync_dir_best_effort(invert_base_path);

        for field in schema.inverted_index_fields() {
            let Some(finch_types::IndexParams::Invert(params)) = field.index_params.as_ref() else {
                continue;
            };
            let path = seg_path.join(format!("{}_invert", field.name));
            let idx =
                InvertIndex::open(&path, field.name.clone(), field.data_type, params.clone())?;
            invert_indexes.insert(field.name.clone(), idx);
        }

        Ok(WritingSegment {
            id,
            schema,
            forward_store,
            invert_indexes,
            dense_indexes,
            binary32_indexes,
            binary64_indexes,
            sparse_indexes,
            doc_count: AtomicU64::new(0),
            min_doc_id: AtomicU64::new(u64::MAX),
            max_doc_id: AtomicU64::new(0),
            docs: parking_lot::RwLock::new(Vec::new()),
            doc_positions: parking_lot::RwLock::new(HashMap::new()),
        })
    }
}

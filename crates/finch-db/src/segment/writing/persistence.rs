use super::*;

impl WritingSegment {
    /// Dump writing segment to disk as Arrow IPC forward store
    pub fn dump(
        &self,
        base_path: &Path,
        forward_file_format: finch_types::FileFormat,
    ) -> ZResult<WrittenSegmentMeta> {
        let seg_path = base_path.join(format!("seg_{}", self.id));
        std::fs::create_dir_all(&seg_path).map_err(|e| Status::io_error(e.to_string()))?;
        // Best-effort crash safety: persist the segment directory entry.
        sync_dir_best_effort(&seg_path);
        sync_dir_best_effort(base_path);

        let forward_path = match forward_file_format {
            finch_types::FileFormat::Parquet => seg_path.join("forward.parquet"),
            _ => seg_path.join("forward.arrow"),
        };
        match forward_file_format {
            finch_types::FileFormat::Parquet => {
                self.forward_store.dump_to_parquet(&forward_path)?
            }
            _ => self.forward_store.dump_to_ipc(&forward_path)?,
        }

        let mut invert_paths: HashMap<String, InvertIndexMeta> = HashMap::new();
        for (name, idx) in &self.invert_indexes {
            idx.freeze()?;
            let meta = InvertIndexMeta {
                path: seg_path.join(format!("{}_invert", name)),
                data_type: idx.data_type,
                params: idx.params.clone(),
            };
            invert_paths.insert(name.clone(), meta);
        }

        Ok(WrittenSegmentMeta {
            segment_id: self.id,
            min_doc_id: self.min_doc_id.load(Ordering::Relaxed),
            max_doc_id: self.max_doc_id.load(Ordering::Relaxed),
            doc_count: self.doc_count.load(Ordering::Relaxed),
            forward_path,
            invert_paths,
        })
    }

    /// Apply a new schema to this writing segment without re-opening invert index DBs.
    ///
    /// This is primarily used when schema changes only affect vector index params
    /// (e.g. switching metric/quantize/index family). Writing segments always use
    /// in-memory flat searchers, but they must use the *current* metric/quantize
    /// settings to match persisted/indexed behavior.
    pub(crate) fn apply_schema_update(&mut self, schema: CollectionSchema) -> ZResult<()> {
        self.schema = schema.clone();

        let (mut dense, mut b32, mut b64, mut sparse) = Self::init_vector_indexes(&schema)?;

        // Rebuild in-memory vector stores from buffered docs to keep results
        // consistent if the writing segment is non-empty.
        let docs = self.docs.read();
        for (doc_id, doc) in docs.iter() {
            for (field, idx) in dense.iter_mut() {
                if let Some(v) = doc.get_vec_f32(field) {
                    let _ = idx.add(*doc_id, v);
                }
            }
            for (field, idx) in b32.iter_mut() {
                if let Some(v) = doc.get_vec_u32(field) {
                    let _ = idx.add(*doc_id, v);
                }
            }
            for (field, idx) in b64.iter_mut() {
                if let Some(v) = doc.get_vec_u64(field) {
                    let _ = idx.add(*doc_id, v);
                }
            }
            for (field, idx) in sparse.iter_mut() {
                if let Some((indices, values)) =
                    doc.get_sparse_f32(field).filter(|(i, _)| !i.is_empty())
                {
                    let sv = SparseVector::new(indices.to_vec(), values.to_vec());
                    let _ = idx.add(*doc_id, &sv);
                }
            }
        }

        self.dense_indexes = dense;
        self.binary32_indexes = b32;
        self.binary64_indexes = b64;
        self.sparse_indexes = sparse;
        Ok(())
    }
}

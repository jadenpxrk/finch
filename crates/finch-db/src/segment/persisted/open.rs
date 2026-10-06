use super::*;
use crate::invert::persisted_index_exists;

impl PersistedSegment {
    #[cfg(test)]
    pub(crate) fn open(meta: &WrittenSegmentMeta) -> ZResult<Self> {
        Self::open_with_forward_store_options(meta, ForwardStoreOpenOptions::default())
    }

    pub fn open_with_forward_store_options(
        meta: &WrittenSegmentMeta,
        forward_opts: ForwardStoreOpenOptions,
    ) -> ZResult<Self> {
        let forward_store = Arc::new(MmapForwardStore::open_with_options(
            &meta.forward_path,
            forward_opts,
        )?);

        // Persisted segments are immutable: open invert indexes read-only.
        let mut invert_indexes: HashMap<String, Arc<InvertIndex>> = HashMap::new();
        for (field_name, inv) in &meta.invert_paths {
            if !persisted_index_exists(&inv.path) {
                continue;
            }
            let idx = InvertIndex::open_read_only(
                &inv.path,
                field_name.clone(),
                inv.data_type,
                inv.params.clone(),
            )?;
            invert_indexes.insert(field_name.clone(), Arc::new(idx));
        }

        Ok(PersistedSegment {
            id: meta.segment_id,
            min_doc_id: meta.min_doc_id,
            max_doc_id: meta.max_doc_id,
            doc_count: meta.doc_count,
            forward_store: parking_lot::RwLock::new(forward_store),
            invert_indexes: parking_lot::RwLock::new(invert_indexes),
            vector_indexes: parking_lot::RwLock::new(HashMap::new()),
        })
    }

    pub fn open_forward_only_with_mmap(
        meta: &WrittenSegmentMeta,
        enable_mmap: bool,
    ) -> ZResult<Self> {
        Self::open_forward_only_with_forward_store_options(
            meta,
            ForwardStoreOpenOptions {
                enable_mmap,
                ..ForwardStoreOpenOptions::default()
            },
        )
    }

    pub fn open_forward_only_with_forward_store_options(
        meta: &WrittenSegmentMeta,
        forward_opts: ForwardStoreOpenOptions,
    ) -> ZResult<Self> {
        let forward_store = Arc::new(MmapForwardStore::open_with_options(
            &meta.forward_path,
            forward_opts,
        )?);
        Ok(PersistedSegment {
            id: meta.segment_id,
            min_doc_id: meta.min_doc_id,
            max_doc_id: meta.max_doc_id,
            doc_count: meta.doc_count,
            forward_store: parking_lot::RwLock::new(forward_store),
            invert_indexes: parking_lot::RwLock::new(HashMap::new()),
            vector_indexes: parking_lot::RwLock::new(HashMap::new()),
        })
    }

    /// Populate invert indexes from `meta` (read-only).
    pub fn load_invert_indexes(&self, meta: &WrittenSegmentMeta) -> ZResult<()> {
        let mut invert_indexes: HashMap<String, Arc<InvertIndex>> = HashMap::new();
        for (field_name, inv) in &meta.invert_paths {
            if !persisted_index_exists(&inv.path) {
                continue;
            }
            let idx = InvertIndex::open_read_only(
                &inv.path,
                field_name.clone(),
                inv.data_type,
                inv.params.clone(),
            )?;
            invert_indexes.insert(field_name.clone(), Arc::new(idx));
        }
        *self.invert_indexes.write() = invert_indexes;
        Ok(())
    }

    pub fn add_vector_index(&self, field_name: String, index: VectorIndex) {
        self.vector_indexes
            .write()
            .insert(field_name, Arc::new(index));
    }
}

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use finch_types::{CollectionSchema, IndexParams, Status, ZResult};

use crate::collection_files::{
    cleanup_old_id_map_checkpoints, effective_forward_file_format, effective_index_enable_mmap,
    id_map_path_for_suffix, wal_path_for_writing_segment,
};
use crate::index::IndexBuilder;
use crate::segment::persisted::PersistedSegment;
use crate::segment::writing::{WritingSegment, WrittenSegmentMeta};
use crate::version::{PersistedSegmentVersion, Version};
use crate::wal::NextWal;

use super::{make_writing_segment, Collection};

// The dumped writing segment, reopened as a persisted segment awaiting the manifest commit.
struct FlushedSegment {
    seg: PersistedSegment,
    active_id_map_path: PathBuf,
    new_id_map_suffix: u32,
    old_delete_suffix: u32,
    new_delete_suffix: u32,
    next_wal: Option<NextWal>,
}

impl Collection {
    /// Force flush the writing segment to disk
    pub fn flush(&self) -> ZResult<()> {
        self.check_not_readonly()?;
        let _guard = self.write_lock.lock();
        self.flush_writing_segment_locked()
    }

    pub(super) fn flush_writing_segment_locked(&self) -> ZResult<()> {
        let Some((meta, indexed_fields)) = self.dump_writing_segment_locked()? else {
            return Ok(());
        };

        let seg = PersistedSegment::open_forward_only_with_mmap(&meta, self.options.enable_mmap)?;
        seg.load_invert_indexes(&meta)?;

        let version = self.cur_version();
        let loaded_indexed_fields =
            self.load_flushed_vector_indexes(&version, &seg, meta.segment_id, &indexed_fields)?;
        let new_seg_id = version.next_segment_id.max(meta.segment_id + 1);
        let next_wal = self
            .wal
            .lock()
            .as_ref()
            .map(|wal| wal.create_next(wal_path_for_writing_segment(&self.path, new_seg_id)))
            .transpose()?;

        // Snapshot delete bitmap before truncating WAL. This guarantees that
        // any tombstones (update/upsert replacements) become durable even if
        // they weren't explicitly flushed via `delete()` calls.
        let old_delete_suffix = version.delete_suffix;
        let new_delete_suffix = {
            let store = self.delete_store.read();
            if store.modified_since_last_snapshot() {
                store.snapshot()?
            } else {
                old_delete_suffix
            }
        };

        let mut new_version = (*version).clone();
        new_version.delete_suffix = new_delete_suffix;

        new_version
            .persisted_segments
            .push(PersistedSegmentVersion {
                segment_id: meta.segment_id,
                min_doc_id: meta.min_doc_id,
                max_doc_id: meta.max_doc_id,
                doc_count: meta.doc_count,
                indexed_vector_fields: loaded_indexed_fields,
                vector_index_generations: Default::default(),
            });
        new_version.next_segment_id = new_seg_id;
        new_version.writing_segment_id = Some(new_seg_id);

        // Snapshot the id_map to a versioned checkpoint before committing the
        // manifest.
        let active_id_map_path = self.id_map.path().to_path_buf();
        let new_id_map_suffix = version.id_map_suffix + 1;
        let new_id_map_path = id_map_path_for_suffix(&self.path, new_id_map_suffix);
        self.id_map.create_snapshot(&new_id_map_path)?;
        new_version.id_map_suffix = new_id_map_suffix;

        let new_writing =
            self.prepare_replacement_writing_segment_locked(Some(new_seg_id), &version.schema)?;
        let flushed = FlushedSegment {
            seg,
            active_id_map_path,
            new_id_map_suffix,
            old_delete_suffix,
            new_delete_suffix,
            next_wal,
        };

        self.commit_manifest_with_prepared(
            &new_version,
            new_writing,
            |new_writing| self.publish_flushed_segment(new_writing, flushed),
            |new_writing| {
                drop(new_writing);
                self.reopen_writing_invert_indexes_best_effort();
            },
        )
    }

    // Dumps the writing segment and builds its cheap vector indexes; `None` when it is empty.
    fn dump_writing_segment_locked(&self) -> ZResult<Option<(WrittenSegmentMeta, Vec<String>)>> {
        let writing = self.writing_segment.write();
        if writing.doc_count() == 0 {
            return Ok(None);
        }
        // Flush Durability::None fjall writes to disk before persisting.
        writing.sync_invert_indexes()?;
        let meta = writing.dump(&self.path, effective_forward_file_format(&self.options))?;

        // Build vector indexes directly from in-memory writing segment (same
        // pattern as compaction). This avoids a full disk round-trip when
        // optimize() runs later.
        let version = self.cur_version();
        let indexed_field_params: Vec<(String, IndexParams)> = version
            .schema
            .indexed_vector_fields()
            .filter_map(|f| f.index_params.clone().map(|p| (f.name.clone(), p)))
            .collect();

        let mut built_fields: Vec<String> = Vec::new();
        let seg_dir = self.path.join(format!("seg_{}", meta.segment_id));
        for (field_name, params) in &indexed_field_params {
            // Skip HNSW during flush - it's expensive and likely to be
            // rebuilt during optimize's compaction when multiple segments
            // are merged. Only build cheap index types (Flat, etc.).
            if matches!(params, IndexParams::Hnsw(_) | IndexParams::HnswSparse(_)) {
                continue;
            }
            let index_path = seg_dir.join(format!("idx_{}", field_name));
            if !index_path.exists()
                && IndexBuilder::build_from_writing(&writing, field_name, params, &index_path)
                    .is_ok()
            {
                built_fields.push(field_name.clone());
            }
        }

        Ok(Some((meta, built_fields)))
    }

    // Load vector indexes that were built during the dump so query paths can use them.
    // Segment metadata is marked indexed only after load succeeds.
    fn load_flushed_vector_indexes(
        &self,
        version: &Version,
        seg: &PersistedSegment,
        segment_id: u32,
        indexed_fields: &[String],
    ) -> ZResult<Vec<String>> {
        let index_enable_mmap = effective_index_enable_mmap(&self.options)?;
        let mut loaded_indexed_fields = Vec::new();
        for field_name in indexed_fields {
            let Some(index_params) = version
                .schema
                .get_field(field_name)
                .and_then(|field| field.index_params.as_ref())
            else {
                continue;
            };
            let index_path = self
                .path
                .join(format!("seg_{}/idx_{}", segment_id, field_name));
            let idx =
                IndexBuilder::load_index(field_name, index_params, &index_path, index_enable_mmap)?;
            seg.add_vector_index(field_name.clone(), idx);
            loaded_indexed_fields.push(field_name.clone());
        }
        Ok(loaded_indexed_fields)
    }

    fn publish_flushed_segment(
        &self,
        new_writing: WritingSegment,
        flushed: FlushedSegment,
    ) -> ZResult<()> {
        // One critical section so a query sees the flushed docs in either the writing or the
        // persisted segment.
        {
            let mut persisted = self.persisted_segments.write();
            let mut writing = self.writing_segment.write();
            drop(std::mem::replace(&mut *writing, new_writing));
            persisted.push(Arc::new(flushed.seg));
        }
        cleanup_old_id_map_checkpoints(
            &self.path,
            &flushed.active_id_map_path,
            flushed.new_id_map_suffix,
        );
        if flushed.old_delete_suffix != flushed.new_delete_suffix {
            self.delete_store
                .write()
                .commit_snapshot(flushed.new_delete_suffix);
            let _ = std::fs::remove_file(
                self.path
                    .join(format!("delete_{}.bitmap", flushed.old_delete_suffix)),
            );
            let _ = fs::File::open(&self.path).and_then(|d| d.sync_all());
        }

        // The manifest now names the next WAL; the old one is never replayed again.
        if let (Some(wal), Some(next_wal)) = (self.wal.lock().as_mut(), flushed.next_wal) {
            let _ = fs::remove_file(wal.switch_to(next_wal));
            let _ = fs::File::open(&self.path).and_then(|d| d.sync_all());
        }

        Ok(())
    }

    pub(super) fn rotate_segment(&self) -> ZResult<()> {
        self.flush_writing_segment_locked()
    }

    pub(super) fn should_rotate_writing(
        &self,
        writing: &WritingSegment,
        max_docs_per_segment: u64,
    ) -> bool {
        if writing.doc_count() >= max_docs_per_segment {
            return true;
        }
        writing.forward_store.approx_bytes() >= self.options.max_buffer_size as usize
    }

    pub(super) fn build_thread_pool(
        concurrency: Option<usize>,
    ) -> ZResult<Option<rayon::ThreadPool>> {
        match concurrency {
            Some(n) if n > 0 => rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .map(Some)
                .map_err(|e| Status::internal(format!("failed to build rayon thread pool: {}", e))),
            _ => Ok(None),
        }
    }

    // Best-effort rollback: re-open invert handles so the collection remains usable for
    // subsequent writes.
    pub(super) fn reopen_writing_invert_indexes_best_effort(&self) {
        let _ = self
            .writing_segment
            .write()
            .reopen_invert_indexes(&self.path);
    }

    // Opens the writing segment that replaces the current one, with id `next_id` or the
    // current id.
    pub(super) fn prepare_replacement_writing_segment_locked(
        &self,
        next_id: Option<u32>,
        schema: &CollectionSchema,
    ) -> ZResult<WritingSegment> {
        let writing_id = {
            let mut writing = self.writing_segment.write();
            let id = next_id.unwrap_or(writing.id);
            // Avoid holding multiple writable fjall-backed invert indexes at
            // once (old writing segment + replacement writing segment).
            writing.drop_invert_indexes();
            id
        };

        match make_writing_segment(writing_id, schema, &self.path, false) {
            Ok(w) => Ok(w),
            Err(e) => {
                self.reopen_writing_invert_indexes_best_effort();
                Err(e)
            }
        }
    }
}

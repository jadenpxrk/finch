use std::sync::atomic::Ordering;
use std::sync::Arc;

use arrow::record_batch::RecordBatch;
use finch_types::ZResult;
use rayon::prelude::*;

use super::OptimizeEnv;
use crate::collection::{
    approx_doc_bytes, ensure_orphan_segment_dir_removed, make_writing_segment, Collection,
};
use crate::collection_files::effective_forward_file_format;
use crate::index::IndexBuilder;
use crate::segment::persisted::PersistedSegment;
use crate::segment::writing::WritingSegment;
use crate::segment::WrittenSegmentMeta;
use crate::version::PersistedSegmentVersion;

pub(super) struct CompactedGroup {
    pub(super) segs: Vec<Arc<PersistedSegment>>,
    pub(super) versions: Vec<PersistedSegmentVersion>,
    pub(super) next_seg_id: u32,
}

struct CompactedSegment {
    seg: Arc<PersistedSegment>,
    version: PersistedSegmentVersion,
}

struct WrittenSegments {
    metas: Vec<WrittenSegmentMeta>,
    next_seg_id: u32,
}

// Streams docs into writing segments, rolling over at the doc-count or byte budget.
struct CompactionWriter<'c, 'e> {
    collection: &'c Collection,
    env: &'e OptimizeEnv<'e>,
    writing: Option<WritingSegment>,
    next_seg_id: u32,
    written: Vec<WrittenSegmentMeta>,
}

impl CompactionWriter<'_, '_> {
    fn push_row(&mut self, doc_id: u64, batch: &RecordBatch, row: usize) -> ZResult<()> {
        let env = self.env;
        let path = &self.collection.path;
        if self.writing.is_none() {
            ensure_orphan_segment_dir_removed(path, self.next_seg_id, env.reserved_segment_ids)?;
            self.writing = Some(make_writing_segment(
                self.next_seg_id,
                env.schema,
                path,
                false,
            )?);
        }

        let doc = finch_storage::MmapForwardStore::extract_doc_from_batch(batch, row, doc_id);
        let doc_bytes = approx_doc_bytes(&doc).max(1);

        if let Some(cur) = self.writing.as_mut() {
            let cur_docs = cur.doc_count.load(Ordering::Relaxed);
            let cur_bytes = cur.forward_store.approx_bytes() as u64;
            let would_exceed_docs = cur_docs >= env.max_docs_per_segment;
            let would_exceed_bytes = env
                .target_segment_size
                .is_some_and(|target| cur_docs > 0 && cur_bytes.saturating_add(doc_bytes) > target);
            if would_exceed_docs || would_exceed_bytes {
                self.collection
                    .flush_compacted_segment(env, cur, &mut self.written)?;
                self.next_seg_id = self.next_seg_id.saturating_add(1);
                ensure_orphan_segment_dir_removed(
                    path,
                    self.next_seg_id,
                    env.reserved_segment_ids,
                )?;
                *cur = make_writing_segment(self.next_seg_id, env.schema, path, false)?;
            }
            cur.insert(doc_id, doc)?;
        }
        Ok(())
    }

    fn finish(mut self) -> ZResult<WrittenSegments> {
        if let Some(mut cur) = self.writing.take() {
            self.collection
                .flush_compacted_segment(self.env, &mut cur, &mut self.written)?;
            if cur.doc_count.load(Ordering::Relaxed) > 0 {
                self.next_seg_id = self.next_seg_id.saturating_add(1);
            }
        }
        Ok(WrittenSegments {
            metas: self.written,
            next_seg_id: self.next_seg_id,
        })
    }
}

impl Collection {
    // Compacts input segments into one or more output segments, optionally dropping deletes.
    pub(super) fn compact_group(
        &self,
        env: &OptimizeEnv<'_>,
        start_seg_id: u32,
        input_segs: &[Arc<PersistedSegment>],
        drop_deleted: bool,
    ) -> ZResult<CompactedGroup> {
        // Phase 1: stream through input segments and write new forward stores.
        let mut writer = CompactionWriter {
            collection: self,
            env,
            writing: None,
            next_seg_id: start_seg_id,
            written: Vec::new(),
        };
        for seg in input_segs {
            let forward = seg.forward_store.read();
            forward.scan_rows(|doc_id, batch, row| {
                if drop_deleted && env.delete_bitmap.contains(doc_id) {
                    return Ok(());
                }
                writer.push_row(doc_id, batch, row)
            })?;
        }
        let written = writer.finish()?;

        // Phase 2: open new persisted segments and build vector indexes.
        let built: Vec<ZResult<CompactedSegment>> = if written.metas.is_empty() {
            Vec::new()
        } else {
            env.pool.install(|| {
                written
                    .metas
                    .into_par_iter()
                    .map(|meta| self.open_compacted_segment(env, meta))
                    .collect()
            })
        };

        let mut segs: Vec<Arc<PersistedSegment>> = Vec::new();
        let mut versions: Vec<PersistedSegmentVersion> = Vec::new();
        for item in built {
            let out = item?;
            segs.push(out.seg);
            versions.push(out.version);
        }

        segs.sort_unstable_by_key(|s| s.id);
        versions.sort_unstable_by_key(|s| s.segment_id);
        Ok(CompactedGroup {
            segs,
            versions,
            next_seg_id: written.next_seg_id,
        })
    }

    fn flush_compacted_segment(
        &self,
        env: &OptimizeEnv<'_>,
        writing: &mut WritingSegment,
        written: &mut Vec<WrittenSegmentMeta>,
    ) -> ZResult<()> {
        if writing.doc_count.load(Ordering::Relaxed) == 0 {
            return Ok(());
        }
        let meta = writing.dump(&self.path, effective_forward_file_format(&self.options))?;

        // Build vector indexes directly from the in-memory writing segment before we
        // drop it. This avoids a second full scan over the persisted forward store
        // during compaction.
        let seg_dir = self.path.join(format!("seg_{}", meta.segment_id));
        for (field, params) in env.indexed_field_params {
            let field_name = &field.name;
            let index_path = seg_dir.join(format!("idx_{}", field_name));
            if !index_path.exists() {
                let _ = IndexBuilder::build_from_writing(writing, field_name, params, &index_path);
            }
        }

        written.push(meta);
        Ok(())
    }

    fn open_compacted_segment(
        &self,
        env: &OptimizeEnv<'_>,
        meta: WrittenSegmentMeta,
    ) -> ZResult<CompactedSegment> {
        let seg_id = meta.segment_id;
        let seg = PersistedSegment::open_with_forward_store_options(&meta, env.forward_opts)?;

        let mut built_fields: Vec<String> = Vec::new();
        for (field, params) in env.indexed_field_params {
            let field_name = &field.name;
            let index_path = self.path.join(format!("seg_{}/idx_{}", seg_id, field_name));

            // Prefer loading an index if it already exists (e.g. built from the
            // in-memory writing segment during compaction). Fall back to building from
            // persisted forward-store data for backward compatibility.
            if !index_path.exists() {
                IndexBuilder::build_from_persisted(&seg, field, params, &index_path)?;
            }
            let idx =
                IndexBuilder::load_index(field_name, params, &index_path, env.index_enable_mmap)?;
            seg.add_vector_index(field_name.clone(), idx);
            built_fields.push(field_name.clone());
        }
        built_fields.sort();
        built_fields.dedup();

        Ok(CompactedSegment {
            seg: Arc::new(seg),
            version: PersistedSegmentVersion {
                segment_id: seg_id,
                min_doc_id: meta.min_doc_id,
                max_doc_id: meta.max_doc_id,
                doc_count: meta.doc_count,
                indexed_vector_fields: built_fields,
                vector_index_generations: Default::default(),
            },
        })
    }
}

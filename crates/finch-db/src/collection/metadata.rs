use std::collections::HashMap;
use std::fs;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use finch_types::{
    CollectionOptions, CollectionSchema, CollectionStats, Status, VectorFieldStats, ZResult,
};

use roaring::RoaringTreemap;

use crate::segment::persisted::PersistedSegment;
use crate::version::Version;

use super::{
    bitmap_count_in_range_inclusive, count_deleted_docs_in_segment, is_contiguous_range, Collection,
};

// Live docs whose segment has `field` indexed, starting from `writing_live`.
fn indexed_live_count(
    version: &Version,
    persisted_live_counts: &[(u32, u64)],
    field: &str,
    writing_live: u64,
) -> u64 {
    let mut indexed_doc_count = writing_live;
    for s in &version.persisted_segments {
        if !s.indexed_vector_fields.iter().any(|name| name == field) {
            continue;
        }
        if let Some((_, live)) = persisted_live_counts
            .iter()
            .find(|(id, _)| *id == s.segment_id)
        {
            indexed_doc_count = indexed_doc_count.saturating_add(*live);
        }
    }
    indexed_doc_count
}

impl Collection {
    // Live doc counts per persisted segment (used to weight index completeness).
    fn persisted_live_counts(
        &self,
        version: &Version,
        delete_bitmap: &RoaringTreemap,
    ) -> ZResult<Vec<(u32, u64)>> {
        let seg_map: HashMap<u32, Arc<PersistedSegment>> = self
            .persisted_segments
            .read()
            .iter()
            .map(|s| (s.id, s.clone()))
            .collect();

        let mut persisted_live_counts: Vec<(u32, u64)> =
            Vec::with_capacity(version.persisted_segments.len());
        for meta in &version.persisted_segments {
            let del = if is_contiguous_range(meta.doc_count, meta.min_doc_id, meta.max_doc_id) {
                bitmap_count_in_range_inclusive(delete_bitmap, meta.min_doc_id, meta.max_doc_id)
            } else if let Some(seg) = seg_map.get(&meta.segment_id) {
                count_deleted_docs_in_segment(meta, seg, delete_bitmap)?
            } else {
                bitmap_count_in_range_inclusive(delete_bitmap, meta.min_doc_id, meta.max_doc_id)
            };
            let live = meta.doc_count.saturating_sub(del);
            persisted_live_counts.push((meta.segment_id, live));
        }
        Ok(persisted_live_counts)
    }

    fn writing_live_count(&self, delete_bitmap: &RoaringTreemap) -> u64 {
        let writing = self.writing_segment.read();
        let writing_doc_count = writing.doc_count();
        let writing_min = writing.min_doc_id.load(Ordering::Relaxed);
        let writing_max = writing.max_doc_id.load(Ordering::Relaxed);
        let writing_del = if writing_doc_count == 0 || writing_min == u64::MAX {
            0
        } else {
            bitmap_count_in_range_inclusive(delete_bitmap, writing_min, writing_max)
        };
        writing_doc_count.saturating_sub(writing_del)
    }

    pub fn stats(&self) -> ZResult<CollectionStats> {
        let version = self.cur_version();
        let delete_bitmap = self.delete_store.read().bitmap();

        let persisted_live_counts = self.persisted_live_counts(&version, delete_bitmap.as_ref())?;
        let writing_live = self.writing_live_count(delete_bitmap.as_ref());

        let live_total: u64 =
            persisted_live_counts.iter().map(|(_, c)| *c).sum::<u64>() + writing_live;

        let mut index_completeness: HashMap<String, f32> = HashMap::new();
        for f in version.schema.vector_fields() {
            if live_total == 0 {
                index_completeness.insert(f.name.clone(), 1.0);
                continue;
            }

            // Writing segment always maintains an in-memory flat store for vector fields.
            let indexed_doc_count =
                indexed_live_count(&version, &persisted_live_counts, &f.name, writing_live);
            index_completeness.insert(
                f.name.clone(),
                (indexed_doc_count as f32) / (live_total as f32),
            );
        }

        let vector_stats = version
            .schema
            .vector_fields()
            .map(|f| VectorFieldStats {
                field_name: f.name.clone(),
                doc_count: live_total,
                indexed: f.is_indexed(),
            })
            .collect();

        Ok(CollectionStats {
            doc_count: live_total,
            segment_count: version.persisted_segments.len() + 1,
            vector_field_stats: vector_stats,
            index_completeness,
        })
    }

    pub fn schema_info(&self) -> CollectionSchema {
        self.version_manager.current().schema.clone()
    }

    pub fn options(&self) -> CollectionOptions {
        self.options.clone()
    }

    pub fn path_string(&self) -> String {
        self.path.display().to_string()
    }

    pub fn destroy(self) -> ZResult<()> {
        let path = self.path.clone();
        // Drop self first to release all file handles (redb, lock file, etc.)
        drop(self);
        fs::remove_dir_all(&path).map_err(|e| Status::io_error(e.to_string()))
    }
}

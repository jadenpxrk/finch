//! Collection: the top-level database object

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use finch_types::{CollectionOptions, CollectionSchema, Doc, IndexParams, Status, Value, ZResult};
use parking_lot::{Mutex, RwLock};

use crate::delete_store::DeleteStore;
use crate::id_map::IdMap;
use crate::invert::InvertIndex;
use crate::row_locator::RowLocator;
use crate::segment::persisted::PersistedSegment;
use crate::segment::writing::WritingSegment;
use crate::version::{PersistedSegmentVersion, Version, VersionManager};
use crate::wal::{Wal, WalEntry};

mod column_ddl;
mod commit;
mod fetch;
mod index_ddl;
mod metadata;
mod multi_query;
mod opening;
mod optimization;
mod query_execution;
mod query_filter_only;
mod query_id;
mod query_int_ids;
mod query_materialization;
mod query_wrappers;
mod segment_lifecycle;
mod write;

// only rebuild (drop deleted docs) during optimize when the
// delete ratio is large enough.
const COMPACT_DELETE_RATIO_THRESHOLD: f64 = 0.3;

// Large monolithic HNSW graphs lose recall at ordinary ef values. Keep the public HNSW
// parameters unchanged, but optimize default dense-HNSW collections into searchable segments.
const HNSW_AUTO_MAX_DOCS_PER_SEGMENT: u64 = 175_000;
const MAX_QUERY_TOPK: usize = 1024;
const MAX_OUTPUT_FIELDS: usize = 1024;

fn bitmap_count_in_range_inclusive(bm: &roaring::RoaringTreemap, min: u64, max: u64) -> u64 {
    if min > max {
        return 0;
    }
    let hi = bm.rank(max);
    let lo = if min == 0 { 0 } else { bm.rank(min - 1) };
    hi.saturating_sub(lo)
}

fn is_contiguous_range(doc_count: u64, min_doc_id: u64, max_doc_id: u64) -> bool {
    if doc_count == 0 {
        return true;
    }
    if min_doc_id > max_doc_id {
        return false;
    }
    doc_count == max_doc_id.saturating_sub(min_doc_id).saturating_add(1)
}

fn count_deleted_docs_in_segment(
    meta: &PersistedSegmentVersion,
    seg: &PersistedSegment,
    delete_bitmap: &roaring::RoaringTreemap,
) -> ZResult<u64> {
    if delete_bitmap.is_empty() || meta.doc_count == 0 || meta.min_doc_id > meta.max_doc_id {
        return Ok(0);
    }

    if is_contiguous_range(meta.doc_count, meta.min_doc_id, meta.max_doc_id) {
        return Ok(bitmap_count_in_range_inclusive(
            delete_bitmap,
            meta.min_doc_id,
            meta.max_doc_id,
        ));
    }

    let forward = seg.forward_store.read();
    let mut deleted: u64 = 0;
    forward.scan_rows(|doc_id, _batch, _row| {
        if delete_bitmap.contains(doc_id) {
            deleted = deleted.saturating_add(1);
        }
        Ok(())
    })?;
    Ok(deleted)
}

// Writes every doc of `seg` into `idx`, with null markers for docs lacking the field.
fn fill_invert_index(idx: &InvertIndex, seg: &PersistedSegment, field_name: &str) -> ZResult<()> {
    let doc_ids = seg.forward_store.read().all_doc_ids();
    let docs = seg.fetch_docs(&doc_ids)?;
    for (doc_id, maybe_doc) in doc_ids.into_iter().zip(docs.into_iter()) {
        let Some(doc) = maybe_doc else {
            continue;
        };
        match doc.fields.get(field_name) {
            Some(val) if !val.is_null() => {
                idx.insert_nonnull_marker(doc_id)?;
                idx.insert(doc_id, val)?;
            }
            _ => idx.insert_null_marker(doc_id)?,
        }
    }
    Ok(())
}

fn ensure_orphan_segment_dir_removed(
    base_path: &Path,
    seg_id: u32,
    reserved_segment_ids: &HashSet<u32>,
) -> ZResult<()> {
    if reserved_segment_ids.contains(&seg_id) {
        return Err(Status::internal(format!(
            "refusing to remove reserved segment directory seg_{}",
            seg_id
        )));
    }

    let seg_path = base_path.join(format!("seg_{}", seg_id));
    if !seg_path.exists() {
        return Ok(());
    }

    std::fs::remove_dir_all(&seg_path).map_err(|e| {
        Status::io_error(format!(
            "failed to remove existing orphan segment directory {}: {}",
            seg_path.display(),
            e
        ))
    })?;
    let _ = fs::File::open(base_path).and_then(|d| d.sync_all());
    Ok(())
}

// ── Collection ────────────────────────────────────────────────────────────────

pub struct Collection {
    pub path: PathBuf,
    version_manager: VersionManager,
    writing_segment: RwLock<WritingSegment>,
    persisted_segments: RwLock<Vec<Arc<PersistedSegment>>>,
    id_map: Arc<IdMap>,
    delete_store: RwLock<DeleteStore>,
    write_lock: Mutex<()>,
    doc_id_allocator: AtomicU64,
    options: CollectionOptions,
    wal: Mutex<Option<Wal>>,
    /// Held open to maintain process-level lock on the collection dir
    _lock_file: fs::File,
}

impl Collection {
    fn cur_version(&self) -> Arc<Version> {
        self.version_manager.current()
    }

    fn row_locator(&self) -> RowLocator {
        let persisted_segments = self.persisted_segments.read();
        let writing_segment = self.writing_segment.read();
        RowLocator::new(
            persisted_segments.as_slice(),
            writing_segment.min_doc_id.load(Ordering::Relaxed),
            writing_segment.max_doc_id.load(Ordering::Relaxed),
        )
    }

    fn allocate_doc_id(&self) -> u64 {
        self.doc_id_allocator.fetch_add(1, Ordering::Relaxed)
    }

    fn check_not_readonly(&self) -> ZResult<()> {
        if self.options.read_only {
            Err(Status::permission_denied("collection is read-only"))
        } else {
            Ok(())
        }
    }

    fn wal_append(&self, entry: &WalEntry) -> ZResult<()> {
        let mut guard = self.wal.lock();
        let Some(wal) = guard.as_mut() else {
            return Ok(());
        };
        wal.append(entry)
    }
}

impl Drop for Collection {
    fn drop(&mut self) {
        // Flush Durability::None redb writes so data survives reopen.
        if !self.options.read_only {
            let _ = self.id_map.sync();
            let _ = self.writing_segment.write().sync_invert_indexes();
        }
    }
}

fn approx_doc_bytes(doc: &Doc) -> u64 {
    let mut bytes = std::mem::size_of::<u64>() as u64 + doc.pk.len() as u64;
    for value in doc.fields.values() {
        bytes = bytes.saturating_add(approx_value_bytes(value));
    }
    bytes
}

fn approx_value_bytes(v: &Value) -> u64 {
    use std::mem::size_of;
    match v {
        Value::Null => 0,
        Value::Bool(_) => size_of::<bool>() as u64,
        Value::I8(_) => size_of::<i8>() as u64,
        Value::I16(_) => size_of::<i16>() as u64,
        Value::I32(_) => size_of::<i32>() as u64,
        Value::I64(_) => size_of::<i64>() as u64,
        Value::U8(_) => size_of::<u8>() as u64,
        Value::U16(_) => size_of::<u16>() as u64,
        Value::U32(_) => size_of::<u32>() as u64,
        Value::U64(_) => size_of::<u64>() as u64,
        Value::F16(_) => size_of::<u16>() as u64,
        Value::F32(_) => size_of::<f32>() as u64,
        Value::F64(_) => size_of::<f64>() as u64,
        Value::String(s) => s.len() as u64,
        Value::Bytes(b) => b.len() as u64,
        Value::VecBool(v) => (v.len() * size_of::<bool>()) as u64,
        Value::VecI8(v) => (v.len() * size_of::<i8>()) as u64,
        Value::VecI16(v) => (v.len() * size_of::<i16>()) as u64,
        Value::VecI32(v) => (v.len() * size_of::<i32>()) as u64,
        Value::VecI64(v) => (v.len() * size_of::<i64>()) as u64,
        Value::VecU32(v) => (v.len() * size_of::<u32>()) as u64,
        Value::VecU64(v) => (v.len() * size_of::<u64>()) as u64,
        Value::VecF16(v) => (v.len() * size_of::<u16>()) as u64,
        Value::VecF32(v) => (v.len() * size_of::<f32>()) as u64,
        Value::VecF64(v) => (v.len() * size_of::<f64>()) as u64,
        Value::VecString(v) => v.iter().map(|s| s.len() as u64).sum(),
        Value::SparseF16 { indices, values } => {
            (indices.len() * size_of::<u32>() + values.len() * size_of::<u16>()) as u64
        }
        Value::SparseF32 { indices, values } => {
            (indices.len() * size_of::<u32>() + values.len() * size_of::<f32>()) as u64
        }
        Value::ArrayBinary(v) => v.iter().map(|b| b.len() as u64).sum(),
        Value::ArrayI32(v) => (v.len() * size_of::<i32>()) as u64,
        Value::ArrayI64(v) => (v.len() * size_of::<i64>()) as u64,
        Value::ArrayU32(v) => (v.len() * size_of::<u32>()) as u64,
        Value::ArrayU64(v) => (v.len() * size_of::<u64>()) as u64,
        Value::ArrayBool(v) => (v.len() * size_of::<bool>()) as u64,
        Value::ArrayF32(v) => (v.len() * size_of::<f32>()) as u64,
        Value::ArrayF64(v) => (v.len() * size_of::<f64>()) as u64,
        Value::ArrayString(v) => v.iter().map(|s| s.len() as u64).sum(),
    }
}

/// Create a writing segment, using inverted index support if the schema requires it
fn make_writing_segment(
    id: u32,
    schema: &CollectionSchema,
    base_path: &Path,
    read_only: bool,
) -> ZResult<WritingSegment> {
    if read_only {
        return WritingSegment::new(id, schema.clone());
    }

    let has_invert = schema.fields.iter().any(|f| {
        f.index_params
            .as_ref()
            .map(|p| matches!(p, IndexParams::Invert(_)))
            .unwrap_or(false)
    });
    if has_invert {
        WritingSegment::new_with_invert(id, schema.clone(), base_path)
    } else {
        WritingSegment::new(id, schema.clone())
    }
}

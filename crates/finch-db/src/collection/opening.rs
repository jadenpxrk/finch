use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};

use finch_storage::ForwardStoreOpenOptions;
use finch_types::{
    CollectionOptions, CollectionSchema, DataType, FileFormat, FlatIndexParams, IndexParams,
    MetricType, QuantizeType, Status, ZResult,
};

use crate::collection_files::{
    acquire_file_lock, effective_forward_file_format, effective_forward_store_open_options,
    effective_index_enable_mmap, forward_path_for_segment, id_map_path_for_suffix,
    remove_stale_read_only_copies, vector_index_path, wal_path_for_writing_segment,
};
use crate::delete_store::DeleteStore;
use crate::id_map::IdMap;
use crate::index::IndexBuilder;
use crate::invert::persisted_index_exists;
use crate::segment::persisted::PersistedSegment;
use crate::segment::writing::{InvertIndexMeta, WritingSegment, WrittenSegmentMeta};
use crate::version::{PersistedSegmentVersion, Version, VersionManager, FORMAT_VERSION};
use crate::wal::{Wal, WalEntry, WalOp};

use super::{make_writing_segment, Collection};

/// Validate collection path characters to reject unexpected inputs early.
///
/// Collection path regex:
/// `^(?:[a-zA-Z]:[/\\])?/?(?:[a-zA-Z0-9_.\\-]+[/\\])*[a-zA-Z0-9_.\\-]+$`
fn validate_collection_path(path: &Path) -> ZResult<()> {
    let raw = path
        .to_str()
        .ok_or_else(|| Status::invalid_argument("collection path must be valid UTF-8"))?;
    let normalized;
    let s = if raw.contains('\\') {
        normalized = raw.replace('\\', "/");
        normalized.as_str()
    } else {
        raw
    };
    if s.is_empty() {
        return Err(Status::invalid_argument("collection path cannot be empty"));
    }

    let trimmed = s.strip_prefix('/').unwrap_or(s);
    #[cfg(windows)]
    let trimmed = {
        let mut trimmed = trimmed;
        let bytes = trimmed.as_bytes();
        if bytes.len() >= 3
            && bytes[0].is_ascii_alphabetic()
            && bytes[1] == b':'
            && bytes[2] == b'/'
        {
            trimmed = &trimmed[3..];
        }
        trimmed
    };
    if trimmed.is_empty() {
        return Err(Status::invalid_argument(
            "collection path must contain at least one path component",
        ));
    }

    for part in trimmed.split('/') {
        if part.is_empty() {
            return Err(Status::invalid_argument(
                "collection path cannot contain empty components",
            ));
        }
        if !part
            .as_bytes()
            .iter()
            .copied()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-'))
        {
            return Err(Status::invalid_argument(format!(
                "collection path contains invalid characters: {:?}",
                path
            )));
        }
    }

    Ok(())
}

// Without a storage override, the `enable_mmap` chosen at creation time applies.
fn with_manifest_mmap(mut options: CollectionOptions, version: &Version) -> CollectionOptions {
    if options.index_storage.is_none() && options.forward_storage.is_none() {
        options.enable_mmap = version.enable_mmap;
    }
    options
}

struct OpenedIdMap {
    id_map: Arc<IdMap>,
    path: PathBuf,
    missing: bool,
}

impl OpenedIdMap {
    fn open(path: &Path, version: &Version, read_only: bool) -> ZResult<Self> {
        let id_map_path = id_map_path_for_suffix(path, version.id_map_suffix);
        // Suffix 0 names no checkpoint: before the first flush the WAL holds the whole map.
        let missing = if version.id_map_suffix == 0 {
            !id_map_path.exists()
        } else {
            !IdMap::has_checkpoint(&id_map_path)
        };
        if missing && read_only {
            return Err(Status::io_error(format!(
                "id_map checkpoint missing at {:?}",
                id_map_path
            )));
        }
        let id_map = Arc::new(if read_only {
            IdMap::open_read_only(&id_map_path)?
        } else {
            IdMap::open(&id_map_path)?
        });
        Ok(Self {
            id_map,
            path: id_map_path,
            missing,
        })
    }
}

struct SegmentOpenContext<'a> {
    path: &'a Path,
    schema: &'a CollectionSchema,
    forward_format: FileFormat,
    forward_opts: ForwardStoreOpenOptions,
    index_enable_mmap: bool,
    read_only: bool,
}

struct LoadedSegments {
    segments: Vec<Arc<PersistedSegment>>,
    max_doc_id: Option<u64>,
}

fn load_persisted_segments(
    ctx: &SegmentOpenContext<'_>,
    version: &Version,
) -> ZResult<LoadedSegments> {
    let mut segments = Vec::new();
    let mut max_doc_id: Option<u64> = None;
    for seg_version in &version.persisted_segments {
        let seg_path = ctx.path.join(format!("seg_{}", seg_version.segment_id));
        let seg = open_persisted_segment(ctx, seg_version, &seg_path)?;
        max_doc_id = Some(max_doc_id.map_or(seg_version.max_doc_id, |cur| {
            cur.max(seg_version.max_doc_id)
        }));
        load_segment_vector_indexes(ctx, seg_version, &seg_path, &seg)?;
        segments.push(Arc::new(seg));
    }
    Ok(LoadedSegments {
        segments,
        max_doc_id,
    })
}

fn open_persisted_segment(
    ctx: &SegmentOpenContext<'_>,
    seg_version: &PersistedSegmentVersion,
    seg_path: &Path,
) -> ZResult<PersistedSegment> {
    let forward_path = forward_path_for_segment(seg_path, ctx.forward_format);
    if !forward_path.exists() {
        return Err(Status::io_error(format!(
            "segment {} forward store missing at {:?}",
            seg_version.segment_id, forward_path
        )));
    }

    let meta = WrittenSegmentMeta {
        segment_id: seg_version.segment_id,
        min_doc_id: seg_version.min_doc_id,
        max_doc_id: seg_version.max_doc_id,
        doc_count: seg_version.doc_count,
        forward_path,
        invert_paths: segment_invert_paths(ctx.schema, seg_version.segment_id, seg_path)?,
    };
    if ctx.read_only {
        PersistedSegment::open_forward_only_with_forward_store_options(&meta, ctx.forward_opts)
    } else {
        PersistedSegment::open_with_forward_store_options(&meta, ctx.forward_opts)
    }
}

// Persisted segments don't currently record invert paths in the manifest.
// The on-disk layout is deterministic: `seg_{id}/{field}_invert`.
fn segment_invert_paths(
    schema: &CollectionSchema,
    segment_id: u32,
    seg_path: &Path,
) -> ZResult<HashMap<String, InvertIndexMeta>> {
    let mut invert_paths: HashMap<String, InvertIndexMeta> = HashMap::new();
    for field in schema.inverted_index_fields() {
        let Some(IndexParams::Invert(params)) = field.index_params.as_ref() else {
            continue;
        };
        let p = seg_path.join(format!("{}_invert", field.name));
        if !persisted_index_exists(&p) {
            return Err(Status::io_error(format!(
                "segment {} invert index missing for field {} at {:?}",
                segment_id, field.name, p
            )));
        }
        invert_paths.insert(
            field.name.clone(),
            InvertIndexMeta {
                path: p,
                data_type: field.data_type,
                params: params.clone(),
            },
        );
    }
    Ok(invert_paths)
}

fn load_segment_vector_indexes(
    ctx: &SegmentOpenContext<'_>,
    seg_version: &PersistedSegmentVersion,
    seg_path: &Path,
    seg: &PersistedSegment,
) -> ZResult<()> {
    for field_name in &seg_version.indexed_vector_fields {
        let Some(index_params) = ctx
            .schema
            .get_field(field_name)
            .and_then(|field| field.index_params.as_ref())
        else {
            continue;
        };
        let index_path = vector_index_path(
            seg_path,
            field_name,
            seg_version.vector_index_generation(field_name),
        );
        if !index_path.exists() {
            return Err(Status::io_error(format!(
                "segment {} vector index missing for {} at {:?}",
                seg_version.segment_id, field_name, index_path
            )));
        }
        let idx =
            IndexBuilder::load_index(field_name, index_params, &index_path, ctx.index_enable_mmap)?;
        seg.add_vector_index(field_name.clone(), idx);
    }
    Ok(())
}

fn ensure_no_pending_wal(path: &Path, writing_seg_id: u32) -> ZResult<()> {
    let wal_path = wal_path_for_writing_segment(path, writing_seg_id);
    let wal_entries = Wal::replay(&wal_path)?;
    if !wal_entries.is_empty() {
        return Err(Status::io_error(format!(
            "read-only open has pending WAL recovery from {:?}; open read-write once to recover",
            wal_path
        )));
    }
    Ok(())
}

fn open_wal(path: &Path, writing_seg_id: u32) -> ZResult<Wal> {
    let cfg = crate::config::global_config();
    Wal::open(
        wal_path_for_writing_segment(path, writing_seg_id),
        cfg.wal_flush_every_docs,
        cfg.wal_fsync_every_docs,
    )
}

// If WAL replay applied deletes, flush the updated bitmap and sync the
// manifest's delete_suffix so they never diverge after a crash.
fn persist_replayed_deletes(
    path: &Path,
    delete_store: &mut DeleteStore,
    version_manager: &VersionManager,
) -> ZResult<()> {
    // Fail the open: without the snapshot the replayed tombstones live only in memory and in the WAL.
    let old_suffix = delete_store.path_suffix;
    let new_suffix = delete_store.snapshot()?;
    let mut updated = (*version_manager.current()).clone();
    updated.delete_suffix = new_suffix;
    // Fail the open: a stale delete_suffix loses the replayed tombstones once the next flush drops the WAL.
    version_manager.flush(&updated)?;
    delete_store.commit_snapshot(new_suffix);
    if old_suffix != new_suffix {
        let _ = std::fs::remove_file(path.join(format!("delete_{}.bitmap", old_suffix)));
        let _ = fs::File::open(path).and_then(|d| d.sync_all());
    }
    Ok(())
}

// Read-write crash recovery: replays the active WAL into the writing segment, id_map and
// delete bitmap.
struct WalRecovery<'a> {
    path: &'a Path,
    writing_seg_id: u32,
    id_map: &'a OpenedIdMap,
    has_persisted_segments: bool,
    max_doc_id: Option<u64>,
    writing_segment: &'a mut WritingSegment,
    delete_store: &'a mut DeleteStore,
    next_doc_id: u64,
}

impl WalRecovery<'_> {
    fn run(&mut self, version_manager: &VersionManager) -> ZResult<()> {
        let wal_path = wal_path_for_writing_segment(self.path, self.writing_seg_id);

        // Replay WAL for crash recovery in read-write mode.
        let wal_entries = Wal::replay(&wal_path)?;
        if self.id_map.missing && wal_entries.is_empty() && self.has_persisted_segments {
            return Err(Status::io_error(format!(
                "id_map checkpoint missing at {:?} and WAL is empty; cannot recover",
                self.id_map.path
            )));
        }
        if !wal_entries.is_empty() {
            tracing::info!("replaying {} WAL entries", wal_entries.len());
        }
        let mut had_wal_deletes = false;
        for entry in wal_entries {
            if self.apply(entry)? {
                had_wal_deletes = true;
            }
        }

        if had_wal_deletes {
            persist_replayed_deletes(self.path, self.delete_store, version_manager)?;
        }
        Ok(())
    }

    // Returns whether the entry tombstoned a doc_id.
    fn apply(&mut self, entry: WalEntry) -> ZResult<bool> {
        match entry.op {
            WalOp::Insert | WalOp::Upsert | WalOp::Update => self.apply_write(entry),
            WalOp::Delete => {
                // Re-apply delete: the on-disk bitmap may be from before the
                // crash, so replay any deletes that arrived after the last
                // bitmap snapshot.
                self.delete_store.mark_deleted(entry.doc_id);
                self.id_map.id_map.delete(&entry.pk)?;
                Ok(true)
            }
        }
    }

    fn apply_write(&mut self, entry: WalEntry) -> ZResult<bool> {
        // A record a persisted segment already holds must not enter the writing segment twice.
        if self
            .max_doc_id
            .is_some_and(|max_persisted| entry.doc_id <= max_persisted)
        {
            return Ok(false);
        }
        if let Some(mut doc) = entry.doc {
            doc.doc_id = entry.doc_id;
            self.writing_segment
                .insert(entry.doc_id, doc)
                .map_err(|e| {
                    Status::new(
                        e.code,
                        format!("WAL replay of pk '{}' failed: {}", entry.pk, e.message),
                    )
                })?;
            // Rebuild id_map from WAL in case the on-disk mapping is missing
            // or the crash happened after WAL append but before id_map insert.
            self.id_map.id_map.insert(&entry.pk, entry.doc_id)?;
            self.next_doc_id = self.next_doc_id.max(entry.doc_id + 1);
        }
        let Some(prev) = entry.prev_doc_id else {
            return Ok(false);
        };
        self.delete_store.mark_deleted(prev);
        Ok(true)
    }
}

impl Collection {
    /// Create a new collection at `path` (must not exist)
    pub fn create_and_open(
        path: &Path,
        schema: CollectionSchema,
        options: CollectionOptions,
    ) -> ZResult<Arc<Self>> {
        crate::config::ensure_rayon_initialized();
        validate_collection_path(path)?;
        if options.read_only {
            return Err(Status::invalid_argument(
                "cannot create a collection in read-only mode",
            ));
        }
        schema.validate()?;
        let schema = assign_default_index_params(schema);

        if path.exists() {
            return Err(Status::already_exists(format!(
                "collection already exists at {:?}",
                path
            )));
        }
        fs::create_dir_all(path).map_err(|e| Status::io_error(e.to_string()))?;
        // Best-effort durability: ensure the newly-created collection
        // directory entry is flushed to its parent directory.
        if let Some(parent) = path.parent() {
            let _ = fs::File::open(parent).and_then(|d| d.sync_all());
        }

        let lock_file = acquire_file_lock(path, false)?;

        let id_map_path = path.join("id_map");
        let id_map = Arc::new(IdMap::open(&id_map_path)?);

        let delete_store = DeleteStore::new(path, 0);
        // Flush an initial empty delete bitmap to disk (suffix 0) so the file
        // always exists before the manifest is written.
        delete_store.snapshot_current()?;
        let version_manager = VersionManager::create(path, schema.clone(), options.enable_mmap)?;

        let writing_segment = make_writing_segment(0, &schema, path, false)?;
        let wal = open_wal(path, 0)?;

        Ok(Arc::new(Collection {
            path: path.to_path_buf(),
            version_manager,
            writing_segment: RwLock::new(writing_segment),
            persisted_segments: RwLock::new(Vec::new()),
            id_map,
            delete_store: RwLock::new(delete_store),
            write_lock: Mutex::new(()),
            doc_id_allocator: AtomicU64::new(0),
            options,
            wal: Mutex::new(Some(wal)),
            _lock_file: lock_file,
        }))
    }

    /// Open an existing collection at `path`
    pub fn open(path: &Path, options: CollectionOptions) -> ZResult<Arc<Self>> {
        crate::config::ensure_rayon_initialized();
        validate_collection_path(path)?;
        if !path.exists() {
            return Err(Status::not_found(format!(
                "collection not found at {:?}",
                path
            )));
        }

        // Allow switching `read_only` at open time, but `enable_mmap`
        // is determined at collection creation time and persisted in the manifest.
        let lock_file = acquire_file_lock(path, options.read_only)?;

        let version_manager = VersionManager::load(path)?;
        let version = version_manager.current();
        if version.format_version != FORMAT_VERSION {
            return Err(Status::invalid_argument(format!(
                "collection at {:?} has index format {}; this build reads only format {}",
                path, version.format_version, FORMAT_VERSION
            )));
        }
        // Only a read-write open may delete: it holds the exclusive lock.
        if !options.read_only {
            remove_stale_read_only_copies(path)?;
        }
        let options = with_manifest_mmap(options, &version);
        let segment_ctx = SegmentOpenContext {
            path,
            schema: &version.schema,
            forward_format: effective_forward_file_format(&options),
            forward_opts: effective_forward_store_open_options(&options),
            index_enable_mmap: effective_index_enable_mmap(&options)?,
            read_only: options.read_only,
        };
        let id_map = OpenedIdMap::open(path, &version, options.read_only)?;
        let mut delete_store = DeleteStore::load(path, version.delete_suffix)?;
        let loaded = load_persisted_segments(&segment_ctx, &version)?;
        let writing_seg_id = version
            .writing_segment_id
            .unwrap_or(version.next_segment_id);

        // A read-only open replays no WAL, so its writing segment stays empty.
        if options.read_only {
            ensure_no_pending_wal(path, writing_seg_id)?;
        }
        let mut writing_segment =
            make_writing_segment(writing_seg_id, &version.schema, path, options.read_only)?;
        let mut next_doc_id = loaded.max_doc_id.map(|v| v.saturating_add(1)).unwrap_or(0);

        if !options.read_only {
            let mut recovery = WalRecovery {
                path,
                writing_seg_id,
                id_map: &id_map,
                has_persisted_segments: !version.persisted_segments.is_empty(),
                max_doc_id: loaded.max_doc_id,
                writing_segment: &mut writing_segment,
                delete_store: &mut delete_store,
                next_doc_id,
            };
            recovery.run(&version_manager)?;
            next_doc_id = recovery.next_doc_id;
        }
        // Compaction drops deleted docs but keeps their tombstones, so a reused doc_id would start out deleted.
        if let Some(max_tombstone) = delete_store.bitmap().max() {
            next_doc_id = next_doc_id.max(max_tombstone.saturating_add(1));
        }

        let wal = if options.read_only {
            None
        } else {
            Some(open_wal(path, writing_seg_id)?)
        };

        Ok(Arc::new(Collection {
            path: path.to_path_buf(),
            version_manager,
            writing_segment: RwLock::new(writing_segment),
            persisted_segments: RwLock::new(loaded.segments),
            id_map: id_map.id_map,
            delete_store: RwLock::new(delete_store),
            write_lock: Mutex::new(()),
            doc_id_allocator: AtomicU64::new(next_doc_id),
            options,
            wal: Mutex::new(wal),
            _lock_file: lock_file,
        }))
    }
}

/// Assign default Flat/FlatSparse index params to vector fields that have none.
fn assign_default_index_params(mut schema: CollectionSchema) -> CollectionSchema {
    let default_flat = IndexParams::Flat(FlatIndexParams {
        metric: MetricType::InnerProduct,
        quantize: QuantizeType::Undefined,
        column_major: false,
    });
    let default_flat_binary = IndexParams::Flat(FlatIndexParams {
        metric: MetricType::Hamming,
        quantize: QuantizeType::Undefined,
        column_major: false,
    });
    let default_flat_sparse = IndexParams::FlatSparse(FlatIndexParams {
        metric: MetricType::InnerProduct,
        quantize: QuantizeType::Undefined,
        column_major: false,
    });
    for field in &mut schema.fields {
        if field.index_params.is_none() {
            if field.data_type.is_sparse() {
                field.index_params = Some(default_flat_sparse.clone());
            } else if field.data_type.is_vector() {
                if matches!(
                    field.data_type,
                    DataType::VectorBinary32 | DataType::VectorBinary64
                ) {
                    field.index_params = Some(default_flat_binary.clone());
                } else {
                    field.index_params = Some(default_flat.clone());
                }
            }
        }
    }
    schema
}

#[cfg(test)]
mod tests {
    use finch_types::{Doc, FieldSchema, InvertIndexParams};

    use super::*;

    #[test]
    fn replay_fails_open_without_mapping_a_doc_the_segment_rejects() {
        let path = std::env::temp_dir().join(format!(
            "finch_replay_rejected_insert_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&path);
        let schema = CollectionSchema::new("test").with_field(
            FieldSchema::new("label", DataType::String)
                .with_index(IndexParams::Invert(InvertIndexParams::default())),
        );
        drop(Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap());

        // An int in the string-indexed field cannot be inserted into the writing segment.
        let mut wal = Wal::open(wal_path_for_writing_segment(&path, 0), 1, 1).unwrap();
        wal.append(&WalEntry {
            op: WalOp::Insert,
            doc_id: 0,
            prev_doc_id: None,
            pk: "bad".to_string(),
            doc: Some(Doc::new("bad").set("label", 7i32)),
        })
        .unwrap();
        drop(wal);

        assert!(Collection::open(&path, CollectionOptions::default()).is_err());
        let id_map = IdMap::open(&path.join("id_map")).unwrap();
        assert_eq!(id_map.get("bad").unwrap(), None);

        drop(id_map);
        fs::remove_dir_all(&path).ok();
    }

    #[test]
    fn replaying_a_wal_record_the_index_rejects_fails_open_naming_the_field() {
        let path = std::env::temp_dir().join(format!(
            "finch_replay_nul_indexed_string_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&path);
        let schema = CollectionSchema::new("test")
            .with_field(
                FieldSchema::new("emb", DataType::VectorFp32)
                    .nullable()
                    .with_dimension(2),
            )
            .with_field(
                FieldSchema::new("label", DataType::String)
                    .nullable()
                    .with_index(IndexParams::Invert(InvertIndexParams::default())),
            );
        drop(Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap());

        // Insert validation rejects a NUL string, so only a hand-written record reaches replay.
        let mut wal = Wal::open(wal_path_for_writing_segment(&path, 0), 1, 1).unwrap();
        wal.append(&WalEntry {
            op: WalOp::Insert,
            doc_id: 0,
            prev_doc_id: None,
            pk: "bad".to_string(),
            doc: Some(
                Doc::new("bad")
                    .set("emb", vec![1.0f32, 0.0])
                    .set("label", "a\0b"),
            ),
        })
        .unwrap();
        drop(wal);

        let err = match Collection::open(&path, CollectionOptions::default()) {
            Ok(_) => panic!("open must fail on a record the index cannot store"),
            Err(e) => e,
        };
        assert!(err.message.contains("label"), "{}", err.message);
        fs::remove_dir_all(&path).ok();
    }

    #[test]
    fn open_rejects_another_format_version() {
        let path =
            std::env::temp_dir().join(format!("finch_other_format_version_{}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        let schema = CollectionSchema::new("test")
            .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(2));
        drop(Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap());

        let manager = VersionManager::load(&path).unwrap();
        let mut other = (*manager.current()).clone();
        other.format_version = FORMAT_VERSION + 1;
        manager.flush(&other).unwrap();
        drop(manager);

        let err = match Collection::open(&path, CollectionOptions::default()) {
            Ok(_) => panic!("open must reject another format version"),
            Err(e) => e,
        };
        assert!(err.message.contains("index format"), "{}", err.message);

        fs::remove_dir_all(&path).ok();
    }
}

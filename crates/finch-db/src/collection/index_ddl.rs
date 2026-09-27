use std::path::{Path, PathBuf};
use std::sync::Arc;

use finch_types::{
    CreateIndexOptions, DataType, FieldSchema, IndexParams, InvertIndexParams, Status, ZResult,
};
use rayon::prelude::*;

use crate::collection_files::{effective_index_enable_mmap, vector_index_path};
use crate::index::IndexBuilder;
use crate::invert::InvertIndex;
use crate::segment::persisted::{PersistedSegment, VectorIndex};
use crate::version::Version;

use super::{fill_invert_index, Collection};

type BuiltInvertIndex = (Arc<PersistedSegment>, PathBuf, Arc<InvertIndex>);
type BuiltVectorIndex = (Arc<PersistedSegment>, PathBuf, VectorIndex);

enum PreparedIndexHandles {
    Invert(Vec<BuiltInvertIndex>),
    Vector(Vec<BuiltVectorIndex>),
}

impl PreparedIndexHandles {
    fn publish(self, field: &str) {
        match self {
            PreparedIndexHandles::Invert(entries) => {
                for (seg, _, idx) in entries {
                    seg.invert_indexes.write().insert(field.to_string(), idx);
                }
            }
            PreparedIndexHandles::Vector(entries) => {
                for (seg, _, idx) in entries {
                    seg.add_vector_index(field.to_string(), idx);
                }
            }
        }
    }
}

pub(super) struct InvertBuildTarget<'a> {
    pub(super) field: &'a str,
    pub(super) data_type: DataType,
    pub(super) params: &'a InvertIndexParams,
}

struct VectorBuildTarget<'a> {
    field: &'a FieldSchema,
    params: &'a IndexParams,
    index_enable_mmap: bool,
}

// Rebuilds one segment's invert index from scratch and reopens it read-only.
pub(super) fn build_segment_invert_index(
    seg: &Arc<PersistedSegment>,
    idx_path: &Path,
    target: &InvertBuildTarget<'_>,
) -> ZResult<BuiltInvertIndex> {
    // Rebuild from scratch (create_index is only called when needed
    // or when rebuild=true).
    let _ = std::fs::remove_dir_all(idx_path);
    let idx = InvertIndex::open(
        idx_path,
        target.field.to_string(),
        target.data_type,
        target.params.clone(),
    )?;
    fill_invert_index(&idx, seg, target.field)?;

    // Flush Durability::None writes before reopening read-only.
    idx.sync()?;
    // Drop the writer handle before reopening
    // (redb file lock is per-Database instance).
    drop(idx);

    // Swap a read-only handle into the segment.
    let ro = InvertIndex::open_read_only(
        idx_path,
        target.field.to_string(),
        target.data_type,
        target.params.clone(),
    )?;
    Ok((seg.clone(), idx_path.to_path_buf(), Arc::new(ro)))
}

// The committed version with `field` indexed by `params`; vector indexes move to a new generation.
fn index_schema_version(
    version: &Version,
    field: &str,
    params: IndexParams,
    is_invert: bool,
) -> Version {
    let mut new_schema = version.schema.clone();
    if let Some(f) = new_schema.get_field_mut(field) {
        f.index_params = Some(params);
    }
    let mut new_version = version.clone();
    new_version.schema = new_schema;
    if !is_invert {
        for seg in &mut new_version.persisted_segments {
            if !seg.indexed_vector_fields.contains(&field.to_string()) {
                seg.indexed_vector_fields.push(field.to_string());
            }
            *seg.vector_index_generations
                .entry(field.to_string())
                .or_insert(0) += 1;
        }
    }
    new_version
}

// Each segment's `field` index directory as recorded in `version`.
fn vector_index_paths(
    collection_path: &Path,
    version: &Version,
    field: &str,
) -> Vec<(u32, PathBuf)> {
    version
        .persisted_segments
        .iter()
        .map(|seg| {
            let seg_dir = collection_path.join(format!("seg_{}", seg.segment_id));
            let path = vector_index_path(&seg_dir, field, seg.vector_index_generation(field));
            (seg.segment_id, path)
        })
        .collect()
}

fn remove_index_dirs(paths: &[(u32, PathBuf)]) {
    for (_, path) in paths {
        let _ = std::fs::remove_dir_all(path);
    }
}

fn all_segments_indexed(version: &Version, field: &str) -> bool {
    version.persisted_segments.iter().all(|segment| {
        segment
            .indexed_vector_fields
            .iter()
            .any(|indexed_field| indexed_field == field)
    })
}

fn build_segment_vector_index(
    seg: &Arc<PersistedSegment>,
    index_path: &Path,
    target: &VectorBuildTarget<'_>,
) -> ZResult<BuiltVectorIndex> {
    let field = target.field;
    // A build that crashed before its manifest commit may have left files at this generation.
    let _ = std::fs::remove_dir_all(index_path);
    IndexBuilder::build_from_persisted(seg, field, target.params, index_path)?;
    let idx = IndexBuilder::load_index(
        &field.name,
        target.params,
        index_path,
        target.index_enable_mmap,
    )?;
    Ok((seg.clone(), index_path.to_path_buf(), idx))
}

fn collect_built<T>(built: Vec<ZResult<T>>) -> ZResult<Vec<T>> {
    let mut prepared = Vec::with_capacity(built.len());
    for r in built {
        prepared.push(r?);
    }
    Ok(prepared)
}

impl Collection {
    fn build_invert_indexes(
        &self,
        segs: &[Arc<PersistedSegment>],
        target: &InvertBuildTarget<'_>,
        pool: &rayon::ThreadPool,
    ) -> ZResult<Vec<BuiltInvertIndex>> {
        let jobs: Vec<(Arc<PersistedSegment>, PathBuf)> = segs
            .iter()
            .map(|seg| {
                let seg_path = self.path.join(format!("seg_{}", seg.id));
                let idx_path = seg_path.join(format!("{}_invert", target.field));
                (seg.clone(), idx_path)
            })
            .collect();

        let built: Vec<ZResult<BuiltInvertIndex>> = pool.install(|| {
            jobs.par_iter()
                .map(|(seg, idx_path)| build_segment_invert_index(seg, idx_path, target))
                .collect()
        });
        collect_built(built)
    }

    fn build_vector_indexes(
        &self,
        segs: &[Arc<PersistedSegment>],
        index_paths: &[(u32, PathBuf)],
        target: &VectorBuildTarget<'_>,
        pool: &rayon::ThreadPool,
    ) -> ZResult<Vec<BuiltVectorIndex>> {
        let jobs = segs
            .iter()
            .map(|seg| {
                index_paths
                    .iter()
                    .find(|(id, _)| *id == seg.id)
                    .map(|(_, path)| (seg.clone(), path.clone()))
                    .ok_or_else(|| Status::internal("index build: missing segment metadata"))
            })
            .collect::<ZResult<Vec<_>>>()?;
        let build_results: Vec<ZResult<BuiltVectorIndex>> = pool.install(|| {
            jobs.par_iter()
                .map(|(seg, index_path)| build_segment_vector_index(seg, index_path, target))
                .collect()
        });
        collect_built(build_results)
    }

    // `abort` runs only when the manifest commit fails.
    fn commit_index_schema_update(
        &self,
        new_version: &Version,
        is_invert: bool,
        abort: impl FnOnce(),
    ) -> ZResult<()> {
        if is_invert {
            let new_writing =
                self.prepare_replacement_writing_segment_locked(None, &new_version.schema)?;
            return self.commit_manifest_with_prepared(
                new_version,
                new_writing,
                |new_writing| {
                    *self.writing_segment.write() = new_writing;
                    Ok(())
                },
                |new_writing| {
                    drop(new_writing);
                    self.reopen_writing_invert_indexes_best_effort();
                    abort();
                },
            );
        }

        self.commit_manifest_with_prepared(
            new_version,
            (),
            |()| {
                self.writing_segment
                    .write()
                    .apply_schema_update(new_version.schema.clone())
            },
            |()| abort(),
        )
    }

    /// Internal index-build helper; caller must already hold `write_lock`.
    pub(super) fn create_index_locked(
        &self,
        field: &str,
        params: IndexParams,
        concurrency: Option<usize>,
    ) -> ZResult<()> {
        let is_invert = matches!(params, IndexParams::Invert(_));
        let index_enable_mmap = effective_index_enable_mmap(&self.options)?;
        self.flush_writing_segment_locked()?;

        let version = self.cur_version();
        let new_version = index_schema_version(&version, field, params.clone(), is_invert);
        let (old_paths, new_paths) = if is_invert {
            (Vec::new(), Vec::new())
        } else {
            (
                vector_index_paths(&self.path, &version, field),
                vector_index_paths(&self.path, &new_version, field),
            )
        };
        let segs = self.persisted_segments.read().clone();
        let maybe_pool = Self::build_thread_pool(concurrency)?;
        let pool = maybe_pool
            .as_ref()
            .unwrap_or_else(|| crate::config::optimize_pool());

        let field_schema = version
            .schema
            .get_field(field)
            .ok_or_else(|| Status::not_found(format!("column {} not found", field)))?;
        let prepared_indexes = match &params {
            IndexParams::Invert(inv_params) => {
                let target = InvertBuildTarget {
                    field,
                    data_type: field_schema.data_type,
                    params: inv_params,
                };
                PreparedIndexHandles::Invert(self.build_invert_indexes(&segs, &target, pool)?)
            }
            _ => {
                let target = VectorBuildTarget {
                    field: field_schema,
                    params: &params,
                    index_enable_mmap,
                };
                match self.build_vector_indexes(&segs, &new_paths, &target, pool) {
                    Ok(built) => PreparedIndexHandles::Vector(built),
                    Err(e) => {
                        remove_index_dirs(&new_paths);
                        return Err(e);
                    }
                }
            }
        };
        drop(segs);

        // Ensure the active writing segment uses the updated schema.
        // For vector index-param changes, avoid re-opening redb-backed invert
        // indexes (can conflict with existing handles) and only refresh the
        // in-memory vector stores.
        self.commit_index_schema_update(&new_version, is_invert, || remove_index_dirs(&new_paths))?;
        prepared_indexes.publish(field);
        remove_index_dirs(&old_paths);
        Ok(())
    }

    /// Build a vector index for an existing field (across all persisted segments)
    pub fn create_index(
        &self,
        field: &str,
        params: IndexParams,
        options: CreateIndexOptions,
    ) -> ZResult<()> {
        self.check_not_readonly()?;
        let _guard = self.write_lock.lock();
        let version = self.cur_version();

        let existing_field = version
            .schema
            .get_field(field)
            .ok_or_else(|| Status::not_found(format!("column {} not found in schema", field)))?;

        // Skip rebuild only when the schema and every persisted segment agree
        // that the index exists. Bulk flushes intentionally defer expensive
        // HNSW construction, so schema configuration alone is not sufficient.
        let index_is_complete = existing_field.index_params.as_ref().is_some_and(|params| {
            matches!(params, IndexParams::Invert(_)) || all_segments_indexed(&version, field)
        });
        if !options.rebuild && index_is_complete {
            return Ok(());
        }

        self.create_index_locked(field, params, options.concurrency)
    }

    /// Remove the vector index for a field.
    pub fn drop_index(&self, field: &str) -> ZResult<()> {
        self.check_not_readonly()?;
        let _guard = self.write_lock.lock();
        let version = self.cur_version();

        let existing_field = version
            .schema
            .get_field(field)
            .ok_or_else(|| Status::not_found(format!("column {} not found in schema", field)))?;

        if existing_field.index_params.is_none() {
            return Ok(());
        }

        let is_invert = matches!(
            existing_field.index_params.as_ref(),
            Some(IndexParams::Invert(_))
        );

        // Flush writing segment so all data is in persisted segments.
        self.flush_writing_segment_locked()?;
        // IMPORTANT: refresh version after flush so we don't overwrite the
        // manifest with stale persisted-segment metadata.
        let version = self.cur_version();

        // Clear index_params from schema and remove field from indexed lists.
        let mut new_schema = version.schema.clone();
        if let Some(f) = new_schema.get_field_mut(field) {
            f.index_params = None;
        }
        let mut new_version = (*version).clone();
        new_version.schema = new_schema;
        let vector_paths = vector_index_paths(&self.path, &version, field);
        if !is_invert {
            for seg in &mut new_version.persisted_segments {
                seg.indexed_vector_fields.retain(|f| f != field);
                seg.vector_index_generations.remove(field);
            }
        }
        // Ensure the active writing segment uses the updated schema.
        self.commit_index_schema_update(&new_version, is_invert, || {})?;

        // Remove index files from each segment directory and from in-memory state
        // only after the manifest no longer references them.
        {
            let segs = self.persisted_segments.read();
            for seg in segs.iter() {
                if is_invert {
                    let idx_path = self.path.join(format!("seg_{}/{}_invert", seg.id, field));
                    seg.invert_indexes.write().remove(field);
                    let _ = std::fs::remove_dir_all(&idx_path);
                } else {
                    seg.vector_indexes.write().remove(field);
                }
            }
        }
        if !is_invert {
            remove_index_dirs(&vector_paths);
        }

        if is_invert {
            let writing_id = self.writing_segment.read().id;
            let writing_idx_path = self
                .path
                .join(format!("seg_{}/{}_invert", writing_id, field));
            let _ = std::fs::remove_dir_all(&writing_idx_path);
        }

        Ok(())
    }
}

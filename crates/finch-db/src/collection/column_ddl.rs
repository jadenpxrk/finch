use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::Arc;

use crate::collection_files::{
    effective_forward_file_format, effective_forward_store_open_options,
    effective_index_enable_mmap, forward_path_for_segment,
};
use crate::index::IndexBuilder;
use crate::invert::{frozen_path, InvertIndex};
use crate::segment::persisted::PersistedSegment;
use crate::segment::writing::{InvertIndexMeta, WritingSegment, WrittenSegmentMeta};
use crate::version::{PersistedSegmentVersion, Version};
use finch_storage::ForwardStoreOpenOptions;
use finch_types::{
    AddColumnOptions, AlterColumnOptions, CollectionSchema, FieldSchema, FileFormat, IndexParams,
    Status, ZResult,
};

use super::{ensure_orphan_segment_dir_removed, fill_invert_index, Collection};

mod rewrite;

use self::rewrite::{
    is_basic_numeric, rewrite_forward_add_numeric_column, rewrite_forward_alter_numeric_column,
};

struct RewrittenSegments {
    segments: Vec<Arc<PersistedSegment>>,
    versions: Vec<PersistedSegmentVersion>,
    old_ids: Vec<u32>,
    new_ids: Vec<u32>,
    next_segment_id: u32,
}

struct RewrittenSegment {
    segment: Arc<PersistedSegment>,
    version: PersistedSegmentVersion,
}

struct ProtectedSegmentIds {
    ids: HashSet<u32>,
    next_segment_id: u32,
}

struct SegmentRewriteTarget<'a> {
    new_schema: &'a CollectionSchema,
    forward_format: FileFormat,
    forward_opts: ForwardStoreOpenOptions,
}

fn sync_dir_best_effort(path: &Path) {
    let _ = fs::File::open(path).and_then(|d| d.sync_all());
}

// The field an ALTER COLUMN produces from the existing field and the caller's rename/schema.
fn alter_target_schema(
    existing: &FieldSchema,
    rename_to: Option<&str>,
    field_schema: Option<FieldSchema>,
) -> ZResult<FieldSchema> {
    if let (Some(new_name), Some(fs)) = (rename_to, field_schema.as_ref()) {
        if new_name != fs.name {
            return Err(Status::invalid_argument(
                "alter_column: new_name must match field_schema.name when both are provided",
            ));
        }
    }

    let Some(mut fs) = field_schema else {
        let mut fs = existing.clone();
        if let Some(new_name) = rename_to {
            fs.name = new_name.to_string();
        }
        return Ok(fs);
    };
    if !fs.data_type.is_scalar() || !is_basic_numeric(fs.data_type) {
        return Err(Status::invalid_argument(
            "alter_column only supports basic numeric scalar field_schema",
        ));
    }
    // If caller didn't specify index params, preserve existing index params by default.
    if fs.index_params.is_none() {
        fs.index_params = existing.index_params.clone();
    }
    Ok(fs)
}

fn copy_forward_store(src: &Path, dst: &Path) -> ZResult<()> {
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent).map_err(|e| Status::io_error(e.to_string()))?;
    }
    fs::copy(src, dst).map_err(|e| Status::io_error(e.to_string()))?;
    if let Ok(file) = fs::File::open(dst) {
        let _ = file.sync_all();
    }
    if let Some(parent) = dst.parent() {
        sync_dir_best_effort(parent);
    }
    Ok(())
}

impl Collection {
    // ── Column DDL ───────────────────────────────────────────────────────────

    fn cleanup_segment_dirs_best_effort(&self, ids: &[u32]) {
        for id in ids {
            let path = self.path.join(format!("seg_{}", id));
            let _ = fs::remove_dir_all(&path);
            if path.exists() {
                let _ = fs::remove_dir_all(&path);
            }
        }
        sync_dir_best_effort(&self.path);
    }

    fn cleanup_empty_uncommitted_segment_dirs_best_effort(&self, version: &Version) {
        let persisted: HashSet<u32> = version
            .persisted_segments
            .iter()
            .map(|seg| seg.segment_id)
            .collect();
        let Ok(entries) = fs::read_dir(&self.path) else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            let Some(id_str) = name.strip_prefix("seg_") else {
                continue;
            };
            let Ok(id) = id_str.parse::<u32>() else {
                continue;
            };
            if persisted.contains(&id) {
                continue;
            }
            let path = entry.path();
            let is_empty = fs::read_dir(&path)
                .map(|mut rd| rd.next().is_none())
                .unwrap_or(false);
            if is_empty {
                let _ = fs::remove_dir_all(path);
            }
        }
        sync_dir_best_effort(&self.path);
    }

    fn build_invert_indexes_for_rewritten_segment(
        &self,
        schema: &CollectionSchema,
        seg: &PersistedSegment,
        seg_path: &Path,
    ) -> ZResult<HashMap<String, InvertIndexMeta>> {
        let mut metas = HashMap::new();
        for field in &schema.fields {
            let Some(IndexParams::Invert(params)) = field.index_params.clone() else {
                continue;
            };
            let idx_path = seg_path.join(format!("{}_invert", field.name));
            let _ = fs::remove_dir_all(&idx_path);
            let idx = InvertIndex::open(
                &idx_path,
                field.name.clone(),
                field.data_type,
                params.clone(),
            )?;

            fill_invert_index(&idx, seg, &field.name)?;
            idx.sync()?;
            idx.freeze()?;
            drop(idx);

            metas.insert(
                field.name.clone(),
                InvertIndexMeta {
                    path: idx_path,
                    data_type: field.data_type,
                    params,
                },
            );
        }
        Ok(metas)
    }

    fn load_vector_indexes_for_rewritten_segment(
        &self,
        schema: &CollectionSchema,
        seg: &PersistedSegment,
        seg_path: &Path,
    ) -> ZResult<Vec<String>> {
        let index_enable_mmap = effective_index_enable_mmap(&self.options)?;
        let mut loaded = Vec::new();
        for field in schema.indexed_vector_fields() {
            let Some(params) = field.index_params.clone() else {
                continue;
            };
            if matches!(params, IndexParams::Invert(_)) {
                continue;
            }
            let index_path = seg_path.join(format!("idx_{}", field.name));
            let _ = fs::remove_dir_all(&index_path);
            IndexBuilder::build_from_persisted(seg, field, &params, &index_path)?;
            let idx =
                IndexBuilder::load_index(&field.name, &params, &index_path, index_enable_mmap)?;
            seg.add_vector_index(field.name.clone(), idx);
            loaded.push(field.name.clone());
        }
        Ok(loaded)
    }

    // Segment ids the rewrite must not reuse, and the first id it may allocate.
    fn protected_segment_ids(&self, version: &Version) -> ProtectedSegmentIds {
        let mut ids: HashSet<u32> = version
            .persisted_segments
            .iter()
            .map(|seg| seg.segment_id)
            .collect();
        if let Some(id) = version.writing_segment_id {
            ids.insert(id);
        }
        ids.insert(self.writing_segment.read().id);

        let mut next_segment_id = version.next_segment_id;
        for id in &ids {
            next_segment_id = next_segment_id.max(id.saturating_add(1));
        }
        ProtectedSegmentIds {
            ids,
            next_segment_id,
        }
    }

    fn rewrite_one_segment(
        &self,
        old: &PersistedSegment,
        new_id: u32,
        target: &SegmentRewriteTarget<'_>,
        rewrite_forward: &mut impl FnMut(&Path, &Path) -> ZResult<()>,
    ) -> ZResult<RewrittenSegment> {
        let old_seg_path = self.path.join(format!("seg_{}", old.id));
        let new_seg_path = self.path.join(format!("seg_{}", new_id));
        fs::create_dir_all(&new_seg_path).map_err(|e| Status::io_error(e.to_string()))?;
        let src_forward = forward_path_for_segment(&old_seg_path, target.forward_format);
        let dst_forward = forward_path_for_segment(&new_seg_path, target.forward_format);
        rewrite_forward(&src_forward, &dst_forward)?;

        let mut meta = WrittenSegmentMeta {
            segment_id: new_id,
            min_doc_id: old.min_doc_id,
            max_doc_id: old.max_doc_id,
            doc_count: old.doc_count,
            forward_path: dst_forward,
            invert_paths: HashMap::new(),
        };
        let persisted = PersistedSegment::open_forward_only_with_forward_store_options(
            &meta,
            target.forward_opts,
        )?;
        meta.invert_paths = self.build_invert_indexes_for_rewritten_segment(
            target.new_schema,
            &persisted,
            &new_seg_path,
        )?;
        persisted.load_invert_indexes(&meta)?;
        let indexed_vector_fields = self.load_vector_indexes_for_rewritten_segment(
            target.new_schema,
            &persisted,
            &new_seg_path,
        )?;

        Ok(RewrittenSegment {
            segment: Arc::new(persisted),
            version: PersistedSegmentVersion {
                segment_id: new_id,
                min_doc_id: old.min_doc_id,
                max_doc_id: old.max_doc_id,
                doc_count: old.doc_count,
                indexed_vector_fields,
                vector_index_generations: Default::default(),
            },
        })
    }

    fn rewrite_persisted_segments_copy_on_write(
        &self,
        version: &Version,
        new_schema: &CollectionSchema,
        mut rewrite_forward: impl FnMut(&Path, &Path) -> ZResult<()>,
    ) -> ZResult<RewrittenSegments> {
        let old_segments = self.persisted_segments.read().clone();
        let target = SegmentRewriteTarget {
            new_schema,
            forward_format: effective_forward_file_format(&self.options),
            forward_opts: effective_forward_store_open_options(&self.options),
        };
        let ProtectedSegmentIds {
            ids: protected_ids,
            mut next_segment_id,
        } = self.protected_segment_ids(version);

        let mut new_ids = Vec::new();
        let result = (|| {
            let mut new_segments = Vec::with_capacity(old_segments.len());
            let mut versions = Vec::with_capacity(old_segments.len());
            let mut old_ids = Vec::with_capacity(old_segments.len());

            for old in &old_segments {
                while protected_ids.contains(&next_segment_id) {
                    next_segment_id = next_segment_id.saturating_add(1);
                }
                let new_id = next_segment_id;
                ensure_orphan_segment_dir_removed(&self.path, new_id, &protected_ids)?;
                next_segment_id = next_segment_id.saturating_add(1);
                new_ids.push(new_id);

                let rewritten =
                    self.rewrite_one_segment(old, new_id, &target, &mut rewrite_forward)?;
                versions.push(rewritten.version);
                old_ids.push(old.id);
                new_segments.push(rewritten.segment);
            }

            Ok(RewrittenSegments {
                segments: new_segments,
                versions,
                old_ids,
                new_ids: new_ids.clone(),
                next_segment_id,
            })
        })();

        match result {
            Ok(rewritten) => Ok(rewritten),
            Err(e) => {
                self.cleanup_segment_dirs_best_effort(&new_ids);
                Err(e)
            }
        }
    }

    fn commit_schema_and_swap_segments(
        &self,
        new_version: &Version,
        rewritten: Option<RewrittenSegments>,
        new_writing: WritingSegment,
    ) -> ZResult<()> {
        // Readers hold this lock, so they never pair the new schema with the old segments.
        let _published = self.delete_store.write();
        self.commit_manifest_with_prepared(
            new_version,
            (new_writing, rewritten),
            |(new_writing, rewritten)| {
                *self.writing_segment.write() = new_writing;
                if let Some(rewritten) = rewritten {
                    *self.persisted_segments.write() = rewritten.segments;
                    self.cleanup_segment_dirs_best_effort(&rewritten.old_ids);
                }
                Ok(())
            },
            |(new_writing, rewritten)| {
                let failed_writing_id = new_writing.id;
                drop(new_writing);
                let failed_writing_path = self.path.join(format!("seg_{}", failed_writing_id));
                let failed_writing_empty = fs::read_dir(&failed_writing_path)
                    .map(|mut rd| rd.next().is_none())
                    .unwrap_or(false);
                if failed_writing_empty {
                    let _ = fs::remove_dir_all(&failed_writing_path);
                }
                if let Some(rewritten) = rewritten {
                    drop(rewritten.segments);
                    self.cleanup_segment_dirs_best_effort(&rewritten.new_ids);
                }
                let committed = self.cur_version();
                self.cleanup_empty_uncommitted_segment_dirs_best_effort(&committed);
                self.reopen_writing_invert_indexes_best_effort();
            },
        )
    }

    /// Add a new field to the schema (does not backfill existing segments)
    pub fn add_column(&self, field: FieldSchema, options: AddColumnOptions) -> ZResult<()> {
        self.add_column_with_expression(field, None, options)
    }

    /// Add a new numeric scalar field, optionally populated by an arithmetic expression.
    ///
    /// `expression` is supported for numeric scalar fields and is evaluated
    /// across all existing persisted segments.
    pub fn add_column_with_expression(
        &self,
        field: FieldSchema,
        expression: Option<&str>,
        _options: AddColumnOptions,
    ) -> ZResult<()> {
        self.check_not_readonly()?;
        let _guard = self.write_lock.lock();

        if !field.data_type.is_scalar() || !is_basic_numeric(field.data_type) {
            return Err(Status::invalid_argument(
                "add_column expression is only supported for basic numeric scalar fields",
            ));
        }

        let expr_trimmed = expression.map(str::trim).unwrap_or("");
        if expr_trimmed.is_empty() && !field.nullable {
            return Err(Status::invalid_argument(
                "add_column with empty expression is not supported for non-nullable columns",
            ));
        }

        // Ensure no in-flight data in the writing segment so DDL applies cleanly
        // to on-disk segments.
        self.flush_writing_segment_locked()?;

        let version = self.cur_version();
        if version.schema.has_field(&field.name) {
            return Err(Status::already_exists(format!(
                "column {} already exists",
                field.name
            )));
        }

        let mut new_schema = version.schema.clone();
        new_schema.fields.push(field.clone());
        new_schema.validate()?;

        let rewritten =
            self.rewrite_persisted_segments_copy_on_write(&version, &new_schema, |src, dst| {
                rewrite_forward_add_numeric_column(src, dst, &new_schema, &field, expression)
            })?;
        self.commit_rewritten_schema(&version, &new_schema, rewritten)
    }

    // Commits `new_schema` with the rewritten segments and swaps in a matching writing segment.
    fn commit_rewritten_schema(
        &self,
        version: &Version,
        new_schema: &CollectionSchema,
        rewritten: RewrittenSegments,
    ) -> ZResult<()> {
        let mut new_version = version.clone();
        new_version.schema = new_schema.clone();
        new_version.persisted_segments = rewritten.versions.clone();
        new_version.next_segment_id = rewritten.next_segment_id.max(
            new_version
                .writing_segment_id
                .map(|id| id.saturating_add(1))
                .unwrap_or(0)
                .max(self.writing_segment.read().id.saturating_add(1)),
        );
        let new_writing = self.prepare_replacement_writing_segment_locked(None, new_schema)?;
        if let Err(e) =
            self.commit_schema_and_swap_segments(&new_version, Some(rewritten), new_writing)
        {
            self.cleanup_empty_uncommitted_segment_dirs_best_effort(version);
            return Err(e);
        }

        Ok(())
    }

    /// Remove a field from the schema (existing data in segments is orphaned but not deleted)
    pub fn drop_column(&self, field_name: &str) -> ZResult<()> {
        self.check_not_readonly()?;
        let _guard = self.write_lock.lock();
        let version = self.cur_version();

        let Some(existing) = version.schema.get_field(field_name) else {
            return Err(Status::not_found(format!(
                "column {} not found",
                field_name
            )));
        };

        // DDL column operations are only supported for basic numeric scalar fields.
        if !existing.data_type.is_scalar() || !is_basic_numeric(existing.data_type) {
            return Err(Status::invalid_argument(
                "drop_column is only supported for basic numeric scalar fields",
            ));
        }

        // Ensure any buffered writes become durable under the old schema before
        // updating the schema.
        self.flush_writing_segment_locked()?;
        // IMPORTANT: refresh version after flush so we don't overwrite the
        // manifest with stale persisted-segment metadata.
        let version = self.cur_version();

        let mut new_schema = version.schema.clone();
        new_schema.fields.retain(|f| f.name != field_name);
        let mut new_version = (*version).clone();
        new_version.schema = new_schema;
        for seg in &mut new_version.persisted_segments {
            seg.indexed_vector_fields.retain(|f| f != field_name);
        }
        let new_writing =
            self.prepare_replacement_writing_segment_locked(None, &new_version.schema)?;
        self.commit_schema_and_swap_segments(&new_version, None, new_writing)?;

        for seg in self.persisted_segments.read().iter() {
            seg.invert_indexes.write().remove(field_name);
            seg.vector_indexes.write().remove(field_name);
            let seg_path = self.path.join(format!("seg_{}", seg.id));
            let invert_path = seg_path.join(format!("{}_invert", field_name));
            let _ = fs::remove_dir_all(&invert_path);
            let _ = fs::remove_file(frozen_path(&invert_path));
            let _ = fs::remove_dir_all(seg_path.join(format!("idx_{}", field_name)));
        }
        sync_dir_best_effort(&self.path);

        Ok(())
    }

    /// Rename a field (optionally) and/or update its schema (basic numeric scalars only).
    pub fn alter_column(
        &self,
        field_name: &str,
        rename_to: Option<&str>,
        field_schema: Option<FieldSchema>,
        _options: AlterColumnOptions,
    ) -> ZResult<()> {
        self.check_not_readonly()?;
        let _guard = self.write_lock.lock();
        // Ensure any buffered writes become durable under the old schema before
        // updating the schema.
        self.flush_writing_segment_locked()?;
        let version = self.cur_version();

        let Some(existing) = version.schema.get_field(field_name).cloned() else {
            return Err(Status::not_found(format!(
                "column {} not found",
                field_name
            )));
        };

        // DDL column operations are only supported for basic numeric scalar fields.
        if !existing.data_type.is_scalar() || !is_basic_numeric(existing.data_type) {
            return Err(Status::invalid_argument(
                "alter_column is only supported for basic numeric scalar fields",
            ));
        }

        let target_schema = alter_target_schema(&existing, rename_to, field_schema)?;
        let old_name = existing.name.clone();
        let new_name = target_schema.name.clone();
        let need_rewrite_forward = old_name != new_name
            || existing.data_type != target_schema.data_type
            || existing.nullable != target_schema.nullable;

        if old_name != new_name && version.schema.has_field(&new_name) {
            return Err(Status::already_exists(format!(
                "column {} already exists",
                new_name
            )));
        }

        // Validate the updated schema.
        let mut new_schema = version.schema.clone();
        if let Some(f) = new_schema.get_field_mut(&old_name) {
            *f = target_schema.clone();
        }
        new_schema.validate()?;

        let rewrite_forward = |src: &Path, dst: &Path| {
            if !need_rewrite_forward {
                return copy_forward_store(src, dst);
            }
            rewrite_forward_alter_numeric_column(src, dst, &old_name, &new_schema, &target_schema)
        };
        let rewritten =
            self.rewrite_persisted_segments_copy_on_write(&version, &new_schema, rewrite_forward)?;
        self.commit_rewritten_schema(&version, &new_schema, rewritten)
    }
}

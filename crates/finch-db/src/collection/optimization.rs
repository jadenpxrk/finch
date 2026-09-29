use std::collections::{HashMap, HashSet};
use std::fs;
use std::sync::Arc;

use finch_storage::ForwardStoreOpenOptions;
use finch_types::{
    CollectionSchema, FieldSchema, IndexParams, OptimizeOptions, Status, ZResult,
    MAX_DOC_COUNT_PER_SEGMENT,
};
use roaring::RoaringTreemap;

use super::{
    count_deleted_docs_in_segment, Collection, COMPACT_DELETE_RATIO_THRESHOLD,
    HNSW_AUTO_MAX_DOCS_PER_SEGMENT,
};
use crate::collection_files::{effective_forward_store_open_options, effective_index_enable_mmap};
use crate::index::IndexBuilder;
use crate::segment::persisted::{PersistedSegment, VectorIndex};
use crate::version::{PersistedSegmentVersion, Version};

mod compaction;

// Read-only inputs shared by every task of one optimize run.
struct OptimizeEnv<'a> {
    schema: &'a CollectionSchema,
    indexed_field_params: &'a [(FieldSchema, IndexParams)],
    delete_bitmap: &'a RoaringTreemap,
    reserved_segment_ids: &'a HashSet<u32>,
    forward_opts: ForwardStoreOpenOptions,
    index_enable_mmap: bool,
    max_docs_per_segment: u64,
    target_segment_size: Option<u64>,
    pool: &'a rayon::ThreadPool,
}

struct IndexedFields {
    params: Vec<(FieldSchema, IndexParams)>,
    names: Vec<String>,
    has_dense_hnsw: bool,
}

impl IndexedFields {
    fn from_schema(schema: &CollectionSchema) -> Self {
        let params: Vec<(FieldSchema, IndexParams)> = schema
            .indexed_vector_fields()
            .filter_map(|f| f.index_params.clone().map(|p| (f.clone(), p)))
            .collect();
        let has_dense_hnsw = params
            .iter()
            .any(|(_, params)| matches!(params, IndexParams::Hnsw(_)));
        let names = params.iter().map(|(field, _)| field.name.clone()).collect();
        Self {
            params,
            names,
            has_dense_hnsw,
        }
    }

    fn missing_in_any(&self, version: &Version) -> bool {
        if self.names.is_empty() {
            return false;
        }
        version.persisted_segments.iter().any(|segment| {
            self.names
                .iter()
                .any(|field| !segment.indexed_vector_fields.iter().any(|x| x == field))
        })
    }
}

// Live counts are computed against docs present in segments, so tombstones for doc_ids that
// no longer exist (e.g. after rebuild compaction) do not force optimize to rebuild again.
struct SegmentLiveCounts {
    meta_by_id: HashMap<u32, PersistedSegmentVersion>,
    live_by_id: HashMap<u32, u64>,
    total_doc_count: u64,
    total_live_count: u64,
}

impl SegmentLiveCounts {
    fn compute(
        version: &Version,
        segs: &[Arc<PersistedSegment>],
        delete_bitmap: &RoaringTreemap,
    ) -> ZResult<Self> {
        let mut meta_by_id: HashMap<u32, PersistedSegmentVersion> = HashMap::new();
        for meta in &version.persisted_segments {
            meta_by_id.insert(meta.segment_id, meta.clone());
        }
        let mut live_by_id: HashMap<u32, u64> = HashMap::new();
        let mut total_doc_count: u64 = 0;
        let mut total_live_count: u64 = 0;
        for seg in segs {
            let meta = meta_by_id.get(&seg.id).ok_or_else(missing_segment_meta)?;
            total_doc_count = total_doc_count.saturating_add(meta.doc_count);
            let deleted = count_deleted_docs_in_segment(meta, seg, delete_bitmap)?;
            let live = meta.doc_count.saturating_sub(deleted);
            total_live_count = total_live_count.saturating_add(live);
            live_by_id.insert(seg.id, live);
        }
        Ok(Self {
            meta_by_id,
            live_by_id,
            total_doc_count,
            total_live_count,
        })
    }
}

struct OptimizeDecision {
    needs_index_build: bool,
    has_deletes_in_segments: bool,
    rebuild: bool,
    auto_hnsw_split: bool,
    wants_split: bool,
}

impl OptimizeDecision {
    fn new(
        options: &OptimizeOptions,
        schema: &CollectionSchema,
        indexed: &IndexedFields,
        counts: &SegmentLiveCounts,
        needs_index_build: bool,
    ) -> Self {
        let wants_target_split = options.target_segment_size.is_some_and(|v| v > 0);
        let deleted_in_segments = counts
            .total_doc_count
            .saturating_sub(counts.total_live_count);
        let rebuild = counts.total_doc_count > 0
            && (counts.total_live_count as f64)
                < (counts.total_doc_count as f64) * (1.0 - COMPACT_DELETE_RATIO_THRESHOLD);
        let auto_hnsw_split = indexed.has_dense_hnsw
            && !wants_target_split
            && options.max_segments.is_none()
            && schema.max_doc_count_per_segment >= MAX_DOC_COUNT_PER_SEGMENT
            && counts.total_live_count > HNSW_AUTO_MAX_DOCS_PER_SEGMENT;
        Self {
            needs_index_build,
            has_deletes_in_segments: deleted_in_segments > 0,
            rebuild,
            auto_hnsw_split,
            wants_split: wants_target_split || auto_hnsw_split,
        }
    }

    fn within_segment_budget(&self, max_segments: Option<usize>, seg_count: usize) -> bool {
        max_segments.is_some_and(|max| {
            seg_count <= max
                && !self.has_deletes_in_segments
                && !self.wants_split
                && !self.needs_index_build
        })
    }

    fn only_missing_indexes(&self, seg_count: usize) -> bool {
        seg_count == 1
            && !self.has_deletes_in_segments
            && !self.wants_split
            && self.needs_index_build
            && !self.rebuild
    }

    fn nothing_to_do(&self, seg_count: usize) -> bool {
        seg_count == 1
            && !self.has_deletes_in_segments
            && !self.wants_split
            && !self.needs_index_build
    }

    fn max_docs_per_segment(&self, schema: &CollectionSchema) -> u64 {
        if self.auto_hnsw_split {
            HNSW_AUTO_MAX_DOCS_PER_SEGMENT
        } else {
            schema.max_doc_count_per_segment.max(1)
        }
    }
}

#[derive(Debug)]
enum OptimizeTask {
    BuildIndexes {
        seg_id: u32,
    },
    Compact {
        input_ids: Vec<u32>,
        drop_deleted: bool,
    },
}

struct OutputSegmentIds {
    active_writing: u32,
    reserved: HashSet<u32>,
    next_out: u32,
}

// The next manifest and segment set, accumulated while tasks run.
struct OptimizeOutcome {
    new_version: Version,
    seg_map: HashMap<u32, Arc<PersistedSegment>>,
    removed_seg_ids: Vec<u32>,
    did_compact: bool,
    next_out_seg_id: u32,
}

impl OptimizeOutcome {
    fn start(version: &Version, segs: &[Arc<PersistedSegment>], ids: &OutputSegmentIds) -> Self {
        let mut new_version = version.clone();
        new_version.writing_segment_id = Some(ids.active_writing);
        Self {
            new_version,
            seg_map: segs.iter().map(|s| (s.id, s.clone())).collect(),
            removed_seg_ids: Vec::new(),
            did_compact: false,
            next_out_seg_id: ids.next_out,
        }
    }
}

fn missing_segment_meta() -> Status {
    Status::internal("optimize: missing segment meta for persisted segment")
}

fn segment_meta(version: &Version, seg_id: u32) -> ZResult<&PersistedSegmentVersion> {
    version
        .persisted_segments
        .iter()
        .find(|s| s.segment_id == seg_id)
        .ok_or_else(missing_segment_meta)
}

fn set_segment_indexed_fields(version: &mut Version, seg_id: u32, fields: Vec<String>) {
    if let Some(meta) = version
        .persisted_segments
        .iter_mut()
        .find(|s| s.segment_id == seg_id)
    {
        meta.indexed_vector_fields = fields;
    }
}

fn push_task_for_group(
    group: &[u32],
    decision: &OptimizeDecision,
    indexed_fields: &[String],
    meta_by_id: &HashMap<u32, PersistedSegmentVersion>,
    tasks: &mut Vec<OptimizeTask>,
) -> ZResult<()> {
    if group.is_empty() {
        return Ok(());
    }

    if group.len() == 1 && !decision.rebuild && !decision.wants_split {
        let seg_id = group[0];
        if indexed_fields.is_empty() {
            return Ok(());
        }
        let meta = meta_by_id.get(&seg_id).ok_or_else(missing_segment_meta)?;
        let all_ready = indexed_fields
            .iter()
            .all(|field| meta.indexed_vector_fields.iter().any(|x| x == field));
        if !all_ready {
            tasks.push(OptimizeTask::BuildIndexes { seg_id });
        }
        return Ok(());
    }

    tasks.push(OptimizeTask::Compact {
        input_ids: group.to_vec(),
        drop_deleted: decision.rebuild,
    });
    Ok(())
}

// Groups doc-id-consecutive segments up to the per-segment doc budget; rebuilds count live docs only.
fn plan_optimize_tasks(
    segs: &[Arc<PersistedSegment>],
    counts: &SegmentLiveCounts,
    decision: &OptimizeDecision,
    indexed_fields: &[String],
    max_docs_per_segment: u64,
) -> ZResult<Vec<OptimizeTask>> {
    let mut tasks: Vec<OptimizeTask> = Vec::new();
    let mut cur_group: Vec<u32> = Vec::new();
    let mut cur_doc_count: u64 = 0;
    let mut cur_live_count: u64 = 0;

    for seg in segs {
        let meta = counts
            .meta_by_id
            .get(&seg.id)
            .ok_or_else(missing_segment_meta)?;
        let seg_doc_count = meta.doc_count;
        let seg_live_count = *counts.live_by_id.get(&seg.id).unwrap_or(&seg_doc_count);

        let would_exceed = if decision.rebuild {
            cur_live_count.saturating_add(seg_live_count) > max_docs_per_segment
        } else {
            cur_doc_count.saturating_add(seg_doc_count) > max_docs_per_segment
        };
        if !cur_group.is_empty() && would_exceed {
            push_task_for_group(
                &cur_group,
                decision,
                indexed_fields,
                &counts.meta_by_id,
                &mut tasks,
            )?;
            cur_group.clear();
            cur_doc_count = 0;
            cur_live_count = 0;
        }

        cur_group.push(seg.id);
        cur_doc_count = cur_doc_count.saturating_add(seg_doc_count);
        cur_live_count = cur_live_count.saturating_add(seg_live_count);
    }

    push_task_for_group(
        &cur_group,
        decision,
        indexed_fields,
        &counts.meta_by_id,
        &mut tasks,
    )?;
    Ok(tasks)
}

impl Collection {
    /// Compact persisted segments, removing deleted docs and rebuilding indexes.
    ///
    /// This is a streaming implementation: it does not materialize all docs in memory.
    pub fn optimize(&self, options: OptimizeOptions) -> ZResult<()> {
        self.check_not_readonly()?;
        let _guard = self.write_lock.lock();

        // First flush writing segment
        self.flush_writing_segment_locked()?;

        let mut segs = self.persisted_segments.read().clone();
        if segs.is_empty() {
            return Ok(()); // Nothing to optimize
        }
        // Compaction streams rows in this order; segment ids do not follow doc ids after a compaction.
        segs.sort_unstable_by_key(|s| (s.min_doc_id, s.id));

        let version = self.cur_version();
        let schema = version.schema.clone();
        let forward_opts = effective_forward_store_open_options(&self.options);
        let index_enable_mmap = effective_index_enable_mmap(&self.options)?;
        let delete_bitmap = self.delete_store.read().bitmap();
        let indexed = IndexedFields::from_schema(&version.schema);
        let needs_index_build = indexed.missing_in_any(&version);
        let counts = SegmentLiveCounts::compute(&version, &segs, delete_bitmap.as_ref())?;
        let decision =
            OptimizeDecision::new(&options, &schema, &indexed, &counts, needs_index_build);

        if decision.within_segment_budget(options.max_segments, segs.len()) {
            return Ok(());
        }
        if decision.only_missing_indexes(segs.len()) {
            return self.build_missing_indexes_and_commit(
                &version,
                &segs[0],
                &indexed.params,
                index_enable_mmap,
            );
        }
        if decision.nothing_to_do(segs.len()) {
            return Ok(());
        }

        let maybe_pool = Self::build_thread_pool(options.concurrency)?;
        let pool = maybe_pool
            .as_ref()
            .unwrap_or_else(|| crate::config::optimize_pool());

        let ids = self.output_segment_ids(&version);
        let max_docs_per_segment = decision.max_docs_per_segment(&schema);

        let tasks = plan_optimize_tasks(
            &segs,
            &counts,
            &decision,
            &indexed.names,
            max_docs_per_segment,
        )?;
        if tasks.is_empty() {
            return Ok(());
        }

        let env = OptimizeEnv {
            schema: &schema,
            indexed_field_params: &indexed.params,
            delete_bitmap: delete_bitmap.as_ref(),
            reserved_segment_ids: &ids.reserved,
            forward_opts,
            index_enable_mmap,
            max_docs_per_segment,
            target_segment_size: options.target_segment_size.filter(|v| *v > 0),
            pool,
        };
        let mut outcome = OptimizeOutcome::start(&version, &segs, &ids);
        for task in tasks {
            self.run_optimize_task(&env, task, &mut outcome)?;
        }
        self.commit_optimize_outcome(outcome)
    }

    // Avoid reusing the open writing segment id: its flush writes `seg_{id}`.
    fn output_segment_ids(&self, version: &Version) -> OutputSegmentIds {
        let active_writing = self.writing_segment.read().id;
        let reserved: HashSet<u32> = version
            .persisted_segments
            .iter()
            .map(|s| s.segment_id)
            .chain(std::iter::once(active_writing))
            .collect();
        let next_out = version
            .next_segment_id
            .max(active_writing.saturating_add(1));
        OutputSegmentIds {
            active_writing,
            reserved,
            next_out,
        }
    }

    // Single-segment fast path: indexes are published only after the manifest records them.
    fn build_missing_indexes_and_commit(
        &self,
        version: &Version,
        seg: &Arc<PersistedSegment>,
        indexed_field_params: &[(FieldSchema, IndexParams)],
        index_enable_mmap: bool,
    ) -> ZResult<()> {
        let seg_id = seg.id;
        let seg_meta = segment_meta(version, seg_id)?;

        let mut prepared_indexes: Vec<(String, VectorIndex)> = Vec::new();
        let built_fields = self.build_missing_segment_indexes(
            seg,
            seg_meta.indexed_vector_fields.clone(),
            indexed_field_params,
            index_enable_mmap,
            |field_name, idx| prepared_indexes.push((field_name, idx)),
        )?;

        let mut new_version = version.clone();
        set_segment_indexed_fields(&mut new_version, seg_id, built_fields);
        self.commit_manifest_with_prepared(
            &new_version,
            prepared_indexes,
            |prepared_indexes| {
                for (field_name, idx) in prepared_indexes {
                    seg.add_vector_index(field_name, idx);
                }
                Ok(())
            },
            |_| {},
        )
    }

    // Builds and loads each index missing from `built_fields`, handing every loaded index
    // to `on_loaded`, and returns the sorted, deduplicated field list.
    fn build_missing_segment_indexes(
        &self,
        seg: &PersistedSegment,
        mut built_fields: Vec<String>,
        indexed_field_params: &[(FieldSchema, IndexParams)],
        index_enable_mmap: bool,
        mut on_loaded: impl FnMut(String, VectorIndex),
    ) -> ZResult<Vec<String>> {
        for (field, params) in indexed_field_params {
            let field_name = &field.name;
            if built_fields.iter().any(|x| x == field_name) {
                continue;
            }
            let index_path = self.path.join(format!("seg_{}/idx_{}", seg.id, field_name));
            IndexBuilder::build_from_persisted(seg, field, params, &index_path)?;
            let idx = IndexBuilder::load_index(field_name, params, &index_path, index_enable_mmap)?;
            on_loaded(field_name.clone(), idx);
            built_fields.push(field_name.clone());
        }
        built_fields.sort();
        built_fields.dedup();
        Ok(built_fields)
    }

    fn run_optimize_task(
        &self,
        env: &OptimizeEnv<'_>,
        task: OptimizeTask,
        outcome: &mut OptimizeOutcome,
    ) -> ZResult<()> {
        match task {
            OptimizeTask::BuildIndexes { seg_id } => {
                self.run_build_indexes_task(env, seg_id, outcome)
            }
            OptimizeTask::Compact {
                input_ids,
                drop_deleted,
            } => self.run_compact_task(env, &input_ids, drop_deleted, outcome),
        }
    }

    fn run_build_indexes_task(
        &self,
        env: &OptimizeEnv<'_>,
        seg_id: u32,
        outcome: &mut OptimizeOutcome,
    ) -> ZResult<()> {
        let seg = outcome.seg_map.get(&seg_id).cloned().ok_or_else(|| {
            Status::internal("optimize: missing persisted segment for index build")
        })?;
        let seg_meta = segment_meta(&outcome.new_version, seg_id)?;

        let built_fields = self.build_missing_segment_indexes(
            &seg,
            seg_meta.indexed_vector_fields.clone(),
            env.indexed_field_params,
            env.index_enable_mmap,
            |field_name, idx| seg.add_vector_index(field_name, idx),
        )?;
        set_segment_indexed_fields(&mut outcome.new_version, seg_id, built_fields);
        Ok(())
    }

    fn run_compact_task(
        &self,
        env: &OptimizeEnv<'_>,
        input_ids: &[u32],
        drop_deleted: bool,
        outcome: &mut OptimizeOutcome,
    ) -> ZResult<()> {
        let mut input_segs: Vec<Arc<PersistedSegment>> = Vec::with_capacity(input_ids.len());
        for id in input_ids {
            let seg = outcome.seg_map.get(id).cloned().ok_or_else(|| {
                Status::internal("optimize: missing persisted segment for compaction")
            })?;
            input_segs.push(seg);
        }
        input_segs.sort_unstable_by_key(|s| (s.min_doc_id, s.id));

        let group = self.compact_group(env, outcome.next_out_seg_id, &input_segs, drop_deleted)?;
        outcome.next_out_seg_id = group.next_seg_id;
        outcome.did_compact = true;

        for id in input_ids {
            outcome.removed_seg_ids.push(*id);
            outcome.seg_map.remove(id);
        }
        for seg in group.segs {
            outcome.seg_map.insert(seg.id, seg);
        }

        outcome
            .new_version
            .persisted_segments
            .retain(|s| !input_ids.contains(&s.segment_id));
        outcome
            .new_version
            .persisted_segments
            .extend(group.versions);
        Ok(())
    }

    fn commit_optimize_outcome(&self, outcome: OptimizeOutcome) -> ZResult<()> {
        let OptimizeOutcome {
            mut new_version,
            seg_map,
            removed_seg_ids,
            did_compact,
            next_out_seg_id,
        } = outcome;
        if did_compact {
            new_version.next_segment_id = next_out_seg_id;
        }
        new_version
            .persisted_segments
            .sort_unstable_by_key(|s| s.segment_id);

        self.commit_manifest_with_prepared(
            &new_version,
            (seg_map, did_compact, removed_seg_ids),
            |(seg_map, did_compact, removed_seg_ids)| {
                self.publish_optimized_segments(seg_map, did_compact, removed_seg_ids);
                Ok(())
            },
            |_| {},
        )
    }

    fn publish_optimized_segments(
        &self,
        seg_map: HashMap<u32, Arc<PersistedSegment>>,
        did_compact: bool,
        mut removed_seg_ids: Vec<u32>,
    ) {
        let mut new_segs: Vec<Arc<PersistedSegment>> = seg_map.into_values().collect();
        new_segs.sort_unstable_by_key(|s| s.id);
        *self.persisted_segments.write() = new_segs;

        // Remove old segment directories (best-effort) only after the manifest is committed.
        if did_compact {
            removed_seg_ids.sort_unstable();
            removed_seg_ids.dedup();
            for seg_id in removed_seg_ids {
                let _ = fs::remove_dir_all(self.path.join(format!("seg_{}", seg_id)));
            }
            let _ = fs::File::open(&self.path).and_then(|dir| dir.sync_all());
        }
    }
}

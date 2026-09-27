use std::collections::HashMap;
use std::sync::Arc;

use rayon::prelude::*;

use finch_types::{CollectionSchema, Doc, MetricType, Status, VectorQuery, ZResult};

use crate::profile::QueryProfile;
use crate::query_filter::prepare_filter_expr;
use crate::query_output::QueryOutputSelection;
use crate::segment::persisted::{AnnSearch, PersistedSegment};
use crate::segment::writing::WritingSegment;
use crate::sqlengine::executor::DocFilterEvaluator;
use crate::sqlengine::parser::FilterExpr;
use crate::vector_normalization::{has_query_vector_payload, validate_and_normalize_query_payload};
use crate::vector_search_field::{resolve_vector_search_field, VectorSearchField};

use super::{Collection, MAX_OUTPUT_FIELDS, MAX_QUERY_TOPK};

struct SegmentSearch<'a> {
    segment_topk: usize,
    ef_or_nprobe: Option<u32>,
    is_linear: bool,
    filter_expr: Option<&'a FilterExpr>,
    delete_bitmap: &'a Arc<roaring::RoaringTreemap>,
    query_pool: &'a rayon::ThreadPool,
}

impl<'a> SegmentSearch<'a> {
    fn ann(&self) -> AnnSearch<'a> {
        AnnSearch {
            topk: self.segment_topk,
            ef_or_nprobe: self.ef_or_nprobe,
            force_linear: self.is_linear,
            delete_bitmap: self.delete_bitmap.clone(),
            filter_expr: self.filter_expr,
        }
    }
}

struct ResultRefinement {
    use_refiner: bool,
    refiner_k: u32,
    radius: Option<f32>,
    topk: usize,
}

impl ResultRefinement {
    fn segment_topk(&self) -> usize {
        let mut segment_topk = if self.use_refiner {
            // Pull a larger candidate set per segment before global refine.
            self.refiner_k as usize
        } else {
            self.topk
        };
        if self.radius.is_some() {
            // Radius is a post-filter; expand the candidate set to avoid
            // trivially returning < topk results when some of the top-k are
            // outside the radius.
            segment_topk =
                segment_topk.max(((self.topk as u32).saturating_mul(10)).min(10_000) as usize);
        }

        segment_topk
    }
}

// Per-query search knobs derived from `QueryParams` and the resolved vector field.
struct VectorSearchParams {
    ef_or_nprobe: Option<u32>,
    bf_pks: Option<Vec<String>>,
    is_linear: bool,
    refinement: ResultRefinement,
}

impl VectorSearchParams {
    fn new(query: &VectorQuery, vector_field: &VectorSearchField<'_>, topk: usize) -> Self {
        let params = &query.query_params;
        let bf_pks = params.bf_pks.clone();
        let use_refiner_requested = params.use_refiner && bf_pks.is_none();
        let is_binary = vector_field.is_binary32 || vector_field.is_binary64;
        // Refinement is only meaningful for indexed dense f32-family vectors.
        let use_refiner = use_refiner_requested && !is_binary && !vector_field.is_sparse;
        Self {
            ef_or_nprobe: params.ef.or(params.n_probe),
            bf_pks,
            is_linear: params.effective_is_linear(),
            refinement: ResultRefinement {
                use_refiner,
                refiner_k: params.effective_refiner_k(topk),
                radius: params.effective_radius(),
                topk,
            },
        }
    }
}

// Query state resolved before choosing between filter-only and vector search.
struct PreparedVectorQuery<'a> {
    schema: &'a CollectionSchema,
    output_selection: &'a QueryOutputSelection,
    filter_expr: Option<&'a FilterExpr>,
    delete_bitmap: &'a Arc<roaring::RoaringTreemap>,
    topk: usize,
}

// The dense query a refine pass recomputes exact distances for.
struct DenseTarget<'a> {
    field: &'a str,
    query_vector: &'a [f32],
    metric: MetricType,
}

struct FinalizedResults {
    candidate_count: usize,
    refine_duration: Option<std::time::Duration>,
}

fn extend_segment_results(
    results: &mut Vec<(u64, f32)>,
    segment_results: Vec<ZResult<Vec<(u64, f32)>>>,
) -> ZResult<()> {
    for res in segment_results {
        results.extend(res?);
    }
    Ok(())
}

impl Collection {
    fn append_query_results(
        &self,
        results: &mut Vec<(u64, f32)>,
        query: &VectorQuery,
        vector_field: &VectorSearchField<'_>,
        bf_pks: Option<&[String]>,
        search: &SegmentSearch<'_>,
    ) -> ZResult<()> {
        use finch_core::algorithm::flat_sparse::SparseVector;

        if let Some(pks) = bf_pks {
            if vector_field.is_sparse {
                return Err(Status::invalid_argument(
                    "bf_pks is only supported for dense vector queries",
                ));
            }
            return self.append_dense_bf_pk_results(
                results,
                pks,
                vector_field,
                query,
                search.filter_expr,
                search.delete_bitmap.as_ref(),
            );
        }

        if vector_field.is_sparse {
            let q_sparse =
                SparseVector::new(query.sparse_indices.clone(), query.sparse_values.clone());

            return self.append_sparse_query_results(results, vector_field.name, &q_sparse, search);
        }

        if vector_field.is_binary32 {
            self.append_binary_u32_query_results(
                results,
                vector_field.name,
                &query.query_vector_u32,
                search,
            )
        } else if vector_field.is_binary64 {
            self.append_binary_u64_query_results(
                results,
                vector_field.name,
                &query.query_vector_u64,
                search,
            )
        } else {
            let target = DenseTarget {
                field: vector_field.name,
                query_vector: &query.query_vector,
                metric: vector_field.metric,
            };
            self.append_dense_query_results(results, &target, search)
        }
    }

    fn append_dense_bf_pk_results(
        &self,
        results: &mut Vec<(u64, f32)>,
        pks: &[String],
        vector_field: &VectorSearchField<'_>,
        query: &VectorQuery,
        filter_expr: Option<&FilterExpr>,
        delete_bitmap: &roaring::RoaringTreemap,
    ) -> ZResult<()> {
        use crate::segment::persisted::{
            compute_distance, compute_hamming_u32, compute_hamming_u64,
        };

        if pks.len() > 1_000_000 {
            return Err(Status::invalid_argument("bf_pks is too large"));
        }

        let mut ids: Vec<u64> = Vec::with_capacity(pks.len());
        for pk in pks {
            if let Ok(Some(id)) = self.id_map.get(pk) {
                if !delete_bitmap.contains(id) {
                    ids.push(id);
                }
            }
        }
        ids.sort_unstable();
        ids.dedup();

        if ids.is_empty() {
            return Ok(());
        }

        let field = vector_field.name;
        let docs = self.fetch_by_ids(&ids)?;
        let row_locator = self.row_locator();
        for id in ids {
            let Some(doc) = docs.get(&id) else {
                continue;
            };
            if let Some(expr) = filter_expr {
                let row_id = row_locator.row_id(id);
                if !DocFilterEvaluator::passes(expr, doc, Some(delete_bitmap), row_id) {
                    continue;
                }
            }
            if vector_field.is_binary32 {
                if let Some(vec) = doc.get_vec_u32(field) {
                    let dist = compute_hamming_u32(vec, &query.query_vector_u32);
                    results.push((id, dist));
                }
            } else if vector_field.is_binary64 {
                if let Some(vec) = doc.get_vec_u64(field) {
                    let dist = compute_hamming_u64(vec, &query.query_vector_u64);
                    results.push((id, dist));
                }
            } else if let Some(vec) = doc.get_vec_f32(field) {
                let dist = compute_distance(vec, &query.query_vector, vector_field.metric);
                results.push((id, dist));
            }
        }

        Ok(())
    }

    fn append_sparse_query_results(
        &self,
        results: &mut Vec<(u64, f32)>,
        field: &str,
        query_vector: &finch_core::algorithm::flat_sparse::SparseVector,
        search: &SegmentSearch<'_>,
    ) -> ZResult<()> {
        self.append_segment_search(
            results,
            search.query_pool,
            |seg| seg.search_sparse_vectors(field, query_vector, search.ann()),
            |writing| {
                writing.search_sparse_vectors(
                    field,
                    query_vector,
                    search.segment_topk,
                    search.delete_bitmap.as_ref(),
                    search.filter_expr,
                )
            },
        )
    }

    // Searches every persisted segment on `pool`, then the writing segment.
    fn append_segment_search(
        &self,
        results: &mut Vec<(u64, f32)>,
        pool: &rayon::ThreadPool,
        persisted: impl Fn(&Arc<PersistedSegment>) -> ZResult<Vec<(u64, f32)>> + Sync + Send,
        writing: impl FnOnce(&WritingSegment) -> ZResult<Vec<(u64, f32)>>,
    ) -> ZResult<()> {
        // Hold the persisted list while reading the writing segment so a concurrent flush cannot hide docs.
        let segs = self.persisted_segments.read();
        let segment_results: Vec<ZResult<Vec<(u64, f32)>>> =
            pool.install(|| segs.par_iter().map(persisted).collect());
        extend_segment_results(results, segment_results)?;

        let writing_segment = self.writing_segment.read();
        results.extend(writing(&writing_segment)?);
        Ok(())
    }

    fn append_binary_u32_query_results(
        &self,
        results: &mut Vec<(u64, f32)>,
        field: &str,
        query_vector: &[u32],
        search: &SegmentSearch<'_>,
    ) -> ZResult<()> {
        let SegmentSearch {
            segment_topk,
            is_linear,
            filter_expr,
            delete_bitmap,
            query_pool,
            ..
        } = *search;
        self.append_segment_search(
            results,
            query_pool,
            |seg| {
                seg.search_binary_vectors_u32(
                    field,
                    query_vector,
                    segment_topk,
                    is_linear,
                    delete_bitmap.clone(),
                    filter_expr,
                )
            },
            |writing| {
                writing.search_binary_u32(
                    field,
                    query_vector,
                    segment_topk,
                    delete_bitmap.as_ref(),
                    filter_expr,
                )
            },
        )
    }

    fn append_binary_u64_query_results(
        &self,
        results: &mut Vec<(u64, f32)>,
        field: &str,
        query_vector: &[u64],
        search: &SegmentSearch<'_>,
    ) -> ZResult<()> {
        let SegmentSearch {
            segment_topk,
            is_linear,
            filter_expr,
            delete_bitmap,
            query_pool,
            ..
        } = *search;
        self.append_segment_search(
            results,
            query_pool,
            |seg| {
                seg.search_binary_vectors_u64(
                    field,
                    query_vector,
                    segment_topk,
                    is_linear,
                    delete_bitmap.clone(),
                    filter_expr,
                )
            },
            |writing| {
                writing.search_binary_u64(
                    field,
                    query_vector,
                    segment_topk,
                    delete_bitmap.as_ref(),
                    filter_expr,
                )
            },
        )
    }

    fn append_dense_query_results(
        &self,
        results: &mut Vec<(u64, f32)>,
        target: &DenseTarget<'_>,
        search: &SegmentSearch<'_>,
    ) -> ZResult<()> {
        let search_one = |seg: &Arc<PersistedSegment>| {
            seg.search_vectors(
                target.field,
                target.query_vector,
                search.ann(),
                target.metric,
            )
        };
        // Hold the persisted list while reading the writing segment so a concurrent flush cannot hide docs.
        let segs = self.persisted_segments.read();
        let segment_results: Vec<ZResult<Vec<(u64, f32)>>> = if segs.len() <= 1 {
            segs.iter().map(search_one).collect()
        } else {
            search
                .query_pool
                .install(|| segs.par_iter().map(search_one).collect())
        };
        extend_segment_results(results, segment_results)?;

        let writing = self.writing_segment.read();
        results.extend(writing.search_dense_vectors(
            target.field,
            target.query_vector,
            search.segment_topk,
            search.delete_bitmap.as_ref(),
            search.filter_expr,
        )?);

        Ok(())
    }

    fn refine_persisted_distances(
        segs: &[Arc<PersistedSegment>],
        refined: &mut HashMap<u64, f32>,
        candidate_ids: &[u64],
        target: &DenseTarget<'_>,
    ) -> ZResult<()> {
        for seg in segs {
            let ids: Vec<u64> = candidate_ids
                .iter()
                .cloned()
                .filter(|&id| id >= seg.min_doc_id && id <= seg.max_doc_id)
                .collect();
            if ids.is_empty() {
                continue;
            }
            let dists = seg.compute_dense_distances_by_doc_ids(
                target.field,
                target.query_vector,
                target.metric,
                &ids,
            )?;
            for (id, dist) in ids.into_iter().zip(dists.into_iter()) {
                if let Some(dist) = dist {
                    refined.insert(id, dist);
                }
            }
        }
        Ok(())
    }

    fn refine_writing_distances(
        &self,
        refined: &mut HashMap<u64, f32>,
        candidate_ids: &[u64],
        target: &DenseTarget<'_>,
    ) {
        use crate::segment::persisted::compute_distance;

        let writing = self.writing_segment.read();
        for &id in candidate_ids {
            if refined.contains_key(&id) {
                continue;
            }
            let Some(doc) = writing.get_doc(id) else {
                continue;
            };
            if let Some(vector) = doc.get_vec_f32(target.field) {
                refined.insert(
                    id,
                    compute_distance(vector, target.query_vector, target.metric),
                );
            }
        }
    }

    fn refine_dense_results(
        &self,
        results: &mut Vec<(u64, f32)>,
        target: &DenseTarget<'_>,
        refiner_k: u32,
    ) -> ZResult<()> {
        results.truncate(refiner_k as usize);
        let candidate_ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();

        let mut refined: HashMap<u64, f32> = HashMap::with_capacity(candidate_ids.len());
        // Hold the persisted list while reading the writing segment so a concurrent flush cannot hide docs.
        let segs = self.persisted_segments.read();
        Self::refine_persisted_distances(&segs, &mut refined, &candidate_ids, target)?;
        self.refine_writing_distances(&mut refined, &candidate_ids, target);
        drop(segs);

        for (id, dist) in results.iter_mut() {
            if let Some(refined_dist) = refined.get(id) {
                *dist = *refined_dist;
            }
        }
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(())
    }

    fn finalize_vector_results(
        &self,
        results: &mut Vec<(u64, f32)>,
        target: &DenseTarget<'_>,
        refinement: &ResultRefinement,
    ) -> ZResult<FinalizedResults> {
        let ResultRefinement {
            use_refiner,
            refiner_k,
            radius,
            topk,
        } = *refinement;
        // Global topk merge
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut refine_duration = None;
        if use_refiner && !results.is_empty() {
            let refine_start = std::time::Instant::now();
            self.refine_dense_results(results, target, refiner_k)?;
            refine_duration = Some(refine_start.elapsed());
        }

        if let Some(r) = radius {
            results.retain(|(_, d)| *d <= r);
        }
        let candidate_count = results.len();
        results.truncate(topk);

        Ok(FinalizedResults {
            candidate_count,
            refine_duration,
        })
    }

    pub(super) fn query_impl(
        &self,
        query: VectorQuery,
        mut profile: Option<&mut QueryProfile>,
    ) -> ZResult<Vec<Arc<Doc>>> {
        let mut query = query;
        let prepare_start = std::time::Instant::now();

        if query.topk > MAX_QUERY_TOPK {
            return Err(Status::invalid_argument(format!(
                "query validate failed: topk[{}] is too large, max is {}",
                query.topk, MAX_QUERY_TOPK
            )));
        }

        let version = self.cur_version();
        let output_selection =
            QueryOutputSelection::prepare(&version.schema, &mut query, MAX_OUTPUT_FIELDS)?;

        let filter_expr = prepare_filter_expr(&version.schema, query.filter.as_deref())?;
        // Held until the scan ends so an upsert's tombstone and new version are seen together.
        let delete_store = self.delete_store.read();
        let delete_bitmap = delete_store.bitmap();

        if let Some(p) = profile.as_deref_mut() {
            p.prepare = prepare_start.elapsed();
            // Vector queries always consider the writing segment plus any persisted segments.
            p.segments_searched = self.persisted_segments.read().len().saturating_add(1);
        }

        let topk = query.topk;

        if query.id.is_some() {
            let resolve_start = std::time::Instant::now();
            self.resolve_query_id_payload(&mut query, &version.schema)?;
            if let Some(p) = profile.as_deref_mut() {
                p.resolve_id = resolve_start.elapsed();
            }
        }

        // Treat the query as filter-only unless it includes an actual query-vector
        // payload. This preserves behavior that allows field_name to be set while
        // query_vector is empty.
        if !has_query_vector_payload(&query) {
            return self.query_filter_only(
                &version.schema,
                query,
                filter_expr.as_ref(),
                delete_bitmap.as_ref(),
                profile,
            );
        }

        let prepared = PreparedVectorQuery {
            schema: &version.schema,
            output_selection: &output_selection,
            filter_expr: filter_expr.as_ref(),
            delete_bitmap: &delete_bitmap,
            topk,
        };
        self.run_vector_query(query, &prepared, profile)
    }

    fn run_vector_query(
        &self,
        query: VectorQuery,
        prepared: &PreparedVectorQuery<'_>,
        mut profile: Option<&mut QueryProfile>,
    ) -> ZResult<Vec<Arc<Doc>>> {
        let mut query = query;
        let field = query.field_name.trim();
        if field.is_empty() {
            return Err(Status::invalid_argument(format!(
                "query validate failed: vector_field[{}] not defined in the collection schema",
                query.field_name
            )));
        }

        let vector_field = resolve_vector_search_field(prepared.schema, field)?;
        let topk = prepared.topk;
        let params = VectorSearchParams::new(&query, &vector_field, topk);

        let mut all_results: Vec<(u64, f32)> = Vec::new();
        let search_start = std::time::Instant::now();
        let maybe_query_pool = Self::build_thread_pool(query.query_params.concurrency)?;
        let query_pool = maybe_query_pool
            .as_ref()
            .unwrap_or_else(|| crate::config::query_pool());

        validate_and_normalize_query_payload(
            &vector_field,
            &mut query.query_vector,
            &query.query_vector_u32,
            &query.query_vector_u64,
            &mut query.sparse_indices,
            &mut query.sparse_values,
        )?;

        self.append_query_results(
            &mut all_results,
            &query,
            &vector_field,
            params.bf_pks.as_deref(),
            &SegmentSearch {
                segment_topk: params.refinement.segment_topk(),
                ef_or_nprobe: params.ef_or_nprobe,
                is_linear: params.is_linear,
                filter_expr: prepared.filter_expr,
                delete_bitmap: prepared.delete_bitmap,
                query_pool,
            },
        )?;

        let target = DenseTarget {
            field: vector_field.name,
            query_vector: &query.query_vector,
            metric: vector_field.metric,
        };
        let finalized =
            self.finalize_vector_results(&mut all_results, &target, &params.refinement)?;
        if let Some(p) = profile.as_deref_mut() {
            if let Some(refine_duration) = finalized.refine_duration {
                p.refine = refine_duration;
            }
            p.search = search_start.elapsed();
            p.candidate_count = finalized.candidate_count;
        }

        let materialize_start = std::time::Instant::now();
        let output = self.materialize_vector_query_results(
            &all_results,
            &query,
            vector_field.metric,
            prepared.output_selection,
        )?;

        if let Some(p) = profile {
            p.materialize = materialize_start.elapsed();
        }

        Ok(output)
    }
}

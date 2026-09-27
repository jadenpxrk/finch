use std::sync::{Arc, OnceLock};

use finch_types::{DataType, MetricType, Status, VectorQuery, ZResult};
use rayon::prelude::*;

use crate::segment::persisted::{AnnSearch, PersistedSegment};

use super::Collection;

impl Collection {
    pub fn query_int_ids(&self, query: VectorQuery) -> ZResult<Vec<i64>> {
        if let Some(ids) = self.try_query_int_ids_fast(&query)? {
            return Ok(ids);
        }

        let mut query = query;
        query.include_vector = false;
        query.include_doc_id = false;
        query.output_fields = Some(Vec::new());
        self.query(query)?
            .into_iter()
            .map(|doc| {
                doc.pk.parse::<i64>().map_err(|e| {
                    Status::invalid_argument(format!("query result pk is not an integer: {}", e))
                })
            })
            .collect()
    }

    fn try_query_int_ids_fast(&self, query: &VectorQuery) -> ZResult<Option<Vec<i64>>> {
        const MAX_QUERY_TOPK: usize = 1024;

        if query.topk > MAX_QUERY_TOPK {
            return Err(Status::invalid_argument(format!(
                "query validate failed: topk[{}] is too large, max is {}",
                query.topk, MAX_QUERY_TOPK
            )));
        }
        if !is_plain_dense_query(query) {
            return Ok(None);
        }

        let version = self.cur_version();
        let field = query.field_name.trim();
        if field.is_empty() {
            return Err(Status::invalid_argument(format!(
                "query validate failed: vector_field[{}] not defined in the collection schema",
                query.field_name
            )));
        }

        let field_schema = version.schema.get_field(field).ok_or_else(|| {
            Status::invalid_argument(format!(
                "query validate failed: vector_field[{}] not defined in the collection schema",
                field
            ))
        })?;
        if field_schema.data_type != DataType::VectorFp32 {
            return Ok(None);
        }
        let dim = field_schema.dimension.unwrap_or(0);
        if dim == 0 || query.query_vector.len() != dim {
            return Err(Status::invalid_argument(
                "query validate failed: dimension is invalid",
            ));
        }

        let metric = field_schema
            .index_params
            .as_ref()
            .and_then(|p| p.metric())
            .unwrap_or(finch_types::MetricType::InnerProduct);
        let delete_store = self.delete_store.read();
        let delete_bitmap = delete_store.bitmap();
        let topk = query.topk;
        let mut all_results: Vec<(u64, f32)> = Vec::new();
        let search = FastDenseSearch {
            query,
            field,
            metric,
            delete_bitmap: &delete_bitmap,
        };
        // Hold the persisted list while reading the writing segment so a concurrent flush cannot hide docs.
        let segs = self.persisted_segments.read();
        Self::append_fast_persisted_results(&segs, &search, &mut all_results)?;
        {
            let writing = self.writing_segment.read();
            all_results.extend(writing.search_dense_vectors(
                field,
                &query.query_vector,
                topk,
                delete_bitmap.as_ref(),
                None,
            )?);
        }
        drop(segs);

        all_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        all_results.truncate(topk);
        let doc_ids: Vec<u64> = all_results.iter().map(|(id, _)| *id).collect();
        self.fetch_i64_pks_by_ids_ordered(&doc_ids).map(Some)
    }

    fn append_fast_persisted_results(
        segs: &[Arc<PersistedSegment>],
        search: &FastDenseSearch<'_>,
        all_results: &mut Vec<(u64, f32)>,
    ) -> ZResult<()> {
        let query = search.query;
        let ef_or_nprobe = query.query_params.ef.or(query.query_params.n_probe);
        let topk = query.topk;
        all_results.reserve(topk.saturating_mul(segs.len().saturating_add(1)));

        let search_segment = |seg: &Arc<PersistedSegment>| {
            seg.search_vectors(
                search.field,
                &query.query_vector,
                AnnSearch {
                    topk,
                    ef_or_nprobe,
                    force_linear: false,
                    delete_bitmap: search.delete_bitmap.clone(),
                    filter_expr: None,
                },
                search.metric,
            )
        };

        let segment_results: Vec<ZResult<Vec<(u64, f32)>>> = if segs.len() <= 1 {
            segs.iter().map(search_segment).collect()
        } else if query.query_params.concurrency.is_none() {
            fast_query_segment_pool().install(|| segs.par_iter().map(search_segment).collect())
        } else {
            let maybe_query_pool = Self::build_thread_pool(query.query_params.concurrency)?;
            let query_pool = maybe_query_pool
                .as_ref()
                .unwrap_or_else(|| crate::config::query_pool());
            query_pool.install(|| segs.par_iter().map(search_segment).collect())
        };
        for res in segment_results {
            all_results.extend(res?);
        }
        Ok(())
    }
}

struct FastDenseSearch<'a> {
    query: &'a VectorQuery,
    field: &'a str,
    metric: MetricType,
    delete_bitmap: &'a Arc<roaring::RoaringTreemap>,
}

// Plain dense top-k with no id lookup, filter, brute-force keys, refine, or radius.
fn is_plain_dense_query(query: &VectorQuery) -> bool {
    !(query.id.is_some()
        || query.query_vector.is_empty()
        || !query.query_vector_u32.is_empty()
        || !query.query_vector_u64.is_empty()
        || !query.sparse_indices.is_empty()
        || !query.sparse_values.is_empty()
        || query
            .filter
            .as_deref()
            .is_some_and(|f| !f.trim().is_empty())
        || query.query_params.bf_pks.is_some()
        || query.query_params.effective_is_linear()
        || query.query_params.use_refiner
        || query.query_params.effective_radius().is_some())
}

fn fast_query_segment_pool() -> &'static rayon::ThreadPool {
    static FAST_QUERY_SEGMENT_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    FAST_QUERY_SEGMENT_POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("failed to build finch fast query segment pool")
    })
}

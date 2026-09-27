use std::sync::Arc;

use finch_types::{CollectionSchema, Doc, Status, VectorQuery, ZResult};

use crate::profile::QueryProfile;
use crate::query_filter::prepare_filter_expr;
use crate::query_output::QueryOutputSelection;
use crate::sqlengine::parser::FilterExpr;
use crate::vector_normalization::has_query_vector_payload;

use super::{Collection, MAX_OUTPUT_FIELDS};

impl Collection {
    /// Scan every document matching a scalar filter.
    ///
    /// Unlike `query`, this is not a ranked top-k operation and is intended for
    /// correctness-sensitive maintenance work that must not silently truncate.
    pub fn scan_filter_only(&self, mut query: VectorQuery) -> ZResult<Vec<Arc<Doc>>> {
        let version = self.cur_version();
        let output_selection =
            QueryOutputSelection::prepare(&version.schema, &mut query, MAX_OUTPUT_FIELDS)?;
        if has_query_vector_payload(&query) {
            return Err(Status::invalid_argument(
                "scan validate failed: scan_filter_only does not accept query vectors",
            ));
        }
        let filter_expr = prepare_filter_expr(&version.schema, query.filter.as_deref())?;
        let delete_store = self.delete_store.read();
        let matched_ids = self.collect_filter_only_doc_ids(
            filter_expr.as_ref(),
            &delete_store.bitmap(),
            usize::MAX,
        )?;
        // Held through materialization so a concurrent column rename cannot split schema and data.
        let docs = self.materialize_filter_only_results(&matched_ids, &query, &output_selection);
        drop(delete_store);
        docs
    }

    fn collect_filter_only_doc_ids(
        &self,
        filter_expr: Option<&FilterExpr>,
        delete_bitmap: &roaring::RoaringTreemap,
        topk: usize,
    ) -> ZResult<Vec<u64>> {
        let mut matched_ids: Vec<u64> = Vec::new();

        // Scan persisted segments first (older doc_ids), then the writing segment.
        // Hold the persisted list while reading the writing segment so a concurrent flush cannot hide docs.
        let segs = self.persisted_segments.read();
        for seg in segs.iter() {
            if matched_ids.len() >= topk {
                break;
            }
            let remain = topk - matched_ids.len();
            let mut ids = seg.scan_filter_ids_limit(filter_expr, delete_bitmap, remain)?;
            matched_ids.append(&mut ids);
        }

        if matched_ids.len() < topk {
            let writing = self.writing_segment.read();
            let limit = topk - matched_ids.len();
            let mut ids = writing.scan_filter_ids_limit(filter_expr, delete_bitmap, limit);
            matched_ids.append(&mut ids);
        }
        drop(segs);

        matched_ids.sort_unstable();
        matched_ids.dedup();
        matched_ids.truncate(topk);

        Ok(matched_ids)
    }

    pub(super) fn query_filter_only(
        &self,
        schema: &CollectionSchema,
        query: VectorQuery,
        filter_expr: Option<&FilterExpr>,
        delete_bitmap: &roaring::RoaringTreemap,
        profile: Option<&mut QueryProfile>,
    ) -> ZResult<Vec<Arc<Doc>>> {
        let start = std::time::Instant::now();
        let mut query = query;
        let output_selection =
            QueryOutputSelection::prepare(schema, &mut query, MAX_OUTPUT_FIELDS)?;

        if has_query_vector_payload(&query) {
            return Err(Status::invalid_argument(
                "query validate failed: filter-only query must not include query vectors",
            ));
        }

        let out = if query.topk == 0 {
            Vec::new()
        } else {
            let matched_ids =
                self.collect_filter_only_doc_ids(filter_expr, delete_bitmap, query.topk)?;
            self.materialize_filter_only_results(&matched_ids, &query, &output_selection)?
        };
        if let Some(p) = profile {
            p.filter_only = start.elapsed();
            p.result_count = out.len();
        }
        Ok(out)
    }
}

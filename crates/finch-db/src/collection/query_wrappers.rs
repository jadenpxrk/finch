use std::sync::Arc;

use finch_types::{Doc, GroupByVectorQuery, GroupResult, Status, VectorQuery, ZResult};

use crate::group_by::GroupByPlan;
use crate::profile::QueryProfile;
use crate::sql_query::SqlQueryPlan;
use crate::vector_normalization::has_query_vector_payload;

use super::{Collection, MAX_QUERY_TOPK};

impl Collection {
    pub fn query(&self, query: VectorQuery) -> ZResult<Vec<Arc<Doc>>> {
        self.query_impl(query, None)
    }

    pub fn query_profiled(&self, query: VectorQuery) -> ZResult<(Vec<Arc<Doc>>, QueryProfile)> {
        let start = std::time::Instant::now();
        let mut profile = QueryProfile::default();
        let docs = self.query_impl(query, Some(&mut profile))?;
        profile.total = start.elapsed();
        profile.result_count = docs.len();
        Ok((docs, profile))
    }

    /// Execute a SQL SELECT statement against this collection.
    ///
    /// Supported syntax:
    /// `SELECT <fields|*> FROM <table> [WHERE <expr>] [ORDER BY ...] [LIMIT n]`
    ///
    /// Notes:
    /// - `ORDER BY` is applied to the returned result set (after vector recall, if any).
    /// - Vector search is supported via a single vector condition in WHERE:
    ///   `<vector_field> = [1,2,3]` (or `[[1,2,3]]`), which must not be under an `OR` ancestor.
    ///   Multi-row matrix literals (e.g. `[[...],[...]]`) are interpreted as a single
    ///   flattened query vector by concatenating the inner vectors (row-major).
    pub fn query_sql(&self, sql: &str) -> ZResult<Vec<Arc<Doc>>> {
        let version = self.cur_version();
        let (query, projection) = SqlQueryPlan::prepare(sql, &version.schema)?.into_parts();
        let docs = self.query(query)?;

        if projection.needs_projection() {
            let row_locator = self.row_locator();
            projection.apply(docs, &row_locator)
        } else {
            Ok(docs)
        }
    }

    /// Group-by vector query: returns groups of nearest neighbors, one group per distinct value
    pub fn group_by_query(&self, query: GroupByVectorQuery) -> ZResult<Vec<GroupResult>> {
        let docs_per_group = query.group_count;
        let group_limit = query.group_topk;

        if docs_per_group == 0 || group_limit == 0 {
            return Ok(Vec::new());
        }

        if query.base.field_name.trim().is_empty() || !has_query_vector_payload(&query.base) {
            return Err(Status::invalid_argument("group by should has vector query"));
        }

        let version = self.cur_version();
        let (plan, base_query) = GroupByPlan::prepare(
            &version.schema,
            query.base,
            query.group_by_field,
            docs_per_group,
            group_limit,
        )?;

        let row_locator = if plan.needs_row_locator() {
            Some(self.row_locator())
        } else {
            None
        };

        // One group can crowd out the rest of the candidates, so widen the window until every
        // group is filled or the query returns fewer candidates than asked for.
        let mut base_query = base_query;
        loop {
            let topk = base_query.topk;
            let results = self.query(base_query.clone())?;
            let exhausted = results.len() < topk || topk >= MAX_QUERY_TOPK;
            let groups = plan.build_results(results, row_locator.as_ref())?;
            if exhausted || plan.is_filled(&groups) {
                return Ok(groups);
            }
            base_query.topk = topk.saturating_mul(2).min(MAX_QUERY_TOPK);
        }
    }
}

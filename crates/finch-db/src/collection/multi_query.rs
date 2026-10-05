use std::collections::HashMap;
use std::sync::Arc;

use finch_core::mixed_reducer::{RrfReducer, WeightedReducer};
use finch_types::{Doc, MetricType, VectorQuery, ZResult};

use crate::multi_query::MultiQuerySetup;

use super::Collection;

impl Collection {
    /// Execute multiple vector queries and fuse results using Reciprocal Rank Fusion (RRF).
    ///
    /// This is a native alternative to the Python-side QueryExecutor+Reranker flow and is
    /// primarily intended for Rust/Node callers.
    pub fn query_multi_rrf(
        &self,
        queries: Vec<VectorQuery>,
        topk: usize,
        rank_constant: u32,
    ) -> ZResult<Vec<Arc<Doc>>> {
        let setup = MultiQuerySetup::new(queries)?;
        let output = setup.output().clone();

        let lists = setup.collect_pk_lists(topk, |query| self.query(query))?;
        let fused = RrfReducer { rank_constant }.fuse(&lists);
        let fused = fused.into_iter().take(topk).collect::<Vec<_>>();

        self.materialize_fused_docs(
            fused,
            output.include_vector,
            output.include_doc_id,
            output.output_fields,
        )
    }

    /// Execute multiple vector queries and fuse results using a normalized weighted sum.
    pub fn query_multi_weighted(
        &self,
        queries: Vec<VectorQuery>,
        topk: usize,
        metric: MetricType,
        weights: HashMap<String, f32>,
    ) -> ZResult<Vec<Arc<Doc>>> {
        let setup = MultiQuerySetup::new(queries)?;
        let output = setup.output().clone();
        let reducer = WeightedReducer { metric, weights };

        let scored_lists = setup.collect_scored_pk_lists(topk, |query| self.query(query))?;
        let fused = reducer.fuse(&scored_lists);
        let fused = fused.into_iter().take(topk).collect::<Vec<_>>();

        self.materialize_fused_docs(
            fused,
            output.include_vector,
            output.include_doc_id,
            output.output_fields,
        )
    }
}

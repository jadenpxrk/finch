use super::*;

impl WritingSegment {
    pub(super) fn invert_prefilter_plan(&self, expr: &FilterExpr) -> ZResult<InvertPrefilterPlan> {
        let max_range_hits = max_range_hits(self.doc_count());
        plan_for(
            expr,
            &|field, lookup| match self.invert_indexes.get(field) {
                Some(idx) => lookup.run(idx, max_range_hits),
                None => Ok(None),
            },
        )
    }
}

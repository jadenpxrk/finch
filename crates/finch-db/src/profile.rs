use std::time::Duration;

#[derive(Debug, Clone, Default)]
pub struct QueryProfile {
    pub prepare: Duration,
    pub resolve_id: Duration,
    pub filter_only: Duration,
    pub search: Duration,
    pub refine: Duration,
    pub materialize: Duration,
    pub total: Duration,

    pub segments_searched: usize,
    pub candidate_count: usize,
    pub result_count: usize,
}

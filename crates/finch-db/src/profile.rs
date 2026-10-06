use std::time::Duration;

/// The time and the counts of each phase of one query.
#[derive(Debug, Clone, Default)]
pub struct QueryProfile {
    /// Time to validate the query and parse its filter and output fields.
    pub prepare: Duration,
    /// Time to load the query vector of a query by document id.
    pub resolve_id: Duration,
    /// Time to run a query that has no query vector.
    pub filter_only: Duration,
    /// Time to search the segments for nearest neighbors, refinement included.
    pub search: Duration,
    /// Time to recompute the distances of the candidates at full precision.
    pub refine: Duration,
    /// Time to load the output fields of the results.
    pub materialize: Duration,
    /// Time for the whole query.
    pub total: Duration,

    /// Segments the query searched, the writing segment included.
    pub segments_searched: usize,
    /// Candidates the segment searches returned before the final top-k.
    pub candidate_count: usize,
    /// Documents the query returned.
    pub result_count: usize,
}

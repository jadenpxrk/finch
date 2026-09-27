use std::sync::Arc;

use crate::segment::persisted::PersistedSegment;

struct PersistedRowRange {
    min_doc_id: u64,
    max_doc_id: u64,
    segment: Arc<PersistedSegment>,
}

pub(crate) struct RowLocator {
    persisted_ranges: Vec<PersistedRowRange>,
    writing_min_doc_id: u64,
    writing_max_doc_id: u64,
}

impl RowLocator {
    pub(crate) fn new(
        persisted_segments: &[Arc<PersistedSegment>],
        writing_min_doc_id: u64,
        writing_max_doc_id: u64,
    ) -> Self {
        let mut persisted_ranges: Vec<PersistedRowRange> = persisted_segments
            .iter()
            .map(|segment| PersistedRowRange {
                min_doc_id: segment.min_doc_id,
                max_doc_id: segment.max_doc_id,
                segment: segment.clone(),
            })
            .collect();
        persisted_ranges.sort_by_key(|range| range.min_doc_id);

        Self {
            persisted_ranges,
            writing_min_doc_id,
            writing_max_doc_id,
        }
    }

    pub(crate) fn row_id(&self, doc_id: u64) -> Option<u64> {
        let mut lo = 0usize;
        let mut hi = self.persisted_ranges.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.persisted_ranges[mid].max_doc_id < doc_id {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        if let Some(range) = self.persisted_ranges.get(lo) {
            if doc_id >= range.min_doc_id && doc_id <= range.max_doc_id {
                return range
                    .segment
                    .forward_store
                    .read()
                    .row_index_of_doc_id(doc_id);
            }
        }

        if self.writing_min_doc_id != u64::MAX
            && doc_id >= self.writing_min_doc_id
            && doc_id <= self.writing_max_doc_id
        {
            return Some(doc_id.saturating_sub(self.writing_min_doc_id));
        }

        None
    }
}

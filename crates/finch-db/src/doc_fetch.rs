use std::collections::HashMap;
use std::sync::Arc;

use finch_types::{Doc, Status, ZResult};
use parking_lot::RwLock;

use crate::segment::persisted::PersistedSegment;
use crate::segment::writing::WritingSegment;

pub(crate) fn fetch_docs_by_ids(
    writing_segment: &RwLock<WritingSegment>,
    persisted_segments: &RwLock<Vec<Arc<PersistedSegment>>>,
    doc_ids: &[u64],
) -> ZResult<HashMap<u64, Doc>> {
    let mut result = HashMap::new();

    {
        let writing = writing_segment.read();
        for &id in doc_ids {
            if let Some(doc) = writing.get_doc(id) {
                result.insert(id, doc);
            }
        }
    }

    let missing: Vec<u64> = doc_ids
        .iter()
        .filter(|&&id| !result.contains_key(&id))
        .cloned()
        .collect();

    if !missing.is_empty() {
        let segs = persisted_segments.read();
        for seg in segs.iter() {
            let still_missing: Vec<u64> = missing
                .iter()
                .filter(|&&id| {
                    !result.contains_key(&id) && id >= seg.min_doc_id && id <= seg.max_doc_id
                })
                .cloned()
                .collect();

            if still_missing.is_empty() {
                continue;
            }

            let docs = seg.fetch_docs(&still_missing)?;
            for (id, maybe_doc) in still_missing.iter().zip(docs.iter()) {
                if let Some(doc) = maybe_doc {
                    result.insert(*id, doc.clone());
                }
            }
        }
    }

    Ok(result)
}

pub(crate) fn fetch_pks_by_ids(
    writing_segment: &RwLock<WritingSegment>,
    persisted_segments: &RwLock<Vec<Arc<PersistedSegment>>>,
    doc_ids: &[u64],
) -> ZResult<HashMap<u64, String>> {
    let mut result: HashMap<u64, String> = HashMap::new();

    {
        let writing = writing_segment.read();
        for &id in doc_ids {
            if let Some(doc) = writing.get_doc(id) {
                result.insert(id, doc.pk);
            }
        }
    }

    let missing: Vec<u64> = doc_ids
        .iter()
        .filter(|&&id| !result.contains_key(&id))
        .cloned()
        .collect();

    if missing.is_empty() {
        return Ok(result);
    }

    let segs = persisted_segments.read();
    for seg in segs.iter() {
        let ids: Vec<u64> = missing
            .iter()
            .filter(|&&id| id >= seg.min_doc_id && id <= seg.max_doc_id)
            .cloned()
            .collect();

        if ids.is_empty() {
            continue;
        }

        let pks = seg.forward_store.read().get_pks_by_doc_ids(&ids)?;
        for (id, pk) in ids.into_iter().zip(pks.into_iter()) {
            if let Some(pk) = pk {
                result.insert(id, pk);
            }
        }
    }

    Ok(result)
}

pub(crate) fn fetch_i64_pks_by_ids_ordered(
    writing_segment: &RwLock<WritingSegment>,
    persisted_segments: &RwLock<Vec<Arc<PersistedSegment>>>,
    doc_ids: &[u64],
) -> ZResult<Vec<i64>> {
    let mut result: Vec<Option<i64>> = vec![None; doc_ids.len()];

    {
        let writing = writing_segment.read();
        for (pos, &id) in doc_ids.iter().enumerate() {
            if let Some(doc) = writing.get_doc(id) {
                result[pos] = Some(doc.pk.parse::<i64>().map_err(|e| {
                    Status::invalid_argument(format!("primary key is not an integer: {}", e))
                })?);
            }
        }
    }

    let segs = persisted_segments.read();
    for seg in segs.iter() {
        let ids: Vec<(usize, u64)> = doc_ids
            .iter()
            .enumerate()
            .filter(|(pos, id)| {
                result[*pos].is_none() && **id >= seg.min_doc_id && **id <= seg.max_doc_id
            })
            .map(|(pos, id)| (pos, *id))
            .collect();

        if ids.is_empty() {
            continue;
        }

        let raw_ids: Vec<u64> = ids.iter().map(|(_, id)| *id).collect();
        let pks = seg.forward_store.read().get_i64_pks_by_doc_ids(&raw_ids)?;
        for ((pos, _), pk) in ids.into_iter().zip(pks.into_iter()) {
            if let Some(pk) = pk {
                result[pos] = Some(pk);
            }
        }
    }

    Ok(result.into_iter().flatten().collect())
}

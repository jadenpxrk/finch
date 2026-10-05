use super::*;

impl InMemoryForwardStore {
    pub(super) fn row_index_of_doc_id(&self, doc_id: u64) -> Option<u64> {
        let (batch_idx, row) = locate_in_batches(&self.batches, &self.batch_doc_id_ranges, doc_id)?;
        let base = *self.batch_row_offsets.get(batch_idx)?;
        Some(base.saturating_add(row) as u64)
    }

    pub(super) fn scan_rows<F>(&self, mut f: F) -> ZResult<()>
    where
        F: FnMut(u64, &RecordBatch, usize) -> ZResult<()>,
    {
        for batch in &self.batches {
            scan_batch_rows(batch, &mut f)?;
        }
        Ok(())
    }

    pub(super) fn scan_large_binary_column_rows<F>(&self, col_name: &str, mut f: F) -> ZResult<()>
    where
        F: FnMut(u64, Option<&[u8]>) -> ZResult<()>,
    {
        require_column(self.has_column(col_name), col_name)?;
        for batch in &self.batches {
            scan_large_binary_batch_rows(batch, col_name, &mut f)?;
        }
        Ok(())
    }

    pub(super) fn has_column(&self, name: &str) -> bool {
        self.batches
            .first()
            .map(|b| b.schema().index_of(name).is_ok())
            .unwrap_or(false)
    }

    pub(super) fn get_by_doc_ids(&self, ids: &[u64]) -> ZResult<Vec<Option<Doc>>> {
        Ok(ids
            .iter()
            .map(|&id| {
                let (batch_idx, row) =
                    locate_in_batches(&self.batches, &self.batch_doc_id_ranges, id)?;
                Some(MmapForwardStore::extract_doc_from_batch(
                    &self.batches[batch_idx],
                    row,
                    id,
                ))
            })
            .collect())
    }

    pub(super) fn compute_dense_distance_by_doc_ids(
        &self,
        query: &DenseDistanceQuery,
        ids: &[u64],
    ) -> ZResult<Vec<Option<f32>>> {
        Ok(ids
            .iter()
            .map(|&id| {
                let (batch_idx, row) =
                    locate_in_batches(&self.batches, &self.batch_doc_id_ranges, id)?;
                query.distance_at(&self.batches[batch_idx], row)
            })
            .collect())
    }

    pub(super) fn get_pks_by_doc_ids(&self, ids: &[u64]) -> ZResult<Vec<Option<String>>> {
        Ok(ids
            .iter()
            .map(|&id| {
                let (batch_idx, row) =
                    locate_in_batches(&self.batches, &self.batch_doc_id_ranges, id)?;
                let batch = &self.batches[batch_idx];
                batch
                    .column_by_name("__pk__")
                    .and_then(|col| col.as_any().downcast_ref::<StringArray>())
                    .and_then(|arr| (!arr.is_null(row)).then(|| arr.value(row).to_string()))
            })
            .collect())
    }

    pub(super) fn get_i64_pks_by_doc_ids(&self, ids: &[u64]) -> ZResult<Vec<Option<i64>>> {
        ids.iter()
            .map(|&id| {
                let Some((batch_idx, row)) =
                    locate_in_batches(&self.batches, &self.batch_doc_id_ranges, id)
                else {
                    return Ok(None);
                };
                let batch = &self.batches[batch_idx];
                let Some(arr) = batch
                    .column_by_name("__pk__")
                    .and_then(|col| col.as_any().downcast_ref::<StringArray>())
                else {
                    return Ok(None);
                };
                if arr.is_null(row) {
                    return Ok(None);
                }
                arr.value(row).parse::<i64>().map(Some).map_err(|e| {
                    Status::invalid_argument(format!("primary key is not an integer: {}", e))
                })
            })
            .collect()
    }

    pub(super) fn all_doc_ids(&self) -> Vec<u64> {
        let mut out: Vec<u64> = Vec::with_capacity(self.doc_count);
        for batch in &self.batches {
            let Some(col) = batch.column_by_name("__doc_id__") else {
                continue;
            };
            let Some(id_col) = col.as_any().downcast_ref::<UInt64Array>() else {
                continue;
            };
            out.extend(id_col.values().iter().copied());
        }
        out
    }

    pub(super) fn len(&self) -> usize {
        self.doc_count
    }
}

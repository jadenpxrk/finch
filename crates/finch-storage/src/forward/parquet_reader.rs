use super::*;

impl RowGroupCache {
    pub(super) fn locate(&self, doc_id: u64) -> Option<(usize, usize)> {
        locate_in_batches(&self.batches, &self.batch_doc_id_ranges, doc_id)
    }
}

impl ParquetBufferedForwardStore {
    pub(super) fn has_column(&self, name: &str) -> bool {
        self.schema.index_of(name).is_ok()
    }

    pub(super) fn find_row_group(&self, doc_id: u64) -> Option<usize> {
        find_range_index(&self.row_group_doc_id_ranges, doc_id)
    }

    pub(super) fn get_or_load_row_group(
        &self,
        row_group_idx: usize,
    ) -> ZResult<Arc<RowGroupCache>> {
        // Fast path: cache hit.
        if let Ok(mut g) = self.cache.lock() {
            if let Some(v) = g.get(row_group_idx) {
                return Ok(v);
            }
        }

        // Miss: load without holding the mutex.
        let loaded = self.load_row_group(row_group_idx)?;

        if let Ok(mut g) = self.cache.lock() {
            // Double-check in case another thread won the race.
            if let Some(v) = g.get(row_group_idx) {
                return Ok(v);
            }
            g.insert(row_group_idx, loaded.clone());
        }

        Ok(loaded)
    }

    pub(super) fn load_row_group(&self, row_group_idx: usize) -> ZResult<Arc<RowGroupCache>> {
        let file = File::open(&self.path).map_err(|e| Status::io_error(e.to_string()))?;
        let mut builder =
            ParquetRecordBatchReaderBuilder::new_with_metadata(file, self.metadata.clone());
        builder = builder
            .with_row_groups(vec![row_group_idx])
            .with_batch_size(4096);
        let batches = read_parquet_batches(builder)?;

        let (batch_doc_id_ranges, batch_row_offsets, _doc_count) =
            MmapForwardStore::compute_batch_ranges(&batches)?;

        Ok(Arc::new(RowGroupCache {
            batches,
            batch_doc_id_ranges,
            batch_row_offsets,
        }))
    }

    pub(super) fn row_index_of_doc_id_result(&self, doc_id: u64) -> ZResult<Option<u64>> {
        let Some(rg_idx) = self.find_row_group(doc_id) else {
            return Ok(None);
        };
        let cache = self.get_or_load_row_group(rg_idx)?;
        let Some((batch_idx, row)) = cache.locate(doc_id) else {
            return Ok(None);
        };
        let base_rg = *self.row_group_row_offsets.get(rg_idx).unwrap_or(&0);
        let base_batch = *cache.batch_row_offsets.get(batch_idx).unwrap_or(&0);
        Ok(Some(
            base_rg.saturating_add(base_batch).saturating_add(row) as u64
        ))
    }

    pub(super) fn row_index_of_doc_id(&self, doc_id: u64) -> Option<u64> {
        // API is infallible; treat IO errors as "not found".
        self.row_index_of_doc_id_result(doc_id).ok().flatten()
    }

    pub(super) fn scan_rows<F>(&self, mut f: F) -> ZResult<()>
    where
        F: FnMut(u64, &RecordBatch, usize) -> ZResult<()>,
    {
        // Stream through all record batches in order. This is used by compaction/DDL
        // and should be memory-bounded.
        let file = File::open(&self.path).map_err(|e| Status::io_error(e.to_string()))?;
        let mut builder =
            ParquetRecordBatchReaderBuilder::new_with_metadata(file, self.metadata.clone());
        builder = builder.with_batch_size(4096);
        let reader = builder
            .build()
            .map_err(|e| Status::io_error(e.to_string()))?;

        for batch in reader {
            let batch = batch.map_err(|e| Status::io_error(e.to_string()))?;
            scan_batch_rows(&batch, &mut f)?;
        }
        Ok(())
    }

    pub(super) fn scan_large_binary_column_rows<F>(&self, col_name: &str, mut f: F) -> ZResult<()>
    where
        F: FnMut(u64, Option<&[u8]>) -> ZResult<()>,
    {
        require_column(self.has_column(col_name), col_name)?;

        let parquet_schema = self.metadata.parquet_schema();
        let file = File::open(&self.path).map_err(|e| Status::io_error(e.to_string()))?;
        let mut builder =
            ParquetRecordBatchReaderBuilder::new_with_metadata(file, self.metadata.clone());
        builder = builder
            .with_batch_size(4096)
            .with_projection(ProjectionMask::columns(
                parquet_schema,
                ["__doc_id__", col_name],
            ));

        let reader = builder
            .build()
            .map_err(|e| Status::io_error(e.to_string()))?;
        for batch in reader {
            let batch = batch.map_err(|e| Status::io_error(e.to_string()))?;
            scan_large_binary_batch_rows(&batch, col_name, &mut f)?;
        }

        Ok(())
    }

    pub(super) fn get_by_doc_ids(&self, ids: &[u64]) -> ZResult<Vec<Option<Doc>>> {
        let mut out: Vec<Option<Doc>> = vec![None; ids.len()];
        let mut groups: HashMap<usize, Vec<(usize, u64)>> = HashMap::new();

        for (pos, &id) in ids.iter().enumerate() {
            if let Some(rg_idx) = self.find_row_group(id) {
                groups.entry(rg_idx).or_default().push((pos, id));
            }
        }

        for (rg_idx, items) in groups {
            let cache = self.get_or_load_row_group(rg_idx)?;
            for (pos, id) in items {
                if let Some((batch_idx, row)) = cache.locate(id) {
                    let batch = &cache.batches[batch_idx];
                    out[pos] = Some(MmapForwardStore::extract_doc_from_batch(batch, row, id));
                }
            }
        }

        Ok(out)
    }

    pub(super) fn compute_dense_distance_by_doc_ids(
        &self,
        query: &DenseDistanceQuery,
        ids: &[u64],
    ) -> ZResult<Vec<Option<f32>>> {
        let mut out: Vec<Option<f32>> = vec![None; ids.len()];
        let mut groups: HashMap<usize, Vec<(usize, u64)>> = HashMap::new();

        for (pos, &id) in ids.iter().enumerate() {
            if let Some(rg_idx) = self.find_row_group(id) {
                groups.entry(rg_idx).or_default().push((pos, id));
            }
        }

        for (rg_idx, items) in groups {
            let cache = self.get_or_load_row_group(rg_idx)?;
            for (pos, id) in items {
                let Some((batch_idx, row)) = cache.locate(id) else {
                    continue;
                };
                out[pos] = query.distance_at(&cache.batches[batch_idx], row);
            }
        }

        Ok(out)
    }

    pub(super) fn get_pks_by_doc_ids(&self, ids: &[u64]) -> ZResult<Vec<Option<String>>> {
        let mut out: Vec<Option<String>> = vec![None; ids.len()];
        let mut groups: HashMap<usize, Vec<(usize, u64)>> = HashMap::new();

        for (pos, &id) in ids.iter().enumerate() {
            if let Some(rg_idx) = self.find_row_group(id) {
                groups.entry(rg_idx).or_default().push((pos, id));
            }
        }

        for (rg_idx, items) in groups {
            let cache = self.get_or_load_row_group(rg_idx)?;
            for (pos, id) in items {
                let Some((batch_idx, row)) = cache.locate(id) else {
                    continue;
                };
                let batch = &cache.batches[batch_idx];
                if let Some(pk) = batch
                    .column_by_name("__pk__")
                    .and_then(|col| col.as_any().downcast_ref::<StringArray>())
                    .and_then(|arr| (!arr.is_null(row)).then(|| arr.value(row).to_string()))
                {
                    out[pos] = Some(pk);
                }
            }
        }

        Ok(out)
    }

    pub(super) fn get_i64_pks_by_doc_ids(&self, ids: &[u64]) -> ZResult<Vec<Option<i64>>> {
        let mut out: Vec<Option<i64>> = vec![None; ids.len()];
        let mut groups: HashMap<usize, Vec<(usize, u64)>> = HashMap::new();

        for (pos, &id) in ids.iter().enumerate() {
            if let Some(rg_idx) = self.find_row_group(id) {
                groups.entry(rg_idx).or_default().push((pos, id));
            }
        }

        for (rg_idx, items) in groups {
            let cache = self.get_or_load_row_group(rg_idx)?;
            for (pos, id) in items {
                let Some((batch_idx, row)) = cache.locate(id) else {
                    continue;
                };
                let batch = &cache.batches[batch_idx];
                let Some(arr) = batch
                    .column_by_name("__pk__")
                    .and_then(|col| col.as_any().downcast_ref::<StringArray>())
                else {
                    continue;
                };
                if arr.is_null(row) {
                    continue;
                }
                out[pos] = Some(arr.value(row).parse::<i64>().map_err(|e| {
                    Status::invalid_argument(format!("primary key is not an integer: {}", e))
                })?);
            }
        }

        Ok(out)
    }

    pub(super) fn all_doc_ids(&self) -> Vec<u64> {
        let mut out: Vec<u64> = Vec::with_capacity(self.doc_count);
        let parquet_schema = self.metadata.parquet_schema();

        let file = match File::open(&self.path) {
            Ok(f) => f,
            Err(_) => return out,
        };
        let mut builder =
            ParquetRecordBatchReaderBuilder::new_with_metadata(file, self.metadata.clone());
        builder = builder
            .with_batch_size(4096)
            .with_projection(ProjectionMask::columns(parquet_schema, ["__doc_id__"]));

        let reader = match builder.build() {
            Ok(r) => r,
            Err(_) => return out,
        };

        for batch in reader {
            let batch = match batch {
                Ok(b) => b,
                Err(_) => return out,
            };
            if let Some(col) = batch.column_by_name("__doc_id__") {
                if let Some(id_col) = col.as_any().downcast_ref::<UInt64Array>() {
                    out.extend(id_col.values().iter().copied());
                }
            }
        }
        out
    }

    pub(super) fn len(&self) -> usize {
        self.doc_count
    }
}

/// Query norms computed once per distance batch.
#[derive(Clone, Copy)]
pub(super) struct QueryNorms {
    norm: Option<f32>,
    sq_norm: Option<f32>,
}

impl QueryNorms {
    pub(super) fn new(query: &[f32], metric: MetricType) -> Self {
        let sq_norm = if matches!(metric, MetricType::Cosine | MetricType::MipsL2) {
            Some(query.iter().map(|x| x * x).sum::<f32>())
        } else {
            None
        };
        let norm = if matches!(metric, MetricType::Cosine) {
            Some(sq_norm.unwrap_or(0.0).sqrt())
        } else {
            None
        };
        Self { norm, sq_norm }
    }
}

/// A dense-vector distance query bound to its `__vec__` column.
pub(super) struct DenseDistanceQuery<'a> {
    col_name: String,
    query: &'a [f32],
    metric: MetricType,
    norms: QueryNorms,
}

impl<'a> DenseDistanceQuery<'a> {
    pub(super) fn new(field: &str, query: &'a [f32], metric: MetricType) -> Self {
        Self {
            col_name: format!("__vec__{}", field),
            query,
            metric,
            norms: QueryNorms::new(query, metric),
        }
    }

    /// `None` when the column or cell is missing, null, or of another dimension.
    pub(super) fn distance_at(&self, batch: &RecordBatch, row: usize) -> Option<f32> {
        let col = batch.column_by_name(&self.col_name)?;
        let bin = col.as_any().downcast_ref::<LargeBinaryArray>()?;
        if bin.is_null(row) {
            return None;
        }
        let bytes = bin.value(row);
        if bytes.len() != self.query.len() * 4 {
            return None;
        }
        Some(compute_distance_bytes(
            bytes,
            self.query,
            self.metric,
            self.norms,
        ))
    }
}

pub(super) fn compute_distance_bytes(
    bytes: &[u8],
    query: &[f32],
    metric: MetricType,
    norms: QueryNorms,
) -> f32 {
    // Fast path: aligned little-endian f32 slice.
    if cfg!(target_endian = "little")
        && (bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>())
    {
        let n = bytes.len() / 4;
        // SAFETY: alignment checked; length is multiple of 4; backing buffer lives with the Arrow batch.
        let v = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, n) };
        return f32_slice_distance(v, query, metric, norms);
    }
    le_bytes_distance(bytes, query, metric, norms)
}

fn f32_slice_distance(v: &[f32], query: &[f32], metric: MetricType, norms: QueryNorms) -> f32 {
    match metric {
        MetricType::InnerProduct => -v.iter().zip(query.iter()).map(|(x, y)| x * y).sum::<f32>(),
        MetricType::MipsL2 => {
            let mut ip = 0.0f32;
            let mut u2 = 0.0f32;
            for (x, y) in v.iter().zip(query.iter()) {
                ip += x * y;
                u2 += x * x;
            }
            let v2 = norms
                .sq_norm
                .unwrap_or_else(|| query.iter().map(|x| x * x).sum::<f32>());
            let denom = u2.max(v2);
            2.0 - 2.0 * ip / denom
        }
        MetricType::Cosine => {
            let dot = v.iter().zip(query.iter()).map(|(x, y)| x * y).sum::<f32>();
            let nv = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nq = norms
                .norm
                .unwrap_or_else(|| query.iter().map(|x| x * x).sum::<f32>().sqrt());
            let denom = nv * nq;
            if denom == 0.0 {
                1.0
            } else {
                1.0 - dot / denom
            }
        }
        _ => v
            .iter()
            .zip(query.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>(),
    }
}

// Portable fallback: decode per element.
fn le_bytes_distance(bytes: &[u8], query: &[f32], metric: MetricType, norms: QueryNorms) -> f32 {
    match metric {
        MetricType::InnerProduct => {
            let mut dot = 0.0f32;
            for (i, &q) in query.iter().enumerate() {
                let off = i * 4;
                let x = f32::from_le_bytes([
                    bytes[off],
                    bytes[off + 1],
                    bytes[off + 2],
                    bytes[off + 3],
                ]);
                dot += x * q;
            }
            -dot
        }
        MetricType::MipsL2 => {
            let mut ip = 0.0f32;
            let mut u2 = 0.0f32;
            for (i, &q) in query.iter().enumerate() {
                let off = i * 4;
                let x = f32::from_le_bytes([
                    bytes[off],
                    bytes[off + 1],
                    bytes[off + 2],
                    bytes[off + 3],
                ]);
                ip += x * q;
                u2 += x * x;
            }
            let v2 = norms
                .sq_norm
                .unwrap_or_else(|| query.iter().map(|x| x * x).sum::<f32>());
            let denom = u2.max(v2);
            2.0 - 2.0 * ip / denom
        }
        MetricType::Cosine => {
            let mut dot = 0.0f32;
            let mut nv = 0.0f32;
            for (i, &q) in query.iter().enumerate() {
                let off = i * 4;
                let x = f32::from_le_bytes([
                    bytes[off],
                    bytes[off + 1],
                    bytes[off + 2],
                    bytes[off + 3],
                ]);
                dot += x * q;
                nv += x * x;
            }
            let nv = nv.sqrt();
            let nq = norms
                .norm
                .unwrap_or_else(|| query.iter().map(|x| x * x).sum::<f32>().sqrt());
            let denom = nv * nq;
            if denom == 0.0 {
                1.0
            } else {
                1.0 - dot / denom
            }
        }
        _ => {
            let mut acc = 0.0f32;
            for (i, &q) in query.iter().enumerate() {
                let off = i * 4;
                let x = f32::from_le_bytes([
                    bytes[off],
                    bytes[off + 1],
                    bytes[off + 2],
                    bytes[off + 3],
                ]);
                let d = x - q;
                acc += d * d;
            }
            acc
        }
    }
}

pub(super) fn extract_scalar_value(col: &dyn Array, row: usize, dt: &ArrowType) -> Option<Value> {
    use ArrowType as AD;
    match dt {
        AD::Int8 => col
            .as_any()
            .downcast_ref::<Int8Array>()
            .map(|a| Value::I8(a.value(row))),
        AD::Int16 => col
            .as_any()
            .downcast_ref::<Int16Array>()
            .map(|a| Value::I16(a.value(row))),
        AD::Int32 => col
            .as_any()
            .downcast_ref::<Int32Array>()
            .map(|a| Value::I32(a.value(row))),
        AD::Int64 => col
            .as_any()
            .downcast_ref::<Int64Array>()
            .map(|a| Value::I64(a.value(row))),
        AD::UInt8 => col
            .as_any()
            .downcast_ref::<UInt8Array>()
            .map(|a| Value::U8(a.value(row))),
        AD::UInt16 => col
            .as_any()
            .downcast_ref::<UInt16Array>()
            .map(|a| Value::U16(a.value(row))),
        AD::UInt32 => col
            .as_any()
            .downcast_ref::<UInt32Array>()
            .map(|a| Value::U32(a.value(row))),
        AD::UInt64 => col
            .as_any()
            .downcast_ref::<UInt64Array>()
            .map(|a| Value::U64(a.value(row))),
        AD::Float16 => col
            .as_any()
            .downcast_ref::<Float16Array>()
            .map(|a| Value::F16(a.value(row))),
        AD::Float32 => col
            .as_any()
            .downcast_ref::<Float32Array>()
            .map(|a| Value::F32(a.value(row))),
        AD::Float64 => col
            .as_any()
            .downcast_ref::<Float64Array>()
            .map(|a| Value::F64(a.value(row))),
        AD::Boolean => col
            .as_any()
            .downcast_ref::<BooleanArray>()
            .map(|a| Value::Bool(a.value(row))),
        AD::Utf8 => col
            .as_any()
            .downcast_ref::<StringArray>()
            .map(|a| Value::String(a.value(row).to_string())),
        AD::Binary => col
            .as_any()
            .downcast_ref::<BinaryArray>()
            .map(|a| Value::Bytes(a.value(row).to_vec())),
        AD::List(field) => {
            let list = col.as_any().downcast_ref::<ListArray>()?;
            if list.is_null(row) {
                return None;
            }
            extract_list_value(list.value(row).as_ref(), field.data_type())
        }
        _ => None,
    }
}

// Null list items are dropped for string/binary/bool lists.
fn extract_list_value(child: &dyn Array, item_type: &ArrowType) -> Option<Value> {
    use ArrowType as AD;
    match item_type {
        AD::Utf8 => {
            let a = child.as_any().downcast_ref::<StringArray>()?;
            let mut out: Vec<String> = Vec::with_capacity(a.len());
            for i in 0..a.len() {
                if !a.is_null(i) {
                    out.push(a.value(i).to_string());
                }
            }
            Some(Value::ArrayString(out))
        }
        AD::Binary => {
            let a = child.as_any().downcast_ref::<BinaryArray>()?;
            let mut out: Vec<Vec<u8>> = Vec::with_capacity(a.len());
            for i in 0..a.len() {
                if !a.is_null(i) {
                    out.push(a.value(i).to_vec());
                }
            }
            Some(Value::ArrayBinary(out))
        }
        AD::Boolean => {
            let a = child.as_any().downcast_ref::<BooleanArray>()?;
            let mut out: Vec<bool> = Vec::with_capacity(a.len());
            for i in 0..a.len() {
                if !a.is_null(i) {
                    out.push(a.value(i));
                }
            }
            Some(Value::ArrayBool(out))
        }
        AD::Int32 => {
            let a = child.as_any().downcast_ref::<Int32Array>()?;
            Some(Value::ArrayI32(a.values().to_vec()))
        }
        AD::Int64 => {
            let a = child.as_any().downcast_ref::<Int64Array>()?;
            Some(Value::ArrayI64(a.values().to_vec()))
        }
        AD::UInt32 => {
            let a = child.as_any().downcast_ref::<UInt32Array>()?;
            Some(Value::ArrayU32(a.values().to_vec()))
        }
        AD::UInt64 => {
            let a = child.as_any().downcast_ref::<UInt64Array>()?;
            Some(Value::ArrayU64(a.values().to_vec()))
        }
        AD::Float32 => {
            let a = child.as_any().downcast_ref::<Float32Array>()?;
            Some(Value::ArrayF32(a.values().to_vec()))
        }
        AD::Float64 => {
            let a = child.as_any().downcast_ref::<Float64Array>()?;
            Some(Value::ArrayF64(a.values().to_vec()))
        }
        _ => None,
    }
}

pub(super) fn approx_value_size(v: &Value) -> usize {
    use std::mem::size_of;
    match v {
        Value::Null => 0,
        Value::Bool(_) => size_of::<bool>(),
        Value::I8(_) => size_of::<i8>(),
        Value::I16(_) => size_of::<i16>(),
        Value::I32(_) => size_of::<i32>(),
        Value::I64(_) => size_of::<i64>(),
        Value::U8(_) => size_of::<u8>(),
        Value::U16(_) => size_of::<u16>(),
        Value::U32(_) => size_of::<u32>(),
        Value::U64(_) => size_of::<u64>(),
        Value::F16(_) => size_of::<u16>(),
        Value::F32(_) => size_of::<f32>(),
        Value::F64(_) => size_of::<f64>(),
        Value::String(s) => s.len(),
        Value::Bytes(b) => b.len(),
        Value::VecBool(v) => v.len() * size_of::<bool>(),
        Value::VecI8(v) => v.len() * size_of::<i8>(),
        Value::VecI16(v) => v.len() * size_of::<i16>(),
        Value::VecI32(v) => v.len() * size_of::<i32>(),
        Value::VecI64(v) => v.len() * size_of::<i64>(),
        Value::VecU32(v) => v.len() * size_of::<u32>(),
        Value::VecU64(v) => v.len() * size_of::<u64>(),
        Value::VecF16(v) => v.len() * size_of::<u16>(),
        Value::VecF32(v) => v.len() * size_of::<f32>(),
        Value::VecF64(v) => v.len() * size_of::<f64>(),
        Value::VecString(v) => v.iter().map(String::len).sum(),
        Value::SparseF16 { indices, values } => {
            indices.len() * size_of::<u32>() + values.len() * size_of::<u16>()
        }
        Value::SparseF32 { indices, values } => {
            indices.len() * size_of::<u32>() + values.len() * size_of::<f32>()
        }
        Value::ArrayBinary(v) => v.iter().map(|b| b.len()).sum(),
        Value::ArrayI32(v) => v.len() * size_of::<i32>(),
        Value::ArrayI64(v) => v.len() * size_of::<i64>(),
        Value::ArrayU32(v) => v.len() * size_of::<u32>(),
        Value::ArrayU64(v) => v.len() * size_of::<u64>(),
        Value::ArrayBool(v) => v.len() * size_of::<bool>(),
        Value::ArrayF32(v) => v.len() * size_of::<f32>(),
        Value::ArrayF64(v) => v.len() * size_of::<f64>(),
        Value::ArrayString(v) => v.iter().map(String::len).sum(),
    }
}

pub(super) fn encode_sparse_f32(indices: &[u32], values: &[f32]) -> Vec<u8> {
    let n = indices.len().min(values.len());
    let mut buf = Vec::with_capacity(4 + n * 8);
    buf.extend_from_slice(&(n as u32).to_le_bytes());
    for &idx in indices.iter().take(n) {
        buf.extend_from_slice(&idx.to_le_bytes());
    }
    for &val in values.iter().take(n) {
        buf.extend_from_slice(&val.to_le_bytes());
    }
    buf
}

/// Decode Finch's sparse vector encoding used in persisted forward stores.
///
/// Format:
/// - u32 count N (little-endian)
/// - N x u32 indices (little-endian)
/// - N x f32 values (little-endian)
pub fn decode_sparse_f32(bytes: &[u8]) -> ZResult<(Vec<u32>, Vec<f32>)> {
    use byteorder::{LittleEndian, ReadBytesExt};
    let mut cur = std::io::Cursor::new(bytes);
    let n = cur
        .read_u32::<LittleEndian>()
        .map_err(|e| Status::io_error(e.to_string()))? as usize;
    let mut indices = Vec::with_capacity(n);
    for _ in 0..n {
        indices.push(
            cur.read_u32::<LittleEndian>()
                .map_err(|e| Status::io_error(e.to_string()))?,
        );
    }
    let mut values = Vec::with_capacity(n);
    for _ in 0..n {
        values.push(
            cur.read_f32::<LittleEndian>()
                .map_err(|e| Status::io_error(e.to_string()))?,
        );
    }
    Ok((indices, values))
}

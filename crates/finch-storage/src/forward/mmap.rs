use super::*;

// Per-batch doc-id ranges, per-batch row offsets, and the total doc count.
type BatchRanges = (Vec<(u64, u64)>, Vec<usize>, usize);

impl MmapForwardStore {
    /// Opens the forward file at `path` with the default options.
    pub fn open(path: &Path) -> ZResult<Self> {
        Self::open_with_options(path, ForwardStoreOpenOptions::default())
    }

    /// Opens the forward file at `path`, with memory maps when `enable_mmap` is true.
    pub fn open_with_mmap(path: &Path, enable_mmap: bool) -> ZResult<Self> {
        Self::open_with_options(
            path,
            ForwardStoreOpenOptions {
                enable_mmap,
                ..ForwardStoreOpenOptions::default()
            },
        )
    }

    /// Opens the forward file at `path`; the file header, not the extension, selects Parquet or Arrow IPC.
    pub fn open_with_options(path: &Path, options: ForwardStoreOpenOptions) -> ZResult<Self> {
        let i64_pk_sidecar = I64PkSidecar::open(path);
        // Sniff file header for Parquet magic bytes.
        //
        // This allows opening files regardless of extension (including `.tmp`)
        // and keeps `open()` as the single entrypoint for finch-db.
        use std::io::Read;
        let mut f = File::open(path).map_err(|e| Status::io_error(e.to_string()))?;
        let mut hdr = [0u8; 4];
        let n = f.read(&mut hdr).unwrap_or(0);
        drop(f);
        if n == 4 && &hdr == b"PAR1" {
            return match options.parquet_read_mode {
                ParquetReadMode::Buffered => Self::open_parquet_buffered(path),
                ParquetReadMode::Eager => Self::open_parquet_eager(path),
            };
        }
        if options.lazy_ipc {
            return Ok(MmapForwardStore {
                inner: ForwardStoreInner::LazyIpc(LazyIpcForwardStore {
                    path: path.to_path_buf(),
                    enable_mmap: options.enable_mmap,
                    inner: OnceLock::new(),
                }),
                i64_pk_sidecar,
            });
        }
        Self::open_ipc(path, options.enable_mmap)
    }

    pub(super) fn materialized_inner(&self) -> ZResult<&ForwardStoreInner> {
        match &self.inner {
            ForwardStoreInner::LazyIpc(lazy) => lazy.materialize(),
            inner => Ok(inner),
        }
    }

    pub(super) fn compute_batch_ranges(batches: &[RecordBatch]) -> ZResult<BatchRanges> {
        let mut batch_doc_id_ranges: Vec<(u64, u64)> = Vec::with_capacity(batches.len());
        let mut batch_row_offsets: Vec<usize> = Vec::with_capacity(batches.len());
        let mut doc_count: usize = 0;
        let mut prev_max_doc_id: Option<u64> = None;

        for (batch_idx, batch) in batches.iter().enumerate() {
            batch_row_offsets.push(doc_count);
            let len = batch.num_rows();
            doc_count = doc_count.saturating_add(len);

            let id_col = non_null_doc_id_column(batch)?;
            if id_col.is_empty() {
                batch_doc_id_ranges.push((0, 0));
                continue;
            }
            let min = id_col.value(0);
            let max = id_col.value(id_col.len() - 1);
            if max < min {
                return Err(Status::io_error(format!(
                    "forward store __doc_id__ column range is invalid (batch={})",
                    batch_idx,
                )));
            }
            if let Some(prev_max) = prev_max_doc_id.filter(|&prev_max| prev_max >= min) {
                return Err(Status::io_error(format!(
                    "forward store record batches must be ordered and non-overlapping (prev_max={}, this_min={})",
                    prev_max, min
                )));
            }
            prev_max_doc_id = Some(max);
            batch_doc_id_ranges.push((min, max));
        }

        Ok((batch_doc_id_ranges, batch_row_offsets, doc_count))
    }

    pub(super) fn open_parquet_buffered(path: &Path) -> ZResult<Self> {
        // Bounded row-group cache to avoid reading the full Parquet file on open.
        const PARQUET_ROW_GROUP_CACHE_CAP: usize = 8;

        let file = File::open(path).map_err(|e| Status::io_error(e.to_string()))?;
        let metadata = ArrowReaderMetadata::load(&file, Default::default())
            .map_err(|e| Status::io_error(e.to_string()))?;

        let parquet_md = metadata.metadata();
        let num_row_groups = parquet_md.num_row_groups();

        let mut row_group_row_offsets: Vec<usize> = Vec::with_capacity(num_row_groups);
        let mut doc_count: usize = 0;
        for i in 0..num_row_groups {
            row_group_row_offsets.push(doc_count);
            let rg_rows = parquet_md
                .row_group(i)
                .num_rows()
                .try_into()
                .unwrap_or(0usize);
            doc_count = doc_count.saturating_add(rg_rows);
        }

        let row_group_doc_id_ranges = parquet_row_group_doc_id_ranges(path, &metadata)?;

        Ok(MmapForwardStore {
            inner: ForwardStoreInner::ParquetBuffered(ParquetBufferedForwardStore {
                path: path.to_path_buf(),
                schema: metadata.schema().clone(),
                metadata,
                row_group_doc_id_ranges,
                row_group_row_offsets,
                doc_count,
                cache: std::sync::Mutex::new(ParquetRowGroupCacheLru::new(
                    PARQUET_ROW_GROUP_CACHE_CAP,
                )),
            }),
            i64_pk_sidecar: I64PkSidecar::open(path),
        })
    }

    pub(super) fn open_parquet_eager(path: &Path) -> ZResult<Self> {
        let file = File::open(path).map_err(|e| Status::io_error(e.to_string()))?;
        let mut builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| Status::io_error(e.to_string()))?;
        builder = builder.with_batch_size(4096);
        let batches = read_parquet_batches(builder)?;

        let (batch_doc_id_ranges, batch_row_offsets, doc_count) =
            Self::compute_batch_ranges(&batches)?;
        Ok(MmapForwardStore {
            inner: ForwardStoreInner::InMemory(InMemoryForwardStore {
                batches,
                batch_doc_id_ranges,
                batch_row_offsets,
                doc_count,
            }),
            i64_pk_sidecar: I64PkSidecar::open(path),
        })
    }

    pub(super) fn open_ipc(path: &Path, enable_mmap: bool) -> ZResult<Self> {
        Ok(MmapForwardStore {
            inner: Self::open_ipc_inner(path, enable_mmap)?,
            i64_pk_sidecar: I64PkSidecar::open(path),
        })
    }

    pub(super) fn open_ipc_inner(path: &Path, enable_mmap: bool) -> ZResult<ForwardStoreInner> {
        let buffer = read_ipc_buffer(path, enable_mmap)?;
        let batches = decode_ipc_batches(&buffer)?;
        let (batch_doc_id_ranges, batch_row_offsets, doc_count) =
            Self::compute_batch_ranges(&batches)?;

        Ok(ForwardStoreInner::InMemory(InMemoryForwardStore {
            batches,
            batch_doc_id_ranges,
            batch_row_offsets,
            doc_count,
        }))
    }

    /// Return the segment-local row index for `doc_id` (0-based), or `None` if
    /// the doc_id is not present in this forward store.
    pub fn row_index_of_doc_id(&self, doc_id: u64) -> Option<u64> {
        match self.materialized_inner().ok()? {
            ForwardStoreInner::InMemory(s) => s.row_index_of_doc_id(doc_id),
            ForwardStoreInner::ParquetBuffered(s) => s.row_index_of_doc_id(doc_id),
            ForwardStoreInner::LazyIpc(_) => None,
        }
    }

    /// Visit every row in doc-id order, exposing the raw `RecordBatch` and row offset.
    ///
    /// This is used by compaction/DDL paths to stream through the forward store without
    /// materializing ID lists first.
    pub fn scan_rows<F>(&self, f: F) -> ZResult<()>
    where
        F: FnMut(u64, &RecordBatch, usize) -> ZResult<()>,
    {
        match self.materialized_inner()? {
            ForwardStoreInner::InMemory(s) => s.scan_rows(f),
            ForwardStoreInner::ParquetBuffered(s) => s.scan_rows(f),
            ForwardStoreInner::LazyIpc(_) => unreachable!("lazy forward store must materialize"),
        }
    }

    /// Visit every row in doc-id order for a `LargeBinary` column.
    ///
    /// This avoids per-row `column_by_name()` lookups and is intended for
    /// streaming index-build paths.
    pub fn scan_large_binary_column_rows<F>(&self, col_name: &str, f: F) -> ZResult<()>
    where
        F: FnMut(u64, Option<&[u8]>) -> ZResult<()>,
    {
        match self.materialized_inner()? {
            ForwardStoreInner::InMemory(s) => s.scan_large_binary_column_rows(col_name, f),
            ForwardStoreInner::ParquetBuffered(s) => s.scan_large_binary_column_rows(col_name, f),
            ForwardStoreInner::LazyIpc(_) => unreachable!("lazy forward store must materialize"),
        }
    }

    /// Return whether the forward store schema contains a column with `name`.
    ///
    /// The Arrow IPC schema is consistent across record batches, so checking
    /// the first batch is sufficient.
    pub fn has_column(&self, name: &str) -> bool {
        match self.materialized_inner() {
            Ok(ForwardStoreInner::InMemory(s)) => s.has_column(name),
            Ok(ForwardStoreInner::ParquetBuffered(s)) => s.has_column(name),
            Ok(ForwardStoreInner::LazyIpc(_)) | Err(_) => false,
        }
    }

    /// The document for each doc id, in input order; `None` for an id that the file does not hold.
    pub fn get_by_doc_ids(&self, ids: &[u64]) -> ZResult<Vec<Option<Doc>>> {
        match self.materialized_inner()? {
            ForwardStoreInner::InMemory(s) => s.get_by_doc_ids(ids),
            ForwardStoreInner::ParquetBuffered(s) => s.get_by_doc_ids(ids),
            ForwardStoreInner::LazyIpc(_) => unreachable!("lazy forward store must materialize"),
        }
    }

    /// Compute exact distances for a dense vector field for each doc_id.
    ///
    /// This is a performance helper used by finch-db's refiner path to avoid
    /// extracting full docs (scalars + vectors) from Arrow.
    pub fn compute_dense_distance_by_doc_ids(
        &self,
        field: &str,
        query: &[f32],
        metric: MetricType,
        ids: &[u64],
    ) -> ZResult<Vec<Option<f32>>> {
        let inner = self.materialized_inner()?;
        let query = DenseDistanceQuery::new(field, query, metric);
        match inner {
            ForwardStoreInner::InMemory(s) => s.compute_dense_distance_by_doc_ids(&query, ids),
            ForwardStoreInner::ParquetBuffered(s) => {
                s.compute_dense_distance_by_doc_ids(&query, ids)
            }
            ForwardStoreInner::LazyIpc(_) => unreachable!("lazy forward store must materialize"),
        }
    }

    /// Fetch only the primary key for each doc_id.
    ///
    /// This avoids extracting scalar/vector fields from Arrow and is much faster
    /// than `get_by_doc_ids()` when callers only need `pk`.
    pub fn get_pks_by_doc_ids(&self, ids: &[u64]) -> ZResult<Vec<Option<String>>> {
        match self.materialized_inner()? {
            ForwardStoreInner::InMemory(s) => s.get_pks_by_doc_ids(ids),
            ForwardStoreInner::ParquetBuffered(s) => s.get_pks_by_doc_ids(ids),
            ForwardStoreInner::LazyIpc(_) => unreachable!("lazy forward store must materialize"),
        }
    }

    /// Fetch integer primary keys for each doc_id without allocating strings.
    pub fn get_i64_pks_by_doc_ids(&self, ids: &[u64]) -> ZResult<Vec<Option<i64>>> {
        if let Some(sidecar) = &self.i64_pk_sidecar {
            return Ok(sidecar.get_by_doc_ids(ids));
        }
        match self.materialized_inner()? {
            ForwardStoreInner::InMemory(s) => s.get_i64_pks_by_doc_ids(ids),
            ForwardStoreInner::ParquetBuffered(s) => s.get_i64_pks_by_doc_ids(ids),
            ForwardStoreInner::LazyIpc(_) => unreachable!("lazy forward store must materialize"),
        }
    }

    /// Return all doc_ids stored in this forward store in row order.
    ///
    /// Used by DDL/index builders to iterate docs without assuming contiguous
    /// ranges.
    pub fn all_doc_ids(&self) -> Vec<u64> {
        match self.materialized_inner() {
            Ok(ForwardStoreInner::InMemory(s)) => s.all_doc_ids(),
            Ok(ForwardStoreInner::ParquetBuffered(s)) => s.all_doc_ids(),
            Ok(ForwardStoreInner::LazyIpc(_)) | Err(_) => Vec::new(),
        }
    }

    /// Count of rows; 0 when the file cannot be read.
    pub fn len(&self) -> usize {
        match self.materialized_inner() {
            Ok(ForwardStoreInner::InMemory(s)) => s.len(),
            Ok(ForwardStoreInner::ParquetBuffered(s)) => s.len(),
            Ok(ForwardStoreInner::LazyIpc(_)) | Err(_) => 0,
        }
    }

    /// Whether the store has no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The document in row `row` of `batch`, with `doc_id` as its id.
    pub fn extract_doc_from_batch(batch: &RecordBatch, row: usize, doc_id: u64) -> Doc {
        let schema = batch.schema();

        let pk = batch
            .column_by_name("__pk__")
            .and_then(|col| col.as_any().downcast_ref::<StringArray>())
            .and_then(|arr| {
                if arr.is_null(row) {
                    None
                } else {
                    Some(arr.value(row).to_string())
                }
            })
            .unwrap_or_default();

        let mut doc = Doc::new(pk);
        doc.doc_id = doc_id;

        for field in schema.fields() {
            let name = field.name();
            if name == "__doc_id__" || name == "__pk__" {
                continue;
            }

            let Some(col) = batch.column_by_name(name) else {
                continue;
            };
            if col.is_null(row) {
                continue;
            }

            if let Some((vec_field, codec)) = vector_column(name) {
                if let Some(v) = decode_vector_cell(col.as_ref(), row, codec) {
                    doc.fields.insert(vec_field.to_string(), v);
                }
                continue;
            }

            let val = extract_scalar_value(col.as_ref(), row, field.data_type());
            if let Some(v) = val {
                doc.fields.insert(name.clone(), v);
            }
        }
        doc
    }
}

impl LazyIpcForwardStore {
    fn materialize(&self) -> ZResult<&ForwardStoreInner> {
        let loaded = self.inner.get_or_init(|| {
            MmapForwardStore::open_ipc_inner(&self.path, self.enable_mmap)
                .map(Box::new)
                .map_err(|e| e.to_string())
        });
        match loaded {
            Ok(inner) => Ok(inner.as_ref()),
            Err(e) => Err(Status::io_error(e.clone())),
        }
    }
}

/// Byte encoding of a persisted vector column.
#[derive(Clone, Copy)]
enum VectorCodec {
    F32,
    U32,
    U64,
    Sparse,
}

// Checked in this order; a column matching a prefix never falls through to scalar decoding.
const VECTOR_COLUMN_PREFIXES: [(&str, VectorCodec); 4] = [
    ("__vec__", VectorCodec::F32),
    ("__bvec32__", VectorCodec::U32),
    ("__bvec64__", VectorCodec::U64),
    ("__svec__", VectorCodec::Sparse),
];

/// Splits a vector column name into its field name and codec.
fn vector_column(name: &str) -> Option<(&str, VectorCodec)> {
    VECTOR_COLUMN_PREFIXES
        .iter()
        .find_map(|&(prefix, codec)| Some((name.strip_prefix(prefix)?, codec)))
}

fn decode_vector_cell(col: &dyn Array, row: usize, codec: VectorCodec) -> Option<Value> {
    let bin_col = col.as_any().downcast_ref::<LargeBinaryArray>()?;
    if bin_col.is_null(row) {
        return None;
    }
    let bytes = bin_col.value(row);
    match codec {
        VectorCodec::F32 if bytes.len() % 4 == 0 => {
            let floats: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            Some(Value::VecF32(floats))
        }
        VectorCodec::U32 if bytes.len() % 4 == 0 => {
            let words: Vec<u32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes(*b))
                .collect();
            Some(Value::VecU32(words))
        }
        VectorCodec::U64 if bytes.len() % 8 == 0 => {
            let words: Vec<u64> = bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| u64::from_le_bytes(*b))
                .collect();
            Some(Value::VecU64(words))
        }
        VectorCodec::Sparse => {
            let (indices, values) = decode_sparse_f32(bytes).ok()?;
            // an encoded empty sparse vector is treated as not set.
            (!indices.is_empty() && !values.is_empty())
                .then_some(Value::SparseF32 { indices, values })
        }
        _ => None,
    }
}

/// The `__doc_id__` column of a forward-store batch.
pub(super) fn doc_id_column(batch: &RecordBatch) -> ZResult<&UInt64Array> {
    let col = batch
        .column_by_name("__doc_id__")
        .ok_or_else(|| Status::io_error("forward store missing __doc_id__ column".to_string()))?;
    col.as_any().downcast_ref::<UInt64Array>().ok_or_else(|| {
        Status::io_error("forward store __doc_id__ column has wrong type".to_string())
    })
}

fn non_null_doc_id_column(batch: &RecordBatch) -> ZResult<&UInt64Array> {
    let id_col = doc_id_column(batch)?;
    if id_col.null_count() != 0 {
        return Err(Status::io_error(
            "forward store __doc_id__ column must not contain NULLs",
        ));
    }
    Ok(id_col)
}

/// Visits every row of `batch` with its doc id.
pub(super) fn scan_batch_rows<F>(batch: &RecordBatch, f: &mut F) -> ZResult<()>
where
    F: FnMut(u64, &RecordBatch, usize) -> ZResult<()>,
{
    let id_col = doc_id_column(batch)?;
    for row in 0..id_col.len() {
        let doc_id = id_col.value(row);
        f(doc_id, batch, row)?;
    }
    Ok(())
}

/// Visits every row of a `LargeBinary` column in `batch` with its doc id.
pub(super) fn scan_large_binary_batch_rows<F>(
    batch: &RecordBatch,
    col_name: &str,
    f: &mut F,
) -> ZResult<()>
where
    F: FnMut(u64, Option<&[u8]>) -> ZResult<()>,
{
    let id_col = batch
        .column_by_name("__doc_id__")
        .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
        .ok_or_else(|| Status::io_error("forward store missing __doc_id__ column".to_string()))?;

    let col = batch
        .column_by_name(col_name)
        .ok_or_else(|| Status::io_error(format!("forward store missing column '{}'", col_name)))?;
    let bin = col
        .as_any()
        .downcast_ref::<LargeBinaryArray>()
        .ok_or_else(|| {
            Status::io_error(format!(
                "forward store column '{}' has wrong type",
                col_name
            ))
        })?;

    for row in 0..id_col.len() {
        let doc_id = id_col.value(row);
        if bin.is_null(row) {
            f(doc_id, None)?;
        } else {
            f(doc_id, Some(bin.value(row)))?;
        }
    }
    Ok(())
}

pub(super) fn require_column(present: bool, col_name: &str) -> ZResult<()> {
    if present {
        return Ok(());
    }
    Err(Status::invalid_argument(format!(
        "forward store missing column '{}'",
        col_name
    )))
}

pub(super) fn read_parquet_batches(
    builder: ParquetRecordBatchReaderBuilder<File>,
) -> ZResult<Vec<RecordBatch>> {
    let reader = builder
        .build()
        .map_err(|e| Status::io_error(e.to_string()))?;

    let mut batches: Vec<RecordBatch> = Vec::new();
    for batch in reader {
        let batch = batch.map_err(|e| Status::io_error(e.to_string()))?;
        batches.push(batch);
    }
    Ok(batches)
}

fn read_ipc_buffer(path: &Path, enable_mmap: bool) -> ZResult<Buffer> {
    let bytes = if enable_mmap {
        // Zero-copy IPC read:
        // - mmap the IPC file
        // - decode record batches against `Buffer` slices (no per-process copies)
        let file = File::open(path).map_err(|e| Status::io_error(e.to_string()))?;
        // SAFETY: forward store files are written once and not modified while mapped.
        let mmap =
            unsafe { memmap2::Mmap::map(&file) }.map_err(|e| Status::io_error(e.to_string()))?;
        bytes::Bytes::from_owner(mmap)
    } else {
        // Heap-backed IPC read (no memory-mapping).
        let data = std::fs::read(path).map_err(|e| Status::io_error(e.to_string()))?;
        bytes::Bytes::from(data)
    };
    Ok(Buffer::from(bytes))
}

fn decode_ipc_batches(buffer: &Buffer) -> ZResult<Vec<RecordBatch>> {
    if buffer.len() < 10 {
        return Err(Status::io_error("invalid Arrow IPC file (too small)"));
    }

    let trailer_start = buffer.len() - 10;
    let footer_len = read_footer_length(
        buffer[trailer_start..]
            .try_into()
            .map_err(|_| Status::io_error("invalid Arrow IPC trailer"))?,
    )
    .map_err(|e| Status::io_error(e.to_string()))?;
    if footer_len > trailer_start {
        return Err(Status::io_error("invalid Arrow IPC footer length"));
    }

    let footer = ipc::root_as_footer(&buffer[trailer_start - footer_len..trailer_start])
        .map_err(|e| Status::io_error(format!("invalid Arrow IPC footer: {e:?}")))?;
    let ipc_schema = footer
        .schema()
        .ok_or_else(|| Status::io_error("Arrow IPC footer missing schema"))?;
    let schema = fb_to_schema(ipc_schema);

    let mut decoder = FileDecoder::new(Arc::new(schema), footer.version());

    // Read dictionaries (if present).
    for block in footer.dictionaries().into_iter().flatten() {
        let block_len = (block.bodyLength() as usize)
            .checked_add(block.metaDataLength() as usize)
            .ok_or_else(|| Status::io_error("Arrow IPC dictionary block overflow"))?;
        let data = buffer.slice_with_length(block.offset() as _, block_len);
        decoder
            .read_dictionary(block, &data)
            .map_err(|e| Status::io_error(e.to_string()))?;
    }

    let batch_blocks: Vec<ipc::Block> = footer
        .recordBatches()
        .map(|b| b.iter().copied().collect())
        .unwrap_or_default();

    let mut batches: Vec<RecordBatch> = Vec::with_capacity(batch_blocks.len());

    for block in &batch_blocks {
        let block_len = (block.bodyLength() as usize)
            .checked_add(block.metaDataLength() as usize)
            .ok_or_else(|| Status::io_error("Arrow IPC record batch block overflow"))?;
        let data = buffer.slice_with_length(block.offset() as _, block_len);
        let batch = decoder
            .read_record_batch(block, &data)
            .map_err(|e| Status::io_error(e.to_string()))?
            .ok_or_else(|| Status::io_error("unexpected empty record batch"))?;

        batches.push(batch);
    }
    Ok(batches)
}

/// Per-row-group `__doc_id__` ranges, checked to be ordered and non-overlapping.
fn parquet_row_group_doc_id_ranges(
    path: &Path,
    metadata: &ArrowReaderMetadata,
) -> ZResult<Vec<(u64, u64)>> {
    let parquet_md = metadata.metadata();
    let num_row_groups = parquet_md.num_row_groups();

    // Find the leaf-column index for __doc_id__ so we can query row-group statistics.
    let doc_id_leaf_idx = metadata
        .parquet_schema()
        .columns()
        .iter()
        .enumerate()
        .find_map(|(i, c)| (c.path().string() == "__doc_id__").then_some(i))
        .ok_or_else(|| {
            Status::io_error("Parquet forward store missing __doc_id__ column".to_string())
        })?;

    let mut row_group_doc_id_ranges: Vec<(u64, u64)> = Vec::with_capacity(num_row_groups);
    let mut prev_max_doc_id: Option<u64> = None;

    for rg_idx in 0..num_row_groups {
        let stats = parquet_md
            .row_group(rg_idx)
            .column(doc_id_leaf_idx)
            .statistics();
        let range = match doc_id_range_from_stats(stats) {
            Some(r) => r,
            None => read_row_group_doc_id_range(path, metadata, rg_idx)?,
        };

        if let Some(prev_max) = prev_max_doc_id.filter(|&prev_max| prev_max >= range.0) {
            return Err(Status::io_error(format!(
                "forward store row groups must be ordered and non-overlapping (prev_max={}, this_min={})",
                prev_max, range.0
            )));
        }
        prev_max_doc_id = Some(range.1);
        row_group_doc_id_ranges.push(range);
    }
    Ok(row_group_doc_id_ranges)
}

/// Exact, non-negative min/max from row-group statistics, if present.
fn doc_id_range_from_stats(
    stats: Option<&parquet::file::statistics::Statistics>,
) -> Option<(u64, u64)> {
    use parquet::file::statistics::Statistics;

    let (min, max) = match stats? {
        Statistics::Int64(s) if s.min_is_exact() && s.max_is_exact() => {
            (*s.min_opt()?, *s.max_opt()?)
        }
        Statistics::Int32(s) if s.min_is_exact() && s.max_is_exact() => {
            (i64::from(*s.min_opt()?), i64::from(*s.max_opt()?))
        }
        _ => return None,
    };
    (min >= 0 && max >= 0).then_some((min as u64, max as u64))
}

// Fallback when statistics are missing; rare for files written by ArrowWriter.
fn read_row_group_doc_id_range(
    path: &Path,
    metadata: &ArrowReaderMetadata,
    rg_idx: usize,
) -> ZResult<(u64, u64)> {
    let file = File::open(path).map_err(|e| Status::io_error(e.to_string()))?;
    let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(file, metadata.clone())
        .with_batch_size(4096)
        .with_row_groups(vec![rg_idx])
        .with_projection(ProjectionMask::columns(
            metadata.parquet_schema(),
            ["__doc_id__"],
        ));

    let reader = builder
        .build()
        .map_err(|e| Status::io_error(e.to_string()))?;
    let mut bounds = DocIdBounds::default();
    for batch in reader {
        let batch = batch.map_err(|e| Status::io_error(e.to_string()))?;
        let id_col = non_null_doc_id_column(&batch)?;
        extend_doc_id_bounds(&mut bounds, id_col, rg_idx)?;
    }

    match (bounds.min, bounds.max) {
        (Some(a), Some(b)) => Ok((a, b)),
        _ => Ok((0, 0)),
    }
}

/// Running bounds of a row group's doc ids, read in row order.
#[derive(Default)]
struct DocIdBounds {
    min: Option<u64>,
    max: Option<u64>,
    prev: Option<u64>,
}

fn extend_doc_id_bounds(
    bounds: &mut DocIdBounds,
    id_col: &UInt64Array,
    rg_idx: usize,
) -> ZResult<()> {
    for row in 0..id_col.len() {
        let v = id_col.value(row);
        if let Some(p) = bounds.prev {
            if v <= p {
                return Err(Status::io_error(format!(
                    "forward store __doc_id__ column must be strictly increasing (row_group={}, row={})",
                    rg_idx, row
                )));
            }
        } else {
            bounds.min = Some(v);
        }
        bounds.prev = Some(v);
        bounds.max = Some(v);
    }
    Ok(())
}

pub(super) fn find_range_index(ranges: &[(u64, u64)], doc_id: u64) -> Option<usize> {
    let mut lo = 0usize;
    let mut hi = ranges.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        let (_min, max) = ranges[mid];
        if max < doc_id {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if lo >= ranges.len() {
        return None;
    }
    let (min, max) = ranges[lo];
    if min <= doc_id && doc_id <= max {
        Some(lo)
    } else {
        None
    }
}

pub(super) fn find_row_in_batch(
    batches: &[RecordBatch],
    batch_doc_id_ranges: &[(u64, u64)],
    batch_idx: usize,
    doc_id: u64,
) -> Option<usize> {
    let batch = batches.get(batch_idx)?;
    let col = batch.column_by_name("__doc_id__")?;
    let id_col = col.as_any().downcast_ref::<UInt64Array>()?;
    if id_col.is_empty() {
        return None;
    }

    let (min, _max) = *batch_doc_id_ranges.get(batch_idx)?;
    if doc_id >= min {
        let guess = (doc_id - min) as usize;
        if guess < id_col.len() && !id_col.is_null(guess) && id_col.value(guess) == doc_id {
            return Some(guess);
        }
    }

    // Fallback: binary search within the batch.
    let mut lo = 0usize;
    let mut hi = id_col.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        let v = id_col.value(mid);
        if v < doc_id {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if lo < id_col.len() && id_col.value(lo) == doc_id {
        Some(lo)
    } else {
        None
    }
}

pub(super) fn locate_in_batches(
    batches: &[RecordBatch],
    batch_doc_id_ranges: &[(u64, u64)],
    doc_id: u64,
) -> Option<(usize, usize)> {
    let batch_idx = find_range_index(batch_doc_id_ranges, doc_id)?;
    let row = find_row_in_batch(batches, batch_doc_id_ranges, batch_idx, doc_id)?;
    Some((batch_idx, row))
}

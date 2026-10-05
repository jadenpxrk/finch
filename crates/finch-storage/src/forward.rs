//! Arrow-based forward document store

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use crate::arrow_schema::data_type_to_arrow;
use arrow::array::*;
use arrow::buffer::Buffer;
use arrow::datatypes::{DataType as ArrowType, Field, Schema};
use arrow::ipc;
use arrow::ipc::convert::fb_to_schema;
use arrow::ipc::reader::{read_footer_length, FileDecoder};
use arrow::ipc::writer::FileWriter;
use arrow::record_batch::RecordBatch;
use finch_types::{CollectionSchema, DataType, Doc, MetricType, Status, Value, ZResult};
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ParquetRecordBatchReaderBuilder};
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::arrow::ProjectionMask;
use parquet::file::properties::WriterProperties;

mod in_memory;
mod mmap;
mod parquet_reader;
mod writer;

use mmap::*;
pub use parquet_reader::*;

pub use writer::MemoryForwardStore;

const PK_I64_SIDECAR_MAGIC: &[u8; 8] = b"FINPKI64";
const PK_I64_SIDECAR_HEADER_LEN: usize = 16;

fn pk_i64_sidecar_path(forward_path: &Path) -> std::path::PathBuf {
    let file_name = forward_path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "forward".to_string());
    forward_path.with_file_name(format!("{}.pk_i64", file_name))
}

fn is_persisted_dense_vector(dt: DataType) -> bool {
    matches!(
        dt,
        DataType::VectorFp32
            | DataType::VectorFp16
            | DataType::VectorFp64
            | DataType::VectorInt8
            | DataType::VectorInt16
            | DataType::VectorInt4
    )
}

/// Arrow IPC schema for a persisted forward store (scalars + dense + sparse vectors).
pub fn ipc_schema_for_collection(schema: &CollectionSchema) -> Arc<Schema> {
    let mut arrow_fields: Vec<Field> = Vec::new();
    arrow_fields.push(Field::new("__doc_id__", ArrowType::UInt64, false));
    arrow_fields.push(Field::new("__pk__", ArrowType::Utf8, false));

    for field in schema.scalar_fields() {
        let arrow_dt = data_type_to_arrow(&field.data_type).unwrap_or(ArrowType::Null);
        arrow_fields.push(Field::new(&field.name, arrow_dt, field.nullable));
    }

    // Vector field columns: stored as LargeBinary (dtype-dependent raw bytes).
    for field in schema.vector_fields() {
        match field.data_type {
            dt if is_persisted_dense_vector(dt) => {
                arrow_fields.push(Field::new(
                    format!("__vec__{}", field.name),
                    ArrowType::LargeBinary,
                    true,
                ));
            }
            DataType::VectorBinary32 => {
                arrow_fields.push(Field::new(
                    format!("__bvec32__{}", field.name),
                    ArrowType::LargeBinary,
                    true,
                ));
            }
            DataType::VectorBinary64 => {
                arrow_fields.push(Field::new(
                    format!("__bvec64__{}", field.name),
                    ArrowType::LargeBinary,
                    true,
                ));
            }
            _ => {}
        }
    }

    // Sparse vector field columns: stored as LargeBinary (serialized sparse bytes)
    for field in schema.vector_fields() {
        if !matches!(field.data_type, DataType::SparseFp32 | DataType::SparseFp16) {
            continue;
        }
        arrow_fields.push(Field::new(
            format!("__svec__{}", field.name),
            ArrowType::LargeBinary,
            true,
        ));
    }

    Arc::new(Schema::new(arrow_fields))
}

/// In-memory forward store, append-only during write phase
struct InMemoryForwardStore {
    batches: Vec<RecordBatch>,
    batch_doc_id_ranges: Vec<(u64, u64)>,
    batch_row_offsets: Vec<usize>,
    doc_count: usize,
}

struct RowGroupCache {
    batches: Vec<RecordBatch>,
    batch_doc_id_ranges: Vec<(u64, u64)>,
    batch_row_offsets: Vec<usize>,
}

struct ParquetRowGroupCacheLru {
    cap: usize,
    entries: HashMap<usize, Arc<RowGroupCache>>,
    order: std::collections::VecDeque<usize>,
}

impl ParquetRowGroupCacheLru {
    fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            entries: HashMap::new(),
            order: std::collections::VecDeque::new(),
        }
    }

    fn get(&mut self, key: usize) -> Option<Arc<RowGroupCache>> {
        let v = self.entries.get(&key).cloned()?;
        self.touch(key);
        Some(v)
    }

    fn touch(&mut self, key: usize) {
        if let Some(pos) = self.order.iter().position(|k| *k == key) {
            self.order.remove(pos);
        }
        self.order.push_back(key);
    }

    fn insert(&mut self, key: usize, val: Arc<RowGroupCache>) {
        self.entries.insert(key, val);
        self.touch(key);
        while self.entries.len() > self.cap {
            if let Some(old) = self.order.pop_front() {
                self.entries.remove(&old);
            } else {
                break;
            }
        }
    }
}

struct ParquetBufferedForwardStore {
    path: std::path::PathBuf,
    metadata: parquet::arrow::arrow_reader::ArrowReaderMetadata,
    schema: Arc<Schema>,
    row_group_doc_id_ranges: Vec<(u64, u64)>,
    row_group_row_offsets: Vec<usize>,
    doc_count: usize,
    cache: std::sync::Mutex<ParquetRowGroupCacheLru>,
}

struct I64PkSidecar {
    mmap: Arc<memmap2::Mmap>,
    len: usize,
    first_doc_id: u64,
    contiguous_doc_ids: bool,
}

impl I64PkSidecar {
    fn open(forward_path: &Path) -> Option<Self> {
        let path = pk_i64_sidecar_path(forward_path);
        let file = File::open(path).ok()?;
        // SAFETY: the sidecar is written once and not modified while mapped.
        let mmap = unsafe { memmap2::Mmap::map(&file).ok()? };
        let raw = mmap.as_ref();
        if raw.len() < PK_I64_SIDECAR_HEADER_LEN || &raw[..8] != PK_I64_SIDECAR_MAGIC {
            return None;
        }
        let len = u64::from_le_bytes(raw[8..16].try_into().ok()?) as usize;
        let expected = PK_I64_SIDECAR_HEADER_LEN.checked_add(len.checked_mul(16)?)?;
        if raw.len() != expected {
            return None;
        }

        let first_doc_id = if len == 0 {
            0
        } else {
            let doc_id_bytes = &raw[PK_I64_SIDECAR_HEADER_LEN..PK_I64_SIDECAR_HEADER_LEN + 8];
            u64::from_le_bytes(doc_id_bytes.try_into().ok()?)
        };
        let doc_ids = {
            let raw = &raw[PK_I64_SIDECAR_HEADER_LEN..PK_I64_SIDECAR_HEADER_LEN + len * 8];
            // SAFETY: `raw` is `len * 8` bytes at page start + 16, so it is u64-aligned.
            unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const u64, len) }
        };
        let contiguous_doc_ids = doc_ids
            .iter()
            .enumerate()
            .all(|(idx, &doc_id)| doc_id == first_doc_id.saturating_add(idx as u64));

        Some(I64PkSidecar {
            mmap: Arc::new(mmap),
            len,
            first_doc_id,
            contiguous_doc_ids,
        })
    }

    fn doc_ids(&self) -> &[u64] {
        let raw = &self.mmap[PK_I64_SIDECAR_HEADER_LEN..PK_I64_SIDECAR_HEADER_LEN + self.len * 8];
        // SAFETY: `open` checked the map holds `len` u64s at page start + 16, which is u64-aligned.
        unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const u64, self.len) }
    }

    fn pks(&self) -> &[i64] {
        let start = PK_I64_SIDECAR_HEADER_LEN + self.len * 8;
        let raw = &self.mmap[start..start + self.len * 8];
        // SAFETY: `open` checked the map holds `len` i64s after the doc ids, at an 8-byte-aligned offset.
        unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const i64, self.len) }
    }

    fn get_by_doc_ids(&self, ids: &[u64]) -> Vec<Option<i64>> {
        let pks = self.pks();
        if self.contiguous_doc_ids {
            return ids
                .iter()
                .map(|id| {
                    let idx = id.checked_sub(self.first_doc_id)? as usize;
                    (idx < self.len).then(|| pks[idx])
                })
                .collect();
        }

        let doc_ids = self.doc_ids();
        ids.iter()
            .map(|id| doc_ids.binary_search(id).ok().map(|idx| pks[idx]))
            .collect()
    }
}

struct LazyIpcForwardStore {
    path: std::path::PathBuf,
    enable_mmap: bool,
    inner: OnceLock<Result<Box<ForwardStoreInner>, String>>,
}

enum ForwardStoreInner {
    InMemory(InMemoryForwardStore),
    ParquetBuffered(ParquetBufferedForwardStore),
    LazyIpc(LazyIpcForwardStore),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParquetReadMode {
    Buffered,
    Eager,
}

/// Open options for `MmapForwardStore`.
///
/// - Arrow IPC: `enable_mmap=true` uses `mmap` for zero-copy reads.
/// - Parquet: `parquet_read_mode` selects between eager load and a bounded
///   row-group cache.
#[derive(Debug, Clone, Copy)]
pub struct ForwardStoreOpenOptions {
    pub enable_mmap: bool,
    pub parquet_read_mode: ParquetReadMode,
    pub lazy_ipc: bool,
}

impl Default for ForwardStoreOpenOptions {
    fn default() -> Self {
        Self {
            enable_mmap: true,
            parquet_read_mode: ParquetReadMode::Buffered,
            lazy_ipc: false,
        }
    }
}

/// Read-only forward store.
///
/// - Arrow IPC: uses `mmap` when enabled for zero-copy reads
/// - Parquet: uses a simple row-group "buffer pool" cache (bounded) to avoid
///   loading the full file into memory on open
pub struct MmapForwardStore {
    inner: ForwardStoreInner,
    i64_pk_sidecar: Option<I64PkSidecar>,
}

#[cfg(test)]
mod tests;

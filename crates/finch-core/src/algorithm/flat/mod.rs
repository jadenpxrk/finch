//! FLAT (brute-force) dense vector index

use super::codec::{read_u32, read_u64, read_u8, u64s_to_le};
use super::{check_query_dim, doc_filter_allows, DocFilter, TopkHeap};
use crate::metric::{make_metric, Metric};
use crate::quantizer::{
    bytes_per_vector, distance_to_quantized_with_query_sq_norm, quantize_append,
    quantize_type_from_u32, query_sq_norm,
};
use finch_types::{FlatIndexParams, QuantizeType, Status, ZResult};
use memmap2::Mmap;
use rayon::prelude::*;
use std::io::Cursor;
use std::marker::PhantomData;
use std::sync::Arc;

const SEGMENT_KEYS: &str = "KEYS";
const SEGMENT_VECTORS: &str = "FEATURES";
const SEGMENT_META: &str = "META";

/// The bytes of one index segment: owned, shared, or memory-mapped.
#[derive(Clone)]
pub enum SegmentBytes {
    /// Bytes the segment owns.
    Owned(Vec<u8>),
    /// Bytes shared with other readers.
    Shared(Arc<Vec<u8>>),
    /// Bytes of a memory-mapped file.
    Mmap(Arc<Mmap>),
}

impl SegmentBytes {
    /// Returns the segment bytes.
    pub fn as_slice(&self) -> &[u8] {
        match self {
            SegmentBytes::Owned(v) => v.as_slice(),
            SegmentBytes::Shared(v) => v.as_slice(),
            SegmentBytes::Mmap(m) => m.as_ref(),
        }
    }
}

pub(crate) enum SegmentArray<T> {
    Borrowed {
        bytes: SegmentBytes,
        len: usize,
        _p: PhantomData<T>,
    },
    Owned(Vec<T>),
}

impl<T: Copy> SegmentArray<T> {
    pub fn len(&self) -> usize {
        match self {
            SegmentArray::Borrowed { len, .. } => *len,
            SegmentArray::Owned(v) => v.len(),
        }
    }

    pub fn as_slice(&self) -> &[T] {
        match self {
            SegmentArray::Owned(v) => v.as_slice(),
            SegmentArray::Borrowed { bytes, len, .. } => {
                let raw = bytes.as_slice();
                let size = std::mem::size_of::<T>();
                let expected = len * size;
                debug_assert_eq!(raw.len(), expected);
                let ptr = raw.as_ptr();
                debug_assert_eq!((ptr as usize) % std::mem::align_of::<T>(), 0);
                // SAFETY: only `segment_array_*` build `Borrowed`, after checking length, alignment, and LE.
                unsafe { std::slice::from_raw_parts(ptr as *const T, *len) }
            }
        }
    }
}

fn parse_le_u64(bytes: &[u8]) -> ZResult<Vec<u64>> {
    if !bytes.len().is_multiple_of(8) {
        return Err(Status::io_error("invalid u64 segment length"));
    }
    let mut out = Vec::with_capacity(bytes.len() / 8);
    for chunk in bytes.as_chunks::<8>().0 {
        out.push(u64::from_le_bytes(*chunk));
    }
    Ok(out)
}

fn parse_le_u32(bytes: &[u8]) -> ZResult<Vec<u32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(Status::io_error("invalid u32 segment length"));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.as_chunks::<4>().0 {
        out.push(u32::from_le_bytes(*chunk));
    }
    Ok(out)
}

fn parse_le_f32(bytes: &[u8]) -> ZResult<Vec<f32>> {
    let u32s = parse_le_u32(bytes)?;
    Ok(u32s.into_iter().map(f32::from_bits).collect())
}

pub(crate) fn segment_array_u64(bytes: SegmentBytes) -> ZResult<SegmentArray<u64>> {
    let raw = bytes.as_slice();
    if !raw.len().is_multiple_of(8) {
        return Err(Status::io_error("invalid u64 segment length"));
    }
    if cfg!(target_endian = "little")
        && (raw.as_ptr() as usize).is_multiple_of(std::mem::align_of::<u64>())
    {
        return Ok(SegmentArray::Borrowed {
            len: raw.len() / 8,
            bytes,
            _p: PhantomData,
        });
    }
    Ok(SegmentArray::Owned(parse_le_u64(raw)?))
}

pub(crate) fn segment_array_u32(bytes: SegmentBytes) -> ZResult<SegmentArray<u32>> {
    let raw = bytes.as_slice();
    if !raw.len().is_multiple_of(4) {
        return Err(Status::io_error("invalid u32 segment length"));
    }
    if cfg!(target_endian = "little")
        && (raw.as_ptr() as usize).is_multiple_of(std::mem::align_of::<u32>())
    {
        return Ok(SegmentArray::Borrowed {
            len: raw.len() / 4,
            bytes,
            _p: PhantomData,
        });
    }
    Ok(SegmentArray::Owned(parse_le_u32(raw)?))
}

pub(crate) fn segment_array_f32(bytes: SegmentBytes) -> ZResult<SegmentArray<f32>> {
    let raw = bytes.as_slice();
    if !raw.len().is_multiple_of(4) {
        return Err(Status::io_error("invalid f32 segment length"));
    }
    if cfg!(target_endian = "little")
        && (raw.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>())
    {
        return Ok(SegmentArray::Borrowed {
            len: raw.len() / 4,
            bytes,
            _p: PhantomData,
        });
    }
    Ok(SegmentArray::Owned(parse_le_f32(raw)?))
}

/// Serializable index metadata
#[derive(Debug, Clone)]
pub struct IndexMeta {
    /// Vector dimension.
    pub dim: usize,
    /// Number of vectors.
    pub count: usize,
    /// True when the vectors are stored column by column.
    pub column_major: bool,
    /// Encoding of the stored vectors.
    pub quantize: QuantizeType,
    /// Bytes of one encoded vector.
    pub vec_bytes: usize,
    /// Storage kind discriminator for the FEATURES segment.
    ///
    /// 0 = f32, 1 = binary32 (u32 words), 2 = binary64 (u64 words)
    pub storage_kind: u32,
}

impl IndexMeta {
    /// Encodes the metadata as META segment bytes.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(self.dim as u64).to_le_bytes());
        buf.extend_from_slice(&(self.count as u64).to_le_bytes());
        buf.push(self.column_major as u8);
        buf.extend_from_slice(&(self.quantize as u32).to_le_bytes());
        buf.extend_from_slice(&(self.vec_bytes as u32).to_le_bytes());
        buf.extend_from_slice(&self.storage_kind.to_le_bytes());
        buf
    }

    /// Decodes metadata from META segment bytes.
    pub fn deserialize(data: &[u8]) -> ZResult<Self> {
        let mut cur = Cursor::new(data);
        let dim = read_u64(&mut cur)? as usize;
        let count = read_u64(&mut cur)? as usize;
        let column_major = read_u8(&mut cur)? != 0;
        let quantize = quantize_type_from_u32(read_u32(&mut cur)?);
        let vec_bytes = read_u32(&mut cur)? as usize;
        let storage_kind = read_u32(&mut cur)?;
        Ok(IndexMeta {
            dim,
            count,
            column_major,
            quantize,
            vec_bytes,
            storage_kind,
        })
    }
}

/// Trait for reading index segments
pub trait StorageReader: Send + Sync {
    /// Returns the bytes of the segment `name`.
    fn read_segment(&self, name: &str) -> ZResult<SegmentBytes>;
    /// Returns true when the segment `name` exists.
    fn exists(&self, name: &str) -> bool;
}

/// Trait for writing index segments
pub trait StorageWriter: Send + Sync {
    /// Writes `data` as the segment `name`.
    fn write_segment(&mut self, name: &str, data: &[u8]) -> ZResult<()>;
}

/// In-memory storage for testing
#[derive(Default)]
pub struct MemoryStorage {
    data: std::collections::HashMap<String, Vec<u8>>,
}

impl MemoryStorage {
    /// Creates an empty storage.
    pub fn new() -> Self {
        MemoryStorage {
            data: std::collections::HashMap::new(),
        }
    }
}

impl StorageReader for MemoryStorage {
    fn read_segment(&self, name: &str) -> ZResult<SegmentBytes> {
        self.data
            .get(name)
            .cloned()
            .map(SegmentBytes::Owned)
            .ok_or_else(|| Status::not_found(format!("segment '{}' not found", name)))
    }

    fn exists(&self, name: &str) -> bool {
        self.data.contains_key(name)
    }
}

impl StorageWriter for MemoryStorage {
    fn write_segment(&mut self, name: &str, data: &[u8]) -> ZResult<()> {
        self.data.insert(name.to_string(), data.to_vec());
        Ok(())
    }
}

/// FLAT index builder (append-only during construction)
pub struct FlatBuilder {
    params: FlatIndexParams,
    dim: usize,
    keys: Vec<u64>,
    vectors: Vec<f32>, // row-major
}

impl FlatBuilder {
    /// Creates an empty builder for `dim`-dimensional vectors.
    pub fn new(dim: usize, params: FlatIndexParams) -> Self {
        FlatBuilder {
            params,
            dim,
            keys: Vec::new(),
            vectors: Vec::new(),
        }
    }

    /// Adds `vector` under `key`; fails if its length differs from the dimension.
    pub fn add(&mut self, key: u64, vector: &[f32]) -> ZResult<()> {
        if vector.len() != self.dim {
            return Err(Status::invalid_argument(format!(
                "expected dim {}, got {}",
                self.dim,
                vector.len()
            )));
        }
        self.keys.push(key);
        self.vectors.extend_from_slice(vector);
        Ok(())
    }

    /// Returns the number of vectors added.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Returns true when no vector was added.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Writes the index segments to `storage`.
    pub fn dump(&self, storage: &mut dyn StorageWriter) -> ZResult<()> {
        storage.write_segment(SEGMENT_KEYS, &u64s_to_le(&self.keys))?;

        // Write vectors (either raw f32 or quantized bytes)
        let mut vec_buf: Vec<u8> =
            Vec::with_capacity(self.keys.len() * bytes_per_vector(self.params.quantize, self.dim));
        for v in self.vectors.chunks_exact(self.dim) {
            quantize_append(self.params.quantize, v, &mut vec_buf);
        }
        storage.write_segment(SEGMENT_VECTORS, &vec_buf)?;

        let meta = IndexMeta {
            dim: self.dim,
            count: self.keys.len(),
            column_major: self.params.column_major,
            quantize: self.params.quantize,
            vec_bytes: bytes_per_vector(self.params.quantize, self.dim),
            storage_kind: 0,
        };
        storage.write_segment(SEGMENT_META, &meta.serialize())?;

        Ok(())
    }
}

/// FLAT index builder for binary vectors represented as u32 words (Hamming distance).
pub struct FlatBinary32Builder {
    params: FlatIndexParams,
    dim: usize,
    keys: Vec<u64>,
    vectors: Vec<u32>, // row-major u32 words
}

impl FlatBinary32Builder {
    /// Creates an empty builder for binary vectors of dimension `dim`.
    pub fn new(dim: usize, params: FlatIndexParams) -> Self {
        FlatBinary32Builder {
            params,
            dim,
            keys: Vec::new(),
            vectors: Vec::new(),
        }
    }

    /// Adds the binary vector `vector` under `key`.
    pub fn add(&mut self, key: u64, vector: &[u32]) -> ZResult<()> {
        if self.params.quantize != QuantizeType::Undefined {
            return Err(Status::invalid_argument(
                "binary flat index does not support quantize",
            ));
        }
        if self.params.metric != finch_types::MetricType::Hamming {
            return Err(Status::invalid_argument(
                "binary flat index requires Hamming metric",
            ));
        }
        if vector.len() != self.dim {
            return Err(Status::invalid_argument(format!(
                "expected dim {}, got {}",
                self.dim,
                vector.len()
            )));
        }
        self.keys.push(key);
        self.vectors.extend_from_slice(vector);
        Ok(())
    }

    /// Writes the index segments to `storage`.
    pub fn dump(&self, storage: &mut dyn StorageWriter) -> ZResult<()> {
        storage.write_segment(SEGMENT_KEYS, &u64s_to_le(&self.keys))?;

        let mut vec_buf: Vec<u8> = Vec::with_capacity(self.keys.len() * self.dim * 4);
        for v in self.vectors.chunks_exact(self.dim) {
            for &w in v {
                vec_buf.extend_from_slice(&w.to_le_bytes());
            }
        }
        storage.write_segment(SEGMENT_VECTORS, &vec_buf)?;

        let meta = IndexMeta {
            dim: self.dim,
            count: self.keys.len(),
            column_major: false,
            quantize: QuantizeType::Undefined,
            vec_bytes: self.dim * 4,
            storage_kind: 1,
        };
        storage.write_segment(SEGMENT_META, &meta.serialize())?;
        Ok(())
    }
}

/// FLAT index builder for binary vectors represented as u64 words (Hamming distance).
pub struct FlatBinary64Builder {
    params: FlatIndexParams,
    dim: usize,
    keys: Vec<u64>,
    vectors: Vec<u64>, // row-major u64 words
}

impl FlatBinary64Builder {
    /// Creates an empty builder for binary vectors of dimension `dim`.
    pub fn new(dim: usize, params: FlatIndexParams) -> Self {
        FlatBinary64Builder {
            params,
            dim,
            keys: Vec::new(),
            vectors: Vec::new(),
        }
    }

    /// Adds the binary vector `vector` under `key`.
    pub fn add(&mut self, key: u64, vector: &[u64]) -> ZResult<()> {
        if self.params.quantize != QuantizeType::Undefined {
            return Err(Status::invalid_argument(
                "binary flat index does not support quantize",
            ));
        }
        if self.params.metric != finch_types::MetricType::Hamming {
            return Err(Status::invalid_argument(
                "binary flat index requires Hamming metric",
            ));
        }
        if vector.len() != self.dim {
            return Err(Status::invalid_argument(format!(
                "expected dim {}, got {}",
                self.dim,
                vector.len()
            )));
        }
        self.keys.push(key);
        self.vectors.extend_from_slice(vector);
        Ok(())
    }

    /// Writes the index segments to `storage`.
    pub fn dump(&self, storage: &mut dyn StorageWriter) -> ZResult<()> {
        storage.write_segment(SEGMENT_KEYS, &u64s_to_le(&self.keys))?;

        let mut vec_buf: Vec<u8> = Vec::with_capacity(self.keys.len() * self.dim * 8);
        for v in self.vectors.chunks_exact(self.dim) {
            for &w in v {
                vec_buf.extend_from_slice(&w.to_le_bytes());
            }
        }
        storage.write_segment(SEGMENT_VECTORS, &vec_buf)?;

        let meta = IndexMeta {
            dim: self.dim,
            count: self.keys.len(),
            column_major: false,
            quantize: QuantizeType::Undefined,
            vec_bytes: self.dim * 8,
            storage_kind: 2,
        };
        storage.write_segment(SEGMENT_META, &meta.serialize())?;
        Ok(())
    }
}

/// FLAT binary index searcher (u32 words, Hamming distance).
pub struct FlatBinary32Searcher {
    keys: SegmentArray<u64>,
    vectors: SegmentArray<u32>,
    dim: usize,
    count: usize,
}

impl FlatBinary32Searcher {
    /// Loads the index from `storage`.
    pub fn load(storage: &dyn StorageReader, params: &FlatIndexParams) -> ZResult<Self> {
        if params.quantize != QuantizeType::Undefined {
            return Err(Status::invalid_argument(
                "binary flat index does not support quantize",
            ));
        }
        if params.metric != finch_types::MetricType::Hamming {
            return Err(Status::invalid_argument(
                "binary flat index requires Hamming metric",
            ));
        }

        let meta_data = storage.read_segment(SEGMENT_META)?;
        let meta = IndexMeta::deserialize(meta_data.as_slice())?;
        if meta.storage_kind != 1 {
            return Err(Status::io_error("flat binary32 meta storage_kind mismatch"));
        }

        let keys_data = storage.read_segment(SEGMENT_KEYS)?;
        let keys = segment_array_u64(keys_data)?;
        if keys.len() != meta.count {
            return Err(Status::io_error("flat keys length mismatch"));
        }

        let vec_data = storage.read_segment(SEGMENT_VECTORS)?;
        if vec_data.as_slice().len() != meta.count * meta.dim * 4 {
            return Err(Status::io_error("flat binary32 vectors length mismatch"));
        }
        let vectors = segment_array_u32(vec_data)?;
        if vectors.len() != meta.count * meta.dim {
            return Err(Status::io_error("flat binary32 vectors length mismatch"));
        }

        Ok(FlatBinary32Searcher {
            keys,
            vectors,
            dim: meta.dim,
            count: meta.count,
        })
    }

    /// Returns up to `topk` keys and distances that pass `filter`, nearest first.
    pub fn search(
        &self,
        query: &[u32],
        topk: usize,
        filter: Option<&dyn DocFilter>,
    ) -> ZResult<Vec<(u64, f32)>> {
        check_query_dim(query.len(), self.dim)?;

        let mut heap = TopkHeap::new(topk);
        let vecs = self.vectors.as_slice();
        let keys = self.keys.as_slice();
        for i in 0..self.count {
            let key = keys[i];
            let allowed = doc_filter_allows(filter, key)?;
            if !allowed {
                continue;
            }
            let row = &vecs[i * self.dim..(i + 1) * self.dim];
            let mut acc: u32 = 0;
            for j in 0..self.dim {
                acc = acc.wrapping_add((row[j] ^ query[j]).count_ones());
            }
            heap.push(acc as f32, key);
        }

        Ok(heap.into_sorted())
    }
}

/// FLAT binary index searcher (u64 words, Hamming distance).
pub struct FlatBinary64Searcher {
    keys: SegmentArray<u64>,
    vectors: SegmentArray<u64>,
    dim: usize,
    count: usize,
}

impl FlatBinary64Searcher {
    /// Loads the index from `storage`.
    pub fn load(storage: &dyn StorageReader, params: &FlatIndexParams) -> ZResult<Self> {
        if params.quantize != QuantizeType::Undefined {
            return Err(Status::invalid_argument(
                "binary flat index does not support quantize",
            ));
        }
        if params.metric != finch_types::MetricType::Hamming {
            return Err(Status::invalid_argument(
                "binary flat index requires Hamming metric",
            ));
        }

        let meta_data = storage.read_segment(SEGMENT_META)?;
        let meta = IndexMeta::deserialize(meta_data.as_slice())?;
        if meta.storage_kind != 2 {
            return Err(Status::io_error("flat binary64 meta storage_kind mismatch"));
        }

        let keys_data = storage.read_segment(SEGMENT_KEYS)?;
        let keys = segment_array_u64(keys_data)?;
        if keys.len() != meta.count {
            return Err(Status::io_error("flat keys length mismatch"));
        }

        let vec_data = storage.read_segment(SEGMENT_VECTORS)?;
        if vec_data.as_slice().len() != meta.count * meta.dim * 8 {
            return Err(Status::io_error("flat binary64 vectors length mismatch"));
        }
        let vectors = segment_array_u64(vec_data)?;
        if vectors.len() != meta.count * meta.dim {
            return Err(Status::io_error("flat binary64 vectors length mismatch"));
        }

        Ok(FlatBinary64Searcher {
            keys,
            vectors,
            dim: meta.dim,
            count: meta.count,
        })
    }

    /// Returns up to `topk` keys and distances that pass `filter`, nearest first.
    pub fn search(
        &self,
        query: &[u64],
        topk: usize,
        filter: Option<&dyn DocFilter>,
    ) -> ZResult<Vec<(u64, f32)>> {
        check_query_dim(query.len(), self.dim)?;

        let mut heap = TopkHeap::new(topk);
        let vecs = self.vectors.as_slice();
        let keys = self.keys.as_slice();
        for i in 0..self.count {
            let key = keys[i];
            let allowed = doc_filter_allows(filter, key)?;
            if !allowed {
                continue;
            }
            let row = &vecs[i * self.dim..(i + 1) * self.dim];
            let mut acc: u32 = 0;
            for j in 0..self.dim {
                acc = acc.wrapping_add((row[j] ^ query[j]).count_ones());
            }
            heap.push(acc as f32, key);
        }

        Ok(heap.into_sorted())
    }
}

fn merge_topk(mut a: TopkHeap, b: TopkHeap) -> TopkHeap {
    for (key, dist) in b.into_sorted() {
        a.push(dist, key);
    }
    a
}

struct QuantizedRows<'a> {
    raw: &'a [u8],
    quantize: QuantizeType,
    stride: usize,
}

/// Shape of a vector segment: `count` rows of `dim` f32s, or of `vec_bytes` quantized bytes.
pub(super) struct RowLayout {
    pub(super) count: usize,
    pub(super) dim: usize,
    pub(super) quantize: QuantizeType,
    pub(super) vec_bytes: usize,
}

/// Validates a vector segment against `layout`; `label` prefixes the error messages.
pub(super) fn load_vector_rows(
    vec_data: SegmentBytes,
    layout: &RowLayout,
    label: &str,
) -> ZResult<FlatVectorStorage> {
    if layout.quantize == QuantizeType::Undefined {
        let vectors = segment_array_f32(vec_data)?;
        if vectors.len() != layout.count * layout.dim {
            return Err(Status::io_error(format!("{label} vectors length mismatch")));
        }
        return Ok(FlatVectorStorage::F32(vectors));
    }
    let expected = layout
        .count
        .checked_mul(layout.vec_bytes)
        .ok_or_else(|| Status::io_error(format!("{label} vector segment overflow")))?;
    if vec_data.as_slice().len() != expected {
        return Err(Status::io_error(format!(
            "{label} quantized vector length mismatch"
        )));
    }
    Ok(FlatVectorStorage::Quantized {
        bytes: vec_data,
        quantize: layout.quantize,
        stride: layout.vec_bytes,
    })
}

pub(super) enum FlatVectorStorage {
    F32(SegmentArray<f32>),
    Quantized {
        bytes: SegmentBytes,
        quantize: QuantizeType,
        stride: usize,
    },
}

/// FLAT index searcher (loaded from storage, read-only)
pub struct FlatSearcher {
    keys: SegmentArray<u64>,
    vectors: FlatVectorStorage,
    dim: usize,
    count: usize,
    metric: Box<dyn Metric>,
}

impl FlatSearcher {
    /// Loads the index from `storage`.
    pub fn load(storage: &dyn StorageReader, params: &FlatIndexParams) -> ZResult<Self> {
        let meta_data = storage.read_segment(SEGMENT_META)?;
        let meta = IndexMeta::deserialize(meta_data.as_slice())?;
        if meta.storage_kind != 0 {
            return Err(Status::io_error("flat meta storage_kind is not f32"));
        }

        let keys_data = storage.read_segment(SEGMENT_KEYS)?;
        let keys = segment_array_u64(keys_data)?;
        if keys.len() != meta.count {
            return Err(Status::io_error("flat keys length mismatch"));
        }

        let layout = RowLayout {
            count: meta.count,
            dim: meta.dim,
            quantize: meta.quantize,
            vec_bytes: meta.vec_bytes,
        };
        let vectors = load_vector_rows(storage.read_segment(SEGMENT_VECTORS)?, &layout, "flat")?;

        Ok(FlatSearcher {
            keys,
            vectors,
            dim: meta.dim,
            count: meta.count,
            metric: make_metric(params.metric),
        })
    }

    /// Returns up to `topk` keys and distances that pass `filter`, nearest first.
    pub fn search(
        &self,
        query: &[f32],
        topk: usize,
        filter: Option<&dyn DocFilter>,
    ) -> ZResult<Vec<(u64, f32)>> {
        check_query_dim(query.len(), self.dim)?;

        match &self.vectors {
            FlatVectorStorage::F32(vectors) => {
                // Threshold for parallelism: only worth spawning threads for
                // large scans (rayon overhead ~2-5 µs per task).
                const PAR_THRESHOLD: usize = 8192;

                if self.count >= PAR_THRESHOLD && filter.is_none() {
                    return Ok(self.par_scan_f32(vectors.as_slice(), query, topk));
                }
                self.scan_f32(vectors.as_slice(), query, topk, filter)
            }
            FlatVectorStorage::Quantized {
                bytes,
                quantize,
                stride,
            } => {
                let rows = QuantizedRows {
                    raw: bytes.as_slice(),
                    quantize: *quantize,
                    stride: *stride,
                };
                self.scan_quantized(&rows, query, topk, filter)
            }
        }
    }

    /// Unfiltered parallel scan: each chunk builds a local TopkHeap, then they merge.
    fn par_scan_f32(&self, vecs: &[f32], query: &[f32], topk: usize) -> Vec<(u64, f32)> {
        let keys = self.keys.as_slice();
        let dim = self.dim;
        let count = self.count;
        let chunk_size = (count / rayon::current_num_threads()).max(1024);
        let heap = (0..count)
            .into_par_iter()
            .with_min_len(chunk_size)
            .fold(
                || TopkHeap::new(topk),
                |mut local_heap, i| {
                    let row = &vecs[i * dim..(i + 1) * dim];
                    let d = self.metric.distance(row, query);
                    local_heap.push(d, keys[i]);
                    local_heap
                },
            )
            .reduce(|| TopkHeap::new(topk), merge_topk);
        heap.into_sorted()
    }

    fn scan_f32(
        &self,
        vecs: &[f32],
        query: &[f32],
        topk: usize,
        filter: Option<&dyn DocFilter>,
    ) -> ZResult<Vec<(u64, f32)>> {
        let keys = self.keys.as_slice();
        let count = self.count;
        let mut dists = vec![0.0f32; count];
        self.metric
            .batch_distance(vecs, query, count, self.dim, &mut dists);
        let mut heap = TopkHeap::new(topk);
        for (i, &d) in dists.iter().enumerate() {
            let key = keys[i];
            let allowed = doc_filter_allows(filter, key)?;
            if allowed {
                heap.push(d, key);
            }
        }
        Ok(heap.into_sorted())
    }

    fn scan_quantized(
        &self,
        rows: &QuantizedRows<'_>,
        query: &[f32],
        topk: usize,
        filter: Option<&dyn DocFilter>,
    ) -> ZResult<Vec<(u64, f32)>> {
        let mt = self.metric.metric_type();
        let q2 = query_sq_norm(mt, query, self.dim);
        let (raw, stride) = (rows.raw, rows.stride);
        let mut heap = TopkHeap::new(topk);
        for (i, &key) in self.keys.as_slice().iter().enumerate().take(self.count) {
            let allowed = doc_filter_allows(filter, key)?;
            if !allowed {
                continue;
            }
            let off = i * stride;
            let d = distance_to_quantized_with_query_sq_norm(
                mt,
                query,
                q2,
                &raw[off..off + stride],
                self.dim,
                rows.quantize,
            );
            heap.push(d, key);
        }
        Ok(heap.into_sorted())
    }

    /// Searches only the vectors stored under `keys`.
    pub fn search_by_keys(
        &self,
        query: &[f32],
        topk: usize,
        keys: &[u64],
    ) -> ZResult<Vec<(u64, f32)>> {
        let key_set: std::collections::HashSet<u64> = keys.iter().cloned().collect();
        let filter = KeySetFilter(key_set);
        self.search(query, topk, Some(&filter))
    }

    /// Returns the number of indexed vectors.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns true when the index holds no vector.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

struct KeySetFilter(std::collections::HashSet<u64>);
impl DocFilter for KeySetFilter {
    fn is_valid(&self, doc_id: u64) -> ZResult<bool> {
        Ok(self.0.contains(&doc_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use finch_types::{MetricType, QuantizeType, Status};

    struct FailingFilter;

    impl DocFilter for FailingFilter {
        fn is_valid(&self, _doc_id: u64) -> ZResult<bool> {
            Err(Status::io_error("filter failed"))
        }
    }

    #[test]
    fn test_flat_build_search() {
        let params = FlatIndexParams::new(MetricType::L2);
        let dim = 4;
        let mut builder = FlatBuilder::new(dim, params.clone());

        builder.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        builder.add(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        builder.add(3, &[0.0, 0.0, 1.0, 0.0]).unwrap();

        let mut storage = MemoryStorage::new();
        builder.dump(&mut storage).unwrap();

        let searcher = FlatSearcher::load(&storage, &params).unwrap();
        let results = searcher.search(&[1.0, 0.0, 0.0, 0.0], 2, None).unwrap();

        assert_eq!(results[0].0, 1);
        assert!(results[0].1 < 1e-6);
    }

    #[test]
    fn test_flat_search_propagates_filter_errors() {
        let params = FlatIndexParams::new(MetricType::L2);
        let mut builder = FlatBuilder::new(2, params.clone());
        builder.add(1, &[1.0, 0.0]).unwrap();

        let mut storage = MemoryStorage::new();
        builder.dump(&mut storage).unwrap();

        let searcher = FlatSearcher::load(&storage, &params).unwrap();
        let err = searcher
            .search(&[1.0, 0.0], 1, Some(&FailingFilter))
            .unwrap_err();
        assert!(
            err.message.contains("filter failed"),
            "expected filter error to propagate, got {err:?}"
        );
    }

    #[test]
    fn test_flat_build_search_quantized_int8() {
        let mut params = FlatIndexParams::new(MetricType::L2);
        params.quantize = QuantizeType::Int8;
        let dim = 4;
        let mut builder = FlatBuilder::new(dim, params.clone());

        builder.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        builder.add(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        builder.add(3, &[0.0, 0.0, 1.0, 0.0]).unwrap();

        let mut storage = MemoryStorage::new();
        builder.dump(&mut storage).unwrap();

        let searcher = FlatSearcher::load(&storage, &params).unwrap();
        let results = searcher.search(&[1.0, 0.0, 0.0, 0.0], 1, None).unwrap();

        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn test_flat_binary32_build_search_hamming() {
        let params = FlatIndexParams::new(MetricType::Hamming);
        let dim = 2;
        let mut builder = FlatBinary32Builder::new(dim, params.clone());

        builder.add(1, &[0u32, 0]).unwrap();
        builder.add(2, &[1u32, 0]).unwrap();

        let mut storage = MemoryStorage::new();
        builder.dump(&mut storage).unwrap();

        let searcher = FlatBinary32Searcher::load(&storage, &params).unwrap();
        let results = searcher.search(&[0u32, 0], 2, None).unwrap();
        assert_eq!(results[0].0, 1);
        assert!(results[0].1 <= results[1].1);
    }

    #[test]
    fn test_flat_binary64_build_search_hamming() {
        let params = FlatIndexParams::new(MetricType::Hamming);
        let dim = 1;
        let mut builder = FlatBinary64Builder::new(dim, params.clone());

        builder.add(1, &[0u64]).unwrap();
        builder.add(2, &[1u64]).unwrap();

        let mut storage = MemoryStorage::new();
        builder.dump(&mut storage).unwrap();

        let searcher = FlatBinary64Searcher::load(&storage, &params).unwrap();
        let results = searcher.search(&[0u64], 2, None).unwrap();
        assert_eq!(results[0].0, 1);
        assert!(results[0].1 <= results[1].1);
    }
}

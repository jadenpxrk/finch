//! FLAT (brute-force) sparse vector index

use super::{doc_filter_allows, DocFilter, TopkHeap};
use finch_types::{FlatIndexParams, QuantizeType, Status, ZResult};
use half::f16;
use std::io::Cursor;

/// A sparse vector with sorted indices
#[derive(Debug, Clone)]
pub struct SparseVector {
    /// Dimension indices of the nonzero values, in ascending order.
    pub indices: Vec<u32>,
    /// The nonzero values, one for each index.
    pub values: SparseValues,
}

/// The nonzero values of a sparse vector, at full or half precision.
#[derive(Debug, Clone)]
pub enum SparseValues {
    /// Full-precision values.
    F32(Vec<f32>),
    /// Half-precision values.
    F16(Vec<f16>),
}

#[inline]
fn value_at(values: &SparseValues, i: usize) -> f32 {
    match values {
        SparseValues::F32(v) => v[i],
        SparseValues::F16(v) => v[i].to_f32(),
    }
}

#[inline]
fn f16_bits_at(values: &SparseValues, i: usize) -> u16 {
    match values {
        SparseValues::F16(v) => v[i].to_bits(),
        SparseValues::F32(v) => f16::from_f32(v[i]).to_bits(),
    }
}

impl SparseVector {
    /// Creates a sparse vector from sorted indices and their f32 values.
    pub fn new(indices: Vec<u32>, values: Vec<f32>) -> Self {
        SparseVector {
            indices,
            values: SparseValues::F32(values),
        }
    }

    pub(crate) fn new_f16(indices: Vec<u32>, values: Vec<f16>) -> Self {
        SparseVector {
            indices,
            values: SparseValues::F16(values),
        }
    }

    /// Returns a copy that holds its values at half precision.
    pub fn quantize_to_f16(&self) -> Self {
        match &self.values {
            SparseValues::F16(v) => SparseVector::new_f16(self.indices.clone(), v.clone()),
            SparseValues::F32(v) => {
                let halfs: Vec<f16> = v.iter().map(|&x| f16::from_f32(x)).collect();
                SparseVector::new_f16(self.indices.clone(), halfs)
            }
        }
    }

    /// Sparse dot product with another sparse vector
    pub fn dot(&self, other: &SparseVector) -> f32 {
        let mut result = 0.0f32;
        let mut i = 0;
        let mut j = 0;
        while i < self.indices.len() && j < other.indices.len() {
            match self.indices[i].cmp(&other.indices[j]) {
                std::cmp::Ordering::Equal => {
                    result += value_at(&self.values, i) * value_at(&other.values, j);
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }
        result
    }

    #[inline]
    pub(crate) fn dot_encoded(&self, encoded: &[u8], quantize: QuantizeType) -> f32 {
        self.dot_encoded_checked(encoded, quantize).unwrap_or(0.0)
    }

    /// `None` when `encoded` is malformed or either side is empty.
    #[inline]
    pub(crate) fn dot_encoded_checked(
        &self,
        encoded: &[u8],
        quantize: QuantizeType,
    ) -> Option<f32> {
        if encoded.len() < 4 {
            return None;
        }
        let n = u32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]) as usize;
        let idx_bytes = n.saturating_mul(4);
        let val_bytes = match quantize {
            QuantizeType::Fp16 => n.saturating_mul(2),
            _ => n.saturating_mul(4),
        };
        let total = 4usize.saturating_add(idx_bytes).saturating_add(val_bytes);
        if total != encoded.len() {
            return None;
        }

        let q_idx = self.indices.as_slice();
        if q_idx.is_empty() || n == 0 {
            return None;
        }

        let mut idx_pos = 4usize;
        let mut val_pos = 4usize + idx_bytes;
        let mut qi = 0usize;
        let mut dot = 0.0f32;

        for _ in 0..n {
            if qi >= q_idx.len() {
                break;
            }
            let b = &encoded[idx_pos..idx_pos + 4];
            let idx = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            idx_pos += 4;

            while qi < q_idx.len() && q_idx[qi] < idx {
                qi += 1;
            }
            if qi >= q_idx.len() {
                break;
            }

            let v = match quantize {
                QuantizeType::Fp16 => {
                    let bits = u16::from_le_bytes([encoded[val_pos], encoded[val_pos + 1]]);
                    val_pos += 2;
                    f16::from_bits(bits).to_f32()
                }
                _ => {
                    let vb = &encoded[val_pos..val_pos + 4];
                    val_pos += 4;
                    f32::from_le_bytes([vb[0], vb[1], vb[2], vb[3]])
                }
            };
            if q_idx[qi] == idx {
                dot += value_at(&self.values, qi) * v;
                qi += 1;
            }
        }

        Some(dot)
    }

    /// Serialize to bytes: [count: u32][indices: u32*count][values: {f32|f16}*count]
    pub(crate) fn serialize_with_quantize(&self, quantize: QuantizeType) -> ZResult<Vec<u8>> {
        let n = self.indices.len();
        let mut buf = Vec::with_capacity(4 + n * 8);
        buf.extend_from_slice(&(n as u32).to_le_bytes());
        for &idx in &self.indices {
            buf.extend_from_slice(&idx.to_le_bytes());
        }

        match quantize {
            QuantizeType::Undefined => {
                for i in 0..n {
                    buf.extend_from_slice(&value_at(&self.values, i).to_le_bytes());
                }
            }
            QuantizeType::Fp16 => {
                for i in 0..n {
                    buf.extend_from_slice(&f16_bits_at(&self.values, i).to_le_bytes());
                }
            }
            _ => {
                return Err(Status::invalid_argument(
                    "sparse quantize only supports Fp16",
                ));
            }
        }
        Ok(buf)
    }

    pub(crate) fn deserialize_with_quantize(
        data: &[u8],
        quantize: QuantizeType,
    ) -> ZResult<(Self, usize)> {
        let mut cur = Cursor::new(data);
        let n = read_u32(&mut cur)? as usize;
        let mut indices = Vec::with_capacity(n);
        for _ in 0..n {
            indices.push(read_u32(&mut cur)?);
        }

        match quantize {
            QuantizeType::Undefined => {
                let mut values = Vec::with_capacity(n);
                for _ in 0..n {
                    values.push(read_f32(&mut cur)?);
                }
                let consumed = 4 + n * 8;
                Ok((SparseVector::new(indices, values), consumed))
            }
            QuantizeType::Fp16 => {
                let mut values = Vec::with_capacity(n);
                for _ in 0..n {
                    values.push(f16::from_bits(read_u16(&mut cur)?));
                }
                let consumed = 4 + n * 6;
                Ok((SparseVector::new_f16(indices, values), consumed))
            }
            _ => Err(Status::invalid_argument(
                "sparse quantize only supports Fp16",
            )),
        }
    }

    /// Encodes the vector at full precision; a vector it cannot encode gives an empty buffer.
    pub fn serialize(&self) -> Vec<u8> {
        self.serialize_with_quantize(QuantizeType::Undefined)
            .unwrap_or_else(|_| Vec::new())
    }

    /// Decodes one vector and returns it with the number of bytes it used.
    pub fn deserialize(data: &[u8]) -> ZResult<(Self, usize)> {
        Self::deserialize_with_quantize(data, QuantizeType::Undefined)
    }
}

use super::codec::{read_f32, read_u16, read_u32, read_u64, u64s_to_le};
use super::flat::{segment_array_u64, SegmentArray, SegmentBytes, StorageReader, StorageWriter};
use crate::quantizer::quantize_type_from_u32;

const SEGMENT_KEYS: &str = "SPARSE_KEYS";
const SEGMENT_VECTORS: &str = "SPARSE_VECTORS";
const SEGMENT_OFFSETS: &str = "SPARSE_OFFSETS";
const SEGMENT_META: &str = "SPARSE_META";

/// Segment names and error label of one sparse index family.
pub(super) struct SparseSegments {
    pub(super) label: &'static str,
    pub(super) keys: &'static str,
    pub(super) vectors: &'static str,
    pub(super) offsets: &'static str,
}

const FLAT_SEGMENTS: SparseSegments = SparseSegments {
    label: "SPARSE",
    keys: SEGMENT_KEYS,
    vectors: SEGMENT_VECTORS,
    offsets: SEGMENT_OFFSETS,
};

/// Writes keys, encoded vectors, and their `count + 1` byte offsets, in that order.
pub(super) fn dump_sparse_rows(
    storage: &mut dyn StorageWriter,
    segments: &SparseSegments,
    keys: &[u64],
    vectors: &[SparseVector],
    quantize: QuantizeType,
) -> ZResult<()> {
    storage.write_segment(segments.keys, &u64s_to_le(keys))?;

    let mut vec_buf = Vec::new();
    let mut offsets = Vec::with_capacity(vectors.len() + 1);
    let mut end = 0u64;
    offsets.push(end);
    for v in vectors {
        let bytes = v.serialize_with_quantize(quantize)?;
        end += bytes.len() as u64;
        offsets.push(end);
        vec_buf.extend_from_slice(&bytes);
    }
    storage.write_segment(segments.vectors, &vec_buf)?;
    storage.write_segment(segments.offsets, &u64s_to_le(&offsets))
}

/// Keys and offset-indexed encoded vectors of a loaded sparse index.
pub(super) struct SparseRows {
    pub(super) keys: SegmentArray<u64>,
    vec_bytes: SegmentBytes,
    offsets: SegmentArray<u64>, // count + 1
}

impl SparseRows {
    pub(super) fn load(
        storage: &dyn StorageReader,
        segments: &SparseSegments,
        count: usize,
        quantize: QuantizeType,
    ) -> ZResult<Self> {
        let keys = segment_array_u64(storage.read_segment(segments.keys)?)?;
        if keys.len() != count {
            return Err(Status::io_error(format!(
                "{} keys length mismatch",
                segments.label
            )));
        }

        let vec_bytes = storage.read_segment(segments.vectors)?;
        let offsets = segment_array_u64(storage.read_segment(segments.offsets)?)?;
        validate_vector_encoding(
            segments.label,
            vec_bytes.as_slice(),
            offsets.as_slice(),
            count,
            quantize,
        )?;
        Ok(SparseRows {
            keys,
            vec_bytes,
            offsets,
        })
    }

    #[inline]
    pub(super) fn vector(&self, i: usize) -> &[u8] {
        let offs = self.offsets.as_slice();
        let start = offs[i] as usize;
        let end = offs[i + 1] as usize;
        &self.vec_bytes.as_slice()[start..end]
    }
}

fn validate_vector_encoding(
    label: &str,
    vec_bytes: &[u8],
    offsets: &[u64],
    count: usize,
    quantize: QuantizeType,
) -> ZResult<()> {
    if offsets.len() != count + 1 {
        return Err(Status::io_error(format!("{label} offsets length mismatch")));
    }
    let total = vec_bytes.len() as u64;
    for i in 0..count {
        let start = offsets[i];
        let end = offsets[i + 1];
        if start > end || end > total {
            return Err(Status::io_error(format!(
                "{label} vector offsets out of range"
            )));
        }
        let len = (end - start) as usize;
        if len < 4 {
            return Err(Status::io_error(format!("{label} vector entry truncated")));
        }
        let base = start as usize;
        let cnt = u32::from_le_bytes([
            vec_bytes[base],
            vec_bytes[base + 1],
            vec_bytes[base + 2],
            vec_bytes[base + 3],
        ]) as usize;
        let expected = match quantize {
            QuantizeType::Fp16 => 4usize.saturating_add(cnt.saturating_mul(6)),
            _ => 4usize.saturating_add(cnt.saturating_mul(8)),
        };
        if expected != len {
            return Err(Status::io_error(format!(
                "{label} vector entry length mismatch"
            )));
        }
    }
    Ok(())
}

/// Builds a flat index of sparse vectors, which a search scans in full.
#[derive(Default)]
pub struct FlatSparseBuilder {
    keys: Vec<u64>,
    vectors: Vec<SparseVector>,
    quantize: QuantizeType,
}

impl FlatSparseBuilder {
    /// Creates an empty builder that stores values at full precision.
    pub fn new() -> Self {
        FlatSparseBuilder {
            keys: Vec::new(),
            vectors: Vec::new(),
            quantize: QuantizeType::Undefined,
        }
    }

    /// Creates an empty builder that stores values in the quantization of `params`.
    pub fn new_with_params(params: FlatIndexParams) -> Self {
        FlatSparseBuilder {
            keys: Vec::new(),
            vectors: Vec::new(),
            quantize: params.quantize,
        }
    }

    /// Adds `vec` under `key`.
    pub fn add(&mut self, key: u64, vec: SparseVector) -> ZResult<()> {
        self.keys.push(key);
        let v = match self.quantize {
            QuantizeType::Fp16 => vec.quantize_to_f16(),
            QuantizeType::Undefined => vec,
            _ => {
                return Err(Status::invalid_argument(
                    "sparse quantize only supports Fp16",
                ))
            }
        };
        self.vectors.push(v);
        Ok(())
    }

    /// Writes the index segments to `storage`.
    pub fn dump(&self, storage: &mut dyn StorageWriter) -> ZResult<()> {
        dump_sparse_rows(
            storage,
            &FLAT_SEGMENTS,
            &self.keys,
            &self.vectors,
            self.quantize,
        )?;

        let mut meta = Vec::new();
        meta.extend_from_slice(&(self.keys.len() as u64).to_le_bytes());
        meta.extend_from_slice(&(self.quantize as u32).to_le_bytes());
        storage.write_segment(SEGMENT_META, &meta)?;

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
}

struct SparseMeta {
    count: usize,
    quantize: QuantizeType,
}

impl SparseMeta {
    fn read(storage: &dyn StorageReader) -> ZResult<Self> {
        let meta = storage.read_segment(SEGMENT_META)?;
        let mut cur = Cursor::new(meta.as_slice());
        let count = read_u64(&mut cur)? as usize;
        let quantize = quantize_type_from_u32(read_u32(&mut cur)?);
        Ok(SparseMeta { count, quantize })
    }
}

/// Searches a flat sparse index by scanning every row.
pub struct FlatSparseSearcher {
    rows: SparseRows,
    quantize: QuantizeType,
}

impl FlatSparseSearcher {
    /// Loads the index from `storage`; fails if its quantization differs from `params`.
    pub fn load_with_params(
        storage: &dyn StorageReader,
        params: &FlatIndexParams,
    ) -> ZResult<Self> {
        let meta = SparseMeta::read(storage)?;
        if meta.quantize != params.quantize {
            return Err(Status::invalid_argument(
                "sparse quantize mismatch between index and params",
            ));
        }
        Ok(FlatSparseSearcher {
            rows: SparseRows::load(storage, &FLAT_SEGMENTS, meta.count, meta.quantize)?,
            quantize: meta.quantize,
        })
    }

    /// Returns up to `topk` keys that pass `filter`, nearest first; the distance is the negative inner product.
    pub fn search(
        &self,
        query: &SparseVector,
        topk: usize,
        filter: Option<&dyn DocFilter>,
    ) -> ZResult<Vec<(u64, f32)>> {
        // An empty query matches nothing; scoring it would rank every row at distance 0.
        if query.indices.is_empty() {
            return Ok(Vec::new());
        }
        let mut heap = TopkHeap::new(topk);
        let keys = self.rows.keys.as_slice();
        for (i, &key) in keys.iter().enumerate() {
            let allowed = doc_filter_allows(filter, key)?;
            if !allowed {
                continue;
            }
            let score = query.dot_encoded(self.rows.vector(i), self.quantize);
            // internal ranking uses a "distance-like" score where smaller is better.
            // For sparse IP, we use neg inner-product (smaller => larger similarity).
            heap.push(-score, key);
        }
        Ok(heap.into_sorted())
    }
}

#[cfg(test)]
mod tests {
    use super::super::flat::MemoryStorage;
    use super::*;

    #[test]
    fn test_sparse_dot() {
        let a = SparseVector::new(vec![0, 2, 4], vec![1.0, 2.0, 3.0]);
        let b = SparseVector::new(vec![0, 1, 2], vec![1.0, 5.0, 2.0]);
        // dot = 1*1 + 2*2 = 5
        assert!((a.dot(&b) - 5.0).abs() < 1e-6);
    }

    fn default_params() -> FlatIndexParams {
        FlatIndexParams {
            metric: finch_types::MetricType::InnerProduct,
            quantize: QuantizeType::Undefined,
            column_major: false,
        }
    }

    #[test]
    fn test_flat_sparse_build_search() {
        let mut builder = FlatSparseBuilder::new();
        builder
            .add(1, SparseVector::new(vec![0, 2], vec![1.0, 1.0]))
            .unwrap();
        builder
            .add(2, SparseVector::new(vec![1, 3], vec![1.0, 1.0]))
            .unwrap();

        let mut storage = MemoryStorage::new();
        builder.dump(&mut storage).unwrap();

        let searcher = FlatSparseSearcher::load_with_params(&storage, &default_params()).unwrap();
        let query = SparseVector::new(vec![0, 2], vec![1.0, 1.0]);
        let results = searcher.search(&query, 2, None).unwrap();
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn empty_query_returns_no_hits() {
        let mut builder = FlatSparseBuilder::new();
        builder
            .add(1, SparseVector::new(vec![0, 2], vec![1.0, 1.0]))
            .unwrap();
        builder.add(2, SparseVector::new(vec![], vec![])).unwrap();
        let mut storage = MemoryStorage::new();
        builder.dump(&mut storage).unwrap();

        let searcher = FlatSparseSearcher::load_with_params(&storage, &default_params()).unwrap();
        let results = searcher
            .search(&SparseVector::new(vec![], vec![]), 2, None)
            .unwrap();
        assert!(results.is_empty(), "{results:?}");
    }

    #[test]
    fn test_flat_sparse_fp16_encoding_round_trip() {
        let params = FlatIndexParams {
            metric: finch_types::MetricType::InnerProduct,
            quantize: QuantizeType::Fp16,
            column_major: false,
        };
        let mut builder = FlatSparseBuilder::new_with_params(params.clone());
        builder
            .add(1, SparseVector::new(vec![0, 2], vec![1.0, 0.25]))
            .unwrap();

        let mut storage = MemoryStorage::new();
        builder.dump(&mut storage).unwrap();

        let searcher = FlatSparseSearcher::load_with_params(&storage, &params).unwrap();
        let query = SparseVector::new(vec![0, 2], vec![1.0, 0.25]);
        let results = searcher.search(&query, 1, None).unwrap();
        assert_eq!(results[0].0, 1);
        // distance-like: negative inner product
        assert!(results[0].1 < 0.0);
    }
}

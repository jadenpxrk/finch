//! Vector index builder: reads docs from a segment and builds index on disk

use crate::segment::persisted::{FileStorageReader, PersistedSegment, VectorIndex};
use crate::segment::writing::WritingSegment;
use finch_core::algorithm::flat::{
    StorageReader as CoreStorageReader, StorageWriter as CoreStorageWriter,
};
use finch_core::{
    FlatBinary32Builder, FlatBinary32Searcher, FlatBinary64Builder, FlatBinary64Searcher,
    FlatBuilder, FlatSearcher, FlatSparseBuilder, FlatSparseSearcher, HnswBuilder, HnswSearcher,
    HnswSparseBuilder, IvfBuilder, IvfSearcher, SparseVector,
};
use finch_storage::{FileStorage, MmapForwardStore};
use finch_types::{
    Doc, FieldSchema, FlatIndexParams, HnswIndexParams, IndexParams, IvfIndexParams, Status,
    ZResult,
};
use std::path::Path;

use finch_storage::forward::decode_sparse_f32;

/// A StorageWriter that delegates to FileStorage
pub struct FileStorageWriter {
    storage: FileStorage,
}

impl FileStorageWriter {
    pub fn new(path: impl AsRef<Path>) -> ZResult<Self> {
        Ok(FileStorageWriter {
            storage: FileStorage::new(path)?,
        })
    }
}

impl CoreStorageWriter for FileStorageWriter {
    fn write_segment(&mut self, name: &str, data: &[u8]) -> ZResult<()> {
        use finch_storage::StorageSegment;
        self.storage.write(name, data)
    }
}

/// Which vector column a persisted segment stores for a field.
#[derive(Clone, Copy, PartialEq, Eq)]
enum VectorKind {
    Binary32,
    Binary64,
    Sparse,
    Dense,
}

/// The forward-store column holding one field's vectors.
struct VectorColumn<'a> {
    forward: &'a MmapForwardStore,
    field: &'a FieldSchema,
    name: String,
    kind: VectorKind,
}

impl<'a> VectorColumn<'a> {
    fn detect(forward: &'a MmapForwardStore, field: &'a FieldSchema) -> ZResult<Self> {
        let candidates = [
            (format!("__bvec32__{}", field.name), VectorKind::Binary32),
            (format!("__bvec64__{}", field.name), VectorKind::Binary64),
            (format!("__svec__{}", field.name), VectorKind::Sparse),
            (format!("__vec__{}", field.name), VectorKind::Dense),
        ];
        for (name, kind) in candidates {
            if forward.has_column(&name) {
                return Ok(VectorColumn {
                    forward,
                    field,
                    name,
                    kind,
                });
            }
        }
        Err(Status::invalid_argument(format!(
            "no vector column found for field '{}'",
            field.name
        )))
    }

    fn require(&self, kind: VectorKind, msg: &'static str) -> ZResult<()> {
        if self.kind != kind {
            return Err(Status::invalid_argument(msg));
        }
        Ok(())
    }

    // A column whose rows are all null still gets an empty index of the schema dimension.
    fn empty_dim(&self) -> ZResult<usize> {
        self.field.dimension.ok_or_else(|| {
            Status::invalid_argument(format!("no vectors found for field '{}'", self.field.name))
        })
    }
}

fn hnsw_concurrency(params: &HnswIndexParams) -> usize {
    params.build_concurrency.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    })
}

fn build_flat_persisted(
    col: &VectorColumn,
    flat_params: &FlatIndexParams,
    writer: &mut FileStorageWriter,
) -> ZResult<()> {
    match col.kind {
        VectorKind::Binary32 => build_flat_b32(col, flat_params, writer),
        VectorKind::Binary64 => build_flat_b64(col, flat_params, writer),
        VectorKind::Sparse => {
            let sparse_params = FlatIndexParams {
                metric: flat_params.metric,
                quantize: flat_params.quantize,
                column_major: false,
            };
            build_flat_sparse(col, sparse_params, writer)
        }
        VectorKind::Dense => build_flat_dense(col, flat_params, writer),
    }
}

fn build_flat_b32(
    col: &VectorColumn,
    flat_params: &FlatIndexParams,
    writer: &mut FileStorageWriter,
) -> ZResult<()> {
    let mut builder: Option<FlatBinary32Builder> = None;
    let mut scratch: Vec<u32> = Vec::new();
    let forward = col.forward;
    forward.scan_large_binary_column_rows(&col.name, |doc_id, bytes| {
        let Some(bytes) = bytes else {
            return Ok(());
        };
        let dim = bytes.len() / 4;
        if dim == 0 || bytes.len() % 4 != 0 {
            return Ok(());
        }
        let b = builder.get_or_insert_with(|| {
            scratch = vec![0u32; dim];
            FlatBinary32Builder::new(dim, flat_params.clone())
        });

        if cfg!(target_endian = "little")
            && (bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<u32>())
        {
            // SAFETY: alignment checked; length is multiple of 4; buffer lives with batch.
            let v = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u32, dim) };
            b.add(doc_id, v)?;
        } else {
            for (dst, chunk) in scratch[..dim].iter_mut().zip(bytes.as_chunks::<4>().0) {
                *dst = u32::from_le_bytes(*chunk);
            }
            b.add(doc_id, &scratch)?;
        }
        Ok(())
    })?;
    let b = match builder {
        Some(b) => b,
        None => FlatBinary32Builder::new(col.empty_dim()?, flat_params.clone()),
    };
    b.dump(writer)
}

fn build_flat_b64(
    col: &VectorColumn,
    flat_params: &FlatIndexParams,
    writer: &mut FileStorageWriter,
) -> ZResult<()> {
    let mut builder: Option<FlatBinary64Builder> = None;
    let mut scratch: Vec<u64> = Vec::new();
    let forward = col.forward;
    forward.scan_large_binary_column_rows(&col.name, |doc_id, bytes| {
        let Some(bytes) = bytes else {
            return Ok(());
        };
        let dim = bytes.len() / 8;
        if dim == 0 || bytes.len() % 8 != 0 {
            return Ok(());
        }
        let b = builder.get_or_insert_with(|| {
            scratch = vec![0u64; dim];
            FlatBinary64Builder::new(dim, flat_params.clone())
        });

        if cfg!(target_endian = "little")
            && (bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<u64>())
        {
            // SAFETY: alignment checked; length is multiple of 8; buffer lives with batch.
            let v = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u64, dim) };
            b.add(doc_id, v)?;
        } else {
            for (dst, chunk) in scratch[..dim].iter_mut().zip(bytes.as_chunks::<8>().0) {
                *dst = u64::from_le_bytes(*chunk);
            }
            b.add(doc_id, &scratch)?;
        }
        Ok(())
    })?;
    let b = match builder {
        Some(b) => b,
        None => FlatBinary64Builder::new(col.empty_dim()?, flat_params.clone()),
    };
    b.dump(writer)
}

fn build_flat_dense(
    col: &VectorColumn,
    flat_params: &FlatIndexParams,
    writer: &mut FileStorageWriter,
) -> ZResult<()> {
    let mut builder: Option<FlatBuilder> = None;
    let mut scratch: Vec<f32> = Vec::new();
    let forward = col.forward;
    forward.scan_large_binary_column_rows(&col.name, |doc_id, bytes| {
        let Some(bytes) = bytes else {
            return Ok(());
        };
        if bytes.len() % 4 != 0 {
            return Ok(());
        }
        let dim = bytes.len() / 4;
        if dim == 0 {
            return Ok(());
        }
        let b = builder.get_or_insert_with(|| {
            scratch = vec![0.0f32; dim];
            FlatBuilder::new(dim, flat_params.clone())
        });

        if cfg!(target_endian = "little")
            && (bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>())
        {
            // SAFETY: alignment checked; length is multiple of 4; buffer lives with batch.
            let v = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, dim) };
            b.add(doc_id, v)?;
        } else {
            for (dst, chunk) in scratch[..dim].iter_mut().zip(bytes.as_chunks::<4>().0) {
                *dst = f32::from_le_bytes(*chunk);
            }
            b.add(doc_id, &scratch)?;
        }
        Ok(())
    })?;
    let b = match builder {
        Some(b) => b,
        None => FlatBuilder::new(col.empty_dim()?, flat_params.clone()),
    };
    b.dump(writer)
}

fn build_flat_sparse(
    col: &VectorColumn,
    flat_params: FlatIndexParams,
    writer: &mut FileStorageWriter,
) -> ZResult<()> {
    let mut builder = FlatSparseBuilder::new_with_params(flat_params);
    let forward = col.forward;
    forward.scan_large_binary_column_rows(&col.name, |doc_id, bytes| {
        let Some(bytes) = bytes else {
            return Ok(());
        };
        let (indices, values) = decode_sparse_f32(bytes)?;
        if indices.is_empty() {
            // treat empty sparse vectors as NULL; skip for indexing.
            return Ok(());
        }
        builder.add(doc_id, SparseVector::new(indices, values))?;
        Ok(())
    })?;
    builder.dump(writer)
}

fn build_hnsw_sparse(
    col: &VectorColumn,
    hnsw_params: &HnswIndexParams,
    writer: &mut FileStorageWriter,
) -> ZResult<()> {
    let mut builder = HnswSparseBuilder::new(hnsw_params.clone());
    let forward = col.forward;
    forward.scan_large_binary_column_rows(&col.name, |doc_id, bytes| {
        let Some(bytes) = bytes else {
            return Ok(());
        };
        let (indices, values) = decode_sparse_f32(bytes)?;
        if indices.is_empty() {
            return Ok(());
        }
        builder.add(doc_id, SparseVector::new(indices, values))?;
        Ok(())
    })?;
    builder.dump(writer)
}

/// Dense vectors gathered up front for the parallel HNSW build.
struct DenseVectors {
    keys: Vec<u64>,
    vectors: Vec<f32>,
    dim: usize,
}

fn collect_dense_vectors(col: &VectorColumn) -> ZResult<DenseVectors> {
    let mut all_keys: Vec<u64> = Vec::new();
    let mut all_vectors: Vec<f32> = Vec::new();
    let mut detected_dim: Option<usize> = None;
    let forward = col.forward;
    forward.scan_large_binary_column_rows(&col.name, |doc_id, bytes| {
        let Some(bytes) = bytes else {
            return Ok(());
        };
        if bytes.len() % 4 != 0 {
            return Ok(());
        }
        let dim = bytes.len() / 4;
        if dim == 0 {
            return Ok(());
        }
        if detected_dim.is_none() {
            detected_dim = Some(dim);
        }

        all_keys.push(doc_id);
        if cfg!(target_endian = "little")
            && (bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>())
        {
            // SAFETY: alignment checked; length is multiple of 4; buffer lives with batch.
            let v = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, dim) };
            all_vectors.extend_from_slice(v);
        } else {
            for i in 0..dim {
                let off = i * 4;
                all_vectors.push(f32::from_le_bytes([
                    bytes[off],
                    bytes[off + 1],
                    bytes[off + 2],
                    bytes[off + 3],
                ]));
            }
        }
        Ok(())
    })?;
    let dim = match detected_dim {
        Some(dim) => dim,
        None => col.empty_dim()?,
    };
    Ok(DenseVectors {
        keys: all_keys,
        vectors: all_vectors,
        dim,
    })
}

fn build_hnsw_dense(
    col: &VectorColumn,
    hnsw_params: &HnswIndexParams,
    writer: &mut FileStorageWriter,
) -> ZResult<()> {
    let dense = collect_dense_vectors(col)?;
    let concurrency = hnsw_concurrency(hnsw_params);
    let b = HnswBuilder::build_parallel(
        &dense.keys,
        &dense.vectors,
        dense.dim,
        hnsw_params.clone(),
        concurrency,
    )?;
    b.dump(writer)
}

fn build_ivf_dense(
    col: &VectorColumn,
    ivf_params: &IvfIndexParams,
    writer: &mut FileStorageWriter,
) -> ZResult<()> {
    let mut builder: Option<IvfBuilder> = None;
    let mut scratch: Vec<f32> = Vec::new();
    let forward = col.forward;
    forward.scan_large_binary_column_rows(&col.name, |doc_id, bytes| {
        let Some(bytes) = bytes else {
            return Ok(());
        };
        if bytes.len() % 4 != 0 {
            return Ok(());
        }
        let dim = bytes.len() / 4;
        if dim == 0 {
            return Ok(());
        }
        let b = builder.get_or_insert_with(|| {
            scratch = vec![0.0f32; dim];
            IvfBuilder::new(dim, ivf_params.clone())
        });

        if cfg!(target_endian = "little")
            && (bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>())
        {
            // SAFETY: alignment checked; length is multiple of 4; buffer lives with batch.
            let v = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, dim) };
            let keys = [doc_id];
            let vecs: [&[f32]; 1] = [v];
            b.add_batch(&keys, &vecs)?;
        } else {
            for (dst, chunk) in scratch[..dim].iter_mut().zip(bytes.as_chunks::<4>().0) {
                *dst = f32::from_le_bytes(*chunk);
            }
            let keys = [doc_id];
            let vecs: [&[f32]; 1] = [&scratch];
            b.add_batch(&keys, &vecs)?;
        }
        Ok(())
    })?;
    let mut b = match builder {
        Some(b) => b,
        None => IvfBuilder::new(col.empty_dim()?, ivf_params.clone()),
    };
    b.train()?;
    b.dump(writer)
}

fn first_dim(docs: &[(u64, Doc)], dim_of: impl Fn(&Doc) -> Option<usize>) -> Option<usize> {
    docs.iter().find_map(|(_, doc)| dim_of(doc))
}

// Storage kind comes from the first doc carrying a u32, then u64, then f32 vector.
fn build_flat_from_docs(
    docs: &[(u64, Doc)],
    field: &str,
    flat_params: &FlatIndexParams,
    writer: &mut FileStorageWriter,
) -> ZResult<()> {
    if let Some(dim) = first_dim(docs, |doc| doc.get_vec_u32(field).map(|v| v.len())) {
        let mut builder = FlatBinary32Builder::new(dim, flat_params.clone());
        for (key, doc) in docs {
            if let Some(vec) = doc.get_vec_u32(field) {
                builder.add(*key, vec)?;
            }
        }
        return builder.dump(writer);
    }
    if let Some(dim) = first_dim(docs, |doc| doc.get_vec_u64(field).map(|v| v.len())) {
        let mut builder = FlatBinary64Builder::new(dim, flat_params.clone());
        for (key, doc) in docs {
            if let Some(vec) = doc.get_vec_u64(field) {
                builder.add(*key, vec)?;
            }
        }
        return builder.dump(writer);
    }
    let dim = first_dim(docs, |doc| doc.get_vec_f32(field).map(|v| v.len())).ok_or_else(|| {
        Status::invalid_argument(format!("no vectors found for field '{}'", field))
    })?;
    let mut builder = FlatBuilder::new(dim, flat_params.clone());
    for (key, doc) in docs {
        if let Some(vec) = doc.get_vec_f32(field) {
            builder.add(*key, vec)?;
        }
    }
    builder.dump(writer)
}

fn build_hnsw_from_docs(
    docs: &[(u64, Doc)],
    field: &str,
    hnsw_params: &HnswIndexParams,
    writer: &mut FileStorageWriter,
) -> ZResult<()> {
    let dim = first_dim(docs, |doc| doc.get_vec_f32(field).map(|v| v.len()))
        .ok_or_else(|| Status::invalid_argument("no vectors found"))?;

    let mut all_keys: Vec<u64> = Vec::with_capacity(docs.len());
    let mut all_vectors: Vec<f32> = Vec::with_capacity(docs.len() * dim);
    for (key, doc) in docs {
        if let Some(vec) = doc.get_vec_f32(field) {
            all_keys.push(*key);
            all_vectors.extend_from_slice(vec);
        }
    }
    let concurrency = hnsw_concurrency(hnsw_params);
    let builder = HnswBuilder::build_parallel(
        &all_keys,
        &all_vectors,
        dim,
        hnsw_params.clone(),
        concurrency,
    )?;
    builder.dump(writer)
}

fn build_ivf_from_docs(
    docs: &[(u64, Doc)],
    field: &str,
    ivf_params: &IvfIndexParams,
    writer: &mut FileStorageWriter,
) -> ZResult<()> {
    let dim = first_dim(docs, |doc| doc.get_vec_f32(field).map(|v| v.len()))
        .ok_or_else(|| Status::invalid_argument("no vectors found"))?;

    let mut builder = IvfBuilder::new(dim, ivf_params.clone());
    let keys: Vec<u64> = docs
        .iter()
        .filter_map(|(k, doc)| doc.get_vec_f32(field).map(|_| *k))
        .collect();
    let vecs: Vec<&[f32]> = docs
        .iter()
        .filter_map(|(_, doc)| doc.get_vec_f32(field))
        .collect();
    builder.add_batch(&keys, &vecs)?;
    builder.train()?;
    builder.dump(writer)
}

fn build_flat_sparse_from_docs(
    docs: &[(u64, Doc)],
    field: &str,
    flat_params: &FlatIndexParams,
    writer: &mut FileStorageWriter,
) -> ZResult<()> {
    let mut builder = FlatSparseBuilder::new_with_params(flat_params.clone());
    for (key, doc) in docs {
        // Empty sparse vectors are NULL in the forward store, so they are not indexed.
        if let Some((indices, values)) = doc.get_sparse_f32(field).filter(|(i, _)| !i.is_empty()) {
            builder.add(*key, SparseVector::new(indices.to_vec(), values.to_vec()))?;
        }
    }
    builder.dump(writer)
}

fn build_hnsw_sparse_from_docs(
    docs: &[(u64, Doc)],
    field: &str,
    hnsw_params: &HnswIndexParams,
    writer: &mut FileStorageWriter,
) -> ZResult<()> {
    let mut builder = HnswSparseBuilder::new(hnsw_params.clone());
    for (key, doc) in docs {
        // Empty sparse vectors are NULL in the forward store, so they are not indexed.
        if let Some((indices, values)) = doc.get_sparse_f32(field).filter(|(i, _)| !i.is_empty()) {
            builder.add(*key, SparseVector::new(indices.to_vec(), values.to_vec()))?;
        }
    }
    builder.dump(writer)
}

// Detect binary vs f32 flat index from META.
fn load_flat(reader: &FileStorageReader, p: &FlatIndexParams) -> ZResult<VectorIndex> {
    let meta_bytes = CoreStorageReader::read_segment(reader, "META")?;
    let meta = finch_core::algorithm::flat::IndexMeta::deserialize(meta_bytes.as_slice())?;
    match meta.storage_kind {
        0 => Ok(VectorIndex::Flat(FlatSearcher::load(reader, p)?)),
        1 => Ok(VectorIndex::FlatBinary32(FlatBinary32Searcher::load(
            reader, p,
        )?)),
        2 => Ok(VectorIndex::FlatBinary64(FlatBinary64Searcher::load(
            reader, p,
        )?)),
        _ => Err(Status::io_error("unknown flat index storage_kind")),
    }
}

pub struct IndexBuilder;

impl IndexBuilder {
    /// Build a vector index from all docs in a WritingSegment.
    /// For HNSW indexes on dense fields, uses the flat in-memory vectors directly
    /// to avoid cloning all docs.
    pub fn build_from_writing(
        seg: &WritingSegment,
        field: &str,
        params: &IndexParams,
        output_path: &Path,
    ) -> ZResult<()> {
        // Fast path: HNSW on dense vectors: read flat buffers directly.
        if let IndexParams::Hnsw(hnsw_params) = params {
            if let Some((keys, vectors, dim)) = seg.dense_vectors(field) {
                std::fs::create_dir_all(output_path)
                    .map_err(|e| Status::io_error(e.to_string()))?;
                let mut writer = FileStorageWriter::new(output_path)?;
                let concurrency = hnsw_concurrency(hnsw_params);
                let builder = HnswBuilder::build_parallel(
                    keys,
                    vectors,
                    dim,
                    hnsw_params.clone(),
                    concurrency,
                )?;
                builder.dump(&mut writer)?;
                return Ok(());
            }
        }
        // Fallback: clone all docs (non-HNSW types, or field not in dense index).
        let docs = seg.all_docs();
        Self::build_from_docs(&docs, field, params, output_path)
    }

    /// Build a vector index from docs in a PersistedSegment
    pub fn build_from_persisted(
        seg: &PersistedSegment,
        field: &FieldSchema,
        params: &IndexParams,
        output_path: &Path,
    ) -> ZResult<()> {
        std::fs::create_dir_all(output_path).map_err(|e| Status::io_error(e.to_string()))?;
        let mut writer = FileStorageWriter::new(output_path)?;

        let forward = seg.forward_store.read();

        if seg.doc_count == 0 {
            return Err(Status::invalid_argument(
                "persisted segment has no docs (cannot build index)",
            ));
        }

        let col = VectorColumn::detect(&forward, field)?;
        let writer = &mut writer;
        match params {
            IndexParams::Flat(flat_params) => build_flat_persisted(&col, flat_params, writer),
            IndexParams::Hnsw(hnsw_params) => {
                col.require(VectorKind::Dense, "HNSW index requires dense vectors")?;
                build_hnsw_dense(&col, hnsw_params, writer)
            }
            IndexParams::Ivf(ivf_params) => {
                col.require(VectorKind::Dense, "IVF index requires dense vectors")?;
                build_ivf_dense(&col, ivf_params, writer)
            }
            IndexParams::FlatSparse(flat_params) => {
                col.require(
                    VectorKind::Sparse,
                    "flat sparse index requires sparse vectors",
                )?;
                build_flat_sparse(&col, flat_params.clone(), writer)
            }
            IndexParams::HnswSparse(hnsw_params) => {
                col.require(
                    VectorKind::Sparse,
                    "HNSW sparse index requires sparse vectors",
                )?;
                build_hnsw_sparse(&col, hnsw_params, writer)
            }
            IndexParams::Invert(_) => Err(Status::invalid_argument(
                "cannot build invert index from persisted vectors",
            )),
        }
    }

    fn build_from_docs(
        docs: &[(u64, Doc)],
        field: &str,
        params: &IndexParams,
        output_path: &Path,
    ) -> ZResult<()> {
        std::fs::create_dir_all(output_path).map_err(|e| Status::io_error(e.to_string()))?;
        let mut writer = FileStorageWriter::new(output_path)?;
        let writer = &mut writer;

        match params {
            IndexParams::Flat(flat_params) => {
                build_flat_from_docs(docs, field, flat_params, writer)
            }
            IndexParams::Hnsw(hnsw_params) => {
                build_hnsw_from_docs(docs, field, hnsw_params, writer)
            }
            IndexParams::Ivf(ivf_params) => build_ivf_from_docs(docs, field, ivf_params, writer),
            IndexParams::FlatSparse(flat_params) => {
                build_flat_sparse_from_docs(docs, field, flat_params, writer)
            }
            IndexParams::HnswSparse(hnsw_params) => {
                build_hnsw_sparse_from_docs(docs, field, hnsw_params, writer)
            }
            IndexParams::Invert(_) => Err(Status::invalid_argument(
                "Invert index is built separately, not via IndexBuilder",
            )),
        }
    }

    /// Load a built vector index from disk
    pub fn load_index(
        _field: &str,
        params: &IndexParams,
        index_path: &Path,
        enable_mmap: bool,
    ) -> ZResult<VectorIndex> {
        let reader = FileStorageReader::new(index_path, enable_mmap);
        match params {
            IndexParams::Flat(p) => load_flat(&reader, p),
            IndexParams::Hnsw(p) => Ok(VectorIndex::Hnsw(HnswSearcher::load(&reader, p)?)),
            IndexParams::Ivf(p) => Ok(VectorIndex::Ivf(IvfSearcher::load(&reader, p)?)),
            IndexParams::FlatSparse(p) => Ok(VectorIndex::FlatSparse(
                FlatSparseSearcher::load_with_params(&reader, p)?,
            )),
            IndexParams::HnswSparse(p) => {
                use finch_core::algorithm::hnsw_sparse::HnswSparseSearcher;
                Ok(VectorIndex::HnswSparse(
                    HnswSparseSearcher::load_with_params(&reader, p)?,
                ))
            }
            IndexParams::Invert(_) => {
                Err(Status::invalid_argument("Invert index loaded separately"))
            }
        }
    }
}

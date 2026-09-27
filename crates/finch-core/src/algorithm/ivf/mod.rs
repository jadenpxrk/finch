//! IVF (Inverted File) dense vector index

use super::cluster::KmeansCluster;
use super::codec::{read_u32, read_u64, read_u8, u64s_to_le};
use super::flat::{
    load_vector_rows, segment_array_f32, segment_array_u32, segment_array_u64, FlatVectorStorage,
    RowLayout, SegmentArray, StorageReader, StorageWriter,
};
use super::{check_query_dim, check_vector_dim, doc_filter_allows, DocFilter, TopkHeap};
use crate::metric::{make_metric, Metric};
use crate::quantizer::{
    bytes_per_vector, distance_to_quantized_with_query_sq_norm, quantize_append,
    quantize_type_from_u32, query_sq_norm,
};
use finch_types::{IndexParams, IvfIndexParams, QuantizeType, Status, ZResult};
use std::io::Cursor;

const SEG_CENTROIDS: &str = "IVF_CENTROIDS";
const SEG_LISTS: &str = "IVF_LISTS";
const SEG_VECTORS: &str = "IVF_VECTORS";
const SEG_KEYS: &str = "IVF_KEYS";
const SEG_META: &str = "IVF_META";
const SEG_L1_CENTROIDS: &str = "IVF_L1_CENTROIDS";
const SEG_L1_OFFSETS: &str = "IVF_L1_OFFSETS";

pub struct IvfBuilder {
    params: IvfIndexParams,
    dim: usize,
    keys: Vec<u64>,
    /// Flat row-major vector storage: keys.len() × dim contiguous f32s
    vectors: Vec<f32>,
    cluster: Option<KmeansCluster>,
}

/// Composite L1 layout: per outer cell, `n_list` inner centroids and `n_list + 1` relative offsets.
struct L1Layout {
    n_list: usize,
    centroids: Vec<u8>,
    rel_offsets: Vec<Vec<u64>>,
}

/// Encoded keys and vectors in storage order, plus absolute L1 offsets for composite IVF.
struct PackedRows {
    keys: Vec<u8>,
    vectors: Vec<u8>,
    l1_offsets: Vec<u64>,
}

fn push_zero_f32s(buf: &mut Vec<u8>, count: usize) {
    for _ in 0..count {
        buf.extend_from_slice(&0.0f32.to_le_bytes());
    }
}

impl IvfBuilder {
    pub fn new(dim: usize, params: IvfIndexParams) -> Self {
        IvfBuilder {
            params,
            dim,
            keys: Vec::new(),
            vectors: Vec::new(),
            cluster: None,
        }
    }

    fn get_vector(&self, i: usize) -> &[f32] {
        &self.vectors[i * self.dim..(i + 1) * self.dim]
    }

    pub fn add_batch(&mut self, keys: &[u64], vectors: &[&[f32]]) -> ZResult<()> {
        if keys.len() != vectors.len() {
            return Err(Status::invalid_argument(
                "keys and vectors must have same length",
            ));
        }
        for (&key, &vec) in keys.iter().zip(vectors.iter()) {
            check_vector_dim(vec.len(), self.dim)?;
            self.keys.push(key);
            self.vectors.extend_from_slice(vec);
        }
        Ok(())
    }

    pub fn train(&mut self) -> ZResult<()> {
        let n = self.keys.len();
        if n == 0 {
            // Zero cells: the index is valid and every search probes nothing.
            self.cluster = Some(KmeansCluster {
                centroids: Vec::new(),
                dim: self.dim,
                n_list: 0,
            });
            return Ok(());
        }
        let cluster = KmeansCluster::train(&self.vectors, n, self.dim, &self.params)?;
        self.cluster = Some(cluster);
        Ok(())
    }

    pub fn dump(&self, storage: &mut dyn StorageWriter) -> ZResult<()> {
        let cluster = self
            .cluster
            .as_ref()
            .ok_or_else(|| Status::internal("IVF index not trained; call train() first"))?;
        let metric = make_metric(self.params.metric);

        let mut lists = self.assign_lists(cluster, metric.as_ref());
        // Composite L1 IVF (SOAR layout only) reorders each outer list by an inner clustering.
        let l1 = match self.l1_params()? {
            Some(inner) => Some(self.build_l1_layout(&mut lists, &inner, metric.as_ref())?),
            None => None,
        };

        // Write centroids: n_list × dim f32, little-endian
        let centroid_bytes: Vec<u8> = cluster
            .centroids
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        storage.write_segment(SEG_CENTROIDS, &centroid_bytes)?;

        let list_offsets = self.write_lists(storage, &lists)?;
        let packed = self.pack_rows(&lists, l1.as_ref());
        storage.write_segment(SEG_KEYS, &packed.keys)?;
        storage.write_segment(SEG_VECTORS, &packed.vectors)?;
        let meta = self.encode_meta(cluster.n_list, &list_offsets, l1.as_ref());
        storage.write_segment(SEG_META, &meta)?;

        if let Some(l1) = &l1 {
            // Persist per-cell inner centroids and absolute offsets (packed positions).
            storage.write_segment(SEG_L1_CENTROIDS, &l1.centroids)?;
            storage.write_segment(SEG_L1_OFFSETS, &u64s_to_le(&packed.l1_offsets))?;
        }

        Ok(())
    }

    /// Assign vectors to lists
    fn assign_lists(&self, cluster: &KmeansCluster, metric: &dyn Metric) -> Vec<Vec<u32>> {
        let mut lists: Vec<Vec<u32>> = vec![Vec::new(); cluster.n_list];
        for i in 0..self.keys.len() {
            let c = cluster.find_nearest(self.get_vector(i), metric);
            lists[c].push(i as u32);
        }
        lists
    }

    /// Inner IVF params of a composite index, validated against the outer params.
    fn l1_params(&self) -> ZResult<Option<IvfIndexParams>> {
        let l1_params = match self.params.l1_index.as_deref() {
            None => None,
            Some(IndexParams::Flat(_)) => None, // flat is the implicit base scan
            Some(IndexParams::Ivf(p)) => Some(p.clone()),
            Some(_) => {
                return Err(Status::invalid_argument(
                    "composite IVF only supports nested IVF (l1_index=IVF)",
                ));
            }
        };
        if let Some(inner) = l1_params.as_ref() {
            if !self.params.use_soar {
                return Err(Status::invalid_argument(
                    "composite IVF requires use_soar=true (packed layout)",
                ));
            }
            if inner.metric != self.params.metric {
                return Err(Status::invalid_argument(
                    "composite IVF requires inner.metric == outer.metric",
                ));
            }
        }
        Ok(l1_params)
    }

    fn build_l1_layout(
        &self,
        lists: &mut [Vec<u32>],
        inner: &IvfIndexParams,
        metric: &dyn Metric,
    ) -> ZResult<L1Layout> {
        let l1_n_list = inner.n_list.max(1);
        let mut layout = L1Layout {
            n_list: l1_n_list,
            centroids: Vec::new(),
            rel_offsets: Vec::with_capacity(lists.len()),
        };
        // Reserve: outer_n_list * inner_n_list * dim f32
        layout
            .centroids
            .reserve(lists.len() * l1_n_list * self.dim * 4);

        for list in lists.iter_mut() {
            if list.is_empty() {
                // Pad centroids with zeros and offsets with 0s.
                push_zero_f32s(&mut layout.centroids, l1_n_list * self.dim);
                layout.rel_offsets.push(vec![0u64; l1_n_list + 1]);
                continue;
            }
            let rel_off = self.reorder_cell(list, inner, metric, &mut layout.centroids)?;
            layout.rel_offsets.push(rel_off);
        }
        Ok(layout)
    }

    /// Regroups one outer cell by inner cluster, appending padded inner centroids; returns offsets.
    fn reorder_cell(
        &self,
        list: &mut Vec<u32>,
        inner: &IvfIndexParams,
        metric: &dyn Metric,
        centroids_out: &mut Vec<u8>,
    ) -> ZResult<Vec<u64>> {
        let dim = self.dim;
        let l1_n_list = inner.n_list.max(1);

        // Collect vectors for this outer cell into a flat buffer (no per-vector alloc)
        let mut cell_vecs: Vec<f32> = Vec::with_capacity(list.len() * dim);
        for &idx in list.iter() {
            cell_vecs.extend_from_slice(self.get_vector(idx as usize));
        }

        let mut cfg = inner.clone();
        cfg.l1_index = None;
        cfg.n_list = l1_n_list;
        let n_inner = list.len();
        let inner_cluster = KmeansCluster::train(&cell_vecs, n_inner, dim, &cfg)?;

        // Persist centroids (pad to l1_n_list)
        for j in 0..l1_n_list {
            if j < inner_cluster.n_list {
                let c = &inner_cluster.centroid(j)[..dim];
                centroids_out.extend(c.iter().flat_map(|x| x.to_le_bytes()));
            } else {
                push_zero_f32s(centroids_out, dim);
            }
        }

        // Assign each vector in this outer cell to an inner cell
        let mut inner_lists: Vec<Vec<u32>> = vec![Vec::new(); l1_n_list];
        for pos in 0..n_inner {
            let v = &cell_vecs[pos * dim..(pos + 1) * dim];
            let inner_id = inner_cluster.find_nearest(v, metric);
            inner_lists[inner_id].push(list[pos]);
        }

        let mut rel_off: Vec<u64> = Vec::with_capacity(l1_n_list + 1);
        rel_off.push(0);
        let mut running: u64 = 0;
        let mut new_list: Vec<u32> = Vec::with_capacity(list.len());
        for bucket in inner_lists {
            running += bucket.len() as u64;
            new_list.extend_from_slice(&bucket);
            rel_off.push(running);
        }
        *list = new_list;
        Ok(rel_off)
    }

    /// List byte offsets; `IVF_LISTS` is written only without SOAR, which packs rows in list order.
    fn write_lists(
        &self,
        storage: &mut dyn StorageWriter,
        lists: &[Vec<u32>],
    ) -> ZResult<Vec<u64>> {
        let mut list_offsets: Vec<u64> = Vec::with_capacity(lists.len() + 1);
        list_offsets.push(0);
        if !self.params.use_soar {
            let mut list_buf: Vec<u8> = Vec::new();
            for list in lists {
                for &idx in list {
                    list_buf.extend_from_slice(&idx.to_le_bytes());
                }
                list_offsets.push(list_buf.len() as u64);
            }
            storage.write_segment(SEG_LISTS, &list_buf)?;
            return Ok(list_offsets);
        }
        let mut bytes = 0u64;
        for list in lists {
            bytes = bytes
                .checked_add((list.len() as u64) * 4)
                .ok_or_else(|| Status::io_error("IVF list offsets overflow"))?;
            list_offsets.push(bytes);
        }
        Ok(list_offsets)
    }

    /// Encodes keys and vectors in list order (SOAR) or insertion order.
    fn pack_rows(&self, lists: &[Vec<u32>], l1: Option<&L1Layout>) -> PackedRows {
        let n = self.keys.len();
        let vec_bytes = bytes_per_vector(self.params.quantize, self.dim);
        let mut packed = PackedRows {
            keys: Vec::with_capacity(n * 8),
            vectors: Vec::with_capacity(n * vec_bytes),
            l1_offsets: Vec::new(),
        };
        if !self.params.use_soar {
            for (i, &k) in self.keys.iter().enumerate() {
                packed.keys.extend_from_slice(&k.to_le_bytes());
                quantize_append(
                    self.params.quantize,
                    self.get_vector(i),
                    &mut packed.vectors,
                );
            }
            return packed;
        }

        let mut global_pos: u64 = 0;
        for (cell, list) in lists.iter().enumerate() {
            if let Some(l1) = l1 {
                // Convert per-cell rel offsets to absolute positions.
                let outer_start = global_pos;
                for &r in &l1.rel_offsets[cell] {
                    packed.l1_offsets.push(outer_start + r);
                }
            }
            for &idx in list {
                let i = idx as usize;
                packed.keys.extend_from_slice(&self.keys[i].to_le_bytes());
                quantize_append(
                    self.params.quantize,
                    self.get_vector(i),
                    &mut packed.vectors,
                );
                global_pos += 1;
            }
        }
        packed
    }

    fn encode_meta(&self, n_list: usize, list_offsets: &[u64], l1: Option<&L1Layout>) -> Vec<u8> {
        let vec_bytes = bytes_per_vector(self.params.quantize, self.dim);
        let mut meta = Vec::new();
        meta.extend_from_slice(&(self.keys.len() as u64).to_le_bytes());
        meta.extend_from_slice(&(self.dim as u64).to_le_bytes());
        meta.extend_from_slice(&(n_list as u64).to_le_bytes());
        meta.extend_from_slice(&(self.params.metric as u32).to_le_bytes());
        meta.extend_from_slice(&u64s_to_le(list_offsets));
        meta.push(self.params.use_soar as u8);
        meta.extend_from_slice(&(self.params.quantize as u32).to_le_bytes());
        meta.extend_from_slice(&(vec_bytes as u32).to_le_bytes());
        meta.push(l1.is_some() as u8);
        if let Some(l1) = l1 {
            meta.extend_from_slice(&(l1.n_list as u64).to_le_bytes());
        }
        meta
    }
}

struct IvfList {
    start: usize,
    end: usize,
}

pub struct IvfSearcher {
    centroids: SegmentArray<f32>, // flat: n_list × dim
    n_list: usize,
    dim: usize,
    keys: SegmentArray<u64>,
    vectors: FlatVectorStorage,
    list_data: Option<SegmentArray<u32>>,
    lists: Vec<IvfList>,
    metric: Box<dyn Metric>,
    count: usize,
    use_soar: bool,
    l1_centroids: Option<SegmentArray<f32>>,
    l1_offsets: Option<SegmentArray<u64>>,
    l1_n_list: usize,
}

/// `IVF_META`; the footer after the list offsets is absent in older indexes.
struct IvfMeta {
    n: usize,
    dim: usize,
    n_list: usize,
    offsets: Vec<u64>,
    use_soar: bool,
    quantize: QuantizeType,
    vec_bytes: usize,
    has_l1: bool,
    l1_n_list: usize,
}

impl IvfMeta {
    fn parse(bytes: &[u8]) -> ZResult<Self> {
        let mut cur = Cursor::new(bytes);
        let has_remaining = |cur: &Cursor<&[u8]>, len: usize| {
            bytes.len().saturating_sub(cur.position() as usize) >= len
        };
        let n = read_u64(&mut cur)? as usize;
        let dim = read_u64(&mut cur)? as usize;
        let n_list = read_u64(&mut cur)? as usize;
        let _metric_raw = read_u32(&mut cur)?;

        // Read list offsets (n_list + 1 entries)
        let mut offsets: Vec<u64> = Vec::with_capacity(n_list + 1);
        for _ in 0..=(n_list) {
            offsets.push(read_u64(&mut cur)?);
        }

        // Optional footer (backward compatible)
        let use_soar = has_remaining(&cur, 1) && read_u8(&mut cur)? != 0;
        let quantize = if has_remaining(&cur, 4) {
            quantize_type_from_u32(read_u32(&mut cur)?)
        } else {
            QuantizeType::Undefined
        };
        let vec_bytes = if has_remaining(&cur, 4) {
            read_u32(&mut cur)? as usize
        } else {
            bytes_per_vector(QuantizeType::Undefined, dim)
        };
        let has_l1 = has_remaining(&cur, 1) && read_u8(&mut cur)? != 0;
        let l1_n_list = if has_l1 {
            if !has_remaining(&cur, 8) {
                return Err(Status::io_error("IVF meta missing l1_n_list"));
            }
            read_u64(&mut cur)? as usize
        } else {
            0
        };
        Ok(IvfMeta {
            n,
            dim,
            n_list,
            offsets,
            use_soar,
            quantize,
            vec_bytes,
            has_l1,
            l1_n_list,
        })
    }

    fn check_params(&self, params: &IvfIndexParams) -> ZResult<()> {
        if self.use_soar != params.use_soar {
            return Err(Status::invalid_argument(
                "IVF use_soar mismatch between index and params",
            ));
        }
        if self.quantize != params.quantize {
            return Err(Status::invalid_argument(
                "IVF quantize mismatch between index and params",
            ));
        }

        let expected_l1 = matches!(params.l1_index.as_deref(), Some(IndexParams::Ivf(_)));
        if self.has_l1 != expected_l1 {
            return Err(Status::invalid_argument(
                "IVF l1_index mismatch between index and params",
            ));
        }
        if !self.has_l1 {
            return Ok(());
        }
        let Some(IndexParams::Ivf(inner)) = params.l1_index.as_deref() else {
            return Err(Status::invalid_argument(
                "IVF l1_index must be IVF when present",
            ));
        };
        if inner.n_list.max(1) != self.l1_n_list {
            return Err(Status::invalid_argument(
                "IVF l1_n_list mismatch between index and params",
            ));
        }
        Ok(())
    }
}

/// Per-query inputs shared by the list scans.
struct Probe<'a> {
    query: &'a [f32],
    query_sq_norm: f32,
    filter: Option<&'a dyn DocFilter>,
}

/// Inner centroids and absolute offsets of a composite index.
struct L1Index<'a> {
    centroids: &'a [f32],
    offsets: &'a [u64],
}

impl IvfSearcher {
    pub fn load(storage: &dyn StorageReader, params: &IvfIndexParams) -> ZResult<Self> {
        let meta = IvfMeta::parse(storage.read_segment(SEG_META)?.as_slice())?;
        meta.check_params(params)?;
        let (n, dim, n_list) = (meta.n, meta.dim, meta.n_list);

        let centroids = segment_array_f32(storage.read_segment(SEG_CENTROIDS)?)?;
        if centroids.len() != n_list * dim {
            return Err(Status::io_error("IVF centroids length mismatch"));
        }

        let mut lists = Vec::with_capacity(n_list);
        for i in 0..n_list {
            let start = (meta.offsets[i] / 4) as usize;
            let end = (meta.offsets[i + 1] / 4) as usize;
            lists.push(IvfList { start, end });
        }

        let list_data = if meta.use_soar {
            None
        } else {
            Some(segment_array_u32(storage.read_segment(SEG_LISTS)?)?)
        };

        let keys = segment_array_u64(storage.read_segment(SEG_KEYS)?)?;
        if keys.len() != n {
            return Err(Status::io_error("IVF keys length mismatch"));
        }

        let layout = RowLayout {
            count: n,
            dim,
            quantize: meta.quantize,
            vec_bytes: meta.vec_bytes,
        };
        let vectors = load_vector_rows(storage.read_segment(SEG_VECTORS)?, &layout, "IVF")?;
        let l1_centroids = if meta.has_l1 {
            Some(segment_array_f32(storage.read_segment(SEG_L1_CENTROIDS)?)?)
        } else {
            None
        };
        let l1_offsets = if meta.has_l1 {
            Some(segment_array_u64(storage.read_segment(SEG_L1_OFFSETS)?)?)
        } else {
            None
        };

        Ok(IvfSearcher {
            centroids,
            n_list,
            dim,
            keys,
            vectors,
            list_data,
            lists,
            metric: make_metric(params.metric),
            count: n,
            use_soar: meta.use_soar,
            l1_centroids,
            l1_offsets,
            l1_n_list: meta.l1_n_list,
        })
    }

    fn get_centroid(&self, i: usize) -> &[f32] {
        &self.centroids.as_slice()[i * self.dim..(i + 1) * self.dim]
    }

    fn distance_to_vector(&self, query: &[f32], query_sq_norm: f32, i: usize) -> f32 {
        match &self.vectors {
            FlatVectorStorage::F32(vectors) => {
                let start = i * self.dim;
                self.metric
                    .distance(query, &vectors.as_slice()[start..start + self.dim])
            }
            FlatVectorStorage::Quantized {
                bytes,
                quantize,
                stride,
            } => {
                let mt = self.metric.metric_type();
                let off = i * (*stride);
                distance_to_quantized_with_query_sq_norm(
                    mt,
                    query,
                    query_sq_norm,
                    &bytes.as_slice()[off..off + *stride],
                    self.dim,
                    *quantize,
                )
            }
        }
    }

    fn topk_by_distance(mut dists: Vec<(usize, f32)>, k: usize) -> Vec<usize> {
        let k = k.min(dists.len());
        if k == 0 {
            return Vec::new();
        }
        if k == dists.len() {
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            return dists.into_iter().map(|(i, _)| i).collect();
        }

        // Partition so the smallest k elements are in the first k positions (unordered),
        // then sort that prefix for stable probe ordering.
        let nth = k - 1;
        dists.select_nth_unstable_by(nth, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        dists.truncate(k);
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.into_iter().map(|(i, _)| i).collect()
    }

    /// Find n_probe nearest centroids
    fn nearest_centroids(&self, query: &[f32], n_probe: usize) -> Vec<usize> {
        let dists: Vec<(usize, f32)> = (0..self.n_list)
            .map(|i| (i, self.metric.distance(query, self.get_centroid(i))))
            .collect();
        Self::topk_by_distance(dists, n_probe)
    }

    pub fn search(
        &self,
        query: &[f32],
        topk: usize,
        n_probe: usize,
        filter: Option<&dyn DocFilter>,
    ) -> ZResult<Vec<(u64, f32)>> {
        check_query_dim(query.len(), self.dim)?;

        let probe = Probe {
            query,
            query_sq_norm: query_sq_norm(self.metric.metric_type(), query, self.dim),
            filter,
        };
        let probe_cells = self.nearest_centroids(query, n_probe);
        let mut heap = TopkHeap::new(topk);

        for cell_idx in probe_cells {
            let list = &self.lists[cell_idx];
            if let (Some(l1_centroids), Some(l1_offsets)) = (&self.l1_centroids, &self.l1_offsets) {
                let l1 = L1Index {
                    centroids: l1_centroids.as_slice(),
                    offsets: l1_offsets.as_slice(),
                };
                let inner_probe = (n_probe / 3).max(1).min(self.l1_n_list);
                self.scan_composite_cell(&probe, &l1, cell_idx, inner_probe, &mut heap)?;
            } else if self.use_soar {
                self.scan_packed(&probe, list.start, list.end, &mut heap)?;
            } else {
                self.scan_listed(&probe, list, &mut heap)?;
            }
        }

        Ok(heap.into_sorted())
    }

    /// Scans the `inner_probe` nearest inner cells of outer cell `cell_idx`.
    fn scan_composite_cell(
        &self,
        probe: &Probe<'_>,
        l1: &L1Index<'_>,
        cell_idx: usize,
        inner_probe: usize,
        heap: &mut TopkHeap,
    ) -> ZResult<()> {
        let list = &self.lists[cell_idx];
        let base = cell_idx * self.l1_n_list * self.dim;
        let off_base = cell_idx * (self.l1_n_list + 1);
        let (l1_c, l1_o) = (l1.centroids, l1.offsets);
        let dists: Vec<(usize, f32)> = (0..self.l1_n_list)
            .filter_map(|j| {
                let start = l1_o[off_base + j] as usize;
                let end = l1_o[off_base + j + 1] as usize;
                if start >= end {
                    return None;
                }
                let c = &l1_c[base + j * self.dim..base + (j + 1) * self.dim];
                Some((j, self.metric.distance(probe.query, c)))
            })
            .collect();
        let inner_cells = Self::topk_by_distance(dists, inner_probe);

        for j in inner_cells {
            let mut start = l1_o[off_base + j] as usize;
            let mut end = l1_o[off_base + j + 1] as usize;
            start = start.max(list.start).min(list.end);
            end = end.max(list.start).min(list.end);
            if start >= end {
                continue;
            }
            self.scan_packed(probe, start, end, heap)?;
        }
        Ok(())
    }

    /// Scans packed rows `start..end`.
    fn scan_packed(
        &self,
        probe: &Probe<'_>,
        start: usize,
        end: usize,
        heap: &mut TopkHeap,
    ) -> ZResult<()> {
        let keys = self.keys.as_slice();
        for (node_idx, &key) in keys.iter().enumerate().take(end).skip(start) {
            let allowed = doc_filter_allows(probe.filter, key)?;
            if !allowed {
                continue;
            }
            let d = self.distance_to_vector(probe.query, probe.query_sq_norm, node_idx);
            heap.push(d, key);
        }
        Ok(())
    }

    /// Scans the rows a non-packed list names by index.
    fn scan_listed(&self, probe: &Probe<'_>, list: &IvfList, heap: &mut TopkHeap) -> ZResult<()> {
        let Some(list_data) = &self.list_data else {
            return Err(Status::internal("IVF missing list_data"));
        };
        let keys = self.keys.as_slice();
        let ld = list_data.as_slice();
        for &node_idx in &ld[list.start..list.end] {
            let key = keys[node_idx as usize];
            let allowed = doc_filter_allows(probe.filter, key)?;
            if !allowed {
                continue;
            }
            let d = self.distance_to_vector(probe.query, probe.query_sq_norm, node_idx as usize);
            heap.push(d, key);
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

#[cfg(test)]
mod tests {
    use super::super::flat::MemoryStorage;
    use super::*;
    use finch_types::{MetricType, QuantizeType};

    #[test]
    fn test_topk_by_distance_matches_full_sort() {
        let dists = vec![
            (0usize, 3.0f32),
            (1usize, 1.0f32),
            (2usize, 2.0f32),
            (3usize, 0.5f32),
            (4usize, 4.0f32),
        ];
        let k = 3;
        let got = IvfSearcher::topk_by_distance(dists.clone(), k);
        let mut full = dists.clone();
        full.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let want: Vec<usize> = full.into_iter().take(k).map(|(i, _)| i).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn test_ivf_build_search() {
        let params = IvfIndexParams {
            n_list: 2,
            n_iters: 5,
            use_soar: false,
            l1_index: None,
            metric: MetricType::L2,
            quantize: finch_types::QuantizeType::Undefined,
        };
        let dim = 4;
        let mut builder = IvfBuilder::new(dim, params.clone());

        let keys: Vec<u64> = (1..=10).collect();
        let vecs: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32, 0.0, 0.0, 0.0]).collect();
        let vec_refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        builder.add_batch(&keys, &vec_refs).unwrap();
        builder.train().unwrap();

        let mut storage = MemoryStorage::new();
        builder.dump(&mut storage).unwrap();

        let searcher = IvfSearcher::load(&storage, &params).unwrap();
        let results = searcher.search(&[5.0, 0.0, 0.0, 0.0], 3, 2, None).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_ivf_build_search_soar_quantized_int8() {
        let params = IvfIndexParams {
            n_list: 2,
            n_iters: 5,
            use_soar: true,
            l1_index: None,
            metric: MetricType::L2,
            quantize: QuantizeType::Int8,
        };
        let dim = 4;
        let mut builder = IvfBuilder::new(dim, params.clone());

        let keys: Vec<u64> = (1..=10).collect();
        let vecs: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32, 0.0, 0.0, 0.0]).collect();
        let vec_refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        builder.add_batch(&keys, &vec_refs).unwrap();
        builder.train().unwrap();

        let mut storage = MemoryStorage::new();
        builder.dump(&mut storage).unwrap();

        let searcher = IvfSearcher::load(&storage, &params).unwrap();
        let results = searcher.search(&[5.0, 0.0, 0.0, 0.0], 3, 2, None).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_ivf_composite_l1_build_search() {
        let inner = IvfIndexParams {
            n_list: 2,
            n_iters: 3,
            use_soar: true,
            l1_index: None,
            metric: MetricType::L2,
            quantize: QuantizeType::Undefined,
        };
        let params = IvfIndexParams {
            n_list: 2,
            n_iters: 3,
            use_soar: true,
            l1_index: Some(Box::new(IndexParams::Ivf(inner))),
            metric: MetricType::L2,
            quantize: QuantizeType::Undefined,
        };
        let dim = 4;
        let mut builder = IvfBuilder::new(dim, params.clone());

        let keys: Vec<u64> = (1..=50).collect();
        let vecs: Vec<Vec<f32>> = (0..50)
            .map(|i| vec![(i % 10) as f32, 0.0, 0.0, 0.0])
            .collect();
        let vec_refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        builder.add_batch(&keys, &vec_refs).unwrap();
        builder.train().unwrap();

        let mut storage = MemoryStorage::new();
        builder.dump(&mut storage).unwrap();

        let searcher = IvfSearcher::load(&storage, &params).unwrap();
        let results = searcher.search(&[5.0, 0.0, 0.0, 0.0], 5, 2, None).unwrap();
        assert!(!results.is_empty());
    }
}

use super::*;
use crate::algorithm::check_query_dim;
use crate::algorithm::codec::{read_u32, read_u64, u32_slice_at};
use crate::quantizer::query_sq_norm;

#[inline]
pub(super) fn align32(bytes: usize) -> usize {
    (bytes + 31) & !31usize
}

#[inline]
pub(super) fn aligned_vector_stride_bytes(raw_bytes: usize) -> usize {
    let mut stride = align32(raw_bytes);
    if stride.is_multiple_of(1024) {
        stride = align32(stride + 1);
    }
    stride
}

#[inline]
pub(super) fn aligned_vector_stride_floats(dim: usize) -> usize {
    let raw_bytes = dim * 4;
    let stride_bytes = aligned_vector_stride_bytes(raw_bytes);
    stride_bytes / 4
}

/// Level-1 routing points that seed the level-0 beam unless a query overrides it.
const DEFAULT_L0_SEEDS: usize = 24;

/// Search-time knobs of one HNSW query.
#[derive(Clone, Copy, Debug)]
pub struct HnswSearchParams {
    /// Level-0 beam width; raised to `topk`.
    pub ef: usize,
    /// Upper-level beam width; `None` descends greedily.
    pub upper_ef: Option<usize>,
    /// Level-1 routing points that seed the level-0 beam; `1` seeds from the single entry.
    pub l0_seeds: usize,
}

impl HnswSearchParams {
    pub fn new(ef: usize) -> Self {
        HnswSearchParams {
            ef,
            upper_ef: None,
            l0_seeds: DEFAULT_L0_SEEDS,
        }
    }
}

struct UpperLevelIndex {
    node_offsets: Vec<u32>,
    fixed_len: u32,
}

pub struct HnswSearcher {
    keys: SegmentArray<u64>,
    vectors: HnswVectorStorage,
    l0_neighbors: SegmentArray<u32>, // flat: count × (2*m)
    upper_bytes: SegmentBytes,
    upper_index: Vec<UpperLevelIndex>,
    entry_point: u32,
    max_level: usize,
    dim: usize,
    count: usize,
    m: usize,
    metric_type: MetricType,
    cosine_normalized: bool,
    vec_stride: usize,
    pub default_ef: usize,
    pub bruteforce_threshold: usize,
}

enum HnswVectorStorage {
    F32(SegmentArray<f32>),
    Quantized {
        bytes: SegmentBytes,
        quantize: QuantizeType,
        stride: usize,
    },
}

/// `HNSW_META`.
struct HnswMeta {
    count: usize,
    dim: usize,
    entry_point: u32,
    max_level: usize,
    m: usize,
    ef_construction: usize,
    quantize: QuantizeType,
    vec_stride: usize,
    cosine_normalized: bool,
}

impl HnswMeta {
    fn parse(bytes: &[u8]) -> ZResult<Self> {
        let mut cur = Cursor::new(bytes);
        let count = read_u64(&mut cur)? as usize;
        let dim = read_u64(&mut cur)? as usize;
        let entry_point = read_u32(&mut cur)?;
        let max_level = read_u32(&mut cur)? as usize;
        let m = read_u32(&mut cur)? as usize;
        let ef_construction = read_u32(&mut cur)? as usize;
        let quantize = quantize_type_from_u32(read_u32(&mut cur)?);
        let vec_stride = read_u32(&mut cur)? as usize;
        let cosine_normalized = read_u32(&mut cur)? != 0;
        Ok(HnswMeta {
            count,
            dim,
            entry_point,
            max_level,
            m,
            ef_construction,
            quantize,
            vec_stride,
            cosine_normalized,
        })
    }

    fn check_params(&self, params: &HnswIndexParams) -> ZResult<()> {
        if self.quantize != params.quantize {
            return Err(Status::invalid_argument(
                "HNSW quantize mismatch between index and params",
            ));
        }
        if self.m != params.m {
            return Err(Status::invalid_argument(
                "HNSW m mismatch between index and params",
            ));
        }
        if self.ef_construction != params.ef_construction {
            return Err(Status::invalid_argument(
                "HNSW ef_construction mismatch between index and params",
            ));
        }
        Ok(())
    }
}

fn load_vector_storage(vec_data: SegmentBytes, meta: &HnswMeta) -> ZResult<HnswVectorStorage> {
    let (count, vec_stride) = (meta.count, meta.vec_stride);
    if meta.quantize == QuantizeType::Undefined {
        if vec_stride % 4 != 0 {
            return Err(Status::io_error(
                "HNSW f32 vector stride is not 4-byte aligned",
            ));
        }
        let stride_floats = vec_stride / 4;
        let vectors = segment_array_f32(vec_data)?;
        if vectors.len() != count * stride_floats {
            return Err(Status::io_error("HNSW vectors length mismatch (stride)"));
        }
        return Ok(HnswVectorStorage::F32(vectors));
    }
    let expected = count
        .checked_mul(vec_stride)
        .ok_or_else(|| Status::io_error("HNSW vector segment overflow"))?;
    if vec_data.as_slice().len() != expected {
        return Err(Status::io_error("HNSW quantized vector length mismatch"));
    }
    Ok(HnswVectorStorage::Quantized {
        bytes: vec_data,
        quantize: meta.quantize,
        stride: vec_stride,
    })
}

fn read_u32_at(raw: &[u8], pos: &mut usize) -> ZResult<u32> {
    let Some(chunk) = raw.get(*pos..).and_then(<[u8]>::first_chunk::<4>) else {
        return Err(Status::io_error("HNSW upper index segment truncated"));
    };
    *pos += 4;
    Ok(u32::from_le_bytes(*chunk))
}

/// Parses `HNSW_UPPER_INDEX`: per level, fixed-length slot offsets into the upper-neighbor bytes.
fn parse_upper_index(raw: &[u8], upper_len: usize) -> ZResult<Vec<UpperLevelIndex>> {
    let mut pos = 0usize;
    let num_levels = read_u32_at(raw, &mut pos)? as usize;
    let mut out: Vec<UpperLevelIndex> = Vec::with_capacity(num_levels);
    for _ in 0..num_levels {
        let node_count = read_u32_at(raw, &mut pos)? as usize;
        let fixed_len = read_u32_at(raw, &mut pos)?;
        let byte_len = node_count
            .checked_mul(4)
            .ok_or_else(|| Status::io_error("HNSW upper index segment overflow"))?;
        let end = pos
            .checked_add(byte_len)
            .ok_or_else(|| Status::io_error("HNSW upper index segment overflow"))?;
        if end > raw.len() {
            return Err(Status::io_error("HNSW upper index segment truncated"));
        }
        let node_offsets = checked_upper_offsets(&raw[pos..end], fixed_len, upper_len)?;
        pos = end;
        out.push(UpperLevelIndex {
            node_offsets,
            fixed_len,
        });
    }
    Ok(out)
}

/// Decodes one level's slot offsets, rejecting slots that would run past `upper_len`.
fn checked_upper_offsets(raw: &[u8], fixed_len: u32, upper_len: usize) -> ZResult<Vec<u32>> {
    let mut offsets = Vec::with_capacity(raw.len() / 4);
    for chunk in raw.as_chunks::<4>().0 {
        let off = u32::from_le_bytes(*chunk);
        if off != u32::MAX {
            let end = (off as usize)
                .checked_add(fixed_len as usize * 4)
                .ok_or_else(|| Status::io_error("HNSW upper index offset overflow"))?;
            if end > upper_len {
                return Err(Status::io_error("HNSW upper index offset out of bounds"));
            }
        }
        offsets.push(off);
    }
    Ok(offsets)
}

/// Rejects an index whose `HNSW_HEADER` disagrees with `params`.
fn check_header(storage: &dyn StorageReader, params: &HnswIndexParams) -> ZResult<()> {
    let header = storage.read_segment(SEG_HEADER)?;
    let Some(rest) = header.as_slice().strip_prefix(&HEADER_MAGIC[..]) else {
        return Err(Status::io_error("HNSW header magic mismatch"));
    };
    let mut cur = Cursor::new(rest);
    let version = read_u32(&mut cur)?;
    if version == 0 || version > HEADER_VERSION {
        return Err(Status::io_error(format!(
            "unsupported HNSW index format version {version}"
        )));
    }
    let metric = read_u32(&mut cur)?;
    if metric != params.metric as u32 {
        return Err(Status::invalid_argument(format!(
            "HNSW metric mismatch: index was built with metric id {metric}, params ask for {:?}",
            params.metric
        )));
    }
    Ok(())
}

/// Rejects links past the last node; the upper-level descent reads their vectors unchecked.
fn check_upper_neighbor_ids(levels: &[UpperLevelIndex], upper: &[u8], count: usize) -> ZResult<()> {
    for level in levels {
        for &off in level.node_offsets.iter().filter(|&&off| off != u32::MAX) {
            let ids = u32_slice_at(upper, off as usize, level.fixed_len as usize);
            if ids.iter().any(|&id| id != u32::MAX && id as usize >= count) {
                return Err(Status::io_error("HNSW upper neighbor id out of range"));
            }
        }
    }
    Ok(())
}

impl HnswSearcher {
    pub fn load(storage: &dyn StorageReader, params: &HnswIndexParams) -> ZResult<Self> {
        check_header(storage, params)?;
        let meta = HnswMeta::parse(storage.read_segment(SEG_META)?.as_slice())?;
        meta.check_params(params)?;
        let count = meta.count;
        if count > 0 && meta.entry_point as usize >= count {
            return Err(Status::io_error("HNSW entry point out of range"));
        }

        let keys = segment_array_u64(storage.read_segment(SEG_KEYS)?)?;
        if keys.len() != count {
            return Err(Status::io_error("HNSW keys length mismatch"));
        }

        let vectors = load_vector_storage(storage.read_segment(SEG_VECTORS)?, &meta)?;
        // A level count that disagrees with the key count means a damaged index.
        let levels = segment_array_u32(storage.read_segment(SEG_LEVELS)?)?;
        if levels.len() != count {
            return Err(Status::io_error("HNSW levels length mismatch"));
        }

        let l0_neighbors = segment_array_u32(storage.read_segment(SEG_L0_NEIGHBORS)?)?;
        if l0_neighbors.len() != count * meta.m * 2 {
            return Err(Status::io_error("HNSW l0 neighbor length mismatch"));
        }

        if !cfg!(target_endian = "little") {
            return Err(Status::unimplemented(
                "HNSW mmap load not supported on big-endian targets",
            ));
        }
        let upper_bytes = storage.read_segment(SEG_UPPER_NEIGHBORS)?;
        let upper_index = parse_upper_index(
            storage.read_segment(SEG_UPPER_INDEX)?.as_slice(),
            upper_bytes.as_slice().len(),
        )?;
        check_upper_neighbor_ids(&upper_index, upper_bytes.as_slice(), count)?;

        Ok(HnswSearcher {
            keys,
            vectors,
            l0_neighbors,
            upper_bytes,
            upper_index,
            entry_point: meta.entry_point,
            max_level: meta.max_level,
            dim: meta.dim,
            count,
            m: meta.m,
            metric_type: params.metric,
            cosine_normalized: meta.cosine_normalized,
            vec_stride: meta.vec_stride,
            // default ef_search is 300 (independent of ef_construction).
            default_ef: 300,
            bruteforce_threshold: 100,
        })
    }

    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.metric_type {
            MetricType::InnerProduct => -simd::ip_f32(a, b),
            MetricType::Cosine => {
                if self.cosine_normalized {
                    1.0 - simd::ip_f32(a, b)
                } else {
                    simd::cosine_f32(a, b)
                }
            }
            MetricType::MipsL2 => simd::mips_l2_f32(a, b),
            MetricType::L2 | MetricType::Undefined | MetricType::Hamming => simd::l2_f32(a, b),
        }
    }

    #[inline]
    fn distance_to_with_query_sq_norm(&self, query: &[f32], query_sq_norm: f32, node: u32) -> f32 {
        match &self.vectors {
            HnswVectorStorage::F32(vectors) => {
                let stride_floats = self.vec_stride / 4;
                let start = node as usize * stride_floats;
                self.distance(query, &vectors.as_slice()[start..start + self.dim])
            }
            HnswVectorStorage::Quantized {
                bytes,
                quantize,
                stride,
            } => {
                let off = (node as usize) * (*stride);
                distance_to_quantized_with_query_sq_norm(
                    self.metric_type,
                    query,
                    query_sq_norm,
                    &bytes.as_slice()[off..off + *stride],
                    self.dim,
                    *quantize,
                )
            }
        }
    }

    #[inline]
    fn get_neighbors(&self, node: u32, level: usize) -> &[u32] {
        if level == 0 {
            return l0_links(self.l0_neighbors.as_slice(), self.m, node);
        }
        let Some(index) = self.upper_index.get(level - 1) else {
            return &[];
        };
        let Some(&off) = index.node_offsets.get(node as usize) else {
            return &[];
        };
        if off == u32::MAX {
            return &[];
        }
        u32_slice_at(
            self.upper_bytes.as_slice(),
            off as usize,
            index.fixed_len as usize,
        )
    }

    fn beam_search_into(
        &self,
        query: &[f32],
        query_sq_norm: f32,
        entry: u32,
        ef: usize,
        scratch: &mut SearchScratch,
    ) {
        scratch.entry_seeds.clear();
        scratch.entry_seeds.push(entry);
        self.beam_search_seeded_into(query, query_sq_norm, ef, scratch);
    }

    fn beam_search_seeded_into(
        &self,
        query: &[f32],
        query_sq_norm: f32,
        ef: usize,
        scratch: &mut SearchScratch,
    ) {
        VISITED_LIST.with(|visited_cell| {
            let mut visited = visited_cell.borrow_mut();
            visited.reset_for_len(self.count);

            scratch.reset(ef);

            for i in 0..scratch.entry_seeds.len() {
                let entry = scratch.entry_seeds[i];
                if !visited.insert_if_absent(entry) {
                    continue;
                }
                let entry_dist = self.distance_to_with_query_sq_norm(query, query_sq_norm, entry);
                scratch.candidates.push(MinDistNode(entry_dist, entry));
                scratch.results.push(DistNode(entry_dist, entry));
            }

            self.expand_l0_beam(query, query_sq_norm, ef, &mut visited, scratch);
            scratch.drain_results_ascending();
        })
    }

    /// Best-first expansion of the seeded level-0 beam.
    fn expand_l0_beam(
        &self,
        query: &[f32],
        query_sq_norm: f32,
        ef: usize,
        visited: &mut VisitedList,
        scratch: &mut SearchScratch,
    ) {
        let l0_m = self.m * 2;
        let l0 = self.l0_neighbors.as_slice();
        while let Some(MinDistNode(dist, node)) = scratch.candidates.pop() {
            // Search-time beam search only traverses level 0. Read the L0
            // neighbor slice directly instead of going through the generic
            // get_neighbors path used by upper-layer greedy descent.
            let nb_start = node as usize * l0_m;
            let nb_ptr = l0.as_ptr().wrapping_add(nb_start) as *const u8;
            prefetch_read(nb_ptr);
            prefetch_read(nb_ptr.wrapping_add(64));
            prefetch_read(nb_ptr.wrapping_add(128));

            let worst = scratch.results.peek().map(|d| d.0).unwrap_or(f32::INFINITY);
            if dist > worst && scratch.results.len() >= ef {
                break;
            }

            let neighbors = &l0[nb_start..nb_start + l0_m];

            // Phase 1: filter unvisited neighbors and prefetch their vectors.
            scratch.unvisited.clear();
            for &nb in neighbors {
                if nb != u32::MAX && visited.insert_if_absent(nb) {
                    scratch.unvisited.push(nb);
                }
            }
            if scratch.unvisited.is_empty() {
                continue;
            }

            // Phase 2: compute distances via scatter-batch (F32) or one-at-a-time (Quantized).
            self.score_unvisited(query, query_sq_norm, ef, scratch);

            // Cross-iteration prefetch: warm up the next candidate's neighbor
            // list so it's in cache when we read it at the top of the next
            // iteration.  Issued here (after results insertion) to maximise the
            // time window between the prefetch and the actual read.
            if let Some(&MinDistNode(_, next_node)) = scratch.candidates.peek() {
                let nb_ptr = l0.as_ptr().wrapping_add(next_node as usize * l0_m) as *const u8;
                prefetch_read(nb_ptr);
                prefetch_read(nb_ptr.wrapping_add(64));
                prefetch_read(nb_ptr.wrapping_add(128));
            }
        }
    }

    /// Scores `scratch.unvisited` and merges them into the beam.
    #[inline(always)]
    fn score_unvisited(
        &self,
        query: &[f32],
        query_sq_norm: f32,
        ef: usize,
        scratch: &mut SearchScratch,
    ) {
        match &self.vectors {
            HnswVectorStorage::F32(ref vectors) => {
                let query = &query[..self.dim];
                self.gather_scatter_ptrs(vectors.as_slice(), scratch);
                self.scatter_distances(query, query_sq_norm, scratch);
                scratch.insert_scored_batch(ef);
            }
            HnswVectorStorage::Quantized { bytes, stride, .. } => {
                // Quantized: prefetch + one-at-a-time (no scatter batch for quantized).
                let base_ptr = bytes.as_slice().as_ptr();
                for &nb in &scratch.unvisited {
                    let ptr = base_ptr.wrapping_add(nb as usize * *stride);
                    prefetch_read(ptr);
                    prefetch_read(ptr.wrapping_add(64));
                }
                self.insert_scored_one_by_one(query, query_sq_norm, ef, scratch);
            }
        }
    }

    /// Collects vector pointers for `scratch.unvisited` and prefetches them.
    #[inline(always)]
    fn gather_scatter_ptrs(&self, vectors: &[f32], scratch: &mut SearchScratch) {
        let stride_floats = self.vec_stride / 4;
        scratch.ptr_buf.clear();
        for &nb in &scratch.unvisited {
            let start = nb as usize * stride_floats;
            let ptr = vectors[start..start + self.dim].as_ptr();
            scratch.ptr_buf.push(ptr);
            let bytes = ptr as *const u8;
            prefetch_read(bytes);
            prefetch_read(bytes.wrapping_add(64));
            prefetch_read(bytes.wrapping_add(128));
            prefetch_read(bytes.wrapping_add(256));
        }
    }

    /// Batch distances from `query` to the gathered pointers, into `scratch.dist_buf`.
    #[inline(always)]
    fn scatter_distances(&self, query: &[f32], query_sq_norm: f32, scratch: &mut SearchScratch) {
        let n_unvisited = scratch.unvisited.len();
        scratch.dist_buf.resize(n_unvisited, 0.0);
        let (ptrs, dists) = (&scratch.ptr_buf, &mut scratch.dist_buf);
        match self.metric_type {
            // SAFETY: each `ptr_buf` entry starts a `dim`-long slice of `vectors`; `query` is `dim` long.
            MetricType::L2 | MetricType::Undefined | MetricType::Hamming => unsafe {
                simd::l2_scatter_f32(ptrs, query, self.dim, dists);
            },
            MetricType::InnerProduct => {
                // SAFETY: each `ptr_buf` entry starts a `dim`-long slice of `vectors`; `query` is `dim` long.
                unsafe {
                    simd::ip_scatter_f32(ptrs, query, self.dim, dists);
                }
                // Negate for min-heap (smaller = better for IP).
                for d in dists.iter_mut().take(n_unvisited) {
                    *d = -*d;
                }
            }
            MetricType::Cosine if self.cosine_normalized => {
                // SAFETY: each `ptr_buf` entry starts a `dim`-long slice of `vectors`; `query` is `dim` long.
                unsafe {
                    simd::ip_scatter_f32(ptrs, query, self.dim, dists);
                }
                for d in dists.iter_mut().take(n_unvisited) {
                    *d = 1.0 - *d;
                }
            }
            // MipsL2 and unnormalized Cosine: fall back to per-vector.
            _ => {
                for (i, &nb) in scratch.unvisited.iter().enumerate() {
                    dists[i] = self.distance_to_with_query_sq_norm(query, query_sq_norm, nb);
                }
            }
        }
    }

    /// Scores and inserts `scratch.unvisited` one at a time (quantized storage).
    #[inline(always)]
    fn insert_scored_one_by_one(
        &self,
        query: &[f32],
        query_sq_norm: f32,
        ef: usize,
        scratch: &mut SearchScratch,
    ) {
        let mut worst = scratch.results.peek().map(|d| d.0).unwrap_or(f32::INFINITY);
        for &nb in &scratch.unvisited {
            let nb_dist = self.distance_to_with_query_sq_norm(query, query_sq_norm, nb);
            if scratch.results.len() < ef {
                scratch.candidates.push(MinDistNode(nb_dist, nb));
                scratch.results.push(DistNode(nb_dist, nb));
                if scratch.results.len() == ef {
                    worst = scratch.results.peek().map(|d| d.0).unwrap_or(f32::INFINITY);
                }
            } else if nb_dist < worst {
                scratch.candidates.push(MinDistNode(nb_dist, nb));
                if let Some(mut top) = scratch.results.peek_mut() {
                    *top = DistNode(nb_dist, nb);
                }
                worst = scratch.results.peek().map(|d| d.0).unwrap_or(f32::INFINITY);
            }
        }
    }

    fn beam_search_layer_into(
        &self,
        query: &[f32],
        query_sq_norm: f32,
        entry: u32,
        ef: usize,
        level: usize,
        scratch: &mut SearchScratch,
    ) {
        let entry_dist = self.distance_to_with_query_sq_norm(query, query_sq_norm, entry);

        VISITED_LIST.with(|visited_cell| {
            let mut visited = visited_cell.borrow_mut();
            visited.reset_for_len(self.count);
            scratch.reset(ef);

            scratch.candidates.push(MinDistNode(entry_dist, entry));
            scratch.results.push(DistNode(entry_dist, entry));
            visited.insert(entry);

            let layer = LayerBeam {
                query,
                query_sq_norm,
                ef,
                level,
            };
            self.expand_layer_beam(&layer, &mut visited, scratch);
            scratch.drain_results_ascending();
        })
    }

    /// Best-first expansion of a single-entry beam on any level.
    fn expand_layer_beam(
        &self,
        layer: &LayerBeam<'_>,
        visited: &mut VisitedList,
        scratch: &mut SearchScratch,
    ) {
        let ef = layer.ef;
        while let Some(MinDistNode(dist, node)) = scratch.candidates.pop() {
            let worst = scratch.results.peek().map(|d| d.0).unwrap_or(f32::INFINITY);
            if dist > worst && scratch.results.len() >= ef {
                break;
            }

            let neighbors = self.get_neighbors(node, layer.level);
            for &nb in neighbors {
                if nb == u32::MAX || !visited.insert_if_absent(nb) {
                    continue;
                }
                let nb_dist =
                    self.distance_to_with_query_sq_norm(layer.query, layer.query_sq_norm, nb);
                let worst = scratch.results.peek().map(|d| d.0).unwrap_or(f32::INFINITY);
                if scratch.results.len() < ef {
                    scratch.candidates.push(MinDistNode(nb_dist, nb));
                    scratch.results.push(DistNode(nb_dist, nb));
                    continue;
                }
                let closer = nb_dist < worst;
                if !closer {
                    continue;
                }
                scratch.candidates.push(MinDistNode(nb_dist, nb));
                if let Some(mut top) = scratch.results.peek_mut() {
                    *top = DistNode(nb_dist, nb);
                }
            }
        }
    }

    pub fn search(
        &self,
        query: &[f32],
        topk: usize,
        params: HnswSearchParams,
        filter: Option<&dyn DocFilter>,
    ) -> ZResult<Vec<(u64, f32)>> {
        if self.count == 0 {
            return Ok(Vec::new());
        }
        check_query_dim(query.len(), self.dim)?;

        HNSW_SEARCH_SCRATCH.with(|scratch_cell| -> ZResult<Vec<(u64, f32)>> {
            let mut scratch = scratch_cell.borrow_mut();
            if self.metric_type == MetricType::Cosine && self.cosine_normalized {
                HNSW_QUERY_BUF.with(|buf_cell| {
                    let mut buf = buf_cell.borrow_mut();
                    buf.clear();
                    buf.extend_from_slice(&query[..self.dim]);
                    normalize_l2(&mut buf);
                    self.search_until_filled(&buf, topk, params, filter, &mut scratch)
                })
            } else {
                self.search_until_filled(query, topk, params, filter, &mut scratch)
            }
        })
    }

    /// Doubles `ef` while the filter leaves fewer than `topk` hits and the beam could still grow.
    fn search_until_filled(
        &self,
        q: &[f32],
        topk: usize,
        params: HnswSearchParams,
        filter: Option<&dyn DocFilter>,
        scratch: &mut SearchScratch,
    ) -> ZResult<Vec<(u64, f32)>> {
        let mut ef = params.ef.max(topk);
        loop {
            let hits =
                self.run_query(q, topk, HnswSearchParams { ef, ..params }, filter, scratch)?;
            // A beam that did not fill has reached every node the entry can reach.
            let beam_exhausted = scratch.out.len() < ef;
            if filter.is_none() || hits.len() >= topk || beam_exhausted || ef >= self.count {
                return Ok(hits);
            }
            ef = ef.saturating_mul(2).min(self.count);
        }
    }

    /// Descends to level 0, searches it with `params.ef`, and collects up to `topk` hits.
    fn run_query(
        &self,
        q: &[f32],
        topk: usize,
        params: HnswSearchParams,
        filter: Option<&dyn DocFilter>,
        scratch: &mut SearchScratch,
    ) -> ZResult<Vec<(u64, f32)>> {
        let q2 = if matches!(&self.vectors, HnswVectorStorage::Quantized { .. }) {
            query_sq_norm(self.metric_type, q, self.dim)
        } else {
            0.0
        };
        let ep = self.descend_to_level1(q, q2, params.upper_ef, scratch);
        self.search_level0(q, q2, ep, params, scratch);
        self.collect_hits(topk, filter, scratch)
    }

    /// Greedy (or `upper_ef`-wide beam) descent through the upper levels.
    fn descend_to_level1(
        &self,
        q: &[f32],
        q2: f32,
        upper_ef: Option<usize>,
        scratch: &mut SearchScratch,
    ) -> u32 {
        let mut ep = Entry {
            node: self.entry_point,
            dist: self.distance_to_with_query_sq_norm(q, q2, self.entry_point),
        };
        if let Some(upper_ef) = upper_ef.map(|ef| ef.max(1)) {
            for lv in (1..=self.max_level).rev() {
                self.beam_search_layer_into(q, q2, ep.node, upper_ef, lv, scratch);
                if let Some(&(_, node)) = scratch.out.first() {
                    ep.node = node;
                }
            }
            return ep.node;
        }
        for lv in (1..=self.max_level).rev() {
            ep = self.descend_level(q, q2, ep, lv);
        }
        ep.node
    }

    /// Greedy walk on one upper level until no neighbor is closer to `q`.
    fn descend_level(&self, q: &[f32], q2: f32, mut ep: Entry, level: usize) -> Entry {
        let mut changed = true;
        while changed {
            changed = false;
            let neighbors = self.get_neighbors(ep.node, level);
            // Prefetch vector data for all neighbors before computing distances.
            self.prefetch_vectors(neighbors);
            for &nb in neighbors {
                if nb == u32::MAX {
                    continue;
                }
                let d = self.distance_to_with_query_sq_norm(q, q2, nb);
                if d < ep.dist {
                    ep = Entry { node: nb, dist: d };
                    changed = true;
                }
            }
        }
        ep
    }

    #[inline]
    fn prefetch_vectors(&self, neighbors: &[u32]) {
        let HnswVectorStorage::F32(ref vectors) = self.vectors else {
            return;
        };
        let stride_floats = self.vec_stride / 4;
        let base = vectors.as_slice().as_ptr();
        for &nb in neighbors {
            if nb != u32::MAX {
                let ptr = base.wrapping_add(nb as usize * stride_floats);
                prefetch_read(ptr as *const u8);
                prefetch_read((ptr as *const u8).wrapping_add(64));
            }
        }
    }

    /// Level-0 beam search seeded with the best level-1 routing points; `ef` is unchanged.
    fn search_level0(
        &self,
        q: &[f32],
        q2: f32,
        ep: u32,
        params: HnswSearchParams,
        scratch: &mut SearchScratch,
    ) {
        let ef = params.ef;
        let l0_seeds = params.l0_seeds.min(ef);
        if l0_seeds <= 1 || self.max_level < 1 {
            self.beam_search_into(q, q2, ep, ef, scratch);
            return;
        }
        self.beam_search_layer_into(q, q2, ep, l0_seeds, 1, scratch);
        scratch.entry_seeds.clear();
        for &(_, node) in scratch.out.iter().take(l0_seeds) {
            if !scratch.entry_seeds.contains(&node) {
                scratch.entry_seeds.push(node);
            }
        }
        if scratch.entry_seeds.is_empty() {
            scratch.entry_seeds.push(ep);
        }
        self.beam_search_seeded_into(q, q2, ef, scratch);
    }

    fn collect_hits(
        &self,
        topk: usize,
        filter: Option<&dyn DocFilter>,
        scratch: &mut SearchScratch,
    ) -> ZResult<Vec<(u64, f32)>> {
        let keys = self.keys.as_slice();
        if let Some(filter) = filter {
            scratch.topk_heap.reset(topk);
            for (dist, node) in scratch.out.iter().copied() {
                let key = keys[node as usize];
                if doc_filter_allows(Some(filter), key)? {
                    scratch.topk_heap.push(dist, key);
                }
            }
            return Ok(scratch.topk_heap.drain_sorted());
        }
        let mut result = Vec::with_capacity(topk);
        for &(dist, node) in scratch.out.iter().take(topk) {
            result.push((keys[node as usize], dist));
        }
        Ok(result)
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

/// Query and bounds of one single-level beam search.
struct LayerBeam<'a> {
    query: &'a [f32],
    query_sq_norm: f32,
    ef: usize,
    level: usize,
}

impl SearchScratch {
    /// Merges `unvisited[i]` at distance `dist_buf[i]` into the beam.
    #[inline(always)]
    fn insert_scored_batch(&mut self, ef: usize) {
        // Worst distance is cached across the loop to avoid repeated heap peeks.
        let mut worst = self.results.peek().map(|d| d.0).unwrap_or(f32::INFINITY);
        for i in 0..self.unvisited.len() {
            let nb = self.unvisited[i];
            let nb_dist = self.dist_buf[i];
            if self.results.len() < ef {
                self.candidates.push(MinDistNode(nb_dist, nb));
                self.results.push(DistNode(nb_dist, nb));
                if self.results.len() == ef {
                    worst = self.results.peek().map(|d| d.0).unwrap_or(f32::INFINITY);
                }
            } else if nb_dist < worst {
                self.candidates.push(MinDistNode(nb_dist, nb));
                if let Some(mut top) = self.results.peek_mut() {
                    *top = DistNode(nb_dist, nb);
                }
                worst = self.results.peek().map(|d| d.0).unwrap_or(f32::INFINITY);
            }
        }
    }

    /// Moves the beam results into `out`, nearest first.
    fn drain_results_ascending(&mut self) {
        self.out.clear();
        self.out.reserve(self.results.len());
        while let Some(d) = self.results.pop() {
            self.out.push((d.0, d.1));
        }
        // Heap pop gives largest-first; reverse to get smallest-first.
        self.out.reverse();
    }
}

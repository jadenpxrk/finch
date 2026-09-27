use super::*;
use crate::algorithm::codec::{
    read_optional_u32, read_u32, read_u64, scan_upper_runs, u32_slice_at, UpperScanErrors,
};
use crate::quantizer::quantize_type_from_u32;

pub struct HnswSparseSearcher {
    rows: SparseRows,
    l0_neighbors: SegmentArray<u32>, // flat: count × (2*m)
    upper_bytes: Option<SegmentBytes>,
    upper_index: Vec<UpperLevelIndex>,
    entry_point: u32,
    max_level: usize,
    count: usize,
    m: usize,
    quantize: QuantizeType,
}

/// `HNSW_SPARSE_META`; fields after `m` are absent in older indexes.
struct SparseGraphMeta {
    count: usize,
    entry_point: u32,
    max_level: usize,
    m: usize,
    quantize: QuantizeType,
    ef_construction: Option<usize>,
    scaling_factor: Option<usize>,
}

impl SparseGraphMeta {
    fn read(storage: &dyn StorageReader) -> ZResult<Self> {
        let meta = storage.read_segment(SEG_META)?;
        let mut cur = Cursor::new(meta.as_slice());
        let count = read_u64(&mut cur)? as usize;
        let entry_point = read_u32(&mut cur)?;
        let max_level = read_u32(&mut cur)? as usize;
        let m = read_u32(&mut cur)? as usize;
        let quantize =
            read_optional_u32(&mut cur)?.map_or(QuantizeType::Undefined, quantize_type_from_u32);
        let ef_construction = read_optional_u32(&mut cur)?.map(|v| v as usize);
        let scaling_factor = read_optional_u32(&mut cur)?.map(|v| v as usize);
        Ok(SparseGraphMeta {
            count,
            entry_point,
            max_level,
            m,
            quantize,
            ef_construction,
            scaling_factor,
        })
    }

    fn check_params(&self, params: &HnswIndexParams) -> ZResult<()> {
        if self.quantize != params.quantize {
            return Err(Status::invalid_argument(
                "sparse quantize mismatch between index and params",
            ));
        }
        if self.m != params.m {
            return Err(Status::invalid_argument(
                "sparse HNSW m mismatch between index and params",
            ));
        }
        if self.ef_construction.unwrap_or(params.ef_construction) != params.ef_construction {
            return Err(Status::invalid_argument(
                "sparse HNSW ef_construction mismatch between index and params",
            ));
        }
        if self.scaling_factor.unwrap_or(params.scaling_factor) != params.scaling_factor {
            return Err(Status::invalid_argument(
                "sparse HNSW scaling_factor mismatch between index and params",
            ));
        }
        Ok(())
    }
}

/// Upper-level neighbor bytes and their per-node runs; empty for indexes written without them.
struct SparseUpper {
    bytes: Option<SegmentBytes>,
    index: Vec<UpperLevelIndex>,
}

fn load_upper(storage: &dyn StorageReader) -> ZResult<SparseUpper> {
    if !storage.exists(SEG_UPPER_NEIGHBORS) {
        return Ok(SparseUpper {
            bytes: None,
            index: Vec::new(),
        });
    }
    if !cfg!(target_endian = "little") {
        return Err(Status::unimplemented(
            "HNSW sparse mmap load not supported on big-endian targets",
        ));
    }
    let bytes = storage.read_segment(SEG_UPPER_NEIGHBORS)?;
    let errors = UpperScanErrors {
        overflow: "HNSW_SPARSE upper neighbor overflow",
        truncated: "HNSW_SPARSE upper neighbor segment truncated",
    };
    let index = scan_upper_runs(bytes.as_slice(), &errors, |run| run)?
        .into_iter()
        .map(|node_offsets| UpperLevelIndex { node_offsets })
        .collect();
    Ok(SparseUpper {
        bytes: Some(bytes),
        index,
    })
}

impl HnswSparseSearcher {
    #[inline]
    fn vector_slice(&self, node: u32) -> &[u8] {
        self.rows.vector(node as usize)
    }

    pub fn load_with_params(
        storage: &dyn StorageReader,
        params: &HnswIndexParams,
    ) -> ZResult<Self> {
        let meta = SparseGraphMeta::read(storage)?;
        meta.check_params(params)?;
        Self::load_graph(storage, &meta, true)
    }

    pub fn load(storage: &dyn StorageReader) -> ZResult<Self> {
        // Backward-compatible load: trusts whatever encoding the index stored.
        let meta = SparseGraphMeta::read(storage)?;
        Self::load_graph(storage, &meta, false)
    }

    fn load_graph(
        storage: &dyn StorageReader,
        meta: &SparseGraphMeta,
        check_levels: bool,
    ) -> ZResult<Self> {
        let n = meta.count;
        let rows = SparseRows::load(storage, &SEGMENTS, n, meta.quantize)?;

        // Levels (optional for backward-compat)
        if check_levels && storage.exists(SEG_LEVELS) {
            let levels = segment_array_u32(storage.read_segment(SEG_LEVELS)?)?;
            if levels.len() != n {
                return Err(Status::io_error("HNSW_SPARSE levels length mismatch"));
            }
        }

        let l0_neighbors = segment_array_u32(storage.read_segment(SEG_L0_NEIGHBORS)?)?;
        if l0_neighbors.len() != n * meta.m * 2 {
            return Err(Status::io_error("HNSW_SPARSE L0 neighbor length mismatch"));
        }

        // Upper neighbors (optional for backward-compat)
        let upper = load_upper(storage)?;
        let max_level_loaded = upper.index.len();
        if max_level_loaded != 0 && max_level_loaded != meta.max_level {
            return Err(Status::io_error("HNSW_SPARSE upper level count mismatch"));
        }

        Ok(Self {
            rows,
            l0_neighbors,
            upper_bytes: upper.bytes,
            upper_index: upper.index,
            entry_point: meta.entry_point,
            max_level: meta.max_level,
            count: n,
            m: meta.m,
            quantize: meta.quantize,
        })
    }

    #[inline]
    fn get_neighbors(&self, node: u32, level: usize) -> &[u32] {
        if level == 0 {
            return l0_links(self.l0_neighbors.as_slice(), self.m, node);
        }
        let lv_idx = level - 1;
        if lv_idx >= self.upper_index.len() {
            return &[];
        }
        let Some(run) = self.upper_index[lv_idx].node_offsets.get(node as usize) else {
            return &[];
        };
        if run.len == 0 {
            return &[];
        }
        let Some(bytes) = self.upper_bytes.as_ref() else {
            return &[];
        };
        u32_slice_at(bytes.as_slice(), run.start as usize, run.len as usize)
    }

    fn beam_search_into(
        &self,
        query: &SparseVector,
        entry: u32,
        ef: usize,
        level: usize,
        scratch: &mut SearchScratch,
    ) {
        VISITED_LIST.with(|visited_cell| {
            let mut visited = visited_cell.borrow_mut();
            beam_search(
                scratch.beam(&mut visited),
                entry,
                ef,
                self.count,
                |nb| sparse_neg_ip_encoded(query, self.vector_slice(nb), self.quantize),
                |node| self.get_neighbors(node, level),
            );
        });
    }

    pub fn search(
        &self,
        query: &SparseVector,
        topk: usize,
        ef: usize,
        filter: Option<&dyn DocFilter>,
    ) -> ZResult<Vec<(u64, f32)>> {
        if self.count == 0 {
            return Ok(Vec::new());
        }
        let ef = ef.max(topk);

        let mut heap = TopkHeap::new(topk);
        HNSW_SPARSE_SEARCH_SCRATCH.with(|scratch_cell| -> ZResult<()> {
            let mut scratch = scratch_cell.borrow_mut();

            // Greedy descent to level 1.
            let ep = greedy_descend(
                self.entry_point,
                (1..=self.max_level).rev(),
                |nb| sparse_neg_ip_encoded(query, self.vector_slice(nb), self.quantize),
                |node, lv| self.get_neighbors(node, lv),
            );

            // Beam search at level 0.
            self.beam_search_into(query, ep, ef, 0, &mut scratch);

            let keys = self.rows.keys.as_slice();
            for (dist, node) in scratch.out.iter().copied() {
                let key = keys[node as usize];
                if doc_filter_allows(filter, key)? {
                    // return distance-like scores (negative inner product).
                    heap.push(dist, key);
                }
            }
            Ok(())
        })?;

        Ok(heap.into_sorted())
    }
}

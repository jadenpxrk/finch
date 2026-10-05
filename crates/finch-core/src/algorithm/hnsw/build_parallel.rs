use super::*;

impl HnswBuilder {
    /// Build an HNSW index in parallel using multiple threads.
    ///
    /// All vectors must be provided upfront. The build uses a sequential seed
    /// phase for the first batch of nodes, then inserts the remainder in
    /// parallel using rayon.
    pub fn build_parallel(
        keys: &[u64],
        vectors: &[f32],
        dim: usize,
        params: HnswIndexParams,
        concurrency: usize,
    ) -> ZResult<Self> {
        let n = keys.len();
        if n == 0 {
            return Ok(HnswBuilder::new(dim, params));
        }
        if vectors.len() != n * dim {
            return Err(Status::invalid_argument(format!(
                "vectors length {} != keys.len() {} * dim {}",
                vectors.len(),
                n,
                dim,
            )));
        }

        // For small inputs or single-threaded, fall back to sequential.
        if concurrency <= 1 || n <= 1 {
            let mut builder = HnswBuilder::new(dim, params);
            for i in 0..n {
                builder.add(keys[i], &vectors[i * dim..(i + 1) * dim])?;
            }
            return Ok(builder.finish_build());
        }

        let graph = SharedGraph::new(keys, vectors, dim, &params);

        // Sequential seed phase: insert the first few points to bootstrap the graph.
        // Qdrant uses a small fixed single-threaded prefix in release builds to
        // avoid disconnected components, then lets the remaining points build in
        // their natural iteration order. Keeping insertion order stable matters:
        // level-sorting the whole build over-emphasizes upper-layer points early
        // and changes the bottom-layer graph shape under the same HNSW params.
        let seed_count = n.min(256);
        graph.insert_seed(seed_count);
        if seed_count < n {
            graph.insert_parallel(seed_count, concurrency)?;
        }

        Ok(graph.into_builder(params).finish_build())
    }

    fn finish_build(mut self) -> Self {
        if self.params.build_tuning.l0_repair {
            self.repair_l0_neighbors();
        }
        self.refine_l0_neighbors();
        self
    }
}

impl SharedGraph {
    fn new(keys: &[u64], vectors: &[f32], dim: usize, params: &HnswIndexParams) -> Self {
        let n = keys.len();
        let m = params.m;
        let vec_stride_floats = aligned_vector_stride_floats(dim);
        let heuristic_dim = heuristic_dim(&params.build_tuning, dim);

        // Pre-roll random levels for all nodes (deterministic per build).
        let mut levels: Vec<u32> = Vec::with_capacity(n);
        {
            let mut rng = rand::thread_rng();
            for _ in 0..n {
                levels.push(random_level_with_scaling(&mut rng, params.scaling_factor) as u32);
            }
        }
        let global_max_level = levels.iter().copied().max().unwrap_or(0) as usize;
        let normalize_cosine = params.metric == MetricType::Cosine;
        let graph_vectors = strided_vectors(vectors, n, dim, vec_stride_floats, normalize_cosine);
        let upper_neighbors = shared_upper_levels(&levels, m, global_max_level);
        let l0_neighbors = vec![u32::MAX; n * m * 2];

        SharedGraph {
            vectors: graph_vectors,
            keys: keys.to_vec(),
            levels,
            l0_neighbors: l0_neighbors.into_iter().map(AtomicU32::new).collect(),
            l0_locks: (0..n).map(|_| Mutex::new(())).collect(),
            upper_neighbors,
            ready: (0..n).map(|_| AtomicBool::new(false)).collect(),
            entry_point: AtomicU32::new(u32::MAX),
            max_level: AtomicUsize::new(0),
            promotion_lock: Mutex::new(()),
            dim,
            vec_stride_floats,
            heuristic_dim,
            tuning: params.build_tuning,
            m,
            ef_construction: params.ef_construction,
            l0_m: m * 2,
            metric_type: params.metric,
        }
    }

    /// Makes node 0 the entry point, then inserts nodes `1..seed_count` on this thread.
    fn insert_seed(&self, seed_count: usize) {
        self.entry_point.store(0, Ordering::Release);
        self.max_level
            .store(self.levels[0] as usize, Ordering::Release);
        self.ready[0].store(true, Ordering::Release);

        let mut scratch = BuildScratch::new();
        for node_id in 1..seed_count {
            par_insert_node(self, node_id as u32, &mut scratch);
        }
    }

    fn insert_parallel(&self, first: usize, concurrency: usize) -> ZResult<()> {
        let parallel_nodes: Vec<u32> = (first as u32..self.keys.len() as u32).collect();

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(concurrency)
            .build()
            .map_err(|e| Status::internal(format!("failed to create thread pool: {}", e)))?;

        pool.install(|| {
            parallel_nodes
                .into_par_iter()
                .for_each_init(BuildScratch::new, |scratch, node_id| {
                    par_insert_node(self, node_id, scratch);
                });
        });
        Ok(())
    }

    fn into_builder(self, params: HnswIndexParams) -> HnswBuilder {
        let final_l0: Vec<u32> = self
            .l0_neighbors
            .iter()
            .map(|a| a.load(Ordering::Relaxed))
            .collect();
        let final_upper: Vec<UpperLevelNeighbors> = self
            .upper_neighbors
            .iter()
            .map(|ul| UpperLevelNeighbors {
                offsets: ul.offsets.clone(),
                neighbors: ul
                    .neighbors
                    .iter()
                    .map(|a| a.load(Ordering::Relaxed))
                    .collect(),
            })
            .collect();

        let mut builder = HnswBuilder::new(self.dim, params);
        builder.vectors = self.vectors;
        builder.keys = self.keys;
        builder.levels = self.levels;
        builder.l0_neighbors = final_l0;
        builder.upper_neighbors = final_upper;
        builder.entry_point = self.entry_point.load(Ordering::Relaxed);
        builder.max_level = self.max_level.load(Ordering::Relaxed);
        builder
    }
}

/// Copies (and for cosine, normalizes) `dim`-long rows into `stride`-aligned storage.
fn strided_vectors(
    vectors: &[f32],
    n: usize,
    dim: usize,
    stride: usize,
    normalize: bool,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n * stride];
    for i in 0..n {
        let src = &vectors[i * dim..(i + 1) * dim];
        let dst_start = i * stride;
        out[dst_start..dst_start + dim].copy_from_slice(src);
        if normalize {
            normalize_l2(&mut out[dst_start..dst_start + dim]);
        }
    }
    out
}

/// Empty upper-level slots for every node whose pre-rolled level reaches each level.
fn shared_upper_levels(levels: &[u32], m: usize, max_level: usize) -> Vec<SharedUpperLevel> {
    let n = levels.len();
    let mut upper: Vec<SharedUpperLevel> = Vec::with_capacity(max_level);
    for lv_idx in 0..max_level {
        let mut offsets = vec![u32::MAX; n];
        let mut neighbor_count = 0usize;
        for node in 0..n {
            if (levels[node] as usize) > lv_idx {
                offsets[node] = (neighbor_count * m) as u32;
                neighbor_count += 1;
            }
        }
        upper.push(SharedUpperLevel {
            offsets,
            neighbors: (0..neighbor_count * m)
                .map(|_| AtomicU32::new(u32::MAX))
                .collect(),
            locks: (0..n).map(|_| Mutex::new(())).collect(),
        });
    }
    upper
}

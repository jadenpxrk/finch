use crate::types::{MetricType, QuantizeType};
use serde::{Deserialize, Serialize};

/// Parameters for HNSW index construction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswIndexParams {
    /// Number of bidirectional links per node (default: 50)
    pub m: usize,
    /// Size of dynamic candidate list during construction (default: 500)
    pub ef_construction: usize,
    /// Scaling factor controlling level probabilities (default: 50)
    pub scaling_factor: usize,
    pub metric: MetricType,
    pub quantize: QuantizeType,
    /// Number of threads for parallel HNSW construction.
    /// `None` = all available cores, `Some(1)` = sequential.
    #[serde(default)]
    pub build_concurrency: Option<usize>,
    /// Neighbor-selection heuristics used while building; recorded in the index file.
    #[serde(default)]
    pub build_tuning: HnswBuildTuning,
}

/// Build-time HNSW heuristics. The defaults are the tuned behavior.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HnswBuildTuning {
    /// Dimension prefix for neighbor-selection distances; `None` uses every dimension.
    pub heuristic_dim: Option<usize>,
    /// Take the nearest candidates without the diversity heuristic.
    pub simple_neighbor_select: bool,
    /// Fill slots the heuristic left free with the candidates it pruned.
    pub keep_pruned_connections: bool,
    /// Candidate cap of a two-hop level-0 refinement pass; `None` skips the pass.
    pub l0_refine_candidate_cap: Option<usize>,
    /// Run the level-0 repair passes after a parallel build.
    pub l0_repair: bool,
    /// A candidate is pruned when `prune_alpha * dist(candidate, kept) < dist(candidate, query)`.
    pub prune_alpha: f32,
    /// Qdrant's incremental backlink heuristic, which keeps existing links in order.
    pub qdrant_backlink: bool,
}

impl Default for HnswBuildTuning {
    fn default() -> Self {
        HnswBuildTuning {
            heuristic_dim: None,
            simple_neighbor_select: false,
            keep_pruned_connections: true,
            l0_refine_candidate_cap: None,
            l0_repair: true,
            prune_alpha: 1.0,
            qdrant_backlink: false,
        }
    }
}

impl HnswIndexParams {
    pub fn new(metric: MetricType) -> Self {
        let m = 50;
        HnswIndexParams {
            m,
            ef_construction: 500,
            scaling_factor: m,
            metric,
            quantize: QuantizeType::Undefined,
            build_concurrency: None,
            build_tuning: HnswBuildTuning::default(),
        }
    }

    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self.scaling_factor = m;
        self
    }

    pub fn with_ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    pub fn with_scaling_factor(mut self, scaling_factor: usize) -> Self {
        self.scaling_factor = scaling_factor;
        self
    }

    pub fn with_quantize(mut self, q: QuantizeType) -> Self {
        self.quantize = q;
        self
    }

    pub fn with_build_concurrency(mut self, c: usize) -> Self {
        self.build_concurrency = Some(c);
        self
    }
}

/// Parameters for IVF index construction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IvfIndexParams {
    /// Number of Voronoi cells (default: 1024)
    pub n_list: usize,
    /// K-means iterations (default: 10)
    pub n_iters: usize,
    /// Enable SOAR (Stable Orderings for Approximate Recall)
    pub use_soar: bool,
    /// Optional nested L1 sub-index (composite IVF).
    ///
    /// When set to `Some(IndexParams::Ivf(_))`, finch-core may build a two-level
    /// IVF layout (outer IVF partitions, inner IVF partitions within each cell).
    #[serde(default)]
    pub l1_index: Option<Box<IndexParams>>,
    pub metric: MetricType,
    pub quantize: QuantizeType,
}

impl IvfIndexParams {
    pub fn new(metric: MetricType) -> Self {
        IvfIndexParams {
            n_list: 1024,
            n_iters: 10,
            use_soar: false,
            l1_index: None,
            metric,
            quantize: QuantizeType::Undefined,
        }
    }

    pub fn with_n_list(mut self, n: usize) -> Self {
        self.n_list = n;
        self
    }

    pub fn with_quantize(mut self, q: QuantizeType) -> Self {
        self.quantize = q;
        self
    }
}

/// Parameters for FLAT (brute-force) index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatIndexParams {
    pub metric: MetricType,
    pub quantize: QuantizeType,
    /// Use column-major storage for batch distance computation
    pub column_major: bool,
}

impl FlatIndexParams {
    pub fn new(metric: MetricType) -> Self {
        FlatIndexParams {
            metric,
            quantize: QuantizeType::Undefined,
            column_major: false,
        }
    }

    pub fn with_quantize(mut self, q: QuantizeType) -> Self {
        self.quantize = q;
        self
    }
}

/// Parameters for inverted (keyword) index
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InvertIndexParams {
    /// Enable range query optimization (sorted storage)
    pub enable_range_optimization: bool,
    /// Enable extended wildcard matching
    pub enable_extended_wildcard: bool,
}

/// Index parameters (enum variant per algorithm)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexParams {
    Hnsw(HnswIndexParams),
    HnswSparse(HnswIndexParams),
    Ivf(IvfIndexParams),
    Flat(FlatIndexParams),
    FlatSparse(FlatIndexParams),
    Invert(InvertIndexParams),
}

impl IndexParams {
    pub fn metric(&self) -> Option<MetricType> {
        match self {
            IndexParams::Hnsw(p) | IndexParams::HnswSparse(p) => Some(p.metric),
            IndexParams::Ivf(p) => Some(p.metric),
            IndexParams::Flat(p) | IndexParams::FlatSparse(p) => Some(p.metric),
            IndexParams::Invert(_) => None,
        }
    }

    pub fn quantize(&self) -> Option<QuantizeType> {
        match self {
            IndexParams::Hnsw(p) | IndexParams::HnswSparse(p) => Some(p.quantize),
            IndexParams::Ivf(p) => Some(p.quantize),
            IndexParams::Flat(p) | IndexParams::FlatSparse(p) => Some(p.quantize),
            IndexParams::Invert(_) => None,
        }
    }
}

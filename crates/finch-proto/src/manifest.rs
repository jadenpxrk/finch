//! Manifest types for collection metadata persistence.
//! Wire format uses protobuf encoding (see proto/manifest.proto).

use std::collections::BTreeMap;

use finch_types::HnswBuildTuning;
use prost::Message;
use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum ManifestCodecError {
    #[error("protobuf encode error: {0}")]
    ProtobufEncode(String),
    #[error("protobuf decode error: {0}")]
    ProtobufDecode(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Manifest {
    pub version: u32,
    pub schema: Option<CollectionSchema>,
    pub enable_mmap: bool,
    pub persisted_segment_metas: Vec<SegmentMeta>,
    pub writing_segment_meta: Option<SegmentMeta>,
    pub id_map_path_suffix: u32,
    pub delete_snapshot_path_suffix: u32,
    pub next_segment_id: u32,
}

impl Manifest {
    pub fn encode(&self) -> Result<Vec<u8>, ManifestCodecError> {
        let pb = to_pb_manifest(self);
        Ok(pb.encode_to_vec())
    }

    pub fn decode(data: &[u8]) -> Result<Self, ManifestCodecError> {
        pb::Manifest::decode(data)
            .map(from_pb_manifest)
            .map_err(|e| ManifestCodecError::ProtobufDecode(e.to_string()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SegmentMeta {
    pub segment_id: u32,
    pub persisted_blocks: Vec<BlockMeta>,
    pub writing_forward_block: Option<BlockMeta>,
    pub indexed_vector_fields: Vec<String>,
    // Rebuilt vector indexes live in a numbered directory; a missing field means generation 0.
    #[serde(default)]
    pub vector_index_generations: BTreeMap<String, u32>,
    // Denormalized segment stats used by finch-db internals.
    pub min_doc_id: u64,
    pub max_doc_id: u64,
    pub doc_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BlockMeta {
    pub block_id: u32,
    pub block_type: u32,
    pub min_doc_id: u64,
    pub max_doc_id: u64,
    pub doc_count: u64,
    pub columns: Vec<String>,
    // Not present in proto BlockMeta, kept for finch internal use.
    pub field_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CollectionSchema {
    pub name: String,
    pub fields: Vec<FieldSchema>,
    pub max_doc_count_per_segment: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FieldSchema {
    pub name: String,
    pub data_type: u32,
    pub nullable: bool,
    pub dimension: u32,
    pub index_params: Option<FieldIndexParams>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum FieldIndexParams {
    Hnsw(HnswIndexParams),
    HnswSparse(HnswIndexParams),
    Ivf(IvfIndexParams),
    Flat(FlatIndexParams),
    FlatSparse(FlatIndexParams),
    Invert(InvertIndexParams),
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HnswIndexParams {
    pub m: u32,
    pub ef_construction: u32,
    pub scaling_factor: u32,
    pub metric: u32,
    pub quantize: u32,
    #[serde(default)]
    pub build_tuning: HnswBuildTuning,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IvfIndexParams {
    pub n_list: u32,
    pub n_iters: u32,
    pub use_soar: bool,
    pub metric: u32,
    pub quantize: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FlatIndexParams {
    pub metric: u32,
    pub quantize: u32,
    pub column_major: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InvertIndexParams {
    pub enable_range_optimization: bool,
    pub enable_extended_wildcard: bool,
}

fn u32_to_i32_sat(v: u32) -> i32 {
    if v > i32::MAX as u32 {
        i32::MAX
    } else {
        v as i32
    }
}

fn i32_to_u32_nonneg(v: i32) -> u32 {
    if v < 0 {
        0
    } else {
        v as u32
    }
}

fn to_pb_base(metric: u32, quantize: u32) -> pb::BaseIndexParams {
    pb::BaseIndexParams {
        metric_type: u32_to_i32_sat(metric),
        quantize_type: u32_to_i32_sat(quantize),
    }
}

fn from_pb_base(base: Option<pb::BaseIndexParams>) -> (u32, u32) {
    match base {
        Some(b) => (
            i32_to_u32_nonneg(b.metric_type),
            i32_to_u32_nonneg(b.quantize_type),
        ),
        None => (0, 0),
    }
}

fn to_pb_block_meta(b: &BlockMeta) -> pb::BlockMeta {
    pb::BlockMeta {
        block_id: b.block_id,
        block_type: u32_to_i32_sat(b.block_type),
        min_doc_id: b.min_doc_id,
        max_doc_id: b.max_doc_id,
        doc_count: b.doc_count,
        columns: b.columns.clone(),
    }
}

fn from_pb_block_meta(b: pb::BlockMeta) -> BlockMeta {
    BlockMeta {
        block_id: b.block_id,
        block_type: i32_to_u32_nonneg(b.block_type),
        min_doc_id: b.min_doc_id,
        max_doc_id: b.max_doc_id,
        doc_count: b.doc_count,
        columns: b.columns,
        field_name: String::new(),
    }
}

const BT_SCALAR: i32 = 1;

fn segment_stats_from_pb(seg: &pb::SegmentMeta) -> (u64, u64, u64) {
    let mut min_doc_id = u64::MAX;
    let mut max_doc_id = 0u64;
    let mut doc_count = 0u64;

    for b in &seg.persisted_blocks {
        if b.block_type == BT_SCALAR {
            min_doc_id = min_doc_id.min(b.min_doc_id);
            max_doc_id = max_doc_id.max(b.max_doc_id);
            doc_count = doc_count.saturating_add(b.doc_count);
        }
    }

    if let Some(wf) = &seg.writing_forward_block {
        if wf.doc_count != 0 {
            max_doc_id = wf.max_doc_id;
        }
        doc_count = doc_count.saturating_add(wf.doc_count);
        if min_doc_id == u64::MAX {
            min_doc_id = wf.min_doc_id;
        }
    }

    if min_doc_id == u64::MAX {
        min_doc_id = 0;
    }

    (min_doc_id, max_doc_id, doc_count)
}

fn to_pb_segment_meta(seg: &SegmentMeta) -> pb::SegmentMeta {
    let mut persisted_blocks: Vec<pb::BlockMeta> =
        seg.persisted_blocks.iter().map(to_pb_block_meta).collect();

    let has_scalar_block = persisted_blocks.iter().any(|b| b.block_type == BT_SCALAR);
    // doc_id ranges are allowed to start at 0, so we must not treat
    // `min_doc_id == 0` as "missing stats". Inject a synthetic scalar block
    // whenever any denormalized segment stats are present.
    let has_stats = seg.doc_count != 0 || seg.min_doc_id != 0 || seg.max_doc_id != 0;
    if !has_scalar_block && has_stats {
        persisted_blocks.push(pb::BlockMeta {
            block_id: 0,
            block_type: BT_SCALAR,
            min_doc_id: seg.min_doc_id,
            max_doc_id: seg.max_doc_id,
            doc_count: seg.doc_count,
            columns: Vec::new(),
        });
    }

    pb::SegmentMeta {
        segment_id: seg.segment_id,
        persisted_blocks,
        writing_forward_block: seg.writing_forward_block.as_ref().map(to_pb_block_meta),
        indexed_vector_fields: seg.indexed_vector_fields.clone(),
        vector_index_generations: seg.vector_index_generations.clone(),
    }
}

fn to_pb_field_index(index: &FieldIndexParams) -> pb::IndexParams {
    let params = match index {
        FieldIndexParams::Hnsw(p) | FieldIndexParams::HnswSparse(p) => {
            let h = pb::HnswIndexParams {
                base: Some(to_pb_base(p.metric, p.quantize)),
                m: u32_to_i32_sat(p.m),
                ef_construction: u32_to_i32_sat(p.ef_construction),
                build_tuning: (p.build_tuning != HnswBuildTuning::default())
                    .then(|| to_pb_build_tuning(&p.build_tuning)),
            };
            Some(pb::index_params::Params::Hnsw(h))
        }
        FieldIndexParams::Ivf(p) => {
            let i = pb::IvfIndexParams {
                base: Some(to_pb_base(p.metric, p.quantize)),
                n_list: u32_to_i32_sat(p.n_list),
                n_iters: u32_to_i32_sat(p.n_iters),
                use_soar: p.use_soar,
            };
            Some(pb::index_params::Params::Ivf(i))
        }
        FieldIndexParams::Flat(p) | FieldIndexParams::FlatSparse(p) => {
            let f = pb::FlatIndexParams {
                base: Some(to_pb_base(p.metric, p.quantize)),
            };
            Some(pb::index_params::Params::Flat(f))
        }
        FieldIndexParams::Invert(p) => {
            let inv = pb::InvertIndexParams {
                enable_range_optimization: p.enable_range_optimization,
                enable_extended_wildcard: p.enable_extended_wildcard,
            };
            Some(pb::index_params::Params::Invert(inv))
        }
    };

    pb::IndexParams { params }
}

fn to_pb_build_tuning(t: &HnswBuildTuning) -> pb::HnswBuildTuning {
    pb::HnswBuildTuning {
        heuristic_dim: t.heuristic_dim.map(usize_to_u32_sat),
        simple_neighbor_select: t.simple_neighbor_select,
        keep_pruned_connections: t.keep_pruned_connections,
        l0_refine_candidate_cap: t.l0_refine_candidate_cap.map(usize_to_u32_sat),
        l0_repair: t.l0_repair,
        prune_alpha: t.prune_alpha,
        qdrant_backlink: t.qdrant_backlink,
    }
}

fn from_pb_build_tuning(t: pb::HnswBuildTuning) -> HnswBuildTuning {
    HnswBuildTuning {
        heuristic_dim: t.heuristic_dim.map(|v| v as usize),
        simple_neighbor_select: t.simple_neighbor_select,
        keep_pruned_connections: t.keep_pruned_connections,
        l0_refine_candidate_cap: t.l0_refine_candidate_cap.map(|v| v as usize),
        l0_repair: t.l0_repair,
        prune_alpha: t.prune_alpha,
        qdrant_backlink: t.qdrant_backlink,
    }
}

fn usize_to_u32_sat(v: usize) -> u32 {
    u32::try_from(v).unwrap_or(u32::MAX)
}

fn is_sparse_vector_data_type(dt: u32) -> bool {
    // Keep this local to avoid pulling finch-types into the manifest codec layer.
    // proto/manifest.proto: DT_SPARSE_VECTOR_FP16 = 30, DT_SPARSE_VECTOR_FP32 = 31
    matches!(dt, 30 | 31)
}

fn from_pb_field_index(index: pb::IndexParams, field_data_type: u32) -> Option<FieldIndexParams> {
    let params = index.params?;
    let is_sparse = is_sparse_vector_data_type(field_data_type);
    match params {
        pb::index_params::Params::Invert(p) => Some(FieldIndexParams::Invert(InvertIndexParams {
            enable_range_optimization: p.enable_range_optimization,
            enable_extended_wildcard: p.enable_extended_wildcard,
        })),
        pb::index_params::Params::Hnsw(p) => {
            let (metric, quantize) = from_pb_base(p.base);
            let m = i32_to_u32_nonneg(p.m);
            // Finch's HNSW uses `scaling_factor` as a derived parameter that
            // defaults to `m`. Protobuf manifests do not persist scaling_factor,
            // so on decode we must recompute the derived value from `m` to avoid
            // reopen-time mismatches.
            let scaling_factor = if m != 0 { m } else { 50 };
            let v = HnswIndexParams {
                m,
                ef_construction: i32_to_u32_nonneg(p.ef_construction),
                scaling_factor,
                metric,
                quantize,
                build_tuning: p.build_tuning.map(from_pb_build_tuning).unwrap_or_default(),
            };
            Some(if is_sparse {
                FieldIndexParams::HnswSparse(v)
            } else {
                FieldIndexParams::Hnsw(v)
            })
        }
        pb::index_params::Params::Flat(p) => {
            let (metric, quantize) = from_pb_base(p.base);
            let v = FlatIndexParams {
                metric,
                quantize,
                column_major: false,
            };
            Some(if is_sparse {
                FieldIndexParams::FlatSparse(v)
            } else {
                FieldIndexParams::Flat(v)
            })
        }
        pb::index_params::Params::Ivf(p) => {
            let (metric, quantize) = from_pb_base(p.base);
            Some(FieldIndexParams::Ivf(IvfIndexParams {
                n_list: i32_to_u32_nonneg(p.n_list),
                n_iters: i32_to_u32_nonneg(p.n_iters),
                use_soar: p.use_soar,
                metric,
                quantize,
            }))
        }
    }
}

fn to_pb_collection_schema(schema: &CollectionSchema) -> pb::CollectionSchema {
    pb::CollectionSchema {
        name: schema.name.clone(),
        fields: schema
            .fields
            .iter()
            .map(|f| pb::FieldSchema {
                name: f.name.clone(),
                data_type: u32_to_i32_sat(f.data_type),
                dimension: f.dimension,
                nullable: f.nullable,
                index_params: f.index_params.as_ref().map(to_pb_field_index),
            })
            .collect(),
        max_doc_count_per_segment: schema.max_doc_count_per_segment,
    }
}

fn from_pb_collection_schema(schema: pb::CollectionSchema) -> CollectionSchema {
    CollectionSchema {
        name: schema.name,
        fields: schema
            .fields
            .into_iter()
            .map(|f| {
                let dt = i32_to_u32_nonneg(f.data_type);
                FieldSchema {
                    name: f.name,
                    data_type: dt,
                    nullable: f.nullable,
                    dimension: f.dimension,
                    index_params: f.index_params.and_then(|ip| from_pb_field_index(ip, dt)),
                }
            })
            .collect(),
        max_doc_count_per_segment: schema.max_doc_count_per_segment,
    }
}

fn to_pb_manifest(m: &Manifest) -> pb::Manifest {
    let persisted_segment_metas = m
        .persisted_segment_metas
        .iter()
        .map(to_pb_segment_meta)
        .collect();

    pb::Manifest {
        version: m.version,
        schema: m.schema.as_ref().map(to_pb_collection_schema),
        enable_mmap: m.enable_mmap,
        persisted_segment_metas,
        writing_segment_meta: m.writing_segment_meta.as_ref().map(to_pb_segment_meta),
        id_map_path_suffix: m.id_map_path_suffix,
        delete_snapshot_path_suffix: m.delete_snapshot_path_suffix,
        next_segment_id: m.next_segment_id,
    }
}

fn from_pb_manifest(m: pb::Manifest) -> Manifest {
    let persisted_segment_metas = m
        .persisted_segment_metas
        .into_iter()
        .map(|seg| {
            let (min_doc_id, max_doc_id, doc_count) = segment_stats_from_pb(&seg);
            SegmentMeta {
                segment_id: seg.segment_id,
                persisted_blocks: seg
                    .persisted_blocks
                    .into_iter()
                    .map(from_pb_block_meta)
                    .collect(),
                writing_forward_block: seg.writing_forward_block.map(from_pb_block_meta),
                indexed_vector_fields: seg.indexed_vector_fields,
                vector_index_generations: seg.vector_index_generations,
                min_doc_id,
                max_doc_id,
                doc_count,
            }
        })
        .collect();

    let writing_segment_meta = m.writing_segment_meta.map(|seg| {
        let (min_doc_id, max_doc_id, doc_count) = segment_stats_from_pb(&seg);
        SegmentMeta {
            segment_id: seg.segment_id,
            persisted_blocks: seg
                .persisted_blocks
                .into_iter()
                .map(from_pb_block_meta)
                .collect(),
            writing_forward_block: seg.writing_forward_block.map(from_pb_block_meta),
            indexed_vector_fields: seg.indexed_vector_fields,
            vector_index_generations: seg.vector_index_generations,
            min_doc_id,
            max_doc_id,
            doc_count,
        }
    });

    Manifest {
        version: m.version,
        schema: m.schema.map(from_pb_collection_schema),
        enable_mmap: m.enable_mmap,
        persisted_segment_metas,
        writing_segment_meta,
        id_map_path_suffix: m.id_map_path_suffix,
        delete_snapshot_path_suffix: m.delete_snapshot_path_suffix,
        next_segment_id: m.next_segment_id,
    }
}

mod pb {
    use prost::Message;

    #[derive(Clone, PartialEq, Message)]
    pub struct Manifest {
        #[prost(uint32, tag = "1")]
        pub version: u32,
        #[prost(message, optional, tag = "2")]
        pub schema: Option<CollectionSchema>,
        #[prost(bool, tag = "3")]
        pub enable_mmap: bool,
        #[prost(message, repeated, tag = "4")]
        pub persisted_segment_metas: Vec<SegmentMeta>,
        #[prost(message, optional, tag = "5")]
        pub writing_segment_meta: Option<SegmentMeta>,
        #[prost(uint32, tag = "6")]
        pub id_map_path_suffix: u32,
        #[prost(uint32, tag = "7")]
        pub delete_snapshot_path_suffix: u32,
        #[prost(uint32, tag = "8")]
        pub next_segment_id: u32,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct SegmentMeta {
        #[prost(uint32, tag = "1")]
        pub segment_id: u32,
        #[prost(message, repeated, tag = "2")]
        pub persisted_blocks: Vec<BlockMeta>,
        #[prost(message, optional, tag = "3")]
        pub writing_forward_block: Option<BlockMeta>,
        #[prost(string, repeated, tag = "4")]
        pub indexed_vector_fields: Vec<String>,
        #[prost(btree_map = "string, uint32", tag = "5")]
        pub vector_index_generations: std::collections::BTreeMap<String, u32>,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct BlockMeta {
        #[prost(uint32, tag = "1")]
        pub block_id: u32,
        #[prost(int32, tag = "2")]
        pub block_type: i32,
        #[prost(uint64, tag = "3")]
        pub min_doc_id: u64,
        #[prost(uint64, tag = "4")]
        pub max_doc_id: u64,
        #[prost(uint64, tag = "5")]
        pub doc_count: u64,
        #[prost(string, repeated, tag = "6")]
        pub columns: Vec<String>,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct CollectionSchema {
        #[prost(string, tag = "1")]
        pub name: String,
        #[prost(message, repeated, tag = "2")]
        pub fields: Vec<FieldSchema>,
        #[prost(uint64, tag = "3")]
        pub max_doc_count_per_segment: u64,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct FieldSchema {
        #[prost(string, tag = "1")]
        pub name: String,
        #[prost(int32, tag = "2")]
        pub data_type: i32,
        #[prost(uint32, tag = "3")]
        pub dimension: u32,
        #[prost(bool, tag = "4")]
        pub nullable: bool,
        #[prost(message, optional, tag = "5")]
        pub index_params: Option<IndexParams>,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct InvertIndexParams {
        #[prost(bool, tag = "1")]
        pub enable_range_optimization: bool,
        #[prost(bool, tag = "2")]
        pub enable_extended_wildcard: bool,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct BaseIndexParams {
        #[prost(int32, tag = "1")]
        pub metric_type: i32,
        #[prost(int32, tag = "2")]
        pub quantize_type: i32,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct HnswIndexParams {
        #[prost(message, optional, tag = "1")]
        pub base: Option<BaseIndexParams>,
        #[prost(int32, tag = "2")]
        pub m: i32,
        #[prost(int32, tag = "3")]
        pub ef_construction: i32,
        #[prost(message, optional, tag = "4")]
        pub build_tuning: Option<HnswBuildTuning>,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct HnswBuildTuning {
        #[prost(uint32, optional, tag = "1")]
        pub heuristic_dim: Option<u32>,
        #[prost(bool, tag = "2")]
        pub simple_neighbor_select: bool,
        #[prost(bool, tag = "3")]
        pub keep_pruned_connections: bool,
        #[prost(uint32, optional, tag = "4")]
        pub l0_refine_candidate_cap: Option<u32>,
        #[prost(bool, tag = "5")]
        pub l0_repair: bool,
        #[prost(float, tag = "6")]
        pub prune_alpha: f32,
        #[prost(bool, tag = "7")]
        pub qdrant_backlink: bool,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct FlatIndexParams {
        #[prost(message, optional, tag = "1")]
        pub base: Option<BaseIndexParams>,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct IvfIndexParams {
        #[prost(message, optional, tag = "1")]
        pub base: Option<BaseIndexParams>,
        #[prost(int32, tag = "2")]
        pub n_list: i32,
        #[prost(int32, tag = "3")]
        pub n_iters: i32,
        #[prost(bool, tag = "4")]
        pub use_soar: bool,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct IndexParams {
        #[prost(oneof = "index_params::Params", tags = "1, 2, 3, 4")]
        pub params: Option<index_params::Params>,
    }

    pub mod index_params {
        use super::{FlatIndexParams, HnswIndexParams, InvertIndexParams, IvfIndexParams};
        use prost::Oneof;

        #[derive(Clone, PartialEq, Oneof)]
        pub enum Params {
            #[prost(message, tag = "1")]
            Invert(InvertIndexParams),
            #[prost(message, tag = "2")]
            Hnsw(HnswIndexParams),
            #[prost(message, tag = "3")]
            Flat(FlatIndexParams),
            #[prost(message, tag = "4")]
            Ivf(IvfIndexParams),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_protobuf_uses_segment_stats_semantics() {
        let pb_manifest = pb::Manifest {
            version: 1,
            schema: None,
            enable_mmap: false,
            persisted_segment_metas: vec![pb::SegmentMeta {
                segment_id: 11,
                persisted_blocks: vec![
                    pb::BlockMeta {
                        block_id: 1,
                        block_type: BT_SCALAR,
                        min_doc_id: 100,
                        max_doc_id: 199,
                        doc_count: 100,
                        columns: vec![],
                    },
                    pb::BlockMeta {
                        block_id: 2,
                        block_type: 3,
                        min_doc_id: 50,
                        max_doc_id: 300,
                        doc_count: 9999,
                        columns: vec![],
                    },
                ],
                writing_forward_block: Some(pb::BlockMeta {
                    block_id: 3,
                    block_type: BT_SCALAR,
                    min_doc_id: 200,
                    max_doc_id: 249,
                    doc_count: 50,
                    columns: vec![],
                }),
                indexed_vector_fields: vec![],
                vector_index_generations: Default::default(),
            }],
            writing_segment_meta: None,
            id_map_path_suffix: 0,
            delete_snapshot_path_suffix: 0,
            next_segment_id: 0,
        };

        let data = pb_manifest.encode_to_vec();
        let m = Manifest::decode(&data).expect("protobuf decode should work");
        let seg = &m.persisted_segment_metas[0];

        assert_eq!(seg.min_doc_id, 100);
        assert_eq!(seg.max_doc_id, 249);
        assert_eq!(seg.doc_count, 150);
    }

    #[test]
    fn protobuf_encode_preserves_stats_for_writing_segment() {
        let manifest = Manifest {
            version: 1,
            schema: None,
            enable_mmap: false,
            persisted_segment_metas: vec![],
            writing_segment_meta: Some(SegmentMeta {
                segment_id: 9,
                persisted_blocks: vec![],
                writing_forward_block: None,
                indexed_vector_fields: vec![],
                vector_index_generations: Default::default(),
                min_doc_id: 1000,
                max_doc_id: 1099,
                doc_count: 100,
            }),
            id_map_path_suffix: 0,
            delete_snapshot_path_suffix: 0,
            next_segment_id: 10,
        };

        let data = manifest.encode().expect("protobuf encode should work");
        let decoded = Manifest::decode(&data).expect("protobuf decode should work");
        let seg = decoded
            .writing_segment_meta
            .expect("writing segment meta should exist");

        assert_eq!(seg.segment_id, 9);
        assert_eq!(seg.min_doc_id, 1000);
        assert_eq!(seg.max_doc_id, 1099);
        assert_eq!(seg.doc_count, 100);
    }

    #[test]
    fn protobuf_encode_injects_scalar_block_when_persisted_stats_present_even_if_min_doc_id_is_zero(
    ) {
        let manifest = Manifest {
            version: 1,
            schema: None,
            enable_mmap: false,
            persisted_segment_metas: vec![SegmentMeta {
                segment_id: 1,
                persisted_blocks: vec![],
                writing_forward_block: None,
                indexed_vector_fields: vec![],
                vector_index_generations: Default::default(),
                min_doc_id: 0,
                max_doc_id: 0,
                doc_count: 1,
            }],
            writing_segment_meta: None,
            id_map_path_suffix: 0,
            delete_snapshot_path_suffix: 0,
            next_segment_id: 2,
        };

        let data = manifest.encode().expect("protobuf encode should work");
        let decoded = Manifest::decode(&data).expect("protobuf decode should work");
        let seg = &decoded.persisted_segment_metas[0];
        assert_eq!(seg.min_doc_id, 0);
        assert_eq!(seg.max_doc_id, 0);
        assert_eq!(seg.doc_count, 1);
    }

    #[test]
    fn protobuf_decode_recomputes_hnsw_scaling_factor_from_m() {
        let manifest = Manifest {
            version: 1,
            schema: Some(CollectionSchema {
                name: "test".to_string(),
                max_doc_count_per_segment: 1000,
                fields: vec![FieldSchema {
                    name: "emb".to_string(),
                    // proto/manifest.proto DataType: VECTOR_FP32 = 23
                    data_type: 23,
                    nullable: false,
                    dimension: 4,
                    index_params: Some(FieldIndexParams::Hnsw(HnswIndexParams {
                        m: 16,
                        ef_construction: 123,
                        // Intentionally set to a non-default value; protobuf doesn't persist it.
                        scaling_factor: 999,
                        // proto/manifest.proto MetricType: MT_IP = 2
                        metric: 2,
                        // proto/manifest.proto QuantizeType: QT_UNDEFINED = 0
                        quantize: 0,
                        build_tuning: HnswBuildTuning::default(),
                    })),
                }],
            }),
            enable_mmap: false,
            persisted_segment_metas: vec![],
            writing_segment_meta: None,
            id_map_path_suffix: 0,
            delete_snapshot_path_suffix: 0,
            next_segment_id: 2,
        };

        let data = manifest.encode().expect("protobuf encode should work");
        let decoded = Manifest::decode(&data).expect("protobuf decode should work");
        let schema = decoded.schema.expect("schema should exist");
        let field = schema
            .fields
            .into_iter()
            .next()
            .expect("field should exist");
        let Some(FieldIndexParams::Hnsw(p)) = field.index_params else {
            panic!("expected HNSW index params");
        };
        assert_eq!(p.m, 16);
        assert_eq!(p.scaling_factor, 16);
    }

    fn hnsw_manifest(build_tuning: HnswBuildTuning) -> Manifest {
        Manifest {
            version: 1,
            schema: Some(CollectionSchema {
                name: "test".to_string(),
                max_doc_count_per_segment: 1000,
                fields: vec![FieldSchema {
                    name: "emb".to_string(),
                    data_type: 23,
                    nullable: false,
                    dimension: 4,
                    index_params: Some(FieldIndexParams::Hnsw(HnswIndexParams {
                        m: 16,
                        ef_construction: 123,
                        scaling_factor: 16,
                        metric: 1,
                        quantize: 0,
                        build_tuning,
                    })),
                }],
            }),
            enable_mmap: false,
            persisted_segment_metas: vec![],
            writing_segment_meta: None,
            id_map_path_suffix: 0,
            delete_snapshot_path_suffix: 0,
            next_segment_id: 2,
        }
    }

    fn decoded_build_tuning(manifest: &Manifest) -> HnswBuildTuning {
        let data = manifest.encode().expect("protobuf encode should work");
        let decoded = Manifest::decode(&data).expect("protobuf decode should work");
        let field = decoded.schema.expect("schema").fields.remove(0);
        let Some(FieldIndexParams::Hnsw(p)) = field.index_params else {
            panic!("expected HNSW index params");
        };
        p.build_tuning
    }

    #[test]
    fn protobuf_round_trips_hnsw_build_tuning() {
        let tuning = HnswBuildTuning {
            heuristic_dim: Some(64),
            simple_neighbor_select: true,
            keep_pruned_connections: false,
            l0_refine_candidate_cap: Some(256),
            l0_repair: false,
            prune_alpha: 1.2,
            qdrant_backlink: true,
        };
        assert_eq!(decoded_build_tuning(&hnsw_manifest(tuning)), tuning);
        let default = HnswBuildTuning::default();
        assert_eq!(decoded_build_tuning(&hnsw_manifest(default)), default);
    }
}

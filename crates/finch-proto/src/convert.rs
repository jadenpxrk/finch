//! Conversions between finch-types and manifest types

use crate::manifest::{
    CollectionSchema as MSchema, FieldIndexParams, FieldSchema as MField, FlatIndexParams as MFlat,
    HnswIndexParams as MHnsw, InvertIndexParams as MInvert, IvfIndexParams as MIvf,
};
use finch_types::{
    CollectionSchema as TypesSchema, DataType, FieldSchema as TypesField,
    FlatIndexParams as TypesFlat, HnswIndexParams as TypesHnsw, IndexParams,
    InvertIndexParams as TypesInvert, IvfIndexParams as TypesIvf, MetricType, QuantizeType, Status,
    ZResult,
};

// ── CollectionSchema ──────────────────────────────────────────────────────────

impl From<TypesSchema> for MSchema {
    fn from(s: TypesSchema) -> Self {
        MSchema {
            name: s.name,
            fields: s.fields.into_iter().map(Into::into).collect(),
            max_doc_count_per_segment: s.max_doc_count_per_segment,
        }
    }
}

impl TryFrom<MSchema> for TypesSchema {
    type Error = Status;
    fn try_from(m: MSchema) -> ZResult<Self> {
        Ok(TypesSchema {
            name: m.name,
            fields: m
                .fields
                .into_iter()
                .map(TypesField::try_from)
                .collect::<ZResult<_>>()?,
            max_doc_count_per_segment: m.max_doc_count_per_segment,
        })
    }
}

// ── FieldSchema ───────────────────────────────────────────────────────────────

impl From<TypesField> for MField {
    fn from(f: TypesField) -> Self {
        let index_params = f.index_params.map(|ip| match ip {
            IndexParams::Hnsw(p) => FieldIndexParams::Hnsw(p.into()),
            IndexParams::HnswSparse(p) => FieldIndexParams::HnswSparse(p.into()),
            IndexParams::Ivf(p) => FieldIndexParams::Ivf(p.into()),
            IndexParams::Flat(p) => FieldIndexParams::Flat(p.into()),
            IndexParams::FlatSparse(p) => FieldIndexParams::FlatSparse(p.into()),
            IndexParams::Invert(p) => FieldIndexParams::Invert(p.into()),
        });
        MField {
            name: f.name,
            data_type: f.data_type as u32,
            nullable: f.nullable,
            dimension: f.dimension.unwrap_or(0) as u32,
            index_params,
        }
    }
}

impl TryFrom<MField> for TypesField {
    type Error = Status;
    fn try_from(m: MField) -> ZResult<Self> {
        let data_type = decode_data_type(m.data_type)?;
        let index_params = m.index_params.map(|ip| match ip {
            FieldIndexParams::Hnsw(p) => IndexParams::Hnsw(p.into()),
            FieldIndexParams::HnswSparse(p) => IndexParams::HnswSparse(p.into()),
            FieldIndexParams::Ivf(p) => IndexParams::Ivf(p.into()),
            FieldIndexParams::Flat(p) => IndexParams::Flat(p.into()),
            FieldIndexParams::FlatSparse(p) => IndexParams::FlatSparse(p.into()),
            FieldIndexParams::Invert(p) => IndexParams::Invert(p.into()),
        });
        Ok(TypesField {
            name: m.name,
            data_type,
            nullable: m.nullable,
            dimension: if m.dimension == 0 {
                None
            } else {
                Some(m.dimension as usize)
            },
            index_params,
        })
    }
}

// ── IndexParams conversions ───────────────────────────────────────────────────

impl From<TypesHnsw> for MHnsw {
    fn from(p: TypesHnsw) -> Self {
        MHnsw {
            m: p.m as u32,
            ef_construction: p.ef_construction as u32,
            scaling_factor: p.scaling_factor as u32,
            metric: p.metric as u32,
            quantize: p.quantize as u32,
        }
    }
}
impl From<MHnsw> for TypesHnsw {
    fn from(p: MHnsw) -> Self {
        TypesHnsw {
            m: p.m as usize,
            ef_construction: p.ef_construction as usize,
            scaling_factor: p.scaling_factor as usize,
            metric: decode_metric(p.metric),
            quantize: decode_quantize(p.quantize),
            build_concurrency: None,
        }
    }
}

impl From<TypesIvf> for MIvf {
    fn from(p: TypesIvf) -> Self {
        MIvf {
            n_list: p.n_list as u32,
            n_iters: p.n_iters as u32,
            use_soar: p.use_soar,
            metric: p.metric as u32,
            quantize: p.quantize as u32,
        }
    }
}
impl From<MIvf> for TypesIvf {
    fn from(p: MIvf) -> Self {
        TypesIvf {
            n_list: p.n_list as usize,
            n_iters: p.n_iters as usize,
            use_soar: p.use_soar,
            l1_index: None,
            metric: decode_metric(p.metric),
            quantize: decode_quantize(p.quantize),
        }
    }
}

impl From<TypesFlat> for MFlat {
    fn from(p: TypesFlat) -> Self {
        MFlat {
            metric: p.metric as u32,
            quantize: p.quantize as u32,
            column_major: p.column_major,
        }
    }
}
impl From<MFlat> for TypesFlat {
    fn from(p: MFlat) -> Self {
        TypesFlat {
            metric: decode_metric(p.metric),
            quantize: decode_quantize(p.quantize),
            column_major: p.column_major,
        }
    }
}

impl From<TypesInvert> for MInvert {
    fn from(p: TypesInvert) -> Self {
        MInvert {
            enable_range_optimization: p.enable_range_optimization,
            enable_extended_wildcard: p.enable_extended_wildcard,
        }
    }
}
impl From<MInvert> for TypesInvert {
    fn from(p: MInvert) -> Self {
        TypesInvert {
            enable_range_optimization: p.enable_range_optimization,
            enable_extended_wildcard: p.enable_extended_wildcard,
        }
    }
}

// ── Enum decoders ─────────────────────────────────────────────────────────────

fn decode_data_type(v: u32) -> ZResult<DataType> {
    match v {
        // Canonical ids
        0 => Ok(DataType::Undefined),
        1 => Ok(DataType::Binary),
        2 => Ok(DataType::String),
        3 => Ok(DataType::Bool),
        4 => Ok(DataType::Int32),
        5 => Ok(DataType::Int64),
        6 => Ok(DataType::Uint32),
        7 => Ok(DataType::Uint64),
        8 => Ok(DataType::Float32),
        9 => Ok(DataType::Float64),
        20 => Ok(DataType::VectorBinary32),
        21 => Ok(DataType::VectorBinary64),
        22 => Ok(DataType::VectorFp16),
        23 => Ok(DataType::VectorFp32),
        24 => Ok(DataType::VectorFp64),
        25 => Ok(DataType::VectorInt4),
        26 => Ok(DataType::VectorInt8),
        27 => Ok(DataType::VectorInt16),
        30 => Ok(DataType::SparseFp16),
        31 => Ok(DataType::SparseFp32),
        40 => Ok(DataType::ArrayBinary),
        41 => Ok(DataType::ArrayString),
        42 => Ok(DataType::ArrayBool),
        43 => Ok(DataType::ArrayInt32),
        44 => Ok(DataType::ArrayInt64),
        45 => Ok(DataType::ArrayUint32),
        46 => Ok(DataType::ArrayUint64),
        47 => Ok(DataType::ArrayFp32),
        48 => Ok(DataType::ArrayFp64),
        // Legacy finch ids (accepted for backward compatibility)
        10 => Ok(DataType::Float16),
        11 => Ok(DataType::Float32),
        12 => Ok(DataType::Float64),
        13 => Ok(DataType::String),
        14 => Ok(DataType::Bytes),
        28 => Ok(DataType::VectorFp32),
        29 => Ok(DataType::VectorFp64),
        _ => Err(Status::invalid_argument(format!("unknown DataType: {v}"))),
    }
}

fn decode_metric(v: u32) -> MetricType {
    match v {
        1 => MetricType::L2,
        2 => MetricType::InnerProduct,
        3 => MetricType::Cosine,
        4 => MetricType::MipsL2,
        5 => MetricType::Hamming,
        _ => MetricType::Undefined,
    }
}

fn decode_quantize(v: u32) -> QuantizeType {
    match v {
        1 => QuantizeType::Fp16,
        2 => QuantizeType::Int8,
        3 => QuantizeType::Int4,
        _ => QuantizeType::Undefined,
    }
}

use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi(object)]
pub struct HnswIndexParams {
    pub metric: i32,
    pub m: u32,
    pub ef_construction: u32,
    pub quantize: i32,
}

#[napi(object)]
pub struct IvfIndexParams {
    pub metric: i32,
    pub n_list: u32,
    pub n_iters: u32,
    pub use_soar: Option<bool>,
    pub quantize: i32,
}

#[napi(object)]
pub struct FlatIndexParams {
    pub metric: i32,
    pub quantize: i32,
}

#[napi(object)]
pub struct InvertIndexParams {
    pub enable_range_optimization: Option<bool>,
    pub enable_extended_wildcard: Option<bool>,
}

#[napi(object)]
pub struct FieldSchemaOptions {
    pub name: String,
    pub data_type: i32,
    pub nullable: Option<bool>,
    pub dimension: Option<u32>,
    pub hnsw_index: Option<HnswIndexParams>,
    pub ivf_index: Option<IvfIndexParams>,
    pub flat_index: Option<FlatIndexParams>,
    pub invert_index: Option<InvertIndexParams>,
}

impl TryFrom<FieldSchemaOptions> for finch_types::FieldSchema {
    type Error = napi::Error;

    fn try_from(opts: FieldSchemaOptions) -> Result<Self> {
        let data_type = decode_data_type(opts.data_type)?;
        let mut schema = finch_types::FieldSchema::new(&opts.name, data_type);
        // reference parity: default nullable is false.
        schema.nullable = opts.nullable.unwrap_or(false);
        schema.dimension = opts.dimension.map(|d| d as usize);

        if let Some(hnsw) = opts.hnsw_index {
            schema.index_params = Some(finch_types::IndexParams::Hnsw(
                finch_types::HnswIndexParams {
                    m: hnsw.m as usize,
                    ef_construction: hnsw.ef_construction as usize,
                    scaling_factor: hnsw.m as usize,
                    metric: decode_metric(hnsw.metric)?,
                    quantize: decode_quantize(hnsw.quantize)?,
                    build_concurrency: None,
                    build_tuning: Default::default(),
                },
            ));
        } else if let Some(ivf) = opts.ivf_index {
            schema.index_params =
                Some(finch_types::IndexParams::Ivf(finch_types::IvfIndexParams {
                    n_list: ivf.n_list as usize,
                    n_iters: ivf.n_iters as usize,
                    use_soar: ivf.use_soar.unwrap_or(false),
                    l1_index: None,
                    metric: decode_metric(ivf.metric)?,
                    quantize: decode_quantize(ivf.quantize)?,
                }));
        } else if let Some(flat) = opts.flat_index {
            schema.index_params = Some(finch_types::IndexParams::Flat(
                finch_types::FlatIndexParams {
                    metric: decode_metric(flat.metric)?,
                    quantize: decode_quantize(flat.quantize)?,
                    column_major: false,
                },
            ));
        } else if let Some(invert) = opts.invert_index {
            schema.index_params = Some(finch_types::IndexParams::Invert(
                finch_types::InvertIndexParams {
                    // reference parity: default enable_range_optimization is false.
                    enable_range_optimization: invert.enable_range_optimization.unwrap_or(false),
                    enable_extended_wildcard: invert.enable_extended_wildcard.unwrap_or(false),
                },
            ));
        }

        Ok(schema)
    }
}

#[napi(object)]
pub struct CollectionSchemaOptions {
    pub name: String,
    pub fields: Vec<FieldSchemaOptions>,
    pub max_doc_count_per_segment: Option<i64>,
}

impl TryFrom<CollectionSchemaOptions> for finch_types::CollectionSchema {
    type Error = napi::Error;

    fn try_from(opts: CollectionSchemaOptions) -> Result<Self> {
        let fields = opts
            .fields
            .into_iter()
            .map(finch_types::FieldSchema::try_from)
            .collect::<Result<Vec<_>>>()?;
        Ok(finch_types::CollectionSchema {
            name: opts.name,
            fields,
            max_doc_count_per_segment: opts
                .max_doc_count_per_segment
                .unwrap_or(finch_types::MAX_DOC_COUNT_PER_SEGMENT as i64)
                as u64,
        })
    }
}

fn decode_data_type(v: i32) -> Result<finch_types::DataType> {
    Ok(match v {
        0 => finch_types::DataType::Undefined,
        1 => finch_types::DataType::Binary,
        2 => finch_types::DataType::String,
        3 => finch_types::DataType::Bool,
        4 => finch_types::DataType::Int32,
        5 => finch_types::DataType::Int64,
        6 => finch_types::DataType::Uint32,
        7 => finch_types::DataType::Uint64,
        8 => finch_types::DataType::Float32,
        9 => finch_types::DataType::Float64,
        100 => finch_types::DataType::Int8,
        101 => finch_types::DataType::Int16,
        102 => finch_types::DataType::Uint8,
        103 => finch_types::DataType::Uint16,
        104 => finch_types::DataType::Float16,
        105 => finch_types::DataType::Bytes,
        20 => finch_types::DataType::VectorBinary32,
        21 => finch_types::DataType::VectorBinary64,
        22 => finch_types::DataType::VectorFp16,
        23 => finch_types::DataType::VectorFp32,
        24 => finch_types::DataType::VectorFp64,
        25 => finch_types::DataType::VectorInt4,
        26 => finch_types::DataType::VectorInt8,
        27 => finch_types::DataType::VectorInt16,
        120 => finch_types::DataType::VectorBool,
        121 => finch_types::DataType::VectorInt32,
        122 => finch_types::DataType::VectorInt64,
        123 => finch_types::DataType::VectorUint32,
        124 => finch_types::DataType::VectorUint64,
        30 => finch_types::DataType::SparseFp16,
        31 => finch_types::DataType::SparseFp32,
        40 => finch_types::DataType::ArrayBinary,
        41 => finch_types::DataType::ArrayString,
        42 => finch_types::DataType::ArrayBool,
        43 => finch_types::DataType::ArrayInt32,
        44 => finch_types::DataType::ArrayInt64,
        45 => finch_types::DataType::ArrayUint32,
        46 => finch_types::DataType::ArrayUint64,
        47 => finch_types::DataType::ArrayFp32,
        48 => finch_types::DataType::ArrayFp64,
        // Legacy finch ids (accepted for backward compatibility)
        11 => finch_types::DataType::Float32,
        12 => finch_types::DataType::Float64,
        13 => finch_types::DataType::String,
        28 => finch_types::DataType::VectorFp32,
        _ => return Err(napi::Error::from_reason(format!("unknown DataType: {}", v))),
    })
}

fn decode_metric(v: i32) -> Result<finch_types::MetricType> {
    Ok(match v {
        0 => finch_types::MetricType::Undefined,
        1 => finch_types::MetricType::L2,
        2 => finch_types::MetricType::InnerProduct,
        3 => finch_types::MetricType::Cosine,
        4 => finch_types::MetricType::MipsL2,
        5 => finch_types::MetricType::Hamming,
        _ => {
            return Err(napi::Error::from_reason(format!(
                "unknown MetricType: {}",
                v
            )))
        }
    })
}

fn decode_quantize(v: i32) -> Result<finch_types::QuantizeType> {
    Ok(match v {
        0 => finch_types::QuantizeType::Undefined,
        1 => finch_types::QuantizeType::Fp16,
        2 => finch_types::QuantizeType::Int8,
        3 => finch_types::QuantizeType::Int4,
        _ => {
            return Err(napi::Error::from_reason(format!(
                "unknown QuantizeType: {}",
                v
            )))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_data_type_accepts_all_public_ids() {
        let ids = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 101, 102, 103, 104, 105, 20, 21, 22, 23, 24, 25, 26,
            27, 120, 121, 122, 123, 124, 30, 31, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        ];
        for id in ids {
            decode_data_type(id).unwrap();
        }
    }

    #[test]
    fn test_hnsw_scaling_factor_is_derived_from_m() {
        let field = FieldSchemaOptions {
            name: "emb".to_string(),
            data_type: 28,
            nullable: None,
            dimension: Some(4),
            hnsw_index: Some(HnswIndexParams {
                metric: 1,
                m: 16,
                ef_construction: 64,
                quantize: 0,
            }),
            ivf_index: None,
            flat_index: None,
            invert_index: None,
        };

        let field = finch_types::FieldSchema::try_from(field).unwrap();
        let Some(finch_types::IndexParams::Hnsw(params)) = field.index_params else {
            panic!("expected HNSW index params");
        };
        assert_eq!(params.scaling_factor, params.m);
    }
}

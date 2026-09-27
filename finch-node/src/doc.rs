use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::collections::HashMap;

mod typed_value;

use typed_value::json_to_typed_value;

#[napi(object)]
pub struct DocObject {
    pub pk: String,
    pub score: Option<f64>,
    pub fields: HashMap<String, serde_json::Value>,
}

pub(crate) fn docobject_to_doc(
    obj: DocObject,
    schema: &finch_types::CollectionSchema,
) -> Result<finch_types::Doc> {
    let mut doc = finch_types::Doc::new(obj.pk);
    doc.score = obj.score.unwrap_or(0.0) as f32;
    for (k, v) in obj.fields {
        let Some(fs) = schema.get_field(&k) else {
            return Err(napi::Error::from_reason(format!(
                "doc validate failed: field[{k}] does not exist in collection's schema"
            )));
        };
        let val = json_to_typed_value(v, fs.data_type)?;
        doc.fields.insert(k, val);
    }
    Ok(doc)
}

impl From<finch_types::Doc> for DocObject {
    fn from(doc: finch_types::Doc) -> Self {
        let fields = doc
            .fields
            .into_iter()
            .map(|(k, v)| (k, value_to_json(v)))
            .collect();
        DocObject {
            pk: doc.pk,
            score: Some(doc.score as f64),
            fields,
        }
    }
}

#[napi(object)]
pub struct VectorQueryOptions {
    pub field_name: String,
    pub query_vector: Vec<f64>,
    /// Binary32 query vector (Hamming); takes precedence over `query_vector` when non-empty.
    pub query_vector_u32: Option<Vec<u32>>,
    /// Binary64 query vector (Hamming), encoded as decimal strings to avoid JS number precision loss.
    /// Takes precedence over `query_vector` when non-empty.
    pub query_vector_u64: Option<Vec<String>>,
    pub topk: u32,
    pub filter: Option<String>,
    pub ef: Option<u32>,
    pub n_probe: Option<u32>,
    pub concurrency: Option<u32>,
    pub bf_pks: Option<Vec<String>>,
    pub radius: Option<f64>,
    pub is_linear: Option<bool>,
    pub use_refiner: Option<bool>,
    pub refiner_k: Option<u32>,
    pub refiner_scale_factor: Option<f64>,
    /// Sparse query indices (for sparse vector fields)
    pub sparse_indices: Option<Vec<u32>>,
    /// Sparse query values (for sparse vector fields)
    pub sparse_values: Option<Vec<f64>>,
    /// Include the stored vector in result docs
    pub include_vector: Option<bool>,
    /// Include internal doc_id in result docs
    pub include_doc_id: Option<bool>,
    /// Whitelist of field names to return:
    /// - `undefined` => all fields
    /// - `[]` => no fields
    /// - `["a", ...]` => selected fields
    pub output_fields: Option<Vec<String>>,
}

impl TryFrom<VectorQueryOptions> for finch_types::VectorQuery {
    type Error = napi::Error;

    fn try_from(opts: VectorQueryOptions) -> Result<Self> {
        let mut query_vector: Vec<f32> = opts.query_vector.into_iter().map(|x| x as f32).collect();
        let mut query_vector_u32: Vec<u32> = Vec::new();
        let mut query_vector_u64: Vec<u64> = Vec::new();

        if let Some(v) = opts.query_vector_u64.as_ref().filter(|v| !v.is_empty()) {
            query_vector.clear();
            query_vector_u64 = v
                .iter()
                .map(|s| s.parse::<u64>())
                .collect::<std::result::Result<Vec<u64>, _>>()
                .map_err(|_| {
                    napi::Error::from_reason("queryVectorU64 contains an invalid u64 string")
                })?;
        } else if let Some(v) = opts.query_vector_u32.filter(|v| !v.is_empty()) {
            query_vector.clear();
            query_vector_u32 = v;
        }

        Ok(finch_types::VectorQuery {
            topk: opts.topk as usize,
            field_name: opts.field_name,
            id: None,
            query_vector,
            query_vector_u32,
            query_vector_u64,
            sparse_indices: opts.sparse_indices.unwrap_or_default(),
            sparse_values: opts
                .sparse_values
                .unwrap_or_default()
                .into_iter()
                .map(|x| x as f32)
                .collect(),
            filter: opts.filter,
            include_vector: opts.include_vector.unwrap_or(false),
            include_doc_id: opts.include_doc_id.unwrap_or(false),
            output_fields: opts.output_fields,
            query_params: finch_types::QueryParams {
                ef: opts.ef,
                n_probe: opts.n_probe,
                concurrency: opts.concurrency.map(|n| n as usize),
                bf_pks: opts.bf_pks,
                radius: opts.radius.map(|x| x as f32),
                is_linear: opts.is_linear,
                use_refiner: opts.use_refiner.unwrap_or(false),
                refiner_k: opts.refiner_k,
                refiner_scale_factor: opts.refiner_scale_factor.map(|x| x as f32),
            },
        })
    }
}

#[napi(object)]
pub struct GroupResultObject {
    pub group_value: serde_json::Value,
    pub docs: Vec<DocObject>,
}

impl From<finch_types::GroupResult> for GroupResultObject {
    fn from(r: finch_types::GroupResult) -> Self {
        GroupResultObject {
            group_value: value_to_json(r.group_value),
            docs: r.docs.into_iter().map(DocObject::from).collect(),
        }
    }
}

fn value_to_json(v: finch_types::Value) -> serde_json::Value {
    use finch_types::Value as V;
    match v {
        V::Null => serde_json::Value::Null,
        V::Bool(b) => serde_json::Value::Bool(b),
        V::I8(x) => serde_json::json!(x),
        V::I16(x) => serde_json::json!(x),
        V::I32(x) => serde_json::json!(x),
        V::I64(x) => i64_to_json(x),
        V::U8(x) => serde_json::json!(x),
        V::U16(x) => serde_json::json!(x),
        V::U32(x) => serde_json::json!(x),
        // JS cannot represent all u64 integers exactly; encode as decimal string.
        V::U64(x) => serde_json::Value::String(x.to_string()),
        V::F16(x) => serde_json::json!(x.to_f32()),
        V::F32(x) => serde_json::json!(x),
        V::F64(x) => serde_json::json!(x),
        V::String(s) => serde_json::Value::String(s),
        // Encode bytes as a JSON number array (u8[]). This is lossless and
        // does not force a base64 convention onto users.
        V::Bytes(b) => serde_json::json!(b),

        // Dense vector types
        V::VecBool(v) => serde_json::json!(v),
        V::VecI8(v) => serde_json::json!(v),
        V::VecI16(v) => serde_json::json!(v),
        V::VecI32(v) => serde_json::json!(v),
        V::VecI64(v) => v.into_iter().map(i64_to_json).collect(),
        V::VecU32(v) => serde_json::json!(v),
        // u64 vectors are encoded as decimal strings to avoid JS precision loss.
        V::VecU64(v) => serde_json::json!(v.into_iter().map(|x| x.to_string()).collect::<Vec<_>>()),
        V::VecF16(v) => serde_json::json!(v.into_iter().map(|x| x.to_f32()).collect::<Vec<_>>()),
        V::VecF32(v) => serde_json::json!(v),
        V::VecF64(v) => serde_json::json!(v),
        V::VecString(v) => serde_json::json!(v),

        // Sparse vector types
        V::SparseF16 { indices, values } => {
            serde_json::json!({
                "indices": indices,
                "values": values.into_iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
            })
        }
        V::SparseF32 { indices, values } => {
            serde_json::json!({
                "indices": indices,
                "values": values,
            })
        }

        // Array types
        V::ArrayBinary(v) => serde_json::json!(v),
        V::ArrayI32(v) => serde_json::json!(v),
        V::ArrayI64(v) => v.into_iter().map(i64_to_json).collect(),
        V::ArrayU32(v) => serde_json::json!(v),
        // u64 arrays are encoded as decimal strings to avoid JS precision loss.
        V::ArrayU64(v) => {
            serde_json::json!(v.into_iter().map(|x| x.to_string()).collect::<Vec<_>>())
        }
        V::ArrayBool(v) => serde_json::json!(v),
        V::ArrayF32(v) => serde_json::json!(v),
        V::ArrayF64(v) => serde_json::json!(v),
        V::ArrayString(v) => serde_json::json!(v),
    }
}

// JS numbers are exact only up to 2^53 - 1, so larger magnitudes go out as decimal strings like u64.
fn i64_to_json(x: i64) -> serde_json::Value {
    const MAX_SAFE_INTEGER: i64 = (1 << 53) - 1;
    if (-MAX_SAFE_INTEGER..=MAX_SAFE_INTEGER).contains(&x) {
        serde_json::json!(x)
    } else {
        serde_json::Value::String(x.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_docobject_preserves_system_u64_columns_as_strings() {
        let mut fields = HashMap::new();
        fields.insert("_finch_g_doc_id_".to_string(), finch_types::Value::U64(42));
        fields.insert("_finch_row_id_".to_string(), finch_types::Value::U64(7));
        fields.insert("_finch_score".to_string(), finch_types::Value::F32(1.25));
        let doc = finch_types::Doc {
            pk: "d0".to_string(),
            score: 1.25,
            doc_id: 42,
            op: finch_types::Operator::Insert,
            fields,
        };

        let out = DocObject::from(doc);
        assert_eq!(out.fields.get("_finch_g_doc_id_").unwrap(), "42");
        assert_eq!(out.fields.get("_finch_row_id_").unwrap(), "7");
        assert_eq!(out.fields.get("_finch_score").unwrap(), 1.25);
    }

    #[test]
    fn test_docobject_to_doc_converts_uint64_from_decimal_string() {
        let schema = finch_types::CollectionSchema::new("test")
            .with_field(finch_types::FieldSchema::new(
                "id",
                finch_types::DataType::Uint64,
            ))
            .with_field(
                finch_types::FieldSchema::new("emb", finch_types::DataType::VectorFp32)
                    .with_dimension(2),
            );

        let mut fields = HashMap::new();
        fields.insert(
            "id".to_string(),
            serde_json::Value::String("18446744073709551615".to_string()),
        );
        fields.insert("emb".to_string(), serde_json::json!([1.0, 2.0]));
        let obj = DocObject {
            pk: "d0".to_string(),
            score: None,
            fields,
        };

        let doc = docobject_to_doc(obj, &schema).expect("doc conversion");
        assert!(matches!(doc.fields.get("id"), Some(finch_types::Value::U64(x)) if *x == u64::MAX));
        assert!(
            matches!(doc.fields.get("emb"), Some(finch_types::Value::VecF32(v)) if v == &vec![1.0, 2.0])
        );
    }
    #[test]
    fn test_int64_beyond_2_pow_53_round_trips_exactly() {
        let schema = finch_types::CollectionSchema::new("test")
            .with_field(finch_types::FieldSchema::new(
                "id",
                finch_types::DataType::Int64,
            ))
            .with_field(finch_types::FieldSchema::new(
                "ids",
                finch_types::DataType::ArrayInt64,
            ));
        let big = "9007199254740993";
        let fields = HashMap::from([
            ("id".to_string(), serde_json::json!(big)),
            ("ids".to_string(), serde_json::json!([big, 7])),
        ]);
        let obj = DocObject {
            pk: "d0".to_string(),
            score: None,
            fields,
        };

        let out = DocObject::from(docobject_to_doc(obj, &schema).expect("doc conversion"));
        assert_eq!(out.fields.get("id").unwrap(), big);
        assert_eq!(out.fields.get("ids").unwrap(), &serde_json::json!([big, 7]));
    }

    #[test]
    fn test_large_js_number_is_rejected_for_64_bit_integer_fields() {
        // napi hands integers outside the i32/u32 range to Rust as f64-backed JSON numbers.
        let schema = finch_types::CollectionSchema::new("test")
            .with_field(finch_types::FieldSchema::new(
                "id",
                finch_types::DataType::Int64,
            ))
            .with_field(finch_types::FieldSchema::new(
                "uid",
                finch_types::DataType::Uint64,
            ));
        let rounded = serde_json::Number::from_f64(9_007_199_254_740_992.0).unwrap();
        for field in ["id", "uid"] {
            let obj = DocObject {
                pk: "d0".to_string(),
                score: None,
                fields: HashMap::from([(
                    field.to_string(),
                    serde_json::Value::Number(rounded.clone()),
                )]),
            };
            assert!(docobject_to_doc(obj, &schema).is_err(), "{field}");
        }
    }
}

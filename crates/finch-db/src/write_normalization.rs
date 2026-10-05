use finch_types::doc::validate_field_value;
use finch_types::{CollectionSchema, DataType, Doc, Value, ZResult};

use crate::vector_normalization::{
    normalize_sparse_f32_pairs, round_f16_f32, truncate_saturating_i16_f32,
    truncate_saturating_i8_f32,
};

// Summing duplicate sparse indices or narrowing to f32 can overflow, so each result is validated again.
pub(crate) fn normalize_vector_fields_for_write(
    schema: &CollectionSchema,
    doc: &mut Doc,
) -> ZResult<()> {
    for field in schema.vector_fields() {
        let Some(v) = doc.fields.remove(&field.name) else {
            continue;
        };
        let v = if v.is_null() {
            v
        } else {
            let v = normalize_vector_value_for_write(field.data_type, v);
            validate_field_value(field, &v)?;
            v
        };
        doc.fields.insert(field.name.clone(), v);
    }
    Ok(())
}

/// Converts a vector value to the f32 storage form of `data_type`; other values pass through.
fn normalize_vector_value_for_write(data_type: DataType, v: Value) -> Value {
    match (data_type, v) {
        (DataType::VectorFp16, Value::VecF16(items)) => {
            Value::VecF32(items.into_iter().map(|x| x.to_f32()).collect())
        }
        (DataType::VectorFp16, Value::VecF32(items)) => {
            Value::VecF32(items.into_iter().map(round_f16_f32).collect())
        }
        (DataType::VectorFp64, Value::VecF64(items)) => {
            Value::VecF32(items.into_iter().map(|x| x as f32).collect())
        }
        (DataType::VectorInt8 | DataType::VectorInt4, Value::VecI8(items)) => {
            Value::VecF32(items.into_iter().map(|x| x as f32).collect())
        }
        (DataType::VectorInt8 | DataType::VectorInt4, Value::VecF32(items)) => {
            Value::VecF32(items.into_iter().map(truncate_saturating_i8_f32).collect())
        }
        (DataType::VectorInt16, Value::VecI16(items)) => {
            Value::VecF32(items.into_iter().map(|x| x as f32).collect())
        }
        (DataType::VectorInt16, Value::VecF32(items)) => {
            Value::VecF32(items.into_iter().map(truncate_saturating_i16_f32).collect())
        }
        (DataType::SparseFp16 | DataType::SparseFp32, Value::SparseF16 { indices, values }) => {
            sparse_f32_value(indices, values.into_iter().map(|x| x.to_f32()).collect())
        }
        (DataType::SparseFp16, Value::SparseF32 { indices, values }) => {
            sparse_f32_value(indices, values.into_iter().map(round_f16_f32).collect())
        }
        (DataType::SparseFp32, Value::SparseF32 { indices, values }) => {
            sparse_f32_value(indices, values)
        }
        (_, other) => other,
    }
}

fn sparse_f32_value(indices: Vec<u32>, values: Vec<f32>) -> Value {
    let (indices, values) = normalize_sparse_f32_pairs(indices, values);
    Value::SparseF32 { indices, values }
}

pub(crate) fn normalize_binary_fields_for_write(schema: &CollectionSchema, doc: &mut Doc) {
    for field in &schema.fields {
        let Some(v) = doc.fields.get(&field.name) else {
            continue;
        };
        // BINARY fields accept string input; store it as raw bytes.
        let converted = match (field.data_type, v) {
            (DataType::Binary | DataType::Bytes, Value::String(s)) => {
                Value::Bytes(s.clone().into_bytes())
            }
            (DataType::ArrayBinary, Value::ArrayString(items)) => {
                Value::ArrayBinary(items.iter().map(|s| s.clone().into_bytes()).collect())
            }
            _ => continue,
        };
        doc.fields.insert(field.name.clone(), converted);
    }
}

use half::f16;

use finch_types::{DataType, Doc, FieldSchema, Status, VectorQuery, ZResult};

use crate::vector_search_field::VectorSearchField;

#[inline]
pub(crate) fn has_query_vector_payload(query: &VectorQuery) -> bool {
    !query.query_vector.is_empty()
        || !query.query_vector_u32.is_empty()
        || !query.query_vector_u64.is_empty()
        || !query.sparse_indices.is_empty()
        || !query.sparse_values.is_empty()
}

pub(crate) fn normalize_sparse_f32_pairs(
    indices: Vec<u32>,
    values: Vec<f32>,
) -> (Vec<u32>, Vec<f32>) {
    if indices.len() != values.len() || indices.len() <= 1 {
        return (indices, values);
    }

    let mut strictly_increasing = true;
    for i in 1..indices.len() {
        if indices[i - 1] >= indices[i] {
            strictly_increasing = false;
            break;
        }
    }
    if strictly_increasing {
        return (indices, values);
    }

    let mut pairs: Vec<(u32, f32)> = indices.into_iter().zip(values).collect();
    pairs.sort_unstable_by_key(|(i, _)| *i);

    let mut out_i: Vec<u32> = Vec::with_capacity(pairs.len());
    let mut out_v: Vec<f32> = Vec::with_capacity(pairs.len());
    for (idx, val) in pairs {
        if out_i.last().copied() == Some(idx) {
            if let Some(last) = out_v.last_mut() {
                *last += val;
            }
        } else {
            out_i.push(idx);
            out_v.push(val);
        }
    }

    (out_i, out_v)
}

#[inline]
pub(crate) fn round_f16_f32(value: f32) -> f32 {
    f16::from_f32(value).to_f32()
}

#[inline]
fn truncate_saturating_i8(value: f32) -> i8 {
    if value.is_nan() {
        0
    } else if value >= i8::MAX as f32 {
        i8::MAX
    } else if value <= i8::MIN as f32 {
        i8::MIN
    } else {
        value.trunc() as i8
    }
}

#[inline]
fn truncate_saturating_i16(value: f32) -> i16 {
    if value.is_nan() {
        0
    } else if value >= i16::MAX as f32 {
        i16::MAX
    } else if value <= i16::MIN as f32 {
        i16::MIN
    } else {
        value.trunc() as i16
    }
}

#[inline]
pub(crate) fn truncate_saturating_i8_f32(value: f32) -> f32 {
    truncate_saturating_i8(value) as f32
}

#[inline]
pub(crate) fn truncate_saturating_i16_f32(value: f32) -> f32 {
    truncate_saturating_i16(value) as f32
}

pub(crate) fn validate_and_normalize_query_payload(
    vector_field: &VectorSearchField<'_>,
    query_vector: &mut [f32],
    query_vector_u32: &[u32],
    query_vector_u64: &[u64],
    sparse_indices: &mut Vec<u32>,
    sparse_values: &mut Vec<f32>,
) -> ZResult<()> {
    let VectorSearchField {
        schema: field_schema,
        is_sparse,
        is_binary32,
        is_binary64,
        ..
    } = *vector_field;

    if is_sparse {
        validate_sparse_query_len(sparse_indices.len(), sparse_values.len())?;
    } else {
        let query_len = if is_binary32 {
            query_vector_u32.len()
        } else if is_binary64 {
            query_vector_u64.len()
        } else {
            query_vector.len()
        };
        let dim = field_schema.dimension.unwrap_or(0);
        if dim == 0 || query_len != dim {
            return Err(Status::invalid_argument(
                "query validate failed: dimension is invalid",
            ));
        }
    }

    round_query_values(field_schema.data_type, query_vector, sparse_values);

    if is_sparse {
        let (idx, vals) = normalize_sparse_f32_pairs(
            std::mem::take(sparse_indices),
            std::mem::take(sparse_values),
        );
        *sparse_indices = idx;
        *sparse_values = vals;
    }

    Ok(())
}

fn validate_sparse_query_len(indices_len: usize, values_len: usize) -> ZResult<()> {
    const MAX_SPARSE_DIM: usize = 4096;
    if indices_len != values_len {
        return Err(Status::invalid_argument(
            "query validate failed: sparse indices/values length mismatch",
        ));
    }
    if indices_len > MAX_SPARSE_DIM {
        return Err(Status::invalid_argument(
            "query validate failed: sparse indices size is too large",
        ));
    }
    Ok(())
}

/// Rounds query values to what the field's storage type can represent.
fn round_query_values(data_type: DataType, query_vector: &mut [f32], sparse_values: &mut [f32]) {
    match data_type {
        DataType::VectorFp16 => {
            for x in query_vector.iter_mut() {
                *x = round_f16_f32(*x);
            }
        }
        DataType::SparseFp16 => {
            for x in sparse_values.iter_mut() {
                *x = round_f16_f32(*x);
            }
        }
        DataType::VectorInt8 => {
            for x in query_vector.iter_mut() {
                *x = truncate_saturating_i8_f32(*x);
            }
        }
        DataType::VectorInt16 => {
            for x in query_vector.iter_mut() {
                *x = truncate_saturating_i16_f32(*x);
            }
        }
        DataType::VectorInt4 => {
            for x in query_vector.iter_mut() {
                let v = truncate_saturating_i8(*x).clamp(-8, 7);
                *x = v as f32;
            }
        }
        _ => {}
    }
}

pub(crate) fn copy_query_id_payload(
    query: &mut VectorQuery,
    field_schema: &FieldSchema,
    field: &str,
    doc: &Doc,
) -> ZResult<()> {
    match field_schema.data_type {
        DataType::VectorBinary32 => {
            let v = doc.get_vec_u32(field).ok_or_else(|| {
                Status::invalid_argument("query validate failed: id doc missing vector value")
            })?;
            query.query_vector_u32 = v.to_vec();
        }
        DataType::VectorBinary64 => {
            let v = doc.get_vec_u64(field).ok_or_else(|| {
                Status::invalid_argument("query validate failed: id doc missing vector value")
            })?;
            query.query_vector_u64 = v.to_vec();
        }
        DataType::SparseFp16 | DataType::SparseFp32 => {
            let (idx, val) = doc.get_sparse_f32(field).ok_or_else(|| {
                Status::invalid_argument("query validate failed: id doc missing vector value")
            })?;
            query.sparse_indices = idx.to_vec();
            query.sparse_values = val.to_vec();
        }
        _ => {
            let v = doc.get_vec_f32(field).ok_or_else(|| {
                Status::invalid_argument("query validate failed: id doc missing vector value")
            })?;
            query.query_vector = v.to_vec();
        }
    }

    Ok(())
}

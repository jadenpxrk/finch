use super::*;
use crate::sqlengine::normalize::{array_element_type, require_field_dtype};

const MAX_ARRAY_FIELD_LEN: usize = 32;

pub fn validate_filter_expr(filter: &FilterExpr, schema: &CollectionSchema) -> ZResult<()> {
    match filter {
        FilterExpr::AlwaysTrue | FilterExpr::AlwaysFalse => Ok(()),
        FilterExpr::And(a, b) | FilterExpr::Or(a, b) => {
            validate_filter_expr(a, schema)?;
            validate_filter_expr(b, schema)
        }
        FilterExpr::Not(inner) => validate_filter_expr(inner, schema),
        FilterExpr::IsNull(field) | FilterExpr::IsNotNull(field) => {
            require_field_dtype(schema, field).map(|_| ())
        }
        FilterExpr::ArrayLengthCompare { field, .. } => validate_array_length(schema, field),
        FilterExpr::HasPrefix { field, .. }
        | FilterExpr::HasSuffix { field, .. }
        | FilterExpr::LikePattern { field, .. } => {
            let dt = require_field_dtype(schema, field)?;
            if dt != DataType::String {
                return Err(Status::invalid_argument(format!(
                    "invalid filter: LIKE requires string field, got '{}' ({:?})",
                    field, dt
                )));
            }
            Ok(())
        }
        FilterExpr::ContainAll { field, values } | FilterExpr::ContainAny { field, values } => {
            validate_contain(schema, field, values)
        }
        FilterExpr::Compare { field, op, value } => validate_compare(schema, field, *op, value),
        FilterExpr::InList { field, values, .. } => validate_in_list(schema, field, values),
    }
}

// SearchCondCheckWalker::array_length_func_check(...)
fn validate_array_length(schema: &CollectionSchema, field: &str) -> ZResult<()> {
    let Some(fs) = schema.get_field(field) else {
        return Err(Status::invalid_argument(format!(
            "array_length argument not found in schema, with {}",
            field
        )));
    };
    if !fs.data_type.is_array() {
        return Err(Status::invalid_argument(format!(
            "array_length only support array, got {}",
            fs.data_type.as_codebook_str()
        )));
    }
    Ok(())
}

fn validate_contain(schema: &CollectionSchema, field: &str, values: &[Value]) -> ZResult<()> {
    let dt = require_field_dtype(schema, field)?;
    if array_element_type(dt).is_none() {
        return Err(Status::invalid_argument(format!(
            "invalid filter: contain requires an array field, got '{}' ({:?})",
            field, dt
        )));
    }
    if values.len() > MAX_ARRAY_FIELD_LEN {
        return Err(Status::invalid_argument(format!(
            "Contain_* rel expr only support list size no more than {}",
            MAX_ARRAY_FIELD_LEN
        )));
    }
    if values.iter().any(|v| !is_value_compatible_for_field(dt, v)) {
        return Err(Status::invalid_argument(format!(
            "invalid filter: contain values are incompatible with field '{}'",
            field
        )));
    }
    Ok(())
}

/// Scalar operators (compare / IN) reject array fields.
fn reject_array_field_for_scalar_op(dt: DataType) -> ZResult<()> {
    if array_element_type(dt).is_some() {
        return Err(Status::invalid_argument(
            "Contain_* rel expr only works with array data type and array data type only works with contain_* op.",
        ));
    }
    Ok(())
}

fn type_mismatch(field: &str) -> Status {
    Status::invalid_argument(format!(
        "invalid filter: type mismatch for field '{}'",
        field
    ))
}

fn validate_compare(
    schema: &CollectionSchema,
    field: &str,
    op: CompareOp,
    value: &Value,
) -> ZResult<()> {
    let dt = require_field_dtype(schema, field)?;
    reject_array_field_for_scalar_op(dt)?;
    if !is_value_compatible_for_field(dt, value) {
        return Err(type_mismatch(field));
    }
    if dt == DataType::Bool && !matches!(op, CompareOp::Equal | CompareOp::NotEqual) {
        // SearchCondCheckWalker bool operator check.
        return Err(Status::invalid_argument("bool type only support EQ and NQ"));
    }
    Ok(())
}

fn validate_in_list(schema: &CollectionSchema, field: &str, values: &[Value]) -> ZResult<()> {
    let dt = require_field_dtype(schema, field)?;
    reject_array_field_for_scalar_op(dt)?;
    if values.is_empty() {
        return Err(Status::invalid_argument(
            "invalid filter: IN list cannot be empty",
        ));
    }
    if values.len() > MAX_IN_LIST_LEN {
        return Err(Status::invalid_argument(format!(
            "In rel expr only support list size no more than {}",
            MAX_IN_LIST_LEN
        )));
    }
    // list-values only support string/numeric/bool, not BINARY.
    if matches!(dt, DataType::Binary | DataType::Bytes) {
        return Err(type_mismatch(field));
    }
    if values.iter().any(|v| !is_value_compatible_for_field(dt, v)) {
        return Err(type_mismatch(field));
    }
    Ok(())
}

pub(super) fn is_value_compatible_for_field(dtype: DataType, value: &Value) -> bool {
    // NULL constants are only valid via `IS NULL` / `IS NOT NULL`.
    if matches!(value, Value::Null) {
        return false;
    }

    // For contain_* on array fields, validate against the element type.
    if let Some(el) = array_element_type(dtype) {
        return is_value_compatible_for_field(el, value);
    }

    use DataType as D;
    match dtype {
        D::String => matches!(value, Value::String(_)),
        D::Bool => matches!(value, Value::Bool(_)),
        // BINARY uses string literals in the SQL dialect (treated as raw bytes).
        D::Bytes | D::Binary => matches!(value, Value::Bytes(_) | Value::String(_)),
        D::Int8 | D::Int16 | D::Int32 | D::Int64 => is_signed_literal_compatible(dtype, value),
        D::Uint8 | D::Uint16 | D::Uint32 | D::Uint64 => {
            is_unsigned_literal_compatible(dtype, value)
        }
        D::Float16 | D::Float32 | D::Float64 => is_float_literal_compatible(dtype, value),
        _ => false,
    }
}

// integer fields only accept integer literals (not floats).
fn is_signed_literal_compatible(dtype: DataType, value: &Value) -> bool {
    use DataType as D;
    match dtype {
        D::Int8 => match value {
            Value::I64(v) => *v >= i8::MIN as i64 && *v <= i8::MAX as i64,
            Value::I32(v) => *v >= i8::MIN as i32 && *v <= i8::MAX as i32,
            Value::I16(v) => *v >= i8::MIN as i16 && *v <= i8::MAX as i16,
            Value::I8(_) => true,
            _ => false,
        },
        D::Int16 => match value {
            Value::I64(v) => *v >= i16::MIN as i64 && *v <= i16::MAX as i64,
            Value::I32(v) => *v >= i16::MIN as i32 && *v <= i16::MAX as i32,
            Value::I16(_) => true,
            Value::I8(_) => true,
            _ => false,
        },
        D::Int32 => match value {
            Value::I64(v) => *v >= i32::MIN as i64 && *v <= i32::MAX as i64,
            Value::I32(_) => true,
            Value::I16(_) => true,
            Value::I8(_) => true,
            _ => false,
        },
        D::Int64 => matches!(
            value,
            Value::I64(_) | Value::I32(_) | Value::I16(_) | Value::I8(_)
        ),
        _ => false,
    }
}

fn is_unsigned_literal_compatible(dtype: DataType, value: &Value) -> bool {
    use DataType as D;
    match dtype {
        D::Uint8 => match value {
            Value::I64(v) => *v >= 0 && *v <= u8::MAX as i64,
            Value::I32(v) => *v >= 0 && (*v as i64) <= u8::MAX as i64,
            Value::U64(v) => *v <= u8::MAX as u64,
            Value::U32(v) => *v <= u8::MAX as u32,
            Value::U16(v) => *v <= u8::MAX as u16,
            Value::U8(_) => true,
            _ => false,
        },
        D::Uint16 => match value {
            Value::I64(v) => *v >= 0 && *v <= u16::MAX as i64,
            Value::I32(v) => *v >= 0 && (*v as i64) <= u16::MAX as i64,
            Value::U64(v) => *v <= u16::MAX as u64,
            Value::U32(v) => *v <= u16::MAX as u32,
            Value::U16(_) => true,
            Value::U8(_) => true,
            _ => false,
        },
        D::Uint32 => match value {
            Value::I64(v) => *v >= 0 && *v <= u32::MAX as i64,
            Value::I32(v) => *v >= 0,
            Value::U64(v) => *v <= u32::MAX as u64,
            Value::U32(_) => true,
            Value::U16(_) => true,
            Value::U8(_) => true,
            _ => false,
        },
        D::Uint64 => match value {
            Value::I64(v) => *v >= 0,
            Value::I32(v) => *v >= 0,
            Value::U64(_) => true,
            Value::U32(_) => true,
            Value::U16(_) => true,
            Value::U8(_) => true,
            _ => false,
        },
        _ => false,
    }
}

// float fields accept INT or FLOAT literals.
fn is_float_literal_compatible(dtype: DataType, value: &Value) -> bool {
    match value {
        Value::I64(_) | Value::I32(_) | Value::I16(_) | Value::I8(_) => true,
        Value::F64(v) if dtype == DataType::Float64 => v.is_finite(),
        Value::F64(v) => v.is_finite() && *v >= f32::MIN as f64 && *v <= f32::MAX as f64,
        Value::F32(v) => v.is_finite(),
        Value::F16(v) => v.to_f32().is_finite(),
        _ => false,
    }
}

pub fn count_filter_terms(expr: &FilterExpr) -> usize {
    match expr {
        FilterExpr::AlwaysTrue | FilterExpr::AlwaysFalse => 0,
        FilterExpr::And(a, b) | FilterExpr::Or(a, b) => {
            count_filter_terms(a) + count_filter_terms(b)
        }
        FilterExpr::Not(inner) => count_filter_terms(inner),
        FilterExpr::Compare { .. }
        | FilterExpr::IsNull(_)
        | FilterExpr::IsNotNull(_)
        | FilterExpr::InList { .. }
        | FilterExpr::ContainAll { .. }
        | FilterExpr::ContainAny { .. }
        | FilterExpr::LikePattern { .. }
        | FilterExpr::HasPrefix { .. }
        | FilterExpr::HasSuffix { .. }
        | FilterExpr::ArrayLengthCompare { .. } => 1,
    }
}

pub fn enforce_max_filter_terms(expr: &FilterExpr) -> ZResult<()> {
    // kMaxNumOfFilters = 4096 (SearchCondCheckWalker in query_analyzer.cc).
    const MAX_NUM_FILTERS: usize = 4096;
    let n = count_filter_terms(expr);
    if n > MAX_NUM_FILTERS {
        return Err(Status::invalid_argument(format!(
            "max number of filters is limited to {}",
            MAX_NUM_FILTERS
        )));
    }
    Ok(())
}

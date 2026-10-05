//! Filter-literal normalization
//!
//! Numeric literals are converted into the field's exact storage
//! type during analysis (e.g. `Uint64` comparisons use uint64 values, not
//! float/double). The parser uses wider literal representations
//! (`I64`/`U64`/`F64`) for convenience; this file provides a schema-aware
//! normalization pass that coerces literals into the schema's exact `Value`
//! variant so later evaluation is exact (no `f64` precision loss).

use finch_types::{
    CollectionSchema, DataType, Status, Value, ZResult, SYS_GLOBAL_DOC_ID, SYS_LOCAL_ROW_ID,
    SYS_USER_ID,
};
use half::f16;

use super::parser::FilterExpr;
pub(super) use crate::invert::array_element_type;

pub fn normalize_filter_expr_literals(
    expr: FilterExpr,
    schema: &CollectionSchema,
) -> ZResult<FilterExpr> {
    match expr {
        FilterExpr::AlwaysTrue | FilterExpr::AlwaysFalse => Ok(expr),
        FilterExpr::And(a, b) => Ok(FilterExpr::And(
            Box::new(normalize_filter_expr_literals(*a, schema)?),
            Box::new(normalize_filter_expr_literals(*b, schema)?),
        )),
        FilterExpr::Or(a, b) => Ok(FilterExpr::Or(
            Box::new(normalize_filter_expr_literals(*a, schema)?),
            Box::new(normalize_filter_expr_literals(*b, schema)?),
        )),
        FilterExpr::Not(inner) => Ok(FilterExpr::Not(Box::new(normalize_filter_expr_literals(
            *inner, schema,
        )?))),
        FilterExpr::IsNull(_)
        | FilterExpr::IsNotNull(_)
        | FilterExpr::ArrayLengthCompare { .. } => Ok(expr),
        FilterExpr::HasPrefix { .. }
        | FilterExpr::HasSuffix { .. }
        | FilterExpr::LikePattern { .. } => Ok(expr),

        FilterExpr::Compare { field, op, value } => {
            let dtype = require_field_dtype(schema, &field)?;
            let value = coerce_value_for_dtype(dtype, value)?;
            Ok(FilterExpr::Compare { field, op, value })
        }

        FilterExpr::InList {
            field,
            values,
            negated,
        } => {
            let dtype = require_field_dtype(schema, &field)?;
            Ok(FilterExpr::InList {
                values: coerce_values_for_dtype(dtype, values)?,
                field,
                negated,
            })
        }

        FilterExpr::ContainAll { field, values } => {
            let el = contain_element_dtype(schema, &field)?;
            let values = coerce_values_for_dtype(el, values)?;
            Ok(FilterExpr::ContainAll { field, values })
        }

        FilterExpr::ContainAny { field, values } => {
            let el = contain_element_dtype(schema, &field)?;
            let values = coerce_values_for_dtype(el, values)?;
            Ok(FilterExpr::ContainAny { field, values })
        }
    }
}

fn unknown_field(field: &str) -> Status {
    Status::invalid_argument(format!("invalid filter: unknown field '{}'", field))
}

/// Element type of a schema array field targeted by contain_*.
fn contain_element_dtype(schema: &CollectionSchema, field: &str) -> ZResult<DataType> {
    let fs = schema
        .get_field(field)
        .ok_or_else(|| unknown_field(field))?;
    array_element_type(fs.data_type).ok_or_else(|| {
        Status::invalid_argument(format!(
            "invalid filter: contain requires an array field, got '{}' ({:?})",
            field, fs.data_type
        ))
    })
}

fn system_column_dtype(field: &str) -> Option<DataType> {
    match field {
        SYS_USER_ID => Some(DataType::String),
        SYS_LOCAL_ROW_ID | SYS_GLOBAL_DOC_ID => Some(DataType::Uint64),
        _ => None,
    }
}

/// Data type of a system column or schema field referenced by a filter.
pub(super) fn require_field_dtype(schema: &CollectionSchema, field: &str) -> ZResult<DataType> {
    if let Some(dt) = system_column_dtype(field) {
        return Ok(dt);
    }
    schema
        .get_field(field)
        .map(|fs| fs.data_type)
        .ok_or_else(|| unknown_field(field))
}

fn coerce_values_for_dtype(dtype: DataType, values: Vec<Value>) -> ZResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for v in values {
        out.push(coerce_value_for_dtype(dtype, v)?);
    }
    Ok(out)
}

fn coerce_value_for_dtype(dtype: DataType, value: Value) -> ZResult<Value> {
    if matches!(value, Value::Null) {
        return Ok(Value::Null);
    }

    // For contain_* on array fields, coerce against the element type.
    if let Some(el) = array_element_type(dtype) {
        return coerce_value_for_dtype(el, value);
    }

    use DataType as D;
    match dtype {
        D::Bool => match value {
            Value::Bool(v) => Ok(Value::Bool(v)),
            _ => Err(Status::invalid_argument(
                "invalid filter: type mismatch for bool field",
            )),
        },
        D::String => match value {
            Value::String(v) => Ok(Value::String(v)),
            _ => Err(Status::invalid_argument(
                "invalid filter: type mismatch for string field",
            )),
        },
        D::Bytes | D::Binary => match value {
            Value::Bytes(v) => Ok(Value::Bytes(v)),
            // BINARY uses SQL string literals in filters.
            Value::String(v) => Ok(Value::Bytes(v.into_bytes())),
            _ => Err(Status::invalid_argument(
                "invalid filter: type mismatch for bytes field",
            )),
        },
        D::Int8 | D::Int16 | D::Int32 | D::Int64 => coerce_signed(dtype, &value),
        D::Uint8 | D::Uint16 | D::Uint32 | D::Uint64 => coerce_unsigned(dtype, &value),
        D::Float16 | D::Float32 | D::Float64 => coerce_float(dtype, &value),
        _ => Err(unsupported_field_type()),
    }
}

fn unsupported_field_type() -> Status {
    Status::invalid_argument("invalid filter: unsupported field type")
}

fn coerce_signed(dtype: DataType, value: &Value) -> ZResult<Value> {
    let v = normalize_i64(value);
    match dtype {
        DataType::Int8 => v
            .and_then(|v| i8::try_from(v).ok())
            .map(Value::I8)
            .ok_or_else(|| Status::invalid_argument("invalid filter: int8 literal out of range")),
        DataType::Int16 => v
            .and_then(|v| i16::try_from(v).ok())
            .map(Value::I16)
            .ok_or_else(|| Status::invalid_argument("invalid filter: int16 literal out of range")),
        DataType::Int32 => v
            .and_then(|v| i32::try_from(v).ok())
            .map(Value::I32)
            .ok_or_else(|| Status::invalid_argument("invalid filter: int32 literal out of range")),
        DataType::Int64 => v
            .map(Value::I64)
            .ok_or_else(|| Status::invalid_argument("invalid filter: int64 literal out of range")),
        _ => Err(unsupported_field_type()),
    }
}

fn coerce_unsigned(dtype: DataType, value: &Value) -> ZResult<Value> {
    let v = normalize_u64(value);
    match dtype {
        DataType::Uint8 => v
            .and_then(|v| u8::try_from(v).ok())
            .map(Value::U8)
            .ok_or_else(|| Status::invalid_argument("invalid filter: uint8 literal out of range")),
        DataType::Uint16 => v
            .and_then(|v| u16::try_from(v).ok())
            .map(Value::U16)
            .ok_or_else(|| Status::invalid_argument("invalid filter: uint16 literal out of range")),
        DataType::Uint32 => v
            .and_then(|v| u32::try_from(v).ok())
            .map(Value::U32)
            .ok_or_else(|| Status::invalid_argument("invalid filter: uint32 literal out of range")),
        DataType::Uint64 => v
            .map(Value::U64)
            .ok_or_else(|| Status::invalid_argument("invalid filter: uint64 literal out of range")),
        _ => Err(unsupported_field_type()),
    }
}

fn fits_f32(v: f64) -> bool {
    v.is_finite() && v >= f32::MIN as f64 && v <= f32::MAX as f64
}

fn coerce_float(dtype: DataType, value: &Value) -> ZResult<Value> {
    let v = normalize_f64(value);
    match dtype {
        DataType::Float16 => v
            .filter(|v| fits_f32(*v))
            .map(|v| Value::F16(f16::from_f32(v as f32)))
            .ok_or_else(|| {
                Status::invalid_argument("invalid filter: float16 literal out of range")
            }),
        DataType::Float32 => v
            .filter(|v| fits_f32(*v))
            .map(|v| Value::F32(v as f32))
            .ok_or_else(|| {
                Status::invalid_argument("invalid filter: float32 literal out of range")
            }),
        DataType::Float64 => v.filter(|v| v.is_finite()).map(Value::F64).ok_or_else(|| {
            Status::invalid_argument("invalid filter: float64 literal out of range")
        }),
        _ => Err(unsupported_field_type()),
    }
}

fn normalize_i64(v: &Value) -> Option<i64> {
    match v {
        Value::I64(x) => Some(*x),
        Value::I32(x) => Some(*x as i64),
        Value::I16(x) => Some(*x as i64),
        Value::I8(x) => Some(*x as i64),
        _ => None,
    }
}

fn normalize_u64(v: &Value) -> Option<u64> {
    match v {
        Value::U64(x) => Some(*x),
        Value::U32(x) => Some(*x as u64),
        Value::U16(x) => Some(*x as u64),
        Value::U8(x) => Some(*x as u64),
        Value::I64(x) if *x >= 0 => Some(*x as u64),
        Value::I32(x) if *x >= 0 => Some(*x as u64),
        Value::I16(x) if *x >= 0 => Some(*x as u64),
        Value::I8(x) if *x >= 0 => Some(*x as u64),
        _ => None,
    }
}

fn normalize_f64(v: &Value) -> Option<f64> {
    match v {
        Value::F64(x) => Some(*x),
        Value::F32(x) => Some(*x as f64),
        Value::F16(x) => Some(x.to_f32() as f64),
        Value::I64(x) => Some(*x as f64),
        Value::I32(x) => Some(*x as f64),
        Value::I16(x) => Some(*x as f64),
        Value::I8(x) => Some(*x as f64),
        Value::U64(x) => Some(*x as f64),
        Value::U32(x) => Some(*x as f64),
        Value::U16(x) => Some(*x as f64),
        Value::U8(x) => Some(*x as f64),
        _ => None,
    }
}

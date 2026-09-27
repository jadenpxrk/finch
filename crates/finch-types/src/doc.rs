use crate::index_params::IndexParams;
use crate::schema::{CollectionSchema, FieldSchema};
use crate::status::{Status, ZResult};
use crate::types::DataType;
use crate::types::Operator;
use half::f16;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A polymorphic value for document fields
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Null,
    Bool(bool),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F16(f16),
    F32(f32),
    F64(f64),
    String(String),
    Bytes(Vec<u8>),
    // Dense vector types
    VecBool(Vec<bool>),
    VecI8(Vec<i8>),
    VecI16(Vec<i16>),
    VecI32(Vec<i32>),
    VecI64(Vec<i64>),
    VecU32(Vec<u32>),
    VecU64(Vec<u64>),
    VecF16(Vec<f16>),
    VecF32(Vec<f32>),
    VecF64(Vec<f64>),
    VecString(Vec<String>),
    // Sparse vector types
    SparseF16 { indices: Vec<u32>, values: Vec<f16> },
    SparseF32 { indices: Vec<u32>, values: Vec<f32> },
    // Array types
    ArrayBinary(Vec<Vec<u8>>),
    ArrayI32(Vec<i32>),
    ArrayI64(Vec<i64>),
    ArrayU32(Vec<u32>),
    ArrayU64(Vec<u64>),
    ArrayBool(Vec<bool>),
    ArrayF32(Vec<f32>),
    ArrayF64(Vec<f64>),
    ArrayString(Vec<String>),
}

impl Value {
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Value::F32(v) => Some(*v),
            Value::F64(v) => Some(*v as f32),
            Value::I32(v) => Some(*v as f32),
            Value::I64(v) => Some(*v as f32),
            Value::U32(v) => Some(*v as f32),
            Value::U64(v) => Some(*v as f32),
            Value::F16(v) => Some(v.to_f32()),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::F64(v) => Some(*v),
            Value::F32(v) => Some(*v as f64),
            Value::I32(v) => Some(*v as f64),
            Value::I64(v) => Some(*v as f64),
            Value::U32(v) => Some(*v as f64),
            Value::U64(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::I64(v) => Some(*v),
            Value::I32(v) => Some(*v as i64),
            Value::U32(v) => Some(*v as i64),
            Value::U64(v) => Some(*v as i64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_vec_f32(&self) -> Option<&[f32]> {
        match self {
            Value::VecF32(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_vec_u32(&self) -> Option<&[u32]> {
        match self {
            Value::VecU32(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_vec_u64(&self) -> Option<&[u64]> {
        match self {
            Value::VecU64(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_sparse_f32(&self) -> Option<(&[u32], &[f32])> {
        match self {
            Value::SparseF32 { indices, values } => Some((indices, values)),
            _ => None,
        }
    }

    /// Returns true for any vector variant (dense or sparse).
    /// Used by output-control stripping in query results.
    pub fn is_vector(&self) -> bool {
        matches!(
            self,
            Value::VecBool(_)
                | Value::VecI8(_)
                | Value::VecI16(_)
                | Value::VecI32(_)
                | Value::VecI64(_)
                | Value::VecU32(_)
                | Value::VecU64(_)
                | Value::VecF16(_)
                | Value::VecF32(_)
                | Value::VecF64(_)
                | Value::SparseF16 { .. }
                | Value::SparseF32 { .. }
        )
    }

    /// Type name for error messages
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Null => "null",
            Value::Bool(_) => "bool",
            Value::I8(_) => "i8",
            Value::I16(_) => "i16",
            Value::I32(_) => "i32",
            Value::I64(_) => "i64",
            Value::U8(_) => "u8",
            Value::U16(_) => "u16",
            Value::U32(_) => "u32",
            Value::U64(_) => "u64",
            Value::F16(_) => "f16",
            Value::F32(_) => "f32",
            Value::F64(_) => "f64",
            Value::String(_) => "string",
            Value::Bytes(_) => "bytes",
            Value::VecBool(_) => "vec_bool",
            Value::VecI8(_) => "vec_i8",
            Value::VecI16(_) => "vec_i16",
            Value::VecI32(_) => "vec_i32",
            Value::VecI64(_) => "vec_i64",
            Value::VecU32(_) => "vec_u32",
            Value::VecU64(_) => "vec_u64",
            Value::VecF16(_) => "vec_f16",
            Value::VecF32(_) => "vec_f32",
            Value::VecF64(_) => "vec_f64",
            Value::VecString(_) => "vec_string",
            Value::SparseF16 { .. } => "sparse_f16",
            Value::SparseF32 { .. } => "sparse_f32",
            Value::ArrayBinary(_) => "array_binary",
            Value::ArrayI32(_) => "array_i32",
            Value::ArrayI64(_) => "array_i64",
            Value::ArrayU32(_) => "array_u32",
            Value::ArrayU64(_) => "array_u64",
            Value::ArrayBool(_) => "array_bool",
            Value::ArrayF32(_) => "array_f32",
            Value::ArrayF64(_) => "array_f64",
            Value::ArrayString(_) => "array_string",
        }
    }
}

/// A document with primary key, fields, and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Doc {
    pub pk: String,
    pub score: f32,
    pub doc_id: u64,
    pub op: Operator,
    pub fields: HashMap<String, Value>,
}

fn is_valid_doc_pk(pk: &str) -> bool {
    // `^[a-zA-Z0-9_!@#$%+=.-]{1,64}$`
    let b = pk.as_bytes();
    if b.is_empty() || b.len() > 64 {
        return false;
    }
    b.iter().all(|&c| {
        c.is_ascii_alphanumeric()
            || matches!(
                c,
                b'_' | b'!' | b'@' | b'#' | b'$' | b'%' | b'+' | b'=' | b'.' | b'-'
            )
    })
}

fn value_matches_data_type(v: &Value, dt: DataType) -> bool {
    use DataType as D;
    match dt {
        D::Bool => matches!(v, Value::Bool(_)),
        D::Int8 => matches!(v, Value::I8(_)),
        D::Int16 => matches!(v, Value::I16(_)),
        D::Int32 => matches!(v, Value::I32(_)),
        D::Int64 => matches!(v, Value::I64(_)),
        D::Uint32 => matches!(v, Value::U32(_)),
        D::Uint64 => matches!(v, Value::U64(_)),
        D::Uint8 => matches!(v, Value::U8(_)),
        D::Uint16 => matches!(v, Value::U16(_)),
        D::Float16 => matches!(v, Value::F16(_)),
        D::Float32 => matches!(v, Value::F32(_)),
        D::Float64 => matches!(v, Value::F64(_)),
        D::String => matches!(v, Value::String(_)),
        D::Bytes | D::Binary => matches!(v, Value::Bytes(_)),
        // Dense vectors: accept the canonical variant per vector dtype, plus VecF32
        // for ergonomic interop (query/index surface is f32).
        D::VectorFp32 => matches!(v, Value::VecF32(_)),
        D::VectorFp16 => matches!(v, Value::VecF16(_) | Value::VecF32(_)),
        D::VectorInt8 => matches!(v, Value::VecI8(_) | Value::VecF32(_)),
        D::VectorInt16 => matches!(v, Value::VecI16(_) | Value::VecF32(_)),
        D::VectorInt4 => matches!(v, Value::VecI8(_) | Value::VecF32(_)),
        D::VectorFp64 => matches!(v, Value::VecF64(_) | Value::VecF32(_)),
        D::VectorBinary32 => matches!(v, Value::VecU32(_)),
        D::VectorBinary64 => matches!(v, Value::VecU64(_)),
        // Other vector dtypes are currently treated as f32 at the API boundary.
        D::VectorBool | D::VectorInt32 | D::VectorInt64 | D::VectorUint32 | D::VectorUint64 => {
            matches!(v, Value::VecF32(_))
        }
        // Sparse vectors: accept either fp32 or fp16 payloads.
        D::SparseFp32 | D::SparseFp16 => {
            matches!(v, Value::SparseF32 { .. } | Value::SparseF16 { .. })
        }
        // Arrays (best-effort)
        D::ArrayInt32 => matches!(v, Value::ArrayI32(_)),
        D::ArrayInt64 => matches!(v, Value::ArrayI64(_)),
        D::ArrayUint32 => match v {
            Value::ArrayU32(_) => true,
            Value::ArrayI32(items) => items.iter().all(|&x| x >= 0),
            Value::ArrayI64(items) => items.iter().all(|&x| x >= 0 && x <= u32::MAX as i64),
            _ => false,
        },
        D::ArrayUint64 => match v {
            Value::ArrayU64(_) => true,
            Value::ArrayI32(items) => items.iter().all(|&x| x >= 0),
            Value::ArrayI64(items) => items.iter().all(|&x| x >= 0),
            _ => false,
        },
        D::ArrayBool => matches!(v, Value::ArrayBool(_)),
        D::ArrayBinary => matches!(v, Value::ArrayBinary(_)),
        D::ArrayFp32 => matches!(v, Value::ArrayF32(_)),
        D::ArrayFp64 => matches!(v, Value::ArrayF64(_)),
        D::ArrayString => matches!(v, Value::ArrayString(_)),
        _ => true,
    }
}

impl Doc {
    pub fn new(pk: impl Into<String>) -> Self {
        Doc {
            pk: pk.into(),
            score: 0.0,
            doc_id: 0,
            op: Operator::Insert,
            fields: HashMap::new(),
        }
    }

    pub fn with_op(mut self, op: Operator) -> Self {
        self.op = op;
        self
    }

    pub fn set(mut self, field: impl Into<String>, value: impl Into<Value>) -> Self {
        self.fields.insert(field.into(), value.into());
        self
    }

    pub fn set_field(&mut self, field: impl Into<String>, value: impl Into<Value>) {
        self.fields.insert(field.into(), value.into());
    }

    pub fn get(&self, field: &str) -> Option<&Value> {
        self.fields.get(field)
    }

    pub fn has(&self, field: &str) -> bool {
        self.fields.contains_key(field)
    }

    pub fn is_null(&self, field: &str) -> bool {
        self.fields.get(field).map(|v| v.is_null()).unwrap_or(true)
    }

    pub fn get_f32(&self, field: &str) -> ZResult<Option<f32>> {
        match self.fields.get(field) {
            None | Some(Value::Null) => Ok(None),
            Some(v) => v.as_f32().map(Some).ok_or_else(|| {
                Status::invalid_argument(format!(
                    "field '{}' is not numeric (got {})",
                    field,
                    v.type_name()
                ))
            }),
        }
    }

    pub fn get_str(&self, field: &str) -> Option<&str> {
        self.fields.get(field)?.as_str()
    }

    pub fn get_vec_f32(&self, field: &str) -> Option<&[f32]> {
        self.fields.get(field)?.as_vec_f32()
    }

    pub fn get_vec_u32(&self, field: &str) -> Option<&[u32]> {
        self.fields.get(field)?.as_vec_u32()
    }

    pub fn get_vec_u64(&self, field: &str) -> Option<&[u64]> {
        self.fields.get(field)?.as_vec_u64()
    }

    pub fn get_sparse_f32(&self, field: &str) -> Option<(&[u32], &[f32])> {
        self.fields.get(field)?.as_sparse_f32()
    }

    pub fn validate(&self, schema: &CollectionSchema, is_update: bool) -> ZResult<()> {
        if self.pk.is_empty() {
            return Err(Status::invalid_argument(
                "doc validate failed: doc_id is not set",
            ));
        }
        if !is_valid_doc_pk(&self.pk) {
            return Err(Status::invalid_argument(format!(
                "doc validate failed: doc_id[{}] cannot pass the regex verification",
                self.pk
            )));
        }

        // Unknown fields not allowed.
        for name in self.fields.keys() {
            if !schema.has_field(name) {
                return Err(Status::invalid_argument(format!(
                    "doc validate failed: field[{}] does not exist in collection's schema",
                    name
                )));
            }
        }

        // Required field presence + basic type/dimension checks.
        for field_schema in &schema.fields {
            let name = field_schema.name.as_str();
            let Some(value) = self.fields.get(name) else {
                if field_schema.nullable || is_update {
                    continue;
                }
                return Err(Status::invalid_argument(format!(
                    "doc validate failed: field[{}] is configured not nullable, but doc does not contain this field",
                    name
                )));
            };

            if value.is_null() {
                if field_schema.nullable {
                    continue;
                }
                return Err(Status::invalid_argument(format!(
                    "doc validate failed: field[{}] is configured not nullable, but doc's field value is empty",
                    name
                )));
            }

            validate_field_value(field_schema, value)?;
        }

        Ok(())
    }
}

/// Type, dimension, and encoding checks for a present, non-null field value.
pub fn validate_field_value(field_schema: &FieldSchema, value: &Value) -> ZResult<()> {
    if !value_matches_data_type(value, field_schema.data_type) {
        return Err(Status::invalid_argument(format!(
            "doc validate failed: field[{}]'s type mismatch (expected {:?}, got {})",
            field_schema.name,
            field_schema.data_type,
            value.type_name()
        )));
    }
    if field_schema.data_type.is_vector() && !field_schema.data_type.is_sparse() {
        validate_dense_vector(field_schema, value)?;
    }
    if field_schema.data_type.is_sparse() {
        validate_sparse_vector(value)?;
    }
    validate_finite(field_schema, value)?;
    validate_invert_encodable(field_schema, value)
}

// The inverted index ends each string term with a NUL byte, so an indexed string cannot contain one.
fn validate_invert_encodable(field_schema: &FieldSchema, value: &Value) -> ZResult<()> {
    if !matches!(field_schema.index_params, Some(IndexParams::Invert(_))) {
        return Ok(());
    }
    let has_nul = match value {
        Value::String(s) => s.contains('\0'),
        Value::ArrayString(items) => items.iter().any(|s| s.contains('\0')),
        _ => false,
    };
    if !has_nul {
        return Ok(());
    }
    Err(Status::invalid_argument(format!(
        "doc validate failed: field[{}] has an inverted index and cannot contain a NUL character",
        field_schema.name
    )))
}

// The WAL encodes docs as JSON, which cannot hold NaN or infinity, so reject values that are or become non-finite.
fn validate_finite(field_schema: &FieldSchema, value: &Value) -> ZResult<()> {
    let stored_finite: fn(f64) -> bool = match field_schema.data_type {
        DataType::VectorFp16 | DataType::SparseFp16 => |x| f16::from_f64(x).is_finite(),
        // Integer vectors store a saturating cast, which is always finite.
        DataType::VectorInt8 | DataType::VectorInt16 => |_| true,
        _ => f64::is_finite,
    };
    let finite = match value {
        Value::F16(x) => stored_finite(x.to_f64()),
        Value::F32(x) => stored_finite(f64::from(*x)),
        Value::F64(x) => stored_finite(*x),
        Value::VecF16(items) | Value::SparseF16 { values: items, .. } => {
            items.iter().all(|x| stored_finite(x.to_f64()))
        }
        Value::VecF32(items) | Value::ArrayF32(items) | Value::SparseF32 { values: items, .. } => {
            items.iter().all(|&x| stored_finite(f64::from(x)))
        }
        Value::VecF64(items) | Value::ArrayF64(items) => items.iter().all(|&x| stored_finite(x)),
        _ => true,
    };
    if finite {
        return Ok(());
    }
    Err(Status::invalid_argument(format!(
        "doc validate failed: field[{}] contains a NaN or infinite value, or one out of range for {:?}",
        field_schema.name, field_schema.data_type
    )))
}

fn validate_dense_vector(field_schema: &FieldSchema, value: &Value) -> ZResult<()> {
    let dim = field_schema.dimension.unwrap_or(0);
    let value_dim = match value {
        Value::VecF32(v) => v.len(),
        Value::VecF16(v) => v.len(),
        Value::VecI8(v) => v.len(),
        Value::VecI16(v) => v.len(),
        Value::VecF64(v) => v.len(),
        Value::VecU32(v) => v.len(),
        Value::VecU64(v) => v.len(),
        _ => {
            return Err(Status::invalid_argument(
                "doc validate failed: dimension is invalid",
            ));
        }
    };
    if dim == 0 || value_dim != dim {
        return Err(Status::invalid_argument(
            "doc validate failed: dimension is invalid",
        ));
    }
    if field_schema.data_type != DataType::VectorInt4 {
        return Ok(());
    }

    // Finch represents int4 vectors using i8/f32 values but must
    // keep the range constraint [-8, 7].
    let ok = match value {
        Value::VecI8(items) => items.iter().all(|&x| (-8..=7).contains(&x)),
        Value::VecF32(items) => items
            .iter()
            .all(|&x| x.is_finite() && (-8.0..=7.0).contains(&x)),
        _ => false,
    };
    if !ok {
        return Err(Status::invalid_argument(
            "doc validate failed: int4 vector contains out-of-range values",
        ));
    }
    Ok(())
}

fn validate_sparse_vector(value: &Value) -> ZResult<()> {
    let lengths_match = match value {
        Value::SparseF32 { indices, values } => indices.len() == values.len(),
        Value::SparseF16 { indices, values } => indices.len() == values.len(),
        _ => false,
    };
    if !lengths_match {
        return Err(Status::invalid_argument(
            "doc validate failed: invalid sparse vector encoding",
        ));
    }
    Ok(())
}

/// Conversion helpers for common Rust types → Value
impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Value::Bool(v)
    }
}
impl From<i8> for Value {
    fn from(v: i8) -> Self {
        Value::I8(v)
    }
}
impl From<i16> for Value {
    fn from(v: i16) -> Self {
        Value::I16(v)
    }
}
impl From<i32> for Value {
    fn from(v: i32) -> Self {
        Value::I32(v)
    }
}
impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Value::I64(v)
    }
}
impl From<u8> for Value {
    fn from(v: u8) -> Self {
        Value::U8(v)
    }
}
impl From<u16> for Value {
    fn from(v: u16) -> Self {
        Value::U16(v)
    }
}
impl From<u32> for Value {
    fn from(v: u32) -> Self {
        Value::U32(v)
    }
}
impl From<u64> for Value {
    fn from(v: u64) -> Self {
        Value::U64(v)
    }
}
impl From<f16> for Value {
    fn from(v: f16) -> Self {
        Value::F16(v)
    }
}
impl From<f32> for Value {
    fn from(v: f32) -> Self {
        Value::F32(v)
    }
}
impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Value::F64(v)
    }
}
impl From<String> for Value {
    fn from(v: String) -> Self {
        Value::String(v)
    }
}
impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Value::String(v.to_string())
    }
}
impl From<Vec<u8>> for Value {
    fn from(v: Vec<u8>) -> Self {
        Value::Bytes(v)
    }
}
impl From<Vec<bool>> for Value {
    fn from(v: Vec<bool>) -> Self {
        Value::VecBool(v)
    }
}
impl From<Vec<i8>> for Value {
    fn from(v: Vec<i8>) -> Self {
        Value::VecI8(v)
    }
}
impl From<Vec<i16>> for Value {
    fn from(v: Vec<i16>) -> Self {
        Value::VecI16(v)
    }
}
impl From<Vec<f32>> for Value {
    fn from(v: Vec<f32>) -> Self {
        Value::VecF32(v)
    }
}
impl From<Vec<f64>> for Value {
    fn from(v: Vec<f64>) -> Self {
        Value::VecF64(v)
    }
}
impl From<Vec<i32>> for Value {
    fn from(v: Vec<i32>) -> Self {
        Value::VecI32(v)
    }
}
impl From<Vec<i64>> for Value {
    fn from(v: Vec<i64>) -> Self {
        Value::VecI64(v)
    }
}
impl From<Vec<u32>> for Value {
    fn from(v: Vec<u32>) -> Self {
        Value::VecU32(v)
    }
}
impl From<Vec<u64>> for Value {
    fn from(v: Vec<u64>) -> Self {
        Value::VecU64(v)
    }
}
impl From<Vec<String>> for Value {
    fn from(v: Vec<String>) -> Self {
        Value::VecString(v)
    }
}
impl From<Vec<f16>> for Value {
    fn from(v: Vec<f16>) -> Self {
        Value::VecF16(v)
    }
}

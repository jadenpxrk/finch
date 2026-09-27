//! JSON field value -> typed finch `Value`, driven by the schema data type.

use finch_types::{DataType as D, Value as V};
use half::f16;
use napi::bindgen_prelude::*;
use serde_json::Value as Json;

pub(super) fn json_to_typed_value(v: Json, dt: D) -> Result<V> {
    if matches!(v, Json::Null) {
        return Ok(V::Null);
    }

    match dt {
        D::Bool => match v {
            Json::Bool(b) => Ok(V::Bool(b)),
            _ => Err(err("bool")),
        },
        D::String => match v {
            Json::String(s) => Ok(V::String(s)),
            _ => Err(err("string")),
        },
        D::Int8 | D::Int16 | D::Int32 | D::Int64 => json_to_signed(v, dt),
        D::Uint8 | D::Uint16 | D::Uint32 | D::Uint64 => json_to_unsigned(v, dt),
        D::Float16 => Ok(V::F16(f16::from_f32(json_to_f64(v, "float16")? as f32))),
        D::Float32 => Ok(V::F32(json_to_f64(v, "float32")? as f32)),
        D::Float64 => Ok(V::F64(json_to_f64(v, "float64")?)),
        D::Bytes | D::Binary => match v {
            // Finch will normalize string -> bytes for binary fields before validation.
            Json::String(s) => Ok(V::String(s)),
            Json::Array(items) => Ok(V::Bytes(json_items(items, |it| {
                json_u8(&it, "bytes (u8 array)")
            })?)),
            _ => Err(err("bytes (string or u8 array)")),
        },
        D::VectorBinary32 => {
            const EXPECTED: &str = "vector_binary32 (u32 array)";
            Ok(V::VecU32(json_array(v, EXPECTED, |it| {
                json_u32(&it, EXPECTED)
            })?))
        }
        D::VectorBinary64 => {
            const EXPECTED: &str = "vector_binary64 (u64 array)";
            Ok(V::VecU64(json_array(v, EXPECTED, |it| {
                json_u64_or_str(it, EXPECTED)
            })?))
        }
        // Dense vectors: for Node ergonomics, accept number arrays and store as VecF32.
        D::VectorFp16
        | D::VectorFp32
        | D::VectorFp64
        | D::VectorInt4
        | D::VectorInt8
        | D::VectorInt16
        | D::VectorBool
        | D::VectorInt32
        | D::VectorInt64
        | D::VectorUint32
        | D::VectorUint64 => {
            const EXPECTED: &str = "vector (number array)";
            Ok(V::VecF32(json_array(v, EXPECTED, |it| {
                json_f32(&it, EXPECTED)
            })?))
        }
        D::SparseFp16 | D::SparseFp32 => json_to_sparse(v),
        D::ArrayBool
        | D::ArrayString
        | D::ArrayInt32
        | D::ArrayInt64
        | D::ArrayUint32
        | D::ArrayUint64
        | D::ArrayFp32
        | D::ArrayFp64
        | D::ArrayBinary => json_to_array(v, dt),
        _ => Err(unsupported(dt)),
    }
}

fn err(expected: &str) -> napi::Error {
    napi::Error::from_reason(format!("invalid value: expected {expected}"))
}

fn unsupported(dt: D) -> napi::Error {
    napi::Error::from_reason(format!(
        "unsupported field data type for Node doc conversion: {dt:?}"
    ))
}

fn num_i64(v: &Json) -> Option<i64> {
    match v {
        Json::Number(n) => n.as_i64(),
        _ => None,
    }
}

fn num_u64(v: &Json) -> Option<u64> {
    match v {
        Json::Number(n) => n.as_u64(),
        _ => None,
    }
}

fn parse_u64_str(s: &str) -> Result<u64> {
    s.parse::<u64>()
        .map_err(|_| napi::Error::from_reason("invalid u64 string"))
}

fn parse_i64_str(s: &str) -> Result<i64> {
    s.parse::<i64>()
        .map_err(|_| napi::Error::from_reason("invalid i64 string"))
}

fn parse_u32_str(s: &str) -> Result<u32> {
    s.parse::<u32>()
        .map_err(|_| napi::Error::from_reason("invalid u32 string"))
}

fn parse_i32_str(s: &str) -> Result<i32> {
    s.parse::<i32>()
        .map_err(|_| napi::Error::from_reason("invalid i32 string"))
}

fn json_to_signed(v: Json, dt: D) -> Result<V> {
    match dt {
        D::Int8 => Ok(V::I8(
            json_to_ranged_i64(v, "int8", i8::MIN as i64, i8::MAX as i64)? as i8,
        )),
        D::Int16 => Ok(V::I16(
            json_to_ranged_i64(v, "int16", i16::MIN as i64, i16::MAX as i64)? as i16,
        )),
        D::Int32 => match v {
            Json::Number(_) => {
                let Some(x) = num_i64(&v) else {
                    return Err(err("int32"));
                };
                if x < i32::MIN as i64 || x > i32::MAX as i64 {
                    return Err(napi::Error::from_reason("int32 out of range"));
                }
                Ok(V::I32(x as i32))
            }
            Json::String(s) => Ok(V::I32(parse_i32_str(&s)?)),
            _ => Err(err("int32")),
        },
        D::Int64 => match v {
            Json::Number(_) => {
                let Some(x) = num_i64(&v) else {
                    return Err(err("int64"));
                };
                Ok(V::I64(x))
            }
            Json::String(s) => Ok(V::I64(parse_i64_str(&s)?)),
            _ => Err(err("int64")),
        },
        _ => Err(unsupported(dt)),
    }
}

/// Integer from a JSON number or decimal string, checked against `[min, max]`.
fn json_to_ranged_i64(v: Json, name: &str, min: i64, max: i64) -> Result<i64> {
    let x = match v {
        Json::Number(_) => {
            let Some(x) = num_i64(&v) else {
                return Err(err(name));
            };
            x
        }
        Json::String(s) => parse_i64_str(&s)?,
        _ => return Err(err(name)),
    };
    if x < min || x > max {
        return Err(napi::Error::from_reason(format!("{name} out of range")));
    }
    Ok(x)
}

fn json_to_unsigned(v: Json, dt: D) -> Result<V> {
    match dt {
        D::Uint8 => Ok(V::U8(json_to_ranged_u64(v, "uint8", u8::MAX as u64)? as u8)),
        D::Uint16 => Ok(V::U16(
            json_to_ranged_u64(v, "uint16", u16::MAX as u64)? as u16
        )),
        D::Uint32 => match v {
            Json::Number(_) => {
                let Some(x) = num_u64(&v) else {
                    return Err(err("uint32"));
                };
                if x > u32::MAX as u64 {
                    return Err(napi::Error::from_reason("uint32 out of range"));
                }
                Ok(V::U32(x as u32))
            }
            Json::String(s) => Ok(V::U32(parse_u32_str(&s)?)),
            _ => Err(err("uint32")),
        },
        D::Uint64 => match v {
            Json::Number(_) => {
                let Some(x) = num_u64(&v) else {
                    return Err(err("uint64"));
                };
                Ok(V::U64(x))
            }
            Json::String(s) => Ok(V::U64(parse_u64_str(&s)?)),
            _ => Err(err("uint64 (number or decimal string)")),
        },
        _ => Err(unsupported(dt)),
    }
}

/// Unsigned integer from a JSON number or decimal string, checked against `max`.
fn json_to_ranged_u64(v: Json, name: &str, max: u64) -> Result<u64> {
    let x = match v {
        Json::Number(_) => {
            let Some(x) = num_u64(&v) else {
                return Err(err(name));
            };
            x
        }
        Json::String(s) => parse_u64_str(&s)?,
        _ => return Err(err(name)),
    };
    if x > max {
        return Err(napi::Error::from_reason(format!("{name} out of range")));
    }
    Ok(x)
}

fn json_to_f64(v: Json, name: &str) -> Result<f64> {
    match v {
        Json::Number(n) => n.as_f64().ok_or_else(|| err(name)),
        _ => Err(err(name)),
    }
}

// Sparse vectors: accept either {index: value, ...} map or {"indices":[...],"values":[...]} shape.
fn json_to_sparse(v: Json) -> Result<V> {
    let Json::Object(map) = v else {
        return Err(err("sparse vector object"));
    };
    if let (Some(idx_v), Some(val_v)) = (map.get("indices"), map.get("values")) {
        return sparse_from_parallel_arrays(idx_v, val_v);
    }
    sparse_from_index_map(map)
}

fn sparse_from_parallel_arrays(idx_v: &Json, val_v: &Json) -> Result<V> {
    let Json::Array(idxs) = idx_v else {
        return Err(err("sparse.indices (u32[])"));
    };
    let Json::Array(vals) = val_v else {
        return Err(err("sparse.values (number[])"));
    };
    if idxs.len() != vals.len() {
        return Err(napi::Error::from_reason(
            "sparse indices/values length mismatch",
        ));
    }
    let mut indices: Vec<u32> = Vec::with_capacity(idxs.len());
    let mut values: Vec<f32> = Vec::with_capacity(vals.len());
    for (i, x) in idxs.iter().enumerate() {
        let Some(ix) = x.as_u64() else {
            return Err(err("sparse.indices (u32[])"));
        };
        if ix > u32::MAX as u64 {
            return Err(napi::Error::from_reason("sparse index out of range"));
        }
        let Some(vx) = vals[i].as_f64() else {
            return Err(err("sparse.values (number[])"));
        };
        indices.push(ix as u32);
        values.push(vx as f32);
    }
    Ok(V::SparseF32 { indices, values })
}

// Map form: {"1": 0.5, "2": 1.0}
fn sparse_from_index_map(map: serde_json::Map<String, Json>) -> Result<V> {
    let mut pairs: Vec<(u32, f32)> = Vec::with_capacity(map.len());
    for (k, vv) in map {
        let idx = k
            .parse::<u32>()
            .map_err(|_| err("sparse index key (u32 string)"))?;
        let Some(val) = vv.as_f64() else {
            return Err(err("sparse value (number)"));
        };
        pairs.push((idx, val as f32));
    }
    pairs.sort_by_key(|(i, _)| *i);
    let (indices, values): (Vec<u32>, Vec<f32>) = pairs.into_iter().unzip();
    Ok(V::SparseF32 { indices, values })
}

fn json_to_array(v: Json, dt: D) -> Result<V> {
    match dt {
        D::ArrayBool => Ok(V::ArrayBool(json_array(v, "bool[]", |it| {
            it.as_bool().ok_or_else(|| err("bool[]"))
        })?)),
        D::ArrayString => Ok(V::ArrayString(json_array(v, "string[]", |it| {
            it.as_str()
                .map(str::to_string)
                .ok_or_else(|| err("string[]"))
        })?)),
        D::ArrayInt32 => Ok(V::ArrayI32(json_array(v, "i32[]", |it| json_i32(&it))?)),
        D::ArrayInt64 => Ok(V::ArrayI64(json_array(v, "i64[]", |it| {
            json_to_ranged_i64(it, "i64[]", i64::MIN, i64::MAX)
        })?)),
        D::ArrayUint32 => Ok(V::ArrayU32(json_array(v, "u32[]", |it| {
            json_u32_or_str(it, "u32[]")
        })?)),
        D::ArrayUint64 => Ok(V::ArrayU64(json_array(v, "u64[]", |it| {
            json_u64_or_str(it, "u64[]")
        })?)),
        D::ArrayFp32 => Ok(V::ArrayF32(json_array(v, "f32[]", |it| {
            json_f32(&it, "f32[]")
        })?)),
        D::ArrayFp64 => Ok(V::ArrayF64(json_array(v, "f64[]", |it| {
            it.as_f64().ok_or_else(|| err("f64[]"))
        })?)),
        D::ArrayBinary => json_to_array_binary(v),
        _ => Err(unsupported(dt)),
    }
}

// Finch will normalize ArrayString -> ArrayBinary before validation.
fn json_to_array_binary(v: Json) -> Result<V> {
    let Json::Array(items) = v else {
        return Err(err("array_binary"));
    };
    // Accept string[] for convenience.
    if items.iter().all(|x| x.is_string()) {
        let out = items
            .into_iter()
            .map(|x| x.as_str().unwrap_or_default().to_string())
            .collect::<Vec<_>>();
        return Ok(V::ArrayString(out));
    }
    // Accept u8[][].
    let mut out: Vec<Vec<u8>> = Vec::with_capacity(items.len());
    for it in items {
        let Json::Array(bytes) = it else {
            return Err(err("array_binary (string[] or u8[][])"));
        };
        out.push(json_items(bytes, |b| {
            json_u8(&b, "array_binary element (u8[])")
        })?);
    }
    Ok(V::ArrayBinary(out))
}

/// Converts each element of a JSON array; a non-array fails with `err(expected)`.
fn json_array<T>(v: Json, expected: &str, item: impl Fn(Json) -> Result<T>) -> Result<Vec<T>> {
    let Json::Array(items) = v else {
        return Err(err(expected));
    };
    json_items(items, item)
}

fn json_items<T>(items: Vec<Json>, item: impl Fn(Json) -> Result<T>) -> Result<Vec<T>> {
    let mut out: Vec<T> = Vec::with_capacity(items.len());
    for it in items {
        out.push(item(it)?);
    }
    Ok(out)
}

fn json_u8(it: &Json, expected: &str) -> Result<u8> {
    let Some(x) = it.as_u64() else {
        return Err(err(expected));
    };
    if x > u8::MAX as u64 {
        return Err(napi::Error::from_reason("byte out of range"));
    }
    Ok(x as u8)
}

fn json_u32(it: &Json, expected: &str) -> Result<u32> {
    let Some(x) = it.as_u64() else {
        return Err(err(expected));
    };
    if x > u32::MAX as u64 {
        return Err(napi::Error::from_reason("u32 out of range"));
    }
    Ok(x as u32)
}

fn json_i32(it: &Json) -> Result<i32> {
    let Some(x) = it.as_i64() else {
        return Err(err("i32[]"));
    };
    if x < i32::MIN as i64 || x > i32::MAX as i64 {
        return Err(napi::Error::from_reason("i32 out of range"));
    }
    Ok(x as i32)
}

fn json_f32(it: &Json, expected: &str) -> Result<f32> {
    let Some(x) = it.as_f64() else {
        return Err(err(expected));
    };
    Ok(x as f32)
}

fn json_u32_or_str(it: Json, expected: &str) -> Result<u32> {
    match it {
        Json::String(s) => parse_u32_str(&s),
        Json::Number(_) => json_u32(&it, expected),
        _ => Err(err(expected)),
    }
}

fn json_u64_or_str(it: Json, expected: &str) -> Result<u64> {
    match it {
        Json::String(s) => parse_u64_str(&s),
        Json::Number(n) => n.as_u64().ok_or_else(|| err(expected)),
        _ => Err(err(expected)),
    }
}

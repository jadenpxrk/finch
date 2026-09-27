use super::*;

/// Convert Python object to finch Value (best-effort type inference)
pub fn py_to_value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(Value::Bool(b));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Value::I64(i));
    }
    if let Ok(u) = obj.extract::<u64>() {
        return Ok(Value::U64(u));
    }
    // A Python int must not fall through to f64, which rounds it beyond 2^53.
    if obj.is_instance_of::<pyo3::types::PyLong>() {
        return Err(pyo3::exceptions::PyOverflowError::new_err(
            "int out of range for int64 and uint64",
        )
        .into());
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(Value::F64(f));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(Value::String(s));
    }
    if let Ok(b) = obj.downcast::<PyBytes>() {
        return Ok(Value::Bytes(b.as_bytes().to_vec()));
    }
    if let Ok(b) = obj.downcast::<PyByteArray>() {
        return Ok(Value::Bytes(b.to_vec()));
    }
    if let Ok(list) = obj.extract::<Vec<Vec<u8>>>() {
        return Ok(Value::ArrayBinary(list));
    }
    if let Ok(list) = obj.extract::<Vec<String>>() {
        return Ok(Value::ArrayString(list));
    }
    if let Ok(list) = obj.extract::<Vec<bool>>() {
        return Ok(Value::ArrayBool(list));
    }
    if let Ok(list) = obj.extract::<Vec<i64>>() {
        return Ok(Value::ArrayI64(list));
    }
    if let Ok(list) = obj.extract::<Vec<u64>>() {
        return Ok(Value::ArrayU64(list));
    }
    if let Ok(list) = obj.extract::<Vec<i32>>() {
        return Ok(Value::ArrayI32(list));
    }
    if let Ok(list) = obj.extract::<Vec<f32>>() {
        return Ok(Value::VecF32(list));
    }
    if let Ok(list) = obj.extract::<Vec<f64>>() {
        return Ok(Value::VecF64(list));
    }
    // Sparse vector: dict[int, float] → SparseF32
    if let Ok(dict) = obj.downcast::<PyDict>() {
        let (indices, values) = sparse_from_py_dict(dict, false)?;
        return Ok(Value::SparseF32 { indices, values });
    }

    Err(pyo3::exceptions::PyValueError::new_err(format!(
        "cannot convert Python type '{}' to finch Value",
        obj.get_type().name()?
    ))
    .into())
}

/// Convert Python object to finch Value using an explicit schema dtype (reference parity).
pub fn py_to_value_typed(obj: &Bound<'_, PyAny>, dt: finch_types::DataType) -> PyResult<Value> {
    use finch_types::DataType as D;
    use finch_types::Value as V;

    if obj.is_none() {
        return Ok(V::Null);
    }

    match dt {
        D::Bool => Ok(V::Bool(obj.extract::<bool>()?)),
        D::String => Ok(V::String(obj.extract::<String>()?)),
        D::Bytes | D::Binary => py_to_bytes(obj),
        D::Int8 => Ok(V::I8(py_to_int_in_range(obj, "int8")?)),
        D::Int16 => Ok(V::I16(py_to_int_in_range(obj, "int16")?)),
        D::Int32 => Ok(V::I32(py_to_int_in_range(obj, "int32")?)),
        D::Int64 => Ok(V::I64(obj.extract::<i64>()?)),
        D::Uint8 => Ok(V::U8(py_to_int_in_range(obj, "uint8")?)),
        D::Uint16 => Ok(V::U16(py_to_int_in_range(obj, "uint16")?)),
        D::Uint32 => py_to_u32(obj),
        D::Uint64 => py_to_u64(obj),
        D::Float16 => Ok(V::F16(f16::from_f64(obj.extract::<f64>()?))),
        D::Float32 => Ok(V::F32(obj.extract::<f64>()? as f32)),
        D::Float64 => Ok(V::F64(obj.extract::<f64>()?)),
        D::VectorFp32
        | D::VectorFp64
        | D::VectorFp16
        | D::VectorInt8
        | D::VectorInt16
        | D::VectorInt4
        | D::VectorBinary32
        | D::VectorBinary64 => py_to_dense_vector(obj, dt),
        D::SparseFp32 | D::SparseFp16 => {
            let dict = obj.downcast::<PyDict>()?;
            let (indices, values) = sparse_from_py_dict(dict, false)?;
            if matches!(dt, D::SparseFp16) {
                let values_f16: Vec<f16> = values.into_iter().map(f16::from_f32).collect();
                Ok(V::SparseF16 {
                    indices,
                    values: values_f16,
                })
            } else {
                Ok(V::SparseF32 { indices, values })
            }
        }
        D::ArrayBinary
        | D::ArrayString
        | D::ArrayBool
        | D::ArrayInt32
        | D::ArrayInt64
        | D::ArrayUint32
        | D::ArrayUint64
        | D::ArrayFp32
        | D::ArrayFp64 => py_to_array(obj, dt),

        // Fallback to inference for any other types.
        _ => py_to_value(obj),
    }
}

fn py_to_int_in_range<T: TryFrom<i64>>(obj: &Bound<'_, PyAny>, type_name: &str) -> PyResult<T> {
    let i: i64 = obj.extract()?;
    T::try_from(i).map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(format!("{type_name} out of range")).into()
    })
}

fn py_to_bytes(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if let Ok(b) = obj.downcast::<PyBytes>() {
        return Ok(Value::Bytes(b.as_bytes().to_vec()));
    }
    if let Ok(b) = obj.downcast::<PyByteArray>() {
        return Ok(Value::Bytes(b.to_vec()));
    }
    Err(pyo3::exceptions::PyTypeError::new_err("expected bytes").into())
}

fn py_to_u32(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if let Ok(u) = obj.extract::<u64>() {
        let v: u32 = u
            .try_into()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("uint32 out of range"))?;
        return Ok(Value::U32(v));
    }
    let i: i64 = obj.extract()?;
    if i < 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("uint32 out of range").into());
    }
    let v: u32 = (i as u64)
        .try_into()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("uint32 out of range"))?;
    Ok(Value::U32(v))
}

fn py_to_u64(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if let Ok(u) = obj.extract::<u64>() {
        return Ok(Value::U64(u));
    }
    let i: i64 = obj.extract()?;
    if i < 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("uint64 out of range").into());
    }
    Ok(Value::U64(i as u64))
}

fn py_to_dense_vector(obj: &Bound<'_, PyAny>, dt: finch_types::DataType) -> PyResult<Value> {
    use finch_types::DataType as D;
    use finch_types::Value as V;

    match dt {
        D::VectorFp32 => Ok(V::VecF32(obj.extract::<Vec<f32>>()?)),
        D::VectorFp64 => Ok(V::VecF64(obj.extract::<Vec<f64>>()?)),
        D::VectorFp16 => {
            // reference parity: fp16 vectors are stored with fp16 rounding.
            // Store as VecF32 to keep the rest of finch's vector pipeline slice-based.
            let v: Vec<f32> = obj.extract::<Vec<f32>>()?;
            let out: Vec<f32> = v.into_iter().map(|x| f16::from_f32(x).to_f32()).collect();
            Ok(V::VecF32(out))
        }
        D::VectorInt8 | D::VectorInt4 => {
            if let Ok(v) = obj.extract::<Vec<i8>>() {
                Ok(V::VecI8(v))
            } else {
                Ok(V::VecF32(obj.extract::<Vec<f32>>()?))
            }
        }
        D::VectorInt16 => {
            if let Ok(v) = obj.extract::<Vec<i16>>() {
                Ok(V::VecI16(v))
            } else {
                Ok(V::VecF32(obj.extract::<Vec<f32>>()?))
            }
        }
        D::VectorBinary32 => Ok(V::VecU32(obj.extract::<Vec<u32>>()?)),
        D::VectorBinary64 => Ok(V::VecU64(obj.extract::<Vec<u64>>()?)),
        _ => py_to_value(obj),
    }
}

fn py_to_array(obj: &Bound<'_, PyAny>, dt: finch_types::DataType) -> PyResult<Value> {
    use finch_types::DataType as D;
    use finch_types::Value as V;

    match dt {
        D::ArrayBinary => Ok(V::ArrayBinary(obj.extract::<Vec<Vec<u8>>>()?)),
        D::ArrayString => Ok(V::ArrayString(obj.extract::<Vec<String>>()?)),
        D::ArrayBool => Ok(V::ArrayBool(obj.extract::<Vec<bool>>()?)),
        D::ArrayInt32 => {
            let items: Vec<i64> = obj.extract()?;
            let mut out = Vec::with_capacity(items.len());
            for i in items {
                let v: i32 = i
                    .try_into()
                    .map_err(|_| pyo3::exceptions::PyValueError::new_err("int32 out of range"))?;
                out.push(v);
            }
            Ok(V::ArrayI32(out))
        }
        D::ArrayInt64 => Ok(V::ArrayI64(obj.extract::<Vec<i64>>()?)),
        D::ArrayUint32 => Ok(V::ArrayU32(obj.extract::<Vec<u32>>()?)),
        D::ArrayUint64 => Ok(V::ArrayU64(obj.extract::<Vec<u64>>()?)),
        D::ArrayFp32 => Ok(V::ArrayF32(obj.extract::<Vec<f32>>()?)),
        D::ArrayFp64 => Ok(V::ArrayF64(obj.extract::<Vec<f64>>()?)),
        _ => py_to_value(obj),
    }
}

/// Parses `dict[int, float]` into index-sorted sparse parts, optionally rounding values through fp16.
pub(crate) fn sparse_from_py_dict(
    dict: &Bound<'_, PyDict>,
    round_fp16: bool,
) -> PyResult<(Vec<u32>, Vec<f32>)> {
    let mut pairs: Vec<(u32, f32)> = Vec::new();
    for (k, v) in dict.iter() {
        let idx: u32 = k.extract()?;
        let mut val: f32 = v.extract()?;
        if round_fp16 {
            val = f16::from_f32(val).to_f32();
        }
        pairs.push((idx, val));
    }
    pairs.sort_by_key(|(i, _)| *i);
    Ok(pairs.into_iter().unzip())
}

/// Convert finch Value to Python object
pub fn value_to_py(py: Python<'_>, v: &Value) -> PyResult<PyObject> {
    match v {
        Value::Null => Ok(py.None()),
        Value::Bool(b) => Ok(b.into_py(py)),
        Value::I8(x) => Ok((*x as i64).into_py(py)),
        Value::I16(x) => Ok((*x as i64).into_py(py)),
        Value::I32(x) => Ok((*x as i64).into_py(py)),
        Value::I64(x) => Ok(x.into_py(py)),
        Value::U8(x) => Ok((*x as u64).into_py(py)),
        Value::U16(x) => Ok((*x as u64).into_py(py)),
        Value::U32(x) => Ok((*x as u64).into_py(py)),
        Value::U64(x) => Ok(x.into_py(py)),
        Value::F16(x) => Ok((x.to_f32() as f64).into_py(py)),
        Value::F32(x) => Ok((*x as f64).into_py(py)),
        Value::F64(x) => Ok(x.into_py(py)),
        Value::String(s) => Ok(s.clone().into_py(py)),
        Value::Bytes(b) => Ok(PyBytes::new_bound(py, b.as_slice()).into()),
        Value::VecF32(v) => Ok(v.clone().into_py(py)),
        Value::VecF16(v) => Ok(v
            .iter()
            .map(|x| x.to_f32() as f64)
            .collect::<Vec<_>>()
            .into_py(py)),
        Value::VecF64(v) => Ok(v.clone().into_py(py)),
        Value::VecU32(v) => Ok(v.clone().into_py(py)),
        Value::VecU64(v) => Ok(v.clone().into_py(py)),
        Value::VecBool(v) => Ok(v.clone().into_py(py)),
        Value::VecI8(v) => Ok(v.iter().map(|x| *x as i64).collect::<Vec<_>>().into_py(py)),
        Value::VecI16(v) => Ok(v.iter().map(|x| *x as i64).collect::<Vec<_>>().into_py(py)),
        Value::VecI32(v) => Ok(v.clone().into_py(py)),
        Value::VecI64(v) => Ok(v.clone().into_py(py)),
        Value::VecString(v) => Ok(v.clone().into_py(py)),
        Value::SparseF32 { indices, values } => {
            let d = PyDict::new_bound(py);
            for (i, v) in indices.iter().zip(values.iter()) {
                d.set_item(i, v)?;
            }
            Ok(d.into())
        }
        Value::SparseF16 { indices, values } => {
            let d = PyDict::new_bound(py);
            for (i, v) in indices.iter().zip(values.iter()) {
                d.set_item(i, v.to_f32())?;
            }
            Ok(d.into())
        }
        Value::ArrayBinary(items) => {
            let out = PyList::empty_bound(py);
            for b in items {
                out.append(PyBytes::new_bound(py, b.as_slice()))?;
            }
            Ok(out.into())
        }
        Value::ArrayBool(v) => Ok(v.clone().into_py(py)),
        Value::ArrayI32(v) => Ok(v.clone().into_py(py)),
        Value::ArrayI64(v) => Ok(v.clone().into_py(py)),
        Value::ArrayU32(v) => Ok(v.clone().into_py(py)),
        Value::ArrayU64(v) => Ok(v.clone().into_py(py)),
        Value::ArrayF32(v) => Ok(v.clone().into_py(py)),
        Value::ArrayF64(v) => Ok(v.clone().into_py(py)),
        Value::ArrayString(v) => Ok(v.clone().into_py(py)),
    }
}

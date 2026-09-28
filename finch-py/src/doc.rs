//! Python Doc class

use crate::schema::{PyFieldSchema, PyQueryParam, QueryParamState};

// Pickled `PyVectorQuery` fields, in `__getstate__` order.
type VectorQueryState = (
    String,
    usize,
    String,
    bool,
    bool,
    Option<Vec<String>>,
    Vec<f32>,
    Vec<u32>,
    Vec<u64>,
    Vec<u32>,
    Vec<f32>,
    QueryParamState,
);
use crate::types::PyDataType;
use crate::PyResult;
use finch_types::{Doc, Operator, Value};
use half::f16;
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyList};
use std::collections::HashMap;

#[pyclass(module = "finch._finch")]
#[derive(Clone)]
pub struct PyDoc {
    pub inner: Doc,
}

fn value_to_pickle(py: Python<'_>, v: &Value) -> PyResult<PyObject> {
    // Return a typed representation: (tag: str, payload: Any)
    // so __setstate__ can reconstruct the exact Value variant.
    match v {
        Value::Null => Ok(("null", py.None()).into_py(py)),
        Value::Bool(b) => Ok(("bool", *b).into_py(py)),
        Value::I8(x) => Ok(("i8", *x as i64).into_py(py)),
        Value::I16(x) => Ok(("i16", *x as i64).into_py(py)),
        Value::I32(x) => Ok(("i32", *x as i64).into_py(py)),
        Value::I64(x) => Ok(("i64", *x).into_py(py)),
        Value::U8(x) => Ok(("u8", *x as u64).into_py(py)),
        Value::U16(x) => Ok(("u16", *x as u64).into_py(py)),
        Value::U32(x) => Ok(("u32", *x as u64).into_py(py)),
        Value::U64(x) => Ok(("u64", *x).into_py(py)),
        Value::F16(x) => Ok(("f16", x.to_f32() as f64).into_py(py)),
        Value::F32(x) => Ok(("f32", *x as f64).into_py(py)),
        Value::F64(x) => Ok(("f64", *x).into_py(py)),
        Value::String(s) => Ok(("string", s.clone()).into_py(py)),
        Value::Bytes(b) => Ok(("bytes", PyBytes::new_bound(py, b.as_slice())).into_py(py)),

        Value::VecBool(v) => Ok(("vec_bool", v.clone()).into_py(py)),
        Value::VecI8(v) => {
            Ok(("vec_i8", v.iter().map(|x| *x as i64).collect::<Vec<_>>()).into_py(py))
        }
        Value::VecI16(v) => {
            Ok(("vec_i16", v.iter().map(|x| *x as i64).collect::<Vec<_>>()).into_py(py))
        }
        Value::VecI32(v) => Ok(("vec_i32", v.clone()).into_py(py)),
        Value::VecI64(v) => Ok(("vec_i64", v.clone()).into_py(py)),
        Value::VecU32(v) => {
            Ok(("vec_u32", v.iter().map(|x| *x as u64).collect::<Vec<_>>()).into_py(py))
        }
        Value::VecU64(v) => Ok(("vec_u64", v.clone()).into_py(py)),
        Value::VecF16(v) => Ok((
            "vec_f16",
            v.iter().map(|x| x.to_f32() as f64).collect::<Vec<_>>(),
        )
            .into_py(py)),
        Value::VecF32(v) => Ok(("vec_f32", v.clone()).into_py(py)),
        Value::VecF64(v) => Ok(("vec_f64", v.clone()).into_py(py)),
        Value::VecString(v) => Ok(("vec_string", v.clone()).into_py(py)),

        Value::SparseF32 { indices, values } => {
            Ok(("sparse_f32", (indices.clone(), values.clone())).into_py(py))
        }
        Value::SparseF16 { indices, values } => Ok((
            "sparse_f16",
            (
                indices.clone(),
                values.iter().map(|x| x.to_f32() as f64).collect::<Vec<_>>(),
            ),
        )
            .into_py(py)),

        Value::ArrayBinary(items) => Ok((
            "array_binary",
            items
                .iter()
                .map(|b| PyBytes::new_bound(py, b.as_slice()).into_py(py))
                .collect::<Vec<_>>(),
        )
            .into_py(py)),
        Value::ArrayI32(v) => Ok(("array_i32", v.clone()).into_py(py)),
        Value::ArrayI64(v) => Ok(("array_i64", v.clone()).into_py(py)),
        Value::ArrayU32(v) => Ok(("array_u32", v.clone()).into_py(py)),
        Value::ArrayU64(v) => Ok(("array_u64", v.clone()).into_py(py)),
        Value::ArrayBool(v) => Ok(("array_bool", v.clone()).into_py(py)),
        Value::ArrayF32(v) => Ok(("array_f32", v.clone()).into_py(py)),
        Value::ArrayF64(v) => Ok(("array_f64", v.clone()).into_py(py)),
        Value::ArrayString(v) => Ok(("array_string", v.clone()).into_py(py)),
    }
}

fn pickle_to_value(state: &Bound<'_, PyAny>) -> PyResult<Value> {
    let (tag, payload): (String, Bound<'_, PyAny>) = state.extract()?;
    let tag = tag.as_str();
    if tag.starts_with("vec_") {
        return pickle_vec_to_value(tag, &payload);
    }
    if tag.starts_with("sparse_") || tag.starts_with("array_") {
        return pickle_collection_to_value(tag, &payload);
    }
    match tag {
        "null" => Ok(Value::Null),
        "bool" => Ok(Value::Bool(payload.extract::<bool>()?)),
        "i8" => Ok(Value::I8(payload.extract::<i64>()? as i8)),
        "i16" => Ok(Value::I16(payload.extract::<i64>()? as i16)),
        "i32" => Ok(Value::I32(payload.extract::<i64>()? as i32)),
        "i64" => Ok(Value::I64(payload.extract::<i64>()?)),
        "u8" => Ok(Value::U8(payload.extract::<u64>()? as u8)),
        "u16" => Ok(Value::U16(payload.extract::<u64>()? as u16)),
        "u32" => Ok(Value::U32(payload.extract::<u64>()? as u32)),
        "u64" => Ok(Value::U64(payload.extract::<u64>()?)),
        "f16" => Ok(Value::F16(f16::from_f32(payload.extract::<f64>()? as f32))),
        "f32" => Ok(Value::F32(payload.extract::<f64>()? as f32)),
        "f64" => Ok(Value::F64(payload.extract::<f64>()?)),
        "string" => Ok(Value::String(payload.extract::<String>()?)),
        "bytes" => {
            if let Ok(b) = payload.downcast::<PyBytes>() {
                return Ok(Value::Bytes(b.as_bytes().to_vec()));
            }
            if let Ok(b) = payload.downcast::<PyByteArray>() {
                return Ok(Value::Bytes(b.to_vec()));
            }
            Err(pyo3::exceptions::PyTypeError::new_err("expected bytes").into())
        }
        _ => Err(unknown_pickle_tag(tag)),
    }
}

fn pickle_vec_to_value(tag: &str, payload: &Bound<'_, PyAny>) -> PyResult<Value> {
    match tag {
        "vec_bool" => Ok(Value::VecBool(payload.extract::<Vec<bool>>()?)),
        "vec_i8" => Ok(Value::VecI8(
            payload
                .extract::<Vec<i64>>()?
                .into_iter()
                .map(|x| x as i8)
                .collect(),
        )),
        "vec_i16" => Ok(Value::VecI16(
            payload
                .extract::<Vec<i64>>()?
                .into_iter()
                .map(|x| x as i16)
                .collect(),
        )),
        "vec_i32" => Ok(Value::VecI32(payload.extract::<Vec<i32>>()?)),
        "vec_i64" => Ok(Value::VecI64(payload.extract::<Vec<i64>>()?)),
        "vec_u32" => Ok(Value::VecU32(
            payload
                .extract::<Vec<u64>>()?
                .into_iter()
                .map(|x| x as u32)
                .collect(),
        )),
        "vec_u64" => Ok(Value::VecU64(payload.extract::<Vec<u64>>()?)),
        "vec_f16" => Ok(Value::VecF16(
            payload
                .extract::<Vec<f64>>()?
                .into_iter()
                .map(|x| f16::from_f32(x as f32))
                .collect(),
        )),
        "vec_f32" => Ok(Value::VecF32(payload.extract::<Vec<f32>>()?)),
        "vec_f64" => Ok(Value::VecF64(payload.extract::<Vec<f64>>()?)),
        "vec_string" => Ok(Value::VecString(payload.extract::<Vec<String>>()?)),
        _ => Err(unknown_pickle_tag(tag)),
    }
}

fn pickle_collection_to_value(tag: &str, payload: &Bound<'_, PyAny>) -> PyResult<Value> {
    match tag {
        "sparse_f32" => {
            let (indices, values): (Vec<u32>, Vec<f32>) = payload.extract()?;
            Ok(Value::SparseF32 { indices, values })
        }
        "sparse_f16" => {
            let (indices, values): (Vec<u32>, Vec<f64>) = payload.extract()?;
            let v16: Vec<f16> = values
                .into_iter()
                .map(|x| f16::from_f32(x as f32))
                .collect();
            Ok(Value::SparseF16 {
                indices,
                values: v16,
            })
        }
        "array_binary" => pickle_array_binary(payload),
        "array_i32" => Ok(Value::ArrayI32(payload.extract::<Vec<i32>>()?)),
        "array_i64" => Ok(Value::ArrayI64(payload.extract::<Vec<i64>>()?)),
        "array_u32" => Ok(Value::ArrayU32(payload.extract::<Vec<u32>>()?)),
        "array_u64" => Ok(Value::ArrayU64(payload.extract::<Vec<u64>>()?)),
        "array_bool" => Ok(Value::ArrayBool(payload.extract::<Vec<bool>>()?)),
        "array_f32" => Ok(Value::ArrayF32(payload.extract::<Vec<f32>>()?)),
        "array_f64" => Ok(Value::ArrayF64(payload.extract::<Vec<f64>>()?)),
        "array_string" => Ok(Value::ArrayString(payload.extract::<Vec<String>>()?)),
        _ => Err(unknown_pickle_tag(tag)),
    }
}

fn pickle_array_binary(payload: &Bound<'_, PyAny>) -> PyResult<Value> {
    let items: Vec<Bound<'_, PyAny>> = payload.extract()?;
    let mut out: Vec<Vec<u8>> = Vec::with_capacity(items.len());
    for item in items {
        if let Ok(b) = item.downcast::<PyBytes>() {
            out.push(b.as_bytes().to_vec());
            continue;
        }
        if let Ok(b) = item.downcast::<PyByteArray>() {
            out.push(b.to_vec());
            continue;
        }
        return Err(
            pyo3::exceptions::PyTypeError::new_err("expected bytes in array_binary").into(),
        );
    }
    Ok(Value::ArrayBinary(out))
}

fn unknown_pickle_tag(tag: &str) -> crate::BindingError {
    pyo3::exceptions::PyValueError::new_err(format!("unknown pickled Value tag: {tag}")).into()
}

fn i8_query_vector(raw: Vec<i64>) -> PyResult<Vec<i8>> {
    let mut out: Vec<i8> = Vec::with_capacity(raw.len());
    for x in raw {
        out.push(
            i8::try_from(x)
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("int8 out of range"))?,
        );
    }
    Ok(out)
}

#[pymethods]
impl PyDoc {
    #[new]
    #[pyo3(signature = (pk="", score=0.0))]
    fn new(pk: &str, score: f32) -> Self {
        PyDoc {
            inner: Doc {
                pk: pk.to_string(),
                score,
                doc_id: 0,
                op: Operator::Insert,
                fields: HashMap::new(),
            },
        }
    }

    // reference-style API: pk()/set_pk(...)
    fn set_pk(&mut self, pk: &str) {
        self.inner.pk = pk.to_string();
    }

    fn pk(&self) -> &str {
        &self.inner.pk
    }

    fn set_score(&mut self, score: f32) {
        self.inner.score = score;
    }

    fn score(&self) -> f32 {
        self.inner.score
    }

    fn doc_id(&self) -> u64 {
        self.inner.doc_id
    }

    fn has_field(&self, name: &str) -> bool {
        self.inner.fields.contains_key(name)
    }

    /// Sets a field without a schema, inferring its type from the Python value.
    ///
    /// None is null; bool, int (int64, else uint64), float, str, and bytes map to their scalar
    /// types; dict[int, float] is a sparse fp32 vector. A list must be non-empty and holds one
    /// kind: bools, strs, bytes, ints (array int64, else array uint64), or floats mixed with ints,
    /// which becomes a float64 vector and rejects any int beyond 2**53. Use `set_any` to target
    /// another type, such as a float32 vector.
    fn set_field(&mut self, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let v = py_to_value(value)?;
        self.inner.fields.insert(name.to_string(), v);
        Ok(())
    }

    fn set_any(
        &mut self,
        name: &str,
        schema: &PyFieldSchema,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let dt = schema.inner.data_type;
        if value.is_none() {
            if schema.inner.nullable {
                self.inner.fields.insert(name.to_string(), Value::Null);
                return Ok(true);
            }
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "doc validate failed: field[{}] is configured not nullable, but doc's field value is empty",
                name
            )).into());
        }
        let v = py_to_value_typed(value, dt)?;
        self.inner.fields.insert(name.to_string(), v);
        Ok(true)
    }

    fn set_vec_u32(&mut self, name: &str, value: Vec<u32>) -> PyResult<()> {
        self.inner
            .fields
            .insert(name.to_string(), Value::VecU32(value));
        Ok(())
    }

    fn set_vec_u64(&mut self, name: &str, value: Vec<u64>) -> PyResult<()> {
        self.inner
            .fields
            .insert(name.to_string(), Value::VecU64(value));
        Ok(())
    }

    fn get_field(&self, py: Python<'_>, name: &str) -> PyResult<PyObject> {
        match self.inner.fields.get(name) {
            None => Ok(py.None()),
            Some(v) => value_to_py(py, v),
        }
    }

    fn get_any(&self, py: Python<'_>, name: &str, _dt: PyDataType) -> PyResult<PyObject> {
        // Finch already stores typed values; for reference parity we accept the dtype
        // parameter but primarily use it to match the signature.
        self.get_field(py, name)
    }

    fn fields_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let d = PyDict::new_bound(py);
        for (k, v) in &self.inner.fields {
            d.set_item(k, value_to_py(py, v)?)?;
        }
        Ok(d.into())
    }

    fn __repr__(&self) -> String {
        format!("Doc(pk='{}', score={:.4})", self.inner.pk, self.inner.score)
    }

    // ── Pickle support (reference parity) ─────────────────────────────────────

    fn __getstate__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let d = PyDict::new_bound(py);
        for (k, v) in &self.inner.fields {
            d.set_item(k, value_to_pickle(py, v)?)?;
        }
        let state = (
            self.inner.pk.clone(),
            self.inner.score,
            self.inner.doc_id,
            self.inner.op as u32,
            d,
        );

        let pickle = PyModule::import_bound(py, "pickle")?;
        let dumps = pickle.getattr("dumps")?;
        Ok(dumps.call1((state,))?.extract()?)
    }

    fn __setstate__(&mut self, py: Python<'_>, state: &Bound<'_, PyAny>) -> PyResult<()> {
        let pickle = PyModule::import_bound(py, "pickle")?;
        let loads = pickle.getattr("loads")?;
        let decoded = loads.call1((state,))?;
        let (pk, score, doc_id, op, fields): (String, f32, u64, u32, Bound<'_, PyDict>) =
            decoded.extract()?;

        self.inner.pk = pk;
        self.inner.score = score;
        self.inner.doc_id = doc_id;
        self.inner.op = match op {
            0 => Operator::Insert,
            1 => Operator::Upsert,
            2 => Operator::Update,
            3 => Operator::Delete,
            _ => Operator::Insert,
        };

        let mut out: HashMap<String, Value> = HashMap::new();
        for (k, v) in fields.iter() {
            let name: String = k.extract()?;
            out.insert(name, pickle_to_value(&v)?);
        }
        self.inner.fields = out;
        Ok(())
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let state = self.__getstate__(py)?;
        let ty = py.get_type_bound::<PyDoc>();
        let args = pyo3::types::PyTuple::empty_bound(py);
        Ok((ty, args, state).into_py(py))
    }
}

mod conversion;

use conversion::sparse_from_py_dict;
pub use conversion::{py_to_value, py_to_value_typed, value_to_py};

#[pyclass(module = "finch._finch")]
pub struct PyGroupResult {
    pub inner: finch_types::GroupResult,
}

#[pymethods]
impl PyGroupResult {
    #[getter]
    fn group_value(&self, py: Python<'_>) -> PyResult<PyObject> {
        value_to_py(py, &self.inner.group_value)
    }

    #[getter]
    fn docs(&self) -> Vec<PyDoc> {
        self.inner
            .docs
            .iter()
            .map(|d| PyDoc { inner: d.clone() })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("GroupResult(count={})", self.inner.docs.len())
    }
}

#[pyclass(module = "finch._finch")]
pub struct PyVectorQuery {
    pub inner: finch_types::VectorQuery,
    // Extra caches for reference-like set_vector/get_vector round-trips.
    dense_i8: Option<Vec<i8>>,
    dense_f64: Option<Vec<f64>>,
}

#[pymethods]
impl PyVectorQuery {
    #[new]
    #[pyo3(signature = (field_name="", query_vector=None, topk=10, filter=None))]
    fn new(
        field_name: &str,
        query_vector: Option<Vec<f32>>,
        topk: usize,
        filter: Option<&str>,
    ) -> Self {
        PyVectorQuery {
            inner: finch_types::VectorQuery {
                topk,
                field_name: field_name.to_string(),
                id: None,
                query_vector: query_vector.unwrap_or_default(),
                query_vector_u32: Vec::new(),
                query_vector_u64: Vec::new(),
                sparse_indices: Vec::new(),
                sparse_values: Vec::new(),
                filter: filter.map(|s| s.to_string()),
                include_vector: false,
                include_doc_id: false,
                output_fields: None,
                query_params: finch_types::QueryParams::default(),
            },
            dense_i8: None,
            dense_f64: None,
        }
    }

    #[staticmethod]
    #[pyo3(signature = (field_name, query_vector, topk, filter=None))]
    fn binary32(
        field_name: &str,
        query_vector: Vec<u32>,
        topk: usize,
        filter: Option<&str>,
    ) -> Self {
        PyVectorQuery {
            inner: finch_types::VectorQuery {
                topk,
                field_name: field_name.to_string(),
                id: None,
                query_vector: Vec::new(),
                query_vector_u32: query_vector,
                query_vector_u64: Vec::new(),
                sparse_indices: Vec::new(),
                sparse_values: Vec::new(),
                filter: filter.map(|s| s.to_string()),
                include_vector: false,
                include_doc_id: false,
                output_fields: None,
                query_params: finch_types::QueryParams::default(),
            },
            dense_i8: None,
            dense_f64: None,
        }
    }

    #[staticmethod]
    #[pyo3(signature = (field_name, query_vector, topk, filter=None))]
    fn binary64(
        field_name: &str,
        query_vector: Vec<u64>,
        topk: usize,
        filter: Option<&str>,
    ) -> Self {
        PyVectorQuery {
            inner: finch_types::VectorQuery {
                topk,
                field_name: field_name.to_string(),
                id: None,
                query_vector: Vec::new(),
                query_vector_u32: Vec::new(),
                query_vector_u64: query_vector,
                sparse_indices: Vec::new(),
                sparse_values: Vec::new(),
                filter: filter.map(|s| s.to_string()),
                include_vector: false,
                include_doc_id: false,
                output_fields: None,
                query_params: finch_types::QueryParams::default(),
            },
            dense_i8: None,
            dense_f64: None,
        }
    }

    #[getter]
    fn topk(&self) -> usize {
        self.inner.topk
    }

    #[setter]
    fn set_topk(&mut self, topk: usize) {
        self.inner.topk = topk;
    }

    #[getter]
    fn field_name(&self) -> &str {
        &self.inner.field_name
    }

    #[setter]
    fn set_field_name(&mut self, field_name: &str) {
        self.inner.field_name = field_name.to_string();
    }

    #[getter]
    fn filter(&self) -> Option<&str> {
        self.inner.filter.as_deref()
    }

    #[setter]
    fn set_filter(&mut self, filter: Option<&str>) {
        self.inner.filter = filter.map(|s| s.to_string());
    }

    fn set_vector(&mut self, schema: &PyFieldSchema, value: &Bound<'_, PyAny>) -> PyResult<()> {
        use finch_types::DataType as D;

        self.dense_i8 = None;
        self.dense_f64 = None;
        self.inner.query_vector_u32.clear();
        self.inner.query_vector_u64.clear();
        self.inner.sparse_indices.clear();
        self.inner.sparse_values.clear();

        let dt = schema.inner.data_type;
        if !dt.is_vector() {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Unsupported vector field type for field: {}",
                schema.inner.name
            ))
            .into());
        }

        let v = if value.hasattr("tolist")? {
            value.call_method0("tolist")?
        } else {
            value.clone()
        };

        match dt {
            D::SparseFp32 | D::SparseFp16 => {
                self.inner.query_vector.clear();
                let dict = v.downcast::<PyDict>()?;
                let (indices, values) = sparse_from_py_dict(dict, matches!(dt, D::SparseFp16))?;
                self.inner.sparse_indices = indices;
                self.inner.sparse_values = values;
            }
            D::VectorFp16 => {
                let raw: Vec<f64> = v.extract()?;
                self.inner.query_vector = raw
                    .into_iter()
                    .map(|x| f16::from_f32(x as f32).to_f32())
                    .collect();
            }
            D::VectorFp32 => {
                let raw: Vec<f64> = v.extract()?;
                self.inner.query_vector = raw.into_iter().map(|x| x as f32).collect();
            }
            D::VectorFp64 => {
                let raw: Vec<f64> = v.extract()?;
                self.dense_f64 = Some(raw.clone());
                self.inner.query_vector = raw.into_iter().map(|x| x as f32).collect();
            }
            D::VectorInt8 => {
                let out = i8_query_vector(v.extract()?)?;
                self.dense_i8 = Some(out.clone());
                self.inner.query_vector = out.into_iter().map(|x| x as f32).collect();
            }
            _ => {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "Unsupported dense vector type for ndarray input: {}",
                    dt as u32
                ))
                .into());
            }
        }
        Ok(())
    }

    fn get_vector(&self, py: Python<'_>, schema: &PyFieldSchema) -> PyResult<PyObject> {
        use finch_types::DataType as D;
        let dt = schema.inner.data_type;
        if !dt.is_vector() {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Unsupported vector field type for field: {}",
                schema.inner.name
            ))
            .into());
        }
        match dt {
            D::SparseFp32 | D::SparseFp16 => {
                let d = PyDict::new_bound(py);
                for (i, v) in self
                    .inner
                    .sparse_indices
                    .iter()
                    .zip(self.inner.sparse_values.iter())
                {
                    d.set_item(i, v)?;
                }
                Ok(d.into())
            }
            D::VectorFp64 => {
                if let Some(v) = &self.dense_f64 {
                    return Ok(v.clone().into_py(py));
                }
                Ok(self
                    .inner
                    .query_vector
                    .iter()
                    .map(|x| *x as f64)
                    .collect::<Vec<_>>()
                    .into_py(py))
            }
            D::VectorInt8 => {
                if let Some(v) = &self.dense_i8 {
                    return Ok(v.iter().map(|x| *x as i64).collect::<Vec<_>>().into_py(py));
                }
                Ok(self
                    .inner
                    .query_vector
                    .iter()
                    .map(|x| *x as i64)
                    .collect::<Vec<_>>()
                    .into_py(py))
            }
            D::VectorFp16 | D::VectorFp32 => Ok(self
                .inner
                .query_vector
                .iter()
                .map(|x| *x as f64)
                .collect::<Vec<_>>()
                .into_py(py)),
            _ => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Unsupported dense vector type for get_vector: {}",
                dt as u32
            ))
            .into()),
        }
    }

    fn set_sparse(&mut self, indices: Vec<u32>, values: Vec<f32>) {
        self.dense_i8 = None;
        self.dense_f64 = None;
        self.inner.sparse_indices = indices;
        self.inner.sparse_values = values;
    }

    fn set_query_vector_u32(&mut self, v: Vec<u32>) {
        self.dense_i8 = None;
        self.dense_f64 = None;
        self.inner.query_vector.clear();
        self.inner.query_vector_u64.clear();
        self.inner.query_vector_u32 = v;
    }

    fn set_query_vector_u64(&mut self, v: Vec<u64>) {
        self.dense_i8 = None;
        self.dense_f64 = None;
        self.inner.query_vector.clear();
        self.inner.query_vector_u32.clear();
        self.inner.query_vector_u64 = v;
    }

    // ── reference-style properties ────────────────────────────────────────────

    #[getter]
    fn include_vector(&self) -> bool {
        self.inner.include_vector
    }

    #[setter(include_vector)]
    fn set_include_vector_prop(&mut self, include: bool) {
        self.inner.include_vector = include;
    }

    fn set_include_vector(&mut self, include: bool) {
        self.inner.include_vector = include;
    }

    #[getter]
    fn include_doc_id(&self) -> bool {
        self.inner.include_doc_id
    }

    #[setter(include_doc_id)]
    fn set_include_doc_id_prop(&mut self, include: bool) {
        self.inner.include_doc_id = include;
    }

    fn set_include_doc_id(&mut self, include: bool) {
        self.inner.include_doc_id = include;
    }

    #[getter]
    fn output_fields(&self) -> Option<Vec<String>> {
        self.inner.output_fields.clone()
    }

    #[setter(output_fields)]
    fn set_output_fields_prop(&mut self, fields: Option<Vec<String>>) {
        self.inner.output_fields = fields;
    }

    fn set_output_fields(&mut self, fields: Vec<String>) {
        self.inner.output_fields = Some(fields);
    }

    fn clear_output_fields(&mut self) {
        self.inner.output_fields = None;
    }

    #[getter]
    fn query_params(&self) -> PyQueryParam {
        PyQueryParam {
            ef: self.inner.query_params.ef,
            n_probe: self.inner.query_params.n_probe,
            concurrency: self.inner.query_params.concurrency,
            bf_pks: self.inner.query_params.bf_pks.clone(),
            radius: self.inner.query_params.radius,
            is_linear: self.inner.query_params.is_linear,
            use_refiner: self.inner.query_params.use_refiner,
            refiner_k: self.inner.query_params.refiner_k,
            refiner_scale_factor: self.inner.query_params.refiner_scale_factor,
        }
    }

    #[setter(query_params)]
    fn set_query_params_prop(&mut self, query_param: &PyQueryParam) {
        self.set_query_param(query_param);
    }

    fn set_query_param(&mut self, query_param: &PyQueryParam) {
        self.inner.query_params.ef = query_param.ef;
        self.inner.query_params.n_probe = query_param.n_probe;
        self.inner.query_params.concurrency = query_param.concurrency;
        self.inner.query_params.bf_pks = query_param.bf_pks.clone();
        self.inner.query_params.radius = query_param.radius;
        self.inner.query_params.is_linear = query_param.is_linear;
        self.inner.query_params.use_refiner = query_param.use_refiner;
        self.inner.query_params.refiner_k = query_param.refiner_k;
        self.inner.query_params.refiner_scale_factor = query_param.refiner_scale_factor;
    }

    // ── Pickle support (reference parity) ─────────────────────────────────────

    fn __getstate__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let qp = &self.inner.query_params;
        let qp_state = (
            qp.ef,
            qp.n_probe,
            qp.concurrency,
            qp.bf_pks.clone(),
            qp.radius,
            qp.is_linear,
            qp.use_refiner,
            qp.refiner_k,
            qp.refiner_scale_factor,
        );

        let filter = self.inner.filter.clone().unwrap_or_default();
        Ok((
            self.inner.field_name.clone(),
            self.inner.topk,
            filter,
            self.inner.include_vector,
            self.inner.include_doc_id,
            self.inner.output_fields.clone(),
            self.inner.query_vector.clone(),
            self.inner.query_vector_u32.clone(),
            self.inner.query_vector_u64.clone(),
            self.inner.sparse_indices.clone(),
            self.inner.sparse_values.clone(),
            qp_state,
        )
            .into_py(py))
    }

    fn __setstate__(&mut self, state: &Bound<'_, PyAny>) -> PyResult<()> {
        let (
            field_name,
            topk,
            filter,
            include_vector,
            include_doc_id,
            output_fields,
            query_vector,
            query_vector_u32,
            query_vector_u64,
            sparse_indices,
            sparse_values,
            qp_state,
        ): VectorQueryState = state.extract()?;

        self.dense_i8 = None;
        self.dense_f64 = None;

        self.inner.field_name = field_name;
        self.inner.topk = topk;
        self.inner.filter = if filter.is_empty() {
            None
        } else {
            Some(filter)
        };
        self.inner.include_vector = include_vector;
        self.inner.include_doc_id = include_doc_id;
        self.inner.output_fields = output_fields;
        self.inner.query_vector = query_vector;
        self.inner.query_vector_u32 = query_vector_u32;
        self.inner.query_vector_u64 = query_vector_u64;
        self.inner.sparse_indices = sparse_indices;
        self.inner.sparse_values = sparse_values;

        self.inner.query_params.ef = qp_state.0;
        self.inner.query_params.n_probe = qp_state.1;
        self.inner.query_params.concurrency = qp_state.2;
        self.inner.query_params.bf_pks = qp_state.3;
        self.inner.query_params.radius = qp_state.4;
        self.inner.query_params.is_linear = qp_state.5;
        self.inner.query_params.use_refiner = qp_state.6;
        self.inner.query_params.refiner_k = qp_state.7;
        self.inner.query_params.refiner_scale_factor = qp_state.8;
        Ok(())
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let state = self.__getstate__(py)?;
        let ty = py.get_type_bound::<PyVectorQuery>();
        let args = pyo3::types::PyTuple::empty_bound(py);
        Ok((ty, args, state).into_py(py))
    }
}

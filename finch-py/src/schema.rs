//! Python schema and index param classes

use crate::types::{PyDataType, PyIndexType, PyMetricType, PyQuantizeType};
use crate::PyResult;
use finch_types::{CollectionSchema, FieldSchema, IndexParams};
use pyo3::prelude::*;
use pyo3::types::PyDict;

// Pickled `PyQueryParam` fields, in `__getstate__` order.
pub(crate) type QueryParamState = (
    Option<u32>,
    Option<u32>,
    Option<usize>,
    Option<Vec<String>>,
    Option<f32>,
    Option<bool>,
    bool,
    Option<u32>,
    Option<f32>,
);

// ── reference-ish option wrappers ────────────────────────────────────────────────

#[pyclass(module = "finch._finch")]
#[derive(Clone, Debug)]
pub struct PyIndexOption {
    concurrency: usize,
}

#[pymethods]
impl PyIndexOption {
    #[new]
    #[pyo3(signature = (concurrency=0))]
    fn new(concurrency: usize) -> Self {
        Self { concurrency }
    }

    #[getter]
    fn concurrency(&self) -> usize {
        self.concurrency
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let d = pyo3::types::PyDict::new_bound(py);
        d.set_item("concurrency", self.concurrency)?;
        Ok(d.into())
    }

    fn __getstate__(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok((self.concurrency,).into_py(py))
    }

    fn __setstate__(&mut self, state: &Bound<'_, PyAny>) -> PyResult<()> {
        let (concurrency,): (usize,) = state.extract()?;
        self.concurrency = concurrency;
        Ok(())
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let state = self.__getstate__(py)?;
        let ty = py.get_type_bound::<PyIndexOption>();
        let args = pyo3::types::PyTuple::empty_bound(py);
        Ok((ty, args, state).into_py(py))
    }
}

impl PyIndexOption {
    pub fn as_concurrency_option(&self) -> Option<usize> {
        if self.concurrency == 0 {
            None
        } else {
            Some(self.concurrency)
        }
    }
}

#[pyclass(module = "finch._finch")]
#[derive(Clone, Debug)]
pub struct PyOptimizeOption {
    concurrency: usize,
}

#[pymethods]
impl PyOptimizeOption {
    #[new]
    #[pyo3(signature = (concurrency=0))]
    fn new(concurrency: usize) -> Self {
        Self { concurrency }
    }

    #[getter]
    fn concurrency(&self) -> usize {
        self.concurrency
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let d = pyo3::types::PyDict::new_bound(py);
        d.set_item("concurrency", self.concurrency)?;
        Ok(d.into())
    }

    fn __getstate__(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok((self.concurrency,).into_py(py))
    }

    fn __setstate__(&mut self, state: &Bound<'_, PyAny>) -> PyResult<()> {
        let (concurrency,): (usize,) = state.extract()?;
        self.concurrency = concurrency;
        Ok(())
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let state = self.__getstate__(py)?;
        let ty = py.get_type_bound::<PyOptimizeOption>();
        let args = pyo3::types::PyTuple::empty_bound(py);
        Ok((ty, args, state).into_py(py))
    }
}

impl PyOptimizeOption {
    pub fn as_concurrency_option(&self) -> Option<usize> {
        if self.concurrency == 0 {
            None
        } else {
            Some(self.concurrency)
        }
    }
}

#[pyclass(module = "finch._finch")]
#[derive(Clone, Debug)]
pub struct PyAddColumnOption {
    concurrency: usize,
}

#[pymethods]
impl PyAddColumnOption {
    #[new]
    #[pyo3(signature = (concurrency=0))]
    fn new(concurrency: usize) -> Self {
        Self { concurrency }
    }

    #[getter]
    fn concurrency(&self) -> usize {
        self.concurrency
    }

    fn __getstate__(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok((self.concurrency,).into_py(py))
    }

    fn __setstate__(&mut self, state: &Bound<'_, PyAny>) -> PyResult<()> {
        let (concurrency,): (usize,) = state.extract()?;
        self.concurrency = concurrency;
        Ok(())
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let state = self.__getstate__(py)?;
        let ty = py.get_type_bound::<PyAddColumnOption>();
        let args = pyo3::types::PyTuple::empty_bound(py);
        Ok((ty, args, state).into_py(py))
    }
}

impl PyAddColumnOption {
    pub fn as_concurrency_option(&self) -> Option<usize> {
        if self.concurrency == 0 {
            None
        } else {
            Some(self.concurrency)
        }
    }
}

#[pyclass(module = "finch._finch")]
#[derive(Clone, Debug)]
pub struct PyAlterColumnOption {
    concurrency: usize,
}

#[pymethods]
impl PyAlterColumnOption {
    #[new]
    #[pyo3(signature = (concurrency=0))]
    fn new(concurrency: usize) -> Self {
        Self { concurrency }
    }

    #[getter]
    fn concurrency(&self) -> usize {
        self.concurrency
    }

    fn __getstate__(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok((self.concurrency,).into_py(py))
    }

    fn __setstate__(&mut self, state: &Bound<'_, PyAny>) -> PyResult<()> {
        let (concurrency,): (usize,) = state.extract()?;
        self.concurrency = concurrency;
        Ok(())
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let state = self.__getstate__(py)?;
        let ty = py.get_type_bound::<PyAlterColumnOption>();
        let args = pyo3::types::PyTuple::empty_bound(py);
        Ok((ty, args, state).into_py(py))
    }
}

impl PyAlterColumnOption {
    pub fn as_concurrency_option(&self) -> Option<usize> {
        if self.concurrency == 0 {
            None
        } else {
            Some(self.concurrency)
        }
    }
}

#[pyclass(module = "finch._finch")]
#[derive(Clone)]
pub struct PyHnswIndexParam {
    pub inner: finch_types::HnswIndexParams,
}

#[pymethods]
impl PyHnswIndexParam {
    #[new]
    #[pyo3(signature = (
        metric=PyMetricType::InnerProduct,
        m=50,
        ef_construction=500,
        quantize=PyQuantizeType::Undefined
    ))]
    fn new(
        metric: PyMetricType,
        m: usize,
        ef_construction: usize,
        quantize: PyQuantizeType,
    ) -> Self {
        PyHnswIndexParam {
            inner: finch_types::HnswIndexParams {
                m,
                ef_construction,
                // reference parity: scaling factor is not exposed in DB HNSW params; it defaults to `m`.
                scaling_factor: m,
                metric: metric.into(),
                quantize: quantize.into(),
                build_concurrency: None,
            },
        }
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let d = PyDict::new_bound(py);
        d.set_item("metric_type", metric_name(self.inner.metric))?;
        d.set_item("m", self.inner.m)?;
        d.set_item("ef_construction", self.inner.ef_construction)?;
        d.set_item("quantize_type", quantize_name(self.inner.quantize))?;
        Ok(d.into())
    }
}

#[pyclass(module = "finch._finch")]
#[derive(Clone)]
pub struct PyIvfIndexParam {
    pub inner: finch_types::IvfIndexParams,
}

#[pymethods]
impl PyIvfIndexParam {
    #[new]
    #[pyo3(signature = (
        metric=PyMetricType::InnerProduct,
        n_list=0,
        n_iters=10,
        use_soar=false,
        quantize=PyQuantizeType::Undefined
    ))]
    fn new(
        metric: PyMetricType,
        n_list: usize,
        n_iters: usize,
        use_soar: bool,
        quantize: PyQuantizeType,
    ) -> Self {
        PyIvfIndexParam {
            inner: finch_types::IvfIndexParams {
                n_list,
                n_iters,
                use_soar,
                l1_index: None,
                metric: metric.into(),
                quantize: quantize.into(),
            },
        }
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let d = PyDict::new_bound(py);
        d.set_item("metric_type", metric_name(self.inner.metric))?;
        d.set_item("n_list", self.inner.n_list)?;
        d.set_item("n_iters", self.inner.n_iters)?;
        d.set_item("use_soar", self.inner.use_soar)?;
        d.set_item("quantize_type", quantize_name(self.inner.quantize))?;
        Ok(d.into())
    }
}

#[pyclass(module = "finch._finch")]
#[derive(Clone)]
pub struct PyFlatIndexParam {
    pub inner: finch_types::FlatIndexParams,
}

#[pymethods]
impl PyFlatIndexParam {
    #[new]
    #[pyo3(signature = (metric=PyMetricType::InnerProduct, quantize=PyQuantizeType::Undefined))]
    fn new(metric: PyMetricType, quantize: PyQuantizeType) -> Self {
        PyFlatIndexParam {
            inner: finch_types::FlatIndexParams {
                metric: metric.into(),
                quantize: quantize.into(),
                column_major: false,
            },
        }
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let d = PyDict::new_bound(py);
        d.set_item("metric_type", metric_name(self.inner.metric))?;
        d.set_item("quantize_type", quantize_name(self.inner.quantize))?;
        Ok(d.into())
    }
}

#[pyclass(module = "finch._finch")]
#[derive(Clone)]
pub struct PyInvertIndexParam {
    pub inner: finch_types::InvertIndexParams,
}

#[pymethods]
impl PyInvertIndexParam {
    #[new]
    #[pyo3(signature = (
        enable_range_optimization=false,
        enable_extended_wildcard=false
    ))]
    fn new(enable_range_optimization: bool, enable_extended_wildcard: bool) -> Self {
        PyInvertIndexParam {
            inner: finch_types::InvertIndexParams {
                enable_range_optimization,
                enable_extended_wildcard,
            },
        }
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let d = PyDict::new_bound(py);
        d.set_item(
            "enable_range_optimization",
            self.inner.enable_range_optimization,
        )?;
        d.set_item(
            "enable_extended_wildcard",
            self.inner.enable_extended_wildcard,
        )?;
        Ok(d.into())
    }

    #[getter]
    fn enable_range_optimization(&self) -> bool {
        self.inner.enable_range_optimization
    }

    #[getter]
    fn enable_extended_wildcard(&self) -> bool {
        self.inner.enable_extended_wildcard
    }
}

fn metric_name(m: finch_types::MetricType) -> &'static str {
    match m {
        finch_types::MetricType::Undefined => "UNDEFINED",
        finch_types::MetricType::L2 => "L2",
        finch_types::MetricType::InnerProduct => "IP",
        finch_types::MetricType::Cosine => "COSINE",
        finch_types::MetricType::MipsL2 => "MIPS_L2",
        finch_types::MetricType::Hamming => "HAMMING",
    }
}

fn quantize_name(q: finch_types::QuantizeType) -> &'static str {
    match q {
        finch_types::QuantizeType::Undefined => "UNDEFINED",
        finch_types::QuantizeType::Fp16 => "FP16",
        finch_types::QuantizeType::Int8 => "INT8",
        finch_types::QuantizeType::Int4 => "INT4",
    }
}

#[pyclass(module = "finch._finch")]
#[derive(Clone)]
pub struct PyFieldSchema {
    pub inner: FieldSchema,
}

#[pymethods]
impl PyFieldSchema {
    #[new]
    #[pyo3(signature = (name, data_type, nullable=false, dimension=None))]
    fn new(name: &str, data_type: PyDataType, nullable: bool, dimension: Option<usize>) -> Self {
        let mut schema = FieldSchema::new(name, data_type.into());
        schema.nullable = nullable;
        schema.dimension = dimension;
        PyFieldSchema { inner: schema }
    }

    fn with_hnsw_index(&self, params: &PyHnswIndexParam) -> Self {
        let mut s = self.clone();
        let is_sparse = matches!(
            s.inner.data_type,
            finch_types::DataType::SparseFp16 | finch_types::DataType::SparseFp32
        );
        s.inner.index_params = Some(if is_sparse {
            IndexParams::HnswSparse(params.inner.clone())
        } else {
            IndexParams::Hnsw(params.inner.clone())
        });
        s
    }

    fn with_ivf_index(&self, params: &PyIvfIndexParam) -> Self {
        let mut s = self.clone();
        s.inner.index_params = Some(IndexParams::Ivf(params.inner.clone()));
        s
    }

    fn with_flat_index(&self, params: &PyFlatIndexParam) -> Self {
        let mut s = self.clone();
        let is_sparse = matches!(
            s.inner.data_type,
            finch_types::DataType::SparseFp16 | finch_types::DataType::SparseFp32
        );
        s.inner.index_params = Some(if is_sparse {
            IndexParams::FlatSparse(params.inner.clone())
        } else {
            IndexParams::Flat(params.inner.clone())
        });
        s
    }

    fn with_invert_index(&self, params: &PyInvertIndexParam) -> Self {
        let mut s = self.clone();
        s.inner.index_params = Some(IndexParams::Invert(params.inner.clone()));
        s
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn data_type(&self) -> PyDataType {
        self.inner.data_type.into()
    }

    #[getter]
    fn nullable(&self) -> bool {
        self.inner.nullable
    }

    #[getter]
    fn dimension(&self) -> Option<usize> {
        self.inner.dimension
    }

    fn is_vector(&self) -> bool {
        self.inner.is_vector()
    }

    fn is_scalar(&self) -> bool {
        self.inner.is_scalar()
    }

    fn index_type(&self) -> Option<PyIndexType> {
        self.inner.index_params.as_ref().map(|p| match p {
            IndexParams::Hnsw(_) => PyIndexType::Hnsw,
            IndexParams::Ivf(_) => PyIndexType::Ivf,
            IndexParams::Flat(_) => PyIndexType::Flat,
            IndexParams::Invert(_) => PyIndexType::Invert,
            IndexParams::HnswSparse(_) => PyIndexType::HnswSparse,
            IndexParams::FlatSparse(_) => PyIndexType::FlatSparse,
        })
    }

    fn index_params_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let d = PyDict::new_bound(py);
        if let Some(p) = self.inner.index_params.as_ref() {
            d.set_item("index_type", self.index_type().map(|t| t as u32))?;
            if let Some(m) = p.metric() {
                d.set_item("metric", PyMetricType::from(m) as u32)?;
            }
            if let Some(q) = p.quantize() {
                d.set_item("quantize", PyQuantizeType::from(q) as u32)?;
            }
            match p {
                IndexParams::Hnsw(h) | IndexParams::HnswSparse(h) => {
                    d.set_item("m", h.m)?;
                    d.set_item("ef_construction", h.ef_construction)?;
                }
                IndexParams::Ivf(ivf) => {
                    d.set_item("n_list", ivf.n_list)?;
                    d.set_item("n_iters", ivf.n_iters)?;
                    d.set_item("use_soar", ivf.use_soar)?;
                }
                IndexParams::Flat(_) | IndexParams::FlatSparse(_) => {}
                IndexParams::Invert(inv) => {
                    d.set_item("enable_range_optimization", inv.enable_range_optimization)?;
                    d.set_item("enable_extended_wildcard", inv.enable_extended_wildcard)?;
                }
            }
        }
        Ok(d.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "FieldSchema(name='{}', type={:?})",
            self.inner.name, self.inner.data_type
        )
    }
}

#[pyclass(module = "finch._finch")]
#[derive(Clone)]
pub struct PyCollectionSchema {
    pub inner: CollectionSchema,
}

#[pymethods]
impl PyCollectionSchema {
    #[new]
    #[pyo3(signature = (name, max_doc_count_per_segment=finch_types::MAX_DOC_COUNT_PER_SEGMENT))]
    fn new(name: &str, max_doc_count_per_segment: u64) -> Self {
        PyCollectionSchema {
            inner: CollectionSchema {
                name: name.to_string(),
                fields: Vec::new(),
                max_doc_count_per_segment,
            },
        }
    }

    fn add_field(&mut self, field: &PyFieldSchema) -> PyResult<()> {
        self.inner.fields.push(field.inner.clone());
        Ok(())
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn max_doc_count_per_segment(&self) -> u64 {
        self.inner.max_doc_count_per_segment
    }

    #[getter]
    fn fields(&self) -> Vec<PyFieldSchema> {
        self.inner
            .fields
            .iter()
            .cloned()
            .map(|inner| PyFieldSchema { inner })
            .collect()
    }

    fn scalar_fields(&self) -> Vec<PyFieldSchema> {
        self.inner
            .fields
            .iter()
            .filter(|f| f.is_scalar())
            .cloned()
            .map(|inner| PyFieldSchema { inner })
            .collect()
    }

    fn vector_fields(&self) -> Vec<PyFieldSchema> {
        self.inner
            .fields
            .iter()
            .filter(|f| f.is_vector())
            .cloned()
            .map(|inner| PyFieldSchema { inner })
            .collect()
    }

    fn get_field(&self, name: &str) -> Option<PyFieldSchema> {
        self.inner
            .get_field(name)
            .cloned()
            .map(|inner| PyFieldSchema { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "CollectionSchema(name='{}', fields={})",
            self.inner.name,
            self.inner.fields.len()
        )
    }
}

#[pyclass(module = "finch._finch")]
#[derive(Clone)]
pub struct PyQueryParam {
    pub ef: Option<u32>,
    pub n_probe: Option<u32>,
    pub concurrency: Option<usize>,
    pub bf_pks: Option<Vec<String>>,
    pub radius: Option<f32>,
    pub is_linear: Option<bool>,
    pub use_refiner: bool,
    pub refiner_k: Option<u32>,
    pub refiner_scale_factor: Option<f32>,
}

#[pymethods]
impl PyQueryParam {
    #[new]
    #[pyo3(signature = (
        ef=None,
        n_probe=None,
        concurrency=None,
        bf_pks=None,
        radius=None,
        is_linear=None,
        use_refiner=false,
        refiner_k=None,
        refiner_scale_factor=None
    ))]
    fn new(
        ef: Option<u32>,
        n_probe: Option<u32>,
        concurrency: Option<usize>,
        bf_pks: Option<Vec<String>>,
        radius: Option<f32>,
        is_linear: Option<bool>,
        use_refiner: bool,
        refiner_k: Option<u32>,
        refiner_scale_factor: Option<f32>,
    ) -> Self {
        PyQueryParam {
            ef,
            n_probe,
            concurrency,
            bf_pks,
            radius,
            is_linear,
            use_refiner,
            refiner_k,
            refiner_scale_factor,
        }
    }

    // ── reference-ish getters for query params ─────────────────────────────────

    #[getter]
    fn ef(&self) -> u32 {
        self.ef.unwrap_or(0)
    }

    #[getter]
    fn n_probe(&self) -> u32 {
        self.n_probe.unwrap_or(0)
    }

    #[getter]
    fn nprobe(&self) -> u32 {
        self.n_probe.unwrap_or(0)
    }

    #[getter]
    fn radius(&self) -> f32 {
        self.radius.unwrap_or(0.0)
    }

    #[getter]
    fn is_linear(&self) -> bool {
        self.is_linear.unwrap_or(false)
    }

    #[getter]
    fn is_using_refiner(&self) -> bool {
        self.use_refiner
    }

    fn __repr__(&self) -> String {
        format!(
            "QueryParam(ef={:?}, n_probe={:?}, radius={:?}, is_linear={:?}, use_refiner={})",
            self.ef, self.n_probe, self.radius, self.is_linear, self.use_refiner
        )
    }

    fn __getstate__(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok((
            self.ef,
            self.n_probe,
            self.concurrency,
            self.bf_pks.clone(),
            self.radius,
            self.is_linear,
            self.use_refiner,
            self.refiner_k,
            self.refiner_scale_factor,
        )
            .into_py(py))
    }

    fn __setstate__(&mut self, state: &Bound<'_, PyAny>) -> PyResult<()> {
        let (
            ef,
            n_probe,
            concurrency,
            bf_pks,
            radius,
            is_linear,
            use_refiner,
            refiner_k,
            refiner_scale_factor,
        ): QueryParamState = state.extract()?;
        self.ef = ef;
        self.n_probe = n_probe;
        self.concurrency = concurrency;
        self.bf_pks = bf_pks;
        self.radius = radius;
        self.is_linear = is_linear;
        self.use_refiner = use_refiner;
        self.refiner_k = refiner_k;
        self.refiner_scale_factor = refiner_scale_factor;
        Ok(())
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let state = self.__getstate__(py)?;
        let ty = py.get_type_bound::<PyQueryParam>();
        let args = pyo3::types::PyTuple::empty_bound(py);
        Ok((ty, args, state).into_py(py))
    }
}

#[pyclass(module = "finch._finch")]
#[derive(Clone)]
pub struct PyCollectionOption {
    pub inner: finch_types::CollectionOptions,
}

#[pymethods]
impl PyCollectionOption {
    #[new]
    #[pyo3(signature = (
        read_only=false,
        enable_mmap=true,
        index_storage=None,
        forward_storage=None,
        max_buffer_size=finch_types::CollectionOptions::DEFAULT_MAX_BUFFER_SIZE
    ))]
    fn new(
        read_only: bool,
        enable_mmap: bool,
        index_storage: Option<crate::types::PyStorageType>,
        forward_storage: Option<crate::types::PyStorageType>,
        max_buffer_size: u32,
    ) -> Self {
        PyCollectionOption {
            inner: finch_types::CollectionOptions {
                read_only,
                enable_mmap,
                index_storage: index_storage.map(Into::into),
                forward_storage: forward_storage.map(Into::into),
                max_buffer_size,
                forward_file_format: None,
            },
        }
    }

    #[getter]
    fn read_only(&self) -> bool {
        self.inner.read_only
    }

    #[getter]
    fn enable_mmap(&self) -> bool {
        self.inner.enable_mmap
    }

    #[getter]
    fn max_buffer_size(&self) -> u32 {
        self.inner.max_buffer_size
    }

    #[getter]
    fn index_storage(&self) -> Option<crate::types::PyStorageType> {
        self.inner.index_storage.map(Into::into)
    }

    #[getter]
    fn forward_storage(&self) -> Option<crate::types::PyStorageType> {
        self.inner.forward_storage.map(Into::into)
    }

    #[getter]
    fn forward_file_format(&self) -> Option<crate::types::PyFileFormat> {
        self.inner.forward_file_format.map(Into::into)
    }

    fn __repr__(&self) -> String {
        format!(
            "CollectionOption(read_only={}, enable_mmap={}, index_storage={:?}, forward_storage={:?}, max_buffer_size={}, forward_file_format={:?})",
            self.inner.read_only,
            self.inner.enable_mmap,
            self.index_storage(),
            self.forward_storage(),
            self.inner.max_buffer_size,
            self.forward_file_format(),
        )
    }

    fn __getstate__(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok((
            self.inner.read_only,
            self.inner.enable_mmap,
            self.index_storage(),
            self.forward_storage(),
            self.inner.max_buffer_size,
            self.forward_file_format(),
        )
            .into_py(py))
    }

    fn __setstate__(&mut self, state: &Bound<'_, PyAny>) -> PyResult<()> {
        // Backward compat: older pickles stored a 3-tuple.
        if let Ok((read_only, enable_mmap, max_buffer_size)) = state.extract::<(bool, bool, u32)>()
        {
            self.inner.read_only = read_only;
            self.inner.enable_mmap = enable_mmap;
            self.inner.index_storage = None;
            self.inner.forward_storage = None;
            self.inner.max_buffer_size = max_buffer_size;
            self.inner.forward_file_format = None;
            return Ok(());
        }

        let (
            read_only,
            enable_mmap,
            index_storage,
            forward_storage,
            max_buffer_size,
            forward_file_format,
        ): (
            bool,
            bool,
            Option<crate::types::PyStorageType>,
            Option<crate::types::PyStorageType>,
            u32,
            Option<crate::types::PyFileFormat>,
        ) = state.extract()?;
        self.inner.read_only = read_only;
        self.inner.enable_mmap = enable_mmap;
        self.inner.index_storage = index_storage.map(Into::into);
        self.inner.forward_storage = forward_storage.map(Into::into);
        self.inner.max_buffer_size = max_buffer_size;
        self.inner.forward_file_format = forward_file_format.map(Into::into);
        Ok(())
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let state = self.__getstate__(py)?;
        let ty = py.get_type_bound::<PyCollectionOption>();
        let args = pyo3::types::PyTuple::empty_bound(py);
        Ok((ty, args, state).into_py(py))
    }
}

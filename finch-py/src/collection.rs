//! Python Collection class

use crate::doc::{PyDoc, PyGroupResult, PyVectorQuery};
use crate::schema::{
    PyAddColumnOption, PyAlterColumnOption, PyCollectionOption, PyCollectionSchema, PyFieldSchema,
    PyFlatIndexParam, PyHnswIndexParam, PyIndexOption, PyInvertIndexParam, PyIvfIndexParam,
    PyOptimizeOption,
};
use crate::stats::PyCollectionStats;
use crate::types::PyStatus;
use crate::{to_py_err, PyResult};
use finch_db::Collection;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::Path;
use std::sync::Arc;

#[pyclass(module = "finch._finch")]
pub struct PyCollection {
    inner: Option<Arc<Collection>>,
}

impl PyCollection {
    fn inner_arc(&self) -> PyResult<&Arc<Collection>> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Collection is closed/destroyed").into())
    }

    fn repeat_status(n: usize, s: finch_types::Status) -> Vec<PyStatus> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(PyStatus::from(s.clone()));
        }
        out
    }
}

#[pymethods]
impl PyCollection {
    #[new]
    fn new() -> Self {
        PyCollection { inner: None }
    }

    // ── DML ──────────────────────────────────────────────────────────────────

    fn insert(&self, docs: Vec<PyDoc>) -> PyResult<Vec<PyStatus>> {
        let n = docs.len();
        let rust_docs: Vec<finch_types::Doc> = docs.into_iter().map(|d| d.inner).collect();
        match self.inner_arc()?.insert(rust_docs) {
            Ok(results) => Ok(results.into_iter().map(PyStatus::from).collect()),
            Err(s) => Ok(Self::repeat_status(n, s)),
        }
    }

    fn upsert(&self, docs: Vec<PyDoc>) -> PyResult<Vec<PyStatus>> {
        let n = docs.len();
        let rust_docs: Vec<finch_types::Doc> = docs.into_iter().map(|d| d.inner).collect();
        match self.inner_arc()?.upsert(rust_docs) {
            Ok(results) => Ok(results.into_iter().map(PyStatus::from).collect()),
            Err(s) => Ok(Self::repeat_status(n, s)),
        }
    }

    fn update(&self, docs: Vec<PyDoc>) -> PyResult<Vec<PyStatus>> {
        let n = docs.len();
        let rust_docs: Vec<finch_types::Doc> = docs.into_iter().map(|d| d.inner).collect();
        match self.inner_arc()?.update(rust_docs) {
            Ok(results) => Ok(results.into_iter().map(PyStatus::from).collect()),
            Err(s) => Ok(Self::repeat_status(n, s)),
        }
    }

    fn delete(&self, pks: Vec<String>) -> PyResult<Vec<PyStatus>> {
        let n = pks.len();
        match self.inner_arc()?.delete(pks) {
            Ok(results) => Ok(results.into_iter().map(PyStatus::from).collect()),
            Err(s) => Ok(Self::repeat_status(n, s)),
        }
    }

    fn delete_by_filter(&self, filter: &str) -> PyResult<PyStatus> {
        let s = self
            .inner_arc()?
            .delete_by_filter(filter)
            .map_err(to_py_err)?;
        Ok(PyStatus::from(s))
    }

    // ── DQL ──────────────────────────────────────────────────────────────────

    fn query(&self, py: Python<'_>, query: &PyVectorQuery) -> PyResult<Vec<PyDoc>> {
        let inner = self.inner_arc()?.clone();
        let q = query.inner.clone();
        let results = py.allow_threads(|| inner.query(q)).map_err(to_py_err)?;
        Ok(results
            .into_iter()
            .map(|doc| PyDoc {
                inner: (*doc).clone(),
            })
            .collect())
    }

    fn query_ids(&self, py: Python<'_>, query: &PyVectorQuery) -> PyResult<Vec<i64>> {
        let inner = self.inner_arc()?.clone();
        let q = query.inner.clone();
        py.allow_threads(|| inner.query_int_ids(q))
            .map_err(to_py_err)
    }

    fn query_sql(&self, py: Python<'_>, sql: &str) -> PyResult<Vec<PyDoc>> {
        let inner = self.inner_arc()?.clone();
        let sql = sql.to_string();
        let results = py
            .allow_threads(|| inner.query_sql(&sql))
            .map_err(to_py_err)?;
        Ok(results
            .into_iter()
            .map(|doc| PyDoc {
                inner: (*doc).clone(),
            })
            .collect())
    }

    fn fetch(
        &self,
        py: Python<'_>,
        pks: Vec<String>,
    ) -> PyResult<std::collections::HashMap<String, PyDoc>> {
        let inner = self.inner_arc()?.clone();
        let results = py.allow_threads(|| inner.fetch(pks)).map_err(to_py_err)?;
        Ok(results
            .into_iter()
            .map(|(pk, doc)| {
                let doc = PyDoc {
                    inner: (*doc).clone(),
                };
                (pk, doc)
            })
            .collect())
    }

    fn group_by_query(
        &self,
        py: Python<'_>,
        query: &PyVectorQuery,
        group_by_field: &str,
        group_count: usize,
        group_topk: usize,
    ) -> PyResult<Vec<PyGroupResult>> {
        let inner = self.inner_arc()?.clone();
        let gbq = finch_types::GroupByVectorQuery {
            base: query.inner.clone(),
            group_by_field: group_by_field.to_string(),
            group_count,
            group_topk,
        };
        let results = py
            .allow_threads(|| inner.group_by_query(gbq))
            .map_err(to_py_err)?;
        Ok(results
            .into_iter()
            .map(|r| PyGroupResult { inner: r })
            .collect())
    }

    // ── DDL ──────────────────────────────────────────────────────────────────

    #[pyo3(signature = (field, params, rebuild=false, concurrency=None, option=None))]
    fn create_hnsw_index(
        &self,
        field: &str,
        params: &PyHnswIndexParam,
        rebuild: bool,
        concurrency: Option<usize>,
        option: Option<&PyIndexOption>,
    ) -> PyResult<()> {
        let mut effective_concurrency = concurrency;
        if effective_concurrency.is_none() {
            effective_concurrency = option.and_then(|o| o.as_concurrency_option());
        }
        self.inner_arc()?
            .create_index(
                field,
                finch_types::IndexParams::Hnsw(params.inner.clone()),
                finch_types::CreateIndexOptions {
                    rebuild,
                    concurrency: effective_concurrency,
                },
            )
            .map_err(to_py_err)
    }

    #[pyo3(signature = (field, params, rebuild=false, concurrency=None, option=None))]
    fn create_ivf_index(
        &self,
        field: &str,
        params: &PyIvfIndexParam,
        rebuild: bool,
        concurrency: Option<usize>,
        option: Option<&PyIndexOption>,
    ) -> PyResult<()> {
        let mut effective_concurrency = concurrency;
        if effective_concurrency.is_none() {
            effective_concurrency = option.and_then(|o| o.as_concurrency_option());
        }
        self.inner_arc()?
            .create_index(
                field,
                finch_types::IndexParams::Ivf(params.inner.clone()),
                finch_types::CreateIndexOptions {
                    rebuild,
                    concurrency: effective_concurrency,
                },
            )
            .map_err(to_py_err)
    }

    #[pyo3(signature = (field, params, rebuild=false, concurrency=None, option=None))]
    fn create_flat_index(
        &self,
        field: &str,
        params: &PyFlatIndexParam,
        rebuild: bool,
        concurrency: Option<usize>,
        option: Option<&PyIndexOption>,
    ) -> PyResult<()> {
        let mut effective_concurrency = concurrency;
        if effective_concurrency.is_none() {
            effective_concurrency = option.and_then(|o| o.as_concurrency_option());
        }
        self.inner_arc()?
            .create_index(
                field,
                finch_types::IndexParams::Flat(params.inner.clone()),
                finch_types::CreateIndexOptions {
                    rebuild,
                    concurrency: effective_concurrency,
                },
            )
            .map_err(to_py_err)
    }

    #[pyo3(signature = (field, params, rebuild=false, concurrency=None, option=None))]
    fn create_hnsw_sparse_index(
        &self,
        field: &str,
        params: &PyHnswIndexParam,
        rebuild: bool,
        concurrency: Option<usize>,
        option: Option<&PyIndexOption>,
    ) -> PyResult<()> {
        let mut effective_concurrency = concurrency;
        if effective_concurrency.is_none() {
            effective_concurrency = option.and_then(|o| o.as_concurrency_option());
        }
        self.inner_arc()?
            .create_index(
                field,
                finch_types::IndexParams::HnswSparse(params.inner.clone()),
                finch_types::CreateIndexOptions {
                    rebuild,
                    concurrency: effective_concurrency,
                },
            )
            .map_err(to_py_err)
    }

    #[pyo3(signature = (field, params, rebuild=false, concurrency=None, option=None))]
    fn create_flat_sparse_index(
        &self,
        field: &str,
        params: &PyFlatIndexParam,
        rebuild: bool,
        concurrency: Option<usize>,
        option: Option<&PyIndexOption>,
    ) -> PyResult<()> {
        let mut effective_concurrency = concurrency;
        if effective_concurrency.is_none() {
            effective_concurrency = option.and_then(|o| o.as_concurrency_option());
        }
        self.inner_arc()?
            .create_index(
                field,
                finch_types::IndexParams::FlatSparse(params.inner.clone()),
                finch_types::CreateIndexOptions {
                    rebuild,
                    concurrency: effective_concurrency,
                },
            )
            .map_err(to_py_err)
    }

    #[pyo3(signature = (field, params, rebuild=false, concurrency=None, option=None))]
    fn create_invert_index(
        &self,
        field: &str,
        params: &PyInvertIndexParam,
        rebuild: bool,
        concurrency: Option<usize>,
        option: Option<&PyIndexOption>,
    ) -> PyResult<()> {
        let mut effective_concurrency = concurrency;
        if effective_concurrency.is_none() {
            effective_concurrency = option.and_then(|o| o.as_concurrency_option());
        }
        self.inner_arc()?
            .create_index(
                field,
                finch_types::IndexParams::Invert(params.inner.clone()),
                finch_types::CreateIndexOptions {
                    rebuild,
                    concurrency: effective_concurrency,
                },
            )
            .map_err(to_py_err)
    }

    fn drop_index(&self, field: &str) -> PyResult<()> {
        self.inner_arc()?.drop_index(field).map_err(to_py_err)
    }

    #[pyo3(signature = (field, rebuild_index=false, concurrency=None, option=None, *, expression=None))]
    fn add_column(
        &self,
        field: &PyFieldSchema,
        rebuild_index: bool,
        concurrency: Option<usize>,
        option: Option<&PyAddColumnOption>,
        expression: Option<&str>,
    ) -> PyResult<()> {
        let mut effective_concurrency = concurrency;
        if effective_concurrency.is_none() {
            effective_concurrency = option.and_then(|o| o.as_concurrency_option());
        }
        self.inner_arc()?
            .add_column_with_expression(
                field.inner.clone(),
                expression,
                finch_types::AddColumnOptions {
                    rebuild_index,
                    concurrency: effective_concurrency,
                },
            )
            .map_err(to_py_err)
    }

    fn drop_column(&self, field: &str) -> PyResult<()> {
        self.inner_arc()?.drop_column(field).map_err(to_py_err)
    }

    #[pyo3(signature = (field, rename_to=None, field_schema=None, rebuild_index=false, concurrency=None, option=None))]
    fn alter_column(
        &self,
        field: &str,
        rename_to: Option<&str>,
        field_schema: Option<&PyFieldSchema>,
        rebuild_index: bool,
        concurrency: Option<usize>,
        option: Option<&PyAlterColumnOption>,
    ) -> PyResult<()> {
        let mut effective_concurrency = concurrency;
        if effective_concurrency.is_none() {
            effective_concurrency = option.and_then(|o| o.as_concurrency_option());
        }
        self.inner_arc()?
            .alter_column(
                field,
                rename_to,
                field_schema.map(|s| s.inner.clone()),
                finch_types::AlterColumnOptions {
                    rebuild_index,
                    concurrency: effective_concurrency,
                },
            )
            .map_err(to_py_err)
    }

    // ── Admin ─────────────────────────────────────────────────────────────────

    #[pyo3(signature = (max_segments=None, concurrency=None, option=None))]
    fn optimize(
        &self,
        max_segments: Option<usize>,
        concurrency: Option<usize>,
        option: Option<&PyOptimizeOption>,
    ) -> PyResult<()> {
        let mut effective_concurrency = concurrency;
        if effective_concurrency.is_none() {
            effective_concurrency = option.and_then(|o| o.as_concurrency_option());
        }
        self.inner_arc()?
            .optimize(finch_types::OptimizeOptions {
                max_segments,
                concurrency: effective_concurrency,
                ..Default::default()
            })
            .map_err(to_py_err)
    }

    fn flush(&self) -> PyResult<()> {
        self.inner_arc()?.flush().map_err(to_py_err)
    }

    fn close(&mut self) -> PyResult<()> {
        // reference parity: close releases resources but does not delete on-disk data.
        self.inner.take();
        Ok(())
    }

    fn stats_dict(&self) -> PyResult<std::collections::HashMap<String, u64>> {
        let stats = self.inner_arc()?.stats().map_err(to_py_err)?;
        let mut m = std::collections::HashMap::new();
        m.insert("doc_count".to_string(), stats.doc_count);
        m.insert("segment_count".to_string(), stats.segment_count as u64);
        Ok(m)
    }

    fn stats(&self) -> PyResult<PyCollectionStats> {
        let stats = self.inner_arc()?.stats().map_err(to_py_err)?;
        Ok(PyCollectionStats { inner: stats })
    }

    fn stats_info(&self, py: Python<'_>) -> PyResult<PyObject> {
        let stats = self.inner_arc()?.stats().map_err(to_py_err)?;
        let d = PyDict::new_bound(py);
        d.set_item("doc_count", stats.doc_count)?;
        d.set_item("segment_count", stats.segment_count)?;
        let ic = PyDict::new_bound(py);
        for (k, v) in stats.index_completeness {
            ic.set_item(k, v)?;
        }
        d.set_item("index_completeness", ic)?;
        Ok(d.into())
    }

    fn path(&self) -> PyResult<String> {
        Ok(self.inner_arc()?.path_string())
    }

    fn options(&self) -> PyResult<PyCollectionOption> {
        Ok(PyCollectionOption {
            inner: self.inner_arc()?.options(),
        })
    }

    fn schema(&self) -> PyResult<PyCollectionSchema> {
        Ok(PyCollectionSchema {
            inner: self.inner_arc()?.schema_info(),
        })
    }

    fn schema_info(&self) -> PyResult<PyCollectionSchema> {
        self.schema()
    }

    fn destroy(&mut self) -> PyResult<()> {
        let arc = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Collection is closed/destroyed"))?;
        match Arc::try_unwrap(arc) {
            Ok(col) => col.destroy().map_err(to_py_err),
            Err(arc) => {
                self.inner = Some(arc);
                Err(PyRuntimeError::new_err(
                    "Collection has other references; drop them before destroy",
                )
                .into())
            }
        }
    }

    fn __repr__(&self) -> String {
        match self.inner.as_ref() {
            Some(inner) => format!("Collection(path='{}')", inner.path.display()),
            None => "Collection(destroyed)".to_string(),
        }
    }

    // ── Pickle support (reference parity) ─────────────────────────────────────

    fn __getstate__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let path = self.inner_arc()?.path_string();
        let opts = PyCollectionOption {
            inner: self.inner_arc()?.options(),
        };
        Ok((path, opts).into_py(py))
    }

    fn __setstate__(&mut self, state: &Bound<'_, PyAny>) -> PyResult<()> {
        let (path, opts): (String, PyCollectionOption) = state.extract()?;
        let collection = Collection::open(Path::new(&path), opts.inner).map_err(to_py_err)?;
        self.inner = Some(collection);
        Ok(())
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let state = self.__getstate__(py)?;
        let ty = py.get_type_bound::<PyCollection>();
        let args = pyo3::types::PyTuple::empty_bound(py);
        Ok((ty, args, state).into_py(py))
    }
}

pub fn create_and_open_impl(
    path: &str,
    schema: &PyCollectionSchema,
    options: Option<&PyCollectionOption>,
) -> PyResult<PyCollection> {
    let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
    let collection = Collection::create_and_open(Path::new(path), schema.inner.clone(), opts)
        .map_err(to_py_err)?;
    Ok(PyCollection {
        inner: Some(collection),
    })
}

pub fn open_impl(path: &str, option: &PyCollectionOption) -> PyResult<PyCollection> {
    let collection = Collection::open(Path::new(path), option.inner.clone()).map_err(to_py_err)?;
    Ok(PyCollection {
        inner: Some(collection),
    })
}

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

    /// Runs `op` with the GIL released so other Python threads keep running.
    fn without_gil<T: Send>(
        &self,
        py: Python<'_>,
        op: impl FnOnce(&Collection) -> finch_types::ZResult<T> + Send,
    ) -> PyResult<T> {
        let inner = self.inner_arc()?.clone();
        py.allow_threads(move || op(&inner)).map_err(to_py_err)
    }

    fn create_index(
        &self,
        py: Python<'_>,
        field: &str,
        params: finch_types::IndexParams,
        rebuild: bool,
        concurrency: Option<usize>,
        option: Option<&PyIndexOption>,
    ) -> PyResult<()> {
        let options = finch_types::CreateIndexOptions {
            rebuild,
            concurrency: concurrency.or_else(|| option.and_then(|o| o.as_concurrency_option())),
        };
        self.without_gil(py, |c| c.create_index(field, params, options))
    }
}

fn statuses(results: Vec<finch_types::Status>) -> Vec<PyStatus> {
    results.into_iter().map(PyStatus::from).collect()
}

fn rust_docs(docs: Vec<PyDoc>) -> Vec<finch_types::Doc> {
    docs.into_iter().map(|d| d.inner).collect()
}

#[pymethods]
impl PyCollection {
    #[new]
    fn new() -> Self {
        PyCollection { inner: None }
    }

    // ── DML ──────────────────────────────────────────────────────────────────

    fn insert(&self, py: Python<'_>, docs: Vec<PyDoc>) -> PyResult<Vec<PyStatus>> {
        let docs = rust_docs(docs);
        self.without_gil(py, |c| c.insert(docs)).map(statuses)
    }

    fn upsert(&self, py: Python<'_>, docs: Vec<PyDoc>) -> PyResult<Vec<PyStatus>> {
        let docs = rust_docs(docs);
        self.without_gil(py, |c| c.upsert(docs)).map(statuses)
    }

    fn update(&self, py: Python<'_>, docs: Vec<PyDoc>) -> PyResult<Vec<PyStatus>> {
        let docs = rust_docs(docs);
        self.without_gil(py, |c| c.update(docs)).map(statuses)
    }

    fn delete(&self, py: Python<'_>, pks: Vec<String>) -> PyResult<Vec<PyStatus>> {
        self.without_gil(py, |c| c.delete(pks)).map(statuses)
    }

    fn delete_by_filter(&self, py: Python<'_>, filter: &str) -> PyResult<PyStatus> {
        self.without_gil(py, |c| c.delete_by_filter(filter))
            .map(PyStatus::from)
    }

    // ── DQL ──────────────────────────────────────────────────────────────────

    fn query(&self, py: Python<'_>, query: &PyVectorQuery) -> PyResult<Vec<PyDoc>> {
        let q = query.inner.clone();
        let results = self.without_gil(py, |c| c.query(q))?;
        Ok(results
            .into_iter()
            .map(|doc| PyDoc {
                inner: (*doc).clone(),
            })
            .collect())
    }

    fn query_ids(&self, py: Python<'_>, query: &PyVectorQuery) -> PyResult<Vec<i64>> {
        let q = query.inner.clone();
        self.without_gil(py, |c| c.query_int_ids(q))
    }

    fn query_sql(&self, py: Python<'_>, sql: &str) -> PyResult<Vec<PyDoc>> {
        let results = self.without_gil(py, |c| c.query_sql(sql))?;
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
        let results = self.without_gil(py, |c| c.fetch(pks))?;
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
        let gbq = finch_types::GroupByVectorQuery {
            base: query.inner.clone(),
            group_by_field: group_by_field.to_string(),
            group_count,
            group_topk,
        };
        let results = self.without_gil(py, |c| c.group_by_query(gbq))?;
        Ok(results
            .into_iter()
            .map(|r| PyGroupResult { inner: r })
            .collect())
    }

    // ── DDL ──────────────────────────────────────────────────────────────────

    #[pyo3(signature = (field, params, rebuild=false, concurrency=None, option=None))]
    fn create_hnsw_index(
        &self,
        py: Python<'_>,
        field: &str,
        params: &PyHnswIndexParam,
        rebuild: bool,
        concurrency: Option<usize>,
        option: Option<&PyIndexOption>,
    ) -> PyResult<()> {
        let params = finch_types::IndexParams::Hnsw(params.inner.clone());
        self.create_index(py, field, params, rebuild, concurrency, option)
    }

    #[pyo3(signature = (field, params, rebuild=false, concurrency=None, option=None))]
    fn create_ivf_index(
        &self,
        py: Python<'_>,
        field: &str,
        params: &PyIvfIndexParam,
        rebuild: bool,
        concurrency: Option<usize>,
        option: Option<&PyIndexOption>,
    ) -> PyResult<()> {
        let params = finch_types::IndexParams::Ivf(params.inner.clone());
        self.create_index(py, field, params, rebuild, concurrency, option)
    }

    #[pyo3(signature = (field, params, rebuild=false, concurrency=None, option=None))]
    fn create_flat_index(
        &self,
        py: Python<'_>,
        field: &str,
        params: &PyFlatIndexParam,
        rebuild: bool,
        concurrency: Option<usize>,
        option: Option<&PyIndexOption>,
    ) -> PyResult<()> {
        let params = finch_types::IndexParams::Flat(params.inner.clone());
        self.create_index(py, field, params, rebuild, concurrency, option)
    }

    #[pyo3(signature = (field, params, rebuild=false, concurrency=None, option=None))]
    fn create_hnsw_sparse_index(
        &self,
        py: Python<'_>,
        field: &str,
        params: &PyHnswIndexParam,
        rebuild: bool,
        concurrency: Option<usize>,
        option: Option<&PyIndexOption>,
    ) -> PyResult<()> {
        let params = finch_types::IndexParams::HnswSparse(params.inner.clone());
        self.create_index(py, field, params, rebuild, concurrency, option)
    }

    #[pyo3(signature = (field, params, rebuild=false, concurrency=None, option=None))]
    fn create_flat_sparse_index(
        &self,
        py: Python<'_>,
        field: &str,
        params: &PyFlatIndexParam,
        rebuild: bool,
        concurrency: Option<usize>,
        option: Option<&PyIndexOption>,
    ) -> PyResult<()> {
        let params = finch_types::IndexParams::FlatSparse(params.inner.clone());
        self.create_index(py, field, params, rebuild, concurrency, option)
    }

    #[pyo3(signature = (field, params, rebuild=false, concurrency=None, option=None))]
    fn create_invert_index(
        &self,
        py: Python<'_>,
        field: &str,
        params: &PyInvertIndexParam,
        rebuild: bool,
        concurrency: Option<usize>,
        option: Option<&PyIndexOption>,
    ) -> PyResult<()> {
        let params = finch_types::IndexParams::Invert(params.inner.clone());
        self.create_index(py, field, params, rebuild, concurrency, option)
    }

    fn drop_index(&self, py: Python<'_>, field: &str) -> PyResult<()> {
        self.without_gil(py, |c| c.drop_index(field))
    }

    #[pyo3(signature = (field, rebuild_index=false, concurrency=None, option=None, *, expression=None))]
    fn add_column(
        &self,
        py: Python<'_>,
        field: &PyFieldSchema,
        rebuild_index: bool,
        concurrency: Option<usize>,
        option: Option<&PyAddColumnOption>,
        expression: Option<&str>,
    ) -> PyResult<()> {
        let field = field.inner.clone();
        let options = finch_types::AddColumnOptions {
            rebuild_index,
            concurrency: concurrency.or_else(|| option.and_then(|o| o.as_concurrency_option())),
        };
        self.without_gil(py, |c| {
            c.add_column_with_expression(field, expression, options)
        })
    }

    fn drop_column(&self, py: Python<'_>, field: &str) -> PyResult<()> {
        self.without_gil(py, |c| c.drop_column(field))
    }

    #[pyo3(signature = (field, rename_to=None, field_schema=None, rebuild_index=false, concurrency=None, option=None))]
    fn alter_column(
        &self,
        py: Python<'_>,
        field: &str,
        rename_to: Option<&str>,
        field_schema: Option<&PyFieldSchema>,
        rebuild_index: bool,
        concurrency: Option<usize>,
        option: Option<&PyAlterColumnOption>,
    ) -> PyResult<()> {
        let field_schema = field_schema.map(|s| s.inner.clone());
        let options = finch_types::AlterColumnOptions {
            rebuild_index,
            concurrency: concurrency.or_else(|| option.and_then(|o| o.as_concurrency_option())),
        };
        self.without_gil(py, |c| {
            c.alter_column(field, rename_to, field_schema, options)
        })
    }

    // ── Admin ─────────────────────────────────────────────────────────────────

    #[pyo3(signature = (max_segments=None, concurrency=None, option=None))]
    fn optimize(
        &self,
        py: Python<'_>,
        max_segments: Option<usize>,
        concurrency: Option<usize>,
        option: Option<&PyOptimizeOption>,
    ) -> PyResult<()> {
        let options = finch_types::OptimizeOptions {
            max_segments,
            concurrency: concurrency.or_else(|| option.and_then(|o| o.as_concurrency_option())),
            ..Default::default()
        };
        self.without_gil(py, |c| c.optimize(options))
    }

    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        self.without_gil(py, |c| c.flush())
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

    fn destroy(&mut self, py: Python<'_>) -> PyResult<()> {
        let arc = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Collection is closed/destroyed"))?;
        match Arc::try_unwrap(arc) {
            Ok(col) => py.allow_threads(|| col.destroy()).map_err(to_py_err),
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
        *self = open_impl(state.py(), &path, &opts)?;
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
    py: Python<'_>,
    path: &str,
    schema: &PyCollectionSchema,
    options: Option<&PyCollectionOption>,
) -> PyResult<PyCollection> {
    let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
    let schema = schema.inner.clone();
    let collection = py
        .allow_threads(|| Collection::create_and_open(Path::new(path), schema, opts))
        .map_err(to_py_err)?;
    Ok(PyCollection {
        inner: Some(collection),
    })
}

pub fn open_impl(
    py: Python<'_>,
    path: &str,
    option: &PyCollectionOption,
) -> PyResult<PyCollection> {
    let opts = option.inner.clone();
    let collection = py
        .allow_threads(|| Collection::open(Path::new(path), opts))
        .map_err(to_py_err)?;
    Ok(PyCollection {
        inner: Some(collection),
    })
}

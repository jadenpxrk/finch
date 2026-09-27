//! Python collection stats classes (reference-ish parity)

use crate::PyResult;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

#[pyclass(module = "finch._finch")]
#[derive(Clone, Debug)]
pub struct PyCollectionStats {
    pub inner: finch_types::CollectionStats,
}

#[pymethods]
impl PyCollectionStats {
    #[new]
    fn new() -> Self {
        // reference parity: CollectionStats is constructible from Python.
        PyCollectionStats {
            inner: finch_types::CollectionStats {
                doc_count: 0,
                segment_count: 0,
                vector_field_stats: Vec::new(),
                index_completeness: HashMap::new(),
            },
        }
    }

    #[getter]
    fn doc_count(&self) -> u64 {
        self.inner.doc_count
    }

    #[getter]
    fn segment_count(&self) -> usize {
        self.inner.segment_count
    }

    #[getter]
    fn index_completeness(&self, py: Python<'_>) -> PyResult<PyObject> {
        let d = PyDict::new_bound(py);
        let mut items: Vec<_> = self.inner.index_completeness.iter().collect();
        items.sort_by(|a, b| a.0.cmp(b.0));
        for (k, v) in items {
            d.set_item(k, *v)?;
        }
        Ok(d.into())
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let d = PyDict::new_bound(py);
        d.set_item("doc_count", self.inner.doc_count)?;
        d.set_item("segment_count", self.inner.segment_count)?;

        let ic = PyDict::new_bound(py);
        let mut items: Vec<_> = self.inner.index_completeness.iter().collect();
        items.sort_by(|a, b| a.0.cmp(b.0));
        for (k, v) in items {
            ic.set_item(k, *v)?;
        }
        d.set_item("index_completeness", ic)?;
        Ok(d.into())
    }

    #[pyo3(signature = (indent_level=0))]
    fn to_string_formatted(&self, indent_level: usize) -> String {
        self.inner.to_string_formatted(indent_level)
    }

    fn __repr__(&self) -> String {
        format!(
            "CollectionStats(doc_count={}, segment_count={})",
            self.inner.doc_count, self.inner.segment_count
        )
    }
}

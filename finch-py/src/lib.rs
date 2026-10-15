// Binding signatures mirror the Python keyword-argument lists, which are the wire contract.
#![allow(clippy::too_many_arguments)]

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use pyo3::prelude::*;

// pyo3 0.22 wraps every binding result in `Into::<PyErr>::into`, spanned onto our
// signatures; a distinct error type keeps that conversion from being a no-op.
pub(crate) struct BindingError(PyErr);

impl From<PyErr> for BindingError {
    fn from(err: PyErr) -> Self {
        BindingError(err)
    }
}

impl From<BindingError> for PyErr {
    fn from(err: BindingError) -> Self {
        err.0
    }
}

impl From<pyo3::DowncastError<'_, '_>> for BindingError {
    fn from(err: pyo3::DowncastError<'_, '_>) -> Self {
        BindingError(err.into())
    }
}

pub(crate) type PyResult<T> = Result<T, BindingError>;

/// The raised exception carries the finch status code as its `code` attribute.
pub(crate) fn to_py_err(e: finch_types::Status) -> BindingError {
    use pyo3::exceptions::{PyOSError, PyPermissionError, PyRuntimeError, PyValueError};
    let err = match e.code {
        finch_types::StatusCode::InvalidArgument => PyValueError::new_err(e.to_string()),
        finch_types::StatusCode::PermissionDenied => PyPermissionError::new_err(e.to_string()),
        finch_types::StatusCode::IoError => PyOSError::new_err(e.to_string()),
        _ => PyRuntimeError::new_err(e.to_string()),
    };
    let code = types::PyStatusCode::from(e.code);
    Python::with_gil(
        |py| match err.value_bound(py).setattr("code", code.into_py(py)) {
            Ok(()) => BindingError(err),
            Err(set_err) => BindingError(set_err),
        },
    )
}

mod memory;
mod types;

use memory::*;
use types::*;

/// finch Python bindings module
#[pymodule]
fn _finch(m: &Bound<'_, PyModule>) -> pyo3::PyResult<()> {
    m.add_class::<PyStatusCode>()?;
    m.add_class::<PyMemoryStore>()?;
    Ok(())
}

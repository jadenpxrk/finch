// Binding signatures mirror the Python keyword-argument lists, which are the wire contract.
#![allow(clippy::too_many_arguments)]

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

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

pub(crate) fn to_py_err(e: finch_types::Status) -> BindingError {
    use pyo3::exceptions::{PyOSError, PyPermissionError, PyRuntimeError, PyValueError};
    BindingError(match e.code {
        finch_types::StatusCode::InvalidArgument => PyValueError::new_err(e.to_string()),
        finch_types::StatusCode::PermissionDenied => PyPermissionError::new_err(e.to_string()),
        finch_types::StatusCode::IoError => PyOSError::new_err(e.to_string()),
        _ => PyRuntimeError::new_err(e.to_string()),
    })
}

mod collection;
mod doc;
mod memory;
mod schema;
mod stats;
mod types;

use collection::*;
use doc::*;
use memory::*;
use schema::*;
use stats::*;
use types::*;

/// finch Python bindings module
#[pymodule]
fn _finch(m: &Bound<'_, PyModule>) -> pyo3::PyResult<()> {
    // Allow importing `finch._finch.*` submodules (reference-style layout) by
    // marking this extension module as a package.
    m.add("__path__", PyList::empty_bound(m.py()))?;

    let parent_name: String = m.getattr("__name__")?.extract()?;

    let typing = add_type_enums(m, &parent_name)?;
    add_schema_and_option_classes(m)?;
    let param = add_doc_and_query_classes(m, &parent_name)?;
    register_submodules(m, &[typing, param])?;
    add_collection_api(m)?;
    Ok(())
}

/// A submodule registered under its dotted name in `sys.modules`.
struct Submodule<'py> {
    name: String,
    module: Bound<'py, PyModule>,
}

/// Type enums, added both to the main module and the `typing` submodule.
fn add_type_enums<'py>(
    m: &Bound<'py, PyModule>,
    parent_name: &str,
) -> pyo3::PyResult<Submodule<'py>> {
    m.add_class::<PyDataType>()?;
    m.add_class::<PyIndexType>()?;
    m.add_class::<PyMetricType>()?;
    m.add_class::<PyQuantizeType>()?;
    m.add_class::<PyFileFormat>()?;
    m.add_class::<PyStorageType>()?;
    m.add_class::<PyStatusCode>()?;
    m.add_class::<PyLogType>()?;
    m.add_class::<PyLogLevel>()?;
    m.add_class::<PyStatus>()?;

    let typing_name = format!("{parent_name}.typing");
    let typing = PyModule::new_bound(m.py(), &typing_name)?;
    typing.add_class::<PyDataType>()?;
    typing.add_class::<PyIndexType>()?;
    typing.add_class::<PyMetricType>()?;
    typing.add_class::<PyQuantizeType>()?;
    typing.add_class::<PyFileFormat>()?;
    typing.add_class::<PyStorageType>()?;
    typing.add_class::<PyStatusCode>()?;
    typing.add_class::<PyLogType>()?;
    typing.add_class::<PyLogLevel>()?;
    typing.add_class::<PyStatus>()?;
    m.add_submodule(&typing)?;
    Ok(Submodule {
        name: typing_name,
        module: typing,
    })
}

fn add_schema_and_option_classes(m: &Bound<'_, PyModule>) -> pyo3::PyResult<()> {
    // Schema types
    m.add_class::<PyFieldSchema>()?;
    m.add_class::<PyCollectionSchema>()?;

    // Index params
    m.add_class::<PyHnswIndexParam>()?;
    m.add_class::<PyIvfIndexParam>()?;
    m.add_class::<PyFlatIndexParam>()?;
    m.add_class::<PyInvertIndexParam>()?;

    // Option objects (reference-ish parity)
    m.add_class::<PyIndexOption>()?;
    m.add_class::<PyOptimizeOption>()?;
    m.add_class::<PyAddColumnOption>()?;
    m.add_class::<PyAlterColumnOption>()?;

    // Query params
    m.add_class::<PyQueryParam>()?;
    m.add_class::<PyCollectionOption>()?;
    Ok(())
}

/// Doc and vector-query classes plus the reference-style `param` submodule.
fn add_doc_and_query_classes<'py>(
    m: &Bound<'py, PyModule>,
    parent_name: &str,
) -> pyo3::PyResult<Submodule<'py>> {
    m.add_class::<PyDoc>()?;
    // reference-style alias: `_Doc`
    m.add("_Doc", m.getattr("PyDoc")?)?;
    m.add_class::<PyVectorQuery>()?;
    // reference-style alias: `_VectorQuery`
    m.add("_VectorQuery", m.getattr("PyVectorQuery")?)?;
    m.add_class::<PyGroupResult>()?;

    // reference-style param submodule: `_reference.param._VectorQuery` equivalent
    let param_name = format!("{parent_name}.param");
    let param = PyModule::new_bound(m.py(), &param_name)?;
    param.add_class::<PyVectorQuery>()?;
    param.add("_VectorQuery", param.getattr("PyVectorQuery")?)?;
    param.add_class::<PyQueryParam>()?;
    m.add_submodule(&param)?;
    Ok(Submodule {
        name: param_name,
        module: param,
    })
}

// Makes submodules importable via `import finch._finch.param` etc.
fn register_submodules(
    m: &Bound<'_, PyModule>,
    submodules: &[Submodule<'_>],
) -> pyo3::PyResult<()> {
    let sys = PyModule::import_bound(m.py(), "sys")?;
    let modules_any = sys.getattr("modules")?;
    let modules = modules_any.downcast::<PyDict>()?;
    for submodule in submodules {
        modules.set_item(&submodule.name, &submodule.module)?;
    }
    Ok(())
}

fn add_collection_api(m: &Bound<'_, PyModule>) -> pyo3::PyResult<()> {
    m.add_class::<PyCollection>()?;
    m.add_class::<PyMemoryStore>()?;
    m.add_function(wrap_pyfunction!(create_and_open, m)?)?;
    m.add_function(wrap_pyfunction!(open_collection, m)?)?;
    m.add_function(wrap_pyfunction!(init_global_config, m)?)?;
    m.add_function(wrap_pyfunction!(init, m)?)?;

    // Stats (reference-ish parity)
    m.add_class::<PyCollectionStats>()?;
    Ok(())
}

/// Create and open a new collection
#[pyfunction(name = "create_and_open")]
#[pyo3(signature = (path, schema, options=None))]
fn create_and_open(
    path: &str,
    schema: &PyCollectionSchema,
    options: Option<&PyCollectionOption>,
) -> PyResult<PyCollection> {
    collection::create_and_open_impl(path, schema, options)
}

/// Open an existing collection
#[pyfunction(name = "open")]
#[pyo3(signature = (path, option))]
fn open_collection(path: &str, option: &PyCollectionOption) -> PyResult<PyCollection> {
    collection::open_impl(path, option)
}

/// Initialize Finch global config (reference parity). This is best called once
/// early in the process before running queries.
#[pyfunction(name = "init_global_config")]
#[pyo3(signature = (
    memory_limit_bytes=None,
    log_level=None,
    log_to_file=None,
    log_dir=None,
    log_basename=None,
    log_file_size_mb=None,
    log_overdue_days=None,
    query_thread_count=None,
    wal_flush_every_docs=None,
    wal_fsync_every_docs=None,
    invert_to_forward_scan_ratio=None,
    brute_force_by_keys_ratio=None,
    optimize_thread_count=None
))]
fn init_global_config(
    memory_limit_bytes: Option<u64>,
    log_level: Option<u8>,
    log_to_file: Option<bool>,
    log_dir: Option<String>,
    log_basename: Option<String>,
    log_file_size_mb: Option<u32>,
    log_overdue_days: Option<u32>,
    query_thread_count: Option<u32>,
    wal_flush_every_docs: Option<u32>,
    wal_fsync_every_docs: Option<u32>,
    invert_to_forward_scan_ratio: Option<f32>,
    brute_force_by_keys_ratio: Option<f32>,
    optimize_thread_count: Option<u32>,
) -> PyResult<()> {
    use pyo3::exceptions::PyRuntimeError;

    let mut cfg = finch_types::GlobalConfigData::default();
    if let Some(v) = memory_limit_bytes {
        cfg.memory_limit_bytes = v;
    }
    if let Some(v) = log_level {
        cfg.log_level = match v {
            0 => finch_types::LogLevel::Debug,
            1 => finch_types::LogLevel::Info,
            2 => finch_types::LogLevel::Warn,
            3 => finch_types::LogLevel::Error,
            _ => finch_types::LogLevel::Fatal,
        };
    }
    if let Some(v) = log_to_file {
        cfg.log_to_file = v;
    }
    if let Some(v) = log_dir {
        cfg.log_dir = v;
    }
    if let Some(v) = log_basename {
        cfg.log_basename = v;
    }
    if let Some(v) = log_file_size_mb {
        cfg.log_file_size_mb = v;
    }
    if let Some(v) = log_overdue_days {
        cfg.log_overdue_days = v;
    }
    if let Some(v) = query_thread_count {
        cfg.query_thread_count = v;
    }
    if let Some(v) = wal_flush_every_docs {
        cfg.wal_flush_every_docs = v;
    }
    if let Some(v) = wal_fsync_every_docs {
        cfg.wal_fsync_every_docs = v;
    }
    if let Some(v) = invert_to_forward_scan_ratio {
        cfg.invert_to_forward_scan_ratio = v;
    }
    if let Some(v) = brute_force_by_keys_ratio {
        cfg.brute_force_by_keys_ratio = v;
    }
    if let Some(v) = optimize_thread_count {
        cfg.optimize_thread_count = v;
    }

    finch_db::initialize_global_config(cfg)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()).into())
}

/// reference-compat `init(...)` API (best-effort mapping to Finch global config).
#[pyfunction(name = "init")]
#[pyo3(signature = (
    log_type=None,
    log_level=None,
    log_dir=None,
    log_basename=None,
    log_file_size=None,
    log_overdue_days=None,
    query_threads=None,
    optimize_threads=None,
    invert_to_forward_scan_ratio=None,
    brute_force_by_keys_ratio=None,
    memory_limit_mb=None,
    wal_flush_every_docs=None,
    wal_fsync_every_docs=None
))]
fn init(
    log_type: Option<PyLogType>,
    log_level: Option<PyLogLevel>,
    log_dir: Option<String>,
    log_basename: Option<String>,
    log_file_size: Option<u32>,
    log_overdue_days: Option<u32>,
    query_threads: Option<u32>,
    optimize_threads: Option<u32>,
    invert_to_forward_scan_ratio: Option<f32>,
    brute_force_by_keys_ratio: Option<f32>,
    memory_limit_mb: Option<u64>,
    wal_flush_every_docs: Option<u32>,
    wal_fsync_every_docs: Option<u32>,
) -> PyResult<()> {
    use pyo3::exceptions::PyRuntimeError;

    let mut cfg = finch_types::GlobalConfigData::default();

    if let Some(v) = log_type {
        cfg.log_to_file = matches!(v, PyLogType::File);
    }
    if let Some(v) = log_level {
        cfg.log_level = v.into();
    }
    if let Some(v) = log_dir {
        cfg.log_dir = v;
    }
    if let Some(v) = log_basename {
        cfg.log_basename = v;
    }
    if let Some(v) = log_file_size {
        cfg.log_file_size_mb = v;
    }
    if let Some(v) = log_overdue_days {
        cfg.log_overdue_days = v;
    }
    if let Some(v) = query_threads {
        cfg.query_thread_count = v;
    }
    if let Some(v) = optimize_threads {
        cfg.optimize_thread_count = v;
    }
    if let Some(v) = invert_to_forward_scan_ratio {
        cfg.invert_to_forward_scan_ratio = v;
    }
    if let Some(v) = brute_force_by_keys_ratio {
        cfg.brute_force_by_keys_ratio = v;
    }
    if let Some(v) = memory_limit_mb {
        cfg.memory_limit_bytes = v.saturating_mul(1024 * 1024);
    }
    if let Some(v) = wal_flush_every_docs {
        cfg.wal_flush_every_docs = v;
    }
    if let Some(v) = wal_fsync_every_docs {
        cfg.wal_fsync_every_docs = v;
    }

    finch_db::initialize_global_config(cfg)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()).into())
}

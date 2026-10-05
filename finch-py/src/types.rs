//! Python enums for finch types

use pyo3::prelude::*;

#[pyclass(module = "finch._finch")]
#[derive(Clone, Debug)]
pub struct PyStatus {
    pub code: PyStatusCode,
    pub message: String,
}

#[pymethods]
impl PyStatus {
    #[new]
    #[pyo3(signature = (code=PyStatusCode::Ok, message=""))]
    fn new(code: PyStatusCode, message: &str) -> Self {
        Self {
            code,
            message: message.to_string(),
        }
    }

    #[getter]
    fn code(&self) -> PyStatusCode {
        self.code
    }

    #[getter]
    fn message(&self) -> &str {
        &self.message
    }

    fn is_ok(&self) -> bool {
        self.code == PyStatusCode::Ok
    }

    fn is_err(&self) -> bool {
        self.code != PyStatusCode::Ok
    }

    fn __repr__(&self) -> String {
        if self.message.is_empty() {
            format!("Status(code={:?})", self.code)
        } else {
            format!("Status(code={:?}, message='{}')", self.code, self.message)
        }
    }
}

impl From<finch_types::Status> for PyStatus {
    fn from(s: finch_types::Status) -> Self {
        Self {
            code: s.code.into(),
            message: s.message,
        }
    }
}

#[pyclass(module = "finch._finch", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PyDataType {
    Undefined = 0,
    Binary = 1,
    String = 2,
    Bool = 3,
    Int32 = 4,
    Int64 = 5,
    Uint32 = 6,
    Uint64 = 7,
    Float32 = 8,
    Float64 = 9,
    Int8 = 100,
    Int16 = 101,
    Uint8 = 102,
    Uint16 = 103,
    Float16 = 104,
    Bytes = 105,
    VectorBinary32 = 20,
    VectorBinary64 = 21,
    VectorFp16 = 22,
    VectorFp32 = 23,
    VectorFp64 = 24,
    VectorInt4 = 25,
    VectorInt8 = 26,
    VectorInt16 = 27,
    VectorBool = 120,
    VectorInt32 = 121,
    VectorInt64 = 122,
    VectorUint32 = 123,
    VectorUint64 = 124,
    SparseFp16 = 30,
    SparseFp32 = 31,
    ArrayBinary = 40,
    ArrayString = 41,
    ArrayBool = 42,
    ArrayInt32 = 43,
    ArrayInt64 = 44,
    ArrayUint32 = 45,
    ArrayUint64 = 46,
    ArrayFp32 = 47,
    ArrayFp64 = 48,
}

impl From<PyDataType> for finch_types::DataType {
    fn from(t: PyDataType) -> Self {
        match t {
            PyDataType::Undefined => finch_types::DataType::Undefined,
            PyDataType::Binary => finch_types::DataType::Binary,
            PyDataType::String => finch_types::DataType::String,
            PyDataType::Bool => finch_types::DataType::Bool,
            PyDataType::Int32 => finch_types::DataType::Int32,
            PyDataType::Int64 => finch_types::DataType::Int64,
            PyDataType::Uint32 => finch_types::DataType::Uint32,
            PyDataType::Uint64 => finch_types::DataType::Uint64,
            PyDataType::Float32 => finch_types::DataType::Float32,
            PyDataType::Float64 => finch_types::DataType::Float64,
            PyDataType::Int8 => finch_types::DataType::Int8,
            PyDataType::Int16 => finch_types::DataType::Int16,
            PyDataType::Uint8 => finch_types::DataType::Uint8,
            PyDataType::Uint16 => finch_types::DataType::Uint16,
            PyDataType::Float16 => finch_types::DataType::Float16,
            PyDataType::Bytes => finch_types::DataType::Bytes,
            PyDataType::VectorBinary32 => finch_types::DataType::VectorBinary32,
            PyDataType::VectorBinary64 => finch_types::DataType::VectorBinary64,
            PyDataType::VectorFp16 => finch_types::DataType::VectorFp16,
            PyDataType::VectorFp32 => finch_types::DataType::VectorFp32,
            PyDataType::VectorFp64 => finch_types::DataType::VectorFp64,
            PyDataType::VectorInt4 => finch_types::DataType::VectorInt4,
            PyDataType::VectorInt8 => finch_types::DataType::VectorInt8,
            PyDataType::VectorInt16 => finch_types::DataType::VectorInt16,
            PyDataType::VectorInt32 => finch_types::DataType::VectorInt32,
            PyDataType::VectorInt64 => finch_types::DataType::VectorInt64,
            PyDataType::VectorUint32 => finch_types::DataType::VectorUint32,
            PyDataType::VectorUint64 => finch_types::DataType::VectorUint64,
            PyDataType::VectorBool => finch_types::DataType::VectorBool,
            PyDataType::SparseFp16 => finch_types::DataType::SparseFp16,
            PyDataType::SparseFp32 => finch_types::DataType::SparseFp32,
            PyDataType::ArrayBinary => finch_types::DataType::ArrayBinary,
            PyDataType::ArrayString => finch_types::DataType::ArrayString,
            PyDataType::ArrayBool => finch_types::DataType::ArrayBool,
            PyDataType::ArrayInt32 => finch_types::DataType::ArrayInt32,
            PyDataType::ArrayInt64 => finch_types::DataType::ArrayInt64,
            PyDataType::ArrayUint32 => finch_types::DataType::ArrayUint32,
            PyDataType::ArrayUint64 => finch_types::DataType::ArrayUint64,
            PyDataType::ArrayFp32 => finch_types::DataType::ArrayFp32,
            PyDataType::ArrayFp64 => finch_types::DataType::ArrayFp64,
        }
    }
}

impl From<finch_types::DataType> for PyDataType {
    fn from(t: finch_types::DataType) -> Self {
        match t {
            finch_types::DataType::Undefined => PyDataType::Undefined,
            finch_types::DataType::Binary => PyDataType::Binary,
            finch_types::DataType::String => PyDataType::String,
            finch_types::DataType::Bool => PyDataType::Bool,
            finch_types::DataType::Int32 => PyDataType::Int32,
            finch_types::DataType::Int64 => PyDataType::Int64,
            finch_types::DataType::Uint32 => PyDataType::Uint32,
            finch_types::DataType::Uint64 => PyDataType::Uint64,
            finch_types::DataType::Float32 => PyDataType::Float32,
            finch_types::DataType::Float64 => PyDataType::Float64,
            finch_types::DataType::Int8 => PyDataType::Int8,
            finch_types::DataType::Int16 => PyDataType::Int16,
            finch_types::DataType::Uint8 => PyDataType::Uint8,
            finch_types::DataType::Uint16 => PyDataType::Uint16,
            finch_types::DataType::Float16 => PyDataType::Float16,
            finch_types::DataType::Bytes => PyDataType::Bytes,
            finch_types::DataType::VectorBinary32 => PyDataType::VectorBinary32,
            finch_types::DataType::VectorBinary64 => PyDataType::VectorBinary64,
            finch_types::DataType::VectorFp16 => PyDataType::VectorFp16,
            finch_types::DataType::VectorFp32 => PyDataType::VectorFp32,
            finch_types::DataType::VectorFp64 => PyDataType::VectorFp64,
            finch_types::DataType::VectorInt4 => PyDataType::VectorInt4,
            finch_types::DataType::VectorInt8 => PyDataType::VectorInt8,
            finch_types::DataType::VectorInt16 => PyDataType::VectorInt16,
            finch_types::DataType::VectorBool => PyDataType::VectorBool,
            finch_types::DataType::VectorInt32 => PyDataType::VectorInt32,
            finch_types::DataType::VectorInt64 => PyDataType::VectorInt64,
            finch_types::DataType::VectorUint32 => PyDataType::VectorUint32,
            finch_types::DataType::VectorUint64 => PyDataType::VectorUint64,
            finch_types::DataType::SparseFp16 => PyDataType::SparseFp16,
            finch_types::DataType::SparseFp32 => PyDataType::SparseFp32,
            finch_types::DataType::ArrayBinary => PyDataType::ArrayBinary,
            finch_types::DataType::ArrayString => PyDataType::ArrayString,
            finch_types::DataType::ArrayBool => PyDataType::ArrayBool,
            finch_types::DataType::ArrayInt32 => PyDataType::ArrayInt32,
            finch_types::DataType::ArrayInt64 => PyDataType::ArrayInt64,
            finch_types::DataType::ArrayUint32 => PyDataType::ArrayUint32,
            finch_types::DataType::ArrayUint64 => PyDataType::ArrayUint64,
            finch_types::DataType::ArrayFp32 => PyDataType::ArrayFp32,
            finch_types::DataType::ArrayFp64 => PyDataType::ArrayFp64,
        }
    }
}

#[pyclass(module = "finch._finch", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PyMetricType {
    Undefined = 0,
    L2 = 1,
    InnerProduct = 2,
    Cosine = 3,
    MipsL2 = 4,
    Hamming = 5,
}

impl From<PyMetricType> for finch_types::MetricType {
    fn from(t: PyMetricType) -> Self {
        match t {
            PyMetricType::Undefined => finch_types::MetricType::Undefined,
            PyMetricType::L2 => finch_types::MetricType::L2,
            PyMetricType::InnerProduct => finch_types::MetricType::InnerProduct,
            PyMetricType::Cosine => finch_types::MetricType::Cosine,
            PyMetricType::MipsL2 => finch_types::MetricType::MipsL2,
            PyMetricType::Hamming => finch_types::MetricType::Hamming,
        }
    }
}

impl From<finch_types::MetricType> for PyMetricType {
    fn from(t: finch_types::MetricType) -> Self {
        match t {
            finch_types::MetricType::Undefined => PyMetricType::Undefined,
            finch_types::MetricType::L2 => PyMetricType::L2,
            finch_types::MetricType::InnerProduct => PyMetricType::InnerProduct,
            finch_types::MetricType::Cosine => PyMetricType::Cosine,
            finch_types::MetricType::MipsL2 => PyMetricType::MipsL2,
            finch_types::MetricType::Hamming => PyMetricType::Hamming,
        }
    }
}

#[pyclass(module = "finch._finch", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PyQuantizeType {
    Undefined = 0,
    Fp16 = 1,
    Int8 = 2,
    Int4 = 3,
}

impl From<PyQuantizeType> for finch_types::QuantizeType {
    fn from(t: PyQuantizeType) -> Self {
        match t {
            PyQuantizeType::Undefined => finch_types::QuantizeType::Undefined,
            PyQuantizeType::Fp16 => finch_types::QuantizeType::Fp16,
            PyQuantizeType::Int8 => finch_types::QuantizeType::Int8,
            PyQuantizeType::Int4 => finch_types::QuantizeType::Int4,
        }
    }
}

impl From<finch_types::QuantizeType> for PyQuantizeType {
    fn from(t: finch_types::QuantizeType) -> Self {
        match t {
            finch_types::QuantizeType::Undefined => PyQuantizeType::Undefined,
            finch_types::QuantizeType::Fp16 => PyQuantizeType::Fp16,
            finch_types::QuantizeType::Int8 => PyQuantizeType::Int8,
            finch_types::QuantizeType::Int4 => PyQuantizeType::Int4,
        }
    }
}

#[pyclass(module = "finch._finch", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PyIndexType {
    Undefined = 0,
    Hnsw = 1,
    Ivf = 3,
    Flat = 4,
    Invert = 10,
    HnswSparse = 11,
    FlatSparse = 12,
}

#[pyclass(module = "finch._finch", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PyStatusCode {
    Ok = 0,
    NotFound = 1,
    AlreadyExists = 2,
    InvalidArgument = 3,
    IoError = 4,
    Internal = 5,
    Unimplemented = 6,
    OutOfRange = 7,
    ResourceExhausted = 8,
    Cancelled = 9,
    Unknown = 10,
    PermissionDenied = 11,
}

#[pyclass(module = "finch._finch", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PyLogType {
    Console = 0,
    File = 1,
}

#[pyclass(module = "finch._finch", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PyLogLevel {
    Debug = 0,
    Info = 1,
    Warn = 2,
    Error = 3,
    Fatal = 4,
}

#[pyclass(module = "finch._finch", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PyFileFormat {
    Raw = 0,
    ArrowIpc = 1,
    Parquet = 2,
    Protobuf = 3,
}

impl From<PyFileFormat> for finch_types::FileFormat {
    fn from(t: PyFileFormat) -> Self {
        match t {
            PyFileFormat::Raw => finch_types::FileFormat::Raw,
            PyFileFormat::ArrowIpc => finch_types::FileFormat::ArrowIpc,
            PyFileFormat::Parquet => finch_types::FileFormat::Parquet,
            PyFileFormat::Protobuf => finch_types::FileFormat::Protobuf,
        }
    }
}

impl From<finch_types::FileFormat> for PyFileFormat {
    fn from(t: finch_types::FileFormat) -> Self {
        match t {
            finch_types::FileFormat::Raw => PyFileFormat::Raw,
            finch_types::FileFormat::ArrowIpc => PyFileFormat::ArrowIpc,
            finch_types::FileFormat::Parquet => PyFileFormat::Parquet,
            finch_types::FileFormat::Protobuf => PyFileFormat::Protobuf,
        }
    }
}

#[pyclass(module = "finch._finch", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PyStorageType {
    None = 0,
    Mmap = 1,
    Memory = 2,
    BufferPool = 3,
}

impl From<PyStorageType> for finch_types::StorageType {
    fn from(t: PyStorageType) -> Self {
        match t {
            PyStorageType::None => finch_types::StorageType::None,
            PyStorageType::Mmap => finch_types::StorageType::Mmap,
            PyStorageType::Memory => finch_types::StorageType::Memory,
            PyStorageType::BufferPool => finch_types::StorageType::BufferPool,
        }
    }
}

impl From<finch_types::StorageType> for PyStorageType {
    fn from(t: finch_types::StorageType) -> Self {
        match t {
            finch_types::StorageType::None => PyStorageType::None,
            finch_types::StorageType::Mmap => PyStorageType::Mmap,
            finch_types::StorageType::Memory => PyStorageType::Memory,
            finch_types::StorageType::BufferPool => PyStorageType::BufferPool,
        }
    }
}

impl From<PyLogLevel> for finch_types::LogLevel {
    fn from(v: PyLogLevel) -> Self {
        match v {
            PyLogLevel::Debug => finch_types::LogLevel::Debug,
            PyLogLevel::Info => finch_types::LogLevel::Info,
            PyLogLevel::Warn => finch_types::LogLevel::Warn,
            PyLogLevel::Error => finch_types::LogLevel::Error,
            PyLogLevel::Fatal => finch_types::LogLevel::Fatal,
        }
    }
}

impl From<finch_types::StatusCode> for PyStatusCode {
    fn from(v: finch_types::StatusCode) -> Self {
        match v {
            finch_types::StatusCode::Ok => PyStatusCode::Ok,
            finch_types::StatusCode::NotFound => PyStatusCode::NotFound,
            finch_types::StatusCode::AlreadyExists => PyStatusCode::AlreadyExists,
            finch_types::StatusCode::InvalidArgument => PyStatusCode::InvalidArgument,
            finch_types::StatusCode::IoError => PyStatusCode::IoError,
            finch_types::StatusCode::Internal => PyStatusCode::Internal,
            finch_types::StatusCode::Unimplemented => PyStatusCode::Unimplemented,
            finch_types::StatusCode::OutOfRange => PyStatusCode::OutOfRange,
            finch_types::StatusCode::ResourceExhausted => PyStatusCode::ResourceExhausted,
            finch_types::StatusCode::Cancelled => PyStatusCode::Cancelled,
            finch_types::StatusCode::Unknown => PyStatusCode::Unknown,
            finch_types::StatusCode::PermissionDenied => PyStatusCode::PermissionDenied,
        }
    }
}

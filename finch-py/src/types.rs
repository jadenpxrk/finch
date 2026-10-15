use pyo3::prelude::*;

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

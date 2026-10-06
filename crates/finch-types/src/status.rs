use std::fmt;
use thiserror::Error;

/// Status codes for operation results
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum StatusCode {
    /// Success.
    Ok = 0,
    /// The item does not exist.
    NotFound = 1,
    /// An item with the same name or key exists.
    AlreadyExists = 2,
    /// The input is not valid.
    InvalidArgument = 3,
    /// A file or storage operation failed.
    IoError = 4,
    /// An internal invariant failed.
    Internal = 5,
    /// The operation is not available.
    Unimplemented = 6,
    /// A value is outside its permitted range.
    OutOfRange = 7,
    /// A memory or other resource limit stopped the operation.
    ResourceExhausted = 8,
    /// The caller stopped the operation.
    Cancelled = 9,
    /// The cause is not known.
    Unknown = 10,
    /// The caller does not have permission, for example on a read-only collection.
    PermissionDenied = 11,
}

/// Operation status with code and message
#[derive(Debug, Clone, Error)]
pub struct Status {
    /// Category of the result.
    pub code: StatusCode,
    /// Text for a person to read.
    pub message: String,
}

impl Status {
    /// A status with `code` and `msg`.
    pub fn new(code: StatusCode, msg: impl Into<String>) -> Self {
        Status {
            code,
            message: msg.into(),
        }
    }

    /// A `NotFound` status.
    pub fn not_found(msg: impl Into<String>) -> Self {
        Status {
            code: StatusCode::NotFound,
            message: msg.into(),
        }
    }

    /// An `AlreadyExists` status.
    pub fn already_exists(msg: impl Into<String>) -> Self {
        Status {
            code: StatusCode::AlreadyExists,
            message: msg.into(),
        }
    }

    /// An `InvalidArgument` status.
    pub fn invalid_argument(msg: impl Into<String>) -> Self {
        Status {
            code: StatusCode::InvalidArgument,
            message: msg.into(),
        }
    }

    /// An `IoError` status.
    pub fn io_error(msg: impl Into<String>) -> Self {
        Status {
            code: StatusCode::IoError,
            message: msg.into(),
        }
    }

    /// An `Internal` status.
    pub fn internal(msg: impl Into<String>) -> Self {
        Status {
            code: StatusCode::Internal,
            message: msg.into(),
        }
    }

    /// An `Unimplemented` status.
    pub fn unimplemented(msg: impl Into<String>) -> Self {
        Status {
            code: StatusCode::Unimplemented,
            message: msg.into(),
        }
    }

    /// An `OutOfRange` status.
    pub fn out_of_range(msg: impl Into<String>) -> Self {
        Status {
            code: StatusCode::OutOfRange,
            message: msg.into(),
        }
    }

    /// A `PermissionDenied` status.
    pub fn permission_denied(msg: impl Into<String>) -> Self {
        Status {
            code: StatusCode::PermissionDenied,
            message: msg.into(),
        }
    }

    /// Whether the code is `Ok`.
    pub fn ok(&self) -> bool {
        self.code == StatusCode::Ok
    }

    /// Category of the result.
    pub fn code(&self) -> StatusCode {
        self.code
    }

    /// Text for a person to read.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Whether the code is `Ok`.
    pub fn is_ok(&self) -> bool {
        self.code == StatusCode::Ok
    }

    /// Whether the code is not `Ok`.
    pub fn is_err(&self) -> bool {
        self.code != StatusCode::Ok
    }

    /// Whether the code is `NotFound`.
    pub fn is_not_found(&self) -> bool {
        self.code == StatusCode::NotFound
    }

    /// Whether the code is `AlreadyExists`.
    pub fn is_already_exists(&self) -> bool {
        self.code == StatusCode::AlreadyExists
    }

    /// default status message for a given code.
    pub fn default_message(code: StatusCode) -> &'static str {
        match code {
            StatusCode::Ok => "OK",
            StatusCode::NotFound => "Not found",
            StatusCode::AlreadyExists => "Already exists",
            StatusCode::InvalidArgument => "Invalid argument",
            StatusCode::PermissionDenied => "Permission denied",
            StatusCode::ResourceExhausted => "Resource exhausted",
            StatusCode::Internal => "Internal error",
            StatusCode::Unimplemented => "Not supported",
            StatusCode::Unknown => "Unknown error",
            StatusCode::IoError => "IO error",
            StatusCode::OutOfRange => "Out of range",
            StatusCode::Cancelled => "Cancelled",
        }
    }
}

impl Default for Status {
    fn default() -> Self {
        Status {
            code: StatusCode::Ok,
            message: String::new(),
        }
    }
}

impl PartialEq for Status {
    fn eq(&self, other: &Self) -> bool {
        if self.code != other.code {
            return false;
        }
        if self.code == StatusCode::Ok {
            return true;
        }
        self.message == other.message
    }
}

impl Eq for Status {}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // stream output prints `OK` for success, otherwise
        // `Status(<default message>, <message>)`.
        if self.code == StatusCode::Ok {
            write!(f, "OK")
        } else {
            write!(
                f,
                "Status({}, {})",
                Status::default_message(self.code),
                self.message
            )
        }
    }
}

impl From<std::io::Error> for Status {
    fn from(e: std::io::Error) -> Self {
        Status::io_error(e.to_string())
    }
}

/// Result type for finch operations
pub type ZResult<T> = Result<T, Status>;

/// Convenience macro for returning errors
#[macro_export]
macro_rules! finch_bail {
    ($code:ident, $msg:expr) => {
        return Err($crate::status::Status::$code($msg))
    };
    ($code:ident, $fmt:literal, $($arg:expr),*) => {
        return Err($crate::status::Status::$code(format!($fmt, $($arg),*)))
    };
}

/// Convenience macro for checking conditions
#[macro_export]
macro_rules! finch_ensure {
    ($cond:expr, $code:ident, $msg:expr) => {
        if !($cond) {
            return Err($crate::status::Status::$code($msg));
        }
    };
}

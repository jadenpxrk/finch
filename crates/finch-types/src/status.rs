use std::fmt;
use thiserror::Error;

/// Status codes for operation results
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum StatusCode {
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

/// Operation status with code and message
#[derive(Debug, Clone, Error)]
pub struct Status {
    pub code: StatusCode,
    pub message: String,
}

impl Status {
    pub fn new(code: StatusCode, msg: impl Into<String>) -> Self {
        Status {
            code,
            message: msg.into(),
        }
    }

    pub fn not_found(msg: impl Into<String>) -> Self {
        Status {
            code: StatusCode::NotFound,
            message: msg.into(),
        }
    }

    pub fn already_exists(msg: impl Into<String>) -> Self {
        Status {
            code: StatusCode::AlreadyExists,
            message: msg.into(),
        }
    }

    pub fn invalid_argument(msg: impl Into<String>) -> Self {
        Status {
            code: StatusCode::InvalidArgument,
            message: msg.into(),
        }
    }

    pub fn io_error(msg: impl Into<String>) -> Self {
        Status {
            code: StatusCode::IoError,
            message: msg.into(),
        }
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Status {
            code: StatusCode::Internal,
            message: msg.into(),
        }
    }

    pub fn unimplemented(msg: impl Into<String>) -> Self {
        Status {
            code: StatusCode::Unimplemented,
            message: msg.into(),
        }
    }

    pub fn out_of_range(msg: impl Into<String>) -> Self {
        Status {
            code: StatusCode::OutOfRange,
            message: msg.into(),
        }
    }

    pub fn resource_exhausted(msg: impl Into<String>) -> Self {
        Status {
            code: StatusCode::ResourceExhausted,
            message: msg.into(),
        }
    }

    pub fn permission_denied(msg: impl Into<String>) -> Self {
        Status {
            code: StatusCode::PermissionDenied,
            message: msg.into(),
        }
    }

    /// `ok()` method name (C++ Status::ok()).
    pub fn ok(&self) -> bool {
        self.code == StatusCode::Ok
    }

    /// `code()` method name (C++ Status::code()).
    pub fn code(&self) -> StatusCode {
        self.code
    }

    /// `message()` method name (C++ Status::message()).
    pub fn message(&self) -> &str {
        &self.message
    }

    /// `c_str()` accessor.
    pub fn c_str(&self) -> &str {
        &self.message
    }

    pub fn is_ok(&self) -> bool {
        self.code == StatusCode::Ok
    }

    pub fn is_err(&self) -> bool {
        self.code != StatusCode::Ok
    }

    pub fn is_not_found(&self) -> bool {
        self.code == StatusCode::NotFound
    }

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
            // Finch-only best-effort messages.
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

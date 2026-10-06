//! The collection manifest and its codec. Writes protobuf through `prost` messages declared in
//! Rust, so the build does not need `protoc`. Used by `finch-db`.

#![deny(missing_docs)]

pub mod convert;
pub mod manifest;

pub use manifest::*;

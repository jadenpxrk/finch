//! The embedded vector database. A `Collection` is a directory of segments, a write-ahead log,
//! a primary-key map, delete bitmaps, and a manifest; this crate implements writes, vector and
//! filter queries, the SQL-like filter language, index builds, and compaction. Used by
//! `finch-memory` and the Python and Node bindings.

#![deny(missing_docs)]

mod collection;
mod collection_files;
mod config;
mod ddl_expr;
mod delete_store;
mod doc_fetch;
mod group_by;
mod id_map;
mod index;
// Integration tests open inverted indexes and manifests directly.
#[doc(hidden)]
pub mod invert;
mod multi_query;
mod output_projection;
mod profile;
mod query_filter;
mod query_output;
mod row_locator;
mod segment;
mod sorted_file;
mod sql_query;
pub mod sqlengine;
mod system_projection;
mod vector_literal;
mod vector_normalization;
mod vector_search_field;
#[doc(hidden)]
pub mod version;
mod wal;
mod write_normalization;

pub use collection::Collection;
pub use config::initialize_global_config;
pub use profile::QueryProfile;

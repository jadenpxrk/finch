//! The embedded vector database. A `Collection` is a directory of segments, a write-ahead log,
//! a primary-key map, delete bitmaps, and a manifest; this crate implements writes, vector and
//! filter queries, the SQL-like filter language, index builds, and compaction. Used by
//! `finch-memory` and the Python and Node bindings.

pub mod collection;
mod collection_files;
pub mod config;
mod ddl_expr;
pub mod delete_store;
mod doc_fetch;
mod group_by;
pub mod id_map;
pub mod index;
pub mod invert;
mod multi_query;
mod output_projection;
pub mod profile;
mod query_filter;
mod query_output;
mod row_locator;
pub mod segment;
mod sorted_file;
mod sql_query;
pub mod sqlengine;
mod system_projection;
mod vector_literal;
mod vector_normalization;
mod vector_search_field;
pub mod version;
mod wal;
mod write_normalization;

pub use collection::Collection;
pub use config::{ensure_rayon_initialized, global_config, init_logging, initialize_global_config};
pub use profile::QueryProfile;

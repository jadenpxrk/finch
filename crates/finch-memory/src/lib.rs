//! Agent memory stored in `finch-db` collections. Records evidence and the claims drawn from it,
//! keeps each claim's history on two timelines (when it held and when Finch learned it), and
//! packs the current state into a context for a language model. Used by the Python and Node
//! bindings and by `finch-memory-bench`.

pub mod context;
pub mod eval;
pub mod ingest;
pub mod json_api;
mod postgres;
pub use postgres::drop_store;
pub mod retrieval;
pub mod row;
pub mod schema;
pub mod source_index;
pub mod state;
pub mod store;
mod table;
pub mod types;

pub use context::*;
pub use eval::*;
pub use ingest::*;
pub use retrieval::*;
pub use row::*;
pub use schema::*;
pub use source_index::*;
pub use state::*;
pub use store::*;
pub use types::*;

#[cfg(test)]
pub(crate) static TEST_STORE_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The Postgres server with pgvector that the tests use.
#[cfg(test)]
pub(crate) fn test_postgres_url() -> String {
    std::env::var("FINCH_MEMORY_TEST_POSTGRES_URL")
        .expect("set FINCH_MEMORY_TEST_POSTGRES_URL to a Postgres server with pgvector")
}

/// A new, empty store named after `dir`; the directory exists so the test can delete it.
#[cfg(test)]
pub(crate) fn test_store(
    dir: &std::path::Path,
    embedding_dim: usize,
) -> finch_types::ZResult<MemoryStore> {
    std::fs::create_dir_all(dir).map_err(|e| finch_types::Status::io_error(e.to_string()))?;
    let name = test_store_name(dir);
    drop_store(&test_postgres_url(), &name)?;
    MemoryStore::create(&test_postgres_url(), &name, embedding_dim, false)
}

/// The store `test_store` created for `dir`.
#[cfg(test)]
pub(crate) fn reopen_test_store(dir: &std::path::Path) -> finch_types::ZResult<MemoryStore> {
    MemoryStore::open(&test_postgres_url(), &test_store_name(dir))
}

#[cfg(test)]
fn test_store_name(dir: &std::path::Path) -> String {
    format!("t_{}", ingest::stable_hash_hex(&[&dir.to_string_lossy()]))
}

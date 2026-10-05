//! Agent memory stored in `finch-db` collections. Records evidence and the claims drawn from it,
//! keeps each claim's history on two timelines (when it held and when Finch learned it), and
//! packs the current state into a context for a language model. Used by the Python and Node
//! bindings and by `finch-memory-bench`.

pub mod context;
pub mod eval;
pub mod ingest;
pub mod json_api;
#[cfg(feature = "postgres")]
mod postgres;
pub mod retrieval;
pub mod row;
pub mod schema;
pub mod source_index;
pub mod state;
pub mod store;
pub mod table;
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

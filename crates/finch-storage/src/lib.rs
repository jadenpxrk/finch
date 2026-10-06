//! Columnar storage for segment documents (Arrow IPC or Parquet files, read through mmap or
//! heap buffers), and the file storage and memory maps for vector index files. Used by
//! `finch-db`.

#![deny(missing_docs)]

pub(crate) mod arrow_schema;
pub mod forward;
pub mod storage;

pub use arrow_schema::data_type_to_arrow;
pub use forward::{
    ipc_schema_for_collection, ForwardStoreOpenOptions, MemoryForwardStore, MmapForwardStore,
    ParquetReadMode,
};
pub use storage::{get_mmap, FileStorage, StorageSegment};

//! Columnar storage for segment documents (Arrow IPC or Parquet files, read through mmap or
//! heap buffers) and the file, mmap, and in-memory byte stores that vector index files are
//! written to and read from. Used by `finch-db`.

pub mod arrow_schema;
pub mod forward;
pub mod storage;

pub use arrow_schema::{collection_schema_to_arrow, data_type_to_arrow};
pub use forward::{
    ipc_schema_for_collection, ForwardStoreOpenOptions, MemoryForwardStore, MmapForwardStore,
    ParquetReadMode,
};
pub use storage::{get_mmap, FileStorage, MemStorage, MmapStorage, StorageSegment};

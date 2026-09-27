//! Deleted document bitmap store

use finch_types::{Status, ZResult};
use roaring::RoaringTreemap;
use std::fs;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub struct DeleteStore {
    bitmap: Arc<RoaringTreemap>,
    base_path: PathBuf,
    pub path_suffix: u32,
    modified_since_last_snapshot: bool,
}

impl DeleteStore {
    pub fn new(base_path: impl AsRef<Path>, suffix: u32) -> Self {
        DeleteStore {
            bitmap: Arc::new(RoaringTreemap::new()),
            base_path: base_path.as_ref().to_path_buf(),
            path_suffix: suffix,
            modified_since_last_snapshot: false,
        }
    }

    pub fn load(base_path: impl AsRef<Path>, suffix: u32) -> ZResult<Self> {
        let path = base_path.as_ref().join(format!("delete_{}.bitmap", suffix));
        if !path.exists() {
            return Err(Status::io_error(format!(
                "delete bitmap missing at {:?}",
                path
            )));
        }

        let file = fs::File::open(&path).map_err(|e| Status::io_error(e.to_string()))?;
        let reader = BufReader::new(file);
        let bitmap = RoaringTreemap::deserialize_from(reader)
            .map_err(|e| Status::io_error(e.to_string()))?;

        Ok(DeleteStore {
            bitmap: Arc::new(bitmap),
            base_path: base_path.as_ref().to_path_buf(),
            path_suffix: suffix,
            modified_since_last_snapshot: false,
        })
    }

    pub fn mark_deleted(&mut self, doc_id: u64) {
        Arc::make_mut(&mut self.bitmap).insert(doc_id);
        self.modified_since_last_snapshot = true;
    }

    pub fn is_deleted(&self, doc_id: u64) -> bool {
        self.bitmap.contains(doc_id)
    }

    /// Writes the bitmap under the next suffix; call `commit_snapshot` once the manifest names it.
    pub fn snapshot(&self) -> ZResult<u32> {
        let new_suffix = self.path_suffix + 1;
        let path = self.base_path.join(format!("delete_{}.bitmap", new_suffix));
        let file = fs::File::create(&path).map_err(|e| Status::io_error(e.to_string()))?;
        let mut writer = BufWriter::new(file);
        self.bitmap
            .serialize_into(&mut writer)
            .map_err(|e| Status::io_error(e.to_string()))?;
        writer
            .flush()
            .map_err(|e| Status::io_error(e.to_string()))?;
        let _ = writer.get_ref().sync_all();
        let _ = fs::File::open(&self.base_path).and_then(|d| d.sync_all());
        Ok(new_suffix)
    }

    // Until the manifest commits, the WAL is the only durable copy of new tombstones, so the store stays dirty.
    pub fn commit_snapshot(&mut self, suffix: u32) {
        self.path_suffix = suffix;
        self.modified_since_last_snapshot = false;
    }

    /// Persist the current bitmap without advancing suffix.
    /// Used for the initial `delete_0.bitmap` snapshot at collection creation.
    pub fn snapshot_current(&self) -> ZResult<u32> {
        let path = self
            .base_path
            .join(format!("delete_{}.bitmap", self.path_suffix));
        let file = fs::File::create(&path).map_err(|e| Status::io_error(e.to_string()))?;
        let mut writer = BufWriter::new(file);
        self.bitmap
            .serialize_into(&mut writer)
            .map_err(|e| Status::io_error(e.to_string()))?;
        writer
            .flush()
            .map_err(|e| Status::io_error(e.to_string()))?;
        let _ = writer.get_ref().sync_all();
        let _ = fs::File::open(&self.base_path).and_then(|d| d.sync_all());
        Ok(self.path_suffix)
    }

    pub fn deleted_count(&self) -> u64 {
        self.bitmap.len()
    }

    pub fn bitmap(&self) -> Arc<RoaringTreemap> {
        self.bitmap.clone()
    }

    pub fn modified_since_last_snapshot(&self) -> bool {
        self.modified_since_last_snapshot
    }
}

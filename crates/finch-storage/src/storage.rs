//! Named byte segments in a directory, one file per segment, and read-only memory maps of files.

use finch_types::{Status, ZResult};
use memmap2::Mmap;
use std::fs;
use std::path::{Path, PathBuf};

/// Read/write abstraction for named byte segments
pub trait StorageSegment: Send + Sync {
    /// The bytes of the segment; a `NotFound` error when it does not exist.
    fn read(&self, segment_id: &str) -> ZResult<Vec<u8>>;
    /// Replaces the segment with `data`.
    fn write(&mut self, segment_id: &str, data: &[u8]) -> ZResult<()>;
    /// Adds `data` at the end of the segment, and creates the segment when needed.
    fn append(&mut self, segment_id: &str, data: &[u8]) -> ZResult<()>;
    /// Whether the segment exists.
    fn exists(&self, segment_id: &str) -> bool;
    /// Size of the segment in bytes.
    fn size(&self, segment_id: &str) -> ZResult<u64>;
    /// Removes the segment; a missing segment is not an error.
    fn delete(&mut self, segment_id: &str) -> ZResult<()>;
}

/// File-system backed storage (one file per segment in a directory)
pub struct FileStorage {
    base_path: PathBuf,
}

impl FileStorage {
    /// Storage in `base_path`, which it creates when needed.
    pub fn new(base_path: impl AsRef<Path>) -> ZResult<Self> {
        let path = base_path.as_ref().to_path_buf();
        fs::create_dir_all(&path).map_err(|e| Status::io_error(e.to_string()))?;
        // Best-effort crash safety for directory creation: sync the new dir
        // and its parent (where the directory entry lives).
        Self::sync_dir_best_effort(&path);
        if let Some(parent) = path.parent() {
            Self::sync_dir_best_effort(parent);
        }
        Ok(FileStorage { base_path: path })
    }

    fn segment_path(&self, segment_id: &str) -> PathBuf {
        self.base_path.join(segment_id)
    }

    fn sync_dir_best_effort(path: &Path) {
        let _ = fs::File::open(path).and_then(|d| d.sync_all());
    }
}

impl StorageSegment for FileStorage {
    fn read(&self, segment_id: &str) -> ZResult<Vec<u8>> {
        let path = self.segment_path(segment_id);
        fs::read(&path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                Status::not_found(format!("segment '{}' not found", segment_id))
            } else {
                Status::io_error(e.to_string())
            }
        })
    }

    fn write(&mut self, segment_id: &str, data: &[u8]) -> ZResult<()> {
        let path = self.segment_path(segment_id);
        let tmp_path = self.segment_path(&format!("{}.tmp", segment_id));
        fs::write(&tmp_path, data).map_err(|e| Status::io_error(e.to_string()))?;

        // Crash safety: sync file, then rename, then sync directory.
        if let Ok(f) = fs::File::open(&tmp_path) {
            let _ = f.sync_all();
        }
        fs::rename(&tmp_path, &path).map_err(|e| Status::io_error(e.to_string()))?;
        Self::sync_dir_best_effort(&self.base_path);
        Ok(())
    }

    fn append(&mut self, segment_id: &str, data: &[u8]) -> ZResult<()> {
        use std::io::Write;
        let path = self.segment_path(segment_id);
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| Status::io_error(e.to_string()))?;
        file.write_all(data)
            .map_err(|e| Status::io_error(e.to_string()))
    }

    fn exists(&self, segment_id: &str) -> bool {
        self.segment_path(segment_id).exists()
    }

    fn size(&self, segment_id: &str) -> ZResult<u64> {
        let path = self.segment_path(segment_id);
        let meta = fs::metadata(&path).map_err(|e| Status::io_error(e.to_string()))?;
        Ok(meta.len())
    }

    fn delete(&mut self, segment_id: &str) -> ZResult<()> {
        let path = self.segment_path(segment_id);
        if path.exists() {
            fs::remove_file(&path).map_err(|e| Status::io_error(e.to_string()))?;
            Self::sync_dir_best_effort(&self.base_path);
        }
        Ok(())
    }
}

/// Get a zero-copy mmap slice for a segment file
pub fn get_mmap(path: impl AsRef<Path>) -> ZResult<Mmap> {
    let file = fs::File::open(path.as_ref()).map_err(|e| Status::io_error(e.to_string()))?;
    // SAFETY: segment files are written once and not modified while mapped.
    unsafe { Mmap::map(&file) }.map_err(|e| Status::io_error(e.to_string()))
}

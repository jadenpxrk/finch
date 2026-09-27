//! Storage abstraction: file, memory, and mmap backends

use finch_types::{Status, ZResult};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Read/write abstraction for named byte segments
pub trait StorageSegment: Send + Sync {
    fn read(&self, segment_id: &str) -> ZResult<Vec<u8>>;
    fn write(&mut self, segment_id: &str, data: &[u8]) -> ZResult<()>;
    fn append(&mut self, segment_id: &str, data: &[u8]) -> ZResult<()>;
    fn exists(&self, segment_id: &str) -> bool;
    fn size(&self, segment_id: &str) -> ZResult<u64>;
    fn delete(&mut self, segment_id: &str) -> ZResult<()>;
}

// ── File Storage ──────────────────────────────────────────────────────────────

/// File-system backed storage (one file per segment in a directory)
pub struct FileStorage {
    base_path: PathBuf,
}

impl FileStorage {
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

// ── Memory Storage ────────────────────────────────────────────────────────────

/// In-memory storage (for testing and writing segments)
pub struct MemStorage {
    map: HashMap<String, Vec<u8>>,
}

impl MemStorage {
    pub fn new() -> Self {
        MemStorage {
            map: HashMap::new(),
        }
    }

    pub fn into_map(self) -> HashMap<String, Vec<u8>> {
        self.map
    }
}

impl Default for MemStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl StorageSegment for MemStorage {
    fn read(&self, segment_id: &str) -> ZResult<Vec<u8>> {
        self.map
            .get(segment_id)
            .cloned()
            .ok_or_else(|| Status::not_found(format!("segment '{}' not found", segment_id)))
    }

    fn write(&mut self, segment_id: &str, data: &[u8]) -> ZResult<()> {
        self.map.insert(segment_id.to_string(), data.to_vec());
        Ok(())
    }

    fn append(&mut self, segment_id: &str, data: &[u8]) -> ZResult<()> {
        self.map
            .entry(segment_id.to_string())
            .or_default()
            .extend_from_slice(data);
        Ok(())
    }

    fn exists(&self, segment_id: &str) -> bool {
        self.map.contains_key(segment_id)
    }

    fn size(&self, segment_id: &str) -> ZResult<u64> {
        self.map
            .get(segment_id)
            .map(|v| v.len() as u64)
            .ok_or_else(|| Status::not_found(format!("segment '{}' not found", segment_id)))
    }

    fn delete(&mut self, segment_id: &str) -> ZResult<()> {
        self.map.remove(segment_id);
        Ok(())
    }
}

// ── Mmap Storage (read-only) ──────────────────────────────────────────────────

/// Memory-mapped file storage (read-only after open)
pub struct MmapStorage {
    base_path: PathBuf,
}

impl MmapStorage {
    pub fn open(base_path: impl AsRef<Path>) -> ZResult<Self> {
        Ok(MmapStorage {
            base_path: base_path.as_ref().to_path_buf(),
        })
    }
}

impl StorageSegment for MmapStorage {
    fn read(&self, segment_id: &str) -> ZResult<Vec<u8>> {
        // Returns a copy; `get_mmap` gives zero-copy access.
        let path = self.base_path.join(segment_id);
        let mut file = fs::File::open(&path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                Status::not_found(format!("segment '{}' not found", segment_id))
            } else {
                Status::io_error(e.to_string())
            }
        })?;
        let mut data = Vec::new();
        std::io::Read::read_to_end(&mut file, &mut data)
            .map_err(|e| Status::io_error(e.to_string()))?;
        Ok(data)
    }

    fn write(&mut self, _segment_id: &str, _data: &[u8]) -> ZResult<()> {
        Err(Status::unimplemented("MmapStorage is read-only"))
    }

    fn append(&mut self, _segment_id: &str, _data: &[u8]) -> ZResult<()> {
        Err(Status::unimplemented("MmapStorage is read-only"))
    }

    fn exists(&self, segment_id: &str) -> bool {
        self.base_path.join(segment_id).exists()
    }

    fn size(&self, segment_id: &str) -> ZResult<u64> {
        let path = self.base_path.join(segment_id);
        let meta = fs::metadata(&path).map_err(|e| Status::io_error(e.to_string()))?;
        Ok(meta.len())
    }

    fn delete(&mut self, _segment_id: &str) -> ZResult<()> {
        Err(Status::unimplemented("MmapStorage is read-only"))
    }
}

/// Get a zero-copy mmap slice for a segment file
pub fn get_mmap(path: impl AsRef<Path>) -> ZResult<Mmap> {
    let file = fs::File::open(path.as_ref()).map_err(|e| Status::io_error(e.to_string()))?;
    // SAFETY: segment files are written once and not modified while mapped.
    unsafe { Mmap::map(&file) }.map_err(|e| Status::io_error(e.to_string()))
}

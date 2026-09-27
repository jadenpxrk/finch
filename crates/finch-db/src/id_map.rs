//! Primary key → doc_id mapping backed by fjall (LSM-tree)

use byteorder::{LittleEndian, ReadBytesExt};
use finch_types::{Status, ZResult};
use fjall::{Database, KeyspaceCreateOptions, PersistMode};
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

static RO_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Temp directory that is deleted on drop.
struct CleanupDir(PathBuf);
impl Drop for CleanupDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> ZResult<()> {
    fs::create_dir_all(dst).map_err(|e| Status::io_error(e.to_string()))?;
    for entry in fs::read_dir(src).map_err(|e| Status::io_error(e.to_string()))? {
        let entry = entry.map_err(|e| Status::io_error(e.to_string()))?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if src_path.is_dir() {
            copy_dir_recursive(&src_path, &dst_path)?;
        } else {
            fs::copy(&src_path, &dst_path).map_err(|e| Status::io_error(e.to_string()))?;
        }
    }
    Ok(())
}

struct OpenIdMap {
    db: Database,
    items: fjall::Keyspace,
    _ro_cleanup: Option<CleanupDir>,
}

enum IdMapState {
    Open(OpenIdMap),
    LazyReadOnly {
        opened: OnceLock<Result<OpenIdMap, String>>,
    },
}

pub struct IdMap {
    state: IdMapState,
    path: PathBuf,
}

impl IdMap {
    pub fn open(path: &Path) -> ZResult<Self> {
        let db = Database::builder(path)
            .open()
            .map_err(|e| Status::io_error(e.to_string()))?;
        let items = db
            .keyspace("id_map", KeyspaceCreateOptions::default)
            .map_err(|e| Status::io_error(e.to_string()))?;
        Ok(IdMap {
            state: IdMapState::Open(OpenIdMap {
                db,
                items,
                _ro_cleanup: None,
            }),
            path: path.to_path_buf(),
        })
    }

    pub fn open_read_only(path: &Path) -> ZResult<Self> {
        Ok(IdMap {
            state: IdMapState::LazyReadOnly {
                opened: OnceLock::new(),
            },
            path: path.to_path_buf(),
        })
    }

    fn open_read_only_copy(path: &Path) -> ZResult<OpenIdMap> {
        // Copy the fjall database directory to a unique temp path so
        // multiple read-only instances can coexist.
        let seq = RO_COUNTER.fetch_add(1, Ordering::Relaxed);
        let ro_name = format!("id_map_ro_{}_{}", std::process::id(), seq);
        let dst = path.parent().unwrap_or(path).join(&ro_name);
        copy_dir_recursive(path, &dst)?;
        let db = Database::builder(&dst).open().map_err(|e| {
            let _ = fs::remove_dir_all(&dst);
            Status::io_error(e.to_string())
        })?;
        let items = db
            .keyspace("id_map", KeyspaceCreateOptions::default)
            .map_err(|e| Status::io_error(e.to_string()))?;
        Ok(OpenIdMap {
            db,
            items,
            _ro_cleanup: Some(CleanupDir(dst)),
        })
    }

    fn open_state(&self) -> ZResult<&OpenIdMap> {
        match &self.state {
            IdMapState::Open(open) => Ok(open),
            IdMapState::LazyReadOnly { opened } => {
                let loaded = opened.get_or_init(|| {
                    Self::open_read_only_copy(&self.path).map_err(|e| e.to_string())
                });
                match loaded {
                    Ok(open) => Ok(open),
                    Err(e) => Err(Status::io_error(e.clone())),
                }
            }
        }
    }

    fn items(&self) -> ZResult<&fjall::Keyspace> {
        Ok(&self.open_state()?.items)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn insert(&self, pk: &str, doc_id: u64) -> ZResult<()> {
        self.items()?
            .insert(pk.as_bytes(), doc_id.to_le_bytes())
            .map_err(|e| Status::io_error(e.to_string()))
    }

    pub fn get(&self, pk: &str) -> ZResult<Option<u64>> {
        let Some(bytes) = self
            .items()?
            .get(pk.as_bytes())
            .map_err(|e| Status::io_error(e.to_string()))?
        else {
            return Ok(None);
        };
        if bytes.len() < 8 {
            return Err(Status::internal("corrupted id map entry"));
        }
        let mut cur = std::io::Cursor::new(bytes.as_ref());
        Ok(Some(
            cur.read_u64::<LittleEndian>()
                .map_err(|e| Status::io_error(e.to_string()))?,
        ))
    }

    pub fn delete(&self, pk: &str) -> ZResult<()> {
        self.items()?
            .remove(pk.as_bytes())
            .map_err(|e| Status::io_error(e.to_string()))
    }

    pub fn multi_get(&self, pks: &[&str]) -> ZResult<Vec<Option<u64>>> {
        pks.iter().map(|pk| self.get(pk)).collect()
    }

    /// Flush all writes to disk so they survive reopen.
    pub fn sync(&self) -> ZResult<()> {
        self.open_state()?
            .db
            .persist(PersistMode::SyncAll)
            .map_err(|e| Status::io_error(e.to_string()))
    }

    /// Create a snapshot by copying the fjall database directory to `dest_path`.
    pub fn create_snapshot(&self, dest_path: &Path) -> ZResult<()> {
        self.sync()?;
        copy_dir_recursive(&self.path, dest_path)?;
        // Best-effort durability
        let _ = fs::File::open(dest_path).and_then(|d| d.sync_all());
        if let Some(parent) = dest_path.parent() {
            let _ = fs::File::open(parent).and_then(|d| d.sync_all());
        }
        Ok(())
    }

    /// Iterate all (pk, doc_id) pairs
    pub fn iter_all(&self) -> Vec<(String, u64)> {
        let mut results = Vec::new();
        let Ok(items) = self.items() else {
            return results;
        };
        for guard in items.iter() {
            let Ok((k, v)) = guard.into_inner() else {
                continue;
            };
            let pk = match String::from_utf8(k.to_vec()) {
                Ok(s) => s,
                Err(_) => continue,
            };
            if v.len() < 8 {
                continue;
            }
            let mut cur = std::io::Cursor::new(v.as_ref());
            let doc_id = match cur.read_u64::<LittleEndian>() {
                Ok(id) => id,
                Err(_) => continue,
            };
            results.push((pk, doc_id));
        }
        results
    }
}

//! Primary key → doc_id mapping.
//!
//! A read-write collection keeps the map in a fjall database under `<dir>/live`. Each flush
//! writes a checkpoint: a sorted file at `<dir>/checkpoint` of a new id-map directory, which
//! the manifest names by suffix. Opening rebuilds the live database from the checkpoint, and
//! WAL replay adds everything after it. Read-only opens read the checkpoint file directly.

use finch_types::{Status, ZResult};
use fjall::{Database, KeyspaceCreateOptions, Readable};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::sorted_file::{sync_dir, write_sorted_file, SortedFile};

const CHECKPOINT_FILE: &str = "checkpoint";
const LIVE_DIR: &str = "live";

fn io_err(e: impl ToString) -> Status {
    Status::io_error(e.to_string())
}

fn decode_doc_id(bytes: &[u8]) -> ZResult<u64> {
    bytes
        .get(..8)
        .and_then(|b| b.try_into().ok())
        .map(u64::from_le_bytes)
        .ok_or_else(|| Status::internal("corrupted id map entry"))
}

fn open_items(dir: &Path) -> ZResult<(Database, fjall::Keyspace)> {
    let db = Database::builder(dir).open().map_err(io_err)?;
    let items = db
        .keyspace("id_map", KeyspaceCreateOptions::default)
        .map_err(io_err)?;
    Ok((db, items))
}

fn write_checkpoint(db: &Database, items: &fjall::Keyspace, path: &Path) -> ZResult<()> {
    let snapshot = db.snapshot();
    let entries = snapshot
        .iter(items)
        .map(|guard| guard.into_inner().map_err(io_err));
    write_sorted_file(path, entries)
}

// Removes everything in `dir` except the checkpoint file, creating `dir` if needed.
fn clear_all_but_checkpoint(dir: &Path) -> ZResult<()> {
    fs::create_dir_all(dir).map_err(io_err)?;
    for entry in fs::read_dir(dir).map_err(io_err)? {
        let entry = entry.map_err(io_err)?;
        if entry.file_name() == CHECKPOINT_FILE {
            continue;
        }
        let path = entry.path();
        let removed = if entry.file_type().map_err(io_err)?.is_dir() {
            fs::remove_dir_all(&path)
        } else {
            fs::remove_file(&path)
        };
        removed.map_err(|e| Status::io_error(format!("could not clear {:?}: {}", path, e)))?;
    }
    Ok(())
}

enum IdMapState {
    Live {
        db: Database,
        items: fjall::Keyspace,
    },
    ReadOnly {
        checkpoint: OnceLock<Result<Option<SortedFile>, String>>,
    },
}

pub struct IdMap {
    state: IdMapState,
    path: PathBuf,
}

impl IdMap {
    /// Opens the id map in `dir` for writing.
    pub fn open(dir: &Path) -> ZResult<Self> {
        let checkpoint = dir.join(CHECKPOINT_FILE);
        // The live database holds nothing that the checkpoint and WAL replay do not rebuild.
        clear_all_but_checkpoint(dir)?;
        let (db, items) = open_items(&dir.join(LIVE_DIR))?;
        if checkpoint.exists() {
            let file = SortedFile::open(&checkpoint)?;
            let mut ingestion = items.start_ingestion().map_err(io_err)?;
            for (pk, doc_id) in file.iter_from(&[]) {
                ingestion.write(pk, doc_id).map_err(io_err)?;
            }
            ingestion.finish().map_err(io_err)?;
        }
        Ok(IdMap {
            state: IdMapState::Live { db, items },
            path: dir.to_path_buf(),
        })
    }

    /// Opens the checkpoint in `dir` without writing anything; it is read on first use.
    pub fn open_read_only(dir: &Path) -> ZResult<Self> {
        Ok(IdMap {
            state: IdMapState::ReadOnly {
                checkpoint: OnceLock::new(),
            },
            path: dir.to_path_buf(),
        })
    }

    /// Whether `dir` holds a checkpoint.
    pub(crate) fn has_checkpoint(dir: &Path) -> bool {
        dir.join(CHECKPOINT_FILE).exists()
    }

    // The checkpoint of a read-only map, loaded on first use; `None` when no flush wrote one.
    fn checkpoint(&self) -> ZResult<Option<&SortedFile>> {
        let IdMapState::ReadOnly { checkpoint } = &self.state else {
            return Ok(None);
        };
        let loaded = checkpoint.get_or_init(|| {
            let path = self.path.join(CHECKPOINT_FILE);
            if !path.exists() {
                return Ok(None);
            }
            SortedFile::open(&path).map(Some).map_err(|e| e.message)
        });
        match loaded {
            Ok(file) => Ok(file.as_ref()),
            Err(e) => Err(Status::io_error(e.clone())),
        }
    }

    fn live_items(&self) -> ZResult<&fjall::Keyspace> {
        match &self.state {
            IdMapState::Live { items, .. } => Ok(items),
            IdMapState::ReadOnly { .. } => {
                Err(Status::permission_denied("id map is open read-only"))
            }
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn insert(&self, pk: &str, doc_id: u64) -> ZResult<()> {
        self.live_items()?
            .insert(pk.as_bytes(), doc_id.to_le_bytes())
            .map_err(io_err)
    }

    pub fn get(&self, pk: &str) -> ZResult<Option<u64>> {
        match &self.state {
            IdMapState::Live { items, .. } => items
                .get(pk.as_bytes())
                .map_err(io_err)?
                .map(|bytes| decode_doc_id(&bytes))
                .transpose(),
            IdMapState::ReadOnly { .. } => self
                .checkpoint()?
                .and_then(|file| file.get(pk.as_bytes()))
                .map(decode_doc_id)
                .transpose(),
        }
    }

    pub fn delete(&self, pk: &str) -> ZResult<()> {
        self.live_items()?.remove(pk.as_bytes()).map_err(io_err)
    }

    pub fn multi_get(&self, pks: &[&str]) -> ZResult<Vec<Option<u64>>> {
        pks.iter().map(|pk| self.get(pk)).collect()
    }

    /// Writes a consistent checkpoint of the live map into `dest_dir`, replacing whatever an
    /// earlier failed flush left there.
    pub fn create_snapshot(&self, dest_dir: &Path) -> ZResult<()> {
        let IdMapState::Live { db, items } = &self.state else {
            return Err(Status::permission_denied("id map is open read-only"));
        };
        match fs::remove_dir_all(dest_dir) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(io_err(e)),
            _ => {}
        }
        fs::create_dir_all(dest_dir).map_err(io_err)?;
        write_checkpoint(db, items, &dest_dir.join(CHECKPOINT_FILE))?;
        sync_dir(dest_dir.parent().unwrap_or(Path::new(".")))
    }
}

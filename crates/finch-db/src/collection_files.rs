use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use finch_storage::{ForwardStoreOpenOptions, ParquetReadMode};
use finch_types::{CollectionOptions, FileFormat, Status, StorageType, ZResult};
use fs4::FileExt;

pub(crate) fn wal_path_for_writing_segment(
    collection_path: &Path,
    writing_segment_id: u32,
) -> PathBuf {
    collection_path.join(format!("wal_{}.log", writing_segment_id))
}

// Generation 0 keeps the original name; field names cannot contain '.', so names never collide.
pub(crate) fn vector_index_path(seg_dir: &Path, field: &str, generation: u32) -> PathBuf {
    if generation == 0 {
        seg_dir.join(format!("idx_{field}"))
    } else {
        seg_dir.join(format!("idx_{field}.{generation}"))
    }
}

pub(crate) fn legacy_wal_path(collection_path: &Path) -> PathBuf {
    collection_path.join("wal.log")
}

pub(crate) fn effective_forward_file_format(options: &CollectionOptions) -> FileFormat {
    options.forward_file_format.unwrap_or(FileFormat::ArrowIpc)
}

pub(crate) fn effective_index_enable_mmap(options: &CollectionOptions) -> ZResult<bool> {
    match options.index_storage {
        None | Some(StorageType::None) => Ok(options.enable_mmap),
        Some(StorageType::Mmap) => Ok(true),
        Some(StorageType::Memory) => Ok(false),
        Some(StorageType::BufferPool) => Err(Status::invalid_argument(
            "index_storage=buffer_pool is not supported for vector indexes yet",
        )),
    }
}

pub(crate) fn effective_forward_store_open_options(
    options: &CollectionOptions,
) -> ForwardStoreOpenOptions {
    let mut enable_mmap = options.enable_mmap;
    let mut parquet_read_mode = ParquetReadMode::Buffered;

    match options.forward_storage {
        None | Some(StorageType::None) => {}
        Some(StorageType::Mmap) => {
            enable_mmap = true;
        }
        Some(StorageType::Memory) => {
            enable_mmap = false;
            parquet_read_mode = ParquetReadMode::Eager;
        }
        Some(StorageType::BufferPool) => {
            enable_mmap = false;
            parquet_read_mode = ParquetReadMode::Buffered;
        }
    }

    ForwardStoreOpenOptions {
        enable_mmap,
        parquet_read_mode,
        lazy_ipc: options.read_only,
    }
}

pub(crate) fn is_parquet_file(path: &Path) -> bool {
    let Ok(mut f) = fs::File::open(path) else {
        return false;
    };
    let mut hdr = [0u8; 4];
    let Ok(n) = f.read(&mut hdr) else {
        return false;
    };
    n == 4 && &hdr == b"PAR1"
}

pub(crate) fn forward_path_for_segment(seg_path: &Path, preferred: FileFormat) -> PathBuf {
    let parquet = seg_path.join("forward.parquet");
    let arrow = seg_path.join("forward.arrow");

    match preferred {
        FileFormat::Parquet => {
            if parquet.exists() {
                parquet
            } else if arrow.exists() {
                arrow
            } else {
                parquet
            }
        }
        _ => {
            if arrow.exists() {
                arrow
            } else if parquet.exists() {
                parquet
            } else {
                arrow
            }
        }
    }
}

/// Versioned id_map directory path. Suffix 0 uses the legacy "id_map" name
/// for backward compatibility; subsequent checkpoints use "id_map_{suffix}".
pub(crate) fn id_map_path_for_suffix(base: &Path, suffix: u32) -> PathBuf {
    if suffix == 0 {
        base.join("id_map")
    } else {
        base.join(format!("id_map_{}", suffix))
    }
}

pub(crate) fn cleanup_old_id_map_checkpoints(base: &Path, active_path: &Path, keep_suffix: u32) {
    if let Ok(entries) = fs::read_dir(base) {
        for entry in entries.flatten() {
            let Ok(ft) = entry.file_type() else {
                continue;
            };
            if !ft.is_dir() {
                continue;
            }
            let name = entry.file_name();
            let name = name.to_string_lossy();
            let Some(suf) = name.strip_prefix("id_map_") else {
                continue;
            };
            let Ok(suf) = suf.parse::<u32>() else {
                continue;
            };
            if suf >= keep_suffix {
                continue;
            }
            let p = entry.path();
            if p == active_path {
                continue;
            }
            let _ = fs::remove_dir_all(&p);
        }
    }

    let _ = fs::File::open(base).and_then(|d| d.sync_all());
}

/// Removes the per-process copies that releases before frozen invert files and id-map
/// checkpoint files made for read-only opens, which a killed process leaves behind.
pub(crate) fn remove_stale_read_only_copies(base: &Path) -> ZResult<()> {
    let io = |p: &Path, e: std::io::Error| Status::io_error(format!("{:?}: {}", p, e));
    let mut dirs = vec![base.to_path_buf()];
    for entry in fs::read_dir(base).map_err(|e| io(base, e))? {
        let path = entry.map_err(|e| io(base, e))?.path();
        let is_seg = path
            .file_name()
            .is_some_and(|n| n.to_string_lossy().starts_with("seg_"));
        if is_seg && path.is_dir() {
            dirs.push(path);
        }
    }
    for dir in dirs {
        for entry in fs::read_dir(&dir).map_err(|e| io(&dir, e))? {
            let path = entry.map_err(|e| io(&dir, e))?.path();
            let name = path.file_name().map(|n| n.to_string_lossy().into_owned());
            let is_copy =
                name.is_some_and(|n| n.starts_with("id_map_ro_") || n.starts_with("invert_ro_"));
            if is_copy && path.is_dir() {
                fs::remove_dir_all(&path).map_err(|e| io(&path, e))?;
            }
        }
    }
    Ok(())
}

/// Acquire a process-level lock on the collection directory.
/// Read-only opens use shared lock; read-write opens use exclusive lock.
pub(crate) fn acquire_file_lock(path: &Path, read_only: bool) -> ZResult<fs::File> {
    let lock_path = path.join("LOCK");
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .map_err(|e| Status::io_error(format!("cannot open lock file: {}", e)))?;
    if read_only {
        file.try_lock_shared().map_err(|_| {
            Status::io_error("collection is locked for read-write access by another process")
        })?;
    } else {
        file.try_lock_exclusive()
            .map_err(|_| Status::io_error("collection is already locked by another process"))?;
    }
    Ok(file)
}

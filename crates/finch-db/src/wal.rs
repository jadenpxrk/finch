//! Write-ahead log (WAL) for crash recovery.

use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use finch_types::{Doc, Status, StatusCode, ZResult};

mod record_file;
use record_file::{WalFileOptions, WalRecordFileReader, WalRecordFileWriter, WAL_HEADER_SIZE};

#[derive(Serialize, Deserialize, Debug)]
pub(crate) enum WalOp {
    Insert,
    Upsert,
    Update,
    Delete,
}

#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct WalEntry {
    pub(crate) op: WalOp,
    pub(crate) doc_id: u64,
    /// When replacing an existing document (update/upsert), this is the prior
    /// doc_id that must be tombstoned in the delete bitmap. This differs from
    /// `WalOp::Delete`, which also removes the PK mapping.
    #[serde(default)]
    pub(crate) prev_doc_id: Option<u64>,
    pub(crate) pk: String,
    pub(crate) doc: Option<Doc>,
}

pub(crate) struct Wal {
    writer: WalWriter,
    path: PathBuf,
    flush_every_docs: u32,
    fsync_every_docs: u32,
    docs_since_flush: u32,
    docs_since_fsync: u32, // legacy jsonl only
    // Set by a failed write or sync; later appends fail until the WAL is reopened or switched.
    poisoned: bool,
}

enum WalWriter {
    Binary(WalRecordFileWriter),
    LegacyJsonl(BufWriter<fs::File>),
}

enum WalFormat {
    Empty,
    Binary,
    LegacyJsonl,
}

// Legacy JSONL WALs start with `{`; any other non-empty file is binary and must hold a whole header.
fn wal_format(path: &Path) -> ZResult<WalFormat> {
    let mut file = fs::File::open(path).map_err(|e| Status::io_error(e.to_string()))?;
    let len = file
        .metadata()
        .map_err(|e| Status::io_error(e.to_string()))?
        .len();
    if len == 0 {
        return Ok(WalFormat::Empty);
    }
    let mut first = [0u8; 1];
    file.read_exact(&mut first)
        .map_err(|e| Status::io_error(e.to_string()))?;
    if first[0] == b'{' {
        return Ok(WalFormat::LegacyJsonl);
    }
    if (len as usize) < WAL_HEADER_SIZE {
        return Err(Status::io_error(format!(
            "WAL {} is {} bytes, shorter than its {}-byte header; inspect or move it aside before reopening",
            path.display(),
            len,
            WAL_HEADER_SIZE
        )));
    }
    Ok(WalFormat::Binary)
}

impl Wal {
    pub(crate) fn open(
        path: PathBuf,
        flush_every_docs: u32,
        fsync_every_docs: u32,
    ) -> ZResult<Self> {
        let writer = if path.exists() {
            Self::open_existing_writer(&path, fsync_every_docs)?
        } else {
            Self::open_binary_writer(&path, true, fsync_every_docs)?
        };
        Ok(Wal {
            writer,
            path,
            flush_every_docs,
            fsync_every_docs,
            docs_since_flush: 0,
            docs_since_fsync: 0,
            poisoned: false,
        })
    }

    fn open_binary_writer(
        path: &Path,
        create_new: bool,
        fsync_every_docs: u32,
    ) -> ZResult<WalWriter> {
        Ok(WalWriter::Binary(WalRecordFileWriter::open(
            path.to_path_buf(),
            WalFileOptions {
                create_new,
                max_docs_wal_flush: fsync_every_docs,
            },
        )?))
    }

    fn open_existing_writer(path: &Path, fsync_every_docs: u32) -> ZResult<WalWriter> {
        match wal_format(path)? {
            WalFormat::Empty => {
                // A crash between create and header write leaves an empty file; give it the header in place.
                let mut file = fs::OpenOptions::new()
                    .write(true)
                    .open(path)
                    .map_err(|e| Status::io_error(e.to_string()))?;
                file.write_all(&[0u8; WAL_HEADER_SIZE])
                    .map_err(|e| Status::io_error(e.to_string()))?;
                file.sync_all()
                    .map_err(|e| Status::io_error(e.to_string()))?;
                Self::open_binary_writer(path, false, fsync_every_docs)
            }
            WalFormat::Binary => Self::open_binary_writer(path, false, fsync_every_docs),
            WalFormat::LegacyJsonl => {
                let file = fs::OpenOptions::new()
                    .append(true)
                    .open(path)
                    .map_err(|e| Status::io_error(e.to_string()))?;
                Ok(WalWriter::LegacyJsonl(BufWriter::new(file)))
            }
        }
    }

    pub(crate) fn append(&mut self, entry: &WalEntry) -> ZResult<()> {
        if self.poisoned {
            return Err(Status::io_error(
                "WAL write or sync failed earlier; reopen the collection before writing",
            ));
        }
        let json = serde_json::to_vec(entry).map_err(|e| Status::io_error(e.to_string()))?;
        let result = self.write_record(&json);
        // After a partial write or failed sync the durable tail of the log is unknown.
        self.poisoned = matches!(&result, Err(e) if e.code == StatusCode::IoError);
        result
    }

    fn write_record(&mut self, json: &[u8]) -> ZResult<()> {
        match &mut self.writer {
            WalWriter::Binary(w) => {
                w.append(json)?;
            }
            WalWriter::LegacyJsonl(w) => {
                w.write_all(json)
                    .map_err(|e| Status::io_error(e.to_string()))?;
                w.write_all(b"\n")
                    .map_err(|e| Status::io_error(e.to_string()))?;
                self.docs_since_fsync = self.docs_since_fsync.saturating_add(1);
            }
        }
        self.docs_since_flush = self.docs_since_flush.saturating_add(1);

        if self.flush_every_docs > 0 && self.docs_since_flush >= self.flush_every_docs {
            match &mut self.writer {
                WalWriter::Binary(w) => w.flush_buffer()?,
                WalWriter::LegacyJsonl(w) => {
                    w.flush().map_err(|e| Status::io_error(e.to_string()))?
                }
            }
            self.docs_since_flush = 0;
        }

        if self.fsync_every_docs > 0 {
            if let WalWriter::LegacyJsonl(w) = &mut self.writer {
                if self.docs_since_fsync >= self.fsync_every_docs {
                    w.flush().map_err(|e| Status::io_error(e.to_string()))?;
                    w.get_ref()
                        .sync_all()
                        .map_err(|e| Status::io_error(e.to_string()))?;
                    self.docs_since_fsync = 0;
                }
            }
        }

        Ok(())
    }

    /// Replay WAL entries from disk.
    ///
    /// A binary WAL replays up to a clean end or a torn final record, and fails on a
    /// corrupt record, because every record after it would be silently dropped.
    /// A legacy JSONL WAL stops at its first malformed line.
    pub(crate) fn replay(path: &Path) -> ZResult<Vec<WalEntry>> {
        if !path.exists() {
            return Ok(Vec::new());
        }
        match wal_format(path)? {
            WalFormat::Empty => Ok(Vec::new()),
            WalFormat::Binary => Self::replay_binary(path),
            WalFormat::LegacyJsonl => Self::replay_jsonl(path),
        }
    }

    fn replay_binary(path: &Path) -> ZResult<Vec<WalEntry>> {
        let mut reader = WalRecordFileReader::open(path.to_path_buf())?;
        // An unreadable header fails the open; returning no entries would drop the unreplayed writes.
        reader.prepare_for_read()?;
        let mut entries = Vec::new();
        while let Some(rec) = reader.next()? {
            let entry = serde_json::from_slice::<WalEntry>(&rec).map_err(|e| {
                Status::io_error(format!(
                    "WAL {} has an unreadable entry ending at offset {}: {}",
                    path.display(),
                    reader.valid_end(),
                    e
                ))
            })?;
            entries.push(entry);
        }
        Ok(entries)
    }

    fn replay_jsonl(path: &Path) -> ZResult<Vec<WalEntry>> {
        let file = fs::File::open(path).map_err(|e| Status::io_error(e.to_string()))?;
        let reader = BufReader::new(file);
        let mut entries = Vec::new();
        for line in reader.lines() {
            let line = line.map_err(|e| Status::io_error(e.to_string()))?;
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<WalEntry>(&line) {
                Ok(entry) => entries.push(entry),
                Err(e) => {
                    tracing::warn!("stopping WAL replay at malformed entry: {}", e);
                    break;
                }
            }
        }
        Ok(entries)
    }

    // Runs before the manifest names `new_path`, so a failure leaves this WAL in use and nothing lost.
    pub(crate) fn create_next(&self, new_path: PathBuf) -> ZResult<NextWal> {
        // Only a flush that failed before its manifest commit leaves a file here.
        if new_path.exists() {
            let _ = fs::remove_file(&new_path);
        }
        let writer = WalRecordFileWriter::open(
            new_path.clone(),
            WalFileOptions {
                create_new: true,
                max_docs_wal_flush: self.fsync_every_docs,
            },
        )?;
        Ok(NextWal {
            path: new_path,
            writer,
        })
    }

    // Runs after the manifest names `next`; the caller deletes the returned old file.
    pub(crate) fn switch_to(&mut self, next: NextWal) -> PathBuf {
        let old = std::mem::replace(&mut self.path, next.path);
        self.writer = WalWriter::Binary(next.writer);
        self.docs_since_flush = 0;
        self.docs_since_fsync = 0;
        self.poisoned = false;
        old
    }
}

/// A synced, empty WAL file for the next writing segment, not yet in use.
pub(crate) struct NextWal {
    path: PathBuf,
    writer: WalRecordFileWriter,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    // A WAL file whose next `sync` fails once `fail_next_sync` is set.
    struct FlakySyncFile {
        file: fs::File,
        fail_next_sync: Arc<AtomicBool>,
    }

    impl Write for FlakySyncFile {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.file.write(buf)
        }
        fn flush(&mut self) -> io::Result<()> {
            self.file.flush()
        }
    }

    impl record_file::WalSink for FlakySyncFile {
        fn sync(&self) -> io::Result<()> {
            if self.fail_next_sync.swap(false, Ordering::SeqCst) {
                return Err(io::Error::other("injected sync failure"));
            }
            self.file.sync_all()
        }
    }

    fn insert_entry(doc_id: u64) -> WalEntry {
        let pk = format!("p{doc_id}");
        WalEntry {
            op: WalOp::Insert,
            doc_id,
            prev_doc_id: None,
            doc: Some(Doc::new(&pk).set("x", doc_id as i32)),
            pk,
        }
    }

    fn temp_wal_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("finch_wal_test_{}_{}.log", name, nanos));
        p
    }

    #[test]
    fn test_replay_keeps_entries_before_first_malformed_line() {
        let path = temp_wal_path("stop_on_malformed");

        let e1 = WalEntry {
            op: WalOp::Insert,
            doc_id: 1,
            prev_doc_id: None,
            pk: "p1".to_string(),
            doc: Some(Doc::new("p1").set("x", 1i32)),
        };
        let e2 = WalEntry {
            op: WalOp::Insert,
            doc_id: 2,
            prev_doc_id: None,
            pk: "p2".to_string(),
            doc: Some(Doc::new("p2").set("x", 2i32)),
        };

        let mut f = fs::File::create(&path).unwrap();
        writeln!(f, "{}", serde_json::to_string(&e1).unwrap()).unwrap();
        writeln!(f, "{{this is not valid json").unwrap();
        writeln!(f, "{}", serde_json::to_string(&e2).unwrap()).unwrap();
        drop(f);

        let got = Wal::replay(&path).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].doc_id, 1);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_replay_keeps_entries_before_truncated_last_line() {
        let path = temp_wal_path("stop_on_trunc");

        let e1 = WalEntry {
            op: WalOp::Insert,
            doc_id: 1,
            prev_doc_id: None,
            pk: "p1".to_string(),
            doc: Some(Doc::new("p1").set("x", 1i32)),
        };

        let mut f = fs::File::create(&path).unwrap();
        writeln!(f, "{}", serde_json::to_string(&e1).unwrap()).unwrap();
        write!(f, "{{\"op\":").unwrap(); // truncated JSON
        drop(f);

        let got = Wal::replay(&path).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].doc_id, 1);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_binary_replay_keeps_entries_before_truncated_last_record() {
        let path = temp_wal_path("binary_trunc");

        let mut wal = Wal::open(path.clone(), 1, 0).unwrap();
        let e1 = WalEntry {
            op: WalOp::Insert,
            doc_id: 1,
            prev_doc_id: None,
            pk: "p1".to_string(),
            doc: Some(Doc::new("p1").set("x", 1i32)),
        };
        let e2 = WalEntry {
            op: WalOp::Insert,
            doc_id: 2,
            prev_doc_id: None,
            pk: "p2".to_string(),
            doc: Some(Doc::new("p2").set("x", 2i32)),
        };
        wal.append(&e1).unwrap();
        wal.append(&e2).unwrap();
        drop(wal);

        let m = fs::metadata(&path).unwrap();
        fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(m.len().saturating_sub(1))
            .unwrap();

        let got = Wal::replay(&path).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].doc_id, 1);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn binary_replay_fails_on_an_unreadable_entry() {
        let path = temp_wal_path("binary_unreadable");
        let mut wal = Wal::open(path.clone(), 1, 1).unwrap();
        wal.append(&insert_entry(1)).unwrap();
        wal.write_record(b"{not a wal entry").unwrap();
        wal.append(&insert_entry(2)).unwrap();
        drop(wal);

        let err = Wal::replay(&path).unwrap_err();
        assert_eq!(err.code, StatusCode::IoError);
        assert!(err.message.contains("unreadable entry"), "{}", err.message);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn failed_sync_poisons_writer_until_reopen() {
        let path = temp_wal_path("failed_sync");
        drop(Wal::open(path.clone(), 1, 1).unwrap());

        let fail_next_sync = Arc::new(AtomicBool::new(false));
        let sink = FlakySyncFile {
            file: fs::OpenOptions::new().append(true).open(&path).unwrap(),
            fail_next_sync: fail_next_sync.clone(),
        };
        let mut wal = Wal {
            writer: WalWriter::Binary(WalRecordFileWriter::from_sink(Box::new(sink), 1)),
            path: path.clone(),
            flush_every_docs: 1,
            fsync_every_docs: 1,
            docs_since_flush: 0,
            docs_since_fsync: 0,
            poisoned: false,
        };
        let fsync_counter = |wal: &Wal| match &wal.writer {
            WalWriter::Binary(w) => w.docs_since_fsync,
            WalWriter::LegacyJsonl(_) => unreachable!(),
        };

        wal.append(&insert_entry(1)).unwrap();
        assert_eq!(fsync_counter(&wal), 0);

        fail_next_sync.store(true, Ordering::SeqCst);
        let err = wal.append(&insert_entry(2)).unwrap_err();
        assert_eq!(err.code, StatusCode::IoError);
        assert!(err.message.contains("injected sync failure"));
        assert_eq!(
            fsync_counter(&wal),
            1,
            "a failed sync must not reset the counter"
        );

        let err = wal.append(&insert_entry(3)).unwrap_err();
        assert_eq!(err.code, StatusCode::IoError);
        assert_eq!(fsync_counter(&wal), 1);
        drop(wal);

        let mut wal = Wal::open(path.clone(), 1, 1).unwrap();
        wal.append(&insert_entry(4)).unwrap();
        drop(wal);

        // Record 2 reached the file before its sync failed; record 3 was refused.
        let ids: Vec<u64> = Wal::replay(&path)
            .unwrap()
            .iter()
            .map(|e| e.doc_id)
            .collect();
        assert_eq!(ids, vec![1, 2, 4]);

        let _ = fs::remove_file(&path);
    }
}

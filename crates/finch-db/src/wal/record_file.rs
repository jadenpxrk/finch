//! WAL record file: fixed header + length/crc framed records.
//!
//! - 64-byte header (version + reserved)
//! - record format: [u32 length][u32 crc32c][payload bytes]
//! - `next()` ends at a clean end or a torn final record, and fails on corruption
//! - optional fsync every N appended records; a failed fsync is returned to the caller

use std::fs;
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use finch_types::{Status, ZResult};

pub(crate) const WAL_HEADER_SIZE: usize = 64;
pub(crate) const WAL_MAX_RECORD_SIZE: usize = 4 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct WalFileOptions {
    pub(crate) create_new: bool,
    /// fsync every N appended records (0 disables auto flush).
    pub(crate) max_docs_wal_flush: u32,
}

fn crc32c_table() -> &'static [u32; 256] {
    static TABLE: OnceLock<[u32; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [0u32; 256];
        // CRC32C (Castagnoli) reversed polynomial.
        let poly = 0x82F63B78u32;
        for (i, slot) in table.iter_mut().enumerate() {
            let mut crc = i as u32;
            for _ in 0..8 {
                crc = if (crc & 1) != 0 {
                    poly ^ (crc >> 1)
                } else {
                    crc >> 1
                };
            }
            *slot = crc;
        }
        table
    })
}

/// CRC32C hash with caller-provided initial crc (no final xor).
pub fn crc32c_hash(data: &[u8], mut crc: u32) -> u32 {
    let table = crc32c_table();
    for &b in data {
        let idx = ((crc as u8) ^ b) as usize;
        crc = table[idx] ^ (crc >> 8);
    }
    crc
}

/// Durable byte target for a WAL writer; a test double can fail `sync`.
pub(crate) trait WalSink: Write + Send {
    fn sync(&self) -> io::Result<()>;
}

impl WalSink for fs::File {
    fn sync(&self) -> io::Result<()> {
        self.sync_all()
    }
}

pub(crate) struct WalRecordFileWriter {
    writer: BufWriter<Box<dyn WalSink>>,
    max_docs_wal_flush: u32,
    pub(super) docs_since_fsync: u32,
}

impl WalRecordFileWriter {
    pub(crate) fn open(path: PathBuf, opts: WalFileOptions) -> ZResult<Self> {
        if opts.create_new {
            if path.exists() {
                return Err(Status::io_error(format!(
                    "WAL create_new failed: file already exists at {}",
                    path.display()
                )));
            }
            let mut file = fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .read(true)
                .open(&path)
                .map_err(|e| Status::io_error(e.to_string()))?;
            // 64-byte header: version=0, reserved=0
            file.write_all(&[0u8; WAL_HEADER_SIZE])
                .map_err(|e| Status::io_error(e.to_string()))?;
            file.flush().map_err(|e| Status::io_error(e.to_string()))?;
            file.sync_all()
                .map_err(|e| Status::io_error(e.to_string()))?;
            // Seek to end for appends.
            file.seek(SeekFrom::End(0))
                .map_err(|e| Status::io_error(e.to_string()))?;
            Ok(Self::from_sink(Box::new(file), opts.max_docs_wal_flush))
        } else {
            if !path.exists() {
                return Err(Status::io_error(format!(
                    "WAL open failed: file does not exist at {}",
                    path.display()
                )));
            }
            // Replay stops at the last whole record, so records appended after a torn tail would be lost.
            let append_at = Self::append_boundary(&path)?;
            let mut file = fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .map_err(|e| Status::io_error(e.to_string()))?;
            let len = file
                .metadata()
                .map_err(|e| Status::io_error(e.to_string()))?
                .len();
            if len != append_at {
                file.set_len(append_at)
                    .and_then(|()| file.sync_all())
                    .map_err(|e| Status::io_error(e.to_string()))?;
            }
            file.seek(SeekFrom::Start(append_at))
                .map_err(|e| Status::io_error(e.to_string()))?;
            Ok(Self::from_sink(Box::new(file), opts.max_docs_wal_flush))
        }
    }

    // End of the last whole record; fails if the log is corrupt before its tail.
    fn append_boundary(path: &Path) -> ZResult<u64> {
        let mut reader = WalRecordFileReader::open(path.to_path_buf())?;
        reader.prepare_for_read()?;
        while reader.next()?.is_some() {}
        Ok(reader.valid_end())
    }

    /// Appends to `sink`, which must already hold a WAL header.
    pub(crate) fn from_sink(sink: Box<dyn WalSink>, max_docs_wal_flush: u32) -> Self {
        WalRecordFileWriter {
            writer: BufWriter::new(sink),
            max_docs_wal_flush,
            docs_since_fsync: 0,
        }
    }

    pub(crate) fn append(&mut self, data: &[u8]) -> ZResult<()> {
        if data.is_empty() || data.len() > WAL_MAX_RECORD_SIZE {
            return Err(Status::invalid_argument(format!(
                "WAL record size out of range: {} bytes",
                data.len()
            )));
        }

        let len_u32: u32 = data
            .len()
            .try_into()
            .map_err(|_| Status::invalid_argument("WAL record too large"))?;
        let crc = crc32c_hash(data, 0);

        self.writer
            .write_all(&len_u32.to_le_bytes())
            .map_err(|e| Status::io_error(e.to_string()))?;
        self.writer
            .write_all(&crc.to_le_bytes())
            .map_err(|e| Status::io_error(e.to_string()))?;
        self.writer
            .write_all(data)
            .map_err(|e| Status::io_error(e.to_string()))?;

        if self.max_docs_wal_flush != 0 {
            self.docs_since_fsync = self.docs_since_fsync.saturating_add(1);
            if self.docs_since_fsync >= self.max_docs_wal_flush {
                self.flush_fsync()?;
                self.docs_since_fsync = 0;
            }
        }

        Ok(())
    }

    /// Flush buffered writes to the OS (no durability guarantee).
    pub(crate) fn flush_buffer(&mut self) -> ZResult<()> {
        self.writer
            .flush()
            .map_err(|e| Status::io_error(e.to_string()))
    }

    /// Flush buffered writes and fsync them.
    pub(crate) fn flush_fsync(&mut self) -> ZResult<()> {
        self.writer
            .flush()
            .map_err(|e| Status::io_error(e.to_string()))?;
        self.writer
            .get_ref()
            .sync()
            .map_err(|e| Status::io_error(e.to_string()))
    }

    #[cfg(test)]
    pub(crate) fn close(mut self) -> ZResult<()> {
        self.flush_fsync()
    }
}

// A crash tears only the last record, so a whole record after a truncated one means corruption.
fn whole_record_follows(tail: &[u8]) -> bool {
    (1..tail.len()).any(|start| {
        let Some((frame, rest)) = tail[start..].split_first_chunk::<8>() else {
            return false;
        };
        let length = u32::from_le_bytes([frame[0], frame[1], frame[2], frame[3]]) as usize;
        let crc = u32::from_le_bytes([frame[4], frame[5], frame[6], frame[7]]);
        length != 0 && length <= rest.len() && crc32c_hash(&rest[..length], 0) == crc
    })
}

pub(crate) struct WalRecordFileReader {
    path: PathBuf,
    reader: BufReader<fs::File>,
    // Offset just past the last whole record read; appends must start here.
    valid_end: u64,
}

impl WalRecordFileReader {
    pub(crate) fn open(path: PathBuf) -> ZResult<Self> {
        if !path.exists() {
            return Err(Status::io_error(format!(
                "WAL open failed: file does not exist at {}",
                path.display()
            )));
        }
        let file = fs::OpenOptions::new()
            .read(true)
            .open(&path)
            .map_err(|e| Status::io_error(e.to_string()))?;
        Ok(WalRecordFileReader {
            path,
            reader: BufReader::new(file),
            valid_end: WAL_HEADER_SIZE as u64,
        })
    }

    pub(crate) fn prepare_for_read(&mut self) -> ZResult<()> {
        self.reader
            .get_mut()
            .seek(SeekFrom::Start(0))
            .map_err(|e| Status::io_error(e.to_string()))?;
        let mut header = [0u8; WAL_HEADER_SIZE];
        self.reader
            .read_exact(&mut header)
            .map_err(|e| Status::io_error(e.to_string()))?;
        // Version is the first 8 bytes (little-endian).
        let mut v = [0u8; 8];
        v.copy_from_slice(&header[..8]);
        let version = u64::from_le_bytes(v);
        if version != 0 {
            return Err(Status::io_error(format!(
                "unsupported WAL version {} in {}",
                version,
                self.path.display()
            )));
        }
        self.valid_end = WAL_HEADER_SIZE as u64;
        Ok(())
    }

    /// The next record, or `None` at a clean end or a final record cut short by a crash.
    /// A bad length or checksum is corruption, not an end, and returns an error.
    pub(crate) fn next(&mut self) -> ZResult<Option<Vec<u8>>> {
        let mut frame = [0u8; 8];
        if self.read_until_eof(&mut frame)? < frame.len() {
            return Ok(None);
        }
        let length = u32::from_le_bytes([frame[0], frame[1], frame[2], frame[3]]) as usize;
        let expected_crc = u32::from_le_bytes([frame[4], frame[5], frame[6], frame[7]]);
        if length == 0 || length > WAL_MAX_RECORD_SIZE {
            return Err(self.corruption(&format!("invalid record length {length}")));
        }

        let mut content = vec![0u8; length];
        let read = self.read_until_eof(&mut content)?;
        if read < length {
            if whole_record_follows(&[&frame[..], &content[..read]].concat()) {
                return Err(self.corruption("truncated record followed by a whole record"));
            }
            return Ok(None);
        }
        if crc32c_hash(&content, 0) != expected_crc {
            return Err(self.corruption("record checksum mismatch"));
        }
        self.valid_end += (frame.len() + length) as u64;
        Ok(Some(content))
    }

    pub(crate) fn valid_end(&self) -> u64 {
        self.valid_end
    }

    // Fills `buf` unless the file ends first; returns how many bytes were read.
    fn read_until_eof(&mut self, buf: &mut [u8]) -> ZResult<usize> {
        let mut filled = 0;
        while filled < buf.len() {
            match self.reader.read(&mut buf[filled..]) {
                Ok(0) => break,
                Ok(n) => filled += n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(Status::io_error(e.to_string())),
            }
        }
        Ok(filled)
    }

    fn corruption(&self, what: &str) -> Status {
        Status::io_error(format!(
            "WAL {} is corrupt at offset {}: {}; inspect or move it aside before reopening",
            self.path.display(),
            self.valid_end,
            what
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("finch_wal_record_file_test_{}_{}.wal", name, nanos));
        p
    }

    #[test]
    fn general_append_reopen_read() {
        let path = tmp_path("general");
        let mut wal = WalRecordFileWriter::open(
            path.clone(),
            WalFileOptions {
                create_new: true,
                max_docs_wal_flush: 0,
            },
        )
        .unwrap();

        for _ in 0..100 {
            wal.append(b"hello").unwrap();
        }
        wal.flush_fsync().unwrap();
        wal.close().unwrap();

        // reopen and append 100..200
        let mut wal = WalRecordFileWriter::open(
            path.clone(),
            WalFileOptions {
                create_new: false,
                max_docs_wal_flush: 1,
            },
        )
        .unwrap();
        for i in 100..200 {
            wal.append(format!("hello{}", i).as_bytes()).unwrap();
        }
        wal.flush_fsync().unwrap();
        wal.close().unwrap();

        // reopen and append 200..300
        let mut wal = WalRecordFileWriter::open(
            path.clone(),
            WalFileOptions {
                create_new: false,
                max_docs_wal_flush: 1,
            },
        )
        .unwrap();
        for i in 200..300 {
            wal.append(format!("hello{}", i).as_bytes()).unwrap();
        }
        wal.flush_fsync().unwrap();
        wal.close().unwrap();

        // reopen and append 300..400 in "batch" mode
        let mut wal = WalRecordFileWriter::open(
            path.clone(),
            WalFileOptions {
                create_new: false,
                max_docs_wal_flush: 10,
            },
        )
        .unwrap();
        for i in 300..400 {
            wal.append(format!("hello{}", i).as_bytes()).unwrap();
        }
        wal.flush_fsync().unwrap();
        wal.close().unwrap();

        // read
        let mut reader = WalRecordFileReader::open(path.clone()).unwrap();
        reader.prepare_for_read().unwrap();
        let mut idx = 0u32;
        loop {
            let Some(rec) = reader.next().unwrap() else {
                break;
            };
            let s = String::from_utf8(rec).unwrap();
            if idx < 100 {
                assert_eq!(s, "hello");
            } else {
                assert_eq!(s, format!("hello{}", idx));
            }
            idx += 1;
        }
        assert_eq!(idx, 400);
        drop(reader);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn multi_thread_append_and_read_count() {
        let path = tmp_path("mt");
        let wal = WalRecordFileWriter::open(
            path.clone(),
            WalFileOptions {
                create_new: true,
                max_docs_wal_flush: 1,
            },
        )
        .unwrap();
        let wal = std::sync::Arc::new(std::sync::Mutex::new(wal));

        let threads = 10usize;
        let total = 10_000usize;
        let chunk = total / threads;
        let mut handles = Vec::new();
        for t in 0..threads {
            let wal = wal.clone();
            let start = t * chunk;
            let end = if t + 1 == threads {
                total
            } else {
                (t + 1) * chunk
            };
            handles.push(std::thread::spawn(move || {
                for i in start..end {
                    let rec = format!("hello{}", i);
                    wal.lock().unwrap().append(rec.as_bytes()).unwrap();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        wal.lock().unwrap().flush_fsync().unwrap();
        drop(wal);

        // reopen for batch mode and append 10_000..20_000
        let wal = WalRecordFileWriter::open(
            path.clone(),
            WalFileOptions {
                create_new: false,
                max_docs_wal_flush: 1000,
            },
        )
        .unwrap();
        let wal = std::sync::Arc::new(std::sync::Mutex::new(wal));
        let total = 10_000usize;
        let chunk = total / threads;
        let mut handles = Vec::new();
        for t in 0..threads {
            let wal = wal.clone();
            let start = 10_000 + t * chunk;
            let end = if t + 1 == threads {
                20_000
            } else {
                10_000 + (t + 1) * chunk
            };
            handles.push(std::thread::spawn(move || {
                for i in start..end {
                    let rec = format!("hello{}", i);
                    wal.lock().unwrap().append(rec.as_bytes()).unwrap();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        wal.lock().unwrap().flush_fsync().unwrap();
        drop(wal);

        // reopen with auto flush disabled and append 20_000..30_000
        let wal = WalRecordFileWriter::open(
            path.clone(),
            WalFileOptions {
                create_new: false,
                max_docs_wal_flush: 0,
            },
        )
        .unwrap();
        let wal = std::sync::Arc::new(std::sync::Mutex::new(wal));
        let total = 10_000usize;
        let chunk = total / threads;
        let mut handles = Vec::new();
        for t in 0..threads {
            let wal = wal.clone();
            let start = 20_000 + t * chunk;
            let end = if t + 1 == threads {
                30_000
            } else {
                20_000 + (t + 1) * chunk
            };
            handles.push(std::thread::spawn(move || {
                for i in start..end {
                    let rec = format!("hello{}", i);
                    wal.lock().unwrap().append(rec.as_bytes()).unwrap();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        wal.lock().unwrap().flush_fsync().unwrap();
        drop(wal);

        let mut reader = WalRecordFileReader::open(path.clone()).unwrap();
        reader.prepare_for_read().unwrap();
        let mut count = 0u32;
        while reader.next().unwrap().is_some() {
            count += 1;
        }
        assert_eq!(count, 30_000);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn boundary_conditions() {
        let path = tmp_path("boundary");

        // empty file read
        let wal = WalRecordFileWriter::open(
            path.clone(),
            WalFileOptions {
                create_new: true,
                max_docs_wal_flush: 1,
            },
        )
        .unwrap();
        wal.close().unwrap();
        let mut reader = WalRecordFileReader::open(path.clone()).unwrap();
        reader.prepare_for_read().unwrap();
        let mut count = 0u32;
        while reader.next().unwrap().is_some() {
            count += 1;
        }
        assert_eq!(count, 0);
        let _ = fs::remove_file(&path);

        // write and read binary payload
        let mut wal = WalRecordFileWriter::open(
            path.clone(),
            WalFileOptions {
                create_new: true,
                max_docs_wal_flush: 1,
            },
        )
        .unwrap();
        wal.append(&[0, 1, 2, 3]).unwrap();
        wal.flush_fsync().unwrap();
        wal.close().unwrap();
        let mut reader = WalRecordFileReader::open(path.clone()).unwrap();
        reader.prepare_for_read().unwrap();
        let rec = reader.next().unwrap().unwrap();
        assert_eq!(rec, vec![0, 1, 2, 3]);
        assert!(reader.next().unwrap().is_none());
        let _ = fs::remove_file(&path);

        // write very large record 4MB
        let mut big = vec![0u8; WAL_MAX_RECORD_SIZE];
        for (i, b) in big.iter_mut().enumerate() {
            *b = (i % 256) as u8;
        }
        let mut wal = WalRecordFileWriter::open(
            path.clone(),
            WalFileOptions {
                create_new: true,
                max_docs_wal_flush: 1,
            },
        )
        .unwrap();
        wal.append(&big).unwrap();
        wal.flush_fsync().unwrap();
        wal.close().unwrap();
        let mut reader = WalRecordFileReader::open(path.clone()).unwrap();
        reader.prepare_for_read().unwrap();
        let got = reader.next().unwrap().unwrap();
        assert_eq!(got.len(), WAL_MAX_RECORD_SIZE);
        assert_eq!(got, big);
        let _ = fs::remove_file(&path);

        // batch model 100, just add 99 record and close
        let mut wal = WalRecordFileWriter::open(
            path.clone(),
            WalFileOptions {
                create_new: true,
                max_docs_wal_flush: 100,
            },
        )
        .unwrap();
        for i in 0..99u32 {
            wal.append(format!("hello{}", i).as_bytes()).unwrap();
        }
        wal.flush_fsync().unwrap();
        wal.close().unwrap();
        let mut reader = WalRecordFileReader::open(path.clone()).unwrap();
        reader.prepare_for_read().unwrap();
        let mut idx = 0u32;
        loop {
            let Some(rec) = reader.next().unwrap() else {
                break;
            };
            assert_eq!(String::from_utf8(rec).unwrap(), format!("hello{}", idx));
            idx += 1;
        }
        assert_eq!(idx, 99);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn not_exist_open_errors() {
        let path = tmp_path("notexist");
        let Err(err) = WalRecordFileWriter::open(
            path,
            WalFileOptions {
                create_new: false,
                max_docs_wal_flush: 0,
            },
        ) else {
            panic!("opening a missing WAL must fail");
        };
        assert_eq!(err.code, finch_types::StatusCode::IoError);
    }

    fn write_hello_records(path: &Path, count: usize) {
        let _ = fs::remove_file(path);
        let mut wal = WalRecordFileWriter::open(
            path.to_path_buf(),
            WalFileOptions {
                create_new: true,
                max_docs_wal_flush: 1,
            },
        )
        .unwrap();
        for _ in 0..count {
            wal.append(b"hello").unwrap();
        }
        wal.close().unwrap();
    }

    fn overwrite_at(path: &Path, offset: usize, bytes: &[u8]) {
        let mut f = fs::OpenOptions::new().write(true).open(path).unwrap();
        f.seek(SeekFrom::Start(offset as u64)).unwrap();
        f.write_all(bytes).unwrap();
        f.sync_all().unwrap();
    }

    // Records read before the reader stopped, and the error when it stopped on corruption.
    fn read_until_stop(path: &Path) -> (usize, Option<Status>) {
        let mut reader = WalRecordFileReader::open(path.to_path_buf()).unwrap();
        reader.prepare_for_read().unwrap();
        let mut count = 0;
        loop {
            match reader.next() {
                Ok(Some(_)) => count += 1,
                Ok(None) => return (count, None),
                Err(e) => return (count, Some(e)),
            }
        }
    }

    // Each "hello" record is 8 frame bytes plus 5 payload bytes.
    const HELLO_RECORD: usize = 13;

    #[test]
    fn corrupt_records_fail_and_torn_tails_end_replay() {
        let path = tmp_path("corrupt");
        let corrupt_at = |offset: usize| format!("corrupt at offset {offset}");

        // Clean end.
        write_hello_records(&path, 10);
        let (count, err) = read_until_stop(&path);
        assert_eq!((count, err.is_none()), (10, true));

        // Checksum mismatch in the first record's payload.
        write_hello_records(&path, 10);
        overwrite_at(&path, WAL_HEADER_SIZE + 8, b"nihao");
        let (count, err) = read_until_stop(&path);
        assert_eq!(count, 0);
        assert!(err.unwrap().message.contains(&corrupt_at(WAL_HEADER_SIZE)));

        // Checksum mismatch in the sixth record, with records after it.
        write_hello_records(&path, 10);
        overwrite_at(&path, WAL_HEADER_SIZE + 5 * HELLO_RECORD + 8, b"nihao");
        let (count, err) = read_until_stop(&path);
        assert_eq!(count, 5);
        assert!(err
            .unwrap()
            .message
            .contains(&corrupt_at(WAL_HEADER_SIZE + 5 * HELLO_RECORD)));

        // Checksum field damaged in the second record.
        write_hello_records(&path, 10);
        overwrite_at(
            &path,
            WAL_HEADER_SIZE + HELLO_RECORD + 4,
            &123u32.to_le_bytes(),
        );
        let (count, err) = read_until_stop(&path);
        assert_eq!(count, 1);
        assert!(err
            .unwrap()
            .message
            .contains(&corrupt_at(WAL_HEADER_SIZE + HELLO_RECORD)));

        // A shorter length makes the checksum cover the wrong bytes.
        write_hello_records(&path, 10);
        overwrite_at(&path, WAL_HEADER_SIZE, &2u32.to_le_bytes());
        let (count, err) = read_until_stop(&path);
        assert_eq!(count, 0);
        assert!(err.unwrap().message.contains("checksum mismatch"));

        // A zero length can never be written.
        write_hello_records(&path, 10);
        overwrite_at(&path, WAL_HEADER_SIZE, &0u32.to_le_bytes());
        let (count, err) = read_until_stop(&path);
        assert_eq!(count, 0);
        assert!(err.unwrap().message.contains("invalid record length 0"));

        // Final record cut short inside its payload: a torn tail.
        write_hello_records(&path, 10);
        let len = fs::metadata(&path).unwrap().len();
        fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(len - 4)
            .unwrap();
        let (count, err) = read_until_stop(&path);
        assert_eq!((count, err.is_none()), (9, true));

        // Final record cut short inside its frame: a torn tail.
        write_hello_records(&path, 10);
        fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(&[0xff, 0xff])
            .unwrap();
        let (count, err) = read_until_stop(&path);
        assert_eq!((count, err.is_none()), (10, true));

        // A length running past the end of the file reads as a torn final record.
        write_hello_records(&path, 10);
        overwrite_at(
            &path,
            WAL_HEADER_SIZE + 9 * HELLO_RECORD,
            &200u32.to_le_bytes(),
        );
        let (count, err) = read_until_stop(&path);
        assert_eq!((count, err.is_none()), (9, true));

        // A truncated record followed by a whole record is corruption, not a torn tail.
        write_hello_records(&path, 2);
        let mut bytes = fs::read(&path).unwrap();
        let second = bytes.split_off(WAL_HEADER_SIZE + HELLO_RECORD);
        bytes.extend_from_slice(&100u32.to_le_bytes());
        bytes.extend_from_slice(&[0, 0, 0, 0, b'p', b'a', b'r']);
        bytes.extend_from_slice(&second);
        fs::write(&path, &bytes).unwrap();
        let (count, err) = read_until_stop(&path);
        assert_eq!(count, 1);
        assert!(err
            .unwrap()
            .message
            .contains("truncated record followed by a whole record"));

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn reopening_truncates_a_torn_tail_before_appending() {
        let path = tmp_path("torn_append");
        write_hello_records(&path, 3);
        let valid_len = fs::metadata(&path).unwrap().len();
        fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(&[5, 0, 0, 0, 1, 2, 3])
            .unwrap();

        let mut wal = WalRecordFileWriter::open(
            path.clone(),
            WalFileOptions {
                create_new: false,
                max_docs_wal_flush: 1,
            },
        )
        .unwrap();
        assert_eq!(fs::metadata(&path).unwrap().len(), valid_len);
        wal.append(b"after").unwrap();
        wal.close().unwrap();

        let (count, err) = read_until_stop(&path);
        assert_eq!((count, err.is_none()), (4, true));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn reopening_a_corrupt_wal_for_append_fails() {
        let path = tmp_path("corrupt_append");
        write_hello_records(&path, 3);
        overwrite_at(&path, WAL_HEADER_SIZE + 8, b"nihao");
        let len = fs::metadata(&path).unwrap().len();

        let result = WalRecordFileWriter::open(
            path.clone(),
            WalFileOptions {
                create_new: false,
                max_docs_wal_flush: 1,
            },
        );
        assert!(result.is_err());
        assert_eq!(
            fs::metadata(&path).unwrap().len(),
            len,
            "corrupt WAL must not be truncated"
        );
        let _ = fs::remove_file(&path);
    }
}

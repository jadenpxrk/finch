//! Immutable file of key/value records in strictly increasing key order, written once and
//! replaced by rename. Id-map checkpoints and persisted invert indexes use it.
//!
//! Layout, little-endian: magic, then per record `[u32 key_len][key][u32 val_len][val]`, then
//! one `u64` record offset per record, then `u64` record count, then `u32` crc32c of all
//! preceding bytes.

use std::cmp::Ordering;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use finch_types::{Status, ZResult};
use memmap2::Mmap;

use crate::wal::crc32c_hash;

const MAGIC: &[u8; 8] = b"FNCHSRT1";
const FOOTER_LEN: usize = 8 + 4;

fn io_err(e: std::io::Error) -> Status {
    Status::io_error(e.to_string())
}

pub(crate) fn sync_dir(dir: &Path) -> ZResult<()> {
    fs::File::open(dir)
        .and_then(|d| d.sync_all())
        .map_err(|e| Status::io_error(format!("could not sync directory {:?}: {}", dir, e)))
}

struct CrcWriter {
    out: BufWriter<fs::File>,
    crc: u32,
    pos: u64,
}

impl CrcWriter {
    fn put(&mut self, bytes: &[u8]) -> ZResult<()> {
        self.out.write_all(bytes).map_err(io_err)?;
        self.crc = crc32c_hash(bytes, self.crc);
        self.pos += bytes.len() as u64;
        Ok(())
    }

    fn put_len(&mut self, len: usize) -> ZResult<()> {
        let len = u32::try_from(len)
            .map_err(|_| Status::invalid_argument("sorted file record exceeds 4 GiB"))?;
        self.put(&len.to_le_bytes())
    }
}

/// Writes `entries` to `path` through a temp file, fsyncs it, renames it into place, and
/// fsyncs the directory. Fails if keys are not strictly increasing.
pub(crate) fn write_sorted_file<K, V>(
    path: &Path,
    entries: impl IntoIterator<Item = ZResult<(K, V)>>,
) -> ZResult<()>
where
    K: AsRef<[u8]>,
    V: AsRef<[u8]>,
{
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    let file = fs::File::create(&tmp).map_err(io_err)?;
    let mut w = CrcWriter {
        out: BufWriter::new(file),
        crc: 0,
        pos: 0,
    };
    w.put(MAGIC)?;
    let mut offsets: Vec<u64> = Vec::new();
    let mut prev: Option<Vec<u8>> = None;
    for entry in entries {
        let (key, val) = entry?;
        let (key, val) = (key.as_ref(), val.as_ref());
        if prev.as_deref().is_some_and(|p| p >= key) {
            return Err(Status::internal(
                "sorted file keys must be strictly increasing",
            ));
        }
        offsets.push(w.pos);
        w.put_len(key.len())?;
        w.put(key)?;
        w.put_len(val.len())?;
        w.put(val)?;
        prev = Some(key.to_vec());
    }
    for off in &offsets {
        w.put(&off.to_le_bytes())?;
    }
    w.put(&(offsets.len() as u64).to_le_bytes())?;
    let crc = w.crc;
    let mut out = w.out;
    out.write_all(&crc.to_le_bytes()).map_err(io_err)?;
    let file = out.into_inner().map_err(|e| io_err(e.into_error()))?;
    file.sync_all().map_err(io_err)?;
    drop(file);
    fs::rename(&tmp, path).map_err(io_err)?;
    sync_dir(path.parent().unwrap_or(Path::new(".")))
}

fn read_u32(data: &[u8], pos: usize) -> Option<u32> {
    Some(u32::from_le_bytes(data.get(pos..pos + 4)?.try_into().ok()?))
}

fn read_u64(data: &[u8], pos: usize) -> Option<u64> {
    Some(u64::from_le_bytes(data.get(pos..pos + 8)?.try_into().ok()?))
}

// The record at `pos` within `data`, and the position after it.
fn read_record(data: &[u8], pos: usize) -> Option<(&[u8], &[u8], usize)> {
    let key_len = read_u32(data, pos)? as usize;
    let key_start = pos + 4;
    let key = data.get(key_start..key_start.checked_add(key_len)?)?;
    let val_len_pos = key_start + key_len;
    let val_len = read_u32(data, val_len_pos)? as usize;
    let val_start = val_len_pos + 4;
    let val_end = val_start.checked_add(val_len)?;
    let val = data.get(val_start..val_end)?;
    Some((key, val, val_end))
}

pub(crate) struct SortedFile {
    data: Mmap,
    offsets_start: usize,
    count: usize,
}

impl SortedFile {
    /// Maps `path` and checks its checksum and every record, so lookups never see a torn file.
    pub(crate) fn open(path: &Path) -> ZResult<Self> {
        let data = finch_storage::get_mmap(path)?;
        let corrupt = || Status::internal(format!("sorted file {:?} is corrupt", path));
        let body_len = data
            .len()
            .checked_sub(4)
            .filter(|&n| n >= MAGIC.len() + 8)
            .ok_or_else(corrupt)?;
        if &data[..MAGIC.len()] != MAGIC
            || read_u32(&data, body_len) != Some(crc32c_hash(&data[..body_len], 0))
        {
            return Err(corrupt());
        }
        let count = read_u64(&data, data.len() - FOOTER_LEN).ok_or_else(corrupt)?;
        let offsets_start = usize::try_from(count)
            .ok()
            .and_then(|c| c.checked_mul(8))
            .and_then(|n| (data.len() - FOOTER_LEN).checked_sub(n))
            .filter(|&n| n >= MAGIC.len())
            .ok_or_else(corrupt)?;
        let file = SortedFile {
            offsets_start,
            count: count as usize,
            data,
        };
        let records = &file.data[..file.offsets_start];
        let mut pos = MAGIC.len();
        let mut prev: Option<&[u8]> = None;
        for i in 0..file.count {
            if read_u64(&file.data, file.offsets_start + i * 8) != Some(pos as u64) {
                return Err(corrupt());
            }
            let (key, _, next) = read_record(records, pos).ok_or_else(corrupt)?;
            if prev.is_some_and(|p| p >= key) {
                return Err(corrupt());
            }
            prev = Some(key);
            pos = next;
        }
        if pos != file.offsets_start {
            return Err(corrupt());
        }
        Ok(file)
    }

    fn entry(&self, i: usize) -> Option<(&[u8], &[u8])> {
        let pos = read_u64(&self.data, self.offsets_start + i * 8)? as usize;
        let (key, val, _) = read_record(&self.data[..self.offsets_start], pos)?;
        Some((key, val))
    }

    // Index of the first record whose key is not less than `key`.
    fn lower_bound(&self, key: &[u8]) -> usize {
        let (mut lo, mut hi) = (0, self.count);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match self.entry(mid).map(|(k, _)| k.cmp(key)) {
                Some(Ordering::Less) => lo = mid + 1,
                _ => hi = mid,
            }
        }
        lo
    }

    pub(crate) fn get(&self, key: &[u8]) -> Option<&[u8]> {
        self.entry(self.lower_bound(key))
            .filter(|(k, _)| *k == key)
            .map(|(_, v)| v)
    }

    /// Records with key `>= start`, in key order.
    pub(crate) fn iter_from(&self, start: &[u8]) -> impl Iterator<Item = (&[u8], &[u8])> + '_ {
        (self.lower_bound(start)..self.count).map_while(|i| self.entry(i))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_rejects_corruption() {
        let dir = std::env::temp_dir().join(format!("finch_sorted_file_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("f");
        let entries = [("a", "1"), ("ab", ""), ("b", "22")];
        write_sorted_file(&path, entries.iter().map(|&(k, v)| Ok((k, v)))).unwrap();

        let file = SortedFile::open(&path).unwrap();
        assert_eq!(file.get(b"ab"), Some(&b""[..]));
        assert_eq!(file.get(b"aa"), None);
        let from_aa: Vec<_> = file.iter_from(b"aa").map(|(k, _)| k.to_vec()).collect();
        assert_eq!(from_aa, vec![b"ab".to_vec(), b"b".to_vec()]);
        drop(file);

        let mut bytes = fs::read(&path).unwrap();
        bytes[10] ^= 1;
        fs::write(&path, &bytes).unwrap();
        assert!(SortedFile::open(&path).is_err());

        let unsorted = [("b", ""), ("a", "")];
        assert!(write_sorted_file(&path, unsorted.iter().map(|&(k, v)| Ok((k, v)))).is_err());
        fs::remove_dir_all(&dir).ok();
    }
}

//! Little-endian segment encoding shared by the index persistence code.

use byteorder::{LittleEndian, ReadBytesExt};
use finch_types::{Status, ZResult};
use std::io::Cursor;

pub(crate) fn read_u64(cur: &mut Cursor<&[u8]>) -> ZResult<u64> {
    cur.read_u64::<LittleEndian>()
        .map_err(|e| Status::io_error(e.to_string()))
}

pub(crate) fn read_u32(cur: &mut Cursor<&[u8]>) -> ZResult<u32> {
    cur.read_u32::<LittleEndian>()
        .map_err(|e| Status::io_error(e.to_string()))
}

pub(crate) fn read_u16(cur: &mut Cursor<&[u8]>) -> ZResult<u16> {
    cur.read_u16::<LittleEndian>()
        .map_err(|e| Status::io_error(e.to_string()))
}

pub(crate) fn read_f32(cur: &mut Cursor<&[u8]>) -> ZResult<f32> {
    cur.read_f32::<LittleEndian>()
        .map_err(|e| Status::io_error(e.to_string()))
}

pub(crate) fn read_u8(cur: &mut Cursor<&[u8]>) -> ZResult<u8> {
    cur.read_u8().map_err(|e| Status::io_error(e.to_string()))
}

pub(crate) fn u64s_to_le(values: &[u64]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(values.len() * 8);
    for &v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

pub(crate) fn u32s_to_le(values: &[u32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(values.len() * 4);
    for &v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

/// Views `len` u32s at byte offset `off`; empty when out of bounds or misaligned.
#[inline]
pub(crate) fn u32_slice_at(bytes: &[u8], off: usize, len: usize) -> &[u32] {
    let byte_len = len * 4;
    if off + byte_len > bytes.len() {
        return &[];
    }
    let ptr = bytes[off..].as_ptr();
    if !(ptr as usize).is_multiple_of(std::mem::align_of::<u32>()) {
        return &[];
    }
    // SAFETY: `off + len * 4 <= bytes.len()` and u32 alignment are checked above.
    unsafe { std::slice::from_raw_parts(ptr as *const u32, len) }
}

/// One node's neighbor run in the `[count][ids...]` upper-level layout.
#[derive(Clone, Copy)]
pub(crate) struct UpperRun {
    pub(crate) start: u32,
    pub(crate) len: u32,
}

/// Error messages for an upper-level scan; they differ per index family.
pub(crate) struct UpperScanErrors {
    pub(crate) overflow: &'static str,
    pub(crate) truncated: &'static str,
}

/// Scans the `[num_levels]([node_count]([count][ids...])*)*` upper-neighbor layout.
pub(crate) fn scan_upper_runs<T>(
    bytes: &[u8],
    errors: &UpperScanErrors,
    map: impl Fn(UpperRun) -> T,
) -> ZResult<Vec<Vec<T>>> {
    let mut cur = Cursor::new(bytes);
    let num_levels = read_u32(&mut cur)? as usize;
    let mut levels = Vec::with_capacity(num_levels);
    for _ in 0..num_levels {
        let node_count = read_u32(&mut cur)? as usize;
        let mut runs = Vec::with_capacity(node_count);
        for _ in 0..node_count {
            let nb_count = read_u32(&mut cur)? as usize;
            let start = cur.position() as usize;
            let skip = nb_count
                .checked_mul(4)
                .ok_or_else(|| Status::io_error(errors.overflow))?;
            let end = start
                .checked_add(skip)
                .ok_or_else(|| Status::io_error(errors.overflow))?;
            if end > bytes.len() {
                return Err(Status::io_error(errors.truncated));
            }
            runs.push(map(UpperRun {
                start: start as u32,
                len: nb_count as u32,
            }));
            cur.set_position(end as u64);
        }
        levels.push(runs);
    }
    Ok(levels)
}

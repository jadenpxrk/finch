//! Inverted index for scalar field filtering (backed by fjall LSM-tree)

use finch_types::{DataType, InvertIndexParams, Status, Value, ZResult};
use fjall::{Database, KeyspaceCreateOptions, PersistMode};
use half::f16;
use roaring::RoaringTreemap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const NS_TERMS: u8 = 1;
const NS_REVERSED_TERMS: u8 = 2;
const NS_ARRAY_LEN: u8 = 3;
const NS_NULL: u8 = 4;
const NS_NONNULL: u8 = 5;
const VALUE_DELIM: u8 = 0;

static INV_RO_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Temp directory that is deleted on drop.
struct CleanupDir(PathBuf);
impl Drop for CleanupDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> ZResult<()> {
    std::fs::create_dir_all(dst).map_err(|e| Status::io_error(e.to_string()))?;
    for entry in std::fs::read_dir(src).map_err(|e| Status::io_error(e.to_string()))? {
        let entry = entry.map_err(|e| Status::io_error(e.to_string()))?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if src_path.is_dir() {
            copy_dir_recursive(&src_path, &dst_path)?;
        } else {
            std::fs::copy(&src_path, &dst_path).map_err(|e| Status::io_error(e.to_string()))?;
        }
    }
    Ok(())
}

pub(crate) fn array_element_type(dt: DataType) -> Option<DataType> {
    use DataType as D;
    match dt {
        D::ArrayBinary => Some(D::Bytes),
        D::ArrayString => Some(D::String),
        D::ArrayBool => Some(D::Bool),
        D::ArrayInt32 => Some(D::Int32),
        D::ArrayInt64 => Some(D::Int64),
        D::ArrayUint32 => Some(D::Uint32),
        D::ArrayUint64 => Some(D::Uint64),
        D::ArrayFp32 => Some(D::Float32),
        D::ArrayFp64 => Some(D::Float64),
        _ => None,
    }
}

fn is_array_type(dt: DataType) -> bool {
    array_element_type(dt).is_some()
}

/// Encode a field value to a byte key for the KV store.
/// Key format:
///   [field_name_len:2 little-endian][field_name_bytes][namespace:1]
///   [value_bytes][value_delim=0x00][doc_id:8 big-endian]
///
/// Notes:
/// - The DB is already per-field (`{field}_invert`), but we keep the field
///   header to make it easy to scan and to allow future multi-field storage.
/// - `namespace` separates forward terms from reversed terms (suffix search).
/// - `value_bytes` are encoded to preserve lexicographic ordering for numeric
///   range scans when `enable_range_optimization` is enabled.
fn encode_header(field: &str, namespace: u8) -> Vec<u8> {
    let mut key = Vec::new();
    let fn_bytes = field.as_bytes();
    key.extend_from_slice(&(fn_bytes.len() as u16).to_le_bytes());
    key.extend_from_slice(fn_bytes);
    key.push(namespace);
    key
}

fn encode_value_bytes_ordered(dt: DataType, value: &Value, out: &mut Vec<u8>) -> Option<()> {
    use DataType as D;
    match (dt, value) {
        (D::Bool, Value::Bool(b)) => out.push(*b as u8),
        (D::Int8, Value::I8(v)) => out.push((*v as u8) ^ 0x80),
        (D::Int16, Value::I16(v)) => out.extend_from_slice(&(((*v as u16) ^ 0x8000).to_be_bytes())),
        (D::Int32, Value::I32(v)) => {
            out.extend_from_slice(&(((*v as u32) ^ 0x8000_0000).to_be_bytes()))
        }
        (D::Int64, Value::I64(v)) => {
            out.extend_from_slice(&(((*v as u64) ^ 0x8000_0000_0000_0000).to_be_bytes()))
        }
        (D::Uint8, Value::U8(v)) => out.push(*v),
        (D::Uint16, Value::U16(v)) => out.extend_from_slice(&v.to_be_bytes()),
        (D::Uint32, Value::U32(v)) => out.extend_from_slice(&v.to_be_bytes()),
        (D::Uint64, Value::U64(v)) => out.extend_from_slice(&v.to_be_bytes()),
        (D::Float16, Value::F16(v)) => {
            let mut bits = v.to_bits();
            if bits == 0x8000 {
                // Canonicalize -0.0 to +0.0 so equality lookups match float semantics.
                bits = 0;
            }
            bits = if (bits & 0x8000) != 0 {
                !bits
            } else {
                bits ^ 0x8000
            };
            out.extend_from_slice(&bits.to_be_bytes());
        }
        (D::Float32, Value::F32(v)) => {
            let mut bits = v.to_bits();
            if bits == 0x8000_0000 {
                bits = 0;
            }
            bits = if (bits & 0x8000_0000) != 0 {
                !bits
            } else {
                bits ^ 0x8000_0000
            };
            out.extend_from_slice(&bits.to_be_bytes());
        }
        (D::Float64, Value::F64(v)) => {
            let mut bits = v.to_bits();
            if bits == 0x8000_0000_0000_0000 {
                bits = 0;
            }
            bits = if (bits & 0x8000_0000_0000_0000) != 0 {
                !bits
            } else {
                bits ^ 0x8000_0000_0000_0000
            };
            out.extend_from_slice(&bits.to_be_bytes());
        }
        (D::String, Value::String(s)) => {
            // Raw bytes + a delimiter so equality keys never match prefixes; document
            // validation rejects NUL in indexed strings.
            if s.as_bytes().contains(&VALUE_DELIM) {
                return None;
            }
            out.extend_from_slice(s.as_bytes());
        }
        (D::Bytes | D::Binary, Value::Bytes(b)) => {
            // Escape NUL as 00 01 and end with 00, so with VALUE_DELIM a term ends in 00 00 and
            // key order equals content order for range scans.
            for &byte in b {
                out.push(byte);
                if byte == VALUE_DELIM {
                    out.push(1);
                }
            }
            out.push(VALUE_DELIM);
        }
        _ => return None,
    }
    Some(())
}

fn encode_full_key(
    field: &str,
    namespace: u8,
    dt: DataType,
    value: &Value,
    doc_id: u64,
) -> Option<Vec<u8>> {
    let mut key = encode_header(field, namespace);
    encode_value_bytes_ordered(dt, value, &mut key)?;
    key.push(VALUE_DELIM);
    key.extend_from_slice(&doc_id.to_be_bytes());
    Some(key)
}

fn encode_marker_key(field: &str, namespace: u8, doc_id: u64) -> Vec<u8> {
    let mut key = encode_header(field, namespace);
    key.push(VALUE_DELIM);
    key.extend_from_slice(&doc_id.to_be_bytes());
    key
}

fn namespace_prefix(field: &str, namespace: u8) -> Vec<u8> {
    encode_header(field, namespace)
}

pub struct InvertIndex {
    db: Database,
    items: fjall::Keyspace,
    _ro_cleanup: Option<CleanupDir>,
    pub field_name: String,
    pub data_type: DataType,
    index_value_type: DataType,
    pub params: InvertIndexParams,
    enable_range_optimization_terms: bool,
    enable_range_optimization_array_len: bool,
    enable_extended_wildcard: bool,
}

impl InvertIndex {
    pub fn open(
        path: &Path,
        field_name: String,
        data_type: DataType,
        params: InvertIndexParams,
    ) -> ZResult<Self> {
        let db = Database::builder(path)
            .open()
            .map_err(|e| Status::io_error(e.to_string()))?;
        let items = db
            .keyspace("invert", KeyspaceCreateOptions::default)
            .map_err(|e| Status::io_error(e.to_string()))?;
        let index_value_type = array_element_type(data_type).unwrap_or(data_type);
        let enable_range_optimization_terms =
            params.enable_range_optimization && !matches!(index_value_type, DataType::Bool);
        let enable_range_optimization_array_len =
            params.enable_range_optimization && is_array_type(data_type);
        let enable_extended_wildcard =
            params.enable_extended_wildcard && matches!(index_value_type, DataType::String);
        Ok(InvertIndex {
            db,
            items,
            _ro_cleanup: None,
            field_name,
            data_type,
            index_value_type,
            params,
            enable_range_optimization_terms,
            enable_range_optimization_array_len,
            enable_extended_wildcard,
        })
    }

    pub fn open_read_only(
        path: &Path,
        field_name: String,
        data_type: DataType,
        params: InvertIndexParams,
    ) -> ZResult<Self> {
        // fjall holds an exclusive lock per Database, so read-only opens
        // copy the database directory to a unique temp path.
        let seq = INV_RO_COUNTER.fetch_add(1, Ordering::Relaxed);
        let ro_name = format!("invert_ro_{}_{}", std::process::id(), seq);
        let dst = path.parent().unwrap_or(path).join(&ro_name);
        copy_dir_recursive(path, &dst)?;
        let db = Database::builder(&dst).open().map_err(|e| {
            let _ = std::fs::remove_dir_all(&dst);
            Status::io_error(e.to_string())
        })?;
        let items = db
            .keyspace("invert", KeyspaceCreateOptions::default)
            .map_err(|e| Status::io_error(e.to_string()))?;
        let index_value_type = array_element_type(data_type).unwrap_or(data_type);
        let enable_range_optimization_terms =
            params.enable_range_optimization && !matches!(index_value_type, DataType::Bool);
        let enable_range_optimization_array_len =
            params.enable_range_optimization && is_array_type(data_type);
        let enable_extended_wildcard =
            params.enable_extended_wildcard && matches!(index_value_type, DataType::String);
        Ok(InvertIndex {
            db,
            items,
            _ro_cleanup: Some(CleanupDir(dst)),
            field_name,
            data_type,
            index_value_type,
            params,
            enable_range_optimization_terms,
            enable_range_optimization_array_len,
            enable_extended_wildcard,
        })
    }

    /// Flush all writes to disk so they survive reopen.
    pub fn sync(&self) -> ZResult<()> {
        self.db
            .persist(PersistMode::SyncAll)
            .map_err(|e| Status::io_error(e.to_string()))
    }

    // ── Write helpers ────────────────────────────────────────────────────

    fn put_kv(&self, key: &[u8], value: &[u8]) -> ZResult<()> {
        self.items
            .insert(key, value)
            .map_err(|e| Status::io_error(e.to_string()))
    }

    fn del_kv(&self, key: &[u8]) -> ZResult<()> {
        self.items
            .remove(key)
            .map_err(|e| Status::io_error(e.to_string()))
    }

    /// Iterate all keys starting with `prefix` and collect doc_ids into a bitmap.
    fn scan_prefix_doc_ids(&self, prefix: &[u8]) -> ZResult<RoaringTreemap> {
        let mut bitmap = RoaringTreemap::new();
        for item in self.items.prefix(prefix) {
            let k = item.key().map_err(|e| Status::io_error(e.to_string()))?;
            if k.len() < prefix.len() + 8 {
                continue;
            }
            let Ok(doc_id_bytes): Result<[u8; 8], _> = k[k.len() - 8..].try_into() else {
                continue;
            };
            let doc_id = u64::from_be_bytes(doc_id_bytes);
            bitmap.insert(doc_id);
        }
        Ok(bitmap)
    }

    // ── Public API ───────────────────────────────────────────────────────

    pub fn insert(&self, doc_id: u64, value: &Value) -> ZResult<()> {
        if is_array_type(self.data_type) {
            self.insert_array(doc_id, value)
        } else {
            self.insert_scalar(doc_id, value)
        }
    }

    pub fn delete(&self, doc_id: u64, value: &Value) -> ZResult<()> {
        if is_array_type(self.data_type) {
            self.delete_array(doc_id, value)
        } else {
            self.delete_scalar(doc_id, value)
        }
    }

    pub fn insert_null_marker(&self, doc_id: u64) -> ZResult<()> {
        let key = encode_marker_key(&self.field_name, NS_NULL, doc_id);
        self.put_kv(&key, b"")
    }

    pub fn delete_null_marker(&self, doc_id: u64) -> ZResult<()> {
        let key = encode_marker_key(&self.field_name, NS_NULL, doc_id);
        let _ = self.del_kv(&key);
        Ok(())
    }

    pub fn insert_nonnull_marker(&self, doc_id: u64) -> ZResult<()> {
        let key = encode_marker_key(&self.field_name, NS_NONNULL, doc_id);
        self.put_kv(&key, b"")
    }

    pub fn delete_nonnull_marker(&self, doc_id: u64) -> ZResult<()> {
        let key = encode_marker_key(&self.field_name, NS_NONNULL, doc_id);
        let _ = self.del_kv(&key);
        Ok(())
    }

    fn insert_scalar(&self, doc_id: u64, value: &Value) -> ZResult<()> {
        let key = encode_full_key(
            &self.field_name,
            NS_TERMS,
            self.index_value_type,
            value,
            doc_id,
        )
        .ok_or_else(|| {
            Status::invalid_argument(format!(
                "invert index: field[{}] cannot store this {} value",
                self.field_name,
                value.type_name()
            ))
        })?;
        self.put_kv(&key, b"")?;

        // Optional reversed-term index for suffix search.
        if self.enable_extended_wildcard && self.index_value_type == DataType::String {
            let Value::String(s) = value else {
                return Ok(());
            };
            let rev: String = s.chars().rev().collect();
            let rev_val = Value::String(rev);
            if let Some(rev_key) = encode_full_key(
                &self.field_name,
                NS_REVERSED_TERMS,
                self.index_value_type,
                &rev_val,
                doc_id,
            ) {
                let _ = self.put_kv(&rev_key, b"");
            }
        }

        Ok(())
    }

    fn delete_scalar(&self, doc_id: u64, value: &Value) -> ZResult<()> {
        if let Some(key) = encode_full_key(
            &self.field_name,
            NS_TERMS,
            self.index_value_type,
            value,
            doc_id,
        ) {
            self.del_kv(&key)?;
        }

        if self.enable_extended_wildcard && self.index_value_type == DataType::String {
            if let Value::String(s) = value {
                let rev: String = s.chars().rev().collect();
                let rev_val = Value::String(rev);
                if let Some(rev_key) = encode_full_key(
                    &self.field_name,
                    NS_REVERSED_TERMS,
                    self.index_value_type,
                    &rev_val,
                    doc_id,
                ) {
                    let _ = self.del_kv(&rev_key);
                }
            }
        }

        Ok(())
    }

    fn insert_array_len(&self, doc_id: u64, len: u32) -> ZResult<()> {
        let v = Value::U32(len);
        let key = encode_full_key(&self.field_name, NS_ARRAY_LEN, DataType::Uint32, &v, doc_id)
            .ok_or_else(|| {
                Status::invalid_argument("invert index: failed to encode array length")
            })?;
        self.put_kv(&key, b"")
    }

    fn delete_array_len(&self, doc_id: u64, len: u32) -> ZResult<()> {
        let v = Value::U32(len);
        if let Some(key) =
            encode_full_key(&self.field_name, NS_ARRAY_LEN, DataType::Uint32, &v, doc_id)
        {
            let _ = self.del_kv(&key);
        }
        Ok(())
    }

    fn insert_array(&self, doc_id: u64, value: &Value) -> ZResult<()> {
        match value {
            Value::ArrayBinary(items) => {
                self.insert_array_len(doc_id, items.len() as u32)?;
                for b in items {
                    self.insert_scalar(doc_id, &Value::Bytes(b.clone()))?;
                }
            }
            Value::ArrayString(items) => {
                self.insert_array_len(doc_id, items.len() as u32)?;
                for s in items {
                    self.insert_scalar(doc_id, &Value::String(s.clone()))?;
                }
            }
            Value::ArrayBool(items) => {
                self.insert_array_len(doc_id, items.len() as u32)?;
                for b in items {
                    self.insert_scalar(doc_id, &Value::Bool(*b))?;
                }
            }
            Value::ArrayI32(items) => {
                self.insert_array_len(doc_id, items.len() as u32)?;
                for x in items {
                    self.insert_scalar(doc_id, &Value::I32(*x))?;
                }
            }
            Value::ArrayI64(items) => {
                self.insert_array_len(doc_id, items.len() as u32)?;
                for x in items {
                    self.insert_scalar(doc_id, &Value::I64(*x))?;
                }
            }
            Value::ArrayU32(items) => {
                self.insert_array_len(doc_id, items.len() as u32)?;
                for x in items {
                    self.insert_scalar(doc_id, &Value::U32(*x))?;
                }
            }
            Value::ArrayU64(items) => {
                self.insert_array_len(doc_id, items.len() as u32)?;
                for x in items {
                    self.insert_scalar(doc_id, &Value::U64(*x))?;
                }
            }
            Value::ArrayF32(items) => {
                self.insert_array_len(doc_id, items.len() as u32)?;
                for x in items {
                    self.insert_scalar(doc_id, &Value::F32(*x))?;
                }
            }
            Value::ArrayF64(items) => {
                self.insert_array_len(doc_id, items.len() as u32)?;
                for x in items {
                    self.insert_scalar(doc_id, &Value::F64(*x))?;
                }
            }
            _ => {
                return Err(Status::invalid_argument(
                    "invert index: array field stored with non-array value",
                ));
            }
        }
        Ok(())
    }

    fn delete_array(&self, doc_id: u64, value: &Value) -> ZResult<()> {
        match value {
            Value::ArrayBinary(items) => {
                self.delete_array_len(doc_id, items.len() as u32)?;
                for b in items {
                    let _ = self.delete_scalar(doc_id, &Value::Bytes(b.clone()));
                }
            }
            Value::ArrayString(items) => {
                self.delete_array_len(doc_id, items.len() as u32)?;
                for s in items {
                    let _ = self.delete_scalar(doc_id, &Value::String(s.clone()));
                }
            }
            Value::ArrayBool(items) => {
                self.delete_array_len(doc_id, items.len() as u32)?;
                for b in items {
                    let _ = self.delete_scalar(doc_id, &Value::Bool(*b));
                }
            }
            Value::ArrayI32(items) => {
                self.delete_array_len(doc_id, items.len() as u32)?;
                for x in items {
                    let _ = self.delete_scalar(doc_id, &Value::I32(*x));
                }
            }
            Value::ArrayI64(items) => {
                self.delete_array_len(doc_id, items.len() as u32)?;
                for x in items {
                    let _ = self.delete_scalar(doc_id, &Value::I64(*x));
                }
            }
            Value::ArrayU32(items) => {
                self.delete_array_len(doc_id, items.len() as u32)?;
                for x in items {
                    let _ = self.delete_scalar(doc_id, &Value::U32(*x));
                }
            }
            Value::ArrayU64(items) => {
                self.delete_array_len(doc_id, items.len() as u32)?;
                for x in items {
                    let _ = self.delete_scalar(doc_id, &Value::U64(*x));
                }
            }
            Value::ArrayF32(items) => {
                self.delete_array_len(doc_id, items.len() as u32)?;
                for x in items {
                    let _ = self.delete_scalar(doc_id, &Value::F32(*x));
                }
            }
            Value::ArrayF64(items) => {
                self.delete_array_len(doc_id, items.len() as u32)?;
                for x in items {
                    let _ = self.delete_scalar(doc_id, &Value::F64(*x));
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Attempt to convert a filter literal into the exact `Value` variant that
    /// is stored for this field's `DataType`.
    ///
    /// Finch filter parsing uses wider numeric literal types (`I64`/`U64`/`F64`)
    /// while documents store the schema's exact scalar variant (e.g. `I32` for
    /// `DataType::Int32`). Invert-index lookups require exact byte-for-byte
    /// matches, so we normalize here for safe prefiltering.
    pub fn normalize_value(&self, value: &Value) -> Option<Value> {
        use DataType as D;

        if matches!(value, Value::Null) {
            return None;
        }

        match self.index_value_type {
            D::Bool => match value {
                Value::Bool(b) => Some(Value::Bool(*b)),
                _ => None,
            },
            D::String => match value {
                Value::String(s) => Some(Value::String(s.clone())),
                _ => None,
            },
            D::Int32 => normalize_i32(value).map(Value::I32),
            D::Int64 => normalize_i64(value).map(Value::I64),
            D::Uint32 => normalize_u32(value).map(Value::U32),
            D::Uint64 => normalize_u64(value).map(Value::U64),
            // inverted index supports float/double equality and range searches.
            // Keep normalization strict (finite only) to avoid NaN/Inf ordering surprises.
            D::Float16 => normalize_f16(value).map(Value::F16),
            D::Float32 => normalize_f32(value).map(Value::F32),
            D::Float64 => normalize_f64(value).map(Value::F64),
            D::Bytes | D::Binary => match value {
                Value::Bytes(b) => Some(Value::Bytes(b.clone())),
                _ => None,
            },
            _ => None,
        }
    }
}

mod lookup;
mod normalize;

use normalize::*;

#[cfg(test)]
mod tests;

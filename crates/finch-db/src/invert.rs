//! Inverted index for scalar field filtering (backed by fjall LSM-tree)

use finch_types::{DataType, InvertIndexParams, Status, Value, ZResult};
use fjall::{Database, KeyspaceCreateOptions, PersistMode, Readable};
use half::f16;
use roaring::RoaringTreemap;
use std::ops::Deref;
use std::path::{Path, PathBuf};

use crate::sorted_file::{write_sorted_file, SortedFile};

const NS_TERMS: u8 = 1;
const NS_REVERSED_TERMS: u8 = 2;
const NS_ARRAY_LEN: u8 = 3;
const NS_NULL: u8 = 4;
const NS_NONNULL: u8 = 5;
const VALUE_DELIM: u8 = 0;

/// Sorted key file that a persisted segment reads instead of the fjall directory at `path`.
pub(crate) fn frozen_path(path: &Path) -> PathBuf {
    let mut p = path.as_os_str().to_owned();
    p.push(".keys");
    PathBuf::from(p)
}

enum Store {
    Live {
        db: Database,
        items: fjall::Keyspace,
    },
    Frozen(SortedFile),
}

/// One index key, borrowed from a frozen file or owned by fjall.
enum Key<'a> {
    Live(fjall::Slice),
    Frozen(&'a [u8]),
}

impl Deref for Key<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Key::Live(k) => k,
            Key::Frozen(k) => k,
        }
    }
}

type Keys<'a> = Box<dyn Iterator<Item = ZResult<Key<'a>>> + 'a>;

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

fn array_items(value: &Value) -> Option<Vec<Value>> {
    Some(match value {
        Value::ArrayBinary(items) => items.iter().cloned().map(Value::Bytes).collect(),
        Value::ArrayString(items) => items.iter().cloned().map(Value::String).collect(),
        Value::ArrayBool(items) => items.iter().copied().map(Value::Bool).collect(),
        Value::ArrayI32(items) => items.iter().copied().map(Value::I32).collect(),
        Value::ArrayI64(items) => items.iter().copied().map(Value::I64).collect(),
        Value::ArrayU32(items) => items.iter().copied().map(Value::U32).collect(),
        Value::ArrayU64(items) => items.iter().copied().map(Value::U64).collect(),
        Value::ArrayF32(items) => items.iter().copied().map(Value::F32).collect(),
        Value::ArrayF64(items) => items.iter().copied().map(Value::F64).collect(),
        _ => return None,
    })
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
    store: Store,
    path: PathBuf,
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
        Ok(Self::with_store(
            Store::Live { db, items },
            path,
            field_name,
            data_type,
            params,
        ))
    }

    /// Opens the frozen file of a persisted index. An index from before frozen files is
    /// frozen first, which needs the only handle on its fjall directory.
    pub fn open_read_only(
        path: &Path,
        field_name: String,
        data_type: DataType,
        params: InvertIndexParams,
    ) -> ZResult<Self> {
        let frozen = frozen_path(path);
        if !frozen.exists() {
            Self::open(path, field_name.clone(), data_type, params.clone())?.freeze()?;
        }
        let file = SortedFile::open(&frozen)?;
        Ok(Self::with_store(
            Store::Frozen(file),
            path,
            field_name,
            data_type,
            params,
        ))
    }

    fn with_store(
        store: Store,
        path: &Path,
        field_name: String,
        data_type: DataType,
        params: InvertIndexParams,
    ) -> Self {
        let index_value_type = array_element_type(data_type).unwrap_or(data_type);
        let enable_range_optimization_terms =
            params.enable_range_optimization && !matches!(index_value_type, DataType::Bool);
        let enable_range_optimization_array_len =
            params.enable_range_optimization && is_array_type(data_type);
        let enable_extended_wildcard =
            params.enable_extended_wildcard && matches!(index_value_type, DataType::String);
        InvertIndex {
            store,
            path: path.to_path_buf(),
            field_name,
            data_type,
            index_value_type,
            params,
            enable_range_optimization_terms,
            enable_range_optimization_array_len,
            enable_extended_wildcard,
        }
    }

    /// Writes the frozen file that `open_read_only` reads, from a consistent snapshot.
    pub fn freeze(&self) -> ZResult<()> {
        let Store::Live { db, items } = &self.store else {
            return Ok(());
        };
        let snapshot = db.snapshot();
        let entries = snapshot.iter(items).map(|guard| {
            guard
                .into_inner()
                .map_err(|e| Status::io_error(e.to_string()))
        });
        write_sorted_file(&frozen_path(&self.path), entries)
    }

    /// Flush all writes to disk so they survive reopen.
    pub fn sync(&self) -> ZResult<()> {
        let Store::Live { db, .. } = &self.store else {
            return Ok(());
        };
        db.persist(PersistMode::SyncAll)
            .map_err(|e| Status::io_error(e.to_string()))
    }

    fn live_items(&self) -> ZResult<&fjall::Keyspace> {
        match &self.store {
            Store::Live { items, .. } => Ok(items),
            Store::Frozen(_) => Err(Status::permission_denied(
                "invert index of a persisted segment is read-only",
            )),
        }
    }

    /// Keys `>= start` in order.
    fn keys_from<'a>(&'a self, start: &[u8]) -> Keys<'a> {
        match &self.store {
            Store::Live { items, .. } => Box::new(items.range(start.to_vec()..).map(|item| {
                item.key()
                    .map(Key::Live)
                    .map_err(|e| Status::io_error(e.to_string()))
            })),
            Store::Frozen(file) => Box::new(file.iter_from(start).map(|(k, _)| Ok(Key::Frozen(k)))),
        }
    }

    /// Keys that start with `prefix`, in order.
    fn keys_with_prefix<'a>(&'a self, prefix: &'a [u8]) -> Keys<'a> {
        Box::new(
            self.keys_from(prefix)
                .take_while(move |k| k.as_ref().map_or(true, |k| k.starts_with(prefix))),
        )
    }

    // ── Write helpers ────────────────────────────────────────────────────

    fn put_kv(&self, key: &[u8], value: &[u8]) -> ZResult<()> {
        self.live_items()?
            .insert(key, value)
            .map_err(|e| Status::io_error(e.to_string()))
    }

    fn del_kv(&self, key: &[u8]) -> ZResult<()> {
        self.live_items()?
            .remove(key)
            .map_err(|e| Status::io_error(e.to_string()))
    }

    /// Iterate all keys starting with `prefix` and collect doc_ids into a bitmap.
    fn scan_prefix_doc_ids(&self, prefix: &[u8]) -> ZResult<RoaringTreemap> {
        let mut bitmap = RoaringTreemap::new();
        for item in self.keys_with_prefix(prefix) {
            let k = item?;
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
        for key in self.keys_for(doc_id, value)? {
            self.put_kv(&key, b"")?;
        }
        Ok(())
    }

    pub fn delete(&self, doc_id: u64, value: &Value) -> ZResult<()> {
        // `insert` writes nothing for a value it cannot encode, so there is nothing to remove.
        let Ok(keys) = self.keys_for(doc_id, value) else {
            return Ok(());
        };
        for key in keys {
            self.del_kv(&key)?;
        }
        Ok(())
    }

    /// Fails when `insert` would reject `value` itself; writes nothing.
    pub fn check_value(&self, value: &Value) -> ZResult<()> {
        self.keys_for(0, value).map(drop)
    }

    pub fn insert_null_marker(&self, doc_id: u64) -> ZResult<()> {
        let key = encode_marker_key(&self.field_name, NS_NULL, doc_id);
        self.put_kv(&key, b"")
    }

    pub fn delete_null_marker(&self, doc_id: u64) -> ZResult<()> {
        let key = encode_marker_key(&self.field_name, NS_NULL, doc_id);
        self.del_kv(&key)
    }

    pub fn insert_nonnull_marker(&self, doc_id: u64) -> ZResult<()> {
        let key = encode_marker_key(&self.field_name, NS_NONNULL, doc_id);
        self.put_kv(&key, b"")
    }

    pub fn delete_nonnull_marker(&self, doc_id: u64) -> ZResult<()> {
        let key = encode_marker_key(&self.field_name, NS_NONNULL, doc_id);
        self.del_kv(&key)
    }

    // Every key `insert` writes for `value`, built in full so a value that fails writes nothing.
    fn keys_for(&self, doc_id: u64, value: &Value) -> ZResult<Vec<Vec<u8>>> {
        let mut keys = Vec::new();
        if !is_array_type(self.data_type) {
            self.push_term_keys(doc_id, value, &mut keys)?;
            return Ok(keys);
        }
        let items = array_items(value).ok_or_else(|| {
            Status::invalid_argument("invert index: array field stored with non-array value")
        })?;
        let len = Value::U32(items.len() as u32);
        let len_key = encode_full_key(
            &self.field_name,
            NS_ARRAY_LEN,
            DataType::Uint32,
            &len,
            doc_id,
        )
        .ok_or_else(|| Status::invalid_argument("invert index: failed to encode array length"))?;
        keys.push(len_key);
        for item in &items {
            self.push_term_keys(doc_id, item, &mut keys)?;
        }
        Ok(keys)
    }

    fn push_term_keys(&self, doc_id: u64, value: &Value, keys: &mut Vec<Vec<u8>>) -> ZResult<()> {
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
        keys.push(key);

        // Optional reversed-term index for suffix search.
        let Value::String(s) = value else {
            return Ok(());
        };
        if !self.enable_extended_wildcard {
            return Ok(());
        }
        let rev_val = Value::String(s.chars().rev().collect());
        if let Some(rev_key) = encode_full_key(
            &self.field_name,
            NS_REVERSED_TERMS,
            self.index_value_type,
            &rev_val,
            doc_id,
        ) {
            keys.push(rev_key);
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

use super::*;
use arrow::datatypes::{
    ArrowPrimitiveType, BinaryType, ByteArrayType, Float32Type, Float64Type, Int32Type, Int64Type,
    Utf8Type,
};
use std::ops::Range;

/// Column buffers for the documents of the segment that takes writes.
pub struct MemoryForwardStore {
    collection_schema: CollectionSchema,
    doc_ids: Vec<u64>,
    pks: Vec<String>,
    /// Scalar field column data
    scalar_fields: HashMap<String, Vec<Option<Value>>>,
    /// Dense vector rows of each vector field, as 32-bit floats.
    vector_fields: HashMap<String, Vec<Option<Vec<f32>>>>,
    /// Binary vector field data stored as u32 words.
    binary_vector_fields_u32: HashMap<String, Vec<Option<Vec<u32>>>>,
    /// Binary vector field data stored as u64 words.
    binary_vector_fields_u64: HashMap<String, Vec<Option<Vec<u64>>>>,
    /// Sparse vector field data stored as serialized bytes (field_name → rows of bytes)
    sparse_vector_fields: HashMap<String, Vec<Option<Vec<u8>>>>,
    /// Running estimate of in-memory payload bytes.
    approx_bytes: usize,
}

impl MemoryForwardStore {
    /// An empty store with one column for each field of `schema`.
    pub fn new(schema: CollectionSchema) -> Self {
        let mut scalar_fields = HashMap::new();
        for f in schema.scalar_fields() {
            scalar_fields.insert(f.name.clone(), Vec::new());
        }

        let mut vector_fields = HashMap::new();
        let mut binary_vector_fields_u32 = HashMap::new();
        let mut binary_vector_fields_u64 = HashMap::new();
        let mut sparse_vector_fields = HashMap::new();
        for f in schema.vector_fields() {
            // Dense float32 vectors stored as raw f32 slices
            if is_persisted_dense_vector(f.data_type) {
                vector_fields.insert(f.name.clone(), Vec::new());
            }
            if f.data_type == DataType::VectorBinary32 {
                binary_vector_fields_u32.insert(f.name.clone(), Vec::new());
            }
            if f.data_type == DataType::VectorBinary64 {
                binary_vector_fields_u64.insert(f.name.clone(), Vec::new());
            }
            // Sparse vectors: persist serialized bytes so persisted segments can
            // build/query sparse indexes after flush/reopen.
            if matches!(f.data_type, DataType::SparseFp32 | DataType::SparseFp16) {
                sparse_vector_fields.insert(f.name.clone(), Vec::new());
            }
        }

        MemoryForwardStore {
            collection_schema: schema,
            doc_ids: Vec::new(),
            pks: Vec::new(),
            scalar_fields,
            vector_fields,
            binary_vector_fields_u32,
            binary_vector_fields_u64,
            sparse_vector_fields,
            approx_bytes: 0,
        }
    }

    /// Appends `doc` as a row with id `doc_id`; a missing field or empty sparse vector becomes null.
    pub fn insert(&mut self, doc_id: u64, doc: &Doc) -> ZResult<()> {
        self.doc_ids.push(doc_id);
        self.pks.push(doc.pk.clone());
        self.approx_bytes += std::mem::size_of::<u64>() + doc.pk.len();

        for (field_name, col_values) in self.scalar_fields.iter_mut() {
            let value = doc.fields.get(field_name).cloned();
            if let Some(v) = &value {
                self.approx_bytes += approx_value_size(v);
            }
            col_values.push(value);
        }

        for (field_name, col_values) in self.vector_fields.iter_mut() {
            let vec = doc.get_vec_f32(field_name).map(|s| s.to_vec());
            if let Some(v) = &vec {
                self.approx_bytes += v.len() * std::mem::size_of::<f32>();
            }
            col_values.push(vec);
        }

        for (field_name, col_values) in self.binary_vector_fields_u32.iter_mut() {
            let vec = match doc.fields.get(field_name) {
                None | Some(Value::Null) => None,
                Some(Value::VecU32(v)) => Some(v.clone()),
                _ => None,
            };
            if let Some(v) = &vec {
                self.approx_bytes += v.len() * std::mem::size_of::<u32>();
            }
            col_values.push(vec);
        }

        for (field_name, col_values) in self.binary_vector_fields_u64.iter_mut() {
            let vec = match doc.fields.get(field_name) {
                None | Some(Value::Null) => None,
                Some(Value::VecU64(v)) => Some(v.clone()),
                _ => None,
            };
            if let Some(v) = &vec {
                self.approx_bytes += v.len() * std::mem::size_of::<u64>();
            }
            col_values.push(vec);
        }

        for (field_name, col_values) in self.sparse_vector_fields.iter_mut() {
            // treat empty sparse vectors as "not set" (NULL) rather than
            // a present-but-empty vector.
            let bytes = doc
                .get_sparse_f32(field_name)
                .filter(|(indices, values)| !indices.is_empty() && !values.is_empty())
                .map(|(indices, values)| encode_sparse_f32(indices, values));
            if let Some(b) = &bytes {
                self.approx_bytes += b.len();
            }
            col_values.push(bytes);
        }

        Ok(())
    }

    /// Count of rows.
    pub fn len(&self) -> usize {
        self.doc_ids.len()
    }

    /// Whether the store has no rows.
    pub fn is_empty(&self) -> bool {
        self.doc_ids.is_empty()
    }

    /// Approximate in-memory bytes currently buffered by this forward store.
    pub fn approx_bytes(&self) -> usize {
        self.approx_bytes
    }

    /// Lookup docs by doc_id from in-memory data (includes vector fields)
    pub fn get_by_doc_ids(&self, ids: &[u64]) -> ZResult<Vec<Option<Doc>>> {
        let id_to_pos: HashMap<u64, usize> = self
            .doc_ids
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, id)| (id, i))
            .collect();

        Ok(ids
            .iter()
            .map(|&id| id_to_pos.get(&id).map(|&pos| self.doc_at(pos, id)))
            .collect())
    }

    fn doc_at(&self, pos: usize, doc_id: u64) -> Doc {
        let mut doc = Doc::new(self.pks[pos].clone());
        doc.doc_id = doc_id;
        for (field_name, values) in &self.scalar_fields {
            if let Some(Some(val)) = values.get(pos) {
                doc.fields.insert(field_name.clone(), val.clone());
            }
        }
        for (field_name, values) in &self.vector_fields {
            if let Some(Some(vec)) = values.get(pos) {
                doc.fields
                    .insert(field_name.clone(), Value::VecF32(vec.clone()));
            }
        }
        for (field_name, values) in &self.sparse_vector_fields {
            let Some(Some(bytes)) = values.get(pos) else {
                continue;
            };
            if let Ok((indices, values)) = decode_sparse_f32(bytes) {
                doc.fields
                    .insert(field_name.clone(), Value::SparseF32 { indices, values });
            }
        }
        doc
    }

    /// Flush to Arrow IPC file (stores both scalar and vector fields)
    pub fn dump_to_ipc(&self, path: &Path) -> ZResult<()> {
        const MAX_BATCH_ROWS: usize = 4096; // kMaxRecordBatchNumRows
        let schema = ipc_schema_for_collection(&self.collection_schema);
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let file_name = path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "forward.arrow".to_string());
        let tmp_path = parent.join(format!("{}.tmp", file_name));

        let file = File::create(&tmp_path).map_err(|e| Status::io_error(e.to_string()))?;
        let mut writer =
            FileWriter::try_new(file, &schema).map_err(|e| Status::io_error(e.to_string()))?;

        let n = self.doc_ids.len();
        let mut start = 0usize;
        while start < n {
            let end = (start + MAX_BATCH_ROWS).min(n);
            let batch = self.build_record_batch_range(schema.clone(), start, end)?;
            writer
                .write(&batch)
                .map_err(|e| Status::io_error(e.to_string()))?;
            start = end;
        }
        writer
            .finish()
            .map_err(|e| Status::io_error(e.to_string()))?;

        // Best-effort durability: sync file contents before rename, then sync dir.
        if let Ok(f) = File::open(&tmp_path) {
            let _ = f.sync_all();
        }

        // Replace any existing file (Windows-compatible).
        let _ = std::fs::remove_file(path);
        std::fs::rename(&tmp_path, path).map_err(|e| Status::io_error(e.to_string()))?;
        self.dump_i64_pk_sidecar(path)?;

        if let Ok(dir) = File::open(parent) {
            let _ = dir.sync_all();
        }

        Ok(())
    }

    /// Flush to Parquet file (stores both scalar and vector fields).
    pub fn dump_to_parquet(&self, path: &Path) -> ZResult<()> {
        const MAX_BATCH_ROWS: usize = 4096; // match IPC chunking
        let schema = ipc_schema_for_collection(&self.collection_schema);
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let file_name = path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "forward.parquet".to_string());
        let tmp_path = parent.join(format!("{}.tmp", file_name));

        let file = File::create(&tmp_path).map_err(|e| Status::io_error(e.to_string()))?;
        let props = WriterProperties::builder().build();
        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))
            .map_err(|e| Status::io_error(e.to_string()))?;

        let n = self.doc_ids.len();
        let mut start = 0usize;
        while start < n {
            let end = (start + MAX_BATCH_ROWS).min(n);
            let batch = self.build_record_batch_range(schema.clone(), start, end)?;
            writer
                .write(&batch)
                .map_err(|e| Status::io_error(e.to_string()))?;
            start = end;
        }
        writer
            .close()
            .map_err(|e| Status::io_error(e.to_string()))?;

        // Best-effort durability: sync file contents before rename, then sync dir.
        if let Ok(f) = File::open(&tmp_path) {
            let _ = f.sync_all();
        }

        // Replace any existing file (Windows-compatible).
        let _ = std::fs::remove_file(path);
        std::fs::rename(&tmp_path, path).map_err(|e| Status::io_error(e.to_string()))?;
        self.dump_i64_pk_sidecar(path)?;

        if let Ok(dir) = File::open(parent) {
            let _ = dir.sync_all();
        }

        Ok(())
    }

    fn dump_i64_pk_sidecar(&self, forward_path: &Path) -> ZResult<()> {
        let sidecar_path = pk_i64_sidecar_path(forward_path);
        let mut parsed_pks: Vec<i64> = Vec::with_capacity(self.pks.len());
        for pk in &self.pks {
            match pk.parse::<i64>() {
                Ok(v) => parsed_pks.push(v),
                Err(_) => {
                    let _ = std::fs::remove_file(&sidecar_path);
                    return Ok(());
                }
            }
        }

        let parent = forward_path.parent().unwrap_or_else(|| Path::new("."));
        let file_name = sidecar_path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "forward.pk_i64".to_string());
        let tmp_path = parent.join(format!("{}.tmp", file_name));
        let mut file = File::create(&tmp_path).map_err(|e| Status::io_error(e.to_string()))?;

        use std::io::Write;
        file.write_all(PK_I64_SIDECAR_MAGIC)
            .map_err(|e| Status::io_error(e.to_string()))?;
        file.write_all(&(self.doc_ids.len() as u64).to_le_bytes())
            .map_err(|e| Status::io_error(e.to_string()))?;
        for &doc_id in &self.doc_ids {
            file.write_all(&doc_id.to_le_bytes())
                .map_err(|e| Status::io_error(e.to_string()))?;
        }
        for pk in parsed_pks {
            file.write_all(&pk.to_le_bytes())
                .map_err(|e| Status::io_error(e.to_string()))?;
        }
        let _ = file.sync_all();

        let _ = std::fs::remove_file(&sidecar_path);
        std::fs::rename(&tmp_path, &sidecar_path).map_err(|e| Status::io_error(e.to_string()))?;
        Ok(())
    }

    fn build_record_batch_range(
        &self,
        schema: Arc<Schema>,
        start: usize,
        end: usize,
    ) -> ZResult<RecordBatch> {
        let mut columns: Vec<Arc<dyn Array>> = Vec::new();

        // doc_id column
        let doc_id_arr: UInt64Array = self.doc_ids[start..end].iter().cloned().collect();
        columns.push(Arc::new(doc_id_arr));

        // pk column
        let pk_arr: StringArray = self.pks[start..end]
            .iter()
            .map(|s| Some(s.as_str()))
            .collect();
        columns.push(Arc::new(pk_arr));

        for field in self.collection_schema.scalar_fields() {
            let rows = column_rows(&self.scalar_fields, &field.name);
            columns.push(scalar_column(field.data_type, rows, start..end));
        }

        // Vector field columns: stored as LargeBinary (dtype-dependent raw bytes).
        for field in self.collection_schema.vector_fields() {
            let column = match field.data_type {
                dt if is_persisted_dense_vector(dt) => {
                    raw_words_column(column_rows(&self.vector_fields, &field.name), start..end)
                }
                DataType::VectorBinary32 => raw_words_column(
                    column_rows(&self.binary_vector_fields_u32, &field.name),
                    start..end,
                ),
                DataType::VectorBinary64 => raw_words_column(
                    column_rows(&self.binary_vector_fields_u64, &field.name),
                    start..end,
                ),
                _ => continue,
            };
            columns.push(column);
        }

        // Sparse vector field columns: stored as LargeBinary (serialized sparse bytes)
        for field in self.collection_schema.vector_fields() {
            if !matches!(field.data_type, DataType::SparseFp32 | DataType::SparseFp16) {
                continue;
            }
            let rows = column_rows(&self.sparse_vector_fields, &field.name);
            columns.push(large_binary_column(rows, start..end));
        }

        RecordBatch::try_new(schema, columns).map_err(|e| Status::io_error(e.to_string()))
    }

    /// Lookup only primary keys by doc_id (cheap projection used by refiner/bench fast paths).
    pub fn get_pks_by_doc_ids(&self, ids: &[u64]) -> ZResult<Vec<Option<String>>> {
        let id_to_pos: HashMap<u64, usize> = self
            .doc_ids
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, id)| (id, i))
            .collect();

        Ok(ids
            .iter()
            .map(|&id| id_to_pos.get(&id).map(|&pos| self.pks[pos].clone()))
            .collect())
    }
}

fn column_rows<'a, T>(columns: &'a HashMap<String, Vec<Option<T>>>, name: &str) -> &'a [Option<T>] {
    columns.get(name).map(|v| v.as_slice()).unwrap_or(&[])
}

fn scalar_column(
    data_type: DataType,
    rows: &[Option<Value>],
    range: Range<usize>,
) -> Arc<dyn Array> {
    match data_type {
        DataType::Int8 => primitive_column::<Int8Array, _>(rows, range, |v| match v {
            Value::I8(x) => Some(*x),
            _ => None,
        }),
        DataType::Int16 => primitive_column::<Int16Array, _>(rows, range, |v| match v {
            Value::I16(x) => Some(*x),
            _ => None,
        }),
        DataType::Int32 => {
            primitive_column::<Int32Array, _>(rows, range, |v| v.as_i64().map(|x| x as i32))
        }
        DataType::Int64 => primitive_column::<Int64Array, _>(rows, range, Value::as_i64),
        DataType::Uint8 => primitive_column::<UInt8Array, _>(rows, range, |v| match v {
            Value::U8(x) => Some(*x),
            _ => None,
        }),
        DataType::Uint16 => primitive_column::<UInt16Array, _>(rows, range, |v| match v {
            Value::U16(x) => Some(*x),
            _ => None,
        }),
        DataType::Float32 => primitive_column::<Float32Array, _>(rows, range, Value::as_f32),
        DataType::Float64 => primitive_column::<Float64Array, _>(rows, range, Value::as_f64),
        DataType::Float16 => primitive_column::<Float16Array, _>(rows, range, |v| match v {
            Value::F16(x) => Some(*x),
            _ => None,
        }),
        DataType::String => primitive_column::<StringArray, _>(rows, range, Value::as_str),
        DataType::Bool => primitive_column::<BooleanArray, _>(rows, range, |v| match v {
            Value::Bool(b) => Some(*b),
            _ => None,
        }),
        DataType::Uint32 => {
            primitive_column::<UInt32Array, _>(rows, range, |v| v.as_i64().map(|x| x as u32))
        }
        DataType::Uint64 => {
            primitive_column::<UInt64Array, _>(rows, range, |v| v.as_i64().map(|x| x as u64))
        }
        DataType::Bytes | DataType::Binary => bytes_column(rows, range),
        DataType::ArrayString
        | DataType::ArrayBinary
        | DataType::ArrayBool
        | DataType::ArrayInt32
        | DataType::ArrayInt64
        | DataType::ArrayUint32
        | DataType::ArrayUint64
        | DataType::ArrayFp32
        | DataType::ArrayFp64 => list_column(data_type, rows, range),
        _ => null_column(data_type, range),
    }
}

fn list_column(data_type: DataType, rows: &[Option<Value>], range: Range<usize>) -> Arc<dyn Array> {
    match data_type {
        DataType::ArrayString => byte_list_column::<Utf8Type, _>(rows, range, |v| match v {
            Value::ArrayString(items) => Some(items.as_slice()),
            _ => None,
        }),
        DataType::ArrayBinary => byte_list_column::<BinaryType, _>(rows, range, |v| match v {
            Value::ArrayBinary(items) => Some(items.as_slice()),
            _ => None,
        }),
        DataType::ArrayBool => bool_list_column(rows, range),
        DataType::ArrayInt32 => primitive_list_column::<Int32Type>(rows, range, |v| match v {
            Value::ArrayI32(items) => Some(items.as_slice()),
            _ => None,
        }),
        DataType::ArrayInt64 => primitive_list_column::<Int64Type>(rows, range, |v| match v {
            Value::ArrayI64(items) => Some(items.as_slice()),
            _ => None,
        }),
        DataType::ArrayUint32 => u32_list_column(rows, range),
        DataType::ArrayUint64 => u64_list_column(rows, range),
        DataType::ArrayFp32 => primitive_list_column::<Float32Type>(rows, range, |v| match v {
            Value::ArrayF32(items) => Some(items.as_slice()),
            _ => None,
        }),
        DataType::ArrayFp64 => primitive_list_column::<Float64Type>(rows, range, |v| match v {
            Value::ArrayF64(items) => Some(items.as_slice()),
            _ => None,
        }),
        _ => null_column(data_type, range),
    }
}

fn null_column(data_type: DataType, range: Range<usize>) -> Arc<dyn Array> {
    let dt = data_type_to_arrow(&data_type).unwrap_or(ArrowType::Null);
    arrow::array::new_null_array(&dt, range.len())
}

fn primitive_column<'a, A, T>(
    rows: &'a [Option<Value>],
    range: Range<usize>,
    get: impl Fn(&'a Value) -> Option<T>,
) -> Arc<dyn Array>
where
    A: FromIterator<Option<T>> + Array + 'static,
{
    let arr: A = range
        .map(|i| rows.get(i).and_then(|v| get(v.as_ref()?)))
        .collect();
    Arc::new(arr)
}

fn bytes_column(rows: &[Option<Value>], range: Range<usize>) -> Arc<dyn Array> {
    let mut b = BinaryBuilder::new();
    for i in range {
        match rows.get(i).and_then(|v| v.as_ref()) {
            Some(Value::Bytes(bytes)) => b.append_value(bytes.as_slice()),
            _ => b.append_null(),
        }
    }
    Arc::new(b.finish())
}

fn primitive_list_column<'a, P: ArrowPrimitiveType>(
    rows: &'a [Option<Value>],
    range: Range<usize>,
    items_of: impl Fn(&'a Value) -> Option<&'a [P::Native]>,
) -> Arc<dyn Array> {
    let mut b = ListBuilder::new(PrimitiveBuilder::<P>::new());
    for i in range {
        match rows.get(i).and_then(|v| items_of(v.as_ref()?)) {
            Some(items) => {
                for &x in items {
                    b.values().append_value(x);
                }
                b.append(true);
            }
            None => b.append(false),
        }
    }
    Arc::new(b.finish())
}

fn byte_list_column<'a, T: ByteArrayType, V: AsRef<T::Native> + 'a>(
    rows: &'a [Option<Value>],
    range: Range<usize>,
    items_of: impl Fn(&'a Value) -> Option<&'a [V]>,
) -> Arc<dyn Array> {
    let mut b = ListBuilder::new(GenericByteBuilder::<T>::new());
    for i in range {
        match rows.get(i).and_then(|v| items_of(v.as_ref()?)) {
            Some(items) => {
                for item in items {
                    b.values().append_value(item);
                }
                b.append(true);
            }
            None => b.append(false),
        }
    }
    Arc::new(b.finish())
}

fn bool_list_column(rows: &[Option<Value>], range: Range<usize>) -> Arc<dyn Array> {
    let mut b = ListBuilder::new(BooleanBuilder::new());
    for i in range {
        match rows.get(i).and_then(|v| v.as_ref()) {
            Some(Value::ArrayBool(items)) => {
                for &x in items {
                    b.values().append_value(x);
                }
                b.append(true);
            }
            _ => b.append(false),
        }
    }
    Arc::new(b.finish())
}

// Signed arrays are kept only when every item fits in u32; otherwise the row is null.
fn u32_list_column(rows: &[Option<Value>], range: Range<usize>) -> Arc<dyn Array> {
    let mut b = ListBuilder::new(UInt32Builder::new());
    for i in range {
        match rows.get(i).and_then(|v| v.as_ref()) {
            Some(Value::ArrayU32(items)) => {
                for &x in items {
                    b.values().append_value(x);
                }
                b.append(true);
            }
            Some(Value::ArrayI32(items)) if items.iter().all(|&x| x >= 0) => {
                for &x in items {
                    b.values().append_value(x as u32);
                }
                b.append(true);
            }
            Some(Value::ArrayI64(items))
                if items.iter().all(|&x| x >= 0 && x <= u32::MAX as i64) =>
            {
                for &x in items {
                    b.values().append_value(x as u32);
                }
                b.append(true);
            }
            _ => b.append(false),
        }
    }
    Arc::new(b.finish())
}

// Signed arrays are kept only when every item is non-negative; otherwise the row is null.
fn u64_list_column(rows: &[Option<Value>], range: Range<usize>) -> Arc<dyn Array> {
    let mut b = ListBuilder::new(UInt64Builder::new());
    for i in range {
        match rows.get(i).and_then(|v| v.as_ref()) {
            Some(Value::ArrayU64(items)) => {
                for &x in items {
                    b.values().append_value(x);
                }
                b.append(true);
            }
            Some(Value::ArrayI32(items)) if items.iter().all(|&x| x >= 0) => {
                for &x in items {
                    b.values().append_value(x as u64);
                }
                b.append(true);
            }
            Some(Value::ArrayI64(items)) if items.iter().all(|&x| x >= 0) => {
                for &x in items {
                    b.values().append_value(x as u64);
                }
                b.append(true);
            }
            _ => b.append(false),
        }
    }
    Arc::new(b.finish())
}

/// Vector element types without padding, so every byte of their storage is initialized.
trait VectorWord: Copy {}
impl VectorWord for f32 {}
impl VectorWord for u32 {}
impl VectorWord for u64 {}

fn raw_words_column<T: VectorWord>(rows: &[Option<Vec<T>>], range: Range<usize>) -> Arc<dyn Array> {
    let mut builder = LargeBinaryBuilder::new();
    for i in range {
        let Some(vec) = rows.get(i).and_then(|v| v.as_ref()) else {
            builder.append_null();
            continue;
        };
        // SAFETY: the view covers exactly `vec`'s padding-free words; u8 has no alignment or validity rules.
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                vec.as_ptr() as *const u8,
                vec.len() * std::mem::size_of::<T>(),
            )
        };
        builder.append_value(bytes);
    }
    Arc::new(builder.finish())
}

fn large_binary_column(rows: &[Option<Vec<u8>>], range: Range<usize>) -> Arc<dyn Array> {
    let mut builder = LargeBinaryBuilder::new();
    for i in range {
        match rows.get(i).and_then(|v| v.as_ref()) {
            Some(bytes) => builder.append_value(bytes.as_slice()),
            None => builder.append_null(),
        }
    }
    Arc::new(builder.finish())
}

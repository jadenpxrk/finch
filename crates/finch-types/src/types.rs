use serde::{Deserialize, Serialize};

/// Data type for document fields and vectors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum DataType {
    /// No type; a schema field must not use it.
    Undefined = 0,
    // Scalar ids
    /// Raw bytes.
    Binary = 1,
    /// UTF-8 text.
    String = 2,
    /// True or false.
    Bool = 3,
    /// Signed 32-bit integer.
    Int32 = 4,
    /// Signed 64-bit integer.
    Int64 = 5,
    /// Unsigned 32-bit integer.
    Uint32 = 6,
    /// Unsigned 64-bit integer.
    Uint64 = 7,
    /// 32-bit float.
    Float32 = 8,
    /// 64-bit float.
    Float64 = 9,
    // Narrow scalar ids
    /// Signed 8-bit integer.
    Int8 = 100,
    /// Signed 16-bit integer.
    Int16 = 101,
    /// Unsigned 8-bit integer.
    Uint8 = 102,
    /// Unsigned 16-bit integer.
    Uint16 = 103,
    /// 16-bit float.
    Float16 = 104,
    /// Raw bytes, stored and named as `Binary`.
    Bytes = 105,
    // Dense vector ids
    /// Dense bit vector in 32-bit words.
    VectorBinary32 = 20,
    /// Dense bit vector in 64-bit words.
    VectorBinary64 = 21,
    /// Dense vector of 16-bit floats.
    VectorFp16 = 22,
    /// Dense vector of 32-bit floats.
    VectorFp32 = 23,
    /// Dense vector of 64-bit floats, stored as 32-bit floats.
    VectorFp64 = 24,
    /// Dense vector of 4-bit integers.
    VectorInt4 = 25,
    /// Dense vector of 8-bit integers.
    VectorInt8 = 26,
    /// Dense vector of 16-bit integers.
    VectorInt16 = 27,
    // Wide integer and bool dense vector ids
    /// Dense vector of booleans.
    VectorBool = 120,
    /// Dense vector of signed 32-bit integers.
    VectorInt32 = 121,
    /// Dense vector of signed 64-bit integers.
    VectorInt64 = 122,
    /// Dense vector of unsigned 32-bit integers.
    VectorUint32 = 123,
    /// Dense vector of unsigned 64-bit integers.
    VectorUint64 = 124,
    // Sparse vector ids
    /// Sparse vector: indices with 16-bit float values.
    SparseFp16 = 30,
    /// Sparse vector: indices with 32-bit float values.
    SparseFp32 = 31,
    // Array ids
    /// List of byte strings.
    ArrayBinary = 40,
    /// List of strings.
    ArrayString = 41,
    /// List of booleans.
    ArrayBool = 42,
    /// List of signed 32-bit integers.
    ArrayInt32 = 43,
    /// List of signed 64-bit integers.
    ArrayInt64 = 44,
    /// List of unsigned 32-bit integers.
    ArrayUint32 = 45,
    /// List of unsigned 64-bit integers.
    ArrayUint64 = 46,
    /// List of 32-bit floats.
    ArrayFp32 = 47,
    /// List of 64-bit floats.
    ArrayFp64 = 48,
}

impl DataType {
    /// Whether the type is a dense or sparse vector.
    pub fn is_vector(&self) -> bool {
        matches!(
            self,
            DataType::VectorBinary32
                | DataType::VectorBinary64
                | DataType::VectorInt4
                | DataType::VectorBool
                | DataType::VectorInt8
                | DataType::VectorInt16
                | DataType::VectorInt32
                | DataType::VectorInt64
                | DataType::VectorUint32
                | DataType::VectorUint64
                | DataType::VectorFp16
                | DataType::VectorFp32
                | DataType::VectorFp64
                | DataType::SparseFp16
                | DataType::SparseFp32
        )
    }

    /// Whether the type is a list of scalars.
    pub fn is_array(&self) -> bool {
        matches!(
            self,
            DataType::ArrayBinary
                | DataType::ArrayString
                | DataType::ArrayBool
                | DataType::ArrayInt32
                | DataType::ArrayInt64
                | DataType::ArrayUint32
                | DataType::ArrayUint64
                | DataType::ArrayFp32
                | DataType::ArrayFp64
        )
    }

    /// Whether the type is a sparse vector.
    pub fn is_sparse(&self) -> bool {
        matches!(self, DataType::SparseFp16 | DataType::SparseFp32)
    }

    /// Whether the type is a defined type that is not a vector.
    pub fn is_scalar(&self) -> bool {
        !self.is_vector() && *self != DataType::Undefined
    }

    /// The type's stable uppercase name, such as `VECTOR_FP32`, for messages and schemas.
    pub fn as_codebook_str(&self) -> &'static str {
        match self {
            DataType::Undefined => "UNDEFINED",

            DataType::Binary | DataType::Bytes => "BINARY",
            DataType::String => "STRING",
            DataType::Bool => "BOOL",
            DataType::Int32 => "INT32",
            DataType::Int64 => "INT64",
            DataType::Uint32 => "UINT32",
            DataType::Uint64 => "UINT64",
            DataType::Float32 => "FLOAT",
            DataType::Float64 => "DOUBLE",

            DataType::VectorBinary32 => "VECTOR_BINARY32",
            DataType::VectorBinary64 => "VECTOR_BINARY64",
            DataType::VectorFp16 => "VECTOR_FP16",
            DataType::VectorFp32 => "VECTOR_FP32",
            DataType::VectorFp64 => "VECTOR_FP64",
            DataType::VectorInt4 => "VECTOR_INT4",
            DataType::VectorInt8 => "VECTOR_INT8",
            DataType::VectorInt16 => "VECTOR_INT16",

            DataType::SparseFp16 => "SPARSE_VECTOR_FP16",
            DataType::SparseFp32 => "SPARSE_VECTOR_FP32",

            DataType::ArrayBinary => "ARRAY_BINARY",
            DataType::ArrayString => "ARRAY_STRING",
            DataType::ArrayBool => "ARRAY_BOOL",
            DataType::ArrayInt32 => "ARRAY_INT32",
            DataType::ArrayInt64 => "ARRAY_INT64",
            DataType::ArrayUint32 => "ARRAY_UINT32",
            DataType::ArrayUint64 => "ARRAY_UINT64",
            DataType::ArrayFp32 => "ARRAY_FLOAT",
            DataType::ArrayFp64 => "ARRAY_DOUBLE",

            DataType::Int8 => "INT8",
            DataType::Int16 => "INT16",
            DataType::Uint8 => "UINT8",
            DataType::Uint16 => "UINT16",
            DataType::Float16 => "FLOAT16",
            DataType::VectorBool => "VECTOR_BOOL",
            DataType::VectorInt32 => "VECTOR_INT32",
            DataType::VectorInt64 => "VECTOR_INT64",
            DataType::VectorUint32 => "VECTOR_UINT32",
            DataType::VectorUint64 => "VECTOR_UINT64",
        }
    }
}

/// Distance metric type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum MetricType {
    /// No metric; a vector index must not use it.
    Undefined = 0,
    /// Squared Euclidean distance.
    L2 = 1,
    /// Negative inner product, so a smaller distance means a closer match.
    InnerProduct = 2,
    /// One minus cosine similarity.
    Cosine = 3,
    /// Inner-product search through a squared L2 distance on vectors scaled into the unit sphere.
    MipsL2 = 4,
    /// Count of different bits.
    Hamming = 5,
}

/// Quantization type for compressed storage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[repr(u32)]
pub enum QuantizeType {
    /// Full precision: no quantization.
    #[default]
    Undefined = 0,
    /// Each component as a 16-bit float.
    Fp16 = 1,
    /// Each component as an 8-bit integer.
    Int8 = 2,
    /// Each component as a 4-bit integer.
    Int4 = 3,
}

/// Document operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum Operator {
    /// Write a new document; fail if its primary key exists.
    Insert = 0,
    /// Write a document whole, replacing any document with its primary key.
    Upsert = 1,
    /// Merge the given fields into the existing document.
    Update = 2,
    /// Remove the document.
    Delete = 3,
}

/// Comparison operator for filters
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompareOp {
    /// `=`
    Equal,
    /// `!=`
    NotEqual,
    /// `<`
    LessThan,
    /// `<=`
    LessEqual,
    /// `>`
    GreaterThan,
    /// `>=`
    GreaterEqual,
}

/// File format for data storage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FileFormat {
    /// Unencoded bytes.
    Raw,
    /// Arrow IPC file.
    ArrowIpc,
    /// Parquet file.
    Parquet,
    /// Protocol Buffers message.
    Protobuf,
}

/// Storage mode for persisted files (forward store / vector indexes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageType {
    /// The mode that `CollectionOptions::enable_mmap` selects.
    None,
    /// Memory-map the files.
    Mmap,
    /// Read the files into memory.
    Memory,
    /// A buffer pool; vector indexes reject it.
    BufferPool,
}

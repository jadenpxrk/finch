use serde::{Deserialize, Serialize};

/// Data type for document fields and vectors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum DataType {
    Undefined = 0,
    // Canonical scalar ids
    Binary = 1,
    String = 2,
    Bool = 3,
    Int32 = 4,
    Int64 = 5,
    Uint32 = 6,
    Uint64 = 7,
    Float32 = 8,
    Float64 = 9,
    // Extended finch-only scalar ids (kept for internal compatibility)
    Int8 = 100,
    Int16 = 101,
    Uint8 = 102,
    Uint16 = 103,
    Float16 = 104,
    Bytes = 105,
    // Canonical dense vector ids
    VectorBinary32 = 20,
    VectorBinary64 = 21,
    VectorFp16 = 22,
    VectorFp32 = 23,
    VectorFp64 = 24,
    VectorInt4 = 25,
    VectorInt8 = 26,
    VectorInt16 = 27,
    // Extended finch-only dense vector ids
    VectorBool = 120,
    VectorInt32 = 121,
    VectorInt64 = 122,
    VectorUint32 = 123,
    VectorUint64 = 124,
    // Canonical sparse vector ids
    SparseFp16 = 30,
    SparseFp32 = 31,
    // Canonical array ids
    ArrayBinary = 40,
    ArrayString = 41,
    ArrayBool = 42,
    ArrayInt32 = 43,
    ArrayInt64 = 44,
    ArrayUint32 = 45,
    ArrayUint64 = 46,
    ArrayFp32 = 47,
    ArrayFp64 = 48,
}

impl DataType {
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

    pub fn is_sparse(&self) -> bool {
        matches!(self, DataType::SparseFp16 | DataType::SparseFp32)
    }

    pub fn is_scalar(&self) -> bool {
        !self.is_vector() && *self != DataType::Undefined
    }

    /// Render data types using a stable, uppercase "codebook" naming.
    ///
    /// This is intentionally *not* `Display` to avoid changing incidental formatting
    /// across the codebase; call sites that need stable type names should opt in.
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

            // Finch-only extensions: best-effort stable names.
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

/// Vector index algorithm type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum IndexType {
    Undefined = 0,
    Hnsw = 1,
    Ivf = 2,
    Flat = 3,
    Invert = 10,
    HnswSparse = 11,
    FlatSparse = 12,
}

/// Distance metric type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum MetricType {
    Undefined = 0,
    L2 = 1,
    InnerProduct = 2,
    Cosine = 3,
    MipsL2 = 4,
    Hamming = 5,
}

/// Quantization type for compressed storage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[repr(u32)]
pub enum QuantizeType {
    #[default]
    Undefined = 0,
    Fp16 = 1,
    Int8 = 2,
    Int4 = 3,
}

/// Block storage type in a segment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum BlockType {
    Undefined = 0,
    Scalar = 1,
    ScalarIndex = 2,
    VectorIndex = 3,
    VectorIndexQuantize = 4,
}

/// Document operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum Operator {
    Insert = 0,
    Upsert = 1,
    Update = 2,
    Delete = 3,
}

/// Comparison operator for filters
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompareOp {
    Equal,
    NotEqual,
    LessThan,
    LessEqual,
    GreaterThan,
    GreaterEqual,
}

/// Logical relation operator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelationOp {
    And,
    Or,
    Not,
}

/// File format for data storage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FileFormat {
    Raw,
    ArrowIpc,
    Parquet,
    Protobuf,
}

/// Storage mode for persisted files (forward store / vector indexes).
///
/// Mirrors Zvec's `StorageOptions::StorageType` at a high level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageType {
    None,
    Mmap,
    Memory,
    BufferPool,
}

/// Column operation type for DDL
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ColumnOp {
    Add,
    Drop,
    Alter,
}

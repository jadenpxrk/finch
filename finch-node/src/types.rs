use napi_derive::napi;

#[napi]
pub enum MetricType {
    Undefined = 0,
    L2 = 1,
    InnerProduct = 2,
    Cosine = 3,
    MipsL2 = 4,
    Hamming = 5,
}

impl From<MetricType> for finch_types::MetricType {
    fn from(m: MetricType) -> Self {
        match m {
            MetricType::Undefined => finch_types::MetricType::Undefined,
            MetricType::L2 => finch_types::MetricType::L2,
            MetricType::InnerProduct => finch_types::MetricType::InnerProduct,
            MetricType::Cosine => finch_types::MetricType::Cosine,
            MetricType::MipsL2 => finch_types::MetricType::MipsL2,
            MetricType::Hamming => finch_types::MetricType::Hamming,
        }
    }
}

#[napi]
pub enum DataType {
    Undefined = 0,
    Binary = 1,
    String = 2,
    Bool = 3,
    Int32 = 4,
    Int64 = 5,
    Uint32 = 6,
    Uint64 = 7,
    Float32 = 8,
    Float64 = 9,
    Int8 = 100,
    Int16 = 101,
    Uint8 = 102,
    Uint16 = 103,
    Float16 = 104,
    Bytes = 105,

    VectorBinary32 = 20,
    VectorBinary64 = 21,
    VectorFp16 = 22,
    VectorFp32 = 23,
    VectorFp64 = 24,
    VectorInt4 = 25,
    VectorInt8 = 26,
    VectorInt16 = 27,
    VectorBool = 120,
    VectorInt32 = 121,
    VectorInt64 = 122,
    VectorUint32 = 123,
    VectorUint64 = 124,

    SparseFp16 = 30,
    SparseFp32 = 31,

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

impl From<DataType> for finch_types::DataType {
    fn from(d: DataType) -> Self {
        match d {
            DataType::Undefined => finch_types::DataType::Undefined,
            DataType::Binary => finch_types::DataType::Binary,
            DataType::String => finch_types::DataType::String,
            DataType::Bool => finch_types::DataType::Bool,
            DataType::Int32 => finch_types::DataType::Int32,
            DataType::Int64 => finch_types::DataType::Int64,
            DataType::Uint32 => finch_types::DataType::Uint32,
            DataType::Uint64 => finch_types::DataType::Uint64,
            DataType::Float32 => finch_types::DataType::Float32,
            DataType::Float64 => finch_types::DataType::Float64,
            DataType::Int8 => finch_types::DataType::Int8,
            DataType::Int16 => finch_types::DataType::Int16,
            DataType::Uint8 => finch_types::DataType::Uint8,
            DataType::Uint16 => finch_types::DataType::Uint16,
            DataType::Float16 => finch_types::DataType::Float16,
            DataType::Bytes => finch_types::DataType::Bytes,
            DataType::VectorBinary32 => finch_types::DataType::VectorBinary32,
            DataType::VectorBinary64 => finch_types::DataType::VectorBinary64,
            DataType::VectorFp16 => finch_types::DataType::VectorFp16,
            DataType::VectorFp32 => finch_types::DataType::VectorFp32,
            DataType::VectorFp64 => finch_types::DataType::VectorFp64,
            DataType::VectorInt4 => finch_types::DataType::VectorInt4,
            DataType::VectorInt8 => finch_types::DataType::VectorInt8,
            DataType::VectorInt16 => finch_types::DataType::VectorInt16,
            DataType::VectorBool => finch_types::DataType::VectorBool,
            DataType::VectorInt32 => finch_types::DataType::VectorInt32,
            DataType::VectorInt64 => finch_types::DataType::VectorInt64,
            DataType::VectorUint32 => finch_types::DataType::VectorUint32,
            DataType::VectorUint64 => finch_types::DataType::VectorUint64,
            DataType::SparseFp16 => finch_types::DataType::SparseFp16,
            DataType::SparseFp32 => finch_types::DataType::SparseFp32,
            DataType::ArrayBinary => finch_types::DataType::ArrayBinary,
            DataType::ArrayString => finch_types::DataType::ArrayString,
            DataType::ArrayBool => finch_types::DataType::ArrayBool,
            DataType::ArrayInt32 => finch_types::DataType::ArrayInt32,
            DataType::ArrayInt64 => finch_types::DataType::ArrayInt64,
            DataType::ArrayUint32 => finch_types::DataType::ArrayUint32,
            DataType::ArrayUint64 => finch_types::DataType::ArrayUint64,
            DataType::ArrayFp32 => finch_types::DataType::ArrayFp32,
            DataType::ArrayFp64 => finch_types::DataType::ArrayFp64,
        }
    }
}

#[napi]
pub enum QuantizeType {
    Undefined = 0,
    Fp16 = 1,
    Int8 = 2,
    Int4 = 3,
}

impl From<QuantizeType> for finch_types::QuantizeType {
    fn from(q: QuantizeType) -> Self {
        match q {
            QuantizeType::Undefined => finch_types::QuantizeType::Undefined,
            QuantizeType::Fp16 => finch_types::QuantizeType::Fp16,
            QuantizeType::Int8 => finch_types::QuantizeType::Int8,
            QuantizeType::Int4 => finch_types::QuantizeType::Int4,
        }
    }
}

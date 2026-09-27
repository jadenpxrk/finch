//! Conversion between CollectionSchema and Arrow Schema

use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
use finch_types::{CollectionSchema, DataType};
use std::sync::Arc;

/// Convert a CollectionSchema (scalar fields only) to an Arrow Schema
pub fn collection_schema_to_arrow(schema: &CollectionSchema) -> Schema {
    let mut fields = Vec::new();

    // Always include a doc_id column
    fields.push(Field::new("__doc_id__", ArrowDataType::UInt64, false));
    fields.push(Field::new("__pk__", ArrowDataType::Utf8, false));

    for field in schema.scalar_fields() {
        if let Some(arrow_type) = data_type_to_arrow(&field.data_type) {
            fields.push(Field::new(&field.name, arrow_type, field.nullable));
        }
        // Vector fields are stored separately in index files, not in Arrow
    }

    Schema::new(fields)
}

/// Map finch DataType to Arrow DataType (for scalar types only)
pub fn data_type_to_arrow(dt: &DataType) -> Option<ArrowDataType> {
    match dt {
        DataType::Bool => Some(ArrowDataType::Boolean),
        DataType::Int8 => Some(ArrowDataType::Int8),
        DataType::Int16 => Some(ArrowDataType::Int16),
        DataType::Int32 => Some(ArrowDataType::Int32),
        DataType::Int64 => Some(ArrowDataType::Int64),
        DataType::Uint8 => Some(ArrowDataType::UInt8),
        DataType::Uint16 => Some(ArrowDataType::UInt16),
        DataType::Uint32 => Some(ArrowDataType::UInt32),
        DataType::Uint64 => Some(ArrowDataType::UInt64),
        DataType::Float16 => Some(ArrowDataType::Float16),
        DataType::Float32 => Some(ArrowDataType::Float32),
        DataType::Float64 => Some(ArrowDataType::Float64),
        DataType::String => Some(ArrowDataType::Utf8),
        DataType::Bytes | DataType::Binary => Some(ArrowDataType::Binary),
        DataType::ArrayBinary => Some(ArrowDataType::List(Arc::new(Field::new(
            "item",
            ArrowDataType::Binary,
            true,
        )))),
        DataType::ArrayString => Some(ArrowDataType::List(Arc::new(Field::new(
            "item",
            ArrowDataType::Utf8,
            true,
        )))),
        DataType::ArrayBool => Some(ArrowDataType::List(Arc::new(Field::new(
            "item",
            ArrowDataType::Boolean,
            true,
        )))),
        DataType::ArrayInt32 => Some(ArrowDataType::List(Arc::new(Field::new(
            "item",
            ArrowDataType::Int32,
            true,
        )))),
        DataType::ArrayInt64 => Some(ArrowDataType::List(Arc::new(Field::new(
            "item",
            ArrowDataType::Int64,
            true,
        )))),
        DataType::ArrayUint32 => Some(ArrowDataType::List(Arc::new(Field::new(
            "item",
            ArrowDataType::UInt32,
            true,
        )))),
        DataType::ArrayUint64 => Some(ArrowDataType::List(Arc::new(Field::new(
            "item",
            ArrowDataType::UInt64,
            true,
        )))),
        DataType::ArrayFp32 => Some(ArrowDataType::List(Arc::new(Field::new(
            "item",
            ArrowDataType::Float32,
            true,
        )))),
        DataType::ArrayFp64 => Some(ArrowDataType::List(Arc::new(Field::new(
            "item",
            ArrowDataType::Float64,
            true,
        )))),
        // Vector fields → not stored in Arrow forward store
        _ => None,
    }
}

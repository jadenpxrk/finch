use std::path::Path;
use std::sync::Arc;

use arrow::array::{new_null_array, Array, PrimitiveArray, PrimitiveBuilder};
use arrow::datatypes::{
    ArrowPrimitiveType, DataType, Field, Float32Type, Float64Type, Int32Type, Int64Type,
    UInt32Type, UInt64Type,
};
use arrow::record_batch::RecordBatch;
use finch_types::{CollectionSchema, FieldSchema, Status, ZResult};

use super::{ColumnRewrite, ForwardRewrite};

fn value<T: ArrowPrimitiveType>(col: &Arc<dyn Array>, row: usize) -> ZResult<T::Native> {
    col.as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .map(|a| a.value(row))
        .ok_or_else(|| Status::internal("alter_column array does not match its data type"))
}

fn unsupported_cast() -> Status {
    Status::invalid_argument("alter_column only supports numeric scalar casts")
}

fn finite_float(v: f64) -> ZResult<f64> {
    if !v.is_finite() {
        return Err(Status::invalid_argument(
            "alter_column cast from non-finite float",
        ));
    }
    Ok(v)
}

fn whole_float(v: f64) -> ZResult<f64> {
    let v = finite_float(v)?;
    if v.fract() != 0.0 {
        return Err(Status::invalid_argument(format!(
            "alter_column cast of fractional value {v} to integer"
        )));
    }
    Ok(v)
}

fn non_negative<T: Default + PartialOrd>(v: T) -> ZResult<T> {
    if v < T::default() {
        return Err(Status::invalid_argument(
            "alter_column cast produced negative value for unsigned",
        ));
    }
    Ok(v)
}

fn get_i128(col: &Arc<dyn Array>, row: usize) -> ZResult<Option<i128>> {
    if col.is_null(row) {
        return Ok(None);
    }
    match col.data_type() {
        DataType::Int32 => Ok(Some(value::<Int32Type>(col, row)? as i128)),
        DataType::Int64 => Ok(Some(value::<Int64Type>(col, row)? as i128)),
        DataType::UInt32 => Ok(Some(value::<UInt32Type>(col, row)? as i128)),
        DataType::UInt64 => {
            let v = value::<UInt64Type>(col, row)?;
            if v > i128::MAX as u64 {
                return Err(Status::invalid_argument(
                    "alter_column cast out of int range",
                ));
            }
            Ok(Some(v as i128))
        }
        DataType::Float32 => {
            let v = whole_float(value::<Float32Type>(col, row)? as f64)?;
            Ok(Some(v as i128))
        }
        DataType::Float64 => {
            let v = whole_float(value::<Float64Type>(col, row)?)?;
            Ok(Some(v as i128))
        }
        _ => Err(unsupported_cast()),
    }
}

fn get_u128(col: &Arc<dyn Array>, row: usize) -> ZResult<Option<u128>> {
    if col.is_null(row) {
        return Ok(None);
    }
    match col.data_type() {
        DataType::Int32 => Ok(Some(non_negative(value::<Int32Type>(col, row)?)? as u128)),
        DataType::Int64 => Ok(Some(non_negative(value::<Int64Type>(col, row)?)? as u128)),
        DataType::UInt32 => Ok(Some(value::<UInt32Type>(col, row)? as u128)),
        DataType::UInt64 => Ok(Some(value::<UInt64Type>(col, row)? as u128)),
        DataType::Float32 => {
            let v = whole_float(value::<Float32Type>(col, row)? as f64)?;
            Ok(Some(non_negative(v)? as u128))
        }
        DataType::Float64 => {
            let v = whole_float(value::<Float64Type>(col, row)?)?;
            Ok(Some(non_negative(v)? as u128))
        }
        _ => Err(unsupported_cast()),
    }
}

fn get_f64(col: &Arc<dyn Array>, row: usize) -> ZResult<Option<f64>> {
    if col.is_null(row) {
        return Ok(None);
    }
    let v = match col.data_type() {
        DataType::Int32 => value::<Int32Type>(col, row)? as f64,
        DataType::Int64 => value::<Int64Type>(col, row)? as f64,
        DataType::UInt32 => value::<UInt32Type>(col, row)? as f64,
        DataType::UInt64 => value::<UInt64Type>(col, row)? as f64,
        DataType::Float32 => value::<Float32Type>(col, row)? as f64,
        DataType::Float64 => value::<Float64Type>(col, row)?,
        _ => return Err(unsupported_cast()),
    };
    finite_float(v).map(Some)
}

fn out_of_range(type_name: &str) -> Status {
    Status::invalid_argument(format!("alter_column cast out of {} range", type_name))
}

// Reads each row through `get` and appends it through `convert`, rejecting nulls if required.
fn build_cast_column<P: ArrowPrimitiveType, V>(
    rows: usize,
    require_nonnull: bool,
    get: impl Fn(usize) -> ZResult<Option<V>>,
    convert: impl Fn(V) -> ZResult<P::Native>,
) -> ZResult<Arc<dyn Array>> {
    let mut b = PrimitiveBuilder::<P>::new();
    for row in 0..rows {
        match get(row)? {
            None if require_nonnull => {
                return Err(Status::invalid_argument(
                    "alter_column produced null for non-nullable column",
                ));
            }
            None => b.append_null(),
            Some(v) => b.append_value(convert(v)?),
        }
    }
    Ok(Arc::new(b.finish()))
}

fn cast_numeric_column(
    old_name: &str,
    new_field: &FieldSchema,
    batch: &RecordBatch,
) -> ZResult<Arc<dyn Array>> {
    use finch_types::DataType as Dt;

    let n = batch.num_rows();
    let Some(col) = batch
        .column_by_name(old_name)
        .or_else(|| batch.column_by_name(&new_field.name))
    else {
        let arrow_dt = finch_storage::data_type_to_arrow(&new_field.data_type)
            .ok_or_else(|| Status::invalid_argument("alter_column unsupported data type"))?;
        return Ok(new_null_array(&arrow_dt, n));
    };
    let nn = !new_field.nullable;
    let as_i128 = |row| get_i128(col, row);
    let as_u128 = |row| get_u128(col, row);
    let as_f64 = |row| get_f64(col, row);

    match new_field.data_type {
        Dt::Int32 => build_cast_column::<Int32Type, _>(n, nn, as_i128, |v| {
            i32::try_from(v).map_err(|_| out_of_range("int32"))
        }),
        Dt::Int64 => build_cast_column::<Int64Type, _>(n, nn, as_i128, |v| {
            i64::try_from(v).map_err(|_| out_of_range("int64"))
        }),
        Dt::Uint32 => build_cast_column::<UInt32Type, _>(n, nn, as_u128, |v| {
            u32::try_from(v).map_err(|_| out_of_range("uint32"))
        }),
        Dt::Uint64 => build_cast_column::<UInt64Type, _>(n, nn, as_u128, |v| {
            u64::try_from(v).map_err(|_| out_of_range("uint64"))
        }),
        Dt::Float32 => build_cast_column::<Float32Type, _>(n, nn, as_f64, |v| {
            let f = v as f32;
            if !f.is_finite() {
                return Err(out_of_range("float32"));
            }
            Ok(f)
        }),
        Dt::Float64 => build_cast_column::<Float64Type, _>(n, nn, as_f64, Ok),
        _ => Err(Status::invalid_argument(
            "alter_column only supports basic numeric scalar field_schema",
        )),
    }
}

// Casts the old column (by old or new name) into the altered field's type.
struct AlterColumn<'a> {
    old_name: &'a str,
    new_field: &'a FieldSchema,
}

impl ColumnRewrite for AlterColumn<'_> {
    fn column(&mut self, batch: &RecordBatch, _field: &Field) -> ZResult<Arc<dyn Array>> {
        cast_numeric_column(self.old_name, self.new_field, batch)
    }
}

pub(in super::super) fn rewrite_forward_alter_numeric_column(
    src_forward_path: &Path,
    dst_forward_path: &Path,
    old_name: &str,
    new_schema: &CollectionSchema,
    new_field: &FieldSchema,
) -> ZResult<()> {
    let rewrite = ForwardRewrite::begin(src_forward_path, dst_forward_path, new_schema)?;
    let mut column = AlterColumn {
        old_name,
        new_field,
    };
    rewrite.run(&new_field.name, &mut column)
}

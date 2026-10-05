use std::path::Path;
use std::sync::Arc;

use arrow::array::{new_null_array, Array, PrimitiveBuilder};
use arrow::datatypes::{
    ArrowPrimitiveType, Field, Float32Type, Float64Type, Int32Type, Int64Type, UInt32Type,
    UInt64Type,
};
use arrow::record_batch::RecordBatch;
use finch_types::{CollectionSchema, FieldSchema, Status, ZResult};

use super::{ColumnRewrite, ForwardRewrite};
use crate::ddl_expr::{bind_expression, parse_expression, BoundExpr, Expr, Num};

fn build_expr_column<P: ArrowPrimitiveType>(
    batch: &RecordBatch,
    bound: Option<&BoundExpr>,
    convert: impl Fn(Num) -> ZResult<P::Native>,
) -> ZResult<Arc<dyn Array>> {
    let mut b = PrimitiveBuilder::<P>::new();
    for row in 0..batch.num_rows() {
        match bound.map(|e| e.eval(batch, row)).transpose()?.flatten() {
            None => b.append_null(),
            Some(Num::Float(v)) if !v.is_finite() => {
                return Err(Status::invalid_argument(
                    "expression produced non-finite value",
                ));
            }
            Some(v) => b.append_value(convert(v)?),
        }
    }
    Ok(Arc::new(b.finish()))
}

fn exact_in_range<T: TryFrom<i128>>(v: Num, type_name: &str) -> ZResult<T> {
    T::try_from(v.to_exact_int()?).map_err(|_| {
        Status::invalid_argument(format!("expression value out of {} range", type_name))
    })
}

fn build_numeric_column(
    dt: finch_types::DataType,
    batch: &RecordBatch,
    bound: Option<&BoundExpr>,
) -> ZResult<Arc<dyn Array>> {
    use finch_types::DataType;
    match dt {
        DataType::Int32 => {
            build_expr_column::<Int32Type>(batch, bound, |v| exact_in_range(v, "int32"))
        }
        DataType::Int64 => {
            build_expr_column::<Int64Type>(batch, bound, |v| exact_in_range(v, "int64"))
        }
        DataType::Uint32 => {
            build_expr_column::<UInt32Type>(batch, bound, |v| exact_in_range(v, "uint32"))
        }
        DataType::Uint64 => {
            build_expr_column::<UInt64Type>(batch, bound, |v| exact_in_range(v, "uint64"))
        }
        DataType::Float32 => build_expr_column::<Float32Type>(batch, bound, |v| {
            let f = v.as_f64() as f32;
            if !f.is_finite() {
                return Err(Status::invalid_argument(
                    "expression value out of float32 range",
                ));
            }
            Ok(f)
        }),
        DataType::Float64 => build_expr_column::<Float64Type>(batch, bound, |v| Ok(v.as_f64())),
        _ => Err(Status::invalid_argument(
            "add_column expression only supports basic numeric types",
        )),
    }
}

// Fills the added column from the expression, bound against the first batch's columns.
struct AddColumn<'a> {
    new_field: &'a FieldSchema,
    expr: Option<Expr>,
    bound: Option<BoundExpr>,
}

impl ColumnRewrite for AddColumn<'_> {
    fn start_batch(&mut self, batch: &RecordBatch) -> ZResult<()> {
        if self.bound.is_none() {
            if let Some(e) = &self.expr {
                self.bound = Some(bind_expression(e, batch)?);
            }
        }
        Ok(())
    }

    fn column(&mut self, batch: &RecordBatch, field: &Field) -> ZResult<Arc<dyn Array>> {
        if self.expr.is_none() {
            return Ok(new_null_array(field.data_type(), batch.num_rows()));
        }
        build_numeric_column(self.new_field.data_type, batch, self.bound.as_ref())
    }
}

pub(in super::super) fn rewrite_forward_add_numeric_column(
    src_forward_path: &Path,
    dst_forward_path: &Path,
    new_schema: &CollectionSchema,
    new_field: &FieldSchema,
    expression: Option<&str>,
) -> ZResult<()> {
    let rewrite = ForwardRewrite::begin(src_forward_path, dst_forward_path, new_schema)?;
    let expr = expression
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(parse_expression)
        .transpose()?;
    let mut column = AddColumn {
        new_field,
        expr,
        bound: None,
    };
    rewrite.run(&new_field.name, &mut column)
}

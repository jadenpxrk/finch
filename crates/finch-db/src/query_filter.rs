use finch_types::{CollectionSchema, Status, ZResult};

use crate::sqlengine::parser::{
    enforce_max_filter_terms, parse_filter, rewrite_expressions, validate_filter_expr, FilterExpr,
};

pub(crate) fn prepare_filter_expr(
    schema: &CollectionSchema,
    filter: Option<&str>,
) -> ZResult<Option<FilterExpr>> {
    let filter_expr = filter
        .map(|filter| {
            parse_filter(filter)
                .map_err(|e| Status::invalid_argument(format!("Invalid filter: {}", e.message)))
        })
        .transpose()?;
    if let Some(expr) = filter_expr.as_ref() {
        validate_filter_expr(expr, schema)?;
    }

    let filter_expr = filter_expr.map(rewrite_expressions);
    if let Some(expr) = filter_expr.as_ref() {
        enforce_max_filter_terms(expr)?;
    }

    filter_expr
        .map(|expr| crate::sqlengine::normalize_filter_expr_literals(expr, schema))
        .transpose()
}

pub(crate) fn prepare_required_filter_expr(
    schema: &CollectionSchema,
    filter: &str,
) -> ZResult<FilterExpr> {
    prepare_filter_expr(schema, Some(filter))?
        .ok_or_else(|| Status::invalid_argument("filter is required"))
}

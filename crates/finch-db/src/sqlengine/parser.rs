//! SQL WHERE clause parser → FilterExpr tree

use finch_types::{CollectionSchema, CompareOp, DataType, Status, Value, ZResult};
use regex::RegexBuilder;
use sqlparser::ast::{BinaryOperator, Expr, UnaryOperator, Value as SqlValue};
use sqlparser::dialect::MySqlDialect;
use sqlparser::parser::Parser;

mod convert;
mod normalize;
mod rewrite;
mod validate;

use convert::*;
use normalize::*;
pub use rewrite::rewrite_expressions;
pub use validate::{count_filter_terms, enforce_max_filter_terms, validate_filter_expr};

/// A filter expression tree
#[derive(Debug, Clone)]
pub enum FilterExpr {
    /// field op literal
    Compare {
        field: String,
        op: CompareOp,
        value: Value,
    },
    /// expr AND expr
    And(Box<FilterExpr>, Box<FilterExpr>),
    /// expr OR expr
    Or(Box<FilterExpr>, Box<FilterExpr>),
    /// NOT expr
    Not(Box<FilterExpr>),
    /// IS NULL
    IsNull(String),
    /// IS NOT NULL
    IsNotNull(String),
    /// field IN (v1, v2, ...)
    InList {
        field: String,
        values: Vec<Value>,
        negated: bool,
    },
    /// field contains all values in list
    ContainAll { field: String, values: Vec<Value> },
    /// field contains any value in list
    ContainAny { field: String, values: Vec<Value> },
    /// string field matched by SQL LIKE pattern (`%` and `_`)
    LikePattern { field: String, pattern: String },
    /// string field starts with prefix
    HasPrefix { field: String, prefix: String },
    /// string field ends with suffix
    HasSuffix { field: String, suffix: String },
    /// Compare array_length(field) against an integer
    ArrayLengthCompare {
        field: String,
        op: CompareOp,
        len: u32,
    },
    /// always true
    AlwaysTrue,
    /// always false
    AlwaysFalse,
}

/// Parse a SQL WHERE expression string into a FilterExpr tree
pub fn parse_filter(sql: &str) -> ZResult<FilterExpr> {
    if sql.trim().is_empty() {
        return Ok(FilterExpr::AlwaysTrue);
    }

    reject_unsupported_syntax(sql)?;

    let normalized_sql =
        normalize_string_escapes(&rewrite_contain_ops(&normalize_identifier_quoting(sql)));

    // Wrap in a full SELECT statement so sqlparser can parse it
    let stmt = format!("SELECT * FROM t WHERE {}", normalized_sql);
    let dialect = MySqlDialect {};
    let ast = Parser::parse_sql(&dialect, &stmt)
        // syntax that can't be parsed is surfaced as a syntax error;
        // avoid leaking sqlparser's dialect-specific message text.
        .map_err(|_| Status::invalid_argument("syntax error"))?;

    if ast.is_empty() {
        return Ok(FilterExpr::AlwaysTrue);
    }

    if let sqlparser::ast::Statement::Query(q) = &ast[0] {
        if let sqlparser::ast::SetExpr::Select(sel) = q.body.as_ref() {
            if let Some(where_expr) = &sel.selection {
                return convert_expr(where_expr);
            }
        }
    }

    Ok(FilterExpr::AlwaysTrue)
}

/// Reject syntax that sqlparser accepts but is not supported by Finch's SQL dialect.
#[cfg(test)]
mod tests;

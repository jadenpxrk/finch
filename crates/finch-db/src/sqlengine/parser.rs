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
pub use validate::{enforce_max_filter_terms, validate_filter_expr};

/// A filter expression tree
#[derive(Debug, Clone)]
pub enum FilterExpr {
    /// field op literal
    Compare {
        /// The field to compare.
        field: String,
        /// The comparison.
        op: CompareOp,
        /// The literal to compare with.
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
        /// The field to test.
        field: String,
        /// The literals of the list.
        values: Vec<Value>,
        /// True for `NOT IN`.
        negated: bool,
    },
    /// field contains all values in list
    ContainAll {
        /// The array field to test.
        field: String,
        /// The values the array must all contain.
        values: Vec<Value>,
    },
    /// field contains any value in list
    ContainAny {
        /// The array field to test.
        field: String,
        /// The values the array must contain at least one of.
        values: Vec<Value>,
    },
    /// string field matched by SQL LIKE pattern (`%` and `_`)
    LikePattern {
        /// The string field to match.
        field: String,
        /// The pattern, where `%` matches any text and `_` matches one character.
        pattern: String,
    },
    /// string field starts with prefix
    HasPrefix {
        /// The string field to test.
        field: String,
        /// The text the value must start with.
        prefix: String,
    },
    /// string field ends with suffix
    HasSuffix {
        /// The string field to test.
        field: String,
        /// The text the value must end with.
        suffix: String,
    },
    /// Compare array_length(field) against an integer
    ArrayLengthCompare {
        /// The array field whose length is compared.
        field: String,
        /// The comparison.
        op: CompareOp,
        /// The length to compare with.
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

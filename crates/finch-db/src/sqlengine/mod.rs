//! The SQL-like filter language: parsing, normalization, and evaluation against documents.

pub(crate) mod executor;
pub(crate) mod normalize;
pub(crate) mod parser;
pub(crate) mod query;

pub(crate) use normalize::normalize_filter_expr_literals;
pub use parser::{parse_filter, FilterExpr};

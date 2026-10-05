mod common;
use common::*;

#[path = "filter_queries_test/array_filters.rs"]
mod array_filters;
#[path = "filter_queries_test/binary_fields.rs"]
mod binary_fields;
#[path = "filter_queries_test/filter_semantics.rs"]
mod filter_semantics;
#[path = "filter_queries_test/filter_syntax.rs"]
mod filter_syntax;
#[path = "filter_queries_test/optimize_deletes.rs"]
mod optimize_deletes;
#[path = "filter_queries_test/vector_query_semantics.rs"]
mod vector_query_semantics;

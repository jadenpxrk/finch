mod common;
use common::*;

#[path = "insert_query_test/sql_queries.rs"]
mod sql_queries;
#[path = "insert_query_test/system_columns.rs"]
mod system_columns;
#[path = "insert_query_test/vector_dtypes.rs"]
mod vector_dtypes;
#[path = "insert_query_test/write_validation.rs"]
mod write_validation;

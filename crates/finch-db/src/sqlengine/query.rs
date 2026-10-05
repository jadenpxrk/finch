mod ast;
mod lexer;
mod parser;

pub use ast::{
    CmpOp, ContainOp, LogicExpr, OrderByItem, RelExpr, SelectItem, SqlSelect, ValueExpr,
    VectorCond, VectorLiteral,
};
pub(super) use lexer::{is_identifier_keyword, scan_numeric_token};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_select_with_where_limit() {
        let s = SqlSelect::parse("SELECT id, name AS n FROM tbl WHERE age >= 18 LIMIT 10").unwrap();
        assert_eq!(s.table, "tbl");
        assert_eq!(s.selected.len(), 2);
        assert_eq!(s.limit, Some(10));
        assert!(s.where_expr.is_some());
    }

    #[test]
    fn test_compilation_unit_allows_trailing_delimiters() {
        SqlSelect::parse("SELECT * FROM tbl;").unwrap();
        SqlSelect::parse("SELECT * FROM tbl /").unwrap();
    }

    #[test]
    fn test_compilation_unit_rejects_second_statement_after_delimiter() {
        assert!(SqlSelect::parse("SELECT * FROM tbl; SELECT * FROM tbl").is_err());
    }

    #[test]
    fn test_compilation_unit_rejects_second_statement_without_delimiter() {
        assert!(SqlSelect::parse("SELECT * FROM tbl SELECT * FROM tbl").is_err());
    }

    #[test]
    fn test_limit_accepts_zero_and_negative_integers() {
        let zero = SqlSelect::parse("SELECT * FROM tbl LIMIT 0").unwrap();
        assert_eq!(zero.limit, Some(0));

        let negative = SqlSelect::parse("SELECT * FROM tbl LIMIT -1").unwrap();
        assert_eq!(negative.limit, Some(-1));
    }

    #[test]
    fn test_parse_single_row_matrix_vector_literal() {
        let s = SqlSelect::parse("SELECT id FROM tbl WHERE emb = [[1, 2, 3]] LIMIT 1").unwrap();
        let (cond, scalar) = s
            .where_expr
            .expect("where expr")
            .split_vector_condition()
            .unwrap();
        assert!(scalar.is_none());
        let cond = cond.expect("vector condition");
        assert_eq!(cond.field, "emb");
        assert!(matches!(cond.literal, VectorLiteral::Matrix(_)));
    }

    #[test]
    fn test_split_vector_condition_requires_no_or_ancestor() {
        let s = SqlSelect::parse("SELECT id FROM tbl WHERE emb = [1,2] OR label = 'x'").unwrap();
        assert!(s
            .where_expr
            .expect("where expr")
            .split_vector_condition()
            .is_err());
    }

    #[test]
    fn test_split_vector_condition_produces_scalar_filter_string_parseable_by_filter_parser() {
        let s =
            SqlSelect::parse("SELECT id FROM tbl WHERE label = 'x' AND emb = [1,2] AND age >= 18")
                .unwrap();
        let (_vector, scalar) = s
            .where_expr
            .expect("where expr")
            .split_vector_condition()
            .unwrap();
        let scalar = scalar.expect("scalar filter");
        assert!(crate::sqlengine::parser::parse_filter(&scalar.to_filter_string()).is_ok());
    }
}

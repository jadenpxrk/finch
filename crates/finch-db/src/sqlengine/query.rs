mod ast;
mod lexer;
mod parser;

pub use ast::{
    CmpOp, LogicExpr, RelExpr, SelectItem, SqlSelect, ValueExpr, VectorCond, VectorLiteral,
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
}

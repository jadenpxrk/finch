use finch_types::{Status, ZResult};

use super::ast::*;
use super::lexer::*;

pub(super) struct Parser<'a> {
    input: &'a str,
    tokens: Vec<Token>,
    pos: usize,
}

impl<'a> Parser<'a> {
    pub(super) fn new(input: &'a str) -> Self {
        let tokens = Lexer::new(input).lex_all().unwrap_or_default();
        Parser {
            input,
            tokens,
            pos: 0,
        }
    }

    pub(super) fn parse_compilation_unit_first_select(mut self) -> ZResult<SqlSelect> {
        // Re-lex with proper error surfacing.
        self.tokens = Lexer::new(self.input).lex_all()?;

        // compilation_unit : (unit_statement (SOLIDUS | SEMI)?)+ EOF
        // unit_statement   : dql_statement
        // dql_statement    : select_statement
        let first = self.parse_select_statement()?;

        // Optional delimiter(s): ';' or '/'
        while self.consume_symbol(Symbol::Semi) || self.consume_symbol(Symbol::Solidus) {
            // allow multiple
        }

        if self.pos < self.tokens.len() {
            return Err(Status::invalid_argument("syntax error"));
        }

        Ok(first)
    }

    fn parse_select_statement(&mut self) -> ZResult<SqlSelect> {
        self.expect_keyword(Keyword::Select)?;
        let selected = self.parse_selected_elements()?;
        self.expect_keyword(Keyword::From)?;
        let table = self.parse_identifier_string()?;

        let where_expr = if self.consume_keyword(Keyword::Where) {
            Some(self.parse_logic_expr()?)
        } else {
            None
        };

        let order_by = if self.consume_keyword(Keyword::Order) {
            self.expect_keyword(Keyword::By)?;
            self.parse_order_by_clause()?
        } else {
            Vec::new()
        };

        let limit = if self.consume_keyword(Keyword::Limit) {
            let n = self.parse_int_i32()?;
            Some(n)
        } else {
            None
        };

        Ok(SqlSelect {
            table,
            selected,
            where_expr,
            order_by,
            limit,
        })
    }

    /// Parses `item (',' item)*`.
    fn parse_comma_list<T>(&mut self, item: fn(&mut Self) -> ZResult<T>) -> ZResult<Vec<T>> {
        let mut out: Vec<T> = Vec::new();
        loop {
            out.push(item(self)?);
            if !self.consume_symbol(Symbol::Comma) {
                break;
            }
        }
        Ok(out)
    }

    fn parse_selected_elements(&mut self) -> ZResult<Vec<SelectItem>> {
        let out = self.parse_comma_list(Self::parse_selected_element)?;
        if out.is_empty() {
            return Err(Status::invalid_argument("syntax error"));
        }
        Ok(out)
    }

    fn parse_selected_element(&mut self) -> ZResult<SelectItem> {
        if self.consume_symbol(Symbol::Asterisk) {
            return Ok(SelectItem::Asterisk);
        }
        let name = self.parse_identifier_string()?;
        let alias = if self.consume_keyword(Keyword::As) {
            Some(self.parse_identifier_string()?)
        } else {
            // Optional bare alias: <field> <alias>, but only when the next token is an identifier-like.
            if self.peek_is_identifier_like() {
                Some(self.parse_identifier_string()?)
            } else {
                None
            }
        };
        Ok(SelectItem::Field { name, alias })
    }

    fn parse_order_by_clause(&mut self) -> ZResult<Vec<OrderByItem>> {
        self.parse_comma_list(Self::parse_order_by_item)
    }

    fn parse_order_by_item(&mut self) -> ZResult<OrderByItem> {
        let field = self.parse_identifier_string()?;
        let desc = if self.consume_keyword(Keyword::Desc) {
            true
        } else {
            let _ = self.consume_keyword(Keyword::Asc);
            false
        };
        Ok(OrderByItem { field, desc })
    }

    // ── WHERE logic_expr ───────────────────────────────────────────────────

    fn parse_logic_expr(&mut self) -> ZResult<LogicExpr> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> ZResult<LogicExpr> {
        let mut left = self.parse_and()?;
        while self.consume_keyword(Keyword::Or) {
            let right = self.parse_and()?;
            left = LogicExpr::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> ZResult<LogicExpr> {
        let mut left = self.parse_primary_logic()?;
        while self.consume_keyword(Keyword::And) {
            let right = self.parse_primary_logic()?;
            left = LogicExpr::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_primary_logic(&mut self) -> ZResult<LogicExpr> {
        if self.consume_symbol(Symbol::LParen) {
            let inner = self.parse_logic_expr()?;
            self.expect_symbol(Symbol::RParen)?;
            return Ok(inner);
        }
        Ok(LogicExpr::Rel(self.parse_relation_expr()?))
    }

    fn parse_relation_expr(&mut self) -> ZResult<RelExpr> {
        let left = self.parse_rel_expr_left()?;

        // IS [NOT] NULL
        if self.consume_keyword(Keyword::Is) {
            // grammar: identifier IS NOT? NULL (function_call is not allowed here).
            reject_function_call_operand(&left)?;
            let negated = self.consume_keyword(Keyword::Not);
            self.expect_keyword(Keyword::Null)?;
            return Ok(RelExpr::IsNull {
                field: left,
                negated,
            });
        }

        // LIKE
        if self.consume_keyword(Keyword::Like) {
            // grammar: identifier LIKE value_expr (function_call is not allowed here).
            reject_function_call_operand(&left)?;
            let pat = self.parse_value_expr()?;
            return Ok(RelExpr::Like {
                field: left,
                pattern: pat,
            });
        }

        let negated = self.consume_keyword(Keyword::Not);
        if let Some(rel) = self.parse_list_relation(&left, negated)? {
            return Ok(rel);
        }

        // Comparison operator
        let op = self.parse_cmp_op()?;
        let right = self.parse_value_expr()?;
        Ok(RelExpr::Compare {
            lhs: left,
            op,
            rhs: right,
        })
    }

    /// Parses `NOT? IN (...)` or `NOT? CONTAIN_* (...)` after `left`, if present.
    fn parse_list_relation(&mut self, left: &ValueExpr, negated: bool) -> ZResult<Option<RelExpr>> {
        if self.consume_keyword(Keyword::In) {
            // grammar: identifier NOT? IN (...) (function_call is not allowed here).
            reject_function_call_operand(left)?;
            self.expect_symbol(Symbol::LParen)?;
            let values = self.parse_in_list_values()?;
            self.expect_symbol(Symbol::RParen)?;
            return Ok(Some(RelExpr::InList {
                field: left.clone(),
                negated,
                values,
            }));
        }

        let op = if self.consume_keyword(Keyword::ContainAll) {
            ContainOp::All
        } else if self.consume_keyword(Keyword::ContainAny) {
            ContainOp::Any
        } else {
            return Ok(None);
        };
        // grammar: identifier NOT? CONTAIN_* (...) (function_call is not allowed here).
        reject_function_call_operand(left)?;
        self.expect_symbol(Symbol::LParen)?;
        let values = if self.peek_symbol(Symbol::RParen) {
            Vec::new()
        } else {
            self.parse_in_list_values()?
        };
        self.expect_symbol(Symbol::RParen)?;
        Ok(Some(RelExpr::Contain {
            field: left.clone(),
            op,
            negated,
            values,
        }))
    }

    fn parse_rel_expr_left(&mut self) -> ZResult<ValueExpr> {
        let name = self.parse_identifier_string()?;
        if !self.consume_symbol(Symbol::LParen) {
            return Ok(ValueExpr::Ident(name));
        }
        let args = if self.peek_symbol(Symbol::RParen) {
            Vec::new()
        } else {
            self.parse_comma_list(Self::parse_function_value_expr)?
        };
        self.expect_symbol(Symbol::RParen)?;
        Ok(ValueExpr::FunctionCall { name, args })
    }

    fn parse_function_value_expr(&mut self) -> ZResult<ValueExpr> {
        // function_value_expr : value_expr | identifier
        match self.peek() {
            Some(TokenKind::Int(_))
            | Some(TokenKind::Float(_))
            | Some(TokenKind::StringLiteral(_))
            | Some(TokenKind::VectorLiteral(_))
            | Some(TokenKind::Keyword(Keyword::True | Keyword::False)) => self.parse_value_expr(),
            _ => {
                let name = self.parse_identifier_string()?;
                Ok(ValueExpr::Ident(name))
            }
        }
    }

    fn parse_value_expr(&mut self) -> ZResult<ValueExpr> {
        // value_expr : constant | function_call
        // function_call starts with identifier followed by '('.
        if self.peek_is_identifier_like() && self.peek_ahead_symbol(1, Symbol::LParen) {
            return self.parse_rel_expr_left();
        }
        self.parse_constant()
    }

    fn parse_constant(&mut self) -> ZResult<ValueExpr> {
        match self.next() {
            Some(TokenKind::Int(s)) => Ok(ValueExpr::Int(s)),
            Some(TokenKind::Float(s)) => Ok(ValueExpr::Float(s)),
            Some(TokenKind::StringLiteral(s)) => Ok(ValueExpr::StringLiteral(s)),
            Some(TokenKind::VectorLiteral(s)) => Ok(ValueExpr::VectorLiteral(s)),
            Some(TokenKind::Keyword(Keyword::True)) => Ok(ValueExpr::BoolLiteral(true)),
            Some(TokenKind::Keyword(Keyword::False)) => Ok(ValueExpr::BoolLiteral(false)),
            Some(TokenKind::Symbol(Symbol::LBracket)) => self.parse_matrix_rows(),
            _ => Err(Status::invalid_argument("syntax error")),
        }
    }

    /// matrix: '[' VECTOR (',' VECTOR)* ']' after the opening bracket.
    fn parse_matrix_rows(&mut self) -> ZResult<ValueExpr> {
        let mut vecs: Vec<String> = vec![self.expect_vector_literal()?];
        while self.consume_symbol(Symbol::Comma) {
            vecs.push(self.expect_vector_literal()?);
        }
        self.expect_symbol(Symbol::RBracket)?;
        Ok(ValueExpr::MatrixLiteral(vecs))
    }

    fn expect_vector_literal(&mut self) -> ZResult<String> {
        match self.next() {
            Some(TokenKind::VectorLiteral(v)) => Ok(v),
            _ => Err(Status::invalid_argument("syntax error")),
        }
    }

    fn parse_in_list_values(&mut self) -> ZResult<Vec<ValueExpr>> {
        let out = self.parse_comma_list(Self::parse_in_list_value)?;
        if out.is_empty() {
            return Err(Status::invalid_argument("syntax error"));
        }
        Ok(out)
    }

    fn parse_in_list_value(&mut self) -> ZResult<ValueExpr> {
        match self.peek() {
            Some(TokenKind::Int(_))
            | Some(TokenKind::Float(_))
            | Some(TokenKind::StringLiteral(_))
            | Some(TokenKind::Keyword(Keyword::True | Keyword::False)) => self.parse_constant(),
            _ => Err(Status::invalid_argument("syntax error")),
        }
    }

    fn parse_cmp_op(&mut self) -> ZResult<CmpOp> {
        match self.next() {
            Some(TokenKind::Symbol(Symbol::Eq)) => Ok(CmpOp::Eq),
            Some(TokenKind::Symbol(Symbol::Ne)) => Ok(CmpOp::Ne),
            Some(TokenKind::Symbol(Symbol::Lt)) => Ok(CmpOp::Lt),
            Some(TokenKind::Symbol(Symbol::Gt)) => Ok(CmpOp::Gt),
            Some(TokenKind::Symbol(Symbol::Le)) => Ok(CmpOp::Le),
            Some(TokenKind::Symbol(Symbol::Ge)) => Ok(CmpOp::Ge),
            _ => Err(Status::invalid_argument("syntax error")),
        }
    }

    // ── Token helpers ─────────────────────────────────────────────────────

    fn peek(&self) -> Option<TokenKind> {
        self.tokens.get(self.pos).map(|t| t.kind.clone())
    }

    fn peek_tok(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn next(&mut self) -> Option<TokenKind> {
        let t = self.tokens.get(self.pos)?.kind.clone();
        self.pos += 1;
        Some(t)
    }

    fn next_tok(&mut self) -> Option<Token> {
        let t = self.tokens.get(self.pos)?.clone();
        self.pos += 1;
        Some(t)
    }

    fn peek_symbol(&self, sym: Symbol) -> bool {
        matches!(self.peek(), Some(TokenKind::Symbol(s)) if s == sym)
    }

    fn peek_ahead_symbol(&self, ahead: usize, sym: Symbol) -> bool {
        matches!(self.tokens.get(self.pos + ahead).map(|t| &t.kind), Some(TokenKind::Symbol(s)) if *s == sym)
    }

    fn peek_is_identifier_like(&self) -> bool {
        match self.peek_tok().map(|t| &t.kind) {
            Some(TokenKind::Ident(_)) => true,
            Some(TokenKind::Keyword(k)) => k.can_be_identifier(),
            _ => false,
        }
    }

    fn parse_identifier_string(&mut self) -> ZResult<String> {
        let tok = self
            .next_tok()
            .ok_or_else(|| Status::invalid_argument("syntax error"))?;
        match tok.kind {
            TokenKind::Ident(s) => Ok(s),
            TokenKind::Keyword(k) if k.can_be_identifier() => Ok(self
                .input
                .get(tok.start..tok.end)
                .ok_or_else(|| Status::invalid_argument("syntax error"))?
                .to_string()),
            _ => Err(Status::invalid_argument("syntax error")),
        }
    }

    fn parse_int_i32(&mut self) -> ZResult<i32> {
        match self.next() {
            Some(TokenKind::Int(s)) => s
                .parse::<i32>()
                .map_err(|_| Status::invalid_argument("syntax error")),
            _ => Err(Status::invalid_argument("syntax error")),
        }
    }

    fn expect_keyword(&mut self, kw: Keyword) -> ZResult<()> {
        match self.next() {
            Some(TokenKind::Keyword(k)) if k == kw => Ok(()),
            _ => Err(Status::invalid_argument("syntax error")),
        }
    }

    fn consume_keyword(&mut self, kw: Keyword) -> bool {
        if matches!(self.peek(), Some(TokenKind::Keyword(k)) if k == kw) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect_symbol(&mut self, sym: Symbol) -> ZResult<()> {
        match self.next() {
            Some(TokenKind::Symbol(s)) if s == sym => Ok(()),
            _ => Err(Status::invalid_argument("syntax error")),
        }
    }

    fn consume_symbol(&mut self, sym: Symbol) -> bool {
        if matches!(self.peek(), Some(TokenKind::Symbol(s)) if s == sym) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
}

/// Grammar positions that take an identifier do not accept a function call.
fn reject_function_call_operand(left: &ValueExpr) -> ZResult<()> {
    if matches!(left, ValueExpr::FunctionCall { .. }) {
        return Err(Status::invalid_argument("syntax error"));
    }
    Ok(())
}

use arrow::array::{
    Array, Float32Array, Float64Array, Int32Array, Int64Array, UInt32Array, UInt64Array,
};
use arrow::datatypes::DataType as ArrowType;
use arrow::record_batch::RecordBatch;
use finch_types::{Status, ZResult};

#[derive(Debug, Clone)]
pub enum Expr {
    Number(Num),
    Column(String),
    Neg(Box<Expr>),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
}

#[derive(Debug, Clone)]
enum Token {
    Ident(String),
    Number(Num),
    Plus,
    Minus,
    Star,
    Slash,
    LParen,
    RParen,
}

fn tokenize(s: &str) -> ZResult<Vec<Token>> {
    let mut out = Vec::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if let Some(tok) = symbol_token(c) {
            out.push(tok);
            i += 1;
            continue;
        }
        let start = i;
        if c.is_ascii_digit() {
            i = scan_number_end(&chars, start);
            let text: String = chars[start..i].iter().collect();
            out.push(Token::Number(number_literal(&text)?));
        } else if c.is_ascii_alphabetic() || c == '_' {
            // DDL arithmetic expression identifiers do not accept '-'.
            i = scan_digits_or(&chars, start + 1, |c| c.is_ascii_alphabetic() || c == '_');
            out.push(Token::Ident(chars[start..i].iter().collect()));
        } else {
            return Err(Status::invalid_argument(format!(
                "invalid character in expression: '{}'",
                c
            )));
        }
    }
    Ok(out)
}

// Digit-only literals parse as integers so values beyond 2^53 are not rounded through f64.
fn number_literal(text: &str) -> ZResult<Num> {
    if let Ok(v) = text.parse::<i128>() {
        return Ok(Num::Int(v));
    }
    text.parse::<f64>().map(Num::literal).map_err(|_| {
        Status::invalid_argument(format!("invalid numeric literal in expression: '{}'", text))
    })
}

fn symbol_token(c: char) -> Option<Token> {
    Some(match c {
        '+' => Token::Plus,
        '-' => Token::Minus,
        '*' => Token::Star,
        '/' => Token::Slash,
        '(' => Token::LParen,
        ')' => Token::RParen,
        _ => return None,
    })
}

/// Advances past ASCII digits and chars matching `also`.
fn scan_digits_or(chars: &[char], mut i: usize, also: fn(char) -> bool) -> usize {
    while i < chars.len() && (chars[i].is_ascii_digit() || also(chars[i])) {
        i += 1;
    }
    i
}

/// End of `\d+(\.\d*)?([eE][+-]?\d+)?` starting at a digit.
fn scan_number_end(chars: &[char], start: usize) -> usize {
    let no_extra = |_: char| false;
    let mut i = scan_digits_or(chars, start + 1, no_extra);
    if chars.get(i) == Some(&'.') {
        i = scan_digits_or(chars, i + 1, no_extra);
    }
    if !matches!(chars.get(i), Some('e' | 'E')) {
        return i;
    }
    let mut digits_start = i + 1;
    if matches!(chars.get(digits_start), Some('+' | '-')) {
        digits_start += 1;
    }
    let end = scan_digits_or(chars, digits_start, no_extra);
    // A trailing `e` without digits is not part of the number.
    if end == digits_start {
        i
    } else {
        end
    }
}

struct Parser {
    toks: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(toks: Vec<Token>) -> Self {
        Parser { toks, pos: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.toks.get(self.pos)
    }

    fn next(&mut self) -> Option<Token> {
        let t = self.toks.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn parse_expr(&mut self, min_bp: u8) -> ZResult<Expr> {
        let mut lhs = match self
            .next()
            .ok_or_else(|| Status::invalid_argument("empty expression"))?
        {
            Token::Number(v) => Expr::Number(v),
            Token::Ident(s) => Expr::Column(s),
            Token::Plus => {
                // unary plus is allowed.

                self.parse_expr(10)?
            }
            Token::Minus => {
                let rhs = self.parse_expr(10)?;
                Expr::Neg(Box::new(rhs))
            }
            Token::LParen => {
                let e = self.parse_expr(0)?;
                match self.next() {
                    Some(Token::RParen) => e,
                    _ => return Err(Status::invalid_argument("missing ')' in expression")),
                }
            }
            other => {
                return Err(Status::invalid_argument(format!(
                    "unexpected token in expression: {:?}",
                    other
                )));
            }
        };

        loop {
            #[derive(Clone, Copy)]
            enum OpKind {
                Add,
                Sub,
                Mul,
                Div,
            }

            let op = match self.peek() {
                Some(Token::Plus) => (1u8, 2u8, OpKind::Add),
                Some(Token::Minus) => (1u8, 2u8, OpKind::Sub),
                Some(Token::Star) => (3u8, 4u8, OpKind::Mul),
                Some(Token::Slash) => (3u8, 4u8, OpKind::Div),
                _ => break,
            };
            let (l_bp, r_bp, _tok) = op;
            if l_bp < min_bp {
                break;
            }
            let _ = self.next();
            let rhs = self.parse_expr(r_bp)?;
            lhs = match op.2 {
                OpKind::Add => Expr::Add(Box::new(lhs), Box::new(rhs)),
                OpKind::Sub => Expr::Sub(Box::new(lhs), Box::new(rhs)),
                OpKind::Mul => Expr::Mul(Box::new(lhs), Box::new(rhs)),
                OpKind::Div => Expr::Div(Box::new(lhs), Box::new(rhs)),
            };
        }

        Ok(lhs)
    }
}

pub fn parse_expression(s: &str) -> ZResult<Expr> {
    let toks = tokenize(s)?;
    let mut p = Parser::new(toks);
    let e = p.parse_expr(0)?;
    if p.peek().is_some() {
        return Err(Status::invalid_argument(
            "unexpected trailing tokens in expression",
        ));
    }
    Ok(e)
}

#[derive(Debug, Clone)]
pub enum BoundExpr {
    Number(Num),
    Column { index: usize, dtype: ArrowType },
    Neg(Box<BoundExpr>),
    Add(Box<BoundExpr>, Box<BoundExpr>),
    Sub(Box<BoundExpr>, Box<BoundExpr>),
    Mul(Box<BoundExpr>, Box<BoundExpr>),
    Div(Box<BoundExpr>, Box<BoundExpr>),
}

pub fn bind_expression(expr: &Expr, batch: &RecordBatch) -> ZResult<BoundExpr> {
    let schema = batch.schema();
    bind_inner(expr, schema.as_ref(), batch)
}

fn bind_inner(
    expr: &Expr,
    schema: &arrow::datatypes::Schema,
    batch: &RecordBatch,
) -> ZResult<BoundExpr> {
    let bind = |e: &Expr| bind_inner(e, schema, batch).map(Box::new);
    Ok(match expr {
        Expr::Number(v) => BoundExpr::Number(*v),
        Expr::Column(name) => bind_column(name, schema, batch)?,
        Expr::Neg(a) => BoundExpr::Neg(bind(a)?),
        Expr::Add(a, b) => BoundExpr::Add(bind(a)?, bind(b)?),
        Expr::Sub(a, b) => BoundExpr::Sub(bind(a)?, bind(b)?),
        Expr::Mul(a, b) => BoundExpr::Mul(bind(a)?, bind(b)?),
        Expr::Div(a, b) => BoundExpr::Div(bind(a)?, bind(b)?),
    })
}

fn bind_column(
    name: &str,
    schema: &arrow::datatypes::Schema,
    batch: &RecordBatch,
) -> ZResult<BoundExpr> {
    let idx = schema.index_of(name).map_err(|_| {
        Status::invalid_argument(format!("expression references unknown column '{}'", name))
    })?;
    let dtype = batch.column(idx).data_type().clone();
    if !matches!(
        dtype,
        ArrowType::Int32
            | ArrowType::Int64
            | ArrowType::UInt32
            | ArrowType::UInt64
            | ArrowType::Float32
            | ArrowType::Float64
    ) {
        return Err(Status::invalid_argument(format!(
            "expression references non-numeric column '{}'",
            name
        )));
    }
    Ok(BoundExpr::Column { index: idx, dtype })
}

// Integers stay exact through arithmetic; f64 cannot hold every Int64 or UInt64 value.
#[derive(Debug, Clone, Copy)]
pub enum Num {
    Int(i128),
    Float(f64),
}

impl Num {
    pub fn as_f64(self) -> f64 {
        match self {
            Num::Int(v) => v as f64,
            Num::Float(v) => v,
        }
    }

    fn literal(v: f64) -> Num {
        const MAX_EXACT: f64 = 9_007_199_254_740_992.0;
        if v.fract() == 0.0 && v.abs() <= MAX_EXACT {
            Num::Int(v as i128)
        } else {
            Num::Float(v)
        }
    }

    fn combine(
        self,
        other: Num,
        int_op: fn(i128, i128) -> Option<i128>,
        float_op: fn(f64, f64) -> f64,
    ) -> Num {
        if let (Num::Int(x), Num::Int(y)) = (self, other) {
            if let Some(v) = int_op(x, y) {
                return Num::Int(v);
            }
        }
        Num::Float(float_op(self.as_f64(), other.as_f64()))
    }
}

fn numeric_value(col: &dyn Array, row: usize, dtype: &ArrowType) -> Option<Num> {
    if col.is_null(row) {
        return None;
    }
    match dtype {
        ArrowType::Int32 => col
            .as_any()
            .downcast_ref::<Int32Array>()
            .map(|a| Num::Int(a.value(row).into())),
        ArrowType::Int64 => col
            .as_any()
            .downcast_ref::<Int64Array>()
            .map(|a| Num::Int(a.value(row).into())),
        ArrowType::UInt32 => col
            .as_any()
            .downcast_ref::<UInt32Array>()
            .map(|a| Num::Int(a.value(row).into())),
        ArrowType::UInt64 => col
            .as_any()
            .downcast_ref::<UInt64Array>()
            .map(|a| Num::Int(a.value(row).into())),
        ArrowType::Float32 => col
            .as_any()
            .downcast_ref::<Float32Array>()
            .map(|a| Num::Float(a.value(row) as f64)),
        ArrowType::Float64 => col
            .as_any()
            .downcast_ref::<Float64Array>()
            .map(|a| Num::Float(a.value(row))),
        _ => None,
    }
}

impl BoundExpr {
    pub fn eval(&self, batch: &RecordBatch, row: usize) -> Option<Num> {
        let pair = |a: &BoundExpr, b: &BoundExpr| Some((a.eval(batch, row)?, b.eval(batch, row)?));
        match self {
            BoundExpr::Number(v) => Some(*v),
            BoundExpr::Column { index, dtype } => {
                let col = batch.column(*index);
                numeric_value(col.as_ref(), row, dtype)
            }
            BoundExpr::Neg(a) => a.eval(batch, row).map(|v| match v {
                Num::Int(i) => i.checked_neg().map_or(Num::Float(-(i as f64)), Num::Int),
                Num::Float(f) => Num::Float(-f),
            }),
            BoundExpr::Add(a, b) => {
                pair(a, b).map(|(x, y)| x.combine(y, i128::checked_add, |x, y| x + y))
            }
            BoundExpr::Sub(a, b) => {
                pair(a, b).map(|(x, y)| x.combine(y, i128::checked_sub, |x, y| x - y))
            }
            BoundExpr::Mul(a, b) => {
                pair(a, b).map(|(x, y)| x.combine(y, i128::checked_mul, |x, y| x * y))
            }
            BoundExpr::Div(a, b) => {
                let (x, y) = pair(a, b)?;
                let y = y.as_f64();
                if y == 0.0 {
                    return None;
                }
                Some(Num::Float(x.as_f64() / y))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_expression_accepts_no_whitespace_arithmetic() {
        let e = parse_expression("1+2*3").unwrap();
        // Just ensure it parsed; evaluation correctness is covered via collection integration tests.
        assert!(matches!(e, Expr::Add(_, _)));
    }

    #[test]
    fn test_parse_expression_accepts_unary_plus() {
        let e = parse_expression("+123").unwrap();
        assert!(matches!(e, Expr::Number(Num::Int(123))));
    }

    #[test]
    fn test_parse_expression_rejects_leading_dot_number() {
        let err = parse_expression(".5").unwrap_err();
        assert!(
            err.message.contains("invalid character")
                || err.message.contains("invalid numeric literal")
        );
    }

    #[test]
    fn test_parse_expression_scientific_notation_roundtrips() {
        let e = parse_expression("1e-3+2").unwrap();
        assert!(matches!(e, Expr::Add(_, _)));
    }
}

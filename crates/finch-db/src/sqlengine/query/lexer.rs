use finch_types::{Status, ZResult};
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(super) enum Keyword {
    Or,
    And,
    Not,
    In,
    ContainAll,
    ContainAny,
    Between,
    Like,
    Where,
    Select,
    From,
    As,
    Order,
    By,
    Asc,
    Desc,
    Limit,
    True,
    False,
    Is,
    Null,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(super) enum Symbol {
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Semi,
    Solidus,
    Asterisk,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum TokenKind {
    Keyword(Keyword),
    Symbol(Symbol),
    Ident(String),
    Int(String),
    Float(String),
    StringLiteral(String),
    VectorLiteral(String),
}

#[derive(Debug, Clone)]
pub(super) struct Token {
    pub(super) kind: TokenKind,
    pub(super) start: usize,
    pub(super) end: usize,
}

pub(super) struct Lexer<'a> {
    input: &'a str,
    bytes: &'a [u8],
    i: usize,
}

impl<'a> Lexer<'a> {
    pub(super) fn new(input: &'a str) -> Self {
        Lexer {
            input,
            bytes: input.as_bytes(),
            i: 0,
        }
    }

    pub(super) fn lex_all(mut self) -> ZResult<Vec<Token>> {
        let mut out: Vec<Token> = Vec::new();
        while self.skip_ws_and_comments()? {
            // keep skipping
        }
        while self.i < self.bytes.len() {
            if self.skip_ws_and_comments()? {
                continue;
            }
            if self.i >= self.bytes.len() {
                break;
            }
            out.push(self.lex_one()?);
        }
        Ok(out)
    }

    fn skip_ws_and_comments(&mut self) -> ZResult<bool> {
        let mut progressed = false;
        while self.i < self.bytes.len() {
            // Whitespace
            if self.bytes[self.i].is_ascii_whitespace() {
                progressed = true;
                self.i += 1;
                continue;
            }

            // Line comment: --
            if self.starts_with_pair(b'-', b'-') {
                progressed = true;
                self.i += 2;
                while self.i < self.bytes.len() && self.bytes[self.i] != b'\n' {
                    self.i += 1;
                }
                continue;
            }

            // Block comment: /* ... */
            if self.starts_with_pair(b'/', b'*') {
                progressed = true;
                self.i += 2;
                self.skip_block_comment_body()?;
                continue;
            }

            break;
        }
        Ok(progressed)
    }

    fn starts_with_pair(&self, first: u8, second: u8) -> bool {
        self.bytes[self.i] == first && self.bytes.get(self.i + 1) == Some(&second)
    }

    /// Advances past the closing `*/` of a block comment.
    fn skip_block_comment_body(&mut self) -> ZResult<()> {
        while self.i + 1 < self.bytes.len() {
            if self.bytes[self.i] == b'*' && self.bytes[self.i + 1] == b'/' {
                self.i += 2;
                return Ok(());
            }
            self.i += 1;
        }
        Err(Status::invalid_argument("syntax error"))
    }

    /// Consumes input up to `end` and returns the token spanning `start..end`.
    fn token_to(&mut self, kind: TokenKind, start: usize, end: usize) -> Token {
        self.i = end;
        Token { kind, start, end }
    }

    fn lex_one(&mut self) -> ZResult<Token> {
        let start = self.i;
        let b = self.bytes[self.i];

        if b == b'[' {
            return Ok(self.lex_bracket(start));
        }
        if let Some(sym) = self.symbol_at(start) {
            let end = start + sym.width();
            return Ok(self.token_to(TokenKind::Symbol(sym), start, end));
        }

        // String literals: '...' or "..."
        if b == b'\'' || b == b'"' {
            return self.lex_string_literal(start);
        }

        // Numbers (int/float). Prefer int when ambiguous.
        if b == b'-' || b.is_ascii_digit() || b == b'.' {
            if let Some(number) = scan_numeric_token(self.bytes, start) {
                let raw = self.input[start..number.end].to_string();
                let kind = if number.is_float {
                    TokenKind::Float(raw)
                } else {
                    TokenKind::Int(raw)
                };
                return Ok(self.token_to(kind, start, number.end));
            }
        }

        // Ident / keyword (REGULAR_ID)
        if is_regular_id_char(b) {
            return Ok(self.lex_word(start));
        }

        Err(Status::invalid_argument("syntax error"))
    }

    /// VECTOR token: `[` <numbers/commas/spaces> `]` with no nested '['; otherwise
    /// an LBracket symbol (used for matrix literals).
    fn lex_bracket(&mut self, start: usize) -> Token {
        if let Some(end) = self.scan_vector_literal(start) {
            let lit = self.input[start..end].to_string();
            return self.token_to(TokenKind::VectorLiteral(lit), start, end);
        }
        self.token_to(TokenKind::Symbol(Symbol::LBracket), start, start + 1)
    }

    /// Two-char operators take precedence over their one-char prefixes.
    fn symbol_at(&self, start: usize) -> Option<Symbol> {
        let b = self.bytes[start];
        if self.bytes.get(start + 1) == Some(&b'=') {
            match b {
                b'!' => return Some(Symbol::Ne),
                b'<' => return Some(Symbol::Le),
                b'>' => return Some(Symbol::Ge),
                _ => {}
            }
        }
        match b {
            b'(' => Some(Symbol::LParen),
            b')' => Some(Symbol::RParen),
            b']' => Some(Symbol::RBracket),
            b',' => Some(Symbol::Comma),
            b';' => Some(Symbol::Semi),
            b'/' => Some(Symbol::Solidus),
            b'*' => Some(Symbol::Asterisk),
            b'=' => Some(Symbol::Eq),
            b'<' => Some(Symbol::Lt),
            b'>' => Some(Symbol::Gt),
            _ => None,
        }
    }

    fn lex_string_literal(&mut self, start: usize) -> ZResult<Token> {
        let quote = self.bytes[start];
        let mut i = start + 1;
        while i < self.bytes.len() {
            let c = self.bytes[i];
            if c == b'\\' {
                // skip escaped char (any)
                i = (i + 2).min(self.bytes.len());
                continue;
            }
            if c == quote {
                // reject SQL-standard quote-doubling.
                if self.bytes.get(i + 1) == Some(&quote) {
                    return Err(Status::invalid_argument("syntax error"));
                }
                let lit = self.input[start..i + 1].to_string();
                return Ok(self.token_to(TokenKind::StringLiteral(lit), start, i + 1));
            }
            i += 1;
        }
        Err(Status::invalid_argument("syntax error"))
    }

    fn lex_word(&mut self, start: usize) -> Token {
        let mut end = start + 1;
        while end < self.bytes.len() && is_regular_id_char(self.bytes[end]) {
            end += 1;
        }
        let raw = &self.input[start..end];
        let kind = match keyword_from_upper(raw.to_ascii_uppercase().as_str()) {
            Some(kw) => TokenKind::Keyword(kw),
            None => TokenKind::Ident(raw.to_string()),
        };
        self.token_to(kind, start, end)
    }

    fn scan_vector_literal(&self, start: usize) -> Option<usize> {
        let mut i = start + 1;
        while i < self.bytes.len() {
            let c = self.bytes[i];
            if c == b']' {
                return Some(i + 1);
            }
            // Reject nested vectors/matrices at this lexer level.
            if c == b'[' {
                return None;
            }
            if !(c.is_ascii_whitespace() || c.is_ascii_digit() || matches!(c, b'-' | b'.' | b',')) {
                return None;
            }
            i += 1;
        }
        None
    }
}

impl Symbol {
    fn width(self) -> usize {
        match self {
            Symbol::Ne | Symbol::Le | Symbol::Ge => 2,
            _ => 1,
        }
    }
}

/// End of a numeric token and whether it lexes as FLOAT rather than INTEGER.
pub(in crate::sqlengine) struct NumericToken {
    pub(in crate::sqlengine) end: usize,
    pub(in crate::sqlengine) is_float: bool,
}

/// Scans an INTEGER/FLOAT token (SQLLexer.g4) at `start` over bytes or chars.
///
/// - INTEGER: MINUS_SIGN? UNSIGNED_INTEGER
/// - FLOAT:   MINUS_SIGN? FLOAT_FRAGMENT ('E' ('+'|'-')? (FLOAT_FRAGMENT | UNSIGNED_INTEGER_FRAGMENT))? ('D' | 'F')?
///
/// REGULAR_ID wins on the longer match, so a token followed by a REGULAR_ID char is not numeric.
pub(in crate::sqlengine) fn scan_numeric_token<T: Copy + Into<char>>(
    input: &[T],
    start: usize,
) -> Option<NumericToken> {
    let at = |i: usize| input.get(i).map(|&c| c.into());
    let mut i = start;
    if at(i) == Some('-') {
        i += 1;
    }
    if i >= input.len() {
        return None;
    }

    // Base FLOAT_FRAGMENT: UNSIGNED_INTEGER* '.'? UNSIGNED_INTEGER+
    let int_end = scan_digits(input, i);
    let saw_leading_digits = int_end > i;
    let mantissa_end = scan_fraction(input, int_end)?;
    let saw_dot = mantissa_end > int_end;
    if !saw_dot && !saw_leading_digits {
        return None;
    }

    let exp_end = scan_exponent(input, mantissa_end)?;
    let saw_exp = exp_end > mantissa_end;

    let mut end = exp_end;
    let saw_suffix = matches!(at(end), Some('d' | 'D' | 'f' | 'F'));
    if saw_suffix {
        end += 1;
    }

    if at(end).is_some_and(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
        return None;
    }

    // Prefer INTEGER when there is no float/exponent/suffix and the token
    // started with at least one digit (avoid treating `.1` as int).
    Some(NumericToken {
        end,
        is_float: saw_dot || saw_exp || saw_suffix || !saw_leading_digits,
    })
}

fn scan_digits<T: Copy + Into<char>>(input: &[T], mut i: usize) -> usize {
    while i < input.len() && input[i].into().is_ascii_digit() {
        i += 1;
    }
    i
}

/// Optional `'.' digits+`; `None` when a dot has no digits after it (`1.`, `-.`).
fn scan_fraction<T: Copy + Into<char>>(input: &[T], i: usize) -> Option<usize> {
    if input.get(i).map(|&c| c.into()) != Some('.') {
        return Some(i);
    }
    let end = scan_digits(input, i + 1);
    (end > i + 1).then_some(end)
}

/// Optional exponent: `e[+-]?` then FLOAT_FRAGMENT or UNSIGNED_INTEGER_FRAGMENT.
fn scan_exponent<T: Copy + Into<char>>(input: &[T], i: usize) -> Option<usize> {
    if !matches!(input.get(i).map(|&c| c.into()), Some('e' | 'E')) {
        return Some(i);
    }
    let mut exp_start = i + 1;
    if matches!(input.get(exp_start).map(|&c| c.into()), Some('+' | '-')) {
        exp_start += 1;
    }
    let end = scan_fraction(input, scan_digits(input, exp_start))?;
    (end > exp_start).then_some(end)
}

fn is_regular_id_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'-'
}

impl Keyword {
    /// `regular_id` includes some keywords in identifier positions (SQLParser.g4).
    pub(super) fn can_be_identifier(self) -> bool {
        matches!(
            self,
            Keyword::Or
                | Keyword::And
                | Keyword::Not
                | Keyword::In
                | Keyword::Between
                | Keyword::Like
                | Keyword::Where
                | Keyword::Select
                | Keyword::As
                | Keyword::By
                | Keyword::Order
                | Keyword::Asc
                | Keyword::Desc
                | Keyword::Limit
        )
    }
}

/// Whether `word` is a keyword that may appear in identifier positions.
pub(in crate::sqlengine) fn is_identifier_keyword(word: &str) -> bool {
    keyword_from_upper(word.to_ascii_uppercase().as_str()).is_some_and(Keyword::can_be_identifier)
}

fn keyword_from_upper(s: &str) -> Option<Keyword> {
    Some(match s {
        "OR" => Keyword::Or,
        "AND" => Keyword::And,
        "NOT" => Keyword::Not,
        "IN" => Keyword::In,
        "CONTAIN_ALL" => Keyword::ContainAll,
        "CONTAIN_ANY" => Keyword::ContainAny,
        "BETWEEN" => Keyword::Between,
        "LIKE" => Keyword::Like,
        "WHERE" => Keyword::Where,
        "SELECT" => Keyword::Select,
        "FROM" => Keyword::From,
        "AS" => Keyword::As,
        "ORDER" => Keyword::Order,
        "BY" => Keyword::By,
        "ASC" => Keyword::Asc,
        "DESC" => Keyword::Desc,
        "LIMIT" => Keyword::Limit,
        "TRUE" => Keyword::True,
        "FALSE" => Keyword::False,
        "IS" => Keyword::Is,
        "NULL" => Keyword::Null,
        _ => return None,
    })
}

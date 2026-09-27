use super::*;
use crate::sqlengine::query::{is_identifier_keyword, scan_numeric_token};

fn syntax_error() -> Status {
    Status::invalid_argument("syntax error")
}

fn next_is(chars: &[char], i: usize, want: char) -> bool {
    chars.get(i + 1) == Some(&want)
}

fn scan_while(chars: &[char], mut i: usize, pred: impl Fn(char) -> bool) -> usize {
    while i < chars.len() && pred(chars[i]) {
        i += 1;
    }
    i
}

fn scan_back_while(chars: &[char], mut i: usize, pred: impl Fn(char) -> bool) -> usize {
    while i > 0 && pred(chars[i - 1]) {
        i -= 1;
    }
    i
}

fn skip_whitespace(chars: &[char], i: usize) -> usize {
    scan_while(chars, i, |c| c.is_ascii_whitespace())
}

fn prev_significant_char(chars: &[char], start: usize) -> Option<char> {
    let j = scan_back_while(chars, start, |c| c.is_ascii_whitespace());
    j.checked_sub(1).map(|p| chars[p])
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum SyntaxScanMode {
    Normal,
    Single,
    Double,
    LineComment,
    BlockComment,
}

struct SyntaxScan<'a> {
    chars: &'a [char],
    mode: SyntaxScanMode,
    square_depth: usize,
}

impl SyntaxScan<'_> {
    /// Scans one position in `Normal` mode and returns the next position.
    fn step_normal(&mut self, i: usize) -> ZResult<usize> {
        let chars = self.chars;
        match chars[i] {
            '-' => {
                // `--` starts a single-line comment and must be ignored.
                if next_is(chars, i, '-') {
                    self.mode = SyntaxScanMode::LineComment;
                    return Ok(i + 2);
                }
                // '-' is only valid inside identifiers (REGULAR_ID includes '-')
                // or as part of a numeric token (`-1`, `-1.2`, `-1e-3`). The parser grammar does
                // not accept a standalone MINUS_SIGN token, so `- 1` / `-(1)` are syntax errors.
                let standalone_minus = chars
                    .get(i + 1)
                    .is_some_and(|&n| n.is_ascii_whitespace() || n == '(');
                if self.square_depth == 0 && standalone_minus {
                    return Err(syntax_error());
                }
            }
            // `/* ... */` is a multi-line comment and must be ignored.
            '/' if next_is(chars, i, '*') => {
                self.mode = SyntaxScanMode::BlockComment;
                return Ok(i + 2);
            }
            '[' => self.square_depth += 1,
            ']' => self.square_depth = self.square_depth.saturating_sub(1),
            '\'' => {
                self.mode = SyntaxScanMode::Single;
                return Ok(i + 1);
            }
            '"' => {
                self.mode = SyntaxScanMode::Double;
                return Ok(i + 1);
            }
            '`' => return Err(syntax_error()),
            '+' => {
                // `+` is not an operator in filters; the only valid
                // use is as part of an exponent token (e.g. `1e+3`).
                let is_exponent_plus =
                    i >= 2 && matches!(chars[i - 1], 'e' | 'E') && chars[i - 2].is_ascii_digit();
                if !is_exponent_plus {
                    return Err(syntax_error());
                }
            }
            '=' if next_is(chars, i, '=') => return Err(syntax_error()),
            '<' if next_is(chars, i, '>') => return Err(syntax_error()),
            c if c.is_ascii_alphabetic() || c == '_' => return reject_word_syntax(chars, i),
            _ => {}
        }
        Ok(i + 1)
    }

    /// Scans one position inside a string literal closed by `quote`.
    fn step_quoted(&mut self, i: usize, quote: char) -> ZResult<usize> {
        let c = self.chars[i];
        if c == '\\' {
            // Skip escaped character.
            return Ok((i + 2).min(self.chars.len()));
        }
        if c == quote {
            // string literals only support backslash escapes
            // (SQUOTA_STRING / DQUOTA_STRING do not accept SQL-standard doubled quotes).
            if next_is(self.chars, i, quote) {
                return Err(syntax_error());
            }
            self.mode = SyntaxScanMode::Normal;
        }
        Ok(i + 1)
    }

    fn step_comment(&mut self, i: usize) -> usize {
        let c = self.chars[i];
        if self.mode == SyntaxScanMode::LineComment && (c == '\n' || c == '\r') {
            self.mode = SyntaxScanMode::Normal;
        }
        if self.mode == SyntaxScanMode::BlockComment && c == '*' && next_is(self.chars, i, '/') {
            self.mode = SyntaxScanMode::Normal;
            return i + 2;
        }
        i + 1
    }
}

pub(super) fn reject_unsupported_syntax(sql: &str) -> ZResult<()> {
    let chars: Vec<char> = sql.chars().collect();
    let mut scan = SyntaxScan {
        chars: &chars,
        mode: SyntaxScanMode::Normal,
        square_depth: 0,
    };
    let mut i = 0usize;
    while i < chars.len() {
        i = match scan.mode {
            SyntaxScanMode::Normal => scan.step_normal(i)?,
            SyntaxScanMode::Single => scan.step_quoted(i, '\'')?,
            SyntaxScanMode::Double => scan.step_quoted(i, '"')?,
            SyntaxScanMode::LineComment | SyntaxScanMode::BlockComment => scan.step_comment(i),
        };
    }

    // unterminated tokens are lexer errors. Surface them
    // as a stable `syntax error` rather than depending on sqlparser behavior.
    match scan.mode {
        SyntaxScanMode::Normal | SyntaxScanMode::LineComment => Ok(()),
        SyntaxScanMode::Single | SyntaxScanMode::Double | SyntaxScanMode::BlockComment => {
            Err(syntax_error())
        }
    }
}

/// Rejects word-level syntax at `start` and returns the position after the word.
fn reject_word_syntax(chars: &[char], start: usize) -> ZResult<usize> {
    let end = scan_while(chars, start + 1, |c| c.is_ascii_alphanumeric() || c == '_');
    let word: String = chars[start..end].iter().collect();
    let word = word.to_ascii_lowercase();
    if word == "in" {
        reject_empty_in_list(chars, end)?;
    }
    if word == "contain_any" || word == "contain_all" {
        reject_contain_function_call(chars, start, end)?;
    }
    Ok(end)
}

/// `IN ()` is a syntax error (in_value_expr_list is non-empty; sqlparser may accept empty lists).
fn reject_empty_in_list(chars: &[char], word_end: usize) -> ZResult<()> {
    let k = skip_whitespace(chars, word_end);
    if chars.get(k) != Some(&'(') {
        return Ok(());
    }
    let k = skip_whitespace(chars, k + 1);
    if chars.get(k) == Some(&')') {
        return Err(syntax_error());
    }
    Ok(())
}

/// CONTAIN_* is an infix operator (`<ident> CONTAIN_ANY (...)`), not a function call.
fn reject_contain_function_call(chars: &[char], start: usize, end: usize) -> ZResult<()> {
    let k = skip_whitespace(chars, end);
    if chars.get(k) != Some(&'(') {
        return Ok(());
    }
    let looks_like_infix = matches!(
        prev_significant_char(chars, start),
        Some(ch) if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-'
    );
    if looks_like_infix {
        // Infix operator form supported; Finch rewrites it later.
        return Ok(());
    }
    Err(syntax_error())
}

pub(super) fn rewrite_contain_ops(input: &str) -> String {
    // Compatibility: accept infix syntax:
    //   field CONTAIN_ANY (v1, v2)
    //   field NOT CONTAIN_ALL (v1)
    // sqlparser parses function calls more reliably, so rewrite to:
    //   contain_any(field, v1, v2)
    //   NOT contain_all(field, v1)
    let re = RegexBuilder::new(
        r"(?i)(`[^`]+`|[A-Za-z0-9_][A-Za-z0-9_\.\-]*)\s+(NOT\s+)?(CONTAIN_ALL|CONTAIN_ANY)\s*\(",
    )
    .build()
    .expect("valid contain rewrite regex");
    let quoted = quoted_ranges(input);

    re.replace_all(input, |caps: &regex::Captures<'_>| {
        let m = caps.get(0).expect("match exists");
        if quoted.iter().any(|range| range.contains(&m.start())) {
            return m.as_str().to_string();
        }
        let field = caps.get(1).map(|m| m.as_str()).unwrap_or_default();
        let has_not = caps.get(2).is_some();
        let func = caps
            .get(3)
            .map(|m| m.as_str().to_ascii_lowercase())
            .unwrap_or_else(|| "contain_any".to_string());
        let mut i = m.end();
        while i < input.len() && input.as_bytes()[i].is_ascii_whitespace() {
            i += 1;
        }
        let is_empty_list = i < input.len() && input.as_bytes()[i] == b')';
        if has_not {
            if is_empty_list {
                format!("NOT {}({}", func, field)
            } else {
                format!("NOT {}({}, ", func, field)
            }
        } else if is_empty_list {
            format!("{}({}", func, field)
        } else {
            format!("{}({}, ", func, field)
        }
    })
    .to_string()
}

/// Byte ranges strictly inside quoted spans, whose text is data rather than operators.
fn quoted_ranges(input: &str) -> Vec<std::ops::Range<usize>> {
    let mut ranges = Vec::new();
    let mut open: Option<(char, usize)> = None;
    let mut chars = input.char_indices();
    while let Some((i, c)) = chars.next() {
        match open {
            None if matches!(c, '\'' | '"' | '`') => open = Some((c, i + 1)),
            None => {}
            // Backtick identifiers have no escapes; string literals skip the escaped character.
            Some((q, _)) if c == '\\' && q != '`' => {
                chars.next();
            }
            Some((q, start)) if c == q => {
                ranges.push(start..i);
                open = None;
            }
            Some(_) => {}
        }
    }
    if let Some((_, start)) = open {
        ranges.push(start..input.len());
    }
    ranges
}

fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.')
}

fn is_word_char(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

/// Lowercased word (letters/underscores) that ends just before the punctuation preceding `j`.
fn preceding_word(chars: &[char], j: usize) -> Option<String> {
    // Skip punctuation like ')', '=', etc.
    let p = scan_back_while(chars, j, |c| !is_word_char(c) && !c.is_ascii_whitespace());
    let w_start = scan_back_while(chars, p, is_word_char);
    if w_start < p {
        let w: String = chars[w_start..p].iter().collect();
        Some(w.to_ascii_lowercase())
    } else {
        None
    }
}

fn looks_like_identifier_context(chars: &[char], start: usize, end: usize) -> bool {
    // Best-effort context detection:
    // - LHS of relation expr: <ident> (=|!=|<|<=|>|>=|IN|LIKE|IS|CONTAIN_*) ...
    // - Function arg ident: func( <ident> , ...) or func( <ident> )
    let j = scan_back_while(chars, start, |c| c.is_ascii_whitespace());
    let prev_sig = j.checked_sub(1).map(|p| chars[p]);
    let k = skip_whitespace(chars, end);
    let next_sig = chars.get(k).copied();

    // Function arg: (ident, ...) or (ident)
    if matches!(prev_sig, None | Some('(') | Some(',')) && matches!(next_sig, Some(',') | Some(')'))
    {
        return true;
    }

    // Keyword identifiers are only possible at the start of a relation expr,
    // not in operator positions like `age NOT IN (...)`; the previous word
    // distinguishes keyword operators like `NOT IN` from identifiers like `not = 1`.
    let start_of_subexpr = matches!(prev_sig, None | Some('(') | Some(','))
        || matches!(
            preceding_word(chars, j).as_deref(),
            Some("and") | Some("or")
        );
    if !start_of_subexpr {
        return false;
    }

    // LHS: followed by a comparison operator char.
    if matches!(next_sig, Some('=') | Some('<') | Some('>') | Some('!')) {
        return true;
    }

    // LHS: followed by keyword operators (IN/LIKE/IS/CONTAIN_*).
    if !matches!(next_sig, Some(c) if is_word_char(c)) {
        return false;
    }
    let w_end = scan_while(chars, k, is_word_char);
    let word: String = chars[k..w_end].iter().collect();
    matches!(
        word.to_ascii_lowercase().as_str(),
        "in" | "like" | "is" | "contain_all" | "contain_any"
    )
}

fn should_quote_ident(token: &str) -> bool {
    let Some(first) = token.chars().next() else {
        return false;
    };
    let has_alpha_or_underscore = token.chars().any(|c| c.is_ascii_alphabetic() || c == '_');
    if !has_alpha_or_underscore {
        // Pure numbers (or punctuation) are literals, not identifiers.
        return false;
    }
    first.is_ascii_digit() || token.contains('-')
}

/// Copies one unquoted token starting at `i` into `out`, quoting loose identifiers.
fn copy_unquoted_token(chars: &[char], i: usize, out: &mut String) -> usize {
    if let Some(token) = scan_numeric_token(chars, i) {
        out.extend(&chars[i..token.end]);
        return token.end;
    }
    if !is_ident_char(chars[i]) {
        out.push(chars[i]);
        return i + 1;
    }
    let end = scan_while(chars, i + 1, is_ident_char);
    let token: String = chars[i..end].iter().collect();
    // SQL keywords can be used as identifiers (regular_id in SQLParser.g4) but
    // sqlparser rejects them, so quote keywords only in identifier positions.
    let quote = should_quote_ident(&token)
        || (is_identifier_keyword(&token) && looks_like_identifier_context(chars, i, end));
    if quote {
        out.push('`');
        out.push_str(&token);
        out.push('`');
    } else {
        out.push_str(&token);
    }
    end
}

/// allow "loose" field identifiers in filters (not strict SQL identifiers),
/// notably leading digits and `-` (dash) characters.
///
/// sqlparser parses these as arithmetic unless quoted, so we normalize them to
/// backtick-quoted identifiers (MySQL-style) outside of string literals.
pub(super) fn normalize_reference_identifier_quoting(sql: &str) -> String {
    let chars: Vec<char> = sql.chars().collect();
    let mut out = String::with_capacity(sql.len());
    // Closing quote of the literal being copied, if any.
    let mut open_quote: Option<char> = None;
    let mut i = 0usize;

    while i < chars.len() {
        let c = chars[i];
        let Some(q) = open_quote else {
            if matches!(c, '\'' | '"' | '`') {
                out.push(c);
                open_quote = Some(c);
                i += 1;
            } else {
                i = copy_unquoted_token(&chars, i, &mut out);
            }
            continue;
        };
        out.push(c);
        i += 1;
        // Backtick identifiers have no escapes; string literals keep `\x` pairs intact.
        if c == '\\' && q != '`' && i < chars.len() {
            out.push(chars[i]);
            i += 1;
        } else if c == q {
            open_quote = None;
        }
    }

    out
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum EscapeMode {
    Normal,
    Single,
    DoubleToSingle,
}

struct EscapeRewriter<'a> {
    chars: &'a [char],
    out: String,
    mode: EscapeMode,
}

impl EscapeRewriter<'_> {
    fn next_is(&self, i: usize, want: char) -> bool {
        next_is(self.chars, i, want)
    }

    /// A backslash run that reaches `quote` reads in pairs like the lexer, so a literal can end in a backslash.
    fn step_backslash_run_before_quote(&mut self, i: usize, quote: char) -> Option<usize> {
        let end = scan_while(self.chars, i, |c| c == '\\');
        let run = end - i;
        if run < 2 || self.chars.get(end) != Some(&quote) {
            return None;
        }
        if run.is_multiple_of(2) {
            self.out.push_str(&"\\\\".repeat(run / 2));
            return Some(end);
        }
        // One extra backslash because string conversion folds `\'` and `\"` into the bare quote.
        self.out.push_str(&"\\\\".repeat(run / 2 + 1));
        self.out.push_str(if quote == '\'' { "''" } else { "\"" });
        Some(end + 1)
    }

    fn step_normal(&mut self, i: usize) -> usize {
        let c = self.chars[i];
        if c == '\'' {
            self.out.push('\'');
            self.mode = EscapeMode::Single;
        } else if c == '"' {
            // treat `"`-quoted strings as string literals, but normalize to
            // single quotes so sqlparser's GenericDialect can parse them as values.
            self.out.push('\'');
            self.mode = EscapeMode::DoubleToSingle;
        } else {
            self.out.push(c);
        }
        i + 1
    }

    fn step_single(&mut self, i: usize) -> usize {
        let c = self.chars[i];
        if c == '\\' {
            if let Some(next) = self.step_backslash_run_before_quote(i, '\'') {
                return next;
            }
            if self.next_is(i, '\'') {
                self.out.push_str("''");
                return i + 2;
            }
            // preserve backslashes in string values (except for explicit quote escapes).
            // MySqlDialect will unescape backslashes in string literals, so emit `\\` to keep a
            // literal `\` in the parsed value; `\"` stays `\"` so it unescapes to `"`.
            if self.next_is(i, '"') {
                self.out.push('\\');
            } else {
                self.out.push_str("\\\\");
            }
            return i + 1;
        }
        if c == '\'' {
            if self.next_is(i, '\'') {
                self.out.push_str("''");
                return i + 2;
            }
            self.out.push('\'');
            self.mode = EscapeMode::Normal;
            return i + 1;
        }
        self.out.push(c);
        i + 1
    }

    /// Inside an original `"..."` string while emitting SQL single-quoted output.
    fn step_double_to_single(&mut self, i: usize) -> usize {
        let c = self.chars[i];
        if c == '\\' {
            if let Some(next) = self.step_backslash_run_before_quote(i, '"') {
                return next;
            }
            if self.next_is(i, '"') {
                self.out.push('"');
                return i + 2;
            }
            if self.next_is(i, '\'') {
                // `\'` is a literal `'` in the source; escape it for single-quoted output.
                self.out.push_str("''");
                return i + 2;
            }
            // Preserve other backslash sequences through MySqlDialect string parsing.
            self.out.push_str("\\\\");
            return i + 1;
        }
        if c == '\'' {
            // Escape embedded single quote in single-quoted output.
            self.out.push_str("''");
            return i + 1;
        }
        if c == '"' {
            if self.next_is(i, '"') {
                // `""` inside a `"`-quoted string means a literal `"`.
                self.out.push('"');
                return i + 2;
            }
            // End of original double-quoted literal.
            self.out.push('\'');
            self.mode = EscapeMode::Normal;
            return i + 1;
        }
        self.out.push(c);
        i + 1
    }
}

/// allow C-style escaping for quotes inside SQL string literals:
/// - in single-quoted strings: `\'` is treated as a literal `'` (rewritten as `''`)
/// - in double-quoted strings: `\"` is treated as a literal `"` and the string is
///   normalized to a single-quoted SQL string (double-quoted strings are also supported)
///
/// Other backslash sequences (notably `\%` / `\_` for LIKE) are preserved.
pub(super) fn normalize_reference_string_escapes(sql: &str) -> String {
    let chars: Vec<char> = sql.chars().collect();
    let mut rewriter = EscapeRewriter {
        chars: &chars,
        out: String::with_capacity(sql.len()),
        mode: EscapeMode::Normal,
    };
    let mut i = 0usize;
    while i < chars.len() {
        i = match rewriter.mode {
            EscapeMode::Normal => rewriter.step_normal(i),
            EscapeMode::Single => rewriter.step_single(i),
            EscapeMode::DoubleToSingle => rewriter.step_double_to_single(i),
        };
    }
    rewriter.out
}

use super::*;

pub(super) const MAX_IN_LIST_LEN: usize = 20_000;

pub(super) fn convert_function_expr(func: &sqlparser::ast::Function) -> ZResult<FilterExpr> {
    let name = func.name.to_string().to_ascii_lowercase();
    let args = extract_function_args(func)?;

    match name.as_str() {
        "contain_all" | "contain_any" => {
            if args.is_empty() {
                return Err(Status::invalid_argument(format!(
                    "{} requires at least the field argument",
                    name
                )));
            }
            let field = extract_field_name(args[0])?;
            let values = args[1..]
                .iter()
                .map(|e| convert_value(e))
                .collect::<ZResult<Vec<_>>>()?;

            if name == "contain_all" {
                Ok(FilterExpr::ContainAll { field, values })
            } else {
                Ok(FilterExpr::ContainAny { field, values })
            }
        }
        _ => Err(Status::invalid_argument(format!(
            "Function is not supported. {}",
            name
        ))),
    }
}

pub(super) fn extract_function_args(func: &sqlparser::ast::Function) -> ZResult<Vec<&Expr>> {
    use sqlparser::ast::{FunctionArg, FunctionArgExpr, FunctionArguments};

    let FunctionArguments::List(list) = &func.args else {
        return Err(Status::invalid_argument("syntax error"));
    };
    list.args
        .iter()
        .map(|arg| match arg {
            FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => Ok(e),
            _ => Err(Status::invalid_argument("syntax error")),
        })
        .collect()
}

pub(super) fn convert_expr(expr: &Expr) -> ZResult<FilterExpr> {
    match expr {
        Expr::BinaryOp { left, op, right } => convert_binary_op(left, op, right),
        Expr::UnaryOp {
            op: UnaryOperator::Not,
            expr,
        } => convert_not_expr(expr),
        // only permits `ID IS NULL`.
        Expr::IsNull(e) => Ok(FilterExpr::IsNull(extract_identifier_operand(e)?)),
        // only permits `ID IS NOT NULL`.
        Expr::IsNotNull(e) => Ok(FilterExpr::IsNotNull(extract_identifier_operand(e)?)),
        Expr::Like {
            expr,
            pattern,
            negated,
            ..
        } => convert_like_expr(expr, pattern, *negated),
        Expr::InList {
            expr,
            list,
            negated,
        } => convert_in_list_expr(expr, list, *negated),
        Expr::Function(func) => convert_function_expr(func),
        Expr::Nested(inner) => convert_expr(inner),
        _ => Err(Status::invalid_argument("syntax error")),
    }
}

/// Field name of an operand that must be a bare identifier.
fn extract_identifier_operand(expr: &Expr) -> ZResult<String> {
    let Expr::Identifier(_) = expr else {
        return Err(Status::invalid_argument("syntax error"));
    };
    extract_field_name(expr)
}

fn convert_not_expr(expr: &Expr) -> ZResult<FilterExpr> {
    // the SQL dialect does not support a general
    // unary `NOT` operator; negation is only supported via:
    // - `NOT IN`
    // - `NOT CONTAIN_*`
    // - `IS NOT NULL`
    //
    // We still accept unary NOT here only for the internal rewrite of
    // `field NOT CONTAIN_* (...)` into `NOT contain_*(field, ...)`.
    let Expr::Function(func) = expr else {
        return Err(Status::invalid_argument("syntax error"));
    };
    let name = func.name.to_string().to_ascii_lowercase();
    if !matches!(name.as_str(), "contain_all" | "contain_any") {
        return Err(Status::invalid_argument("syntax error"));
    }
    Ok(FilterExpr::Not(Box::new(convert_expr(expr)?)))
}

fn convert_like_expr(expr: &Expr, pattern: &Expr, negated: bool) -> ZResult<FilterExpr> {
    if negated {
        // upstream grammar does not support `NOT LIKE`.
        return Err(Status::invalid_argument("syntax error"));
    }
    // treat non-identifier LIKE operands as syntax errors.
    // Example: `array_length(tags) like '%'`.
    let field = extract_identifier_operand(expr)?;
    let pat = match pattern {
        Expr::Value(SqlValue::SingleQuotedString(p))
        | Expr::Value(SqlValue::DoubleQuotedString(p)) => p.clone(),
        _ => {
            return Err(Status::invalid_argument(
                "LIKE pattern must be a string literal",
            ));
        }
    };
    Ok(like_pattern_to_filter_expr(field, pat))
}

fn convert_in_list_expr(expr: &Expr, list: &[Expr], negated: bool) -> ZResult<FilterExpr> {
    // only permits `ID [NOT] IN (...)`.
    let field = extract_identifier_operand(expr)?;
    // `IN ()` is a syntax error (in_value_expr_list is non-empty).
    if list.is_empty() {
        return Err(Status::invalid_argument("syntax error"));
    }
    if list.len() > MAX_IN_LIST_LEN {
        return Err(Status::invalid_argument(format!(
            "In rel expr only support list size no more than {}",
            MAX_IN_LIST_LEN
        )));
    }
    let mut values = Vec::with_capacity(list.len());
    for v in list {
        values.push(convert_value(v)?);
    }
    Ok(FilterExpr::InList {
        field,
        values,
        negated,
    })
}

pub(super) fn convert_binary_op(
    left: &Expr,
    op: &BinaryOperator,
    right: &Expr,
) -> ZResult<FilterExpr> {
    match op {
        BinaryOperator::And => Ok(FilterExpr::And(
            Box::new(convert_expr(left)?),
            Box::new(convert_expr(right)?),
        )),
        BinaryOperator::Or => Ok(FilterExpr::Or(
            Box::new(convert_expr(left)?),
            Box::new(convert_expr(right)?),
        )),
        BinaryOperator::Eq => compare_expr(left, right, CompareOp::Equal),
        BinaryOperator::NotEq => compare_expr(left, right, CompareOp::NotEqual),
        BinaryOperator::Lt => compare_expr(left, right, CompareOp::LessThan),
        BinaryOperator::LtEq => compare_expr(left, right, CompareOp::LessEqual),
        BinaryOperator::Gt => compare_expr(left, right, CompareOp::GreaterThan),
        BinaryOperator::GtEq => compare_expr(left, right, CompareOp::GreaterEqual),
        _ => Err(Status::invalid_argument("syntax error")),
    }
}

pub(super) fn try_extract_array_length_field(expr: &Expr) -> ZResult<Option<String>> {
    use sqlparser::ast::{FunctionArg, FunctionArgExpr, FunctionArguments};

    let Expr::Function(func) = expr else {
        return Ok(None);
    };
    if !func.name.to_string().eq_ignore_ascii_case("array_length") {
        return Ok(None);
    }

    let args = match &func.args {
        FunctionArguments::List(list) => &list.args,
        _ => {
            return Err(Status::invalid_argument(
                "array_length function should have only one argument. ",
            ));
        }
    };
    if args.len() != 1 {
        return Err(Status::invalid_argument(
            "array_length function should have only one argument. ",
        ));
    }
    let FunctionArg::Unnamed(FunctionArgExpr::Expr(inner)) = &args[0] else {
        return Err(Status::invalid_argument(
            "array_length function argument must be a field name",
        ));
    };
    let Expr::Identifier(_) = inner else {
        // include the stable prefix, but avoid leaking a sqlparser AST debug dump.
        return Err(Status::invalid_argument(
            "array_length function argument must be a field name",
        ));
    };
    Ok(Some(extract_field_name(inner)?))
}

pub(super) fn parse_array_length_rhs_u32(expr: &Expr) -> Option<u32> {
    // array_length RHS must be an integer literal token.
    // Treat everything else (floats, functions, identifiers, negative numbers) as invalid.
    match expr {
        Expr::Value(SqlValue::Number(raw, _)) => {
            if raw.chars().all(|c| c.is_ascii_digit()) {
                raw.parse::<u64>().ok().and_then(|v| u32::try_from(v).ok())
            } else {
                None
            }
        }
        _ => None,
    }
}

pub(super) fn compare_expr(left: &Expr, right: &Expr, op: CompareOp) -> ZResult<FilterExpr> {
    // parentheses are only valid around whole sub-expressions
    // (`(a = 1) AND (b = 2)`), not around identifiers/values inside relation exprs.
    if matches!(left, Expr::Nested(_)) || matches!(right, Expr::Nested(_)) {
        return Err(Status::invalid_argument("syntax error"));
    }

    // array_length(field) op <int>
    if let Some(field) = try_extract_array_length_field(left)? {
        let Some(len) = parse_array_length_rhs_u32(right) else {
            // `QueryInfoTest.ArrayLengthInvalidArgument`
            return Err(Status::invalid_argument(
                "array_length right side only support integer",
            ));
        };
        return Ok(FilterExpr::ArrayLengthCompare { field, op, len });
    }

    // relation expr LHS is either a bare identifier or a function call.
    let field = match left {
        Expr::Identifier(id) => id.value.clone(),
        Expr::Function(func) => {
            return Err(Status::invalid_argument(format!(
                "Function is not supported. {}",
                func.name
            )));
        }
        _ => {
            return Err(Status::invalid_argument("syntax error"));
        }
    };

    match right {
        Expr::Identifier(_) => return Err(Status::invalid_argument("syntax error")),
        Expr::Function(func) => {
            return Err(Status::invalid_argument(format!(
                "Function is not supported. {}",
                func.name
            )));
        }
        Expr::Value(_) => {}
        Expr::UnaryOp {
            op: UnaryOperator::Minus,
            expr,
        } => {
            // numeric tokens can include a leading '-' (INTEGER/FLOAT). sqlparser
            // represents `-1` as a unary op, so accept `-(number)` as a numeric literal.
            let Expr::Value(SqlValue::Number(_, _)) = expr.as_ref() else {
                return Err(Status::invalid_argument("syntax error"));
            };
        }
        _ => return Err(Status::invalid_argument("syntax error")),
    }

    let value = convert_value(right)?;

    Ok(FilterExpr::Compare { field, op, value })
}

pub(super) fn extract_field_name(expr: &Expr) -> ZResult<String> {
    match expr {
        Expr::Identifier(id) => {
            // identifiers cannot contain '.' (dot) characters.
            // Dots must not be permitted even if sqlparser treats them as a single
            // identifier (e.g. backtick-quoted) because the dialect has no identifier-quoting.
            if id.value.contains('.') {
                return Err(Status::invalid_argument("syntax error"));
            }
            Ok(id.value.clone())
        }
        _ => Err(Status::invalid_argument("syntax error")),
    }
}

pub(super) fn convert_value(expr: &Expr) -> ZResult<Value> {
    match expr {
        Expr::Value(v) => match v {
            SqlValue::Number(n, _) => convert_number(n),
            SqlValue::SingleQuotedString(s) | SqlValue::DoubleQuotedString(s) => {
                // treat \" and \' inside SQL strings as escaped quotes.
                let s = s.replace("\\\"", "\"").replace("\\'", "'");
                Ok(Value::String(s))
            }
            SqlValue::Boolean(b) => Ok(Value::Bool(*b)),
            // NULL is not a constant in relation expressions; only
            // `IS NULL` / `IS NOT NULL` are supported for NULL checks.
            SqlValue::Null => Err(Status::invalid_argument("syntax error")),
            _ => Err(Status::invalid_argument(format!(
                "unsupported value type: {:?}",
                v
            ))),
        },
        Expr::UnaryOp {
            op: UnaryOperator::Minus,
            expr,
        } => match convert_value(expr)? {
            Value::I64(v) => Ok(Value::I64(-v)),
            Value::U64(v) if v == (i64::MAX as u64) + 1 => Ok(Value::I64(i64::MIN)),
            Value::F64(v) => Ok(Value::F64(-v)),
            _ => Err(Status::invalid_argument(
                "negation only valid for numeric literals",
            )),
        },
        _ => Err(Status::invalid_argument(format!(
            "expected literal value, got: {:?}",
            expr
        ))),
    }
}

fn convert_number(n: &str) -> ZResult<Value> {
    // lexer:
    // - INT tokens: digits with optional leading '-'
    // - FLOAT tokens: decimal and/or scientific notation, optionally suffixed with 'D'/'F'
    let is_float = n.contains('.')
        || n.contains('e')
        || n.contains('E')
        || matches!(n.chars().last(), Some('d' | 'D' | 'f' | 'F'));

    if !is_float {
        return n
            .parse::<i64>()
            .map(Value::I64)
            .or_else(|_| n.parse::<u64>().map(Value::U64))
            .map_err(|_| Status::invalid_argument(format!("invalid integer: {}", n)));
    }
    let stripped = n
        .strip_suffix('d')
        .or_else(|| n.strip_suffix('D'))
        .or_else(|| n.strip_suffix('f'))
        .or_else(|| n.strip_suffix('F'))
        .unwrap_or(n);
    stripped
        .parse::<f64>()
        .map(Value::F64)
        .map_err(|_| Status::invalid_argument(format!("invalid number: {}", n)))
}

pub(super) fn like_pattern_to_filter_expr(field: String, pattern: String) -> FilterExpr {
    // treat `\` as an escape inside LIKE patterns (notably for `\%` / `\_`).
    fn scan_unescaped_wildcards(pat: &str) -> (Vec<usize>, Vec<usize>) {
        let bytes = pat.as_bytes();
        let mut percents: Vec<usize> = Vec::new();
        let mut underscores: Vec<usize> = Vec::new();
        let mut i = 0usize;
        while i < bytes.len() {
            match bytes[i] {
                b'\\' => {
                    i = (i + 2).min(bytes.len());
                }
                b'%' => {
                    percents.push(i);
                    i += 1;
                }
                b'_' => {
                    underscores.push(i);
                    i += 1;
                }
                _ => i += 1,
            }
        }
        (percents, underscores)
    }

    fn unescape_like_literal(pat: &str) -> String {
        let bytes = pat.as_bytes();
        let mut out = Vec::with_capacity(bytes.len());
        let mut i = 0usize;
        while i < bytes.len() {
            if bytes[i] == b'\\' {
                if i + 1 < bytes.len() {
                    out.push(bytes[i + 1]);
                    i += 2;
                } else {
                    // Trailing '\': keep it literally.
                    out.push(b'\\');
                    i += 1;
                }
            } else {
                out.push(bytes[i]);
                i += 1;
            }
        }
        String::from_utf8_lossy(&out).to_string()
    }

    let (percents, underscores) = scan_unescaped_wildcards(&pattern);

    // No unescaped wildcards => exact match.
    if percents.is_empty() && underscores.is_empty() {
        return FilterExpr::Compare {
            field,
            op: CompareOp::Equal,
            value: Value::String(unescape_like_literal(&pattern)),
        };
    }

    // Single unescaped '%' and no '_' => prefix/suffix optimization.
    if underscores.is_empty() && percents.len() == 1 {
        let loc = percents[0];
        if loc == pattern.len().saturating_sub(1) {
            let prefix = unescape_like_literal(&pattern[..loc]);
            return FilterExpr::HasPrefix { field, prefix };
        }
        if loc == 0 {
            let suffix = unescape_like_literal(&pattern[1..]);
            return FilterExpr::HasSuffix { field, suffix };
        }
    }

    FilterExpr::LikePattern { field, pattern }
}

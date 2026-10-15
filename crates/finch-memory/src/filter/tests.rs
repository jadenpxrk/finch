use super::*;
use finch_types::StatusCode;

#[test]
fn test_parse_simple() {
    let expr = parse_filter("age > 30").unwrap();
    if let FilterExpr::Compare { field, op, value } = &expr {
        assert_eq!(field, "age");
        assert_eq!(*op, CompareOp::GreaterThan);
        assert_eq!(*value, Value::I64(30));
    } else {
        panic!("expected Compare");
    }
}

#[test]
fn test_parse_i64_min_literal() {
    let expr = parse_filter("timestamp >= -9223372036854775808").unwrap();
    assert!(matches!(
        expr,
        FilterExpr::Compare {
            value: Value::I64(i64::MIN),
            ..
        }
    ));
}

#[test]
fn test_parse_and() {
    let expr = parse_filter("age > 30 AND name = 'alice'").unwrap();
    assert!(matches!(expr, FilterExpr::And(_, _)));
}

#[test]
fn test_parse_empty() {
    let expr = parse_filter("").unwrap();
    assert!(matches!(expr, FilterExpr::AlwaysTrue));
}

#[test]
fn test_parse_in_list() {
    let expr = parse_filter("age IN (10, 20)").unwrap();
    assert!(matches!(expr, FilterExpr::InList { .. }));
}

#[test]
fn test_parse_not_in_list() {
    let expr = parse_filter("age NOT IN (10, 20)").unwrap();
    if let FilterExpr::InList { negated, .. } = expr {
        assert!(negated);
    } else {
        panic!("expected InList");
    }
}

#[test]
fn test_general_unary_not_is_rejected() {
    let err = parse_filter("NOT (age = 10)").unwrap_err();
    assert!(err.to_string().contains("syntax error"));
}

#[test]
fn test_keyword_identifiers_parse_in_identifier_positions() {
    // SQL keywords can be used as identifiers; ensure we
    // normalize them so sqlparser accepts them.
    let expr = parse_filter("or = 1").unwrap();
    assert!(
        matches!(expr, FilterExpr::Compare { field, op: CompareOp::Equal, .. } if field == "or")
    );

    let expr = parse_filter("select != 2").unwrap();
    assert!(
        matches!(expr, FilterExpr::Compare { field, op: CompareOp::NotEqual, .. } if field == "select")
    );

    // Keyword ident inside a function call (via CONTAIN rewrite).
    let expr = parse_filter("or contain_any ('a')").unwrap();
    assert!(
        matches!(expr, FilterExpr::ContainAny { field, values } if field == "or" && values.len() == 1)
    );

    // Keyword ident inside array_length(...) function.
    let expr = parse_filter("array_length(and) >= 1").unwrap();
    assert!(
        matches!(expr, FilterExpr::ArrayLengthCompare { field, op: CompareOp::GreaterEqual, len: 1 } if field == "and")
    );
}

#[test]
fn test_parse_double_quoted_string_literals_are_accepted() {
    // normalize_string_escapes rewrites them so
    // sqlparser can parse under GenericDialect.
    let expr = parse_filter(r#"name IN ("user_1", "user_2")"#).unwrap();
    match expr {
        FilterExpr::InList {
            field,
            values,
            negated,
        } => {
            assert_eq!(field, "name");
            assert!(!negated);
            assert_eq!(values.len(), 2);
            assert!(matches!(&values[0], Value::String(s) if s == "user_1"));
            assert!(matches!(&values[1], Value::String(s) if s == "user_2"));
        }
        other => panic!("expected InList, got {:?}", other),
    }

    let expr = parse_filter(r#"category NOT IN ("a", "b", "c")"#).unwrap();
    match expr {
        FilterExpr::InList {
            field,
            values,
            negated,
        } => {
            assert_eq!(field, "category");
            assert!(negated);
            assert_eq!(values.len(), 3);
        }
        other => panic!("expected InList, got {:?}", other),
    }
}

#[test]
fn test_parse_contain_any_infix() {
    let expr = parse_filter("tags contain_any ('a', 'b')").unwrap();
    assert!(matches!(expr, FilterExpr::ContainAny { .. }));
}

#[test]
fn test_parse_contain_empty_list_infix_is_accepted() {
    let expr = parse_filter("tags contain_any ()").unwrap();
    assert!(matches!(expr, FilterExpr::ContainAny { .. }));
    let expr = parse_filter("tags contain_all ()").unwrap();
    assert!(matches!(expr, FilterExpr::ContainAll { .. }));
}

#[test]
fn test_parse_string_backslash_quote_escape() {
    let expr = parse_filter("name = 'it\\'s'").unwrap();
    assert!(matches!(
        expr,
        FilterExpr::Compare { field, op: CompareOp::Equal, value: Value::String(s) }
        if field == "name" && s == "it's"
    ));

    let expr = parse_filter("name = 'a\\\"b'").unwrap();
    assert!(matches!(
        expr,
        FilterExpr::Compare { field, op: CompareOp::Equal, value: Value::String(s) }
        if field == "name" && s == "a\"b"
    ));
}

#[test]
fn test_parse_string_backslash_run_before_quote_reads_in_pairs() {
    let cases = [
        (r"name = 'a\\'", r"a\"),
        (r#"name = "a\\""#, r"a\"),
        (r"name = 'a\\\\\'b'", r"a\\'b"),
        (r"name = 'a\\\'b'", r"a\'b"),
        (r"name = 'a\\b'", r"a\\b"),
    ];
    for (sql, want) in cases {
        let expr = parse_filter(sql).unwrap();
        assert!(
            matches!(&expr, FilterExpr::Compare { value: Value::String(s), .. } if s == want),
            "{sql} parsed as {expr:?}"
        );
    }
}

#[test]
fn test_parse_string_sql_standard_quote_doubling_is_rejected() {
    // only backslash escapes are supported inside strings.
    // SQL-standard quote doubling (e.g. 'it''s') is not accepted by the lexer.
    let err = parse_filter("name = 'it''s'").unwrap_err();
    assert!(err.to_string().contains("syntax error"));

    let err = parse_filter(r#"name = "a""b""#).unwrap_err();
    assert!(err.to_string().contains("syntax error"));
}

#[test]
fn test_parse_filter_ignores_single_line_comments() {
    // `-- ...` comments are ignored, and any dialect-shaper
    // checks must not fire on tokens that appear only inside comments.
    let expr = parse_filter("age = 1 -- this contains == and `backticks`\nAND name = 'x'").unwrap();
    assert!(matches!(expr, FilterExpr::And(_, _)));

    // EOF-terminated comment is also allowed.
    let expr = parse_filter("age = 1 -- trailing == is ignored").unwrap();
    assert!(matches!(expr, FilterExpr::Compare { .. }));
}

#[test]
fn test_parse_filter_ignores_block_comments() {
    let expr =
        parse_filter("age = 1 /* this contains == and `backticks` */ AND name = 'x'").unwrap();
    assert!(matches!(expr, FilterExpr::And(_, _)));
}

#[test]
fn test_parse_filter_unterminated_strings_and_comments_are_rejected() {
    // unterminated quote/comment is a lexer error; surface as syntax error.
    let err = parse_filter("name = 'unterminated").unwrap_err();
    assert!(err.to_string().contains("syntax error"));

    let err = parse_filter("name = \"unterminated").unwrap_err();
    assert!(err.to_string().contains("syntax error"));

    let err = parse_filter("age = 1 /* unterminated").unwrap_err();
    assert!(err.to_string().contains("syntax error"));
}

#[test]
fn test_like_pattern_escaped_wildcards_can_be_exact_or_prefix() {
    // Escaped '%' is literal => no wildcard => equality.
    let expr = parse_filter("name LIKE 'abc\\%'").unwrap();
    assert!(matches!(
        expr,
        FilterExpr::Compare { field, op: CompareOp::Equal, value: Value::String(s) }
        if field == "name" && s == "abc%"
    ));

    // Unescaped trailing '%' => prefix optimization.
    let expr = parse_filter("name LIKE 'abc%'").unwrap();
    assert!(matches!(
        expr,
        FilterExpr::HasPrefix { field, prefix }
        if field == "name" && prefix == "abc"
    ));
}

#[test]
fn test_parse_like_pattern() {
    let expr = parse_filter("name LIKE '%ali_%'").unwrap();
    assert!(matches!(expr, FilterExpr::LikePattern { .. }));
}

#[test]
fn test_parse_like_exact_becomes_eq() {
    let expr = parse_filter("name LIKE 'alice'").unwrap();
    assert!(matches!(
        expr,
        FilterExpr::Compare {
            op: CompareOp::Equal,
            value: Value::String(_),
            ..
        }
    ));
}

#[test]
fn test_parse_like_prefix_becomes_has_prefix() {
    let expr = parse_filter("name LIKE 'ali%'").unwrap();
    assert!(matches!(expr, FilterExpr::HasPrefix { .. }));
}

#[test]
fn test_parse_like_suffix_becomes_has_suffix() {
    let expr = parse_filter("name LIKE '%ice'").unwrap();
    assert!(matches!(expr, FilterExpr::HasSuffix { .. }));
}

#[test]
fn test_parse_has_prefix_function() {
    let err = parse_filter("has_prefix(name, 'ali')").unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument);
}

#[test]
fn test_parse_has_suffix_function() {
    let err = parse_filter("has_suffix(name, 'ice')").unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument);
}

#[test]
fn test_parse_array_length_compare() {
    let expr = parse_filter("array_length(tags) >= 2").unwrap();
    assert!(matches!(expr, FilterExpr::ArrayLengthCompare { .. }));
}

#[test]
fn test_parse_loose_identifier_with_leading_digit_and_dash_is_accepted() {
    let expr = parse_filter("1-dash_score_field = 'test'").unwrap();
    assert!(matches!(
        expr,
        FilterExpr::Compare { field, op: CompareOp::Equal, value: Value::String(s) }
        if field == "1-dash_score_field" && s == "test"
    ));
}

#[test]
fn test_parse_is_null_rejects_compound_identifier() {
    let err = parse_filter("a.b is null").unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument);
}

#[test]
fn test_parse_is_not_null_rejects_compound_identifier() {
    let err = parse_filter("a.b is not null").unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument);
}

#[test]
fn test_parse_backtick_quoted_dot_identifier_is_rejected() {
    let err = parse_filter("`a.b` is null").unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument);
}

#[test]
fn test_parse_angle_bracket_not_equal_is_rejected() {
    let err = parse_filter("age <> 1").unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument);
}

#[test]
fn test_parse_double_equals_is_rejected() {
    let err = parse_filter("age == 1").unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument);
}

#[test]
fn test_parse_contain_any_function_call_is_rejected() {
    let err = parse_filter("contain_any(tags, 'x')").unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument);
}

#[test]
fn test_parse_like_custom_escape() {
    let expr = parse_filter(r"name LIKE 'a\b!%' ESCAPE '!'").unwrap();
    assert!(matches!(
        expr,
        FilterExpr::Compare { field, op: CompareOp::Equal, value: Value::String(s) }
        if field == "name" && s == r"a\b%"
    ));
    let expr = parse_filter(r"name LIKE 'a\_%' ESCAPE '\\'").unwrap();
    assert!(matches!(
        expr,
        FilterExpr::HasPrefix { field, prefix } if field == "name" && prefix == "a_"
    ));
    assert!(parse_filter("name LIKE 'ab!' ESCAPE '!'").is_err());
    assert!(parse_filter("name LIKE 'ab' ESCAPE '!!'").is_err());
}

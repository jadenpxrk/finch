use super::*;
use finch_types::{FieldSchema, StatusCode};

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
fn test_enforce_max_filter_terms_matches_reference_limit() {
    // Build a balanced AND tree with 4097 leaf predicates, which should be rejected.
    fn leaf(i: i64) -> FilterExpr {
        FilterExpr::Compare {
            field: "age".to_string(),
            op: CompareOp::Equal,
            value: Value::I64(i),
        }
    }

    let mut nodes: Vec<FilterExpr> = (0..4097).map(leaf).collect();
    while nodes.len() > 1 {
        let mut next: Vec<FilterExpr> = Vec::with_capacity(nodes.len().div_ceil(2));
        for chunk in nodes.chunks(2) {
            if chunk.len() == 2 {
                next.push(FilterExpr::And(
                    Box::new(chunk[0].clone()),
                    Box::new(chunk[1].clone()),
                ));
            } else {
                next.push(chunk[0].clone());
            }
        }
        nodes = next;
    }

    let expr = nodes.pop().unwrap();
    let err = enforce_max_filter_terms(&expr).unwrap_err();
    assert!(err
        .to_string()
        .contains("max number of filters is limited to 4096"));
}

#[test]
fn test_parse_double_quoted_string_literals_are_accepted() {
    // normalize_reference_string_escapes rewrites them so
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
fn test_rewrite_contain_empty_list_matches_reference_semantics() {
    let expr = rewrite_expressions(parse_filter("tags contain_any ()").unwrap());
    assert!(matches!(expr, FilterExpr::AlwaysFalse));

    let expr = rewrite_expressions(parse_filter("tags contain_all ()").unwrap());
    assert!(matches!(expr, FilterExpr::IsNotNull(field) if field == "tags"));

    let expr = rewrite_expressions(parse_filter("tags NOT CONTAIN_ANY ()").unwrap());
    assert!(matches!(expr, FilterExpr::IsNotNull(field) if field == "tags"));

    let expr = rewrite_expressions(parse_filter("tags NOT CONTAIN_ALL ()").unwrap());
    assert!(matches!(expr, FilterExpr::AlwaysFalse));
}

#[test]
fn test_rewrite_eq_or_becomes_in_list() {
    let expr = rewrite_expressions(parse_filter("age = 10 OR age = 20").unwrap());
    assert!(matches!(
        expr,
        FilterExpr::InList { field, values, negated: false }
        if field == "age" && values.len() == 2
    ));
}

#[test]
fn test_rewrite_ne_or_becomes_not_in_list() {
    // `age != 10 OR age != 20` holds for every non-null age, so it must stay an OR.
    let expr = rewrite_expressions(parse_filter("age != 10 OR age != 20").unwrap());
    assert!(matches!(
        expr,
        FilterExpr::Or(a, b)
        if matches!(&*a, FilterExpr::Compare { field, op: CompareOp::NotEqual, .. } if field == "age")
            && matches!(&*b, FilterExpr::Compare { field, op: CompareOp::NotEqual, .. } if field == "age")
    ));
}

#[test]
fn test_rewrite_eq_and_ne_or_becomes_or_of_lists() {
    let expr = rewrite_expressions(
        parse_filter("age != 10 OR age != 20 OR age = 30 OR age = 40").unwrap(),
    );
    // Only the `=` terms merge; the `!=` terms stay as an OR ahead of the IN list.
    assert!(matches!(
        expr,
        FilterExpr::Or(a, b)
        if matches!(&*a, FilterExpr::Or(x, y)
            if matches!(&**x, FilterExpr::Compare { op: CompareOp::NotEqual, .. })
                && matches!(&**y, FilterExpr::Compare { op: CompareOp::NotEqual, .. }))
            && matches!(&*b, FilterExpr::InList { field, negated: false, values } if field == "age" && values.len() == 2)
    ));
}

#[test]
fn test_rewrite_does_not_merge_preexisting_in() {
    let expr = rewrite_expressions(parse_filter("age IN (10, 20) OR age = 30").unwrap());
    assert!(matches!(expr, FilterExpr::Or(_, _)));
}

#[test]
fn test_rewrite_mixed_or_and() {
    // simple_rewriter_test: "gender =1 and age = 10 or age = 20 or age = 30 or age = 40"
    let expr = rewrite_expressions(
        parse_filter("gender = 1 AND age = 10 OR age = 20 OR age = 30 OR age = 40").unwrap(),
    );
    assert!(matches!(
        expr,
        FilterExpr::Or(a, b)
        if matches!(&*a, FilterExpr::And(_, _))
            && matches!(&*b, FilterExpr::InList { field, negated: false, values } if field == "age" && values.len() == 3)
    ));

    // simple_rewriter_test: "age = 10 or age = 20 or age = 30 or age = 40 and gender = 1"
    let expr = rewrite_expressions(
        parse_filter("age = 10 OR age = 20 OR age = 30 OR age = 40 AND gender = 1").unwrap(),
    );
    assert!(matches!(
        expr,
        FilterExpr::Or(a, b)
        if matches!(&*a, FilterExpr::InList { field, negated: false, values } if field == "age" && values.len() == 3)
            && matches!(&*b, FilterExpr::And(_, _))
    ));
}

fn contains_in_list(expr: &FilterExpr) -> bool {
    match expr {
        FilterExpr::InList { .. } => true,
        FilterExpr::And(a, b) | FilterExpr::Or(a, b) => {
            contains_in_list(a.as_ref()) || contains_in_list(b.as_ref())
        }
        FilterExpr::Not(inner) => contains_in_list(inner.as_ref()),
        _ => false,
    }
}

#[test]
fn test_rewrite_eq_or_flattens_parenthesized_or_groups() {
    // simple_rewriter_test: parenthesized OR groups are still merged into a single IN list.
    let expr = rewrite_expressions(parse_filter("age = 1 OR (age = 2 OR age = 3)").unwrap());
    assert!(matches!(
        expr,
        FilterExpr::InList { field, negated: false, values }
        if field == "age" && values.len() == 3
    ));
}

#[test]
fn test_rewrite_eq_or_does_not_merge_non_consecutive_same_field() {
    // simple_rewriter_test: interleaved different-field EQ terms do not cause a non-local merge.
    let expr = rewrite_expressions(parse_filter("a = 1 OR b = 2 OR a = 3").unwrap());
    assert!(matches!(expr, FilterExpr::Or(_, _)));
    assert!(
        !contains_in_list(&expr),
        "did not expect an IN-list rewrite: {expr:?}"
    );
}

#[test]
fn test_rewrite_eq_or_merges_empty_string_literals() {
    // user case: (strAttr ='' or strAttr = 'prd') and ...
    let expr = rewrite_expressions(parse_filter("strAttr = '' OR strAttr = 'prd'").unwrap());
    assert!(matches!(
        expr,
        FilterExpr::InList { field, negated: false, values }
        if field == "strAttr"
            && values.len() == 2
            && values.iter().any(|v| matches!(v, Value::String(s) if s.is_empty()))
            && values.iter().any(|v| matches!(v, Value::String(s) if s == "prd"))
    ));
}

#[test]
fn test_rewrite_eq_or_does_not_merge_mixed_eq_and_ne() {
    // NotChanged4: (a = 1 or a != 2) should not be rewritten.
    let expr = rewrite_expressions(parse_filter("a = 1 OR a != 2").unwrap());
    assert!(matches!(expr, FilterExpr::Or(_, _)));
    assert!(
        !contains_in_list(&expr),
        "did not expect an IN-list rewrite: {expr:?}"
    );
}

#[test]
fn test_rewrite_empty_contain_any_drops_out_of_or() {
    // MiscOr: `... OR contain_any ()` simplifies away (contain_any() == false).
    let expr = rewrite_expressions(
        parse_filter("a = 1 OR a = 2 OR a = 3 OR tags contain_any ()").unwrap(),
    );
    assert!(matches!(
        expr,
        FilterExpr::InList { field, negated: false, values }
        if field == "a" && values.len() == 3
    ));
}

#[test]
fn test_rewrite_empty_contain_any_makes_and_always_false() {
    // MiscAnd: `(a=... OR ...) AND contain_any()` is unsatisfiable.
    let expr = rewrite_expressions(
        parse_filter("(a = 1 OR a = 2 OR a = 3) AND tags contain_any ()").unwrap(),
    );
    assert!(matches!(expr, FilterExpr::AlwaysFalse));
}

#[test]
fn test_parse_string_backslash_quote_escape_matches_reference() {
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
fn test_in_list_size_limit_matches_reference() {
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("age", DataType::Int32))
        .with_field(FieldSchema::new("name", DataType::String));

    let values = (0..=20_000).map(Value::I64).collect::<Vec<_>>();
    let expr = FilterExpr::InList {
        field: "age".to_string(),
        values,
        negated: false,
    };

    let err = validate_filter_expr(&expr, &schema).unwrap_err();
    assert!(err
        .message
        .contains("In rel expr only support list size no more than 20000"));
}

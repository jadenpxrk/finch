use super::*;

/// Applies analyzer-style rewrites after schema validation.
///
/// Rewriting after validation preserves every field reference needed by the
/// validator.
pub fn rewrite_expressions(expr: FilterExpr) -> FilterExpr {
    let expr = rewrite_eq_or(expr);
    rewrite_contain_and_simplify(expr)
}

fn simplify_not(inner: FilterExpr) -> FilterExpr {
    match inner {
        FilterExpr::AlwaysTrue => FilterExpr::AlwaysFalse,
        FilterExpr::AlwaysFalse => FilterExpr::AlwaysTrue,
        FilterExpr::IsNull(f) => FilterExpr::IsNotNull(f),
        FilterExpr::IsNotNull(f) => FilterExpr::IsNull(f),
        other => FilterExpr::Not(Box::new(other)),
    }
}

fn simplify_and(a: FilterExpr, b: FilterExpr) -> FilterExpr {
    match (a, b) {
        (FilterExpr::AlwaysFalse, _) | (_, FilterExpr::AlwaysFalse) => FilterExpr::AlwaysFalse,
        (FilterExpr::AlwaysTrue, x) | (x, FilterExpr::AlwaysTrue) => x,
        (x, y) => FilterExpr::And(Box::new(x), Box::new(y)),
    }
}

fn simplify_or(a: FilterExpr, b: FilterExpr) -> FilterExpr {
    match (a, b) {
        (FilterExpr::AlwaysTrue, _) | (_, FilterExpr::AlwaysTrue) => FilterExpr::AlwaysTrue,
        (FilterExpr::AlwaysFalse, x) | (x, FilterExpr::AlwaysFalse) => x,
        (x, y) => FilterExpr::Or(Box::new(x), Box::new(y)),
    }
}

fn fold_or(mut terms: Vec<FilterExpr>) -> FilterExpr {
    if terms.is_empty() {
        return FilterExpr::AlwaysFalse;
    }
    let first = terms.remove(0);
    terms.into_iter().fold(first, simplify_or)
}

fn collect_or_terms(expr: FilterExpr, out: &mut Vec<FilterExpr>) {
    match expr {
        FilterExpr::Or(a, b) => {
            collect_or_terms(*a, out);
            collect_or_terms(*b, out);
        }
        other => out.push(other),
    }
}

/// (EqualOrRewriteRule): rewrite sequences of `field = v` under OR into `IN`.
///
/// `!=` terms are left alone: `a != 1 OR a != 2` is not `a NOT IN (1, 2)`.
/// This is implemented as a *top-level OR flatten + single sequential pass*
/// Uses DFS traversal across nested OR nodes.
fn rewrite_eq_or(expr: FilterExpr) -> FilterExpr {
    match expr {
        FilterExpr::And(a, b) => {
            FilterExpr::And(Box::new(rewrite_eq_or(*a)), Box::new(rewrite_eq_or(*b)))
        }
        FilterExpr::Not(inner) => FilterExpr::Not(Box::new(rewrite_eq_or(*inner))),
        FilterExpr::Or(a, b) => {
            let mut terms = Vec::new();
            collect_or_terms(FilterExpr::Or(a, b), &mut terms);
            // Recurse into non-OR subtrees, then do the OR merge once.
            let terms = terms.into_iter().map(rewrite_eq_or).collect::<Vec<_>>();
            let terms = merge_terms(terms);
            fold_or(terms)
        }
        other => other,
    }
}

/// A `field = v` term.
struct EqCompare<'a> {
    field: &'a str,
    value: &'a Value,
}

fn as_eq_compare(expr: &FilterExpr) -> Option<EqCompare<'_>> {
    match expr {
        FilterExpr::Compare {
            field,
            op: CompareOp::Equal,
            value,
        } => Some(EqCompare { field, value }),
        _ => None,
    }
}

fn push_unique(dst: &mut Vec<Value>, v: &Value) {
    if !dst.contains(v) {
        dst.push(v.clone());
    }
}

/// Merges `cur` into the accumulated term; returns false when `cur` starts a new run.
fn merge_into_accumulator(acc: &mut FilterExpr, cur: &EqCompare<'_>) -> bool {
    let (acc_field, mut acc_vals) = match &*acc {
        FilterExpr::Compare {
            field,
            op: CompareOp::Equal,
            value,
        } => (field.as_str(), vec![value.clone()]),
        FilterExpr::InList {
            field,
            values,
            negated: false,
        } => (field.as_str(), values.clone()),
        // Unreachable in practice (the accumulator only points at prior `=` terms).
        _ => return false,
    };
    if acc_field != cur.field {
        return false;
    }
    push_unique(&mut acc_vals, cur.value);
    if acc_vals.len() >= 2 {
        *acc = FilterExpr::InList {
            field: acc_field.to_string(),
            values: acc_vals,
            negated: false,
        };
    }
    true
}

fn merge_terms(terms: Vec<FilterExpr>) -> Vec<FilterExpr> {
    let mut out: Vec<FilterExpr> = Vec::with_capacity(terms.len());
    let mut cur_pos: Option<usize> = None;

    for term in terms {
        let Some(cur) = as_eq_compare(&term) else {
            // do not reset accumulator for non `=` nodes.
            out.push(term);
            continue;
        };
        let merged = match cur_pos {
            Some(pos) => merge_into_accumulator(&mut out[pos], &cur),
            None => false,
        };
        if !merged {
            out.push(term);
            cur_pos = Some(out.len() - 1);
        }
    }

    out
}

fn rewrite_contain_and_simplify(expr: FilterExpr) -> FilterExpr {
    match expr {
        FilterExpr::And(a, b) => simplify_and(
            rewrite_contain_and_simplify(*a),
            rewrite_contain_and_simplify(*b),
        ),
        FilterExpr::Or(a, b) => simplify_or(
            rewrite_contain_and_simplify(*a),
            rewrite_contain_and_simplify(*b),
        ),
        FilterExpr::Not(inner) => {
            match *inner {
                // (ContainRewriteRule):
                // `not contain_all ()` evaluates to false
                FilterExpr::ContainAll { values, .. } if values.is_empty() => {
                    FilterExpr::AlwaysFalse
                }
                // `not contain_any ()` rewrites to `is not null`
                FilterExpr::ContainAny { field, values } if values.is_empty() => {
                    FilterExpr::IsNotNull(field)
                }
                other => simplify_not(rewrite_contain_and_simplify(other)),
            }
        }
        // (ContainRewriteRule): `contain_any()` evaluates to false.
        FilterExpr::ContainAny { values, .. } if values.is_empty() => FilterExpr::AlwaysFalse,
        // (ContainRewriteRule): `contain_all()` rewrites to `is not null`.
        FilterExpr::ContainAll { field, values } if values.is_empty() => {
            FilterExpr::IsNotNull(field)
        }
        other => other,
    }
}

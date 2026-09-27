//! Filter expression evaluator

use super::parser::FilterExpr;
use finch_types::{CompareOp, Doc, Value, SYS_GLOBAL_DOC_ID, SYS_LOCAL_ROW_ID, SYS_USER_ID};
use roaring::{RoaringBitmap, RoaringTreemap};

/// Evaluates filter expressions against documents or bitmaps
pub struct DocFilterEvaluator;

impl DocFilterEvaluator {
    /// Check if a single doc passes the filter (for writing segment, forward scan)
    pub fn passes(
        filter: &FilterExpr,
        doc: &Doc,
        delete_bitmap: Option<&RoaringTreemap>,
        row_id: Option<u64>,
    ) -> bool {
        // Check delete first
        if let Some(bitmap) = delete_bitmap {
            if bitmap.contains(doc.doc_id) {
                return false;
            }
        }
        eval_expr(filter, doc, row_id).is_true()
    }

    /// Evaluate a filter against a bitmap of candidates (within-segment u32 doc_ids)
    pub fn filter_bitmap(
        filter: &FilterExpr,
        candidates: &RoaringBitmap,
        doc_fetch: &impl Fn(u64) -> Option<Doc>,
        delete_bitmap: Option<&RoaringTreemap>,
        row_id_for_doc_id: Option<&dyn Fn(u64) -> Option<u64>>,
    ) -> RoaringBitmap {
        let mut result = RoaringBitmap::new();
        for doc_id in candidates.iter() {
            let doc_id_u64 = doc_id as u64;

            // Check delete
            if let Some(bitmap) = delete_bitmap {
                if bitmap.contains(doc_id_u64) {
                    continue;
                }
            }

            if let Some(doc) = doc_fetch(doc_id_u64) {
                let row_id = row_id_for_doc_id.and_then(|f| f(doc_id_u64));
                if eval_expr(filter, &doc, row_id).is_true() {
                    result.insert(doc_id);
                }
            }
        }
        result
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Truth {
    True,
    False,
    Unknown,
}

impl Truth {
    fn is_true(self) -> bool {
        matches!(self, Truth::True)
    }
}

fn truth_from_bool(v: bool) -> Truth {
    if v {
        Truth::True
    } else {
        Truth::False
    }
}

fn truth_not(v: Truth) -> Truth {
    match v {
        Truth::True => Truth::False,
        Truth::False => Truth::True,
        Truth::Unknown => Truth::Unknown,
    }
}

fn truth_and(a: Truth, b: Truth) -> Truth {
    match (a, b) {
        (Truth::False, _) | (_, Truth::False) => Truth::False,
        (Truth::True, Truth::True) => Truth::True,
        _ => Truth::Unknown,
    }
}

fn truth_or(a: Truth, b: Truth) -> Truth {
    match (a, b) {
        (Truth::True, _) | (_, Truth::True) => Truth::True,
        (Truth::False, Truth::False) => Truth::False,
        _ => Truth::Unknown,
    }
}

// SQL-ish 3-valued logic (NULL handling):
// - predicates on NULL/missing values evaluate to Unknown
// - Unknown is treated as non-matching at the top level
// - NOT Unknown stays Unknown
fn eval_expr(filter: &FilterExpr, doc: &Doc, row_id: Option<u64>) -> Truth {
    match filter {
        FilterExpr::AlwaysTrue => Truth::True,
        FilterExpr::AlwaysFalse => Truth::False,
        FilterExpr::Compare { field, op, value } => eval_compare(field, op, value, doc, row_id),
        FilterExpr::And(a, b) => truth_and(eval_expr(a, doc, row_id), eval_expr(b, doc, row_id)),
        FilterExpr::Or(a, b) => truth_or(eval_expr(a, doc, row_id), eval_expr(b, doc, row_id)),
        FilterExpr::Not(inner) => truth_not(eval_expr(inner, doc, row_id)),
        FilterExpr::IsNull(field) => eval_is_null(field, doc, row_id),
        FilterExpr::IsNotNull(field) => truth_not(eval_is_null(field, doc, row_id)),
        FilterExpr::ArrayLengthCompare { field, op, len } => {
            eval_array_length(field, op, *len, doc)
        }
        FilterExpr::InList {
            field,
            values,
            negated,
        } => eval_in_list(field, values, *negated, doc, row_id),
        FilterExpr::HasPrefix { field, prefix } => string_operand(field, doc)
            .map_or(Truth::Unknown, |v| {
                truth_from_bool(v.starts_with(prefix.as_str()))
            }),
        FilterExpr::HasSuffix { field, suffix } => string_operand(field, doc)
            .map_or(Truth::Unknown, |v| {
                truth_from_bool(v.ends_with(suffix.as_str()))
            }),
        FilterExpr::LikePattern { field, pattern } => string_operand(field, doc)
            .map_or(Truth::Unknown, |v| {
                truth_from_bool(sql_like_match(v, pattern))
            }),
        FilterExpr::ContainAll { field, values } => array_operand(field, doc)
            .map_or(Truth::Unknown, |v| {
                truth_from_bool(values.iter().all(|needle| array_contains(v, needle)))
            }),
        FilterExpr::ContainAny { field, values } => array_operand(field, doc)
            .map_or(Truth::Unknown, |v| {
                truth_from_bool(values.iter().any(|needle| array_contains(v, needle)))
            }),
    }
}

#[inline]
fn eval_compare(
    field: &str,
    op: &CompareOp,
    value: &Value,
    doc: &Doc,
    row_id: Option<u64>,
) -> Truth {
    if value.is_null() {
        return Truth::Unknown;
    }

    match field {
        SYS_USER_ID => match value {
            Value::String(rhs) => truth_from_bool(compare_ord(doc.pk.as_str(), op, rhs.as_str())),
            _ => Truth::False,
        },
        SYS_GLOBAL_DOC_ID => compare_system_u64(doc.doc_id, op, value),
        SYS_LOCAL_ROW_ID => match row_id {
            Some(row_id) => compare_system_u64(row_id, op, value),
            None => Truth::Unknown,
        },
        _ => match doc.fields.get(field) {
            None | Some(Value::Null) => Truth::Unknown,
            Some(v) => truth_from_bool(compare_values(v, op, value)),
        },
    }
}

/// Compares a u64 system column against a non-negative integer literal.
#[inline]
fn compare_system_u64(actual: u64, op: &CompareOp, value: &Value) -> Truth {
    let rhs = match value {
        Value::U64(rhs) => *rhs,
        Value::U32(rhs) => *rhs as u64,
        Value::I64(rhs) if *rhs >= 0 => *rhs as u64,
        Value::I32(rhs) if *rhs >= 0 => *rhs as u64,
        _ => return Truth::False,
    };
    truth_from_bool(compare_ord(&actual, op, &rhs))
}

#[inline]
fn eval_is_null(field: &str, doc: &Doc, row_id: Option<u64>) -> Truth {
    match field {
        SYS_USER_ID | SYS_GLOBAL_DOC_ID => Truth::False,
        SYS_LOCAL_ROW_ID => {
            if row_id.is_some() {
                Truth::False
            } else {
                Truth::Unknown
            }
        }
        _ => truth_from_bool(doc.fields.get(field).map(|v| v.is_null()).unwrap_or(true)),
    }
}

#[inline]
fn eval_array_length(field: &str, op: &CompareOp, len: u32, doc: &Doc) -> Truth {
    let Some(v) = doc.fields.get(field) else {
        return Truth::Unknown;
    };
    if v.is_null() {
        return Truth::Unknown;
    }
    let Some(actual) = array_len(v) else {
        return Truth::False;
    };
    truth_from_bool(compare_ord(&actual, op, &len))
}

#[inline]
fn array_len(v: &Value) -> Option<u32> {
    match v {
        Value::ArrayBinary(items) => Some(items.len() as u32),
        Value::ArrayString(items) => Some(items.len() as u32),
        Value::ArrayBool(items) => Some(items.len() as u32),
        Value::ArrayI32(items) => Some(items.len() as u32),
        Value::ArrayI64(items) => Some(items.len() as u32),
        Value::ArrayU32(items) => Some(items.len() as u32),
        Value::ArrayU64(items) => Some(items.len() as u32),
        Value::ArrayF32(items) => Some(items.len() as u32),
        Value::ArrayF64(items) => Some(items.len() as u32),
        _ => None,
    }
}

#[inline]
fn eval_in_list(
    field: &str,
    values: &[Value],
    negated: bool,
    doc: &Doc,
    row_id: Option<u64>,
) -> Truth {
    let Some(doc_val) = in_list_operand(field, doc, row_id) else {
        return Truth::Unknown;
    };
    let mut has_null = false;
    for v in values {
        if v.is_null() {
            has_null = true;
            continue;
        }
        if compare_values(&doc_val, &CompareOp::Equal, v) {
            return if negated { Truth::False } else { Truth::True };
        }
    }

    let base = if has_null {
        Truth::Unknown
    } else {
        Truth::False
    };
    if negated {
        truth_not(base)
    } else {
        base
    }
}

/// Value an IN list is matched against; `None` when missing or NULL.
#[inline]
fn in_list_operand(field: &str, doc: &Doc, row_id: Option<u64>) -> Option<Value> {
    match field {
        SYS_USER_ID => Some(Value::String(doc.pk.clone())),
        SYS_GLOBAL_DOC_ID => Some(Value::U64(doc.doc_id)),
        SYS_LOCAL_ROW_ID => row_id.map(Value::U64),
        _ => doc.fields.get(field).filter(|v| !v.is_null()).cloned(),
    }
}

#[inline]
fn string_operand<'a>(field: &str, doc: &'a Doc) -> Option<&'a str> {
    match field {
        SYS_USER_ID => Some(doc.pk.as_str()),
        _ => doc.fields.get(field).and_then(|v| v.as_str()),
    }
}

#[inline]
fn array_operand<'a>(field: &str, doc: &'a Doc) -> Option<&'a Value> {
    doc.fields.get(field).filter(|v| !matches!(v, Value::Null))
}

fn array_contains(array: &Value, needle: &Value) -> bool {
    match array {
        Value::ArrayBinary(arr) => match needle {
            Value::Bytes(b) => arr.iter().any(|v| v == b),
            _ => false,
        },
        Value::ArrayString(arr) => match needle {
            Value::String(s) => arr.contains(s),
            _ => false,
        },
        Value::ArrayBool(arr) => match needle {
            Value::Bool(b) => arr.contains(b),
            _ => false,
        },
        Value::ArrayI32(arr) => to_i128(needle)
            .map(|n| arr.iter().any(|v| (*v as i128) == n))
            .unwrap_or(false),
        Value::ArrayI64(arr) => to_i128(needle)
            .map(|n| arr.iter().any(|v| (*v as i128) == n))
            .unwrap_or(false),
        Value::ArrayU32(arr) => to_i128(needle)
            .map(|n| arr.iter().any(|v| (*v as i128) == n))
            .unwrap_or(false),
        Value::ArrayU64(arr) => to_i128(needle)
            .map(|n| arr.iter().any(|v| (*v as i128) == n))
            .unwrap_or(false),
        Value::ArrayF32(arr) => to_f32(needle).map(|n| arr.contains(&n)).unwrap_or(false),
        Value::ArrayF64(arr) => to_f64(needle).map(|n| arr.contains(&n)).unwrap_or(false),
        _ => false,
    }
}

fn sql_like_match(text: &str, pattern: &str) -> bool {
    // support `\` escaping for `%` and `_` (and any next char).
    let t: Vec<char> = text.chars().collect();
    let p: Vec<char> = pattern.chars().collect();
    let m = t.len();
    let n = p.len();

    let mut dp = vec![vec![false; n + 1]; m + 1];
    dp[0][0] = true;

    for i in 0..=m {
        for j in 0..n {
            if !dp[i][j] {
                continue;
            }
            // `%` matches empty.
            if p[j] == '%' {
                dp[i][j + 1] = true;
            }
            // Every other transition consumes one text char.
            if i == m {
                continue;
            }
            match p[j] {
                '%' => dp[i + 1][j] = true,
                '_' => dp[i + 1][j + 1] = true,
                '\\' if j + 1 < n => dp[i + 1][j + 2] |= t[i] == p[j + 1],
                // Trailing '\': treat as matching literal '\' if present.
                '\\' => dp[i + 1][j + 1] |= t[i] == '\\',
                lit => dp[i + 1][j + 1] |= t[i] == lit,
            }
        }
    }

    dp[m][n]
}

fn compare_values(a: &Value, op: &CompareOp, b: &Value) -> bool {
    // Exact integer comparisons (avoid `f64` precision loss for large integers).
    if let (Some(ia), Some(ib)) = (to_i128(a), to_i128(b)) {
        return compare_ord(&ia, op, &ib);
    }

    // Float comparisons (best-effort for cross-float variants; schema-aware
    // filter normalization ensures the common path compares same-width floats).
    if let (Some(fa), Some(fb)) = (to_f64(a), to_f64(b)) {
        return compare_ord(&fa, op, &fb);
    }

    // String comparison
    if let (Some(sa), Some(sb)) = (a.as_str(), b.as_str()) {
        return compare_ord(sa, op, sb);
    }

    // Bool comparison
    if let (Value::Bool(ba), Value::Bool(bb)) = (a, b) {
        return match op {
            CompareOp::Equal => ba == bb,
            CompareOp::NotEqual => ba != bb,
            _ => false,
        };
    }

    // Bytes comparison (lexicographic).
    if let (Value::Bytes(ba), Value::Bytes(bb)) = (a, b) {
        return compare_ord(ba.as_slice(), op, bb.as_slice());
    }

    false
}

#[inline]
fn compare_ord<T: PartialOrd + ?Sized>(a: &T, op: &CompareOp, b: &T) -> bool {
    match op {
        CompareOp::Equal => a == b,
        CompareOp::NotEqual => a != b,
        CompareOp::LessThan => a < b,
        CompareOp::LessEqual => a <= b,
        CompareOp::GreaterThan => a > b,
        CompareOp::GreaterEqual => a >= b,
    }
}

fn to_i128(v: &Value) -> Option<i128> {
    match v {
        Value::I8(x) => Some(*x as i128),
        Value::I16(x) => Some(*x as i128),
        Value::I32(x) => Some(*x as i128),
        Value::I64(x) => Some(*x as i128),
        Value::U8(x) => Some(*x as i128),
        Value::U16(x) => Some(*x as i128),
        Value::U32(x) => Some(*x as i128),
        Value::U64(x) => Some(*x as i128),
        _ => None,
    }
}

fn to_f32(v: &Value) -> Option<f32> {
    match v {
        Value::F32(x) => Some(*x),
        Value::F16(x) => Some(x.to_f32()),
        Value::I8(x) => Some(*x as f32),
        Value::I16(x) => Some(*x as f32),
        Value::I32(x) => Some(*x as f32),
        Value::I64(x) => Some(*x as f32),
        Value::U8(x) => Some(*x as f32),
        Value::U16(x) => Some(*x as f32),
        Value::U32(x) => Some(*x as f32),
        Value::U64(x) => Some(*x as f32),
        _ => None,
    }
}

#[inline]
pub(crate) fn to_f64(v: &Value) -> Option<f64> {
    match v {
        Value::I8(x) => Some(*x as f64),
        Value::I16(x) => Some(*x as f64),
        Value::I32(x) => Some(*x as f64),
        Value::I64(x) => Some(*x as f64),
        Value::U8(x) => Some(*x as f64),
        Value::U16(x) => Some(*x as f64),
        Value::U32(x) => Some(*x as f64),
        Value::U64(x) => Some(*x as f64),
        Value::F32(x) => Some(*x as f64),
        Value::F64(x) => Some(*x),
        Value::F16(x) => Some(x.to_f32() as f64),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sqlengine::parser::parse_filter;

    #[test]
    fn test_like_escape_semantics_match_reference() {
        let mut doc = Doc::new("a");
        doc.doc_id = 1;
        doc.fields
            .insert("name".to_string(), Value::String("user-%22".to_string()));
        doc.fields
            .insert("name2".to_string(), Value::String("user-_22".to_string()));

        // Escaped '%' matches literal '%'.
        let expr = parse_filter("name LIKE 'user-\\%22'").unwrap();
        assert!(DocFilterEvaluator::passes(&expr, &doc, None, None));

        // Escaped '_' matches literal '_'.
        let expr = parse_filter("name2 LIKE 'user-\\_22'").unwrap();
        assert!(DocFilterEvaluator::passes(&expr, &doc, None, None));

        // Unescaped '_' matches any single character.
        let expr = parse_filter("name LIKE 'user-_22'").unwrap();
        assert!(DocFilterEvaluator::passes(&expr, &doc, None, None));
    }

    #[test]
    fn test_null_propagation_three_valued_logic_basics() {
        let mut doc = Doc::new("pk1");
        doc.doc_id = 1;
        doc.fields.insert("x".to_string(), Value::I64(2));

        // Missing field comparisons evaluate to Unknown => non-matching at top-level.
        let expr = parse_filter("missing = 1").unwrap();
        assert!(!DocFilterEvaluator::passes(&expr, &doc, None, None));

        // Missing is treated as NULL for IS NULL / IS NOT NULL .
        let expr = parse_filter("missing IS NULL").unwrap();
        assert!(DocFilterEvaluator::passes(&expr, &doc, None, None));
        let expr = parse_filter("missing IS NOT NULL").unwrap();
        assert!(!DocFilterEvaluator::passes(&expr, &doc, None, None));

        // Unknown AND True => Unknown (non-matching); Unknown OR True => True (matching).
        let expr = parse_filter("missing = 1 AND x = 2").unwrap();
        assert!(!DocFilterEvaluator::passes(&expr, &doc, None, None));
        let expr = parse_filter("missing = 1 OR x = 2").unwrap();
        assert!(DocFilterEvaluator::passes(&expr, &doc, None, None));

        // Unknown OR False => Unknown (non-matching).
        let expr = parse_filter("missing = 1 OR x = 3").unwrap();
        assert!(!DocFilterEvaluator::passes(&expr, &doc, None, None));

        // Unknown AND False => False (non-matching).
        let expr = parse_filter("missing = 1 AND x = 3").unwrap();
        assert!(!DocFilterEvaluator::passes(&expr, &doc, None, None));

        // False AND Unknown => False; False OR Unknown => Unknown.
        let expr = parse_filter("x = 3 AND missing = 1").unwrap();
        assert!(!DocFilterEvaluator::passes(&expr, &doc, None, None));
        let expr = parse_filter("x = 3 OR missing = 1").unwrap();
        assert!(!DocFilterEvaluator::passes(&expr, &doc, None, None));

        // True AND Unknown => Unknown; True OR Unknown => True.
        let expr = parse_filter("x = 2 AND missing = 1").unwrap();
        assert!(!DocFilterEvaluator::passes(&expr, &doc, None, None));
        let expr = parse_filter("x = 2 OR missing = 1").unwrap();
        assert!(DocFilterEvaluator::passes(&expr, &doc, None, None));
    }

    #[test]
    fn test_null_literal_is_rejected_outside_is_null() {
        // NULL is not a constant token in relation expressions; it's only valid via
        // `IS NULL` / `IS NOT NULL`.
        assert!(parse_filter("x IN (1, NULL)").is_err());
        assert!(parse_filter("x NOT IN (1, NULL)").is_err());
        assert!(parse_filter("x = NULL").is_err());
    }

    #[test]
    fn test_missing_field_predicates_are_unknown_and_do_not_match() {
        let mut doc = Doc::new("pk1");
        doc.doc_id = 1;
        doc.fields.insert("x".to_string(), Value::I64(2));

        // IN/NOT IN on missing => Unknown (non-matching).
        let expr = parse_filter("missing IN (1, 2)").unwrap();
        assert!(!DocFilterEvaluator::passes(&expr, &doc, None, None));
        let expr = parse_filter("missing NOT IN (1, 2)").unwrap();
        assert!(!DocFilterEvaluator::passes(&expr, &doc, None, None));

        // LIKE on missing => Unknown (non-matching).
        let expr = parse_filter("missing LIKE '%'").unwrap();
        assert!(!DocFilterEvaluator::passes(&expr, &doc, None, None));

        // CONTAIN_* on missing => Unknown (non-matching).
        let expr = parse_filter("missing CONTAIN_ANY (1)").unwrap();
        assert!(!DocFilterEvaluator::passes(&expr, &doc, None, None));
        let expr = parse_filter("missing CONTAIN_ALL (1)").unwrap();
        assert!(!DocFilterEvaluator::passes(&expr, &doc, None, None));
    }

    #[test]
    fn test_three_valued_logic_truth_tables_are_exhaustive() {
        // Truth tables (SQL 3-valued logic).
        let t = Truth::True;
        let f = Truth::False;
        let u = Truth::Unknown;

        // NOT
        assert_eq!(truth_not(t), f);
        assert_eq!(truth_not(f), t);
        assert_eq!(truth_not(u), u);

        // AND
        assert_eq!(truth_and(t, t), t);
        assert_eq!(truth_and(t, f), f);
        assert_eq!(truth_and(t, u), u);
        assert_eq!(truth_and(f, t), f);
        assert_eq!(truth_and(f, f), f);
        assert_eq!(truth_and(f, u), f);
        assert_eq!(truth_and(u, t), u);
        assert_eq!(truth_and(u, f), f);
        assert_eq!(truth_and(u, u), u);

        // OR
        assert_eq!(truth_or(t, t), t);
        assert_eq!(truth_or(t, f), t);
        assert_eq!(truth_or(t, u), t);
        assert_eq!(truth_or(f, t), t);
        assert_eq!(truth_or(f, f), f);
        assert_eq!(truth_or(f, u), u);
        assert_eq!(truth_or(u, t), t);
        assert_eq!(truth_or(u, f), u);
        assert_eq!(truth_or(u, u), u);
    }

    #[test]
    fn test_null_semantics_missing_vs_explicit_null_vs_present() {
        let mut doc = Doc::new("pk1");
        doc.doc_id = 1;
        doc.fields.insert("present".to_string(), Value::I64(1));
        doc.fields.insert("nullish".to_string(), Value::Null);

        // Missing behaves like NULL for IS NULL / IS NOT NULL.
        let expr = parse_filter("missing IS NULL").unwrap();
        assert!(DocFilterEvaluator::passes(&expr, &doc, None, None));
        let expr = parse_filter("missing IS NOT NULL").unwrap();
        assert!(!DocFilterEvaluator::passes(&expr, &doc, None, None));

        // Explicit NULL behaves the same.
        let expr = parse_filter("nullish IS NULL").unwrap();
        assert!(DocFilterEvaluator::passes(&expr, &doc, None, None));
        let expr = parse_filter("nullish IS NOT NULL").unwrap();
        assert!(!DocFilterEvaluator::passes(&expr, &doc, None, None));

        // Present value.
        let expr = parse_filter("present IS NULL").unwrap();
        assert!(!DocFilterEvaluator::passes(&expr, &doc, None, None));
        let expr = parse_filter("present IS NOT NULL").unwrap();
        assert!(DocFilterEvaluator::passes(&expr, &doc, None, None));

        // Predicates on NULL/missing are Unknown (non-matching at top level).
        let expr = parse_filter("missing = 1").unwrap();
        assert!(!DocFilterEvaluator::passes(&expr, &doc, None, None));
        let expr = parse_filter("nullish = 1").unwrap();
        assert!(!DocFilterEvaluator::passes(&expr, &doc, None, None));
    }
}

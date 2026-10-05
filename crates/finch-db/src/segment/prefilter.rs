use finch_types::{CompareOp, Value, ZResult};
use roaring::RoaringTreemap;

use crate::invert::InvertIndex;
use crate::sqlengine::parser::FilterExpr;

#[derive(Debug, Clone)]
pub(super) struct InvertPrefilterPlan {
    /// Doc IDs that satisfy a subset of the predicate (AND case) or the full
    /// predicate (when `exact=true`).
    pub(super) allowlist: Option<roaring::RoaringTreemap>,
    /// True only when the entire expression was translated to invert lookups.
    pub(super) exact: bool,
}

impl InvertPrefilterPlan {
    pub(super) fn none() -> Self {
        InvertPrefilterPlan {
            allowlist: None,
            exact: false,
        }
    }

    fn exact_or_none(bm: Option<RoaringTreemap>) -> Self {
        match bm {
            Some(bm) => InvertPrefilterPlan {
                allowlist: Some(bm),
                exact: true,
            },
            None => InvertPrefilterPlan::none(),
        }
    }
}

/// One inverted-index lookup requested by the planner for a single field.
pub(super) enum InvertLookup<'a> {
    Eq(&'a Value),
    Lt(&'a Value, bool),
    Gt(&'a Value, bool),
    Prefix(&'a str),
    Suffix(&'a str),
    ArrayLen(CompareOp, u32),
    IsNull,
    IsNotNull,
}

impl InvertLookup<'_> {
    /// Run against `idx`; range lookups give up (Ok(None)) past `max_range_hits`.
    pub(super) fn run(
        self,
        idx: &InvertIndex,
        max_range_hits: Option<u64>,
    ) -> ZResult<Option<RoaringTreemap>> {
        match self {
            InvertLookup::Eq(value) => {
                let Some(norm) = idx.normalize_value(value) else {
                    return Ok(None);
                };
                idx.lookup_eq(&norm).map(Some)
            }
            InvertLookup::Lt(value, include_eq) => {
                let Some(norm) = idx.normalize_value(value) else {
                    return Ok(None);
                };
                idx.lookup_lt_bounded(&norm, include_eq, max_range_hits)
                    .or(Ok(None))
            }
            InvertLookup::Gt(value, include_eq) => {
                let Some(norm) = idx.normalize_value(value) else {
                    return Ok(None);
                };
                idx.lookup_gt_bounded(&norm, include_eq, max_range_hits)
                    .or(Ok(None))
            }
            InvertLookup::Prefix(prefix) => Ok(idx.lookup_prefix(prefix).ok()),
            InvertLookup::Suffix(suffix) => Ok(idx.lookup_suffix(suffix).ok()),
            InvertLookup::ArrayLen(op, len) => run_array_len(idx, op, len),
            InvertLookup::IsNull => idx.lookup_is_null().map(Some),
            InvertLookup::IsNotNull => idx.lookup_is_not_null().map(Some),
        }
    }
}

fn run_array_len(idx: &InvertIndex, op: CompareOp, len: u32) -> ZResult<Option<RoaringTreemap>> {
    match op {
        CompareOp::Equal => idx.lookup_array_len_eq(len).map(Some),
        // ratio_rule does not apply to function-call LHS like
        // `array_length(field) <op> N`; keep the inverted lookup unbounded.
        CompareOp::LessThan => idx
            .lookup_array_len_lt_bounded(len, false, None)
            .or(Ok(None)),
        CompareOp::LessEqual => idx
            .lookup_array_len_lt_bounded(len, true, None)
            .or(Ok(None)),
        CompareOp::GreaterThan => idx
            .lookup_array_len_gt_bounded(len, false, None)
            .or(Ok(None)),
        CompareOp::GreaterEqual => idx
            .lookup_array_len_gt_bounded(len, true, None)
            .or(Ok(None)),
        CompareOp::NotEqual => Ok(None),
    }
}

/// Range allowlists at or above `invert_to_forward_scan_ratio` of `total_docs` are skipped.
pub(super) fn max_range_hits(total_docs: u64) -> Option<u64> {
    let cfg = crate::config::global_config();
    let threshold_hits =
        ((total_docs as f64) * (cfg.invert_to_forward_scan_ratio as f64)).ceil() as u64;
    Some(threshold_hits.saturating_sub(1))
}

/// First `limit` ids from an exact candidate stream that are not deleted.
pub(super) fn take_live_ids(
    ids: impl IntoIterator<Item = u64>,
    deleted: &RoaringTreemap,
    limit: usize,
) -> Vec<u64> {
    let mut out: Vec<u64> = Vec::new();
    for doc_id in ids {
        if deleted.contains(doc_id) {
            continue;
        }
        out.push(doc_id);
        if out.len() >= limit {
            break;
        }
    }
    out
}

/// Runs one lookup on the named field's index; Ok(None) when the field has no index.
pub(super) type InvertLookups<'a> =
    &'a dyn Fn(&str, InvertLookup<'_>) -> ZResult<Option<RoaringTreemap>>;

// Build a conservative prefilter allowlist:
// - For AND: intersect whatever child allowlists we can compute.
// - For OR: only safe if we can compute BOTH child allowlists (union).
// - Leaf: supports `=` and `IN` on inverted-index fields.
pub(super) fn plan_for(expr: &FilterExpr, lookups: InvertLookups) -> ZResult<InvertPrefilterPlan> {
    let exact = InvertPrefilterPlan::exact_or_none;
    match expr {
        FilterExpr::AlwaysTrue => Ok(InvertPrefilterPlan {
            allowlist: None,
            exact: true,
        }),
        FilterExpr::AlwaysFalse => Ok(InvertPrefilterPlan {
            allowlist: Some(roaring::RoaringTreemap::new()),
            exact: true,
        }),
        FilterExpr::IsNull(field) => Ok(exact(lookups(field, InvertLookup::IsNull)?)),
        FilterExpr::IsNotNull(field) => Ok(exact(lookups(field, InvertLookup::IsNotNull)?)),
        FilterExpr::Compare { field, op, value } => {
            let Some(lookup) = compare_lookup(*op, value) else {
                return Ok(InvertPrefilterPlan::none());
            };
            Ok(exact(lookups(field, lookup)?))
        }
        FilterExpr::ArrayLengthCompare { field, op, len } => {
            Ok(exact(lookups(field, InvertLookup::ArrayLen(*op, *len))?))
        }
        FilterExpr::HasPrefix { field, prefix } => {
            Ok(exact(lookups(field, InvertLookup::Prefix(prefix))?))
        }
        FilterExpr::HasSuffix { field, suffix } => {
            Ok(exact(lookups(field, InvertLookup::Suffix(suffix))?))
        }
        FilterExpr::InList {
            field,
            values,
            negated,
        } => {
            if *negated {
                return Ok(InvertPrefilterPlan::none());
            }
            plan_eq_values(field, values, Combine::Union, lookups)
        }
        FilterExpr::ContainAny { field, values } => {
            plan_eq_values(field, values, Combine::Union, lookups)
        }
        FilterExpr::ContainAll { field, values } => {
            plan_eq_values(field, values, Combine::Intersect, lookups)
        }
        FilterExpr::And(a, b) => plan_and(plan_for(a, lookups)?, plan_for(b, lookups)?),
        FilterExpr::Or(a, b) => plan_or(plan_for(a, lookups)?, plan_for(b, lookups)?),
        FilterExpr::LikePattern { field, pattern } => plan_like(field, pattern, lookups),
        _ => Ok(InvertPrefilterPlan::none()),
    }
}

fn compare_lookup(op: CompareOp, value: &Value) -> Option<InvertLookup<'_>> {
    match op {
        CompareOp::Equal => Some(InvertLookup::Eq(value)),
        CompareOp::LessThan => Some(InvertLookup::Lt(value, false)),
        CompareOp::LessEqual => Some(InvertLookup::Lt(value, true)),
        CompareOp::GreaterThan => Some(InvertLookup::Gt(value, false)),
        CompareOp::GreaterEqual => Some(InvertLookup::Gt(value, true)),
        CompareOp::NotEqual => None,
    }
}

enum Combine {
    Union,
    Intersect,
}

// Every value must be indexable, otherwise the allowlist would be incomplete.
fn plan_eq_values(
    field: &str,
    values: &[Value],
    combine: Combine,
    lookups: InvertLookups,
) -> ZResult<InvertPrefilterPlan> {
    let mut out: Option<RoaringTreemap> = None;
    for v in values {
        let Some(bm) = lookups(field, InvertLookup::Eq(v))? else {
            return Ok(InvertPrefilterPlan::none());
        };
        match (&mut out, &combine) {
            (None, _) => out = Some(bm),
            (Some(acc), Combine::Union) => *acc |= bm,
            (Some(acc), Combine::Intersect) => *acc &= bm,
        }
    }
    Ok(InvertPrefilterPlan::exact_or_none(out))
}

fn plan_and(pa: InvertPrefilterPlan, pb: InvertPrefilterPlan) -> ZResult<InvertPrefilterPlan> {
    let allowlist = match (pa.allowlist, pb.allowlist) {
        (None, None) => None,
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (Some(mut x), Some(y)) => {
            x &= y;
            Some(x)
        }
    };
    Ok(InvertPrefilterPlan {
        allowlist,
        exact: pa.exact && pb.exact,
    })
}

fn plan_or(pa: InvertPrefilterPlan, pb: InvertPrefilterPlan) -> ZResult<InvertPrefilterPlan> {
    match (pa.allowlist, pb.allowlist) {
        (Some(mut x), Some(y)) => {
            x |= y;
            Ok(InvertPrefilterPlan {
                allowlist: Some(x),
                exact: pa.exact && pb.exact,
            })
        }
        _ => Ok(InvertPrefilterPlan::none()),
    }
}

struct LikeScan {
    percent_count: usize,
    underscore_count: usize,
    percent_loc: Option<usize>,
}

fn scan_like(pat: &str) -> LikeScan {
    let mut scan = LikeScan {
        percent_count: 0,
        underscore_count: 0,
        percent_loc: None,
    };
    let bytes = pat.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'\\' {
            i += 2;
            continue;
        }
        if c == b'%' {
            scan.percent_count += 1;
            scan.percent_loc = Some(i);
        } else if c == b'_' {
            scan.underscore_count += 1;
        }
        i += 1;
    }
    scan
}

fn unescape(pat: &str) -> String {
    let bytes = pat.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] != b'\\' {
            out.push(bytes[i]);
            i += 1;
        } else if i + 1 < bytes.len() {
            out.push(bytes[i + 1]);
            i += 2;
        } else {
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).to_string()
}

// invert supports at most one '%' and no '_' for pushdown.
// Without extended wildcard, '%' must be at the end (prefix match).
fn plan_like(field: &str, pattern: &str, lookups: InvertLookups) -> ZResult<InvertPrefilterPlan> {
    let scan = scan_like(pattern);
    if scan.percent_count > 1 || scan.underscore_count > 0 {
        return Ok(InvertPrefilterPlan::none());
    }

    if scan.percent_count == 0 {
        // No wildcard => equality.
        let unescaped = Value::String(unescape(pattern));
        let bm = lookups(field, InvertLookup::Eq(&unescaped))?;
        return Ok(InvertPrefilterPlan::exact_or_none(bm));
    }

    let Some(loc) = scan.percent_loc else {
        return Ok(InvertPrefilterPlan::none());
    };
    let (lhs, rhs) = pattern.split_at(loc);
    let rhs = &rhs[1..]; // skip '%'
    let lhs = unescape(lhs);
    let rhs = unescape(rhs);

    // Suffix-only: %xyz
    if loc == 0 {
        let bm = lookups(field, InvertLookup::Suffix(rhs.as_str()))?;
        return Ok(InvertPrefilterPlan::exact_or_none(bm));
    }

    // Prefix-only: xyz%
    if loc == pattern.len() - 1 {
        let bm = lookups(field, InvertLookup::Prefix(lhs.as_str()))?;
        return Ok(InvertPrefilterPlan::exact_or_none(bm));
    }

    // Infix: abc%xyz => prefix AND suffix (requires extended wildcard).
    let Some(mut bm) = lookups(field, InvertLookup::Prefix(lhs.as_str()))? else {
        return Ok(InvertPrefilterPlan::none());
    };
    let Some(bm_suffix) = lookups(field, InvertLookup::Suffix(rhs.as_str()))? else {
        return Ok(InvertPrefilterPlan::none());
    };
    bm &= bm_suffix;
    // Prefix and suffix may overlap in short values ('aba' for 'ab%ba'), so rows need a recheck.
    Ok(InvertPrefilterPlan {
        allowlist: Some(bm),
        exact: false,
    })
}

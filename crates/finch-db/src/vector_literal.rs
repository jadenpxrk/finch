use finch_types::{Status, ZResult};

fn is_finch_vector_number_token(t: &str) -> bool {
    // VECTOR token is built from:
    // - MINUS_SIGN
    // - UNSIGNED_INTEGER_FRAGMENT ([0-9]+)
    // - FLOAT_FRAGMENT (UNSIGNED_INTEGER* '.'? UNSIGNED_INTEGER+)
    //
    // This means:
    // - no '+' sign
    // - no exponent
    // - '.' must have digits after it (disallow `1.` / `-.`)
    // - either digits, or digits '.' digits, or '.' digits, with optional leading '-'
    let t = t.trim();
    if t.is_empty() {
        return false;
    }
    if t.starts_with('+') {
        return false;
    }
    if t.bytes()
        .any(|b| matches!(b, b'e' | b'E' | b'd' | b'D' | b'f' | b'F'))
    {
        return false;
    }

    let t = t.strip_prefix('-').unwrap_or(t);
    if t.is_empty() {
        return false;
    }
    let dot_count = t.bytes().filter(|b| *b == b'.').count();
    if dot_count > 1 {
        return false;
    }
    if dot_count == 0 {
        return t.bytes().all(|b| b.is_ascii_digit());
    }
    let (left, right) = t.split_once('.').unwrap_or(("", ""));
    if right.is_empty() {
        return false;
    }
    if !right.bytes().all(|b| b.is_ascii_digit()) {
        return false;
    }
    if !left.is_empty() && !left.bytes().all(|b| b.is_ascii_digit()) {
        return false;
    }
    true
}

pub(crate) fn parse_vector_literal_f32(lit: &str) -> ZResult<Vec<f32>> {
    let s = lit.trim();
    if !s.starts_with('[') || !s.ends_with(']') {
        return Err(Status::invalid_argument("vector format error"));
    }
    let inner = &s[1..s.len().saturating_sub(1)];
    let inner = inner.trim();
    if inner.is_empty() {
        return Err(Status::invalid_argument("vector format error"));
    }
    let mut out: Vec<f32> = Vec::new();
    for part in inner.split(',') {
        let t = part.trim();
        if t.is_empty() {
            return Err(Status::invalid_argument("vector format error"));
        }
        if !is_finch_vector_number_token(t) {
            return Err(Status::invalid_argument("vector format error"));
        }
        let v: f32 = t
            .parse()
            .map_err(|_| Status::invalid_argument("vector format error"))?;
        out.push(v);
    }
    Ok(out)
}

pub(crate) fn parse_vector_literal_u32(lit: &str) -> ZResult<Vec<u32>> {
    let s = lit.trim();
    if !s.starts_with('[') || !s.ends_with(']') {
        return Err(Status::invalid_argument("vector format error"));
    }
    let inner = &s[1..s.len().saturating_sub(1)];
    let inner = inner.trim();
    if inner.is_empty() {
        return Err(Status::invalid_argument("vector format error"));
    }
    let mut out: Vec<u32> = Vec::new();
    for part in inner.split(',') {
        let t = part.trim();
        if t.is_empty() {
            return Err(Status::invalid_argument("vector format error"));
        }
        // VECTOR disallows '+', exponent, and requires digits.
        // For u32 payloads, additionally disallow '-' and '.'.
        if t.starts_with('+') || t.starts_with('-') {
            return Err(Status::invalid_argument("vector format error"));
        }
        if t.bytes()
            .any(|b| matches!(b, b'.' | b'e' | b'E' | b'd' | b'D' | b'f' | b'F'))
        {
            return Err(Status::invalid_argument("vector format error"));
        }
        if !t.bytes().all(|b| b.is_ascii_digit()) {
            return Err(Status::invalid_argument("vector format error"));
        }
        let v: u32 = t
            .parse()
            .map_err(|_| Status::invalid_argument("vector format error"))?;
        out.push(v);
    }
    Ok(out)
}

pub(crate) fn parse_vector_literal_u64(lit: &str) -> ZResult<Vec<u64>> {
    let s = lit.trim();
    if !s.starts_with('[') || !s.ends_with(']') {
        return Err(Status::invalid_argument("vector format error"));
    }
    let inner = &s[1..s.len().saturating_sub(1)];
    let inner = inner.trim();
    if inner.is_empty() {
        return Err(Status::invalid_argument("vector format error"));
    }
    let mut out: Vec<u64> = Vec::new();
    for part in inner.split(',') {
        let t = part.trim();
        if t.is_empty() {
            return Err(Status::invalid_argument("vector format error"));
        }
        if t.starts_with('+') || t.starts_with('-') {
            return Err(Status::invalid_argument("vector format error"));
        }
        if t.bytes()
            .any(|b| matches!(b, b'.' | b'e' | b'E' | b'd' | b'D' | b'f' | b'F'))
        {
            return Err(Status::invalid_argument("vector format error"));
        }
        if !t.bytes().all(|b| b.is_ascii_digit()) {
            return Err(Status::invalid_argument("vector format error"));
        }
        let v: u64 = t
            .parse()
            .map_err(|_| Status::invalid_argument("vector format error"))?;
        out.push(v);
    }
    Ok(out)
}

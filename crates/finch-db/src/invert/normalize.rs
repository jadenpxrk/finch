use super::*;

pub(super) fn normalize_i32(v: &Value) -> Option<i32> {
    match v {
        Value::I32(x) => Some(*x),
        Value::I64(x) => i32::try_from(*x).ok(),
        Value::U32(x) => i32::try_from(*x).ok(),
        Value::U64(x) => i32::try_from(*x).ok(),
        Value::F32(x) => {
            if x.is_finite() && x.fract() == 0.0 {
                i32::try_from(*x as i64).ok()
            } else {
                None
            }
        }
        Value::F64(x) => {
            if x.is_finite() && x.fract() == 0.0 {
                i32::try_from(*x as i64).ok()
            } else {
                None
            }
        }
        _ => None,
    }
}

pub(super) fn normalize_i64(v: &Value) -> Option<i64> {
    match v {
        Value::I64(x) => Some(*x),
        Value::I32(x) => Some(*x as i64),
        Value::U32(x) => Some(*x as i64),
        Value::U64(x) => i64::try_from(*x).ok(),
        Value::F32(x) => {
            if x.is_finite() && x.fract() == 0.0 {
                Some(*x as i64)
            } else {
                None
            }
        }
        Value::F64(x) => {
            if x.is_finite() && x.fract() == 0.0 {
                Some(*x as i64)
            } else {
                None
            }
        }
        _ => None,
    }
}

pub(super) fn normalize_u32(v: &Value) -> Option<u32> {
    match v {
        Value::U32(x) => Some(*x),
        Value::U64(x) => u32::try_from(*x).ok(),
        Value::I32(x) => u32::try_from(*x).ok(),
        Value::I64(x) => u32::try_from(*x).ok(),
        Value::F32(x) => {
            if x.is_finite() && *x >= 0.0 && x.fract() == 0.0 {
                u32::try_from(*x as u64).ok()
            } else {
                None
            }
        }
        Value::F64(x) => {
            if x.is_finite() && *x >= 0.0 && x.fract() == 0.0 {
                u32::try_from(*x as u64).ok()
            } else {
                None
            }
        }
        _ => None,
    }
}

pub(super) fn normalize_u64(v: &Value) -> Option<u64> {
    match v {
        Value::U64(x) => Some(*x),
        Value::U32(x) => Some(*x as u64),
        Value::I32(x) => u64::try_from(*x).ok(),
        Value::I64(x) => u64::try_from(*x).ok(),
        Value::F32(x) => {
            if x.is_finite() && *x >= 0.0 && x.fract() == 0.0 {
                Some(*x as u64)
            } else {
                None
            }
        }
        Value::F64(x) => {
            if x.is_finite() && *x >= 0.0 && x.fract() == 0.0 {
                Some(*x as u64)
            } else {
                None
            }
        }
        _ => None,
    }
}

pub(super) fn normalize_f16(v: &Value) -> Option<f16> {
    let f = normalize_f64(v)?;
    if !f.is_finite() {
        return None;
    }
    let f32v = f as f32;
    if !f32v.is_finite() {
        return None;
    }
    Some(f16::from_f32(f32v))
}

pub(super) fn normalize_f32(v: &Value) -> Option<f32> {
    let f = normalize_f64(v)?;
    if !f.is_finite() {
        return None;
    }
    let f32v = f as f32;
    if !f32v.is_finite() {
        return None;
    }
    Some(f32v)
}

pub(super) fn normalize_f64(v: &Value) -> Option<f64> {
    let f = crate::sqlengine::executor::to_f64(v)?;

    // Canonicalize -0.0 to +0.0. This avoids producing a distinct key
    // for values that are equal under normal float equality semantics.
    if f == 0.0 {
        Some(0.0)
    } else {
        Some(f)
    }
}

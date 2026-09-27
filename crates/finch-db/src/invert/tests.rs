use super::*;
use std::fs;

fn temp_db_dir(name: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("finch_invert_test_{}_{}", name, nanos));
    p
}

#[test]
fn test_lookup_eq_string_is_exact_not_prefix() {
    let path = temp_db_dir("lookup_eq_string_exact");
    fs::create_dir_all(&path).unwrap();

    let idx = InvertIndex::open(
        &path,
        "tag".to_string(),
        DataType::String,
        InvertIndexParams::default(),
    )
    .unwrap();
    idx.insert(1, &Value::String("a".to_string())).unwrap();
    idx.insert(2, &Value::String("ab".to_string())).unwrap();

    let bm = idx.lookup_eq(&Value::String("a".to_string())).unwrap();
    assert!(bm.contains(1));
    assert!(!bm.contains(2));

    let _ = fs::remove_dir_all(&path);
}

#[test]
fn test_lookup_range_bounded_early_exits() {
    let path = temp_db_dir("lookup_range_bounded");
    fs::create_dir_all(&path).unwrap();

    let idx = InvertIndex::open(
        &path,
        "n".to_string(),
        DataType::Int32,
        InvertIndexParams {
            enable_range_optimization: true,
            enable_extended_wildcard: false,
        },
    )
    .unwrap();

    for i in 0..10 {
        idx.insert(i as u64, &Value::I32(i)).unwrap();
    }

    // Bound smaller than match count => returns None (caller should fall back).
    let res = idx
        .lookup_lt_bounded(&Value::I32(100), true, Some(2))
        .unwrap();
    assert!(res.is_none());

    // Large enough bound => returns a bitmap.
    let res = idx
        .lookup_lt_bounded(&Value::I32(3), false, Some(10))
        .unwrap()
        .unwrap();
    assert!(res.contains(0));
    assert!(res.contains(1));
    assert!(res.contains(2));
    assert!(!res.contains(3));

    let _ = fs::remove_dir_all(&path);
}

#[test]
fn test_array_field_indexes_elements_and_length() {
    let path = temp_db_dir("array_string_elements");
    fs::create_dir_all(&path).unwrap();

    let idx = InvertIndex::open(
        &path,
        "tags".to_string(),
        DataType::ArrayString,
        InvertIndexParams::default(),
    )
    .unwrap();

    idx.insert(
        1,
        &Value::ArrayString(vec!["a".to_string(), "b".to_string()]),
    )
    .unwrap();
    idx.insert(2, &Value::ArrayString(vec!["b".to_string()]))
        .unwrap();

    let bm_a = idx.lookup_eq(&Value::String("a".to_string())).unwrap();
    assert!(bm_a.contains(1));
    assert!(!bm_a.contains(2));

    let bm_b = idx.lookup_eq(&Value::String("b".to_string())).unwrap();
    assert!(bm_b.contains(1));
    assert!(bm_b.contains(2));

    let len2 = idx.lookup_array_len_eq(2).unwrap();
    assert!(len2.contains(1));
    assert!(!len2.contains(2));

    let _ = fs::remove_dir_all(&path);
}

#[test]
fn test_array_len_range_is_bounded() {
    let path = temp_db_dir("array_len_bounded");
    fs::create_dir_all(&path).unwrap();

    let idx = InvertIndex::open(
        &path,
        "xs".to_string(),
        DataType::ArrayInt32,
        InvertIndexParams {
            enable_range_optimization: true,
            enable_extended_wildcard: false,
        },
    )
    .unwrap();

    for i in 0..10u32 {
        let items: Vec<i32> = (0..i).map(|v| v as i32).collect();
        idx.insert(100 + (i as u64), &Value::ArrayI32(items))
            .unwrap();
    }

    // This would match most docs; bounded lookup should early-exit.
    let res = idx.lookup_array_len_gt_bounded(0, true, Some(2)).unwrap();
    assert!(res.is_none());

    // Small range with enough headroom should return a bitmap.
    let bm = idx
        .lookup_array_len_lt_bounded(3, false, Some(10))
        .unwrap()
        .unwrap();
    // lengths 0,1,2 match
    assert!(bm.contains(100));
    assert!(bm.contains(101));
    assert!(bm.contains(102));
    assert!(!bm.contains(103));

    let _ = fs::remove_dir_all(&path);
}

#[test]
fn test_lookup_range_float32_orders_negatives_correctly() {
    let path = temp_db_dir("lookup_range_f32_neg");
    fs::create_dir_all(&path).unwrap();

    let idx = InvertIndex::open(
        &path,
        "x".to_string(),
        DataType::Float32,
        InvertIndexParams {
            enable_range_optimization: true,
            enable_extended_wildcard: false,
        },
    )
    .unwrap();

    let vals = [-2.0f32, -1.0, 0.0, 1.0, 2.0];
    for (doc_id, v) in vals.iter().enumerate() {
        idx.insert(doc_id as u64, &Value::F32(*v)).unwrap();
        idx.insert_nonnull_marker(doc_id as u64).unwrap();
    }

    // x < 0 should match the two negative docs.
    let bm = idx.lookup_lt(&Value::F32(0.0), false).unwrap();
    assert!(bm.contains(0));
    assert!(bm.contains(1));
    assert!(!bm.contains(2));
    assert!(!bm.contains(3));
    assert!(!bm.contains(4));

    // x >= 0 should match 0,1,2.
    let bm = idx.lookup_gt(&Value::F32(0.0), true).unwrap();
    assert!(!bm.contains(0));
    assert!(!bm.contains(1));
    assert!(bm.contains(2));
    assert!(bm.contains(3));
    assert!(bm.contains(4));

    let _ = fs::remove_dir_all(&path);
}

#[test]
fn test_float_negative_zero_is_canonicalized_for_equality() {
    let path = temp_db_dir("float_neg_zero_eq");
    fs::create_dir_all(&path).unwrap();

    let idx = InvertIndex::open(
        &path,
        "x".to_string(),
        DataType::Float32,
        InvertIndexParams {
            enable_range_optimization: true,
            enable_extended_wildcard: false,
        },
    )
    .unwrap();

    idx.insert(1, &Value::F32(-0.0)).unwrap();
    idx.insert(2, &Value::F32(0.0)).unwrap();

    let bm = idx.lookup_eq(&Value::F32(0.0)).unwrap();
    assert!(bm.contains(1));
    assert!(bm.contains(2));

    let bm = idx.lookup_eq(&Value::F32(-0.0)).unwrap();
    assert!(bm.contains(1));
    assert!(bm.contains(2));

    let norm = idx.normalize_value(&Value::F64(-0.0)).unwrap();
    assert!(matches!(norm, Value::F32(v) if v == 0.0));
    let bm = idx.lookup_eq(&norm).unwrap();
    assert!(bm.contains(1));
    assert!(bm.contains(2));

    let _ = fs::remove_dir_all(&path);
}

#[test]
fn test_normalize_value_allows_float_prefilter_literals() {
    let path = temp_db_dir("normalize_float");
    fs::create_dir_all(&path).unwrap();

    let idx = InvertIndex::open(
        &path,
        "x".to_string(),
        DataType::Float32,
        InvertIndexParams {
            enable_range_optimization: true,
            enable_extended_wildcard: false,
        },
    )
    .unwrap();

    idx.insert(1, &Value::F32(3.0)).unwrap();
    idx.insert_nonnull_marker(1).unwrap();

    let norm = idx.normalize_value(&Value::F64(3.0)).unwrap();
    assert!(matches!(norm, Value::F32(v) if v == 3.0));

    let bm = idx.lookup_eq(&norm).unwrap();
    assert!(bm.contains(1));

    let _ = fs::remove_dir_all(&path);
}

#[test]
fn test_lookup_ne_excludes_nulls_like_sql_semantics() {
    let path = temp_db_dir("lookup_ne_nulls");
    fs::create_dir_all(&path).unwrap();

    let idx = InvertIndex::open(
        &path,
        "x".to_string(),
        DataType::Int32,
        InvertIndexParams {
            enable_range_optimization: true,
            enable_extended_wildcard: false,
        },
    )
    .unwrap();

    idx.insert_null_marker(0).unwrap();
    idx.insert(1, &Value::I32(1)).unwrap();
    idx.insert_nonnull_marker(1).unwrap();
    idx.insert(2, &Value::I32(2)).unwrap();
    idx.insert_nonnull_marker(2).unwrap();

    let bm = idx.lookup_ne(&Value::I32(1)).unwrap();
    assert!(!bm.contains(0));
    assert!(!bm.contains(1));
    assert!(bm.contains(2));

    let _ = fs::remove_dir_all(&path);
}

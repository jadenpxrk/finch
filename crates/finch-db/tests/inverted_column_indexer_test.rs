use std::fs;

use finch_db::invert::InvertIndex;
use finch_types::{DataType, InvertIndexParams, Value};

fn temp_db_dir(name: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("finch_invert_indexer_{}_{}", name, nanos));
    p
}

fn generate_string(doc_id: u32) -> String {
    let prefix = match doc_id % 4 {
        0 => "One",
        1 => "Two",
        2 => "Three",
        _ => "Four",
    };
    let suffix = format!("{:02}", doc_id % 20);
    format!("{prefix}_{suffix}")
}

fn generate_number_array(doc_id: u32) -> (u32, Vec<u64>) {
    let mut out: Vec<u64> = (0..5).map(|i| (doc_id + i) as u64).collect();
    if doc_id > 999 {
        out.push((doc_id + 5) as u64);
    }
    (out.len() as u32, out)
}

fn value_array_i32(doc_id: u32) -> (u32, Value) {
    let (len, arr) = generate_number_array(doc_id);
    let v = Value::ArrayI32(arr.into_iter().map(|x| x as i32).collect());
    (len, v)
}

fn value_array_i64(doc_id: u32) -> (u32, Value) {
    let (len, arr) = generate_number_array(doc_id);
    let v = Value::ArrayI64(arr.into_iter().map(|x| x as i64).collect());
    (len, v)
}

fn value_array_u32(doc_id: u32) -> (u32, Value) {
    let (len, arr) = generate_number_array(doc_id);
    let v = Value::ArrayU32(arr.into_iter().map(|x| x as u32).collect());
    (len, v)
}

fn value_array_u64(doc_id: u32) -> (u32, Value) {
    let (len, arr) = generate_number_array(doc_id);
    let v = Value::ArrayU64(arr);
    (len, v)
}

fn run_array_numbers_case(
    case_name: &str,
    data_type: DataType,
    array_value: fn(u32) -> (u32, Value),
    scalar_value: fn(u32) -> Value,
) {
    let path = temp_db_dir(case_name);
    fs::create_dir_all(&path).unwrap();

    let params = InvertIndexParams {
        enable_range_optimization: true,
        enable_extended_wildcard: false,
    };
    let mut idx = InvertIndex::open(&path, "xs".to_string(), data_type, params.clone()).unwrap();

    let n: u32 = 2000;
    let nulls = n / 100;
    let nonnull = n - nulls;

    for doc_id in 0..n {
        if doc_id % 100 == 0 {
            idx.insert_null_marker(doc_id as u64).unwrap();
        } else {
            let (_len, v) = array_value(doc_id);
            idx.insert(doc_id as u64, &v).unwrap();
            idx.insert_nonnull_marker(doc_id as u64).unwrap();
        }
    }

    // Non-existent value.
    let v = scalar_value(n + 100);
    assert!(idx
        .lookup_contain_any(std::slice::from_ref(&v))
        .unwrap()
        .is_empty());

    // Contain value "2".
    let v2 = scalar_value(2);
    let bm = idx.lookup_contain_any(std::slice::from_ref(&v2)).unwrap();
    assert_eq!(bm.len(), 2);
    assert!(bm.contains(1));
    assert!(bm.contains(2));
    let bm = idx.lookup_contain_all(std::slice::from_ref(&v2)).unwrap();
    assert_eq!(bm.len(), 2);
    assert!(bm.contains(1));
    assert!(bm.contains(2));

    // Contain any of "2", "3", "10".
    let bm = idx
        .lookup_contain_any(&[scalar_value(2), scalar_value(3), scalar_value(10)])
        .unwrap();
    assert_eq!(bm.len(), 8);
    for id in [1u64, 2, 3, 6, 7, 8, 9, 10] {
        assert!(bm.contains(id));
    }
    let bm = idx
        .lookup_contain_all(&[scalar_value(2), scalar_value(3), scalar_value(10)])
        .unwrap();
    assert!(bm.is_empty());

    // Contain any/all of "3" and "6".
    let bm = idx
        .lookup_contain_any(&[scalar_value(3), scalar_value(6)])
        .unwrap();
    assert_eq!(bm.len(), 6);
    for id in 1u64..=6u64 {
        assert!(bm.contains(id));
    }
    let bm = idx
        .lookup_contain_all(&[scalar_value(3), scalar_value(6)])
        .unwrap();
    assert_eq!(bm.len(), 2);
    assert!(bm.contains(2));
    assert!(bm.contains(3));

    // Not contain value "1".
    let bm = idx.lookup_not_contain_any(&[scalar_value(1)]).unwrap();
    assert_eq!(bm.len(), (nonnull - 1) as u64);
    assert!(!bm.contains(0));
    assert!(!bm.contains(1));
    let bm = idx.lookup_not_contain_all(&[scalar_value(1)]).unwrap();
    assert_eq!(bm.len(), (nonnull - 1) as u64);
    assert!(!bm.contains(1));

    // Not contain any of "10" and "14".
    let bm = idx
        .lookup_not_contain_any(&[scalar_value(10), scalar_value(14)])
        .unwrap();
    assert_eq!(bm.len(), (nonnull - 9) as u64);
    for id in 6u64..=14u64 {
        assert!(!bm.contains(id));
    }
    let bm = idx
        .lookup_not_contain_all(&[scalar_value(10), scalar_value(14)])
        .unwrap();
    assert_eq!(bm.len(), (nonnull - 1) as u64);
    assert!(!bm.contains(10));

    // Array length queries.
    let len5_nonnull = 1000 - (1000 / 100);
    let len6_nonnull = nonnull - len5_nonnull;
    assert_eq!(
        idx.lookup_array_len_eq(5).unwrap().len(),
        len5_nonnull as u64
    );
    assert_eq!(
        idx.lookup_array_len_ne(5).unwrap().len(),
        len6_nonnull as u64
    );
    assert_eq!(
        idx.lookup_array_len_lt(6, false).unwrap().len(),
        len5_nonnull as u64
    );
    assert_eq!(
        idx.lookup_array_len_lt(6, true).unwrap().len(),
        nonnull as u64
    );
    assert!(idx.lookup_array_len_gt(6, false).unwrap().is_empty());
    assert_eq!(
        idx.lookup_array_len_gt(6, true).unwrap().len(),
        len6_nonnull as u64
    );

    // Freeze, reopen read-only, and re-check a couple of invariants.
    idx.freeze().unwrap();
    drop(idx);
    let idx = InvertIndex::open_read_only(&path, "xs".to_string(), data_type, params).unwrap();
    assert_eq!(
        idx.lookup_array_len_eq(5).unwrap().len(),
        len5_nonnull as u64
    );
    assert_eq!(idx.lookup_contain_any(&[scalar_value(2)]).unwrap().len(), 2);

    let _ = fs::remove_dir_all(&path);
}

#[test]
fn test_invert_int32_arrays_answer_contain_and_length_lookups() {
    run_array_numbers_case(
        "array_numbers_i32",
        DataType::ArrayInt32,
        value_array_i32,
        |x| Value::I32(x as i32),
    );
}

#[test]
fn test_invert_int64_arrays_answer_contain_and_length_lookups() {
    run_array_numbers_case(
        "array_numbers_i64",
        DataType::ArrayInt64,
        value_array_i64,
        |x| Value::I64(x as i64),
    );
}

#[test]
fn test_invert_uint32_arrays_answer_contain_and_length_lookups() {
    run_array_numbers_case(
        "array_numbers_u32",
        DataType::ArrayUint32,
        value_array_u32,
        Value::U32,
    );
}

#[test]
fn test_invert_uint64_arrays_answer_contain_and_length_lookups() {
    run_array_numbers_case(
        "array_numbers_u64",
        DataType::ArrayUint64,
        value_array_u64,
        |x| Value::U64(x as u64),
    );
}

#[test]
fn test_invert_int32_eq_and_range_lookups_exclude_null_docs() {
    let path = temp_db_dir("seq_i32_nulls");
    fs::create_dir_all(&path).unwrap();

    let mut idx = InvertIndex::open(
        &path,
        "age".to_string(),
        DataType::Int32,
        InvertIndexParams {
            enable_range_optimization: true,
            enable_extended_wildcard: false,
        },
    )
    .unwrap();

    let n: u32 = 200;
    for doc_id in 0..n {
        if doc_id % 100 == 0 {
            idx.insert_null_marker(doc_id as u64).unwrap();
        } else {
            idx.insert(doc_id as u64, &Value::I32(doc_id as i32))
                .unwrap();
            idx.insert_nonnull_marker(doc_id as u64).unwrap();
        }
    }

    // EQ: null docs have no term entry.
    assert!(idx.lookup_eq(&Value::I32(0)).unwrap().is_empty());
    assert!(idx.lookup_eq(&Value::I32(100)).unwrap().is_empty());

    for doc_id in [1u32, 2, 50, 99, 101, 150, 199] {
        let bm = idx.lookup_eq(&Value::I32(doc_id as i32)).unwrap();
        assert!(bm.contains(doc_id as u64));
        assert_eq!(bm.len(), 1);
    }

    // Null markers.
    let nulls = idx.lookup_is_null().unwrap();
    assert!(nulls.contains(0));
    assert!(nulls.contains(100));
    assert_eq!(nulls.len(), 2);

    let not_null = idx.lookup_is_not_null().unwrap();
    assert!(!not_null.contains(0));
    assert!(!not_null.contains(100));
    assert!(not_null.contains(1));
    assert_eq!(not_null.len(), (n - 2) as u64);

    // Range: x < 10 should include 1..9 (0 is NULL).
    let lt10 = idx.lookup_lt(&Value::I32(10), false).unwrap();
    for doc_id in 0..10u32 {
        let expected = doc_id != 0;
        assert_eq!(lt10.contains(doc_id as u64), expected);
    }

    // Range: x <= 1 should include only doc 1 (0 is NULL).
    let le1 = idx.lookup_lt(&Value::I32(1), true).unwrap();
    assert!(!le1.contains(0));
    assert!(le1.contains(1));
    assert!(!le1.contains(2));

    // Range: x >= 198 should include 198 and 199.
    let ge198 = idx.lookup_gt(&Value::I32(198), true).unwrap();
    assert!(!ge198.contains(197));
    assert!(ge198.contains(198));
    assert!(ge198.contains(199));

    let _ = fs::remove_dir_all(&path);
}

#[test]
fn test_invert_strings_answer_prefix_suffix_and_range_lookups() {
    let path = temp_db_dir("strings_like");
    fs::create_dir_all(&path).unwrap();

    let mut idx = InvertIndex::open(
        &path,
        "tag".to_string(),
        DataType::String,
        InvertIndexParams {
            enable_range_optimization: true,
            enable_extended_wildcard: true,
        },
    )
    .unwrap();

    let n: u32 = 200;
    for doc_id in 0..n {
        idx.insert(doc_id as u64, &Value::String(generate_string(doc_id)))
            .unwrap();
    }

    // HAS_PREFIX: "Three" matches doc_id % 4 == 2.
    let three = idx.lookup_prefix("Three").unwrap();
    assert_eq!(three.len(), (n / 4) as u64);
    for doc_id in 0..n {
        assert_eq!(three.contains(doc_id as u64), doc_id % 4 == 2);
    }

    // HAS_SUFFIX: "06" matches doc_id % 20 == 6.
    let s06 = idx.lookup_suffix("06").unwrap();
    assert_eq!(s06.len(), (n / 20) as u64);
    for doc_id in 0..n {
        assert_eq!(s06.contains(doc_id as u64), doc_id % 20 == 6);
    }

    // HAS_SUFFIX: "6" matches suffix "06" or "16" => doc_id % 20 in {6,16}.
    let s6 = idx.lookup_suffix("6").unwrap();
    assert_eq!(s6.len(), (n / 10) as u64);
    for doc_id in 0..n {
        assert_eq!(
            s6.contains(doc_id as u64),
            doc_id % 20 == 6 || doc_id % 20 == 16
        );
    }

    // Non-existent suffix.
    let s21 = idx.lookup_suffix("21").unwrap();
    assert!(s21.is_empty());

    // Range: "Two" is greater than the other prefixes in lexicographic order.
    let lt_two = idx
        .lookup_lt(&Value::String("Two".to_string()), false)
        .unwrap();
    assert_eq!(lt_two.len(), (n / 4 * 3) as u64);
    for doc_id in 0..n {
        assert_eq!(lt_two.contains(doc_id as u64), doc_id % 4 != 1);
    }

    let _ = fs::remove_dir_all(&path);
}

fn generate_bool(doc_id: u32) -> bool {
    doc_id.is_multiple_of(2)
}

fn generate_bool_array(doc_id: u32) -> Vec<bool> {
    match doc_id % 10 {
        0 => vec![true],
        1 => vec![true, true],
        2 => vec![true, false],
        3 => vec![true, true, true],
        4 => vec![false, false, false],
        5 => vec![false, true, false],
        6 => vec![true, true, true, true],
        7 => vec![false, false, false, false],
        8 => vec![true, false, true, false],
        9 => vec![false, true, false, true],
        _ => Vec::new(),
    }
}

#[test]
fn test_invert_bools_and_bool_arrays_answer_eq_ne_and_contain_lookups() {
    let params = InvertIndexParams {
        enable_range_optimization: true,
        enable_extended_wildcard: false,
    };

    let path = temp_db_dir("bools_scalar");
    fs::create_dir_all(&path).unwrap();

    let mut idx =
        InvertIndex::open(&path, "b".to_string(), DataType::Bool, params.clone()).unwrap();
    let n: u32 = 1000;
    for doc_id in 0..n {
        idx.insert(doc_id as u64, &Value::Bool(generate_bool(doc_id)))
            .unwrap();
        idx.insert_nonnull_marker(doc_id as u64).unwrap();
    }

    let trues = idx.lookup_eq(&Value::Bool(true)).unwrap();
    assert_eq!(trues.len(), (n / 2) as u64);
    for doc_id in 0..n {
        assert_eq!(trues.contains(doc_id as u64), doc_id % 2 == 0);
    }

    let ne_false = idx.lookup_ne(&Value::Bool(false)).unwrap();
    assert_eq!(ne_false.len(), (n / 2) as u64);
    for doc_id in 0..n {
        assert_eq!(ne_false.contains(doc_id as u64), doc_id % 2 == 0);
    }

    idx.freeze().unwrap();

    drop(idx);
    let idx = InvertIndex::open_read_only(&path, "b".to_string(), DataType::Bool, params.clone())
        .unwrap();
    assert_eq!(
        idx.lookup_eq(&Value::Bool(true)).unwrap().len(),
        (n / 2) as u64
    );
    drop(idx);
    let _ = fs::remove_dir_all(&path);

    // Bool arrays.
    let path = temp_db_dir("bool_arrays");
    fs::create_dir_all(&path).unwrap();

    let mut idx =
        InvertIndex::open(&path, "ba".to_string(), DataType::ArrayBool, params.clone()).unwrap();
    for doc_id in 0..n {
        idx.insert(
            doc_id as u64,
            &Value::ArrayBool(generate_bool_array(doc_id)),
        )
        .unwrap();
        idx.insert_nonnull_marker(doc_id as u64).unwrap();
    }

    let contain_true = idx.lookup_contain_all(&[Value::Bool(true)]).unwrap();
    assert_eq!(contain_true.len(), (n / 10 * 8) as u64);
    for doc_id in 0..n {
        assert_eq!(
            contain_true.contains(doc_id as u64),
            doc_id % 10 != 4 && doc_id % 10 != 7
        );
    }

    let contain_true_any = idx.lookup_contain_any(&[Value::Bool(true)]).unwrap();
    assert_eq!(contain_true_any.len(), contain_true.len());
    for doc_id in 0..n {
        assert_eq!(
            contain_true_any.contains(doc_id as u64),
            doc_id % 10 != 4 && doc_id % 10 != 7
        );
    }

    let contain_both = idx
        .lookup_contain_all(&[Value::Bool(true), Value::Bool(false)])
        .unwrap();
    assert_eq!(contain_both.len(), (n / 10 * 4) as u64);
    for doc_id in 0..n {
        let expected = matches!(doc_id % 10, 2 | 5 | 8 | 9);
        assert_eq!(contain_both.contains(doc_id as u64), expected);
    }

    let contain_any = idx
        .lookup_contain_any(&[Value::Bool(true), Value::Bool(false)])
        .unwrap();
    assert_eq!(contain_any.len(), n as u64);

    assert_eq!(idx.lookup_array_len_eq(1).unwrap().len(), (n / 10) as u64);
    assert_eq!(
        idx.lookup_array_len_eq(2).unwrap().len(),
        (n / 10 * 2) as u64
    );
    assert_eq!(
        idx.lookup_array_len_eq(3).unwrap().len(),
        (n / 10 * 3) as u64
    );
    assert_eq!(
        idx.lookup_array_len_eq(4).unwrap().len(),
        (n / 10 * 4) as u64
    );

    assert_eq!(idx.lookup_array_len_ne(5).unwrap().len(), n as u64);
    assert_eq!(
        idx.lookup_array_len_ne(3).unwrap().len(),
        (n / 10 * 7) as u64
    );

    assert!(idx.lookup_array_len_lt(1, false).unwrap().is_empty());
    assert_eq!(
        idx.lookup_array_len_lt(1, true).unwrap().len(),
        (n / 10) as u64
    );
    assert_eq!(
        idx.lookup_array_len_lt(4, false).unwrap().len(),
        (n / 10 * 6) as u64
    );
    assert_eq!(idx.lookup_array_len_lt(4, true).unwrap().len(), n as u64);

    assert_eq!(
        idx.lookup_array_len_gt(1, false).unwrap().len(),
        (n / 10 * 9) as u64
    );
    assert_eq!(idx.lookup_array_len_gt(1, true).unwrap().len(), n as u64);
    assert!(idx.lookup_array_len_gt(4, false).unwrap().is_empty());
    assert_eq!(
        idx.lookup_array_len_gt(4, true).unwrap().len(),
        (n / 10 * 4) as u64
    );

    idx.freeze().unwrap();
    drop(idx);
    let idx =
        InvertIndex::open_read_only(&path, "ba".to_string(), DataType::ArrayBool, params).unwrap();
    assert_eq!(
        idx.lookup_array_len_eq(4).unwrap().len(),
        (n / 10 * 4) as u64
    );

    let _ = fs::remove_dir_all(&path);
}

fn cyclic_value_i32(doc_id: u32) -> i32 {
    let base = (doc_id / 100) * 100;
    (base + (doc_id % 10)) as i32
}

fn cyclic_value_u64(doc_id: u32) -> u64 {
    let base = (doc_id / 100) * 100;
    (base + (doc_id % 10)) as u64
}

fn cyclic_value_f32(doc_id: u32) -> f32 {
    let base = (doc_id / 100) * 100;
    (base as f32) + (doc_id % 10) as f32 + 0.666
}

fn cyclic_value_f64(doc_id: u32) -> f64 {
    let base = (doc_id / 100) * 100;
    (base as f64) + (doc_id % 10) as f64 + 0.666
}

fn run_cyclic_case(
    case_name: &str,
    data_type: DataType,
    insert_value: fn(u32) -> Value,
    existing_value: fn(u32) -> Value,
    literal_value: fn(u32) -> Value,
    include_nulls: bool,
) {
    let path = temp_db_dir(case_name);
    fs::create_dir_all(&path).unwrap();

    let params = InvertIndexParams {
        enable_range_optimization: true,
        enable_extended_wildcard: false,
    };
    let mut idx = InvertIndex::open(&path, "x".to_string(), data_type, params).unwrap();

    let n: u32 = 1000;
    let nulls = if include_nulls { n / 100 } else { 0 };
    let nonnull = n - nulls;
    for doc_id in 0..n {
        if include_nulls && doc_id % 100 == 0 {
            idx.insert_null_marker(doc_id as u64).unwrap();
        } else {
            idx.insert(doc_id as u64, &insert_value(doc_id)).unwrap();
            idx.insert_nonnull_marker(doc_id as u64).unwrap();
        }
    }

    // EQ checks per cycle (first and 4th value) and a non-existent value.
    for cycle in 0..(n / 100) {
        let first_doc = cycle * 100;
        let bm = idx.lookup_eq(&existing_value(first_doc)).unwrap();
        let expected = if include_nulls { 9 } else { 10 };
        assert_eq!(bm.len(), expected);
        for j in 0..10 {
            let id = first_doc + j * 10;
            let should = !(include_nulls && id % 100 == 0);
            assert_eq!(bm.contains(id as u64), should);
        }

        let bm = idx.lookup_eq(&existing_value(first_doc + 3)).unwrap();
        assert_eq!(bm.len(), 10);
        for j in 0..10 {
            assert!(bm.contains((first_doc + 3 + j * 10) as u64));
        }

        let bm = idx.lookup_eq(&literal_value(first_doc + 11)).unwrap();
        assert!(bm.is_empty());
    }

    // NE with a non-existent value returns all non-null docs.
    let bm = idx.lookup_ne(&existing_value(n)).unwrap();
    assert_eq!(bm.len(), nonnull as u64);
    for doc_id in 0..n {
        assert_eq!(
            bm.contains(doc_id as u64),
            !(include_nulls && doc_id % 100 == 0)
        );
    }

    // NE with a value existing in a random-ish cycle.
    let random_cycle = (n / 100) / 2;
    let v = existing_value(random_cycle * 100 + 1);
    let bm = idx.lookup_ne(&v).unwrap();
    for doc_id in 0..n {
        let is_null = include_nulls && doc_id % 100 == 0;
        if is_null {
            assert!(!bm.contains(doc_id as u64));
            continue;
        }
        if (doc_id / 100) == random_cycle && (doc_id % 10) == 1 {
            assert!(!bm.contains(doc_id as u64));
        } else {
            assert!(bm.contains(doc_id as u64));
        }
    }

    // Range semantics for min.
    let min_v = existing_value(0);
    assert!(idx.lookup_lt(&min_v, false).unwrap().is_empty());
    assert_eq!(
        idx.lookup_lt(&min_v, true).unwrap().len(),
        (if include_nulls { 9 } else { 10 }) as u64
    );
    assert_eq!(
        idx.lookup_gt(&min_v, false).unwrap().len(),
        (nonnull - (if include_nulls { 9 } else { 10 })) as u64
    );
    assert_eq!(idx.lookup_gt(&min_v, true).unwrap().len(), nonnull as u64);

    // Range semantics for a mid-cycle threshold (cycle_base + 1).
    let mid = (n / 100) / 2;
    let mid_v = existing_value(mid * 100 + 1);
    let lt = idx.lookup_lt(&mid_v, false).unwrap();
    for doc_id in 0..n {
        let is_null = include_nulls && doc_id % 100 == 0;
        if is_null {
            assert!(!lt.contains(doc_id as u64));
            continue;
        }
        let cycle = doc_id / 100;
        let rem = doc_id % 10;
        let expected = if cycle < mid {
            true
        } else if cycle == mid {
            rem < 1
        } else {
            false
        };
        assert_eq!(lt.contains(doc_id as u64), expected);
    }

    let le = idx.lookup_lt(&mid_v, true).unwrap();
    for doc_id in 0..n {
        let is_null = include_nulls && doc_id % 100 == 0;
        if is_null {
            assert!(!le.contains(doc_id as u64));
            continue;
        }
        let cycle = doc_id / 100;
        let rem = doc_id % 10;
        let expected = if cycle < mid {
            true
        } else if cycle == mid {
            rem <= 1
        } else {
            false
        };
        assert_eq!(le.contains(doc_id as u64), expected);
    }

    let gt = idx.lookup_gt(&mid_v, false).unwrap();
    for doc_id in 0..n {
        let is_null = include_nulls && doc_id % 100 == 0;
        if is_null {
            assert!(!gt.contains(doc_id as u64));
            continue;
        }
        let cycle = doc_id / 100;
        let rem = doc_id % 10;
        let expected = if cycle < mid {
            false
        } else if cycle == mid {
            rem > 1
        } else {
            true
        };
        assert_eq!(gt.contains(doc_id as u64), expected);
    }

    let ge = idx.lookup_gt(&mid_v, true).unwrap();
    for doc_id in 0..n {
        let is_null = include_nulls && doc_id % 100 == 0;
        if is_null {
            assert!(!ge.contains(doc_id as u64));
            continue;
        }
        let cycle = doc_id / 100;
        let rem = doc_id % 10;
        let expected = if cycle < mid {
            false
        } else if cycle == mid {
            rem >= 1
        } else {
            true
        };
        assert_eq!(ge.contains(doc_id as u64), expected);
    }

    // Range semantics for max.
    let max_v = existing_value(n - 1);
    let max_eq = idx.lookup_eq(&max_v).unwrap().len() as u32;
    assert_eq!(max_eq, 10);
    assert_eq!(
        idx.lookup_lt(&max_v, false).unwrap().len(),
        (nonnull - max_eq) as u64
    );
    assert_eq!(idx.lookup_lt(&max_v, true).unwrap().len(), nonnull as u64);
    assert!(idx.lookup_gt(&max_v, false).unwrap().is_empty());
    assert_eq!(idx.lookup_gt(&max_v, true).unwrap().len(), max_eq as u64);

    let _ = fs::remove_dir_all(&path);
}

#[test]
fn test_invert_cyclic_int32_answer_eq_ne_and_range_lookups() {
    run_cyclic_case(
        "cyclic_i32_no_null",
        DataType::Int32,
        |id| Value::I32(cyclic_value_i32(id)),
        |id| Value::I32(cyclic_value_i32(id)),
        |id| Value::I32(id as i32),
        false,
    );
    run_cyclic_case(
        "cyclic_i32_with_null",
        DataType::Int32,
        |id| Value::I32(cyclic_value_i32(id)),
        |id| Value::I32(cyclic_value_i32(id)),
        |id| Value::I32(id as i32),
        true,
    );
}

#[test]
fn test_invert_cyclic_uint64_answer_eq_ne_and_range_lookups() {
    run_cyclic_case(
        "cyclic_u64_no_null",
        DataType::Uint64,
        |id| Value::U64(cyclic_value_u64(id)),
        |id| Value::U64(cyclic_value_u64(id)),
        |id| Value::U64(id as u64),
        false,
    );
    run_cyclic_case(
        "cyclic_u64_with_null",
        DataType::Uint64,
        |id| Value::U64(cyclic_value_u64(id)),
        |id| Value::U64(cyclic_value_u64(id)),
        |id| Value::U64(id as u64),
        true,
    );
}

#[test]
fn test_invert_cyclic_floats_answer_eq_ne_and_range_lookups() {
    run_cyclic_case(
        "cyclic_f32_no_null",
        DataType::Float32,
        |id| Value::F32(cyclic_value_f32(id)),
        |id| Value::F32(cyclic_value_f32(id)),
        |id| Value::F32(id as f32),
        false,
    );
    run_cyclic_case(
        "cyclic_f32_with_null",
        DataType::Float32,
        |id| Value::F32(cyclic_value_f32(id)),
        |id| Value::F32(cyclic_value_f32(id)),
        |id| Value::F32(id as f32),
        true,
    );
    run_cyclic_case(
        "cyclic_f64_no_null",
        DataType::Float64,
        |id| Value::F64(cyclic_value_f64(id)),
        |id| Value::F64(cyclic_value_f64(id)),
        |id| Value::F64(id as f64),
        false,
    );
    run_cyclic_case(
        "cyclic_f64_with_null",
        DataType::Float64,
        |id| Value::F64(cyclic_value_f64(id)),
        |id| Value::F64(cyclic_value_f64(id)),
        |id| Value::F64(id as f64),
        true,
    );
}

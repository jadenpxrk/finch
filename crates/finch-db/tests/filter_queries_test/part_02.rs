use super::*;

#[test]
fn test_filter_only_query_like_semantics_match_reference() {
    use finch_types::{IndexParams, InvertIndexParams};

    let path = temp_dir("filter_only_like");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("dummy_vec", DataType::VectorFp32).with_dimension(1))
        .with_field(FieldSchema::new("name", DataType::String).not_null())
        .with_field(
            FieldSchema::new("invert_name", DataType::String)
                .not_null()
                .with_index(IndexParams::Invert(InvertIndexParams {
                    enable_range_optimization: false,
                    enable_extended_wildcard: false,
                })),
        )
        .with_field(
            FieldSchema::new("extended_invert_name", DataType::String)
                .not_null()
                .with_index(IndexParams::Invert(InvertIndexParams {
                    enable_range_optimization: false,
                    enable_extended_wildcard: true,
                })),
        );

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    fn make_doc(i: u64) -> Doc {
        let mut name = String::from("user-");
        if (5000..8000).contains(&i) {
            name.push('%');
        } else if i >= 8000 {
            name.push('_');
        }
        name.push_str(&(i % 100).to_string());

        Doc::new(format!("pk_{i}"))
            .set("name", name.clone())
            .set("invert_name", name.clone())
            .set("extended_invert_name", name)
    }

    let mut batch: Vec<Doc> = Vec::new();
    for i in 0..10_000u64 {
        batch.push(make_doc(i));
        if batch.len() == 1024 {
            col.insert(std::mem::take(&mut batch)).unwrap();
        }
    }
    if !batch.is_empty() {
        col.insert(batch).unwrap();
    }

    // Force persisted path (reference tests run against persisted segments).
    col.flush().unwrap();

    // LIKE '%' matches all.
    let q = VectorQuery::new("", vec![], 200)
        .with_filter("name LIKE '%'")
        .with_output_fields(vec!["name".to_string()]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    for i in 0..200u64 {
        assert_eq!(docs[i as usize].pk, format!("pk_{i}"));
    }

    let q = VectorQuery::new("", vec![], 200)
        .with_filter("invert_name LIKE '%'")
        .with_output_fields(vec!["name".to_string()]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    for i in 0..200u64 {
        assert_eq!(docs[i as usize].pk, format!("pk_{i}"));
    }

    // Prefix: user-22% matches only docs with name "user-22..." (i < 5000).
    let q = VectorQuery::new("", vec![], 200)
        .with_filter("name LIKE 'user-22%'")
        .with_output_fields(vec!["name".to_string()]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 50);
    for i in 0..50u64 {
        let id = i * 100 + 22;
        assert_eq!(docs[i as usize].pk, format!("pk_{id}"));
    }

    // Invert prefix like: should behave identically (and can be pushed down).
    let q = VectorQuery::new("", vec![], 200)
        .with_filter("invert_name LIKE 'user-22%'")
        .with_output_fields(vec!["name".to_string()]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 50);
    for i in 0..50u64 {
        let id = i * 100 + 22;
        assert_eq!(docs[i as usize].pk, format!("pk_{id}"));
    }

    // Suffix: '%ser-22'
    let q = VectorQuery::new("", vec![], 200)
        .with_filter("name LIKE '%ser-22'")
        .with_output_fields(vec!["name".to_string()]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 50);
    for i in 0..50u64 {
        let id = i * 100 + 22;
        assert_eq!(docs[i as usize].pk, format!("pk_{id}"));
    }

    // Not-extended invert suffix like: should still behave correctly (but may run as forward).
    let q = VectorQuery::new("", vec![], 200)
        .with_filter("invert_name LIKE '%ser-22'")
        .with_output_fields(vec!["name".to_string()]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 50);
    for i in 0..50u64 {
        let id = i * 100 + 22;
        assert_eq!(docs[i as usize].pk, format!("pk_{id}"));
    }

    // Extended invert suffix like: can be pushed down.
    let q = VectorQuery::new("", vec![], 200)
        .with_filter("extended_invert_name LIKE '%ser-22'")
        .with_output_fields(vec!["name".to_string()]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 50);
    for i in 0..50u64 {
        let id = i * 100 + 22;
        assert_eq!(docs[i as usize].pk, format!("pk_{id}"));
    }

    // Middle: 'user%2' => last digit 2 (topk=200).
    let q = VectorQuery::new("", vec![], 200)
        .with_filter("name LIKE 'user%2'")
        .with_output_fields(vec!["name".to_string()]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    for i in 0..200u64 {
        let id = i * 10 + 2;
        assert_eq!(docs[i as usize].pk, format!("pk_{id}"));
    }

    let q = VectorQuery::new("", vec![], 200)
        .with_filter("extended_invert_name LIKE 'user%2'")
        .with_output_fields(vec!["name".to_string()]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    for i in 0..200u64 {
        let id = i * 10 + 2;
        assert_eq!(docs[i as usize].pk, format!("pk_{id}"));
    }

    // UnderScore: 'user-_2' => i < 5000, 2-digit suffix, last digit 2.
    let q = VectorQuery::new("", vec![], 200)
        .with_filter("name LIKE 'user-_2'")
        .with_output_fields(vec!["name".to_string()]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    let mut expected_id = 0u64;
    for doc in docs.iter() {
        while expected_id < 5000 && (expected_id % 100 % 10 != 2 || expected_id % 100 < 10) {
            expected_id += 1;
        }
        assert!(
            expected_id < 5000,
            "expected more matches before doc_id 5000"
        );
        assert_eq!(doc.pk, format!("pk_{expected_id}"));
        expected_id += 1;
    }

    // Invert UnderScore: should still behave correctly (but may run as forward).
    let q = VectorQuery::new("", vec![], 200)
        .with_filter("invert_name LIKE 'user-_2'")
        .with_output_fields(vec!["name".to_string()]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    let mut expected_id = 0u64;
    for doc in docs.iter() {
        while expected_id < 5000 && (expected_id % 100 % 10 != 2 || expected_id % 100 < 10) {
            expected_id += 1;
        }
        assert!(
            expected_id < 5000,
            "expected more matches before doc_id 5000"
        );
        assert_eq!(doc.pk, format!("pk_{expected_id}"));
        expected_id += 1;
    }

    // Escape percent: user-\%% matches doc_ids [5000..5200).
    let q = VectorQuery::new("", vec![], 200)
        .with_filter(r#"name LIKE 'user-\%%'"#)
        .with_output_fields(vec!["name".to_string()]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    for i in 0..200u64 {
        let id = 5000 + i;
        assert_eq!(docs[i as usize].pk, format!("pk_{id}"));
    }

    let q = VectorQuery::new("", vec![], 200)
        .with_filter(r#"invert_name LIKE 'user-\%%'"#)
        .with_output_fields(vec!["name".to_string()]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    for i in 0..200u64 {
        let id = 5000 + i;
        assert_eq!(docs[i as usize].pk, format!("pk_{id}"));
    }

    // Escape underscore: user-\_% matches doc_ids [8000..8200).
    let q = VectorQuery::new("", vec![], 200)
        .with_filter(r#"name LIKE 'user-\_%'"#)
        .with_output_fields(vec!["name".to_string()]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    for i in 0..200u64 {
        let id = 8000 + i;
        assert_eq!(docs[i as usize].pk, format!("pk_{id}"));
    }

    let q = VectorQuery::new("", vec![], 200)
        .with_filter(r#"invert_name LIKE 'user-\_%'"#)
        .with_output_fields(vec!["name".to_string()]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    for i in 0..200u64 {
        let id = 8000 + i;
        assert_eq!(docs[i as usize].pk, format!("pk_{id}"));
    }

    // No wildcard: runs as equality (invert_name like 'user-22').
    let q = VectorQuery::new("", vec![], 200)
        .with_filter(r#"invert_name LIKE 'user-22'"#)
        .with_output_fields(vec!["name".to_string()]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 50);
    for i in 0..50u64 {
        let id = i * 100 + 22;
        assert_eq!(docs[i as usize].pk, format!("pk_{id}"));
    }

    // upstream SQL grammar does not support `NOT LIKE`.
    let q = VectorQuery::new("", vec![], 200)
        .with_filter("name NOT LIKE '%'")
        .with_output_fields(vec![]);
    let err = col.query(q).unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument);
    assert!(err.message.contains("syntax error"));

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_query_filter_unsupported_sql_constructs_are_syntax_errors() {
    let path = temp_dir("filter_unsupported_sql_constructs");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("dummy_vec", DataType::VectorFp32).with_dimension(1))
        .with_field(FieldSchema::new("name", DataType::String))
        .with_field(FieldSchema::new("age", DataType::Int32));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    // BETWEEN is not part of the filter grammar.
    let q = VectorQuery::new("", vec![], 10)
        .with_filter("age BETWEEN 1 AND 2")
        .with_output_fields(vec![]);
    let err = col.query(q).unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument);
    assert!(err.message.contains("syntax error"));

    // string concatenation operator `||` is not part of the filter grammar.
    let q = VectorQuery::new("", vec![], 10)
        .with_filter("name = 'a' || 'b'")
        .with_output_fields(vec![]);
    let err = col.query(q).unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument);
    assert!(err.message.contains("syntax error"));

    // compound identifiers (with dot) are rejected.
    let q = VectorQuery::new("", vec![], 10)
        .with_filter("t.name = 'x'")
        .with_output_fields(vec![]);
    let err = col.query(q).unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument);
    assert!(err.message.contains("syntax error"));

    // identifier quoting is not supported.
    let q = VectorQuery::new("", vec![], 10)
        .with_filter("`name` = 'x'")
        .with_output_fields(vec![]);
    let err = col.query(q).unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument);
    assert!(err.message.contains("syntax error"));

    // `==` is not a valid equality operator.
    let q = VectorQuery::new("", vec![], 10)
        .with_filter("age == 1")
        .with_output_fields(vec![]);
    let err = col.query(q).unwrap_err();
    assert_eq!(err.code, StatusCode::InvalidArgument);
    assert!(err.message.contains("syntax error"));

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_filter_only_query_contain_all_semantics_match_reference() {
    let path = temp_dir("filter_only_contain_all");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("dummy_vec", DataType::VectorFp32).with_dimension(1))
        .with_field(FieldSchema::new("str_array", DataType::ArrayString))
        .with_field(FieldSchema::new("i32_array", DataType::ArrayInt32))
        .with_field(FieldSchema::new("i64_array", DataType::ArrayInt64))
        .with_field(FieldSchema::new("u32_array", DataType::ArrayUint32))
        .with_field(FieldSchema::new("u64_array", DataType::ArrayUint64))
        .with_field(FieldSchema::new("fp32_array", DataType::ArrayFp32))
        .with_field(FieldSchema::new("fp64_array", DataType::ArrayFp64));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    fn make_doc(i: u64) -> Doc {
        let size = (i % 100) as u32;
        let mut doc = Doc::new(format!("pk_{i}"));
        if size > 0 {
            let i32_arr: Vec<i32> = (1..=size).map(|x| x as i32).collect();
            let i64_arr: Vec<i64> = (1..=size).map(|x| x as i64).collect();
            let u32_arr: Vec<u32> = (1..=size).collect();
            let u64_arr: Vec<u64> = (1..=size).map(|x| x as u64).collect();
            let f32_arr: Vec<f32> = (1..=size).map(|x| x as f32).collect();
            let f64_arr: Vec<f64> = (1..=size).map(|x| x as f64).collect();
            let str_arr: Vec<String> = (1..=size).map(|x| format!("name{x}")).collect();

            doc = doc
                .set("str_array", Value::ArrayString(str_arr))
                .set("i32_array", Value::ArrayI32(i32_arr))
                .set("i64_array", Value::ArrayI64(i64_arr))
                .set("u32_array", Value::ArrayU32(u32_arr))
                .set("u64_array", Value::ArrayU64(u64_arr))
                .set("fp32_array", Value::ArrayF32(f32_arr))
                .set("fp64_array", Value::ArrayF64(f64_arr));
        }
        doc
    }

    let mut batch: Vec<Doc> = Vec::new();
    for i in 0..10_000u64 {
        batch.push(make_doc(i));
        if batch.len() == 1024 {
            col.insert(std::mem::take(&mut batch)).unwrap();
        }
    }
    if !batch.is_empty() {
        col.insert(batch).unwrap();
    }
    col.flush().unwrap();

    fn assert_threshold_filter(col: &Collection, filter: String, start: u64, threshold: u64) {
        let q = VectorQuery::new("", vec![], 200)
            .with_filter(filter)
            .with_output_fields(vec![]);
        let docs = col.query(q).unwrap();
        assert_eq!(docs.len(), 200);

        let mut i = start;
        for doc in docs.iter() {
            assert_eq!(doc.pk, format!("pk_{i}"));
            i += 1;
            while i % 100 < threshold {
                i += 1;
            }
        }
    }

    // contain_all (1..32): first 200 doc_ids in order where `doc_id % 100 >= 32`.
    let mut list_1_32 = String::new();
    for i in 1..=32 {
        if i > 1 {
            list_1_32.push_str(", ");
        }
        list_1_32.push_str(&i.to_string());
    }
    assert_threshold_filter(&col, format!("i32_array contain_all ({list_1_32})"), 32, 32);
    assert_threshold_filter(&col, format!("i64_array contain_all ({list_1_32})"), 32, 32);
    assert_threshold_filter(&col, format!("u32_array contain_all ({list_1_32})"), 32, 32);
    assert_threshold_filter(&col, format!("u64_array contain_all ({list_1_32})"), 32, 32);
    assert_threshold_filter(
        &col,
        format!("fp32_array contain_all ({list_1_32})"),
        32,
        32,
    );
    assert_threshold_filter(
        &col,
        format!("fp64_array contain_all ({list_1_32})"),
        32,
        32,
    );
    {
        let mut list = String::new();
        for i in 1..=32 {
            if i > 1 {
                list.push_str(", ");
            }
            list.push_str(&format!("'name{i}'"));
        }
        assert_threshold_filter(&col, format!("str_array contain_all ({list})"), 32, 32);
    }

    // contain_any (98,99,100): first 200 doc_ids where `doc_id % 100 >= 98`.
    assert_threshold_filter(
        &col,
        "i32_array contain_any (98,99,100)".to_string(),
        98,
        98,
    );
    assert_threshold_filter(
        &col,
        "i64_array contain_any (98,99,100)".to_string(),
        98,
        98,
    );
    assert_threshold_filter(
        &col,
        "u32_array contain_any (98,99,100)".to_string(),
        98,
        98,
    );
    assert_threshold_filter(
        &col,
        "u64_array contain_any (98,99,100)".to_string(),
        98,
        98,
    );
    assert_threshold_filter(
        &col,
        "fp32_array contain_any (98,99,100)".to_string(),
        98,
        98,
    );
    assert_threshold_filter(
        &col,
        "fp64_array contain_any (98,99,100)".to_string(),
        98,
        98,
    );
    assert_threshold_filter(
        &col,
        "str_array contain_any ('name98','name99','name100')".to_string(),
        98,
        98,
    );

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_filter_numeric_literals_are_normalized_no_f64_rounding() {
    let path = temp_dir("filter_normalize_u64");

    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("dummy_vec", DataType::VectorFp32).with_dimension(1))
        .with_field(FieldSchema::new("x", DataType::Uint64));
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    // Values around 2^53 to catch precision loss when comparing via f64.
    let a = 9_007_199_254_740_992u64; // 2^53
    let b = 9_007_199_254_740_993u64; // 2^53 + 1 (not exactly representable in f64)
    let c = 9_007_199_254_740_994u64;

    let statuses = col
        .insert(vec![
            Doc::new("a").set("x", a),
            Doc::new("b").set("x", b),
            Doc::new("c").set("x", c),
        ])
        .unwrap();
    assert!(statuses.iter().all(|s| s.is_ok()));

    fn pks(col: &Collection, filter: &str) -> Vec<String> {
        let q = VectorQuery::new("", vec![], 10)
            .with_filter(filter)
            .with_output_fields(vec![]);
        let mut out: Vec<String> = col.query(q).unwrap().iter().map(|d| d.pk.clone()).collect();
        out.sort();
        out
    }

    assert_eq!(pks(&col, "x = 9007199254740993"), vec!["b".to_string()]);

    col.flush().unwrap();
    assert_eq!(pks(&col, "x = 9007199254740993"), vec!["b".to_string()]);

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_filter_only_query_is_null_semantics_match_reference() {
    use finch_types::{IndexParams, InvertIndexParams};

    let path = temp_dir("filter_only_is_null");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("dummy_vec", DataType::VectorFp32).with_dimension(1))
        .with_field(FieldSchema::new("age", DataType::Int32).not_null())
        .with_field(
            FieldSchema::new("optional_age", DataType::Int32)
                .with_index(IndexParams::Invert(InvertIndexParams::default())),
        );

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    fn make_doc(i: u64) -> Doc {
        let mut doc = Doc::new(format!("pk_{i}")).set("age", (i % 100) as i32);
        if !i.is_multiple_of(100) {
            doc = doc.set("optional_age", (i % 100) as i32);
        }
        doc
    }

    let mut batch: Vec<Doc> = Vec::new();
    for i in 0..10_000u64 {
        batch.push(make_doc(i));
        if batch.len() == 1024 {
            col.insert(std::mem::take(&mut batch)).unwrap();
        }
    }
    if !batch.is_empty() {
        col.insert(batch).unwrap();
    }
    col.flush().unwrap();

    // `optional_age IS NULL` matches doc_ids 0, 100, 200, ...
    let q = VectorQuery::new("", vec![], 200)
        .with_filter("optional_age IS NULL")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 100);
    for (j, doc) in docs.iter().enumerate() {
        let id = (j as u64) * 100;
        assert_eq!(doc.pk, format!("pk_{id}"));
    }

    // `optional_age IS NOT NULL` skips multiples of 100.
    let q = VectorQuery::new("", vec![], 200)
        .with_filter("optional_age IS NOT NULL")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    let mut expected_id = 0u64;
    for doc in docs.iter() {
        expected_id += 1;
        if expected_id.is_multiple_of(100) {
            expected_id += 1;
        }
        assert_eq!(doc.pk, format!("pk_{expected_id}"));
    }

    // Non-nullable `age` should never be null.
    let q = VectorQuery::new("", vec![], 200)
        .with_filter("age IS NULL")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert!(docs.is_empty());

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_filter_only_query_str_in_and_not_in_semantics_match_reference() {
    let path = temp_dir("filter_only_str_in_not_in");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("dummy_vec", DataType::VectorFp32).with_dimension(1))
        .with_field(FieldSchema::new("id", DataType::Int64).not_null())
        .with_field(FieldSchema::new("name", DataType::String).not_null());

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    fn make_doc(i: u64) -> Doc {
        Doc::new(format!("pk_{i}"))
            .set("id", i as i64)
            .set("name", format!("user_{}", i % 100))
    }

    let mut batch: Vec<Doc> = Vec::new();
    for i in 0..10_000u64 {
        batch.push(make_doc(i));
        if batch.len() == 1024 {
            col.insert(std::mem::take(&mut batch)).unwrap();
        }
    }
    if !batch.is_empty() {
        col.insert(batch).unwrap();
    }
    col.flush().unwrap();

    // reference ForwardRecallTest.StrIn
    let q = VectorQuery::new("", vec![], 200)
        .with_filter("name IN ('user_1', 'user_2')")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    let mut expected = 1u64;
    for doc in docs.iter() {
        assert_eq!(doc.pk, format!("pk_{expected}"));
        if expected % 100 == 1 {
            expected += 1;
        } else if expected % 100 == 2 {
            expected += 99;
        } else {
            panic!("unexpected expected id state: {expected}");
        }
    }

    // reference ForwardRecallTest.StrNotIn
    let q = VectorQuery::new("", vec![], 200)
        .with_filter("name NOT IN ('user_1', 'user_2')")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    let mut expected = 0u64;
    for doc in docs.iter() {
        assert_eq!(doc.pk, format!("pk_{expected}"));
        if expected.is_multiple_of(100) {
            expected += 3;
        } else {
            expected += 1;
        }
    }

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_filter_only_query_invert_str_in_and_not_in_semantics_match_reference() {
    let path = temp_dir("filter_only_invert_str_in_not_in");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("dummy_vec", DataType::VectorFp32).with_dimension(1))
        .with_field(FieldSchema::new("id", DataType::Int64).not_null())
        .with_field(
            FieldSchema::new("invert_name", DataType::String)
                .not_null()
                .with_index(IndexParams::Invert(InvertIndexParams {
                    enable_range_optimization: false,
                    enable_extended_wildcard: false,
                })),
        );

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    fn make_doc(i: u64) -> Doc {
        Doc::new(format!("pk_{i}"))
            .set("id", i as i64)
            .set("invert_name", format!("user_{}", i % 100))
    }

    let mut batch: Vec<Doc> = Vec::new();
    for i in 0..10_000u64 {
        batch.push(make_doc(i));
        if batch.len() == 1024 {
            col.insert(std::mem::take(&mut batch)).unwrap();
        }
    }
    if !batch.is_empty() {
        col.insert(batch).unwrap();
    }
    col.flush().unwrap();

    // reference InvertRecallTest.StrIn / StrNotIn (behavior mirrors forward path).
    let q = VectorQuery::new("", vec![], 200)
        .with_filter("invert_name IN ('user_1', 'user_2')")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    let mut expected = 1u64;
    for doc in docs.iter() {
        assert_eq!(doc.pk, format!("pk_{expected}"));
        if expected % 100 == 1 {
            expected += 1;
        } else if expected % 100 == 2 {
            expected += 99;
        } else {
            panic!("unexpected expected id state: {expected}");
        }
    }

    let q = VectorQuery::new("", vec![], 200)
        .with_filter("invert_name NOT IN ('user_1', 'user_2')")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    let mut expected = 0u64;
    for doc in docs.iter() {
        assert_eq!(doc.pk, format!("pk_{expected}"));
        if expected.is_multiple_of(100) {
            expected += 3;
        } else {
            expected += 1;
        }
    }

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_filter_only_query_double_quoted_strings_work() {
    let path = temp_dir("filter_only_double_quote");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("dummy_vec", DataType::VectorFp32).with_dimension(1))
        .with_field(FieldSchema::new("name", DataType::String).not_null());

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![
        Doc::new("pk_0").set("name", "user_0"),
        Doc::new("pk_1").set("name", "user_1"),
        Doc::new("pk_2").set("name", "user_2"),
        Doc::new("pk_3").set("name", "user_3"),
    ])
    .unwrap();
    col.flush().unwrap();

    let q = VectorQuery::new("", vec![], 10)
        .with_filter(r#"name IN ("user_1", "user_3")"#)
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    let got: std::collections::HashSet<_> = docs.iter().map(|d| d.pk.as_str()).collect();
    assert_eq!(got.len(), 2);
    assert!(got.contains("pk_1"));
    assert!(got.contains("pk_3"));

    let q = VectorQuery::new("", vec![], 10)
        .with_filter(r#"name = "user_2""#)
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].pk, "pk_2");

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

mod common;
use common::*;
use finch_types::Status;

const TWO_53: i64 = 9_007_199_254_740_992;

fn scalar_collection(
    name: &str,
    schema: CollectionSchema,
) -> (std::path::PathBuf, Arc<Collection>) {
    let path = temp_dir(name);
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    (path, col)
}

fn fetch_field(col: &Collection, pk: &str, field: &str) -> Option<Value> {
    let docs = col.fetch(vec![pk.to_string()]).unwrap();
    docs[pk].fields.get(field).cloned()
}

#[test]
fn ddl_backfill_rejects_inexact_integer_results() {
    // An expression result that is fractional, overflowed, or rounded through f64 fails the DDL.
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("value", DataType::Int64))
        .with_field(FieldSchema::new("big", DataType::Uint64))
        .with_field(FieldSchema::new("ratio", DataType::Float64));
    let (path, col) = scalar_collection("int_exact_ddl", schema);
    let statuses = col
        .insert(vec![Doc::new("row")
            .set("value", TWO_53 + 1)
            .set("big", u64::MAX)
            .set("ratio", 1.5f64)])
        .unwrap();
    assert!(statuses.iter().all(Status::is_ok));

    let add = |name: &str, dt: DataType, expr: &str| {
        col.add_column_with_expression(
            FieldSchema::new(name, dt),
            Some(expr),
            AddColumnOptions::default(),
        )
    };
    let rejected = [
        ("half", DataType::Int64, "value + 0.5"),
        ("scaled", DataType::Int64, "value * (ratio - 0.5)"),
        (
            "wrapped",
            DataType::Int64,
            "value * value * 4194304 / (value * 4194304)",
        ),
        ("squared", DataType::Float64, "big * big * 2"),
        (
            "negated",
            DataType::Float64,
            "-(-170141183460469231731687303715884105727 - 1)",
        ),
        (
            "quotient",
            DataType::Float64,
            "(-170141183460469231731687303715884105727 - 1) / -1",
        ),
        ("narrow", DataType::Uint32, "ratio * 3"),
        ("too_big", DataType::Int64, "big"),
    ];
    for (name, dt, expr) in rejected {
        let result = add(name, dt, expr);
        assert!(result.is_err(), "{expr} into {dt:?} returned {result:?}");
        assert!(!col.schema_info().has_field(name), "{expr} added {name}");
    }

    add(
        "near_max",
        DataType::Int64,
        "value - 9007199254740993 + 9223372036854775807",
    )
    .unwrap();
    add("big_copy", DataType::Uint64, "big / 1").unwrap();
    add("exact_half", DataType::Int64, "(value + 1) / 2").unwrap();
    add("float_whole", DataType::Int64, "ratio * 2").unwrap();

    let altered = col.alter_column(
        "ratio",
        None,
        Some(FieldSchema::new("ratio", DataType::Int64)),
        AlterColumnOptions::default(),
    );
    assert!(altered.is_err(), "1.5 cast to int64 returned {altered:?}");

    drop(col);
    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    let values = [
        fetch_field(&col, "row", "near_max"),
        fetch_field(&col, "row", "big_copy"),
        fetch_field(&col, "row", "exact_half"),
        fetch_field(&col, "row", "float_whole"),
        fetch_field(&col, "row", "ratio"),
    ];
    drop(col);
    std::fs::remove_dir_all(path).unwrap();
    assert_eq!(
        values,
        [
            Some(Value::I64(i64::MAX)),
            Some(Value::U64(u64::MAX)),
            Some(Value::I64(TWO_53 / 2 + 1)),
            Some(Value::I64(3)),
            Some(Value::F64(1.5)),
        ]
    );
}

fn filter_pks(col: &Collection, filter: &str) -> Vec<String> {
    let query = VectorQuery::new("", Vec::<f32>::new(), 1000).with_filter(filter);
    let mut pks: Vec<String> = col
        .query(query)
        .unwrap_or_else(|e| panic!("{filter}: {e:?}"))
        .iter()
        .map(|d| d.pk.clone())
        .filter(|pk| !pk.starts_with("pad"))
        .collect();
    pks.sort();
    pks
}

#[test]
fn integer_filters_match_exactly_with_and_without_inverted_index() {
    // Literals beyond 2^53 compare exactly on Int64 and UInt64 columns, indexed or not.
    let rows: [(&str, i64, u64); 5] = [
        ("a", TWO_53, u64::MAX - 2),
        ("b", TWO_53 + 1, u64::MAX - 1),
        ("c", TWO_53 + 2, u64::MAX),
        ("d", i64::MAX - 1, TWO_53 as u64),
        ("e", i64::MAX, TWO_53 as u64 + 1),
    ];
    let cases: [(&str, &[&str]); 12] = [
        ("value = 9007199254740993", &["b"]),
        ("value != 9007199254740993", &["a", "c", "d", "e"]),
        ("value > 9007199254740992", &["b", "c", "d", "e"]),
        ("value >= 9007199254740993", &["b", "c", "d", "e"]),
        ("value < 9007199254740994", &["a", "b"]),
        ("value <= 9007199254740992", &["a"]),
        (
            "value IN (9007199254740993, 9223372036854775806)",
            &["b", "d"],
        ),
        (
            "value >= 9007199254740993 AND value <= 9007199254740993",
            &["b"],
        ),
        ("value = 9223372036854775807", &["e"]),
        ("big = 18446744073709551614", &["b"]),
        ("big > 18446744073709551613", &["b", "c"]),
        (
            "big IN (9007199254740993, 18446744073709551615)",
            &["c", "e"],
        ),
    ];
    for indexed in [false, true] {
        let field = |name: &str, dt: DataType| {
            let f = FieldSchema::new(name, dt);
            if indexed {
                f.with_index(IndexParams::Invert(InvertIndexParams::default()))
            } else {
                f
            }
        };
        let schema = CollectionSchema::new("test")
            .with_field(field("value", DataType::Int64))
            .with_field(field("big", DataType::Uint64));
        let (path, col) = scalar_collection("int_exact_filter", schema);
        // Small padding rows keep range hits under the ratio at which the index is skipped.
        let pad = (0..95).map(|i| (format!("pad{i}"), -(i as i64), i as u64));
        let docs = rows
            .iter()
            .map(|(pk, v, b)| (pk.to_string(), *v, *b))
            .chain(pad)
            .map(|(pk, v, b)| Doc::new(pk).set("value", v).set("big", b))
            .collect();
        assert!(col.insert(docs).unwrap().iter().all(Status::is_ok));
        for flushed in [false, true] {
            if flushed {
                col.flush().unwrap();
            }
            for (filter, expected) in cases {
                assert_eq!(
                    filter_pks(&col, filter),
                    expected.to_vec(),
                    "{filter} (indexed: {indexed}, flushed: {flushed})"
                );
            }
        }
        drop(col);
        std::fs::remove_dir_all(path).unwrap();
    }
}

#[test]
fn group_by_and_order_by_keep_large_integer_keys_distinct() {
    // Group keys and sort keys must not merge integers that share an f64 representation.
    let schema = basic_schema(2)
        .with_field(FieldSchema::new("value", DataType::Int64))
        .with_field(FieldSchema::new("big", DataType::Uint64));
    let (path, col) = scalar_collection("int_exact_group", schema);
    let rows: [(&str, i64, u64); 4] = [
        ("a", TWO_53, u64::MAX),
        ("b", TWO_53 + 1, u64::MAX - 1),
        ("c", i64::MAX, u64::MAX),
        ("d", i64::MAX - 1, u64::MAX - 1),
    ];
    let docs = rows
        .iter()
        .enumerate()
        .map(|(i, (pk, v, b))| {
            make_doc(pk, vec![1.0, i as f32], "x")
                .set("value", *v)
                .set("big", *b)
        })
        .collect();
    assert!(col.insert(docs).unwrap().iter().all(Status::is_ok));

    let groups = |field: &str| {
        let query = GroupByVectorQuery {
            base: VectorQuery::new("emb", vec![1.0, 0.0], 10),
            group_by_field: field.to_string(),
            group_count: 10,
            group_topk: 10,
        };
        let mut out: Vec<(String, Vec<String>)> = col
            .group_by_query(query)
            .unwrap()
            .into_iter()
            .map(|g| {
                let mut pks: Vec<String> = g.docs.iter().map(|d| d.pk.clone()).collect();
                pks.sort();
                (format!("{:?}", g.group_value), pks)
            })
            .collect();
        out.sort();
        out
    };
    let value_groups = groups("value");
    let big_groups = groups("big");
    let ordered: Vec<String> = col
        .query_sql("SELECT value FROM test ORDER BY value DESC LIMIT 10")
        .unwrap()
        .iter()
        .map(|d| d.pk.clone())
        .collect();
    drop(col);
    std::fs::remove_dir_all(path).unwrap();

    let key = |v: Value| format!("{v:?}");
    let pks = |p: &[&str]| p.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let mut expected_value = vec![
        (key(Value::I64(TWO_53)), pks(&["a"])),
        (key(Value::I64(TWO_53 + 1)), pks(&["b"])),
        (key(Value::I64(i64::MAX)), pks(&["c"])),
        (key(Value::I64(i64::MAX - 1)), pks(&["d"])),
    ];
    expected_value.sort();
    let mut expected_big = vec![
        (key(Value::U64(u64::MAX)), pks(&["a", "c"])),
        (key(Value::U64(u64::MAX - 1)), pks(&["b", "d"])),
    ];
    expected_big.sort();
    assert_eq!(value_groups, expected_value);
    assert_eq!(big_groups, expected_big);
    assert_eq!(ordered, pks(&["c", "d", "b", "a"]));
}

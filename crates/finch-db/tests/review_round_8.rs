mod common;
use common::*;

#[test]
fn sql_order_by_limit_discards_the_best_matching_row_before_sorting() {
    // A scalar ORDER BY must sort all matching rows before applying LIMIT.
    let path = temp_dir("round_8_order_by_limit");
    let schema =
        CollectionSchema::new("test").with_field(FieldSchema::new("value", DataType::Int64));
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let statuses = col
        .insert(vec![
            Doc::new("first").set("value", 10i64),
            Doc::new("best").set("value", 20i64),
        ])
        .unwrap();
    assert!(statuses.iter().all(|status| status.ok()));
    col.flush().unwrap();
    drop(col);
    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    let all = col
        .query_sql("SELECT value FROM test WHERE value > 0 ORDER BY value DESC LIMIT 2")
        .unwrap();
    let limited = col
        .query_sql("SELECT value FROM test WHERE value > 0 ORDER BY value DESC LIMIT 1")
        .unwrap();
    drop(col);
    std::fs::remove_dir_all(path).unwrap();
    assert_eq!(all[0].get("value"), Some(&Value::I64(20)));
    assert_eq!(limited.len(), 1);
    assert_eq!(limited[0].get("value"), all[0].get("value"));
}

#[test]
fn concurrent_column_rename_makes_an_acknowledged_value_disappear_from_fetch() {
    // A column rename must publish its schema and forward data as one readable version.
    let path = temp_dir("round_8_rename_fetch");
    let schema =
        CollectionSchema::new("test").with_field(FieldSchema::new("left", DataType::Int64));
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let statuses = col
        .insert(vec![Doc::new("row").set("left", 42i64)])
        .unwrap();
    assert!(statuses.iter().all(|status| status.ok()));
    col.flush().unwrap();
    let barrier = std::sync::Barrier::new(2);
    let done = std::sync::atomic::AtomicBool::new(false);
    let missing = std::thread::scope(|threads| {
        let writer = threads.spawn(|| {
            barrier.wait();
            for attempt in 0..128 {
                let (from, to) = if attempt % 2 == 0 {
                    ("left", "right")
                } else {
                    ("right", "left")
                };
                col.alter_column(from, Some(to), None, AlterColumnOptions::default())
                    .unwrap();
            }
            done.store(true, Ordering::Release);
        });
        barrier.wait();
        let mut missing = None;
        for attempt in 0..100_000 {
            let docs = col.fetch(vec!["row".to_string()]).unwrap();
            let values = docs
                .get("row")
                .map(|doc| (doc.get("left").cloned(), doc.get("right").cloned()));
            if values != Some((Some(Value::I64(42)), None))
                && values != Some((None, Some(Value::I64(42))))
            {
                missing = Some((attempt, values));
                break;
            }
            if done.load(Ordering::Acquire) {
                break;
            }
        }
        writer.join().unwrap();
        missing
    });
    drop(col);
    std::fs::remove_dir_all(path).unwrap();
    assert!(
        missing.is_none(),
        "fetch observed a partial rename: {missing:?}"
    );
}

mod common;
use common::*;

use std::sync::atomic::AtomicBool;
use std::sync::{Barrier, Mutex};

const ROWS: i64 = 8;
const RENAMES: usize = 64;

// A query path under test: `Some(ok)` when it returned rows, `None` when the column was renamed away.
type Reader<'a> = &'a (dyn Fn(&Collection) -> Option<bool> + Sync);

fn renamed_collection(name: &str, rows: i64) -> (std::path::PathBuf, Arc<Collection>) {
    let path = temp_dir(name);
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(2))
        .with_field(FieldSchema::new("left", DataType::Int64));
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let docs = (1..=rows)
        .map(|i| {
            Doc::new(i.to_string())
                .set("emb", vec![1.0f32, i as f32])
                .set("left", i * 10)
        })
        .collect();
    assert!(col.insert(docs).unwrap().iter().all(|s| s.ok()));
    col.flush().unwrap();
    (path, col)
}

fn rename_back_and_forth(col: &Collection, barrier: &Barrier, done: &AtomicBool) {
    barrier.wait();
    for attempt in 0..RENAMES {
        let (from, to) = if attempt % 2 == 0 {
            ("left", "right")
        } else {
            ("right", "left")
        };
        col.alter_column(from, Some(to), None, AlterColumnOptions::default())
            .unwrap();
    }
    done.store(true, Ordering::Release);
}

// The row carries its value under exactly one of the two names.
fn has_one_name(doc: &Doc) -> bool {
    let Ok(pk) = doc.pk.parse::<i64>() else {
        return false;
    };
    let value = Some(&Value::I64(pk * 10));
    matches!(
        (doc.get("left"), doc.get("right")),
        (v, None) | (None, v) if v == value
    )
}

fn all_rows(docs: &[Arc<Doc>]) -> bool {
    docs.len() == ROWS as usize && docs.iter().all(|doc| has_one_name(doc))
}

#[test]
fn queries_during_a_column_rename_see_one_schema() {
    let (path, col) = renamed_collection("schema_publication_queries", ROWS);
    let readers: [(&str, Reader); 5] = [
        ("vector query", &|col| {
            let query = VectorQuery::new("emb", vec![1.0, 0.0], 100).with_filter("left > 0");
            col.query(query).ok().map(|docs| all_rows(&docs))
        }),
        ("filter-only scan", &|col| {
            let query = VectorQuery::new("emb", Vec::new(), 100).with_filter("left > 0");
            col.scan_filter_only(query).ok().map(|docs| all_rows(&docs))
        }),
        ("int-id query", &|col| {
            let query = VectorQuery::new("emb", vec![1.0, 0.0], 100).with_filter("left > 0");
            col.query_int_ids(query)
                .ok()
                .map(|ids| ids.len() == ROWS as usize)
        }),
        ("group-by", &|col| {
            let query = GroupByVectorQuery {
                base: VectorQuery::new("emb", vec![1.0, 0.0], 100),
                group_by_field: "left".to_string(),
                group_count: 1,
                group_topk: ROWS as usize,
            };
            col.group_by_query(query)
                .ok()
                .map(|groups| groups.len() == ROWS as usize)
        }),
        ("sql order by", &|col| {
            let docs = col.query_sql("SELECT * FROM test ORDER BY left DESC LIMIT 1");
            docs.ok()
                .map(|docs| docs.len() == 1 && docs[0].pk == ROWS.to_string())
        }),
    ];

    let barrier = Barrier::new(readers.len() + 1);
    let done = AtomicBool::new(false);
    let failures = Mutex::new(Vec::new());
    std::thread::scope(|threads| {
        threads.spawn(|| rename_back_and_forth(&col, &barrier, &done));
        for (name, read) in readers {
            let (col, barrier, done, failures) = (&col, &barrier, &done, &failures);
            threads.spawn(move || {
                barrier.wait();
                while !done.load(Ordering::Acquire) {
                    // An error means the query ran after its column name was renamed away.
                    if read(col) == Some(false) {
                        failures.lock().unwrap().push(name);
                        return;
                    }
                }
            });
        }
    });
    drop(col);
    std::fs::remove_dir_all(path).unwrap();
    let failures = failures.into_inner().unwrap();
    assert!(
        failures.is_empty(),
        "these paths mixed two schemas: {failures:?}"
    );
}

#[test]
fn delete_by_filter_during_a_column_rename_deletes_its_match() {
    let (path, col) = renamed_collection("schema_publication_delete", 1000);
    let barrier = Barrier::new(2);
    let done = AtomicBool::new(false);
    let kept = std::thread::scope(|threads| {
        threads.spawn(|| rename_back_and_forth(&col, &barrier, &done));
        barrier.wait();
        let mut pk = 1;
        while pk <= 1000 && !done.load(Ordering::Acquire) {
            // Only the name the column has when the filter is checked succeeds.
            let deleted = ["left", "right"].iter().any(|name| {
                col.delete_by_filter(&format!("{name} = {}", pk * 10))
                    .is_ok()
            });
            if !deleted {
                continue;
            }
            if col
                .fetch(vec![pk.to_string()])
                .unwrap()
                .contains_key(&pk.to_string())
            {
                return Some(pk);
            }
            pk += 1;
        }
        None
    });
    drop(col);
    std::fs::remove_dir_all(path).unwrap();
    assert!(
        kept.is_none(),
        "delete_by_filter returned Ok but kept row {kept:?}"
    );
}

#[test]
fn fetch_during_column_renames_always_sees_the_value() {
    // A column rename must publish its schema and forward data as one readable version.
    let path = temp_dir("rename_during_fetch");
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

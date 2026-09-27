mod common;
use common::*;

fn tags_schema() -> CollectionSchema {
    CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(2))
        .with_field(
            FieldSchema::new("tags", DataType::ArrayUint32)
                .with_index(IndexParams::Invert(InvertIndexParams::default())),
        )
}

fn tags_doc(pk: &str, tags: Value) -> Doc {
    Doc::new(pk).set("emb", vec![1.0f32, 0.0]).set("tags", tags)
}

// Validation accepts non-negative i32 items for an ARRAY_UINT32 field, but its inverted index
// cannot encode them.
fn rejected_tags() -> Value {
    Value::ArrayI32(vec![7])
}

fn tags_of(col: &Collection, pk: &str) -> Option<Value> {
    col.fetch(vec![pk.to_string()])
        .unwrap()
        .get(pk)
        .map(|doc| doc.fields["tags"].clone())
}

fn assert_contents(col: &Collection, expected: &[(&str, u32)]) {
    assert_eq!(col.stats().unwrap().doc_count, expected.len() as u64);
    for (pk, tag) in expected {
        assert_eq!(
            tags_of(col, pk),
            Some(Value::ArrayU32(vec![*tag])),
            "pk {pk}"
        );
    }
}

#[test]
fn test_rejected_document_leaves_no_trace() {
    let path = temp_dir("write_atomicity");
    let col =
        Collection::create_and_open(&path, tags_schema(), CollectionOptions::default()).unwrap();

    let results = col
        .insert(vec![
            tags_doc("a", Value::ArrayU32(vec![1])),
            tags_doc("b", rejected_tags()),
            tags_doc("c", Value::ArrayU32(vec![3])),
        ])
        .unwrap();
    assert!(results[0].is_ok());
    assert!(!results[1].is_ok());
    assert!(results[2].is_ok());
    assert_eq!(tags_of(&col, "b"), None);

    let retry = col.insert(vec![tags_doc("b", Value::ArrayU32(vec![2]))]);
    assert!(retry.unwrap()[0].is_ok());

    // A rejected replacement keeps the previous version.
    assert!(!col.upsert(vec![tags_doc("a", rejected_tags())]).unwrap()[0].is_ok());
    assert!(!col
        .update(vec![Doc::new("c").set("tags", rejected_tags())])
        .unwrap()[0]
        .is_ok());
    let expected = [("a", 1), ("b", 2), ("c", 3)];
    assert_contents(&col, &expected);

    // Reopen replays the WAL, which must not hold the rejected documents.
    drop(col);
    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    assert_contents(&col, &expected);

    col.flush().unwrap();
    drop(col);
    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    assert_contents(&col, &expected);

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_fetch_racing_writes_never_sees_a_key_without_its_document() {
    let path = temp_dir("fetch_write_race");
    // The label index makes each segment insert slow enough for readers to land inside it.
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(
            FieldSchema::new("label", DataType::String)
                .with_index(IndexParams::Invert(InvertIndexParams::default())),
        );
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    col.insert(vec![make_doc("k", vec![1.0, 0.0, 0.0, 0.0], "v0")])
        .unwrap();
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let readers: Vec<_> = (0..8)
        .map(|reader| {
            let col = col.clone();
            let stop = stop.clone();
            std::thread::spawn(move || {
                let mut misses = 0usize;
                let mut i = 0usize;
                while !stop.load(Ordering::Relaxed) {
                    let fresh = format!("n{}", (i + reader) % 2000);
                    i += 1;
                    let fetched = col.fetch(vec!["k".to_string(), fresh.clone()]).unwrap();
                    if !fetched.contains_key("k") {
                        misses += 1;
                    }
                    for doc in fetched.values() {
                        assert_eq!(doc.fields.len(), 2, "incomplete doc for pk {}", doc.pk);
                    }
                }
                misses
            })
        })
        .collect();
    for i in 0..2000 {
        let label = format!("v{i}");
        let upsert = col.upsert(vec![make_doc("k", vec![1.0, 0.0, 0.0, 0.0], &label)]);
        assert!(upsert.unwrap()[0].is_ok());
        let insert = col.insert(vec![make_doc(
            &format!("n{i}"),
            vec![0.0, 1.0, 0.0, 0.0],
            &label,
        )]);
        assert!(insert.unwrap()[0].is_ok());
    }
    stop.store(true, Ordering::Relaxed);
    let misses: usize = readers.into_iter().map(|r| r.join().unwrap()).sum();
    drop(col);
    std::fs::remove_dir_all(&path).ok();
    assert_eq!(misses, 0, "fetch missed a key that always exists");
}

mod common;
use common::*;

fn invert(range: bool) -> IndexParams {
    IndexParams::Invert(InvertIndexParams {
        enable_range_optimization: range,
        enable_extended_wildcard: false,
    })
}

fn pks(docs: &[Arc<Doc>]) -> Vec<String> {
    let mut pks: Vec<String> = docs.iter().map(|d| d.pk.clone()).collect();
    pks.sort();
    pks
}

#[test]
fn nul_in_indexed_string_is_rejected_before_the_wal() {
    let path = temp_dir("nul_indexed_string");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(2))
        .with_field(FieldSchema::new("label", DataType::String).with_index(invert(false)));
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    let doc = Doc::new("bad")
        .set("emb", vec![1.0f32, 0.0])
        .set("label", "a\0b");
    let status = col.insert(vec![doc]).unwrap().remove(0);
    assert!(!status.is_ok(), "NUL string must be rejected");
    assert!(status.message.contains("label"), "{}", status.message);
    drop(col);

    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    assert!(col.fetch(vec!["bad".into()]).unwrap().is_empty());
    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn replaying_a_wal_record_the_index_rejects_fails_open_naming_the_field() {
    let path = temp_dir("nul_indexed_string_replay");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(2))
        .with_field(FieldSchema::new("label", DataType::String).with_index(invert(false)));
    drop(Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap());

    // A legacy JSONL WAL is plain text, so a damaged record can be written by hand.
    std::fs::remove_file(path.join("wal_0.log")).unwrap();
    let doc = Doc::new("bad")
        .set("emb", vec![1.0f32, 0.0])
        .set("label", "a\0b");
    let record = serde_json::json!({"op": "Insert", "doc_id": 0, "pk": "bad", "doc": doc});
    std::fs::write(path.join("wal.log"), format!("{record}\n")).unwrap();

    let err = Collection::open(&path, CollectionOptions::default())
        .err()
        .expect("open must fail on a record the index cannot store");
    assert!(err.message.contains("label"), "{}", err.message);
    std::fs::remove_dir_all(&path).ok();
}

fn binary_range_collection(name: &str) -> (std::path::PathBuf, Arc<Collection>) {
    let path = temp_dir(name);
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(2))
        .with_field(FieldSchema::new("payload", DataType::Binary).with_index(invert(true)));
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let docs = [
        ("short", b"000".as_slice()),
        ("equal", b"aa"),
        ("long", b"z"),
    ]
    .into_iter()
    .map(|(pk, bytes)| {
        Doc::new(pk)
            .set("emb", vec![1.0f32, 0.0])
            .set("payload", Value::Bytes(bytes.to_vec()))
    })
    .collect();
    col.insert(docs).unwrap();
    col.flush().unwrap();
    (path, col)
}

#[test]
fn binary_range_filter_compares_contents() {
    let (path, col) = binary_range_collection("binary_range_query");
    let q = VectorQuery::new("emb", vec![1.0, 0.0], 10).with_filter("payload > 'aa'");
    assert_eq!(pks(&col.query(q).unwrap()), vec!["long"]);
    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn binary_range_delete_by_filter_compares_contents() {
    let (path, col) = binary_range_collection("binary_range_delete");
    col.delete_by_filter("payload > 'aa'").unwrap();
    let q = VectorQuery::new("emb", vec![1.0, 0.0], 10);
    assert_eq!(pks(&col.query(q).unwrap()), vec!["equal", "short"]);
    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn dropped_column_is_not_materialized() {
    let path = temp_dir("dropped_column_hidden");
    let col =
        Collection::create_and_open(&path, basic_schema(2), CollectionOptions::default()).unwrap();
    col.add_column(
        FieldSchema::new("x", DataType::Int32),
        AddColumnOptions::default(),
    )
    .unwrap();
    col.insert(vec![make_doc("a", vec![1.0, 0.0], "l").set("x", 7i32)])
        .unwrap();
    col.drop_column("x").unwrap();

    let fetched = col.fetch(vec!["a".into()]).unwrap();
    assert!(!fetched["a"].fields.contains_key("x"), "{:?}", fetched["a"]);
    let q = VectorQuery::new("emb", vec![1.0, 0.0], 10);
    let docs = col.query(q).unwrap();
    assert!(!docs[0].fields.contains_key("x"), "{:?}", docs[0]);
    assert!(docs[0].fields.contains_key("label"));

    let q = VectorQuery::new("emb", vec![1.0, 0.0], 10).with_output_fields(vec!["x".into()]);
    assert!(col.query(q).is_err());
    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn group_by_widens_candidates_until_groups_are_filled() {
    let path = temp_dir("group_by_widens");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(2))
        .with_field(FieldSchema::new("category", DataType::String));
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let mut docs: Vec<Doc> = (0..4)
        .map(|i| {
            Doc::new(format!("a{i}"))
                .set("emb", vec![1.0f32, i as f32 * 0.01])
                .set("category", "A")
        })
        .collect();
    docs.push(
        Doc::new("b0")
            .set("emb", vec![1.0f32, 0.5])
            .set("category", "B"),
    );
    col.insert(docs).unwrap();

    let results = col
        .group_by_query(GroupByVectorQuery {
            base: VectorQuery::new("emb", vec![1.0, 0.0], 1),
            group_by_field: "category".to_string(),
            group_count: 1,
            group_topk: 2,
        })
        .unwrap();
    let groups: Vec<_> = results
        .iter()
        .map(|g| (g.group_value.clone(), g.docs[0].pk.clone()))
        .collect();
    assert_eq!(
        groups,
        vec![
            (Value::String("A".into()), "a0".to_string()),
            (Value::String("B".into()), "b0".to_string()),
        ]
    );
    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

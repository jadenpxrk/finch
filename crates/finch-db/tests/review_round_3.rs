mod common;
use common::*;

#[test]
fn or_of_inequalities_is_rewritten_as_not_in() {
    // Every non-null value satisfies at least one inequality, so all three rows must match.
    let path = temp_dir("round_3_inequality_or");
    let schema =
        CollectionSchema::new("test").with_field(FieldSchema::new("n", DataType::Int64).nullable());
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let statuses = col
        .insert(vec![
            Doc::new("one").set("n", 1i64),
            Doc::new("two").set("n", 2i64),
            Doc::new("three").set("n", 3i64),
            Doc::new("missing"),
        ])
        .unwrap();
    assert!(statuses.iter().all(|status| status.ok()));
    col.flush().unwrap();
    let filter = "n != 1 OR n != 2";
    let matched = col
        .query(VectorQuery::new("", vec![], 10).with_filter(filter))
        .unwrap();
    assert!(col.delete_by_filter(filter).unwrap().ok());
    drop(col);
    let reopened = Collection::open(&path, CollectionOptions::default()).unwrap();
    let survivors = reopened
        .fetch(
            ["one", "two", "three", "missing"]
                .map(str::to_string)
                .to_vec(),
        )
        .unwrap();
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
    let mut matched = matched
        .iter()
        .map(|doc| doc.pk.as_str())
        .collect::<Vec<_>>();
    matched.sort_unstable();
    assert_eq!(
        (matched, survivors.len()),
        (vec!["one", "three", "two"], 1),
        "OR inequalities must match and delete all non-null rows"
    );
}

mod common;
use common::*;

#[test]
fn indexed_like_skips_rows_whose_prefix_and_suffix_overlap() {
    // LIKE literals must consume distinct characters, or filtered deletion loses nonmatching rows.
    let path = temp_dir("like_prefix_suffix_overlap");
    let schema =
        CollectionSchema::new("test")
            .with_field(FieldSchema::new("plain", DataType::String))
            .with_field(FieldSchema::new("indexed", DataType::String).with_index(
                IndexParams::Invert(InvertIndexParams {
                    enable_extended_wildcard: true,
                    ..Default::default()
                }),
            ));
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let statuses = col
        .insert(vec![
            Doc::new("overlap")
                .set("plain", "aba")
                .set("indexed", "aba"),
            Doc::new("match")
                .set("plain", "abba")
                .set("indexed", "abba"),
        ])
        .unwrap();
    assert!(statuses.iter().all(|status| status.ok()));
    col.flush().unwrap();
    drop(col);
    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    let plain = col
        .query(VectorQuery::new("", vec![], 10).with_filter("plain LIKE 'ab%ba'"))
        .unwrap();
    let indexed = col
        .query(VectorQuery::new("", vec![], 10).with_filter("indexed LIKE 'ab%ba'"))
        .unwrap();
    assert!(col.delete_by_filter("indexed LIKE 'ab%ba'").unwrap().ok());
    drop(col);
    let reopened = Collection::open(&path, CollectionOptions::default()).unwrap();
    let survivors = reopened
        .fetch(vec!["overlap".to_string(), "match".to_string()])
        .unwrap();
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
    assert_eq!(plain.len(), 1);
    assert_eq!(plain[0].pk, "match");
    assert_eq!(
        (
            indexed.len(),
            survivors.contains_key("overlap"),
            survivors.contains_key("match")
        ),
        (1, true, false),
        "indexed LIKE matched and deleted the nonmatching overlap row"
    );
}

#[test]
fn or_of_inequalities_matches_every_non_null_row() {
    // Every non-null value satisfies at least one inequality, so all three rows must match.
    let path = temp_dir("inequality_or");
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

#[test]
fn like_escape_clause_deletes_only_matching_rows() {
    // An accepted ESCAPE clause must not silently change which rows a delete targets.
    let path = temp_dir("like_escape");
    let schema =
        CollectionSchema::new("test").with_field(FieldSchema::new("label", DataType::String));
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let statuses = col
        .insert(vec![
            Doc::new("target").set("label", "a_b"),
            Doc::new("bystander").set("label", "a!xb"),
        ])
        .unwrap();
    assert!(statuses.iter().all(|status| status.ok()));
    col.flush().unwrap();
    let filter = "label LIKE 'a!_b' ESCAPE '!'";
    let matched = col.query(VectorQuery::new("", vec![], 10).with_filter(filter));
    let deleted = col.delete_by_filter(filter);
    drop(col);
    let reopened = Collection::open(&path, CollectionOptions::default()).unwrap();
    let survivors = reopened
        .fetch(vec!["target".to_string(), "bystander".to_string()])
        .unwrap();
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
    assert!(
        survivors.contains_key("bystander"),
        "ESCAPE was ignored: query matched {:?}; survivors: {:?}",
        matched
            .as_ref()
            .map(|docs| docs.iter().map(|doc| &doc.pk).collect::<Vec<_>>()),
        survivors.keys().collect::<std::collections::BTreeSet<_>>()
    );
    if let Ok(status) = deleted {
        assert!(status.ok());
        assert!(!survivors.contains_key("target"));
    } else {
        assert!(survivors.contains_key("target"));
    }
    if let Ok(matched) = matched {
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].pk, "target");
    }
}

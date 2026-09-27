mod common;
use common::*;

#[test]
fn like_escape_clause_is_ignored_and_deletes_a_nonmatching_row() {
    // An accepted ESCAPE clause must not silently change which rows a delete targets.
    let path = temp_dir("round_10_like_escape");
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

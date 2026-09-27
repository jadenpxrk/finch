mod common;
use common::*;

#[test]
fn repeated_compaction_loses_documents_when_segment_ids_reorder_rows() {
    // Compaction must preserve doc-id order even when newer writes use a reserved lower segment id.
    let path = temp_dir("round_4_compaction_order");
    let schema = CollectionSchema::new("test").with_field(FieldSchema::new("n", DataType::Int64));
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    for (pk, n) in [("a", 0i64), ("b", 1)] {
        assert!(col.insert(vec![Doc::new(pk).set("n", n)]).unwrap()[0].ok());
        col.flush().unwrap();
    }
    col.optimize(OptimizeOptions::default()).unwrap();
    for (pk, n) in [("c", 2i64), ("d", 3)] {
        assert!(col.insert(vec![Doc::new(pk).set("n", n)]).unwrap()[0].ok());
        col.flush().unwrap();
    }
    let pks = ["a", "b", "c", "d"].map(str::to_string).to_vec();
    let before = col.fetch(pks.clone()).unwrap();
    col.optimize(OptimizeOptions::default()).unwrap();
    drop(col);
    let reopened = Collection::open(&path, CollectionOptions::default()).unwrap();
    let after = reopened.fetch(pks).unwrap();
    drop(reopened);
    std::fs::remove_dir_all(path).unwrap();
    let mut before_keys = before.keys().cloned().collect::<Vec<_>>();
    let mut after_keys = after.keys().cloned().collect::<Vec<_>>();
    before_keys.sort();
    after_keys.sort();
    assert_eq!(before_keys, vec!["a", "b", "c", "d"]);
    assert_eq!(after_keys, before_keys);
}

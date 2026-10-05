use super::*;

#[test]
fn test_doc_validation_rejects_bad_pk_and_missing_required_fields() {
    let path = temp_dir("doc_validate");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("label", DataType::String).not_null());
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    // Bad PK (illegal char '/')
    let statuses = col
        .insert(vec![Doc::new("bad/pk")
            .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
            .set("label", "x")])
        .unwrap();
    assert!(statuses[0].is_err());

    // Missing required (not-null) field
    let statuses = col
        .insert(vec![Doc::new("ok").set("emb", vec![1.0f32, 0.0, 0.0, 0.0])])
        .unwrap();
    assert!(statuses[0].is_err());

    // Unknown field
    let statuses = col
        .insert(vec![Doc::new("ok2")
            .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
            .set("label", "x")
            .set("unknown", 1i64)])
        .unwrap();
    assert!(statuses[0].is_err());

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_insert_duplicate_pk() {
    let path = temp_dir("dup_pk");
    let schema = basic_schema(4);
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![make_doc("d1", vec![1.0, 0.0, 0.0, 0.0], "a")])
        .unwrap();
    let results = col
        .insert(vec![make_doc("d1", vec![0.0, 1.0, 0.0, 0.0], "b")])
        .unwrap();
    assert!(results[0].is_err(), "duplicate pk should fail");

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_write_batch_size_limit_is_enforced() {
    let path = temp_dir("write_batch_limit");
    let schema = basic_schema(4);
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    let docs: Vec<Doc> = (0..1025)
        .map(|i| {
            Doc::new(format!("d{}", i))
                .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
                .set("label", "x")
        })
        .collect();
    assert!(col.insert(docs).is_err());

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

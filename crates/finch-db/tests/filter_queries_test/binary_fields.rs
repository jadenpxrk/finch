use super::*;

#[test]
fn test_query_binary_and_array_binary_filters_accept_string_literals() {
    let path = temp_dir("filter_binary_literals");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("blob", DataType::Binary))
        .with_field(FieldSchema::new("blobs", DataType::ArrayBinary));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![
        Doc::new("a")
            .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
            .set("blob", Value::Bytes(b"abc".to_vec()))
            .set(
                "blobs",
                Value::ArrayBinary(vec![b"red".to_vec(), b"blue".to_vec()]),
            ),
        Doc::new("b")
            .set("emb", vec![0.9f32, 0.1, 0.0, 0.0])
            .set("blob", Value::Bytes(b"xyz".to_vec()))
            .set("blobs", Value::ArrayBinary(vec![b"green".to_vec()])),
    ])
    .unwrap();

    // BINARY/ARRAY_BINARY use string literals in filters (treated as raw bytes).
    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("blob = 'abc'");
    let r = col.query(q).unwrap();
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].pk, "a");

    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10)
        .with_filter("blobs contain_any ('blue')");
    let r = col.query(q).unwrap();
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].pk, "a");

    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10)
        .with_filter("blobs contain_all ('red', 'blue')");
    let r = col.query(q).unwrap();
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].pk, "a");

    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10)
        .with_filter("blobs contain_any ('missing')");
    let r = col.query(q).unwrap();
    assert!(r.is_empty());

    // `IN (...)` list-values are only supported for string/numeric/bool,
    // not for BINARY.
    let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10).with_filter("blob IN ('abc')");
    assert!(col.query(q).is_err());

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_binary_fields_accept_string_values_on_write_and_round_trip() {
    let path = temp_dir("binary_write_round_trip");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("blob", DataType::Binary))
        .with_field(FieldSchema::new("blobs", DataType::ArrayBinary));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    // BINARY/ARRAY_BINARY setters accept string payloads.
    col.insert(vec![Doc::new("a")
        .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
        .set("blob", "abc")
        .set(
            "blobs",
            Value::ArrayString(vec!["red".into(), "blue".into()]),
        )])
    .unwrap();

    fn assert_round_trip(col: &Collection) {
        let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10)
            .with_output_fields(vec!["blob".into(), "blobs".into()]);
        let r = col.query(q).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].pk, "a");
        match r[0].fields.get("blob") {
            Some(Value::Bytes(b)) => assert_eq!(b.as_slice(), b"abc"),
            other => panic!("expected blob bytes, got: {:?}", other),
        }
        match r[0].fields.get("blobs") {
            Some(Value::ArrayBinary(items)) => {
                let as_slices: Vec<&[u8]> = items.iter().map(|v| v.as_slice()).collect();
                assert_eq!(as_slices, vec![b"red".as_slice(), b"blue".as_slice()]);
            }
            other => panic!("expected blobs array_binary, got: {:?}", other),
        }
    }

    assert_round_trip(&col);
    col.flush().unwrap();
    assert_round_trip(&col);

    drop(col);
    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    assert_round_trip(&col);

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

use super::*;

#[test]
fn test_insert_vector_fp16_and_int8_are_accepted_and_searchable() {
    let path = temp_dir("insert_vec_fp16_int8");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb16", DataType::VectorFp16).with_dimension(4))
        .with_field(FieldSchema::new("emb8", DataType::VectorInt8).with_dimension(4));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    col.insert(vec![
        Doc::new("a")
            .set(
                "emb16",
                vec![
                    f16::from_f32(1.0),
                    f16::from_f32(0.0),
                    f16::from_f32(0.0),
                    f16::from_f32(0.0),
                ],
            )
            .set("emb8", vec![1i8, 0, 0, 0]),
        Doc::new("b")
            .set(
                "emb16",
                vec![
                    f16::from_f32(0.0),
                    f16::from_f32(1.0),
                    f16::from_f32(0.0),
                    f16::from_f32(0.0),
                ],
            )
            .set("emb8", vec![0i8, 1, 0, 0]),
    ])
    .unwrap();

    // Query against both fields (writing segment).
    let q16 = VectorQuery::new("emb16", vec![1.0f32, 0.0, 0.0, 0.0], 2);
    let r16 = col.query(q16).unwrap();
    assert_eq!(r16[0].pk, "a");

    let q8 = VectorQuery::new("emb8", vec![1.0f32, 0.0, 0.0, 0.0], 2);
    let r8 = col.query(q8).unwrap();
    assert_eq!(r8[0].pk, "a");

    // Persist and query again.
    col.flush().unwrap();
    let q16 = VectorQuery::new("emb16", vec![1.0f32, 0.0, 0.0, 0.0], 2);
    let r16 = col.query(q16).unwrap();
    assert_eq!(r16[0].pk, "a");

    let q8 = VectorQuery::new("emb8", vec![1.0f32, 0.0, 0.0, 0.0], 2);
    let r8 = col.query(q8).unwrap();
    assert_eq!(r8[0].pk, "a");

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_fp16_fields_round_f32_payloads_on_write() {
    let path = temp_dir("fp16_rounding_on_write");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb16", DataType::VectorFp16).with_dimension(4))
        .with_field(FieldSchema::new("sparse16", DataType::SparseFp16).with_dimension(8));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    let emb_in = vec![0.1f32, 0.2, 0.3, 0.4];
    let emb_expected: Vec<f32> = emb_in.iter().map(|&x| f16::from_f32(x).to_f32()).collect();

    let sparse_indices = vec![1u32, 3u32, 7u32];
    let sparse_values_in = vec![0.3333f32, 0.12345, -0.9876];
    let sparse_values_expected: Vec<f32> = sparse_values_in
        .iter()
        .map(|&x| f16::from_f32(x).to_f32())
        .collect();

    col.insert(vec![Doc::new("d1").set("emb16", emb_in.clone()).set(
        "sparse16",
        Value::SparseF32 {
            indices: sparse_indices.clone(),
            values: sparse_values_in.clone(),
        },
    )])
    .unwrap();

    // Validate in-memory write normalization via fetch().
    let fetched = col.fetch(vec!["d1".to_string()]).unwrap();
    let d1 = fetched.get("d1").unwrap();
    match d1.fields.get("emb16") {
        Some(Value::VecF32(v)) => assert_eq!(v, &emb_expected),
        other => panic!("expected emb16 VecF32, got {other:?}"),
    }
    match d1.fields.get("sparse16") {
        Some(Value::SparseF32 { indices, values }) => {
            assert_eq!(indices, &sparse_indices);
            assert_eq!(values, &sparse_values_expected);
        }
        other => panic!("expected sparse16 SparseF32, got {other:?}"),
    }

    // Validate persisted round-trip.
    col.flush().unwrap();
    drop(col);

    let reopened = Collection::open(&path, CollectionOptions::default()).unwrap();
    let fetched = reopened.fetch(vec!["d1".to_string()]).unwrap();
    let d1 = fetched.get("d1").unwrap();
    match d1.fields.get("emb16") {
        Some(Value::VecF32(v)) => assert_eq!(v, &emb_expected),
        other => panic!("expected emb16 VecF32, got {other:?}"),
    }
    match d1.fields.get("sparse16") {
        Some(Value::SparseF32 { indices, values }) => {
            assert_eq!(indices, &sparse_indices);
            assert_eq!(values, &sparse_values_expected);
        }
        other => panic!("expected sparse16 SparseF32, got {other:?}"),
    }

    drop(reopened);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_integer_vector_fields_normalize_f32_payloads_on_write() {
    let path = temp_dir("int_vec_normalize_on_write");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("i8v", DataType::VectorInt8).with_dimension(4));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    let statuses = col
        .insert(vec![
            Doc::new("d1").set("i8v", vec![1.9f32, -1.9, 9999.0, f32::NAN])
        ])
        .unwrap();
    assert!(statuses.iter().all(|s| s.is_ok()), "statuses={statuses:?}");

    fn assert_doc(doc: &Doc) {
        match doc.fields.get("i8v") {
            Some(Value::VecF32(v)) => assert_eq!(v, &vec![1.0, -1.0, i8::MAX as f32, 0.0]),
            other => panic!("expected i8v VecF32, got {other:?}"),
        }
    }

    let fetched = col.fetch(vec!["d1".to_string()]).unwrap();
    assert_doc(fetched.get("d1").unwrap());

    col.flush().unwrap();
    drop(col);

    let reopened = Collection::open(&path, CollectionOptions::default()).unwrap();
    let fetched = reopened.fetch(vec!["d1".to_string()]).unwrap();
    assert_doc(fetched.get("d1").unwrap());

    drop(reopened);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_schema_rejects_unsupported_dense_vector_dtypes() {
    let path = temp_dir("insert_vec_fp64_int16_int4");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb64", DataType::VectorFp64).with_dimension(4))
        .with_field(FieldSchema::new("emb16", DataType::VectorInt16).with_dimension(4))
        .with_field(FieldSchema::new("emb4", DataType::VectorInt4).with_dimension(4));

    let err = match Collection::create_and_open(&path, schema, CollectionOptions::default()) {
        Ok(_) => panic!("expected schema validation error"),
        Err(err) => err,
    };
    assert!(
        err.message.contains("dense_vector's data type"),
        "err={err}"
    );
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_schema_rejects_binary_dense_vectors() {
    let path = temp_dir("insert_vec_binary");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("b32", DataType::VectorBinary32).with_dimension(2))
        .with_field(FieldSchema::new("b64", DataType::VectorBinary64).with_dimension(1));

    let err = match Collection::create_and_open(&path, schema, CollectionOptions::default()) {
        Ok(_) => panic!("expected schema validation error"),
        Err(err) => err,
    };
    assert!(
        err.message.contains("dense_vector's data type"),
        "err={err}"
    );
    std::fs::remove_dir_all(&path).ok();
}

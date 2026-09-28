use super::*;

#[test]
fn test_filter_only_query_recall_base_scalar_semantics_match_reference() {
    let path = temp_dir("filter_only_recall_base");
    let schema = CollectionSchema::new("test")
        .with_field(
            FieldSchema::new("dummy_vec", DataType::VectorFp32)
                .nullable()
                .with_dimension(1),
        )
        .with_field(
            FieldSchema::new("id", DataType::Int64)
                .nullable()
                .not_null(),
        )
        .with_field(
            FieldSchema::new("invert_id", DataType::Int64)
                .nullable()
                .not_null()
                .with_index(IndexParams::Invert(InvertIndexParams::default())),
        )
        .with_field(
            FieldSchema::new("bool", DataType::Bool)
                .nullable()
                .not_null(),
        )
        .with_field(
            FieldSchema::new("invert_bool", DataType::Bool)
                .nullable()
                .not_null()
                .with_index(IndexParams::Invert(InvertIndexParams::default())),
        )
        .with_field(
            FieldSchema::new("bool_array", DataType::ArrayBool)
                .nullable()
                .not_null(),
        )
        .with_field(
            FieldSchema::new("invert_bool_array", DataType::ArrayBool)
                .nullable()
                .not_null()
                .with_index(IndexParams::Invert(InvertIndexParams::default())),
        )
        .with_field(
            FieldSchema::new("name", DataType::String)
                .nullable()
                .not_null(),
        )
        .with_field(
            FieldSchema::new("invert_name", DataType::String)
                .nullable()
                .not_null()
                .with_index(IndexParams::Invert(InvertIndexParams::default())),
        )
        .with_field(FieldSchema::new("optional_age", DataType::Uint32).nullable())
        .with_field(
            FieldSchema::new("invert_optional_age", DataType::Uint32)
                .nullable()
                .with_index(IndexParams::Invert(InvertIndexParams::default())),
        )
        .with_field(FieldSchema::new("category_set", DataType::ArrayInt32).nullable())
        .with_field(
            FieldSchema::new("invert_category_set", DataType::ArrayInt32)
                .nullable()
                .with_index(IndexParams::Invert(InvertIndexParams::default())),
        );

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    fn make_doc(i: u64) -> Doc {
        let mut doc = Doc::new(format!("pk_{i}"))
            .set("id", i as i64)
            .set("invert_id", i as i64)
            .set("bool", i.is_multiple_of(100))
            .set("invert_bool", i.is_multiple_of(100))
            .set("name", format!("user_{}", i % 100))
            .set("invert_name", format!("user_{}", i % 100));

        if !i.is_multiple_of(100) {
            doc = doc.set("optional_age", (i % 100) as u32);
            doc = doc.set("invert_optional_age", (i % 100) as u32);
        }

        let category_size = i % 100;
        if category_size > 0 {
            let category: Vec<i32> = (1..=category_size as i32).collect();
            doc = doc
                .set("category_set", Value::ArrayI32(category.clone()))
                .set("invert_category_set", Value::ArrayI32(category));
        }

        let bool_array = if i.is_multiple_of(3) {
            vec![true, false, true]
        } else if i % 3 == 1 {
            vec![true, true, true]
        } else {
            vec![false, false, false]
        };
        doc = doc
            .set("bool_array", Value::ArrayBool(bool_array.clone()))
            .set("invert_bool_array", Value::ArrayBool(bool_array));

        doc
    }

    let mut batch: Vec<Doc> = Vec::new();
    for i in 0..10_000u64 {
        batch.push(make_doc(i));
        if batch.len() == 1024 {
            col.insert(std::mem::take(&mut batch)).unwrap();
        }
    }
    if !batch.is_empty() {
        col.insert(batch).unwrap();
    }
    col.flush().unwrap();

    // ForwardRecallTest.BoolContainAll: ids 0,3,6,...
    let q = VectorQuery::new("", vec![], 200)
        .with_filter("bool_array contain_all (true, false)")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    for (j, doc) in docs.iter().enumerate() {
        let id = (j as u64) * 3;
        assert_eq!(doc.pk, format!("pk_{id}"));
    }

    // ForwardRecallTest.BoolContainAny: skip i%3==2.
    let q = VectorQuery::new("", vec![], 200)
        .with_filter("bool_array contain_any (true)")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    let mut expected = 0u64;
    for doc in docs.iter() {
        assert_eq!(doc.pk, format!("pk_{expected}"));
        expected += 1;
        if expected % 3 == 2 {
            expected += 1;
        }
    }

    // ForwardRecallTest.BoolEqTrue: 0,100,200,... (100 matches total).
    let q = VectorQuery::new("", vec![], 200)
        .with_filter("bool = TRuE")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 100);
    for (j, doc) in docs.iter().enumerate() {
        let id = (j as u64) * 100;
        assert_eq!(doc.pk, format!("pk_{id}"));
    }

    // ForwardRecallTest.BoolEqFalse: start at 1, skip i%100==0.
    let q = VectorQuery::new("", vec![], 200)
        .with_filter("bool = false")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    let mut expected = 1u64;
    for doc in docs.iter() {
        assert_eq!(doc.pk, format!("pk_{expected}"));
        expected += 1;
        if expected.is_multiple_of(100) {
            expected += 1;
        }
    }

    // ForwardRecallTest.ArrayLengthEq
    let q = VectorQuery::new("", vec![], 200)
        .with_filter("array_length(category_set) = 32")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 100);
    for (j, doc) in docs.iter().enumerate() {
        let id = 32u64 + (j as u64) * 100;
        assert_eq!(doc.pk, format!("pk_{id}"));
    }

    // ForwardRecallTest.ArrayLengthGe
    let q = VectorQuery::new("", vec![], 200)
        .with_filter("array_length(category_set) >= 32")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    let mut expected = 32u64;
    for doc in docs.iter() {
        assert_eq!(doc.pk, format!("pk_{expected}"));
        expected += 1;
        while expected % 100 < 32 {
            expected += 1;
        }
    }

    // ForwardRecallTest.NotContainAny: excludes null category_set rows (i%100==0).
    let q = VectorQuery::new("", vec![], 200)
        .with_filter("category_set not contain_any (98,99,100)")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    let mut expected = 1u64;
    for doc in docs.iter() {
        assert_eq!(doc.pk, format!("pk_{expected}"));
        expected += 1;
        while expected % 100 >= 98 || expected.is_multiple_of(100) {
            expected += 1;
        }
    }

    // ForwardRecallTest.StrGe (forward): skip i%100==0
    let q = VectorQuery::new("", vec![], 200)
        .with_filter("name >= 'user_1'")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    let mut expected = 0u64;
    for doc in docs.iter() {
        if expected.is_multiple_of(100) {
            expected += 1;
        }
        assert_eq!(doc.pk, format!("pk_{expected}"));
        expected += 1;
    }

    // ForwardRecallTest.StrIn but with double quotes (reference query_info_test.cc style).
    let q = VectorQuery::new("", vec![], 200)
        .with_filter(r#"name IN ("user_1", "user_2")"#)
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    let mut expected = 1u64;
    for doc in docs.iter() {
        assert_eq!(doc.pk, format!("pk_{expected}"));
        if expected % 100 == 1 {
            expected += 1;
        } else if expected % 100 == 2 {
            expected += 99;
        } else {
            panic!("unexpected expected id state: {expected}");
        }
    }

    // InvertRecallTest.StrGe
    let q = VectorQuery::new("", vec![], 200)
        .with_filter("invert_name >= 'user_1'")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    let mut expected = 0u64;
    for doc in docs.iter() {
        if expected.is_multiple_of(100) {
            expected += 1;
        }
        assert_eq!(doc.pk, format!("pk_{expected}"));
        expected += 1;
    }

    // InvertRecallTest.NotContainAny: excludes null invert_category_set rows (i%100==0).
    let q = VectorQuery::new("", vec![], 200)
        .with_filter("invert_category_set not contain_any (98,99,100)")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    let mut expected = 1u64;
    for doc in docs.iter() {
        assert_eq!(doc.pk, format!("pk_{expected}"));
        expected += 1;
        while expected % 100 >= 98 || expected.is_multiple_of(100) {
            expected += 1;
        }
    }

    // InvertRecallTest.StrLike: escaped '_' is treated as literal underscore.
    let q = VectorQuery::new("", vec![], 200)
        .with_filter(r#"invert_name like 'user\_9%'"#)
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    let mut expected = 9u64;
    for doc in docs.iter() {
        assert_eq!(doc.pk, format!("pk_{expected}"));
        if expected % 100 == 9 {
            expected += 81;
        } else if expected % 100 == 99 {
            expected += 10;
        } else {
            expected += 1;
        }
    }

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_vector_query_recall_base_dense_semantics_match_reference() {
    let path = temp_dir("vector_recall_base_dense");
    let schema = CollectionSchema::new("test")
        .with_field(
            FieldSchema::new("id", DataType::Int64)
                .nullable()
                .not_null(),
        )
        .with_field(
            FieldSchema::new("invert_id", DataType::Int64)
                .nullable()
                .not_null()
                .with_index(IndexParams::Invert(InvertIndexParams::default())),
        )
        .with_field(
            FieldSchema::new("dense", DataType::VectorFp32)
                .nullable()
                .with_dimension(4)
                .with_index(IndexParams::Flat(FlatIndexParams::new(MetricType::L2))),
        );

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    fn make_doc(i: u64) -> Doc {
        let v = vec![i as f32; 4];
        Doc::new(format!("pk_{i}"))
            .set("id", i as i64)
            .set("invert_id", i as i64)
            .set("dense", v)
    }

    // Keep this smaller than reference's 10k doc recall_base to avoid ballooning test runtime,
    // while still validating ordering + score semantics.
    let mut batch: Vec<Doc> = Vec::new();
    for i in 0..2_000u64 {
        batch.push(make_doc(i));
        if batch.len() == 1024 {
            col.insert(std::mem::take(&mut batch)).unwrap();
        }
    }
    if !batch.is_empty() {
        col.insert(batch).unwrap();
    }
    col.flush().unwrap();

    // VectorRecallTest.Basic (dense, L2 squared distance).
    let q =
        VectorQuery::new("dense", vec![0.0f32; 4], 200).with_output_fields(vec!["id".to_string()]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    for (i, doc) in docs.iter().enumerate() {
        let id = i as u64;
        assert_eq!(doc.pk, format!("pk_{id}"));
        let expected = (id as f32) * (id as f32) * 4.0;
        assert!(
            (doc.score - expected).abs() < 1e-5,
            "score mismatch for {id}: got={}, expected={}",
            doc.score,
            expected
        );
    }

    // VectorRecallTest.HybridInvertFilter: invert_id >= 1.
    let q = VectorQuery::new("dense", vec![0.0f32; 4], 200)
        .with_filter("invert_id >= 1")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    for (j, doc) in docs.iter().enumerate() {
        let id = (j as u64) + 1;
        assert_eq!(doc.pk, format!("pk_{id}"));
        let expected = (id as f32) * (id as f32) * 4.0;
        assert!((doc.score - expected).abs() < 1e-5);
    }

    // VectorRecallTest.HybridInvertFilterBfByKeys: invert_id < 199, topk=199.
    let q = VectorQuery::new("dense", vec![0.0f32; 4], 199)
        .with_filter("invert_id < 199")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 199);
    for (j, doc) in docs.iter().enumerate() {
        let id = j as u64;
        assert_eq!(doc.pk, format!("pk_{id}"));
        let expected = (id as f32) * (id as f32) * 4.0;
        assert!((doc.score - expected).abs() < 1e-5);
    }

    // VectorRecallTest.HybridForwardFilter: id >= 1.
    let q = VectorQuery::new("dense", vec![0.0f32; 4], 200)
        .with_filter("id >= 1")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    for (j, doc) in docs.iter().enumerate() {
        let id = (j as u64) + 1;
        assert_eq!(doc.pk, format!("pk_{id}"));
        let expected = (id as f32) * (id as f32) * 4.0;
        assert!((doc.score - expected).abs() < 1e-5);
    }

    // VectorRecallTest.HybridInvertForwardFilter: invert_id >= 1 AND id <= 100.
    let q = VectorQuery::new("dense", vec![0.0f32; 4], 200)
        .with_filter("invert_id >= 1 and id <= 100")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 100);
    for (j, doc) in docs.iter().enumerate() {
        let id = (j as u64) + 1;
        assert_eq!(doc.pk, format!("pk_{id}"));
        let expected = (id as f32) * (id as f32) * 4.0;
        assert!((doc.score - expected).abs() < 1e-5);
    }

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_vector_query_recall_base_sparse_semantics_match_reference() {
    let path = temp_dir("vector_recall_base_sparse");
    let schema = CollectionSchema::new("test")
        .with_field(
            FieldSchema::new("id", DataType::Int64)
                .nullable()
                .not_null(),
        )
        .with_field(
            FieldSchema::new("sparse", DataType::SparseFp32)
                .nullable()
                .with_dimension(100)
                .with_index(IndexParams::FlatSparse(FlatIndexParams::new(
                    MetricType::InnerProduct,
                ))),
        );

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    fn make_doc(i: u64) -> Doc {
        let nnz = (i % 100) as usize;
        let mut indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);
        for j in 0..nnz {
            indices.push(j as u32);
            values.push(i as f32);
        }

        Doc::new(format!("pk_{i}"))
            .set("id", i as i64)
            // docs where i%100==0 set an "empty sparse vector" (treated as NULL)
            .set("sparse", Value::SparseF32 { indices, values })
    }

    // Keep this smaller than reference's 10k recall_base to avoid ballooning test runtime.
    let mut batch: Vec<Doc> = Vec::new();
    for i in 0..2_000u64 {
        batch.push(make_doc(i));
        if batch.len() == 1024 {
            col.insert(std::mem::take(&mut batch)).unwrap();
        }
    }
    if !batch.is_empty() {
        col.insert(batch).unwrap();
    }
    col.flush().unwrap();

    // reference forward recall expects empty sparse vectors to behave like NULL / not present.
    let fetched = col.fetch(vec!["pk_0".to_string()]).unwrap();
    let d0 = fetched.get("pk_0").expect("pk_0 exists");
    assert!(
        !d0.fields.contains_key("sparse"),
        "empty sparse vector should not round-trip as a present value"
    );

    // VectorRecallTest.Sparse: query sparse [0,1,2,3] values [1,1,1,1] with IP.
    let mut q = VectorQuery::new("sparse", Vec::new(), 200).with_output_fields(vec![]);
    q.sparse_indices = vec![0, 1, 2, 3];
    q.sparse_values = vec![1.0, 1.0, 1.0, 1.0];

    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);

    let mut expected = 1_999u64;
    for doc in docs.iter() {
        assert_eq!(doc.pk, format!("pk_{expected}"));
        let expected_score = (expected as f32) * 4.0;
        assert!((doc.score - expected_score).abs() < 1e-4);

        expected = expected.saturating_sub(1);
        while expected % 100 <= 3 {
            expected = expected.saturating_sub(1);
        }
    }

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_vector_query_dense_inner_product_score_semantics_match_reference() {
    let path = temp_dir("vector_dense_ip_score");
    let schema = CollectionSchema::new("test").with_field(
        FieldSchema::new("emb", DataType::VectorFp32)
            .nullable()
            .with_dimension(2)
            .with_index(IndexParams::Flat(FlatIndexParams::new(
                MetricType::InnerProduct,
            ))),
    );

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    col.insert(vec![
        Doc::new("d1").set("emb", vec![1.0f32, 0.0]), // dot=1.0
        Doc::new("d2").set("emb", vec![0.5f32, 0.0]), // dot=0.5
        Doc::new("d3").set("emb", vec![0.1f32, 0.0]), // dot=0.1
    ])
    .unwrap();
    col.flush().unwrap();

    let q = VectorQuery::new("emb", vec![1.0, 0.0], 3).with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 3);
    assert_eq!(docs[0].pk, "d1");
    assert!((docs[0].score - 1.0).abs() < 1e-6);
    assert_eq!(docs[1].pk, "d2");
    assert!((docs[1].score - 0.5).abs() < 1e-6);
    assert_eq!(docs[2].pk, "d3");
    assert!((docs[2].score - 0.1).abs() < 1e-6);

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_vector_query_dense_mips_l2_distance_semantics_match_reference() {
    fn run_case(case_name: &str, quantize: QuantizeType) {
        let path = temp_dir(&format!("vector_dense_mips_l2_{case_name}"));
        let schema = CollectionSchema::new("test").with_field(
            FieldSchema::new("emb", DataType::VectorFp32)
                .nullable()
                .with_dimension(2)
                .with_index(IndexParams::Flat(
                    FlatIndexParams::new(MetricType::MipsL2).with_quantize(quantize),
                )),
        );

        let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
        col.insert(vec![
            Doc::new("d0").set("emb", vec![2.0f32, 4.0]),
            Doc::new("d1").set("emb", vec![3.0f32, 4.0]),
            Doc::new("d2").set("emb", vec![1.0f32, 0.0]),
        ])
        .unwrap();
        col.flush().unwrap();

        let qv = vec![1.0f32, 2.0];
        let q = VectorQuery::new("emb", qv.clone(), 3).with_output_fields(vec![]);
        let docs = col.query(q).unwrap();
        assert_eq!(docs.len(), 3);

        fn expected(v: &[f32], q: &[f32]) -> f32 {
            let mut ip = 0.0f32;
            let mut u2 = 0.0f32;
            let mut v2 = 0.0f32;
            for (x, y) in v.iter().zip(q.iter()) {
                ip += x * y;
                u2 += x * x;
                v2 += y * y;
            }
            let denom = u2.max(v2);
            2.0 - 2.0 * ip / denom
        }

        let d0 = expected(&[2.0, 4.0], &qv);
        let d1 = expected(&[3.0, 4.0], &qv);
        let d2 = expected(&[1.0, 0.0], &qv);

        assert_eq!(docs[0].pk, "d0");
        assert!(
            (docs[0].score - d0).abs() < 1e-4,
            "got={} expected={}",
            docs[0].score,
            d0
        );
        assert_eq!(docs[1].pk, "d1");
        assert!(
            (docs[1].score - d1).abs() < 1e-4,
            "got={} expected={}",
            docs[1].score,
            d1
        );
        assert_eq!(docs[2].pk, "d2");
        assert!(
            (docs[2].score - d2).abs() < 1e-4,
            "got={} expected={}",
            docs[2].score,
            d2
        );

        drop(col);
        std::fs::remove_dir_all(&path).ok();
    }

    run_case("f32", QuantizeType::Undefined);
    run_case("fp16", QuantizeType::Fp16);
}

#[test]
fn test_query_ip_scores_upsert_replacement_and_include_vector_across_indexes() {
    // Port of reference's `vector_column_indexer_test.cc` at the observable boundary:
    // - inner-product scores returned as similarity (dot, larger is better),
    // - duplicate PK upsert replaces prior vector (tombstone semantics),
    // - `include_vector` round-trips the stored vector values.
    fn run_case(case_name: &str, index_params: IndexParams) {
        let path = temp_dir(&format!("vector_column_indexer_{case_name}"));
        let schema = CollectionSchema::new("test").with_field(
            FieldSchema::new("emb", DataType::VectorFp32)
                .nullable()
                .with_dimension(4),
        );

        let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

        col.insert(vec![
            Doc::new("d1").set("emb", vec![1.0f32, 2.0, 3.0, 0.0]),
            Doc::new("d2").set("emb", vec![1.0f32, 2000.0, 3.0, 0.0]),
        ])
        .unwrap();

        let d2_old_id = col
            .fetch(vec!["d2".to_string()])
            .unwrap()
            .get("d2")
            .unwrap()
            .doc_id;

        // reference's column indexer overwrites by doc_id; Finch overwrites by PK via upsert, producing
        // a new doc_id and tombstoning the previous version.
        col.upsert(vec![Doc::new("d2").set("emb", vec![1.0f32, 0.0, 3.0, 0.0])])
            .unwrap();

        let fetched = col.fetch(vec!["d1".to_string(), "d2".to_string()]).unwrap();
        let d1_id = fetched.get("d1").unwrap().doc_id;
        let d2_id = fetched.get("d2").unwrap().doc_id;
        assert_ne!(d2_old_id, d2_id, "upsert must allocate a new doc_id");

        col.create_index(
            "emb",
            index_params.clone(),
            CreateIndexOptions {
                rebuild: true,
                concurrency: None,
            },
        )
        .unwrap();
        col.flush().unwrap();

        let wants_refiner = index_params
            .quantize()
            .is_some_and(|q| q != QuantizeType::Undefined);

        let qp = QueryParams {
            use_refiner: wants_refiner,
            ..QueryParams::default()
        };

        let mut q = VectorQuery::new("emb", vec![1.0, 2.0, 3.0, 0.0], 10).with_params(qp);
        q.include_vector = true;
        q.include_doc_id = true;
        q.output_fields = None;

        let docs = col.query(q).unwrap();
        assert_eq!(docs.len(), 2);

        assert_eq!(docs[0].pk, "d1");
        assert!(
            (docs[0].score - 14.0).abs() < 1e-4,
            "score={}",
            docs[0].score
        );
        assert_eq!(docs[0].doc_id, d1_id);
        match docs[0].fields.get("emb") {
            Some(Value::VecF32(v)) => assert_eq!(v, &vec![1.0, 2.0, 3.0, 0.0]),
            other => panic!("expected emb VecF32, got {other:?}"),
        }

        assert_eq!(docs[1].pk, "d2");
        assert!(
            (docs[1].score - 10.0).abs() < 1e-4,
            "score={}",
            docs[1].score
        );
        assert_eq!(docs[1].doc_id, d2_id);
        assert_ne!(docs[0].doc_id, docs[1].doc_id);
        match docs[1].fields.get("emb") {
            Some(Value::VecF32(v)) => assert_eq!(v, &vec![1.0, 0.0, 3.0, 0.0]),
            other => panic!("expected emb VecF32, got {other:?}"),
        }

        drop(col);
        std::fs::remove_dir_all(&path).ok();
    }

    let metric = MetricType::InnerProduct;
    let quantizes = [
        QuantizeType::Undefined,
        QuantizeType::Fp16,
        QuantizeType::Int8,
        QuantizeType::Int4,
    ];

    for q in quantizes {
        run_case(
            &format!("flat_{q:?}"),
            IndexParams::Flat(FlatIndexParams::new(metric).with_quantize(q)),
        );
        run_case(
            &format!("hnsw_{q:?}"),
            IndexParams::Hnsw(
                HnswIndexParams::new(metric)
                    .with_m(10)
                    .with_ef_construction(100)
                    .with_quantize(q),
            ),
        );
        run_case(
            &format!("ivf_{q:?}"),
            IndexParams::Ivf(IvfIndexParams::new(metric).with_n_list(2).with_quantize(q)),
        );
    }
}

#[test]
fn test_fp16_vector_field_query_returns_stored_vectors() {
    // Mirrors reference's `DenseDataTypeFP16` behavior (fp16 storage + fetch/search).
    fn run_case(case_name: &str, index_params: IndexParams) {
        let path = temp_dir(&format!("vector_column_indexer_fp16_{case_name}"));
        let schema = CollectionSchema::new("test").with_field(
            FieldSchema::new("emb", DataType::VectorFp16)
                .nullable()
                .with_dimension(4),
        );

        let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

        // Insert fp32 payloads into an fp16 field (reference python-style behavior: cast to fp16).
        col.insert(vec![
            Doc::new("d1").set("emb", vec![1.0f32, 2.0, 3.0, 0.0]),
            Doc::new("d2").set("emb", vec![1.0f32, 2000.0, 3.0, 0.0]),
        ])
        .unwrap();

        col.upsert(vec![Doc::new("d2").set("emb", vec![1.0f32, 0.0, 3.0, 0.0])])
            .unwrap();

        col.create_index(
            "emb",
            index_params,
            CreateIndexOptions {
                rebuild: true,
                concurrency: None,
            },
        )
        .unwrap();
        col.flush().unwrap();

        let mut q = VectorQuery::new("emb", vec![1.0, 2.0, 3.0, 0.0], 10);
        q.include_vector = true;
        q.include_doc_id = true;

        let docs = col.query(q).unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].pk, "d1");
        assert_eq!(docs[1].pk, "d2");
        assert!(
            (docs[0].score - 14.0).abs() < 1e-4,
            "score={}",
            docs[0].score
        );
        assert!(
            (docs[1].score - 10.0).abs() < 1e-4,
            "score={}",
            docs[1].score
        );

        match docs[0].fields.get("emb") {
            Some(Value::VecF32(v)) => assert_eq!(v, &vec![1.0, 2.0, 3.0, 0.0]),
            other => panic!("expected emb VecF32, got {other:?}"),
        }
        match docs[1].fields.get("emb") {
            Some(Value::VecF32(v)) => assert_eq!(v, &vec![1.0, 0.0, 3.0, 0.0]),
            other => panic!("expected emb VecF32, got {other:?}"),
        }

        drop(col);
        std::fs::remove_dir_all(&path).ok();
    }

    let metric = MetricType::InnerProduct;
    run_case("flat", IndexParams::Flat(FlatIndexParams::new(metric)));
    run_case(
        "hnsw",
        IndexParams::Hnsw(
            HnswIndexParams::new(metric)
                .with_m(10)
                .with_ef_construction(100),
        ),
    );
    run_case(
        "ivf",
        IndexParams::Ivf(IvfIndexParams::new(metric).with_n_list(2)),
    );
}

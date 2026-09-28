use super::*;

#[test]
fn test_int8_vector_field_query_returns_stored_vectors() {
    // Mirrors reference's `DenseDataTypeINT8` behavior at Finch's API boundary.
    fn run_case(case_name: &str, index_params: IndexParams) {
        let path = temp_dir(&format!("vector_column_indexer_int8_{case_name}"));
        let schema = CollectionSchema::new("test").with_field(
            FieldSchema::new("emb", DataType::VectorInt8)
                .nullable()
                .with_dimension(4),
        );

        let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

        col.insert(vec![
            Doc::new("d1").set("emb", vec![1i8, 2, 3, 0]),
            Doc::new("d2").set("emb", vec![1i8, 100, 3, 0]),
        ])
        .unwrap();

        col.upsert(vec![Doc::new("d2").set("emb", vec![1i8, 0, 3, 0])])
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
}

#[test]
fn test_sparse_vector_field_query_returns_stored_vectors() {
    fn run_case(case_name: &str, index_params: IndexParams) {
        let path = temp_dir(&format!("vector_column_indexer_sparse_{case_name}"));
        let schema = CollectionSchema::new("test").with_field(
            FieldSchema::new("s", DataType::SparseFp32)
                .nullable()
                .with_dimension(8),
        );

        let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

        let q_idx = vec![0u32, 1u32, 2u32];
        let q_vals = vec![1.0f32, 2.0, 3.0];

        col.insert(vec![
            Doc::new("d1").set(
                "s",
                Value::SparseF32 {
                    indices: q_idx.clone(),
                    values: q_vals.clone(),
                },
            ),
            Doc::new("d2").set(
                "s",
                Value::SparseF32 {
                    indices: vec![0u32, 2u32],
                    values: vec![1.0f32, 3.0],
                },
            ),
        ])
        .unwrap();

        col.create_index(
            "s",
            index_params,
            CreateIndexOptions {
                rebuild: true,
                concurrency: None,
            },
        )
        .unwrap();
        col.flush().unwrap();

        let mut q = VectorQuery::new("s", Vec::new(), 10);
        q.sparse_indices = q_idx.clone();
        q.sparse_values = q_vals.clone();
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

        match docs[0].fields.get("s") {
            Some(Value::SparseF32 { indices, values }) => {
                assert_eq!(indices, &q_idx);
                assert_eq!(values, &q_vals);
            }
            other => panic!("expected s SparseF32, got {other:?}"),
        }

        drop(col);
        std::fs::remove_dir_all(&path).ok();
    }

    let metric = MetricType::InnerProduct;
    run_case(
        "flat_sparse",
        IndexParams::FlatSparse(FlatIndexParams::new(metric)),
    );
    run_case(
        "hnsw_sparse",
        IndexParams::HnswSparse(
            HnswIndexParams::new(metric)
                .with_m(10)
                .with_ef_construction(100),
        ),
    );
}

#[test]
fn test_int8_query_vector_f32_inputs_are_cast_to_int8() {
    let path = temp_dir("int8_query_cast");
    let schema = CollectionSchema::new("test").with_field(
        FieldSchema::new("emb", DataType::VectorInt8)
            .nullable()
            .with_dimension(4)
            .with_index(IndexParams::Flat(FlatIndexParams::new(
                MetricType::InnerProduct,
            ))),
    );

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    col.insert(vec![
        Doc::new("d1").set("emb", vec![1i8, 2, 3, 0]),
        Doc::new("d2").set("emb", vec![1i8, 0, 3, 0]),
    ])
    .unwrap();
    col.flush().unwrap();

    let mut q = VectorQuery::new("emb", vec![1.9f32, 2.1, 3.0, 0.0], 2);
    q.include_doc_id = true;

    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 2);
    assert_eq!(docs[0].pk, "d1");
    // If query floats were not cast to int8, the dot would be 15.1; reference-style
    // int8 query semantics require truncation to (1,2,3,0) => dot 14.0.
    assert!(
        (docs[0].score - 14.0).abs() < 1e-4,
        "score={}",
        docs[0].score
    );

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_vector_query_dense_cosine_distance_semantics_match_reference() {
    let path = temp_dir("vector_dense_cosine_score");
    let schema = CollectionSchema::new("test").with_field(
        FieldSchema::new("emb", DataType::VectorFp32)
            .nullable()
            .with_dimension(2)
            .with_index(IndexParams::Flat(FlatIndexParams::new(MetricType::Cosine))),
    );

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    col.insert(vec![
        Doc::new("d1").set("emb", vec![1.0f32, 0.0]), // cosine dist = 0
        Doc::new("d2").set("emb", vec![0.0f32, 1.0]), // cosine dist = 1
        Doc::new("d3").set("emb", vec![-1.0f32, 0.0]), // cosine dist = 2
    ])
    .unwrap();
    col.flush().unwrap();

    let q = VectorQuery::new("emb", vec![1.0, 0.0], 3).with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 3);

    assert_eq!(docs[0].pk, "d1");
    assert!((docs[0].score - 0.0).abs() < 1e-6);
    assert_eq!(docs[1].pk, "d2");
    assert!((docs[1].score - 1.0).abs() < 1e-6);
    assert_eq!(docs[2].pk, "d3");
    assert!((docs[2].score - 2.0).abs() < 1e-6);

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_filter_only_query_forward_and_invert_recall_semantics_match_reference() {
    let path = temp_dir("filter_only_forward_invert_recall");

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
            FieldSchema::new("age", DataType::Int32)
                .nullable()
                .not_null(),
        )
        .with_field(
            FieldSchema::new("invert_age", DataType::Int32)
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
        );

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    fn make_doc(i: u64) -> Doc {
        let mut doc = Doc::new(format!("pk_{i}"))
            .set("id", i as i64)
            .set("invert_id", i as i64)
            .set("age", (i % 100) as i32)
            .set("invert_age", (i % 100) as i32)
            .set("name", format!("user_{}", i % 100))
            .set("invert_name", format!("user_{}", i % 100));

        if !i.is_multiple_of(100) {
            doc = doc.set("optional_age", (i % 100) as u32);
            doc = doc.set("invert_optional_age", (i % 100) as u32);
        }

        doc
    }

    // Smaller than reference's 10k, but exercises all modulo-100 patterns + >1000 cutoffs.
    const N: u64 = 2_000;
    let mut batch: Vec<Doc> = Vec::new();
    for i in 0..N {
        batch.push(make_doc(i));
        if batch.len() == 1024 {
            col.insert(std::mem::take(&mut batch)).unwrap();
        }
    }
    if !batch.is_empty() {
        col.insert(batch).unwrap();
    }
    col.flush().unwrap();

    fn pk(i: u64) -> String {
        format!("pk_{i}")
    }

    // ── Forward numeric recall suite (reference forward_recall_test.cc patterns) ──
    let q = VectorQuery::new("", vec![], 200)
        .with_filter("age = 1")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 20);
    for (j, doc) in docs.iter().enumerate() {
        let id = 1u64 + (j as u64) * 100;
        assert_eq!(doc.pk, pk(id));
    }

    let q = VectorQuery::new("", vec![], 200)
        .with_filter("id > 1000")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    for (j, doc) in docs.iter().enumerate() {
        let id = 1001u64 + (j as u64);
        assert_eq!(doc.pk, pk(id));
    }

    let q = VectorQuery::new("", vec![], 200)
        .with_filter("id >= 1000")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    for (j, doc) in docs.iter().enumerate() {
        let id = 1000u64 + (j as u64);
        assert_eq!(doc.pk, pk(id));
    }

    let q = VectorQuery::new("", vec![], 200)
        .with_filter("id < 100")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 100);
    for (j, doc) in docs.iter().enumerate() {
        let id = j as u64;
        assert_eq!(doc.pk, pk(id));
    }

    let q = VectorQuery::new("", vec![], 200)
        .with_filter("id <= 100")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 101);
    for (j, doc) in docs.iter().enumerate() {
        let id = j as u64;
        assert_eq!(doc.pk, pk(id));
    }

    let q = VectorQuery::new("", vec![], 200)
        .with_filter("id <= 100 and id > 50")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 50);
    for (j, doc) in docs.iter().enumerate() {
        let id = 51u64 + (j as u64);
        assert_eq!(doc.pk, pk(id));
    }

    let q = VectorQuery::new("", vec![], 200)
        .with_filter("id < 100 or id > 200")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    for (j, doc) in docs.iter().enumerate() {
        let id = if j < 100 { j as u64 } else { (j as u64) + 101 };
        assert_eq!(doc.pk, pk(id));
    }

    // ── Forward string recall suite ──
    let q = VectorQuery::new("", vec![], 200)
        .with_filter("name = 'user_1'")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 20);
    for (j, doc) in docs.iter().enumerate() {
        let id = 1u64 + (j as u64) * 100;
        assert_eq!(doc.pk, pk(id));
    }

    let q = VectorQuery::new("", vec![], 200)
        .with_filter("name IN ('user_1', 'user_2')")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 40);
    let mut expected = 1u64;
    for doc in docs.iter() {
        assert_eq!(doc.pk, pk(expected));
        if expected % 100 == 1 {
            expected += 1;
        } else if expected % 100 == 2 {
            expected += 99;
        } else {
            panic!("unexpected expected id state: {expected}");
        }
    }

    let q = VectorQuery::new("", vec![], 200)
        .with_filter("name NOT IN ('user_1', 'user_2')")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    let mut expected = 0u64;
    for doc in docs.iter() {
        assert_eq!(doc.pk, pk(expected));
        if expected.is_multiple_of(100) {
            expected += 3;
        } else {
            expected += 1;
        }
    }

    // reference forward_recall_test.cc: name like 'user_9%'
    let q = VectorQuery::new("", vec![], 200)
        .with_filter("name like 'user_9%'")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    let mut expected = 9u64;
    for doc in docs.iter() {
        assert_eq!(doc.pk, pk(expected));
        if expected % 100 == 9 {
            expected += 81;
        } else if expected % 100 == 99 {
            expected += 10;
        } else {
            expected += 1;
        }
    }

    // ── NULL semantics (forward) ──
    let q = VectorQuery::new("", vec![], 200)
        .with_filter("optional_age is null")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 20);
    for (j, doc) in docs.iter().enumerate() {
        let id = (j as u64) * 100;
        assert_eq!(doc.pk, pk(id));
    }

    // ── Invert numeric recall suite (invert_* fields) ──
    let q = VectorQuery::new("", vec![], 200)
        .with_filter("invert_id > 1000")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    for (j, doc) in docs.iter().enumerate() {
        let id = 1001u64 + (j as u64);
        assert_eq!(doc.pk, pk(id));
    }

    let q = VectorQuery::new("", vec![], 200)
        .with_filter("invert_id <= 100 and invert_id > 50")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 50);
    for (j, doc) in docs.iter().enumerate() {
        let id = 51u64 + (j as u64);
        assert_eq!(doc.pk, pk(id));
    }

    let q = VectorQuery::new("", vec![], 200)
        .with_filter("invert_name NOT IN ('user_1', 'user_2')")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    let mut expected = 0u64;
    for doc in docs.iter() {
        assert_eq!(doc.pk, pk(expected));
        if expected.is_multiple_of(100) {
            expected += 3;
        } else {
            expected += 1;
        }
    }

    // reference invert_recall_test.cc: invert_name like 'user\\_9%'
    let q = VectorQuery::new("", vec![], 200)
        .with_filter(r#"invert_name like 'user\_9%'"#)
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 200);
    let mut expected = 9u64;
    for doc in docs.iter() {
        assert_eq!(doc.pk, pk(expected));
        if expected % 100 == 9 {
            expected += 81;
        } else if expected % 100 == 99 {
            expected += 10;
        } else {
            expected += 1;
        }
    }

    // ── NULL semantics (invert) ──
    let q = VectorQuery::new("", vec![], 200)
        .with_filter("invert_optional_age is null")
        .with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 20);
    for (j, doc) in docs.iter().enumerate() {
        let id = (j as u64) * 100;
        assert_eq!(doc.pk, pk(id));
    }

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_optimize_removes_deleted_docs_and_old_segments() {
    let path = temp_dir("optimize_deletes_cleanup");
    let schema = CollectionSchema::new("test")
        .with_field(
            FieldSchema::new("id", DataType::Int64)
                .nullable()
                .not_null(),
        )
        .with_field(
            FieldSchema::new("emb", DataType::VectorFp32)
                .nullable()
                .with_dimension(4)
                .with_index(IndexParams::Flat(FlatIndexParams::new(MetricType::L2))),
        );

    let opts = CollectionOptions {
        max_buffer_size: 1024, // force multiple flush/rotate during insert
        ..CollectionOptions::default()
    };
    let col = Collection::create_and_open(&path, schema, opts).unwrap();

    let mut docs = Vec::new();
    for i in 0..200u64 {
        docs.push(
            Doc::new(format!("d{i}"))
                .set("id", i as i64)
                .set("emb", vec![i as f32; 4]),
        );
        if docs.len() == 128 {
            col.insert(std::mem::take(&mut docs)).unwrap();
        }
    }
    if !docs.is_empty() {
        col.insert(docs).unwrap();
    }

    // Ensure we actually created more than one persisted segment directory.
    col.flush().unwrap();
    let seg_dirs_before: Vec<_> = std::fs::read_dir(&path)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .filter(|e| e.file_name().to_string_lossy().starts_with("seg_"))
        .collect();
    assert!(
        seg_dirs_before.len() >= 2,
        "expected multiple persisted segments before optimize"
    );

    // Delete a prefix of docs.
    let to_delete: Vec<String> = (0..10u64).map(|i| format!("d{i}")).collect();
    let statuses = col.delete(to_delete).unwrap();
    assert!(statuses.iter().all(|s| s.is_ok()));

    // Optimize into a single segment and verify tombstones are applied.
    col.optimize(OptimizeOptions {
        target_segment_size: Some(u64::MAX),
        ..Default::default()
    })
    .unwrap();

    let seg_dirs_after: Vec<_> = std::fs::read_dir(&path)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .filter(|e| e.file_name().to_string_lossy().starts_with("seg_"))
        .collect();
    assert_eq!(
        seg_dirs_after.len(),
        1,
        "optimize should remove old segment directories"
    );

    let stats = col.stats().unwrap();
    assert_eq!(
        stats.doc_count, 190,
        "deleted docs should not count as live after optimize"
    );

    // Query should skip deleted docs (d0..d9) and return d10.. first.
    let q = VectorQuery::new("emb", vec![0.0f32; 4], 5).with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 5);
    assert_eq!(docs[0].pk, "d10");
    assert_eq!(docs[1].pk, "d11");

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_optimize_resets_delete_bitmap_and_stats_for_middle_deletes() {
    let path = temp_dir("optimize_delete_reset_middle");
    let schema = CollectionSchema::new("test")
        .with_field(
            FieldSchema::new("id", DataType::Int64)
                .nullable()
                .not_null(),
        )
        .with_field(
            FieldSchema::new("emb", DataType::VectorFp32)
                .nullable()
                .with_dimension(4)
                .with_index(IndexParams::Flat(FlatIndexParams::new(MetricType::L2))),
        );

    let opts = CollectionOptions {
        max_buffer_size: 1024, // force multiple persisted segments
        ..CollectionOptions::default()
    };
    let col = Collection::create_and_open(&path, schema, opts).unwrap();

    let mut docs = Vec::new();
    for i in 0..200u64 {
        docs.push(
            Doc::new(format!("d{i}"))
                .set("id", i as i64)
                .set("emb", vec![i as f32; 4]),
        );
        if docs.len() == 128 {
            col.insert(std::mem::take(&mut docs)).unwrap();
        }
    }
    if !docs.is_empty() {
        col.insert(docs).unwrap();
    }

    col.flush().unwrap();

    fn current_delete_suffix(dir: &std::path::Path) -> u32 {
        let mut suffixes: Vec<u32> = std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                let n = name.strip_prefix("delete_")?.strip_suffix(".bitmap")?;
                n.parse::<u32>().ok()
            })
            .collect();
        suffixes.sort_unstable();
        assert!(
            !suffixes.is_empty(),
            "expected at least one delete_*.bitmap in collection dir"
        );
        *suffixes.last().unwrap()
    }

    let suffix_before_delete = current_delete_suffix(&path);

    // Delete a doc in the middle of the doc_id range (not prefix/suffix).
    let statuses = col.delete(vec!["d50".to_string()]).unwrap();
    assert_eq!(statuses.len(), 1);
    assert!(statuses[0].is_ok());
    let suffix_after_delete = current_delete_suffix(&path);
    assert_eq!(
        suffix_after_delete,
        suffix_before_delete + 1,
        "delete() should rotate the delete bitmap suffix"
    );
    assert!(
        !path
            .join(format!("delete_{}.bitmap", suffix_before_delete))
            .exists(),
        "expected previous delete bitmap to be removed after delete() rotation"
    );

    col.optimize(OptimizeOptions {
        target_segment_size: Some(u64::MAX),
        ..Default::default()
    })
    .unwrap();

    let suffix_after_optimize = current_delete_suffix(&path);
    assert_eq!(
        suffix_after_optimize, suffix_after_delete,
        "reference parity: optimize() should not rotate/reset delete tombstones"
    );
    assert!(
        path.join(format!("delete_{}.bitmap", suffix_after_optimize))
            .exists(),
        "expected current delete bitmap to exist after optimize()"
    );

    let stats = col.stats().unwrap();
    assert_eq!(
        stats.doc_count, 199,
        "stats should not undercount after optimize()"
    );

    // Query near the deleted point should not return d50.
    let q = VectorQuery::new("emb", vec![50.0f32; 4], 5).with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert!(docs.iter().all(|d| d.pk != "d50"));
    assert!(
        docs.iter().any(|d| d.pk == "d49") || docs.iter().any(|d| d.pk == "d51"),
        "expected nearest neighbors around the deleted point"
    );

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_optimize_rebuild_drops_deleted_docs_when_delete_ratio_is_high() {
    use finch_db::version::VersionManager;

    let path = temp_dir("optimize_rebuild_threshold");
    let schema = CollectionSchema::new("test")
        .with_field(
            FieldSchema::new("id", DataType::Int64)
                .nullable()
                .not_null(),
        )
        .with_field(
            FieldSchema::new("emb", DataType::VectorFp32)
                .nullable()
                .with_dimension(4)
                .with_index(IndexParams::Flat(FlatIndexParams::new(MetricType::L2))),
        );

    let opts = CollectionOptions {
        max_buffer_size: 1024,
        ..CollectionOptions::default()
    };
    let col = Collection::create_and_open(&path, schema, opts).unwrap();

    // Insert 100 docs and flush.
    let docs: Vec<Doc> = (0..100u64)
        .map(|i| {
            Doc::new(format!("d{i}"))
                .set("id", i as i64)
                .set("emb", vec![i as f32; 4])
        })
        .collect();
    for chunk in docs.chunks(128) {
        col.insert(chunk.to_vec()).unwrap();
    }
    col.flush().unwrap();

    // Delete 50% of docs -> should cross rebuild threshold (0.3) and physically drop deleted docs.
    let to_delete: Vec<String> = (0..50u64).map(|i| format!("d{i}")).collect();
    let statuses = col.delete(to_delete).unwrap();
    assert!(statuses.iter().all(|s| s.is_ok()));

    let suffix_after_delete = VersionManager::load(&path).unwrap().current().delete_suffix;

    col.optimize(OptimizeOptions {
        target_segment_size: Some(u64::MAX),
        ..Default::default()
    })
    .unwrap();

    let vm = VersionManager::load(&path).unwrap();
    let v = vm.current();
    assert_eq!(
        v.delete_suffix, suffix_after_delete,
        "reference parity: optimize should not rotate/reset delete tombstones"
    );

    let stats = col.stats().unwrap();
    assert_eq!(
        stats.doc_count, 50,
        "rebuild optimize should drop deleted docs"
    );

    let persisted_count: u64 = v.persisted_segments.iter().map(|s| s.doc_count).sum();
    assert_eq!(
        persisted_count, 50,
        "persisted forward stores should only contain live docs"
    );

    // Query near the deleted region should not return d0..d49; nearest should be d50.
    let q = VectorQuery::new("emb", vec![0.0f32; 4], 1).with_output_fields(vec![]);
    let docs = col.query(q).unwrap();
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].pk, "d50");

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

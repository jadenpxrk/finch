mod common;
use common::*;

#[test]
fn test_collection_lifecycle() {
    let path = temp_dir("lifecycle");
    let schema = basic_schema(4);

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default())
        .expect("create_and_open");

    let stats = col.stats().expect("stats");
    assert_eq!(stats.doc_count, 0);
    assert_eq!(stats.segment_count, 1);

    // Double-create should fail
    let schema2 = basic_schema(4);
    assert!(Collection::create_and_open(&path, schema2, CollectionOptions::default()).is_err());

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_destroy_removes_collection_dir() {
    let path = temp_dir("destroy");
    let schema = basic_schema(4);

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default())
        .expect("create_and_open");
    assert!(path.exists());

    let col = match Arc::try_unwrap(col) {
        Ok(col) => col,
        Err(_) => panic!("collection should have a single Arc owner in this test"),
    };
    col.destroy().expect("destroy should succeed");
    assert!(!path.exists());
}

#[test]
fn test_open_missing() {
    let path = temp_dir("missing");
    assert!(Collection::open(&path, CollectionOptions::default()).is_err());
}

#[test]
fn test_open_empty_dir_fails() {
    let path = temp_dir("empty_dir");
    std::fs::create_dir_all(&path).unwrap();
    assert!(Collection::open(&path, CollectionOptions::default()).is_err());
}

#[test]
fn test_open_fails_when_segment_forward_store_is_missing() {
    fn find_forward_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        let Ok(rd) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in rd.flatten() {
            let path = entry.path();
            if path.is_dir() {
                find_forward_files(&path, out);
            } else if path.file_name().and_then(|s| s.to_str()) == Some("forward.arrow") {
                out.push(path);
            }
        }
    }

    let path = temp_dir("open_corrupt_forward");
    let schema = CollectionSchema::new("corruption")
        .with_field(
            FieldSchema::new("id", DataType::Int64).with_index(IndexParams::Invert(
                InvertIndexParams {
                    enable_range_optimization: true,
                    enable_extended_wildcard: false,
                },
            )),
        )
        .with_field(
            FieldSchema::new("name", DataType::String).with_index(IndexParams::Invert(
                InvertIndexParams {
                    enable_range_optimization: false,
                    enable_extended_wildcard: false,
                },
            )),
        )
        .with_field(
            FieldSchema::new("emb", DataType::VectorFp32)
                .with_dimension(8)
                .with_index(IndexParams::Hnsw(HnswIndexParams::new(MetricType::L2))),
        );

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let docs = vec![Doc::new("d1")
        .set("id", 1i64)
        .set("name", "test")
        .set("emb", vec![0.1f32, 0.2, 0.0, 0.3, 0.0, 0.0, 0.4, 0.0])];
    let st = col.insert(docs).unwrap();
    assert!(
        st.iter().all(|s| s.is_ok()),
        "expected insert to succeed, got statuses: {st:?}"
    );
    col.flush().unwrap();
    drop(col);

    let mut forwards = Vec::new();
    find_forward_files(&path, &mut forwards);
    assert!(
        !forwards.is_empty(),
        "expected at least one forward.arrow file"
    );

    std::fs::remove_file(&forwards[0]).unwrap();

    assert!(Collection::open(&path, CollectionOptions::default()).is_err());
}

#[test]
fn test_open_fails_when_segment_invert_index_is_missing() {
    let path = temp_dir("open_corrupt_invert");
    let schema = CollectionSchema::new("corruption")
        .with_field(
            FieldSchema::new("id", DataType::Int64).with_index(IndexParams::Invert(
                InvertIndexParams {
                    enable_range_optimization: true,
                    enable_extended_wildcard: false,
                },
            )),
        )
        .with_field(
            FieldSchema::new("name", DataType::String).with_index(IndexParams::Invert(
                InvertIndexParams {
                    enable_range_optimization: false,
                    enable_extended_wildcard: false,
                },
            )),
        )
        .with_field(
            FieldSchema::new("emb", DataType::VectorFp32)
                .with_dimension(8)
                .with_index(IndexParams::Hnsw(HnswIndexParams::new(MetricType::L2))),
        );

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let docs = vec![Doc::new("d1")
        .set("id", 1i64)
        .set("name", "test")
        .set("emb", vec![0.1f32, 0.2, 0.0, 0.3, 0.0, 0.0, 0.4, 0.0])];
    let st = col.insert(docs).unwrap();
    assert!(st.iter().all(|s| s.is_ok()));
    col.flush().unwrap();
    drop(col);

    // Persisted segment layout is deterministic: seg_{id}/{field}_invert.keys
    let invert_file = path.join("seg_0").join("name_invert.keys");
    assert!(
        invert_file.exists(),
        "expected invert file at {invert_file:?}"
    );
    std::fs::remove_file(&invert_file).unwrap();

    assert!(Collection::open(&path, CollectionOptions::default()).is_err());
}

#[test]
fn test_open_fails_when_segment_vector_index_is_missing() {
    let path = temp_dir("open_corrupt_vector_index");
    let schema = CollectionSchema::new("corruption")
        .with_field(FieldSchema::new("id", DataType::Int64))
        .with_field(
            FieldSchema::new("emb", DataType::VectorFp32)
                .with_dimension(8)
                .with_index(IndexParams::Hnsw(HnswIndexParams::new(MetricType::L2))),
        );

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let docs = vec![Doc::new("d1")
        .set("id", 1i64)
        .set("emb", vec![0.1f32, 0.2, 0.0, 0.3, 0.0, 0.0, 0.4, 0.0])];
    let st = col.insert(docs).unwrap();
    assert!(st.iter().all(|s| s.is_ok()));
    col.flush().unwrap();

    // optimize builds the vector indexes the schema declares.
    col.optimize(OptimizeOptions::default()).unwrap();
    drop(col);

    let idx_dir = path.join("seg_0").join("idx_emb");
    assert!(idx_dir.exists(), "expected vector index dir at {idx_dir:?}");
    std::fs::remove_dir_all(&idx_dir).unwrap();

    assert!(Collection::open(&path, CollectionOptions::default()).is_err());
}

#[test]
fn test_scalar_only_collection_lifecycle_fetch_and_sql() {
    let path = temp_dir("scalar_only");
    let schema = CollectionSchema::new("state")
        .with_field(FieldSchema::new("kind", DataType::String))
        .with_field(FieldSchema::new("counter", DataType::Int64))
        .with_field(FieldSchema::new("active", DataType::Bool));

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let docs = vec![
        Doc::new("s1")
            .set("kind", "session")
            .set("counter", 1i64)
            .set("active", true),
        Doc::new("s2")
            .set("kind", "session")
            .set("counter", 2i64)
            .set("active", false),
    ];
    let st = col.insert(docs).unwrap();
    assert!(
        st.iter().all(|s| s.is_ok()),
        "expected insert to succeed, got {st:?}"
    );

    let fetched = col.fetch(vec!["s1".to_string(), "s2".to_string()]).unwrap();
    assert_eq!(fetched.len(), 2);
    assert_eq!(
        fetched.get("s1").and_then(|d| d.get("kind")),
        Some(&Value::String("session".to_string()))
    );
    assert_eq!(
        fetched.get("s2").and_then(|d| d.get("counter")),
        Some(&Value::I64(2))
    );

    let sql = col
        .query_sql("SELECT kind, counter FROM state WHERE counter = 2 LIMIT 10")
        .unwrap();
    assert_eq!(sql.len(), 1);
    assert_eq!(sql[0].pk, "s2");
    assert_eq!(
        sql[0].get("kind"),
        Some(&Value::String("session".to_string()))
    );
    assert_eq!(sql[0].get("counter"), Some(&Value::I64(2)));

    col.flush().unwrap();
    drop(col);

    let reopened = Collection::open(&path, CollectionOptions::default()).unwrap();
    let fetched = reopened.fetch(vec!["s2".to_string()]).unwrap();
    assert_eq!(fetched.len(), 1);
    assert_eq!(
        fetched.get("s2").and_then(|d| d.get("active")),
        Some(&Value::Bool(false))
    );
}

#[test]
fn test_open_fails_when_delete_bitmap_missing_for_manifest_suffix() {
    let path = temp_dir("open_missing_delete_bitmap");
    let schema = basic_schema(4);

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    col.insert(vec![make_doc("d0", vec![1.0, 0.0, 0.0, 0.0], "d0")])
        .unwrap();

    // delete() snapshots the bitmap: advance the delete suffix to 1 and remove suffix 0.
    let st = col.delete(vec!["d0".to_string()]).unwrap();
    assert!(
        st.iter().all(|s| s.is_ok()),
        "expected delete to succeed, got {st:?}"
    );
    drop(col);

    let delete_1 = path.join("delete_1.bitmap");
    assert!(
        delete_1.exists(),
        "expected delete snapshot at {delete_1:?}"
    );
    std::fs::remove_file(&delete_1).unwrap();

    assert!(Collection::open(&path, CollectionOptions::default()).is_err());
}

#[test]
fn test_open_fails_when_latest_manifest_is_corrupt_even_if_previous_exists() {
    // VersionManager recovery loads only the max-id manifest. If it's corrupt,
    // open fails even if an older manifest exists.
    let path = temp_dir("open_corrupt_manifest_fallback");
    let schema = basic_schema(4);

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let st = col
        .insert(vec![make_doc("d0", vec![1.0, 0.0, 0.0, 0.0], "d0")])
        .unwrap();
    assert!(st.iter().all(|s| s.is_ok()));
    col.flush().unwrap();
    drop(col);

    let manifest_1 = path.join("manifest.1");
    assert!(manifest_1.exists(), "expected manifest.1 to exist");

    // Recreate an older manifest on disk to simulate the crash window where both remain.
    let bytes = std::fs::read(&manifest_1).unwrap();
    let manifest_0 = path.join("manifest.0");
    std::fs::write(&manifest_0, &bytes).unwrap();
    assert!(manifest_0.exists());

    // Corrupt the newest manifest.
    std::fs::write(&manifest_1, b"\x00").unwrap();

    assert!(Collection::open(&path, CollectionOptions::default()).is_err());

    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_open_ignores_leftover_manifest_tmp_file() {
    // Crash window: `manifest.tmp` may exist if the process died between writing tmp and renaming.
    // Open should ignore it and load the latest valid rolling manifest.
    let path = temp_dir("open_manifest_tmp_leftover");
    let schema = basic_schema(4);

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    let st = col
        .insert(vec![make_doc("d0", vec![1.0, 0.0, 0.0, 0.0], "d0")])
        .unwrap();
    assert!(st.iter().all(|s| s.is_ok()));
    col.flush().unwrap();
    drop(col);

    // Leave a junk tmp file behind.
    std::fs::write(path.join("manifest.tmp"), b"junk").unwrap();

    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    assert_eq!(col.stats().unwrap().doc_count, 1);

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_open_ignores_orphan_segment_dirs_not_in_manifest() {
    let path = temp_dir("open_orphan_seg_dir");
    let schema = basic_schema(4);

    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();
    col.insert(vec![make_doc("d0", vec![1.0, 0.0, 0.0, 0.0], "d0")])
        .unwrap();
    col.flush().unwrap();
    drop(col);

    // Create an orphan segment directory that is not referenced by the manifest.
    let orphan = path.join("seg_9999");
    std::fs::create_dir_all(&orphan).unwrap();
    // A junk forward store should not affect open because open enumerates from manifest.
    std::fs::write(orphan.join("forward.arrow"), b"junk").unwrap();

    let col = Collection::open(&path, CollectionOptions::default()).unwrap();
    assert_eq!(col.stats().unwrap().doc_count, 1);

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_collection_path_validation() {
    let base = temp_dir("path_val");
    std::fs::create_dir_all(&base).unwrap();

    let good = base.join("ok_subdir");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4));
    let col = Collection::create_and_open(&good, schema, CollectionOptions::default()).unwrap();
    drop(col);

    #[cfg(windows)]
    {
        let good_backslash = std::path::PathBuf::from(format!("{}\\ok_backslash", base.display()));
        let schema = CollectionSchema::new("test")
            .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4));
        let col =
            Collection::create_and_open(&good_backslash, schema, CollectionOptions::default())
                .unwrap();
        drop(col);
    }

    let bad = base.join("bad dir"); // contains space
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4));
    let err = match Collection::create_and_open(&bad, schema, CollectionOptions::default()) {
        Ok(_) => panic!("expected create_and_open to reject invalid path"),
        Err(e) => e,
    };
    assert!(err.message.contains("collection path"), "err={err:?}");

    std::fs::remove_dir_all(&base).ok();
}

#[test]
fn test_schema_rejects_reserved_system_and_internal_names() {
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(1))
        .with_field(FieldSchema::new(SYS_USER_ID, DataType::Int32));
    let err = schema.validate().unwrap_err();
    assert!(err.message.contains("reserved"), "err={err:?}");

    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(1))
        .with_field(FieldSchema::new(FINCH_IPC_DOC_ID, DataType::Int32));
    let err = schema.validate().unwrap_err();
    assert!(err.message.contains("reserved"), "err={err:?}");

    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(1))
        .with_field(FieldSchema::new(SYS_SCORE, DataType::Float32));
    let err = schema.validate().unwrap_err();
    assert!(err.message.contains("reserved"), "err={err:?}");

    // SQL engine internal/planner columns must also be reserved.
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(1))
        .with_field(FieldSchema::new("_finch_vector", DataType::Bytes));
    let err = schema.validate().unwrap_err();
    assert!(err.message.contains("reserved"), "err={err:?}");

    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(1))
        .with_field(FieldSchema::new("_finch_group_id", DataType::String));
    let err = schema.validate().unwrap_err();
    assert!(err.message.contains("reserved"), "err={err:?}");
}

use super::*;

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
        "optimize() should not rotate/reset delete tombstones"
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
        "optimize should not rotate/reset delete tombstones"
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

#[test]
fn test_optimize_rebuild_threshold_is_strictly_greater_than_0_3() {
    use finch_db::version::VersionManager;

    // Case 1: exactly 30% deleted -> should NOT rebuild (no physical drop).
    {
        let path = temp_dir("optimize_rebuild_threshold_boundary_no_rebuild");
        let schema = CollectionSchema::new("test")
            .with_field(FieldSchema::new("id", DataType::Int64).not_null())
            .with_field(
                FieldSchema::new("emb", DataType::VectorFp32)
                    .with_dimension(4)
                    .with_index(IndexParams::Flat(FlatIndexParams::new(MetricType::L2))),
            );
        let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

        let docs: Vec<Doc> = (0..10u64)
            .map(|i| {
                Doc::new(format!("d{i}"))
                    .set("id", i as i64)
                    .set("emb", vec![i as f32; 4])
            })
            .collect();
        col.insert(docs).unwrap();
        col.flush().unwrap();

        let statuses = col
            .delete(vec!["d0".to_string(), "d1".to_string(), "d2".to_string()])
            .unwrap();
        assert!(statuses.iter().all(|s| s.is_ok()));

        col.optimize(OptimizeOptions::default()).unwrap();

        let vm = VersionManager::load(&path).unwrap();
        let v = vm.current();
        let persisted_count: u64 = v.persisted_segments.iter().map(|s| s.doc_count).sum();
        assert_eq!(
            persisted_count, 10,
            "at exactly 0.3 delete ratio, optimize must not rebuild and should retain tombstoned docs in forward stores"
        );
        assert_eq!(v.persisted_segments.len(), 1);
        assert_eq!(v.persisted_segments[0].segment_id, 0);

        let stats = col.stats().unwrap();
        assert_eq!(
            stats.doc_count, 7,
            "stats should still reflect live docs only"
        );

        drop(col);
        std::fs::remove_dir_all(&path).ok();
    }

    // Case 2: 40% deleted -> must rebuild (physical drop).
    {
        let path = temp_dir("optimize_rebuild_threshold_boundary_rebuild");
        let schema = CollectionSchema::new("test")
            .with_field(FieldSchema::new("id", DataType::Int64).not_null())
            .with_field(
                FieldSchema::new("emb", DataType::VectorFp32)
                    .with_dimension(4)
                    .with_index(IndexParams::Flat(FlatIndexParams::new(MetricType::L2))),
            );
        let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

        let docs: Vec<Doc> = (0..10u64)
            .map(|i| {
                Doc::new(format!("d{i}"))
                    .set("id", i as i64)
                    .set("emb", vec![i as f32; 4])
            })
            .collect();
        col.insert(docs).unwrap();
        col.flush().unwrap();

        let statuses = col
            .delete(vec![
                "d0".to_string(),
                "d1".to_string(),
                "d2".to_string(),
                "d3".to_string(),
            ])
            .unwrap();
        assert!(statuses.iter().all(|s| s.is_ok()));

        col.optimize(OptimizeOptions::default()).unwrap();

        let vm = VersionManager::load(&path).unwrap();
        let v = vm.current();
        let persisted_count: u64 = v.persisted_segments.iter().map(|s| s.doc_count).sum();
        assert_eq!(
            persisted_count, 6,
            "when delete ratio is > 0.3, optimize must rebuild and physically drop deleted docs"
        );
        assert_eq!(v.persisted_segments.len(), 1);
        assert_ne!(
            v.persisted_segments[0].segment_id, 0,
            "rebuild should create a new segment id"
        );
        assert!(
            !path.join("seg_0").exists(),
            "old segment dir should be removed after rebuild"
        );

        let stats = col.stats().unwrap();
        assert_eq!(stats.doc_count, 6);

        drop(col);
        std::fs::remove_dir_all(&path).ok();
    }
}

#[test]
fn test_optimize_removes_orphan_output_segment_dir_before_writing() {
    use finch_db::version::VersionManager;

    let path = temp_dir("optimize_orphan_output_dir");
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("id", DataType::Int64).not_null());

    let opts = CollectionOptions {
        max_buffer_size: 1024, // force multiple persisted segments
        ..CollectionOptions::default()
    };
    let col = Collection::create_and_open(&path, schema, opts).unwrap();

    let docs: Vec<Doc> = (0..200u64)
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

    let vm = VersionManager::load(&path).unwrap();
    let v = vm.current();
    let active_writing = v.writing_segment_id.unwrap_or(0);
    let base_out = v.next_segment_id.max(active_writing.saturating_add(1));

    // Simulate a previous crashed optimize that left behind an orphan output directory.
    let orphan = path.join(format!("seg_{}", base_out));
    std::fs::create_dir_all(&orphan).unwrap();
    std::fs::write(orphan.join("ORPHAN"), b"stale").unwrap();
    assert!(orphan.join("ORPHAN").exists());

    col.optimize(OptimizeOptions {
        target_segment_size: Some(u64::MAX),
        ..Default::default()
    })
    .unwrap();

    // Optimize should have removed the orphan and written a valid segment directory in its place.
    assert!(path.join(format!("seg_{}", base_out)).exists());
    assert!(!path.join(format!("seg_{}/ORPHAN", base_out)).exists());

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

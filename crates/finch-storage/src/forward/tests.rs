use super::*;
use finch_types::{CollectionSchema, DataType, Doc, FieldSchema, MetricType};

#[test]
fn test_forward_ipc_is_chunked_and_mmap_lookup_works_across_batches() {
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("emb16", DataType::VectorFp16).with_dimension(4))
        .with_field(FieldSchema::new("emb8", DataType::VectorInt8).with_dimension(4))
        .with_field(FieldSchema::new("label", DataType::String));

    let mut store = MemoryForwardStore::new(schema);
    let n = 5000usize;
    for i in 0..n {
        let doc_id = i as u64;
        let doc = Doc::new(format!("d{}", i))
            .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
            .set("emb16", vec![0.0f32, 1.0, 0.0, 0.0])
            .set("emb8", vec![0.0f32, 0.0, 1.0, 0.0])
            .set("label", "x");
        store.insert(doc_id, &doc).unwrap();
    }

    let path = std::env::temp_dir().join("finch_forward_ipc_chunk_test.arrow");
    let _ = std::fs::remove_file(&path);
    store.dump_to_ipc(&path).unwrap();

    // Validate the file contains multiple record batches (ceil(5000/4096) = 2).
    let f = std::fs::File::open(&path).unwrap();
    let reader = arrow::ipc::reader::FileReader::try_new(f, None).unwrap();
    let batch_count = reader.count();
    assert_eq!(batch_count, 2);

    // Validate MmapForwardStore can locate doc_ids that span batches.
    let mmap = MmapForwardStore::open(&path).unwrap();
    assert_eq!(mmap.len(), n);

    let ids = vec![0u64, 4095u64, 4096u64, 4999u64];
    let docs = mmap.get_by_doc_ids(&ids).unwrap();
    let got: Vec<String> = docs.into_iter().map(|d| d.unwrap().pk).collect();
    assert_eq!(got, vec!["d0", "d4095", "d4096", "d4999"]);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_forward_parquet_roundtrips_and_lookup_works_across_batches() {
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("label", DataType::String));

    let mut store = MemoryForwardStore::new(schema);
    let n = 5000usize;
    for i in 0..n {
        let doc_id = i as u64;
        let doc = Doc::new(format!("d{}", i))
            .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
            .set("label", "x");
        store.insert(doc_id, &doc).unwrap();
    }

    let path = std::env::temp_dir().join("finch_forward_parquet_roundtrip_test.parquet");
    let _ = std::fs::remove_file(&path);
    store.dump_to_parquet(&path).unwrap();

    let opened = MmapForwardStore::open(&path).unwrap();

    // Locate doc_ids spanning multiple batches.
    assert_eq!(opened.row_index_of_doc_id(0), Some(0));
    assert_eq!(opened.row_index_of_doc_id(4095), Some(4095));
    assert_eq!(opened.row_index_of_doc_id(4096), Some(4096));
    assert_eq!(
        opened.row_index_of_doc_id((n - 1) as u64),
        Some((n - 1) as u64)
    );
    assert_eq!(opened.row_index_of_doc_id(999_999), None);

    let ids = vec![0u64, 4096u64, (n - 1) as u64];
    let pks = opened.get_pks_by_doc_ids(&ids).unwrap();
    assert_eq!(
        pks,
        vec![
            Some("d0".to_string()),
            Some("d4096".to_string()),
            Some(format!("d{}", n - 1))
        ]
    );

    // Distance helper should work for the dense bytes column.
    let d = opened
        .compute_dense_distance_by_doc_ids(
            "emb",
            &[1.0, 0.0, 0.0, 0.0],
            MetricType::InnerProduct,
            &ids,
        )
        .unwrap();
    assert_eq!(d.len(), ids.len());
    assert!(d.iter().all(|x| x.is_some()));

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_parquet_forward_store_eager_mode_survives_file_deletion() {
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4))
        .with_field(FieldSchema::new("label", DataType::String));

    let mut store = MemoryForwardStore::new(schema);
    let n = 5000usize;
    for i in 0..n {
        let doc_id = i as u64;
        let doc = Doc::new(format!("d{}", i))
            .set("emb", vec![1.0f32, 0.0, 0.0, 0.0])
            .set("label", "x");
        store.insert(doc_id, &doc).unwrap();
    }

    let path = std::env::temp_dir().join("finch_forward_parquet_eager_delete_test.parquet");
    let _ = std::fs::remove_file(&path);
    store.dump_to_parquet(&path).unwrap();

    let buffered = MmapForwardStore::open_with_options(
        &path,
        ForwardStoreOpenOptions {
            enable_mmap: true,
            parquet_read_mode: ParquetReadMode::Buffered,
            lazy_ipc: false,
        },
    )
    .unwrap();
    let eager = MmapForwardStore::open_with_options(
        &path,
        ForwardStoreOpenOptions {
            enable_mmap: true,
            parquet_read_mode: ParquetReadMode::Eager,
            lazy_ipc: false,
        },
    )
    .unwrap();

    std::fs::remove_file(&path).unwrap();

    // Buffered mode loads row groups on-demand and should now fail.
    let ids = vec![0u64, 4096u64, (n - 1) as u64];
    assert!(buffered.get_pks_by_doc_ids(&ids).is_err());

    // Eager mode has already materialized all record batches and should still work.
    let pks = eager.get_pks_by_doc_ids(&ids).unwrap();
    assert_eq!(
        pks,
        vec![
            Some("d0".to_string()),
            Some("d4096".to_string()),
            Some(format!("d{}", n - 1)),
        ]
    );
}

#[test]
fn test_mmap_forward_store_rejects_unsorted_doc_ids() {
    // Construct a minimal Arrow IPC file with an out-of-order __doc_id__ column.
    let schema = Arc::new(Schema::new(vec![
        Field::new("__doc_id__", ArrowType::UInt64, false),
        Field::new("__pk__", ArrowType::Utf8, false),
    ]));

    let doc_ids: UInt64Array = vec![2u64, 1u64].into();
    let pks: StringArray = vec![Some("a"), Some("b")].into();
    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(doc_ids), Arc::new(pks)]).unwrap();

    let path = std::env::temp_dir().join("finch_forward_unsorted_doc_ids.arrow");
    let _ = std::fs::remove_file(&path);
    let f = File::create(&path).unwrap();
    let mut writer = FileWriter::try_new(f, &schema).unwrap();
    writer.write(&batch).unwrap();
    writer.finish().unwrap();

    let err = match MmapForwardStore::open(&path) {
        Ok(_) => panic!("expected open to fail for unsorted doc ids"),
        Err(e) => e,
    };
    assert!(err.message.contains("range is invalid"), "err={err:?}");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_mmap_forward_store_compute_dense_distance_mips_l2_matches_reference_formula() {
    let schema = CollectionSchema::new("test")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(2));

    let mut store = MemoryForwardStore::new(schema);
    store
        .insert(0, &Doc::new("d0").set("emb", vec![3.0f32, 4.0]))
        .unwrap();
    store
        .insert(1, &Doc::new("d1").set("emb", vec![1.0f32, 0.0]))
        .unwrap();

    let path = std::env::temp_dir().join("finch_forward_mips_l2.arrow");
    let _ = std::fs::remove_file(&path);
    store.dump_to_ipc(&path).unwrap();

    let mmap = MmapForwardStore::open(&path).unwrap();
    let query = vec![1.0f32, 2.0];
    let ids = vec![0u64, 1u64];
    let dists = mmap
        .compute_dense_distance_by_doc_ids("emb", &query, MetricType::MipsL2, &ids)
        .unwrap();

    let expected0 = {
        let ip = 1.0f32 * 3.0 + 2.0 * 4.0;
        let u2 = 3.0f32 * 3.0 + 4.0 * 4.0;
        let v2 = 1.0f32 * 1.0 + 2.0 * 2.0;
        2.0 - 2.0 * ip / u2.max(v2)
    };
    let expected1 = {
        let ip = 1.0f32 * 1.0 + 2.0 * 0.0;
        let u2 = 1.0f32 * 1.0 + 0.0 * 0.0;
        let v2 = 1.0f32 * 1.0 + 2.0 * 2.0;
        2.0 - 2.0 * ip / u2.max(v2)
    };

    assert!((dists[0].unwrap() - expected0).abs() < 1e-6);
    assert!((dists[1].unwrap() - expected1).abs() < 1e-6);

    let _ = std::fs::remove_file(&path);
}

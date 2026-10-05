use super::*;
use crate::segment::WritingSegment;
use finch_core::algorithm::flat::{FlatBuilder, MemoryStorage};
use finch_types::{
    CollectionSchema, CompareOp, DataType, Doc, FieldSchema, FlatIndexParams, IndexParams,
    InvertIndexParams, Value,
};
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_base_dir(name: &str) -> std::path::PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let mut p = std::env::temp_dir();
    p.push(format!(
        "finch_persisted_{name}_{}_{}",
        std::process::id(),
        ts
    ));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).expect("create temp dir");
    p
}

#[test]
fn test_expr_doc_filter_propagates_doc_fetch_errors() {
    let filter = ExprDocFilter::new(
        FilterExpr::AlwaysTrue,
        Arc::new(|_| Err(Status::io_error("forward fetch failed"))),
        Arc::new(roaring::RoaringTreemap::new()),
        None,
    );

    let err = filter.is_valid(1).unwrap_err();
    assert!(
        err.message.contains("forward fetch failed"),
        "expected fetch error to propagate, got {err:?}"
    );
}

#[test]
fn test_invert_prefilter_plan_and_or_semantics() {
    // Fake lookup: field "a" has doc IDs {1,2}; field "b" has {2,3}.
    let lookup_eq = |field: &str, _value: &Value| -> ZResult<Option<roaring::RoaringTreemap>> {
        let mut bm = roaring::RoaringTreemap::new();
        match field {
            "a" => {
                bm.insert(1);
                bm.insert(2);
                Ok(Some(bm))
            }
            "b" => {
                bm.insert(2);
                bm.insert(3);
                Ok(Some(bm))
            }
            _ => Ok(None),
        }
    };

    // Pull the planner out via the inner function by reconstructing it
    // (kept here as a regression check for correctness of subset/superset rules).
    fn plan_for(
        expr: &FilterExpr,
        lookup_eq: &dyn Fn(&str, &Value) -> ZResult<Option<roaring::RoaringTreemap>>,
    ) -> ZResult<(Option<roaring::RoaringTreemap>, bool)> {
        // Mirror PersistedSegment::invert_prefilter_plan semantics.
        fn inner(
            expr: &FilterExpr,
            lookup_eq: &dyn Fn(&str, &Value) -> ZResult<Option<roaring::RoaringTreemap>>,
        ) -> ZResult<InvertPrefilterPlan> {
            match expr {
                FilterExpr::Compare { field, op, value } => {
                    if *op != CompareOp::Equal {
                        return Ok(InvertPrefilterPlan::none());
                    }
                    let Some(bm) = lookup_eq(field, value)? else {
                        return Ok(InvertPrefilterPlan::none());
                    };
                    Ok(InvertPrefilterPlan {
                        allowlist: Some(bm),
                        exact: true,
                    })
                }
                FilterExpr::And(a, b) => {
                    let pa = inner(a, lookup_eq)?;
                    let pb = inner(b, lookup_eq)?;
                    let allowlist = match (pa.allowlist, pb.allowlist) {
                        (None, None) => None,
                        (Some(x), None) => Some(x),
                        (None, Some(y)) => Some(y),
                        (Some(mut x), Some(y)) => {
                            x &= y;
                            Some(x)
                        }
                    };
                    Ok(InvertPrefilterPlan {
                        allowlist,
                        exact: pa.exact && pb.exact,
                    })
                }
                FilterExpr::Or(a, b) => {
                    let pa = inner(a, lookup_eq)?;
                    let pb = inner(b, lookup_eq)?;
                    match (pa.allowlist, pb.allowlist) {
                        (Some(mut x), Some(y)) => {
                            x |= y;
                            Ok(InvertPrefilterPlan {
                                allowlist: Some(x),
                                exact: pa.exact && pb.exact,
                            })
                        }
                        _ => Ok(InvertPrefilterPlan::none()),
                    }
                }
                _ => Ok(InvertPrefilterPlan::none()),
            }
        }
        let p = inner(expr, lookup_eq)?;
        Ok((p.allowlist, p.exact))
    }

    let expr_and = FilterExpr::And(
        Box::new(FilterExpr::Compare {
            field: "a".to_string(),
            op: CompareOp::Equal,
            value: Value::I32(1),
        }),
        Box::new(FilterExpr::Compare {
            field: "b".to_string(),
            op: CompareOp::Equal,
            value: Value::I32(1),
        }),
    );
    let (bm_and, exact_and) = plan_for(&expr_and, &lookup_eq).unwrap();
    assert!(exact_and);
    let bm_and = bm_and.unwrap();
    assert!(bm_and.contains(2));
    assert!(!bm_and.contains(1));
    assert!(!bm_and.contains(3));

    let expr_or = FilterExpr::Or(
        Box::new(FilterExpr::Compare {
            field: "a".to_string(),
            op: CompareOp::Equal,
            value: Value::I32(1),
        }),
        Box::new(FilterExpr::Compare {
            field: "b".to_string(),
            op: CompareOp::Equal,
            value: Value::I32(1),
        }),
    );
    let (bm_or, exact_or) = plan_for(&expr_or, &lookup_eq).unwrap();
    assert!(exact_or);
    let bm_or = bm_or.unwrap();
    assert!(bm_or.contains(1));
    assert!(bm_or.contains(2));
    assert!(bm_or.contains(3));
}

#[test]
fn test_invert_prefilter_plan_ratio_uses_doc_count_and_array_length_is_unbounded() {
    let cfg = crate::config::global_config();
    if cfg.invert_to_forward_scan_ratio == 0.0 {
        // Ratio=0 degenerates into "always forward"; this test asserts the
        // doc_count-vs-span boundary behavior and is meaningless under 0.
        return;
    }

    let base = temp_base_dir("ratio_doc_count");

    let inv = InvertIndexParams {
        enable_range_optimization: true,
        enable_extended_wildcard: false,
    };
    let schema = CollectionSchema::new("test")
        .with_field(
            FieldSchema::new("x", DataType::Int64).with_index(IndexParams::Invert(inv.clone())),
        )
        .with_field(
            FieldSchema::new("arr", DataType::ArrayInt32).with_index(IndexParams::Invert(inv)),
        );

    // Insert a small number of docs with a deliberately sparse doc_id span.
    // We choose the max doc_id span so that (span * ratio) comfortably exceeds
    // the number of matching docs, ensuring the span-based heuristic would
    // keep the inverted allowlist.
    let doc_count: u64 = 10;
    let hits: u64 = doc_count;
    let ratio = cfg.invert_to_forward_scan_ratio as f64;
    let span_docs = (((hits + 1) as f64) / ratio).ceil() as u64 + 100;
    let sparse_max_doc_id = span_docs.saturating_sub(1);

    let mut seg = WritingSegment::new_with_invert(1, schema, &base).unwrap();
    for i in 0..(doc_count - 1) {
        seg.insert(
            i,
            Doc::new(format!("pk{i}"))
                .set("x", i as i64)
                .set("arr", Value::ArrayI32(vec![0i32; (i as usize) % 3])),
        )
        .unwrap();
    }
    seg.insert(
        sparse_max_doc_id,
        Doc::new("pk_last")
            .set("x", (doc_count - 1) as i64)
            .set("arr", Value::ArrayI32(vec![1i32, 2, 3])),
    )
    .unwrap();

    let meta = seg.dump(&base, finch_types::FileFormat::ArrowIpc).unwrap();
    drop(seg); // close redb handles before opening persisted indexes

    assert_eq!(meta.doc_count, doc_count);
    assert_eq!(meta.min_doc_id, 0);
    assert_eq!(meta.max_doc_id, sparse_max_doc_id);

    let persisted = PersistedSegment::open(&meta).unwrap();

    // Scalar range: doc_count-based ratio rule should skip building an allowlist
    // for a large hit set (falls back to forward scan).
    let plan_x = persisted
        .invert_prefilter_plan(&FilterExpr::Compare {
            field: "x".to_string(),
            op: CompareOp::GreaterEqual,
            value: Value::I64(0),
        })
        .unwrap();
    assert!(
        plan_x.allowlist.is_none(),
        "expected forward-scan fallback for range prefilter"
    );
    assert!(!plan_x.exact);

    // array_length(...) comparisons should not be subject to the ratio bound.
    let plan_arr = persisted
        .invert_prefilter_plan(&FilterExpr::ArrayLengthCompare {
            field: "arr".to_string(),
            op: CompareOp::GreaterEqual,
            len: 0,
        })
        .unwrap();
    assert!(plan_arr.exact);
    let al = plan_arr
        .allowlist
        .expect("expected unbounded allowlist for array_length");
    assert_eq!(al.len(), doc_count);

    drop(persisted);
    std::fs::remove_dir_all(&base).ok();
}

#[test]
fn test_brute_force_by_keys_threshold_uses_doc_count_not_doc_id_span() {
    let cfg = crate::config::global_config();
    if cfg.brute_force_by_keys_ratio == 0.0 {
        // Ratio=0 disables bf-by-keys, so doc_count vs span can't be observed.
        return;
    }

    let base = temp_base_dir("bf_by_keys_doc_count");

    let schema = CollectionSchema::new("test")
        .with_field(
            FieldSchema::new("x", DataType::Int64)
                .with_index(IndexParams::Invert(InvertIndexParams::default())),
        )
        .with_field(FieldSchema::new("v", DataType::VectorFp32).with_dimension(1));

    // Sparse span to make (doc_id span * ratio) much larger than (doc_count * ratio).
    let doc_count: u64 = 10;
    let sparse_max_doc_id: u64 = 1000;

    let mut seg = WritingSegment::new_with_invert(1, schema, &base).unwrap();
    for i in 0..(doc_count - 1) {
        seg.insert(
            i,
            Doc::new(format!("pk{i}"))
                .set("x", i as i64)
                .set("v", vec![i as f32]),
        )
        .unwrap();
    }
    seg.insert(
        sparse_max_doc_id,
        Doc::new("pk_last")
            .set("x", (doc_count - 1) as i64)
            .set("v", vec![(doc_count - 1) as f32]),
    )
    .unwrap();

    let meta = seg.dump(&base, finch_types::FileFormat::ArrowIpc).unwrap();
    drop(seg);

    let persisted = PersistedSegment::open(&meta).unwrap();

    // Attach a FLAT index with dim=1, then issue a query with dim=2.
    // If bf-by-keys triggers, we return `Ok([])` (forward distance returns None on dim mismatch).
    // If bf-by-keys does NOT trigger, the index search path is taken and returns an error.
    let params = FlatIndexParams::new(MetricType::L2);
    let builder = FlatBuilder::new(1, params.clone());
    let mut storage = MemoryStorage::new();
    builder.dump(&mut storage).unwrap();
    let searcher = FlatSearcher::load(&storage, &params).unwrap();
    persisted.add_vector_index("v".to_string(), VectorIndex::Flat(searcher));

    // Allowlist size=5 is:
    // - <= (doc_id span * ratio) for any reasonable span (old, buggy behavior => bf-by-keys triggers)
    // - > (doc_count * ratio) with doc_count=10 and default ratio=0.1 (new behavior => no bf-by-keys)
    let filter = FilterExpr::InList {
        field: "x".to_string(),
        values: vec![
            Value::I64(0),
            Value::I64(1),
            Value::I64(2),
            Value::I64(3),
            Value::I64(4),
        ],
        negated: false,
    };

    let res = persisted.search_vectors(
        "v",
        &[0.0, 1.0], // dim mismatch vs index dim=1
        AnnSearch {
            topk: 3,
            index_params: IndexQueryParams::default(),
            force_linear: false,
            delete_bitmap: Arc::new(roaring::RoaringTreemap::new()),
            filter_expr: Some(&filter),
        },
        MetricType::L2,
    );
    assert!(
        res.is_err(),
        "expected index-search path (no bf-by-keys) when threshold uses doc_count"
    );

    drop(persisted);
    std::fs::remove_dir_all(&base).ok();
}

#[test]
fn test_file_storage_reader_honors_enable_mmap_flag() {
    use finch_core::algorithm::flat::StorageReader as _;

    let dir = std::env::temp_dir().join(format!(
        "finch_test_file_storage_reader_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("META"), b"abc").unwrap();

    let r = FileStorageReader::new(&dir, false);
    let seg = r.read_segment("META").unwrap();
    assert!(
        !matches!(seg, SegmentBytes::Mmap(_)),
        "expected non-mmap segment bytes when enable_mmap=false"
    );
}

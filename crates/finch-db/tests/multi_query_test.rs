use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

use finch_db::Collection;
use finch_types::{
    CollectionOptions, CollectionSchema, DataType, Doc, FieldSchema, FlatIndexParams, IndexParams,
    MetricType, Value, VectorQuery, SYS_GLOBAL_DOC_ID, SYS_LOCAL_ROW_ID, SYS_SCORE, SYS_USER_ID,
};

fn temp_dir(name: &str) -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir =
        std::env::temp_dir().join(format!("finch_test_{}_{}_{}", name, std::process::id(), n));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn two_vec_schema(dim: usize) -> CollectionSchema {
    CollectionSchema::new("test_multi")
        .with_field(
            FieldSchema::new("emb_a", DataType::VectorFp32)
                .with_dimension(dim)
                .with_index(IndexParams::Flat(FlatIndexParams::new(MetricType::L2))),
        )
        .with_field(
            FieldSchema::new("emb_b", DataType::VectorFp32)
                .with_dimension(dim)
                .with_index(IndexParams::Flat(FlatIndexParams::new(MetricType::L2))),
        )
        .with_field(FieldSchema::new("label", DataType::String))
}

fn make_doc(pk: &str, a: Vec<f32>, b: Vec<f32>) -> Doc {
    Doc::new(pk)
        .set("emb_a", a)
        .set("emb_b", b)
        .set("label", pk)
}

#[test]
fn test_query_multi_rrf_prefers_doc_present_in_both_lists_and_injects_system_columns() {
    let path = temp_dir("multi_rrf");
    let schema = two_vec_schema(2);
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    // Construct two ranked lists that only share "c" in the top-3.
    // - Query emb_a: a (best), c, d
    // - Query emb_b: b (best), c, e
    let docs = vec![
        make_doc("a", vec![0.0, 0.0], vec![100.0, 0.0]),
        make_doc("b", vec![100.0, 0.0], vec![0.0, 0.0]),
        make_doc("c", vec![1.0, 0.0], vec![1.0, 0.0]),
        make_doc("d", vec![2.0, 0.0], vec![100.0, 0.0]),
        make_doc("e", vec![100.0, 0.0], vec![2.0, 0.0]),
    ];
    let st = col.insert(docs).unwrap();
    assert!(
        st.iter().all(|s| s.is_ok()),
        "expected insert to succeed, got statuses: {st:?}"
    );

    // Force persisted segment path so row-id resolution uses forward-store lookups.
    col.flush().unwrap();

    let output_fields = vec![
        SYS_USER_ID.to_string(),
        SYS_GLOBAL_DOC_ID.to_string(),
        SYS_LOCAL_ROW_ID.to_string(),
        SYS_SCORE.to_string(),
    ];

    let mut q1 = VectorQuery::new("emb_a", vec![0.0, 0.0], 3);
    q1.output_fields = Some(output_fields.clone());
    let mut q2 = VectorQuery::new("emb_b", vec![0.0, 0.0], 3);
    q2.output_fields = Some(output_fields.clone());

    let out = col.query_multi_rrf(vec![q1, q2], 3, 60).unwrap();
    assert_eq!(out.len(), 3);
    assert_eq!(out[0].pk, "c", "expected consensus doc to rank first");

    // Remaining top-3 should be "a" and "b" (order may tie).
    let tail: HashSet<&str> = out[1..].iter().map(|d| d.pk.as_str()).collect();
    assert_eq!(tail, HashSet::from(["a", "b"]));

    for d in &out {
        // include_vector=false by default; multi-query should strip vector values.
        assert!(
            !d.fields.values().any(|v| v.is_vector()),
            "expected no vector fields in output"
        );

        // System columns should be injected.
        assert_eq!(
            d.fields.get(SYS_USER_ID),
            Some(&Value::String(d.pk.clone()))
        );
        let Some(Value::U64(global)) = d.fields.get(SYS_GLOBAL_DOC_ID) else {
            panic!("missing or wrong type for SYS_GLOBAL_DOC_ID");
        };
        assert_eq!(*global, d.doc_id, "global_doc_id must match doc_id member");

        assert!(
            matches!(d.fields.get(SYS_LOCAL_ROW_ID), Some(Value::U64(_))),
            "expected SYS_LOCAL_ROW_ID to be present"
        );
        assert!(
            matches!(d.fields.get(SYS_SCORE), Some(Value::F32(_))),
            "expected SYS_SCORE to be present"
        );
    }
}

#[test]
fn test_query_multi_weighted_respects_weights_and_hides_doc_id_by_default() {
    let path = temp_dir("multi_weighted");
    let schema = two_vec_schema(2);
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    let docs = vec![
        make_doc("a", vec![0.0, 0.0], vec![100.0, 0.0]),
        make_doc("b", vec![100.0, 0.0], vec![0.0, 0.0]),
        make_doc("c", vec![1.0, 0.0], vec![1.0, 0.0]),
        make_doc("d", vec![2.0, 0.0], vec![100.0, 0.0]),
        make_doc("e", vec![100.0, 0.0], vec![2.0, 0.0]),
    ];
    let st = col.insert(docs).unwrap();
    assert!(
        st.iter().all(|s| s.is_ok()),
        "expected insert to succeed, got statuses: {st:?}"
    );
    col.flush().unwrap();

    // Favor emb_a heavily: "a" should beat "c" and "b".
    let weights = HashMap::from([("emb_a".to_string(), 2.0f32), ("emb_b".to_string(), 1.0f32)]);

    let mut q1 = VectorQuery::new("emb_a", vec![0.0, 0.0], 3);
    q1.output_fields = Some(vec!["label".to_string()]);
    let mut q2 = VectorQuery::new("emb_b", vec![0.0, 0.0], 3);
    q2.output_fields = Some(vec!["label".to_string()]);

    let out = col
        .query_multi_weighted(vec![q1, q2], 3, MetricType::L2, weights)
        .unwrap();
    assert_eq!(out.len(), 3);

    let pks: Vec<&str> = out.iter().map(|d| d.pk.as_str()).collect();
    assert_eq!(pks, vec!["a", "c", "b"]);

    for d in &out {
        // include_doc_id defaults to false, and we did not request SYS_GLOBAL_DOC_ID.
        assert_eq!(d.doc_id, 0, "expected doc_id hidden by default");
        assert_eq!(d.fields.get("label"), Some(&Value::String(d.pk.clone())));
    }
}

#[test]
fn test_query_multi_requires_shared_output_context() {
    let path = temp_dir("multi_shared_ctx");
    let schema = two_vec_schema(2);
    let col = Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap();

    let st = col
        .insert(vec![make_doc("a", vec![0.0, 0.0], vec![0.0, 0.0])])
        .unwrap();
    assert!(st.iter().all(|s| s.is_ok()));
    col.flush().unwrap();

    let mut q1 = VectorQuery::new("emb_a", vec![0.0, 0.0], 3);
    q1.output_fields = Some(vec!["label".to_string()]);
    let mut q2 = VectorQuery::new("emb_b", vec![0.0, 0.0], 3);
    q2.output_fields = None; // mismatch

    assert!(col.query_multi_rrf(vec![q1, q2], 3, 60).is_err());
}

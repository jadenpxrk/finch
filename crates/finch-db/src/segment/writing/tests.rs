use super::*;
use finch_types::{
    CollectionSchema, FieldSchema, FlatIndexParams, IndexParams, MetricType, QuantizeType, Value,
};

#[test]
fn test_upsert_updates_in_memory_vector_store() {
    let schema = CollectionSchema::new("c").with_field({
        let mut f = FieldSchema::new("v", DataType::VectorFp32);
        f.dimension = Some(2);
        f.index_params = Some(IndexParams::Flat(FlatIndexParams {
            metric: MetricType::L2,
            quantize: QuantizeType::Undefined,
            column_major: false,
        }));
        f
    });

    let mut seg = WritingSegment::new(0, schema).unwrap();

    let mut d1 = Doc::new("a");
    d1.fields
        .insert("v".to_string(), Value::VecF32(vec![1.0, 0.0]));
    seg.insert(1, d1).unwrap();

    // Upsert with a different vector should replace (not duplicate) the in-memory entry.
    let mut d2 = Doc::new("a");
    d2.fields
        .insert("v".to_string(), Value::VecF32(vec![0.0, 1.0]));
    let old = seg.get_doc(1);
    seg.upsert(1, d2, old).unwrap();

    let deleted = roaring::RoaringTreemap::new();
    let res = seg
        .search_dense_vectors("v", &[0.0, 1.0], 10, &deleted, None)
        .unwrap();
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].0, 1);
    assert!(res[0].1 <= 1e-6);

    // Upsert removing the vector should remove it from the in-memory store.
    let d3 = Doc::new("a");
    let old = seg.get_doc(1);
    seg.upsert(1, d3, old).unwrap();
    let res = seg
        .search_dense_vectors("v", &[0.0, 1.0], 10, &deleted, None)
        .unwrap();
    assert!(res.is_empty());
}

use super::*;

#[cfg(test)]
mod builder_pruning_tests {
    use super::*;
    use finch_types::{HnswIndexParams, MetricType, QuantizeType};

    fn sv(indices: &[u32], values: &[f32]) -> SparseVector {
        SparseVector::new(indices.to_vec(), values.to_vec())
    }

    #[test]
    fn backlink_inserts_when_slot_available_even_if_prune_would_reject() {
        // Build a tiny, deterministic state:
        // - backlink insertion does NOT prune unless the neighbor list is full.
        //
        // We construct a case where the diversity heuristic would reject `C`
        // (because C is closer to A than to B), but B still has empty slots.
        let params = HnswIndexParams {
            m: 2,
            ef_construction: 8,
            scaling_factor: 50,
            metric: MetricType::InnerProduct,
            quantize: QuantizeType::Undefined,
            build_concurrency: None,
            build_tuning: Default::default(),
        };
        let mut b = HnswSparseBuilder::new(params);

        // Nodes:
        // A: idx0 val1.0
        // B: idx0 val0.5
        // C: idx0 val0.8
        //
        // Distances (neg IP):
        // dist(B,A) = -0.5 (A is close to B)
        // dist(B,C) = -0.4
        // dist(C,A) = -0.8  => C would be pruned if we ran the heuristic.
        b.keys = vec![1, 2, 3];
        b.vectors = vec![sv(&[0], &[1.0]), sv(&[0], &[0.5]), sv(&[0], &[0.8])];
        b.levels = vec![0, 0, 0];
        b.entry_point = 0;
        b.max_level = 0;
        let l0_m = b.params.m * 2;
        b.l0_neighbors = vec![u32::MAX; b.keys.len() * l0_m];

        // Set B's neighbors to include A, leaving free slots.
        b.set_neighbors(1, 0, &[0]);

        let mut scratch = BuilderScratch::default();
        b.add_backlink(1, 0, 2, l0_m, &mut scratch);

        let slot = b.get_neighbors(1, 0);
        assert!(
            slot.contains(&2),
            "expected backlink insertion to include C when slots are free"
        );
        assert!(
            slot.contains(&0),
            "expected original neighbor A to remain present"
        );
    }

    #[test]
    fn pruning_rejects_ties() {
        // Construct a tie case where the candidate is exactly as close to a
        // selected neighbor as it is to the query. Ties are rejected (<=).
        let params = HnswIndexParams {
            m: 1,
            ef_construction: 8,
            scaling_factor: 50,
            metric: MetricType::InnerProduct,
            quantize: QuantizeType::Undefined,
            build_concurrency: None,
            build_tuning: Default::default(),
        };
        let mut b = HnswSparseBuilder::new(params);
        b.vectors = vec![
            sv(&[0], &[1.0]), // Q/S
            sv(&[0], &[1.0]), // identical to Q/S
            sv(&[0], &[0.3]), // C
        ];

        // Candidate set sorted by dist_to_query (neg IP) w.r.t query = vector[0]:
        // S (id=1) first, then C (id=2).
        let candidates = vec![
            (sparse_neg_ip(&b.vectors[0], &b.vectors[1]), 1),
            (sparse_neg_ip(&b.vectors[0], &b.vectors[2]), 2),
        ];

        // Force pruning path by setting prune_cnt < candidates.len().
        let mut out = Vec::new();
        b.select_neighbors_heuristic(&candidates, 2, 1, &mut out);

        assert_eq!(
            out,
            vec![1],
            "expected tie to be rejected; only the first neighbor should remain"
        );
    }
}

#[cfg(test)]
mod build_dump_load_tests {
    use super::*;
    use crate::algorithm::flat::MemoryStorage;
    use finch_types::MetricType;

    fn make_unit_norm_sparse_vec(seed: u64) -> SparseVector {
        let indices: Vec<u32> = (0..32).collect();
        let mut values: Vec<f32> = Vec::with_capacity(indices.len());
        for j in 0..indices.len() {
            let x = (seed as u32)
                .wrapping_mul(1_315_423_911)
                .wrapping_add((j as u32).wrapping_mul(2_654_435_761));
            let v = (x % 1000) as f32 / 1000.0 - 0.5;
            values.push(v);
        }
        let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        for v in &mut values {
            *v /= norm;
        }
        SparseVector::new(indices, values)
    }

    #[test]
    fn test_hnsw_sparse_build_dump_load_search() {
        let params = HnswIndexParams {
            m: 16,
            ef_construction: 128,
            // scaling_factor=2 => the most upper levels random_level allows.
            scaling_factor: 2,
            metric: MetricType::InnerProduct,
            quantize: QuantizeType::Undefined,
            build_concurrency: None,
            build_tuning: Default::default(),
        };

        let mut builder = HnswSparseBuilder::new(params.clone());
        for i in 0..64u64 {
            // Deterministic unit-norm vectors => self inner-product is exactly 1.0 and is the unique best match.
            builder.add(i, make_unit_norm_sparse_vec(i)).unwrap();
        }

        let mut stg = MemoryStorage::new();
        builder.dump(&mut stg).unwrap();

        let searcher = HnswSparseSearcher::load_with_params(&stg, &params).unwrap();
        let q = make_unit_norm_sparse_vec(10);
        let out = searcher.search(&q, 5, 256, None).unwrap();
        assert!(!out.is_empty());
        assert_eq!(out[0].0, 10);
    }

    #[test]
    fn m_of_one_does_not_put_every_node_on_the_top_level() {
        let params = HnswIndexParams::new(MetricType::InnerProduct).with_m(1);
        let mut builder = HnswSparseBuilder::new(params);
        for i in 0..200u64 {
            builder.add(i, make_unit_norm_sparse_vec(i)).unwrap();
        }
        let mut stg = MemoryStorage::new();
        builder.dump(&mut stg).unwrap();
        let levels = stg.read_segment(SEG_LEVELS).unwrap().as_slice().to_vec();
        let levels: Vec<u32> = levels
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect();
        assert_eq!(levels.len(), 200);
        let at_level_zero = levels.iter().filter(|&&level| level == 0).count();
        assert!(
            at_level_zero > 50,
            "only {at_level_zero} of 200 nodes stay on level 0"
        );
        assert!(levels.iter().all(|&level| level < 32));
    }
}

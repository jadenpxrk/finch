use super::*;
use crate::algorithm::codec::{u32s_to_le, u64s_to_le};

/// `HNSW_UPPER_NEIGHBORS` (`[count][ids...]` runs) and its fixed-slot `HNSW_UPPER_INDEX`.
struct UpperSegments {
    neighbors: Vec<u8>,
    index: Vec<u8>,
}

impl HnswBuilder {
    pub fn dump(&self, storage: &mut dyn StorageWriter) -> ZResult<()> {
        let n = self.keys.len();
        storage.write_segment(SEG_HEADER, &self.encode_header())?;
        storage.write_segment(SEG_KEYS, &u64s_to_le(&self.keys))?;

        // Write vectors (either raw f32 or quantized bytes)
        let raw_vec_bytes = bytes_per_vector(self.params.quantize, self.dim);
        let vec_stride_bytes = aligned_vector_stride_bytes(raw_vec_bytes);
        storage.write_segment(SEG_VECTORS, &self.encode_vectors(vec_stride_bytes))?;
        storage.write_segment(SEG_LEVELS, &u32s_to_le(&self.levels))?;

        // Write L0 neighbors (n × 2m u32 entries)
        let l0_len = self.l0_neighbors.len().min(n * self.params.m * 2);
        storage.write_segment(SEG_L0_NEIGHBORS, &u32s_to_le(&self.l0_neighbors[..l0_len]))?;

        let upper = self.encode_upper();
        storage.write_segment(SEG_UPPER_NEIGHBORS, &upper.neighbors)?;
        storage.write_segment(SEG_UPPER_INDEX, &upper.index)?;
        storage.write_segment(SEG_META, &self.encode_meta(vec_stride_bytes))?;

        Ok(())
    }

    /// Quantizes each vector and zero-pads it to `vec_stride_bytes`.
    fn encode_vectors(&self, vec_stride_bytes: usize) -> Vec<u8> {
        let n = self.keys.len();
        let raw_vec_bytes = bytes_per_vector(self.params.quantize, self.dim);
        let mut vec_buf: Vec<u8> = Vec::with_capacity(n * vec_stride_bytes);
        for i in 0..n {
            let start = i * self.vec_stride_floats;
            let v = &self.vectors[start..start + self.dim];
            quantize_append(self.params.quantize, v, &mut vec_buf);
            if vec_stride_bytes > raw_vec_bytes {
                vec_buf.resize(vec_buf.len() + (vec_stride_bytes - raw_vec_bytes), 0u8);
            }
        }
        vec_buf
    }

    fn encode_upper(&self) -> UpperSegments {
        let m = self.params.m;
        let mut upper_buf: Vec<u8> = Vec::new();
        let mut upper_idx_buf: Vec<u8> = Vec::new();
        upper_buf.extend_from_slice(&(self.max_level as u32).to_le_bytes());
        upper_idx_buf.extend_from_slice(&(self.max_level as u32).to_le_bytes());
        for level in &self.upper_neighbors {
            upper_buf.extend_from_slice(&(level.offsets.len() as u32).to_le_bytes());
            upper_idx_buf.extend_from_slice(&(level.offsets.len() as u32).to_le_bytes());
            upper_idx_buf.extend_from_slice(&(m as u32).to_le_bytes());
            for &off in &level.offsets {
                if off == u32::MAX {
                    upper_buf.extend_from_slice(&0u32.to_le_bytes());
                    upper_idx_buf.extend_from_slice(&u32::MAX.to_le_bytes());
                    continue;
                }
                upper_buf.extend_from_slice(&(m as u32).to_le_bytes());
                upper_idx_buf.extend_from_slice(&(upper_buf.len() as u32).to_le_bytes());
                let off = off as usize;
                for &nb in level.neighbors[off..off + m].iter() {
                    upper_buf.extend_from_slice(&nb.to_le_bytes());
                }
            }
        }
        UpperSegments {
            neighbors: upper_buf,
            index: upper_idx_buf,
        }
    }

    /// Magic, format version, metric, then the build heuristics that shaped the graph.
    fn encode_header(&self) -> Vec<u8> {
        let t = &self.params.build_tuning;
        let flags = u32::from(t.simple_neighbor_select)
            | (u32::from(t.keep_pruned_connections) << 1)
            | (u32::from(t.l0_repair) << 2)
            | (u32::from(t.qdrant_backlink) << 3);
        let refine_cap = t.l0_refine_candidate_cap.map_or(0, |cap| cap as u32);
        let mut buf = HEADER_MAGIC.to_vec();
        for word in [
            HEADER_VERSION,
            self.params.metric as u32,
            self.heuristic_dim as u32,
            flags,
            refine_cap,
            t.prune_alpha.to_bits(),
        ] {
            buf.extend_from_slice(&word.to_le_bytes());
        }
        buf
    }

    fn encode_meta(&self, vec_stride_bytes: usize) -> Vec<u8> {
        let mut meta_buf = Vec::new();
        meta_buf.extend_from_slice(&(self.keys.len() as u64).to_le_bytes());
        meta_buf.extend_from_slice(&(self.dim as u64).to_le_bytes());
        meta_buf.extend_from_slice(&self.entry_point.to_le_bytes());
        meta_buf.extend_from_slice(&(self.max_level as u32).to_le_bytes());
        meta_buf.extend_from_slice(&(self.params.m as u32).to_le_bytes());
        meta_buf.extend_from_slice(&(self.params.ef_construction as u32).to_le_bytes());
        meta_buf.extend_from_slice(&(self.params.quantize as u32).to_le_bytes());
        meta_buf.extend_from_slice(&(vec_stride_bytes as u32).to_le_bytes());
        meta_buf.extend_from_slice(&u32::from(self.normalize_cosine).to_le_bytes());
        meta_buf
    }
}

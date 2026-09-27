//! Writing (mutable) segment: accepts inserts until full, then dumps to disk

use super::prefilter::{max_range_hits, plan_for, take_live_ids, InvertPrefilterPlan};
use crate::invert::InvertIndex;
use crate::sqlengine::executor::DocFilterEvaluator;
use crate::sqlengine::parser::FilterExpr;
use finch_core::algorithm::flat_sparse::SparseVector;
use finch_core::algorithm::TopkHeap;
use finch_core::metric::{make_metric, Metric};
use finch_core::quantizer::{bytes_per_vector, distance_to_quantized, quantize_append};
use finch_storage::MemoryForwardStore;
use finch_types::{
    CollectionSchema, DataType, Doc, InvertIndexParams, MetricType, QuantizeType, Status, ZResult,
};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Metadata describing an inverted index persisted in a segment directory.
#[derive(Debug, Clone)]
pub struct InvertIndexMeta {
    pub path: PathBuf,
    pub data_type: DataType,
    pub params: InvertIndexParams,
}

/// Metadata describing a dumped segment on disk
#[derive(Debug, Clone)]
pub struct WrittenSegmentMeta {
    pub segment_id: u32,
    pub min_doc_id: u64,
    pub max_doc_id: u64,
    pub doc_count: u64,
    pub forward_path: PathBuf,
    pub invert_paths: HashMap<String, InvertIndexMeta>,
}

/// The in-memory write buffer for a single segment
pub struct WritingSegment {
    pub id: u32,
    schema: CollectionSchema,
    pub forward_store: MemoryForwardStore,
    invert_indexes: HashMap<String, InvertIndex>,
    dense_indexes: HashMap<String, DenseMemIndex>,
    binary32_indexes: HashMap<String, Binary32MemIndex>,
    binary64_indexes: HashMap<String, Binary64MemIndex>,
    sparse_indexes: HashMap<String, SparseMemIndex>,
    pub doc_count: AtomicU64,
    pub min_doc_id: AtomicU64,
    pub max_doc_id: AtomicU64,
    // All docs in insertion order for filter/scan
    docs: parking_lot::RwLock<Vec<(u64, Doc)>>,
    doc_positions: parking_lot::RwLock<HashMap<u64, usize>>,
}

mod filter;
mod indexes;
mod mutation;
mod persistence;
mod search;
mod setup;

use indexes::*;

fn sync_dir_best_effort(path: &Path) {
    let _ = std::fs::File::open(path).and_then(|d| d.sync_all());
}

#[cfg(test)]
mod tests;

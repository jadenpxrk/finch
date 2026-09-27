//! Neighbor selection heuristics shared by the sequential and parallel builders.

use super::*;

pub(super) struct VectorView<'a> {
    pub(super) metric_type: MetricType,
    pub(super) vectors: &'a [f32],
    pub(super) vec_stride_floats: usize,
    pub(super) dim: usize,
    pub(super) heuristic_dim: usize,
}

impl<'a> VectorView<'a> {
    pub(super) fn new(
        metric_type: MetricType,
        vectors: &'a [f32],
        vec_stride_floats: usize,
        dim: usize,
        heuristic_dim: usize,
    ) -> Self {
        VectorView {
            metric_type,
            vectors,
            vec_stride_floats,
            dim,
            heuristic_dim,
        }
    }
}

/// Scratch buffers for pruning a full neighbor slot.
pub(super) struct BacklinkBuffers<'a> {
    pub(super) scored: &'a mut Vec<(f32, u32)>,
    pub(super) selected: &'a mut Vec<u32>,
    pub(super) selected_is_new: &'a mut Vec<bool>,
}

/// Takes candidates in order, skipping empty slots, until `out` holds `m`.
fn push_first_valid(candidates: &[(f32, u32)], m: usize, out: &mut Vec<u32>) {
    for &(_, cand) in candidates.iter() {
        if cand == u32::MAX {
            continue;
        }
        out.push(cand);
        if out.len() >= m {
            break;
        }
    }
}

/// keepPrunedConnections: the heuristic often leaves slots free in high dimensions.
fn backfill_pruned(pruned: &[u32], m: usize, out: &mut Vec<u32>) {
    if !keep_pruned_connections_enabled() || out.len() >= m {
        return;
    }
    for &cand in pruned {
        out.push(cand);
        if out.len() >= m {
            break;
        }
    }
}

/// Candidate selection for backlink pruning.
///
/// `candidates` must be sorted by ascending precomputed distance-to-query,
/// which is stored in `candidates[i].0` (computed using `heuristic_dim`).
pub(super) fn select_neighbors_precomputed_into(
    view: &VectorView<'_>,
    candidates: &[(f32, u32)],
    m: usize,
    out: &mut Vec<u32>,
) {
    let VectorView {
        metric_type,
        vectors,
        vec_stride_floats,
        dim,
        heuristic_dim,
    } = *view;
    out.reserve(m);
    if simple_neighbor_select_enabled() {
        push_first_valid(candidates, m, out);
        return;
    }
    // Pruned candidates stay in distance order for the backfill.
    let mut pruned: Vec<u32> = Vec::new();
    for &(dist_to_query, cand) in candidates.iter() {
        if cand == u32::MAX {
            continue;
        }
        let cand_vec = vec_slice_raw(vectors, vec_stride_floats, dim, cand);
        let mut ok = true;
        for &s in out.iter() {
            let s_vec = vec_slice_raw(vectors, vec_stride_floats, dim, s);
            let d = hnsw_heuristic_distance(metric_type, heuristic_dim, dim, cand_vec, s_vec);
            if should_prune_neighbor(d, dist_to_query) {
                ok = false;
                break;
            }
        }
        if ok {
            out.push(cand);
            if out.len() >= m {
                break;
            }
        } else {
            pruned.push(cand);
        }
    }
    backfill_pruned(&pruned, m, out);
}

/// Like `select_neighbors_precomputed_into`, scoring against `query` with the same dimension prefix.
pub(super) fn select_neighbors_query_into(
    view: &VectorView<'_>,
    candidates: &[(f32, u32)],
    m: usize,
    query: &[f32],
    out: &mut Vec<u32>,
) {
    let VectorView {
        metric_type,
        vectors,
        vec_stride_floats,
        dim,
        heuristic_dim,
    } = *view;
    out.reserve(m);
    if simple_neighbor_select_enabled() {
        push_first_valid(candidates, m, out);
        return;
    }
    let mut pruned: Vec<u32> = Vec::new();
    for &(_, cand) in candidates.iter() {
        if cand == u32::MAX {
            continue;
        }
        let cand_vec = vec_slice_raw(vectors, vec_stride_floats, dim, cand);
        let dist_to_query =
            hnsw_heuristic_distance(metric_type, heuristic_dim, dim, cand_vec, query);
        let mut ok = true;
        for &s in out.iter() {
            let s_vec = vec_slice_raw(vectors, vec_stride_floats, dim, s);
            let d = hnsw_heuristic_distance(metric_type, heuristic_dim, dim, cand_vec, s_vec);
            if should_prune_neighbor(d, dist_to_query) {
                ok = false;
                break;
            }
        }
        if ok {
            out.push(cand);
            if out.len() >= m {
                break;
            }
        } else {
            pruned.push(cand);
        }
    }
    backfill_pruned(&pruned, m, out);
}

/// Links a full slot keeps once `new_item` arrives; `buf.scored` holds its current links.
pub(super) fn select_backlinks_into(
    view: &VectorView<'_>,
    new_item: (f32, u32),
    max_m: usize,
    buf: BacklinkBuffers<'_>,
) {
    buf.selected.clear();
    if qdrant_backlink_heuristic_enabled() {
        let VectorView {
            metric_type,
            vectors,
            vec_stride_floats,
            dim,
            heuristic_dim,
        } = *view;
        select_backlink_preserving_existing_order(
            buf.scored,
            new_item,
            max_m,
            buf.selected,
            buf.selected_is_new,
            |a, b| {
                hnsw_heuristic_distance(
                    metric_type,
                    heuristic_dim,
                    dim,
                    vec_slice_raw(vectors, vec_stride_floats, dim, a),
                    vec_slice_raw(vectors, vec_stride_floats, dim, b),
                )
            },
        );
    } else {
        buf.scored.push(new_item);
        buf.scored
            .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        select_neighbors_precomputed_into(view, buf.scored, max_m, buf.selected);
    }
}

/// Links picked so far by the incremental backlink heuristic.
struct BacklinkPicks<'a, F> {
    out: &'a mut Vec<u32>,
    out_is_new: &'a mut Vec<bool>,
    distance_between: F,
}

impl<F: FnMut(u32, u32) -> f32> BacklinkPicks<'_, F> {
    #[inline]
    fn try_push(&mut self, cand_dist: f32, cand: u32, cand_is_new: bool) {
        for (idx, &selected) in self.out.iter().enumerate() {
            // Existing links have already passed the heuristic in a previous
            // backlink update. Match Qdrant's incremental path by preserving
            // that old-old relationship and only testing pairs involving the
            // candidate new link.
            if !cand_is_new && !self.out_is_new[idx] {
                continue;
            }
            if should_prune_neighbor((self.distance_between)(cand, selected), cand_dist) {
                return;
            }
        }
        self.out.push(cand);
        self.out_is_new.push(cand_is_new);
    }
}

fn select_backlink_preserving_existing_order<F>(
    existing: &[(f32, u32)],
    new_item: (f32, u32),
    m: usize,
    out: &mut Vec<u32>,
    out_is_new: &mut Vec<bool>,
    distance_between: F,
) where
    F: FnMut(u32, u32) -> f32,
{
    out.clear();
    out_is_new.clear();
    out.reserve(m);
    out_is_new.reserve(m);
    let mut picks = BacklinkPicks {
        out,
        out_is_new,
        distance_between,
    };

    let (new_dist, new_node) = new_item;
    let mut inserted_new = false;
    for &(old_dist, old_node) in existing {
        if !inserted_new && new_dist < old_dist {
            picks.try_push(new_dist, new_node, true);
            inserted_new = true;
            if picks.out.len() >= m {
                return;
            }
        }
        picks.try_push(old_dist, old_node, false);
        if picks.out.len() >= m {
            return;
        }
    }

    if !inserted_new && picks.out.len() < m {
        picks.try_push(new_dist, new_node, true);
    }
}

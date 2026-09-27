use std::sync::Arc;

use finch_types::{Doc, MetricType, VectorQuery, ZResult};

use crate::collection::Collection;
use crate::query_output::{public_score, wants_pk_only_output, QueryOutputSelection};

impl Collection {
    pub(super) fn materialize_vector_query_results(
        &self,
        results: &[(u64, f32)],
        query: &VectorQuery,
        metric: MetricType,
        output_selection: &QueryOutputSelection,
    ) -> ZResult<Vec<Arc<Doc>>> {
        let wants_pk_only = wants_pk_only_output(
            query.include_vector,
            query.include_doc_id,
            query.output_fields.as_deref(),
        );

        let mut output: Vec<Doc> = Vec::with_capacity(results.len());
        if wants_pk_only {
            let doc_ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
            let pks = self.fetch_pks_by_ids(&doc_ids)?;
            for (doc_id, score) in results {
                if let Some(pk) = pks.get(doc_id) {
                    let mut doc = Doc::new(pk.clone());
                    doc.score = public_score(*score, metric);
                    output.push(doc);
                }
            }
        } else {
            let doc_ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
            let mut fetched_docs = self.fetch_by_ids(&doc_ids)?;

            for (doc_id, score) in results {
                if let Some(mut doc) = fetched_docs.remove(doc_id) {
                    doc.score = public_score(*score, metric);
                    output.push(doc);
                }
            }
        }

        let row_locator = if output_selection.wants_row_locator() {
            Some(self.row_locator())
        } else {
            None
        };
        output_selection.finish_docs(
            output,
            query.include_vector,
            query.include_doc_id,
            query.output_fields.as_deref(),
            row_locator.as_ref(),
        )
    }

    pub(super) fn materialize_filter_only_results(
        &self,
        matched_ids: &[u64],
        query: &VectorQuery,
        output_selection: &QueryOutputSelection,
    ) -> ZResult<Vec<Arc<Doc>>> {
        let wants_pk_only = wants_pk_only_output(
            query.include_vector,
            query.include_doc_id,
            query.output_fields.as_deref(),
        );

        let mut output: Vec<Doc> = Vec::with_capacity(matched_ids.len());
        if wants_pk_only {
            let pks = self.fetch_pks_by_ids(matched_ids)?;
            for doc_id in matched_ids {
                if let Some(pk) = pks.get(doc_id) {
                    let mut doc = Doc::new(pk.clone());
                    doc.doc_id = *doc_id;
                    doc.score = 0.0;
                    output.push(doc);
                }
            }
        } else {
            let mut fetched_docs = self.fetch_by_ids(matched_ids)?;
            for doc_id in matched_ids {
                if let Some(mut doc) = fetched_docs.remove(doc_id) {
                    doc.score = 0.0;
                    output.push(doc);
                }
            }
        }

        let row_locator = if output_selection.wants_row_locator() {
            Some(self.row_locator())
        } else {
            None
        };
        output_selection.finish_docs(
            output,
            query.include_vector,
            query.include_doc_id,
            query.output_fields.as_deref(),
            row_locator.as_ref(),
        )
    }

    pub(super) fn materialize_fused_docs(
        &self,
        fused: Vec<(String, f32)>,
        include_vector: bool,
        include_doc_id: bool,
        output_fields: Option<Vec<String>>,
    ) -> ZResult<Vec<Arc<Doc>>> {
        let wants_pk_only =
            wants_pk_only_output(include_vector, include_doc_id, output_fields.as_deref());
        let output_selection =
            QueryOutputSelection::from_output_fields(include_doc_id, output_fields.as_deref());

        if wants_pk_only {
            let mut out: Vec<Arc<Doc>> = Vec::with_capacity(fused.len());
            for (pk, score) in fused {
                let mut doc = Doc::new(pk);
                doc.score = score;
                out.push(Arc::new(doc));
            }
            return Ok(out);
        }

        let pks: Vec<String> = fused.iter().map(|(pk, _)| pk.clone()).collect();
        let fetched = self.fetch(pks)?;

        let row_locator = if output_selection.wants_row_locator() {
            Some(self.row_locator())
        } else {
            None
        };

        let mut docs: Vec<Doc> = Vec::with_capacity(fused.len());
        for (pk, score) in fused {
            let Some(doc_arc) = fetched.get(&pk) else {
                continue;
            };
            let mut doc = (**doc_arc).clone();
            doc.score = score;
            docs.push(doc);
        }

        output_selection.finish_docs(
            docs,
            include_vector,
            include_doc_id,
            output_fields.as_deref(),
            row_locator.as_ref(),
        )
    }
}

use std::sync::Arc;

use finch_types::{Doc, Status, VectorQuery, ZResult};

type ScoredPkList = (String, Vec<(String, f32)>);

#[derive(Clone)]
pub(crate) struct MultiQueryOutput {
    pub(crate) include_vector: bool,
    pub(crate) include_doc_id: bool,
    pub(crate) output_fields: Option<Vec<String>>,
}

pub(crate) struct MultiQuerySetup {
    queries: Vec<VectorQuery>,
    output: MultiQueryOutput,
}

impl MultiQuerySetup {
    pub(crate) fn new(queries: Vec<VectorQuery>) -> ZResult<Self> {
        if queries.is_empty() {
            return Err(Status::invalid_argument("No query to execute"));
        }

        let base_filter = queries[0].filter.clone();
        let output = MultiQueryOutput {
            include_vector: queries[0].include_vector,
            include_doc_id: queries[0].include_doc_id,
            output_fields: normalize_output_fields(queries[0].output_fields.clone()),
        };

        for query in &queries {
            if query.filter != base_filter {
                return Err(Status::invalid_argument(
                    "multi query requires identical filter across VectorQuery items",
                ));
            }
            if query.include_vector != output.include_vector
                || query.include_doc_id != output.include_doc_id
            {
                return Err(Status::invalid_argument(
                    "multi query requires identical include_vector/include_doc_id across VectorQuery items",
                ));
            }
            if normalize_output_fields(query.output_fields.clone()) != output.output_fields {
                return Err(Status::invalid_argument(
                    "multi query requires identical output_fields across VectorQuery items",
                ));
            }
        }

        Ok(Self { queries, output })
    }

    pub(crate) fn output(&self) -> &MultiQueryOutput {
        &self.output
    }

    pub(crate) fn collect_pk_lists<F>(
        self,
        topk: usize,
        mut execute: F,
    ) -> ZResult<Vec<Vec<String>>>
    where
        F: FnMut(VectorQuery) -> ZResult<Vec<Arc<Doc>>>,
    {
        let mut lists: Vec<Vec<String>> = Vec::with_capacity(self.queries.len());
        for query in self.into_pk_only_queries(topk) {
            let docs = execute(query)?;
            lists.push(docs.into_iter().map(|d| d.pk.clone()).collect());
        }

        Ok(lists)
    }

    pub(crate) fn collect_scored_pk_lists<F>(
        self,
        topk: usize,
        mut execute: F,
    ) -> ZResult<Vec<ScoredPkList>>
    where
        F: FnMut(VectorQuery) -> ZResult<Vec<Arc<Doc>>>,
    {
        let mut lists: Vec<ScoredPkList> = Vec::with_capacity(self.queries.len());
        for query in self.into_pk_only_queries(topk) {
            let name = query.field_name.clone();
            let docs = execute(query)?;
            lists.push((
                name,
                docs.into_iter().map(|d| (d.pk.clone(), d.score)).collect(),
            ));
        }

        Ok(lists)
    }

    pub(crate) fn into_pk_only_queries(self, topk: usize) -> Vec<VectorQuery> {
        self.queries
            .into_iter()
            .map(|mut query| {
                query.topk = query.topk.max(topk);
                query.include_vector = false;
                query.include_doc_id = false;
                query.output_fields = Some(Vec::new());
                query
            })
            .collect()
    }
}

fn normalize_output_fields(output_fields: Option<Vec<String>>) -> Option<Vec<String>> {
    match output_fields {
        Some(fields) if fields.iter().any(|field| field.as_str() == "*") => None,
        other => other,
    }
}

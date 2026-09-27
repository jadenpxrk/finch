use std::sync::Arc;

use finch_types::{CollectionSchema, Doc, MetricType, Status, VectorQuery, ZResult};

use crate::output_projection::{validate_and_normalize_output_fields, OutputFieldProjection};
use crate::row_locator::RowLocator;
use crate::system_projection::SystemColumnOutputs;

pub(crate) struct QueryOutputSelection {
    system_outputs: SystemColumnOutputs,
    expose_doc_id_member: bool,
}

impl QueryOutputSelection {
    pub(crate) fn prepare(
        schema: &CollectionSchema,
        query: &mut VectorQuery,
        max_output_fields: usize,
    ) -> ZResult<Self> {
        let selection =
            Self::from_output_fields(query.include_doc_id, query.output_fields.as_deref());
        if selection.needs_doc_id_for_system_output() {
            query.include_doc_id = true;
        }

        if let Some(fields) = query.output_fields.as_ref() {
            if fields.len() > max_output_fields {
                return Err(Status::invalid_argument(format!(
                    "query validate failed: output_fields is too large, max is {}",
                    max_output_fields
                )));
            }
        }

        validate_and_normalize_output_fields(schema, &mut query.output_fields)?;

        Ok(selection)
    }

    pub(crate) fn from_output_fields(
        include_doc_id: bool,
        output_fields: Option<&[String]>,
    ) -> Self {
        let system_outputs = SystemColumnOutputs::from_output_fields(output_fields);
        let wants_global_doc_id = system_outputs.wants_global_doc_id();
        Self {
            system_outputs,
            expose_doc_id_member: include_doc_id || wants_global_doc_id,
        }
    }

    pub(crate) fn wants_row_locator(&self) -> bool {
        self.system_outputs.wants_row_id()
    }

    pub(crate) fn finish_docs(
        &self,
        mut output: Vec<Doc>,
        include_vector: bool,
        include_doc_id: bool,
        output_fields: Option<&[String]>,
        row_locator: Option<&RowLocator>,
    ) -> ZResult<Vec<Arc<Doc>>> {
        let keep_doc_id_for_system_output = include_doc_id || self.needs_doc_id_for_system_output();
        if !keep_doc_id_for_system_output {
            for doc in &mut output {
                doc.doc_id = 0;
            }
        }

        let field_projection = OutputFieldProjection::new(include_vector, output_fields);
        for doc in &mut output {
            field_projection.apply(doc);
        }

        if self.system_outputs.wants_any() {
            for doc in &mut output {
                self.system_outputs.insert_into_doc(doc, row_locator)?;
                if !self.expose_doc_id_member {
                    doc.doc_id = 0;
                }
            }
        }

        Ok(output.into_iter().map(Arc::new).collect())
    }

    fn needs_doc_id_for_system_output(&self) -> bool {
        self.system_outputs.wants_row_id() || self.system_outputs.wants_global_doc_id()
    }
}

#[inline]
pub(crate) fn wants_pk_only_output(
    include_vector: bool,
    include_doc_id: bool,
    output_fields: Option<&[String]>,
) -> bool {
    output_fields.is_some_and(|fields| fields.is_empty()) && !include_vector && !include_doc_id
}

#[inline]
pub(crate) fn public_score(dist: f32, metric: MetricType) -> f32 {
    match metric {
        MetricType::InnerProduct => -dist,
        _ => dist,
    }
}

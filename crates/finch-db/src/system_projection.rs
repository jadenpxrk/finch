use crate::row_locator::RowLocator;
use finch_types::{Doc, Status, Value, ZResult};
use finch_types::{SYS_GLOBAL_DOC_ID, SYS_LOCAL_ROW_ID, SYS_SCORE, SYS_USER_ID};

#[derive(Clone, Debug, Default)]
pub(crate) struct SystemColumnOutputs {
    uid: Vec<String>,
    row_id: Vec<String>,
    global_doc_id: Vec<String>,
    score: Vec<String>,
}

impl SystemColumnOutputs {
    pub(crate) fn from_output_fields(output_fields: Option<&[String]>) -> Self {
        let Some(output_fields) = output_fields else {
            return Self::default();
        };

        let mut outputs = Self::default();
        for field in output_fields {
            if let Some(names) = outputs.output_names_mut(field) {
                names.push(field.clone());
            }
        }
        outputs
    }

    /// Output names for system column `source`; `None` when it is not a system column.
    pub(crate) fn output_names_mut(&mut self, source: &str) -> Option<&mut Vec<String>> {
        match source {
            SYS_USER_ID => Some(&mut self.uid),
            SYS_LOCAL_ROW_ID => Some(&mut self.row_id),
            SYS_GLOBAL_DOC_ID => Some(&mut self.global_doc_id),
            SYS_SCORE => Some(&mut self.score),
            _ => None,
        }
    }

    pub(crate) fn wants_any(&self) -> bool {
        self.wants_uid() || self.wants_row_id() || self.wants_global_doc_id() || self.wants_score()
    }

    pub(crate) fn wants_uid(&self) -> bool {
        !self.uid.is_empty()
    }

    pub(crate) fn wants_row_id(&self) -> bool {
        !self.row_id.is_empty()
    }

    pub(crate) fn wants_global_doc_id(&self) -> bool {
        !self.global_doc_id.is_empty()
    }

    pub(crate) fn wants_score(&self) -> bool {
        !self.score.is_empty()
    }

    pub(crate) fn insert_into_doc(
        &self,
        doc: &mut Doc,
        row_locator: Option<&RowLocator>,
    ) -> ZResult<()> {
        for out_name in &self.uid {
            doc.fields
                .insert(out_name.clone(), Value::String(doc.pk.clone()));
        }
        for out_name in &self.global_doc_id {
            doc.fields.insert(out_name.clone(), Value::U64(doc.doc_id));
        }
        for out_name in &self.score {
            doc.fields.insert(out_name.clone(), Value::F32(doc.score));
        }
        if !self.row_id.is_empty() {
            let row_id = resolve_row_id(row_locator, doc.doc_id)?;
            for out_name in &self.row_id {
                doc.fields.insert(out_name.clone(), Value::U64(row_id));
            }
        }

        Ok(())
    }
}

/// Row id of `doc_id`; an internal error when no segment owns it.
pub(crate) fn resolve_row_id(row_locator: Option<&RowLocator>, doc_id: u64) -> ZResult<u64> {
    row_locator
        .and_then(|locator| locator.row_id(doc_id))
        .ok_or_else(|| Status::internal(format!("failed to resolve segment for doc_id={}", doc_id)))
}

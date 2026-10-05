use std::collections::HashSet;

use finch_types::system_columns::is_system_column;
use finch_types::{CollectionSchema, Doc, Status, ZResult, SYS_SCORE};

pub(crate) struct OutputFieldProjection<'a> {
    include_vector: bool,
    output_fields: Option<&'a [String]>,
    scalar_fields: HashSet<&'a str>,
}

impl<'a> OutputFieldProjection<'a> {
    pub(crate) fn new(include_vector: bool, output_fields: Option<&'a [String]>) -> Self {
        let scalar_fields = output_fields
            .filter(|fields| !fields.is_empty())
            .map(|fields| fields.iter().map(|field| field.as_str()).collect())
            .unwrap_or_default();

        Self {
            include_vector,
            output_fields,
            scalar_fields,
        }
    }

    pub(crate) fn apply(&self, doc: &mut Doc) {
        if !self.include_vector {
            doc.fields.retain(|_, value| !value.is_vector());
        }

        match self.output_fields {
            None => {}
            Some([]) => {
                doc.fields.retain(|_, value| value.is_vector());
            }
            Some(_) => {
                doc.fields.retain(|field, value| {
                    value.is_vector() || self.scalar_fields.contains(field.as_str())
                });
            }
        }
    }
}

pub(crate) fn validate_and_normalize_output_fields(
    schema: &CollectionSchema,
    output_fields: &mut Option<Vec<String>>,
) -> ZResult<()> {
    let Some(fields) = output_fields.as_ref() else {
        return Ok(());
    };

    for field in fields.iter().filter(|field| field.as_str() != "*") {
        if is_system_column(field.as_str()) || field.as_str() == SYS_SCORE {
            continue;
        }
        if !schema.has_field(field) {
            return Err(Status::invalid_argument(format!(
                "{field} not defined in schema"
            )));
        }
    }

    if fields.iter().any(|field| field == "*") {
        *output_fields = None;
    }

    Ok(())
}

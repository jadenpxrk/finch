use finch_types::{CollectionSchema, Status, VectorQuery, ZResult};

use crate::vector_normalization::{copy_query_id_payload, has_query_vector_payload};

use super::Collection;

impl Collection {
    pub(super) fn resolve_query_id_payload(
        &self,
        query: &mut VectorQuery,
        schema: &CollectionSchema,
    ) -> ZResult<()> {
        let Some(pk) = query.id.clone() else {
            return Ok(());
        };

        if has_query_vector_payload(query) {
            return Err(Status::invalid_argument(
                "Cannot provide both id and vector",
            ));
        }

        let field = query.field_name.trim().to_string();
        if field.is_empty() {
            return Err(Status::invalid_argument(
                "query validate failed: vector_field[] not defined in the collection schema",
            ));
        }

        let doc = self
            .fetch(vec![pk.clone()])?
            .remove(&pk)
            .ok_or_else(|| Status::invalid_argument("query validate failed: id not found"))?;

        let field_schema = schema.get_field(&field).ok_or_else(|| {
            Status::invalid_argument(format!(
                "query validate failed: vector_field[{}] not defined in the collection schema",
                field
            ))
        })?;
        if !field_schema.data_type.is_vector() {
            return Err(Status::invalid_argument(
                "query validate failed: field is not vector",
            ));
        }

        copy_query_id_payload(query, field_schema, &field, &doc)
    }
}

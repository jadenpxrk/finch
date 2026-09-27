use finch_types::{CollectionSchema, DataType, FieldSchema, MetricType, Status, ZResult};

pub(crate) struct VectorSearchField<'a> {
    pub(crate) name: &'a str,
    pub(crate) schema: &'a FieldSchema,
    pub(crate) is_sparse: bool,
    pub(crate) is_binary32: bool,
    pub(crate) is_binary64: bool,
    pub(crate) metric: MetricType,
}

pub(crate) fn resolve_vector_search_field<'a>(
    schema: &'a CollectionSchema,
    field: &'a str,
) -> ZResult<VectorSearchField<'a>> {
    let field_schema = schema.get_field(field).ok_or_else(|| {
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

    let is_sparse = field_schema.data_type.is_sparse();
    let is_binary32 = field_schema.data_type == DataType::VectorBinary32;
    let is_binary64 = field_schema.data_type == DataType::VectorBinary64;
    let is_binary = is_binary32 || is_binary64;

    if is_binary {
        let configured = field_schema
            .index_params
            .as_ref()
            .and_then(|p| p.metric())
            .unwrap_or(MetricType::Hamming);
        if configured != MetricType::Hamming && configured != MetricType::Undefined {
            return Err(Status::invalid_argument(
                "query validate failed: binary vectors only support Hamming metric",
            ));
        }
    }

    let metric = if is_binary {
        MetricType::Hamming
    } else {
        field_schema
            .index_params
            .as_ref()
            .and_then(|p| p.metric())
            .unwrap_or(MetricType::InnerProduct)
    };

    Ok(VectorSearchField {
        name: field,
        schema: field_schema,
        is_sparse,
        is_binary32,
        is_binary64,
        metric,
    })
}

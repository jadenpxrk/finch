use crate::index_params::IndexParams;
use crate::status::{Status, ZResult};
use crate::system_columns::is_reserved_field_name;
use crate::types::{DataType, MetricType, QuantizeType};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Default/validation bounds for segment sizing.
pub const MAX_DOC_COUNT_PER_SEGMENT: u64 = 10_000_000;
pub const MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD: u64 = 1_000;

const COLLECTION_NAME_MIN_LEN: usize = 3;
const COLLECTION_NAME_MAX_LEN: usize = 64;
const FIELD_NAME_MIN_LEN: usize = 1;
const FIELD_NAME_MAX_LEN: usize = 32;

const MAX_DENSE_DIM_SIZE: usize = 20_000;
const MAX_SCALAR_FIELD_SIZE: usize = 1024;
const MAX_VECTOR_FIELD_SIZE: usize = 5;

fn is_valid_name_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'-'
}

fn is_valid_collection_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    if bytes.len() < COLLECTION_NAME_MIN_LEN || bytes.len() > COLLECTION_NAME_MAX_LEN {
        return false;
    }
    bytes.iter().copied().all(is_valid_name_char)
}

fn is_valid_field_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    if bytes.len() < FIELD_NAME_MIN_LEN || bytes.len() > FIELD_NAME_MAX_LEN {
        return false;
    }
    bytes.iter().copied().all(is_valid_name_char)
}

fn is_supported_dense_vector_type(dt: DataType) -> bool {
    matches!(
        dt,
        DataType::VectorFp32 | DataType::VectorFp16 | DataType::VectorInt8
    )
}

fn is_supported_sparse_vector_type(dt: DataType) -> bool {
    matches!(dt, DataType::SparseFp32 | DataType::SparseFp16)
}

/// Schema for a single document field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldSchema {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    /// Vector dimension (for vector fields)
    pub dimension: Option<usize>,
    /// Index parameters (if this field is indexed)
    pub index_params: Option<IndexParams>,
}

impl FieldSchema {
    pub fn new(name: impl Into<String>, data_type: DataType) -> Self {
        FieldSchema {
            name: name.into(),
            data_type,
            nullable: true,
            dimension: None,
            index_params: None,
        }
    }

    pub fn not_null(mut self) -> Self {
        self.nullable = false;
        self
    }

    pub fn with_dimension(mut self, dim: usize) -> Self {
        self.dimension = Some(dim);
        self
    }

    pub fn with_index(mut self, params: IndexParams) -> Self {
        self.index_params = Some(params);
        self
    }

    pub fn is_vector(&self) -> bool {
        self.data_type.is_vector()
    }

    pub fn is_scalar(&self) -> bool {
        self.data_type.is_scalar()
    }

    pub fn is_indexed(&self) -> bool {
        self.index_params.is_some()
    }
}

/// Schema for a collection (table of documents)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionSchema {
    pub name: String,
    pub fields: Vec<FieldSchema>,
    /// Maximum documents per segment before creating a new one
    pub max_doc_count_per_segment: u64,
}

impl CollectionSchema {
    pub fn new(name: impl Into<String>) -> Self {
        CollectionSchema {
            name: name.into(),
            fields: Vec::new(),
            max_doc_count_per_segment: MAX_DOC_COUNT_PER_SEGMENT,
        }
    }

    pub fn with_field(mut self, field: FieldSchema) -> Self {
        self.fields.push(field);
        self
    }

    pub fn with_max_docs_per_segment(mut self, n: u64) -> Self {
        self.max_doc_count_per_segment = n;
        self
    }

    pub fn get_field(&self, name: &str) -> Option<&FieldSchema> {
        self.fields.iter().find(|f| f.name == name)
    }

    pub fn get_field_mut(&mut self, name: &str) -> Option<&mut FieldSchema> {
        self.fields.iter_mut().find(|f| f.name == name)
    }

    pub fn has_field(&self, name: &str) -> bool {
        self.fields.iter().any(|f| f.name == name)
    }

    pub fn vector_fields(&self) -> impl Iterator<Item = &FieldSchema> {
        self.fields.iter().filter(|f| f.is_vector())
    }

    pub fn scalar_fields(&self) -> impl Iterator<Item = &FieldSchema> {
        self.fields.iter().filter(|f| f.is_scalar())
    }

    pub fn indexed_vector_fields(&self) -> impl Iterator<Item = &FieldSchema> {
        self.fields
            .iter()
            .filter(|f| f.is_vector() && f.is_indexed())
    }

    pub fn inverted_index_fields(&self) -> impl Iterator<Item = &FieldSchema> {
        self.fields.iter().filter(|f| {
            f.index_params
                .as_ref()
                .map(|p| matches!(p, IndexParams::Invert(_)))
                .unwrap_or(false)
        })
    }

    pub fn validate(&self) -> ZResult<()> {
        if self.name.is_empty() {
            return Err(Status::invalid_argument(
                "schema validate failed: name is empty",
            ));
        }
        if !is_valid_collection_name(&self.name) {
            return Err(Status::invalid_argument(format!(
                "schema validate failed: collection[{}]'s name cannot pass the regex verification",
                self.name
            )));
        }
        if self.max_doc_count_per_segment < MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD {
            return Err(Status::invalid_argument(format!(
                "schema validate failed: max_doc_count_per_segment must >= {}",
                MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD
            )));
        }

        let scalar_fields = self.scalar_fields().count();
        if scalar_fields > MAX_SCALAR_FIELD_SIZE {
            return Err(Status::invalid_argument(format!(
                "schema validate failed: collection[{}]'s field size must <= {}",
                self.name, MAX_SCALAR_FIELD_SIZE
            )));
        }

        let vector_field_count = self.vector_fields().count();
        if vector_field_count > MAX_VECTOR_FIELD_SIZE {
            return Err(Status::invalid_argument(format!(
                "schema validate failed: collection[{}]'s vector field size must <= {}",
                self.name, MAX_VECTOR_FIELD_SIZE
            )));
        }

        // field names must be unique across scalar + vector fields.
        let mut seen_names: HashSet<&str> = HashSet::with_capacity(self.fields.len());
        for f in &self.fields {
            if !seen_names.insert(f.name.as_str()) {
                return Err(Status::invalid_argument(format!(
                    "schema validate failed: duplicate field name '{}': field names must be unique",
                    f.name
                )));
            }
        }

        for field in &self.fields {
            validate_field_schema(field)?;
        }
        Ok(())
    }
}

fn validate_field_schema(field: &FieldSchema) -> ZResult<()> {
    if field.name.is_empty() {
        return Err(Status::invalid_argument(format!(
            "schema validate failed: field[{}]'s name is empty",
            field.name
        )));
    }
    if is_reserved_field_name(field.name.as_str()) {
        return Err(Status::invalid_argument(format!(
            "schema validate failed: field[{}]'s name is reserved",
            field.name
        )));
    }
    if !is_valid_field_name(&field.name) {
        return Err(Status::invalid_argument(format!(
            "schema validate failed: field[{}]'s name cannot pass the regex verification",
            field.name
        )));
    }
    if field.data_type == DataType::Undefined {
        return Err(Status::invalid_argument(format!(
            "schema validate failed: field[{}]'s data_type is not defined",
            field.name
        )));
    }

    if field.is_vector() {
        validate_vector_field_schema(field)
    } else {
        validate_scalar_field_index(field)
    }
}

fn validate_vector_field_schema(field: &FieldSchema) -> ZResult<()> {
    let is_sparse = field.data_type.is_sparse();
    if is_sparse {
        validate_sparse_vector_type(field)?;
    } else {
        validate_dense_vector_type_and_dimension(field)?;
    }

    if let Some(index_params) = field.index_params.as_ref() {
        validate_vector_index_params(field, is_sparse, index_params)?;
    }
    Ok(())
}

fn validate_dense_vector_type_and_dimension(field: &FieldSchema) -> ZResult<()> {
    let dim = field.dimension.ok_or_else(|| {
        Status::invalid_argument(format!(
            "schema validate failed: field[{}]'s dimension must be in (0,20000]",
            field.name
        ))
    })?;
    if dim == 0 || dim > MAX_DENSE_DIM_SIZE {
        return Err(Status::invalid_argument(format!(
            "schema validate failed: field[{}]'s dimension must be in (0,20000]",
            field.name
        )));
    }

    if !is_supported_dense_vector_type(field.data_type) {
        return Err(Status::invalid_argument(format!(
            "schema validate failed: dense_vector's data type only support FP32, but field[{}]'s data type is {:?}",
            field.name, field.data_type
        )));
    }
    Ok(())
}

fn validate_sparse_vector_type(field: &FieldSchema) -> ZResult<()> {
    if !is_supported_sparse_vector_type(field.data_type) {
        return Err(Status::invalid_argument(format!(
            "schema validate failed: sparse_vector's data type only support FP32, but field[{}]'s data type is {:?}",
            field.name, field.data_type
        )));
    }
    Ok(())
}

fn validate_vector_index_params(
    field: &FieldSchema,
    is_sparse: bool,
    index_params: &IndexParams,
) -> ZResult<()> {
    validate_index_param_shape(field, index_params)?;

    let metric = index_params.metric().unwrap_or(MetricType::Undefined);
    let quantize = index_params.quantize().unwrap_or(QuantizeType::Undefined);

    validate_vector_index_kind(field, is_sparse, index_params, metric)?;
    validate_vector_quantize(field, is_sparse, quantize)?;
    validate_ivf_inner_product_dtype(field, index_params, metric)
}

fn validate_index_param_shape(field: &FieldSchema, index_params: &IndexParams) -> ZResult<()> {
    match index_params {
        IndexParams::Hnsw(p) | IndexParams::HnswSparse(p) => {
            // Finch treats `scaling_factor` as a derived parameter and does not
            // persist it in protobuf manifests. Reject overrides to avoid
            // reopen-time mismatches.
            if p.scaling_factor != p.m {
                return Err(Status::invalid_argument(format!(
                    "schema validate failed: HNSW scaling_factor must equal m (field[{}])",
                    field.name
                )));
            }
        }
        IndexParams::Flat(p) | IndexParams::FlatSparse(p) => {
            if p.column_major {
                return Err(Status::invalid_argument(format!(
                    "schema validate failed: FLAT column_major is not supported (field[{}])",
                    field.name
                )));
            }
        }
        IndexParams::Ivf(p) => {
            if p.l1_index.is_some() {
                return Err(Status::invalid_argument(format!(
                    "schema validate failed: composite IVF is not supported (field[{}])",
                    field.name
                )));
            }
        }
        IndexParams::Invert(_) => {}
    }
    Ok(())
}

fn validate_vector_index_kind(
    field: &FieldSchema,
    is_sparse: bool,
    index_params: &IndexParams,
    metric: MetricType,
) -> ZResult<()> {
    if is_sparse {
        let ok_index = matches!(
            index_params,
            IndexParams::FlatSparse(_) | IndexParams::HnswSparse(_)
        );
        if !ok_index {
            return Err(Status::invalid_argument(format!(
                "schema validate failed: sparse_vector's index_params only support FLAT|HNSW index, but field[{}]'s index_params is {:?}",
                field.name, index_params
            )));
        }
        if metric != MetricType::InnerProduct {
            return Err(Status::invalid_argument(format!(
                "schema validate failed: sparse_vector's index_params only support IP metric, but field[{}]'s metric is {:?}",
                field.name, metric
            )));
        }
    } else {
        let ok_index = matches!(
            index_params,
            IndexParams::Flat(_) | IndexParams::Hnsw(_) | IndexParams::Ivf(_)
        );
        if !ok_index {
            return Err(Status::invalid_argument(format!(
                "schema validate failed: dense_vector's index_params only support FLAT|HNSW|IVF index, but field[{}]'s index_params is {:?}",
                field.name, index_params
            )));
        }

        // Finch schema does not allow binary dense vectors at all (VECTOR_BINARY32/64),
        // so there are no additional binary-specific index constraints here.
    }
    Ok(())
}

fn validate_vector_quantize(
    field: &FieldSchema,
    is_sparse: bool,
    quantize: QuantizeType,
) -> ZResult<()> {
    if quantize == QuantizeType::Undefined {
        return Ok(());
    }

    match field.data_type {
        DataType::VectorFp32 => {
            let ok = matches!(
                quantize,
                QuantizeType::Fp16 | QuantizeType::Int4 | QuantizeType::Int8
            );
            if !ok {
                return Err(Status::invalid_argument(format!(
                    "schema validate failed: dense_vector's index_params of VectorFp32 support Fp16/Int4/Int8 quantize, but field[{}]'s quantize_type is {:?}",
                    field.name, quantize
                )));
            }
        }
        DataType::SparseFp32 => {
            let ok = matches!(quantize, QuantizeType::Fp16);
            if !ok {
                return Err(Status::invalid_argument(format!(
                    "schema validate failed: sparse_vector's index_params of SparseFp32 only support Fp16 quantize, but field[{}]'s quantize_type is {:?}",
                    field.name, quantize
                )));
            }
        }
        _ => {
            return Err(Status::invalid_argument(format!(
                "schema validate failed: {}'s index_params of {:?} do not support quantize, but field[{}]'s quantize_type is {:?}",
                if is_sparse { "sparse_vector" } else { "dense_vector" },
                field.data_type,
                field.name,
                quantize
            )));
        }
    }
    Ok(())
}

fn validate_ivf_inner_product_dtype(
    field: &FieldSchema,
    index_params: &IndexParams,
    metric: MetricType,
) -> ZResult<()> {
    if matches!(index_params, IndexParams::Ivf(_))
        && metric == MetricType::InnerProduct
        && !matches!(field.data_type, DataType::VectorFp16 | DataType::VectorFp32)
    {
        return Err(Status::invalid_argument(
            "schema validate failed: IVF index only support FP32/FP16 data types according to the IP metric",
        ));
    }
    Ok(())
}

fn validate_scalar_field_index(field: &FieldSchema) -> ZResult<()> {
    if let Some(index_params) = field.index_params.as_ref() {
        if !matches!(index_params, IndexParams::Invert(_)) {
            return Err(Status::invalid_argument(format!(
                "schema validate failed: scalar_field's index_params only support INVERT index, but field[{}]'s index_params is {:?}",
                field.name, index_params
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index_params::{FlatIndexParams, HnswIndexParams, IvfIndexParams};

    #[test]
    fn validate_rejects_bad_collection_name() {
        let schema = CollectionSchema::new("x")
            .with_field(FieldSchema::new("vec", DataType::VectorFp32).with_dimension(3));
        let err = schema.validate().unwrap_err();
        assert!(err.message.contains("schema validate failed"));
    }

    #[test]
    fn validate_requires_vector_fields_non_empty() {
        let schema = CollectionSchema::new("abc")
            .with_field(FieldSchema::new("kind", DataType::String))
            .with_field(FieldSchema::new("counter", DataType::Int64));
        schema.validate().unwrap();
    }

    #[test]
    fn validate_dense_vector_dim_bounds() {
        let schema = CollectionSchema::new("abc")
            .with_field(FieldSchema::new("vec", DataType::VectorFp32).with_dimension(0));
        let err = schema.validate().unwrap_err();
        assert!(err.message.contains("dimension must be in (0,20000]"));
    }

    #[test]
    fn validate_sparse_vector_does_not_require_dimension() {
        let schema = CollectionSchema::new("abc").with_field(
            FieldSchema::new("svec", DataType::SparseFp32).with_index(IndexParams::FlatSparse(
                FlatIndexParams::new(MetricType::InnerProduct),
            )),
        );
        // Note: SparseFp32 is supported, but quantize is still rejected (if set).
        schema.validate().unwrap();
    }

    #[test]
    fn validate_rejects_hnsw_scaling_factor_override() {
        let p = HnswIndexParams::new(MetricType::L2).with_scaling_factor(123);
        let schema = CollectionSchema::new("abc").with_field(
            FieldSchema::new("vec", DataType::VectorFp32)
                .with_dimension(8)
                .with_index(IndexParams::Hnsw(p)),
        );
        let err = schema.validate().unwrap_err();
        assert!(err.message.contains("HNSW scaling_factor must equal m"));
    }

    #[test]
    fn validate_rejects_flat_column_major() {
        let p = FlatIndexParams {
            metric: MetricType::L2,
            quantize: QuantizeType::Undefined,
            column_major: true,
        };
        let schema = CollectionSchema::new("abc").with_field(
            FieldSchema::new("vec", DataType::VectorFp32)
                .with_dimension(8)
                .with_index(IndexParams::Flat(p)),
        );
        let err = schema.validate().unwrap_err();
        assert!(err.message.contains("FLAT column_major is not supported"));
    }

    #[test]
    fn validate_rejects_composite_ivf() {
        let mut ivf = IvfIndexParams::new(MetricType::L2).with_n_list(16);
        ivf.l1_index = Some(Box::new(IndexParams::Flat(FlatIndexParams::new(
            MetricType::L2,
        ))));
        let schema = CollectionSchema::new("abc").with_field(
            FieldSchema::new("vec", DataType::VectorFp32)
                .with_dimension(8)
                .with_index(IndexParams::Ivf(ivf)),
        );
        let err = schema.validate().unwrap_err();
        assert!(err.message.contains("composite IVF is not supported"));
    }

    #[test]
    fn validate_rejects_sparse_metric_non_ip() {
        let schema = CollectionSchema::new("abc").with_field(
            FieldSchema::new("svec", DataType::SparseFp32).with_index(IndexParams::FlatSparse(
                FlatIndexParams::new(MetricType::L2),
            )),
        );
        let err = schema.validate().unwrap_err();
        assert!(err.message.contains("only support IP metric"));
    }

    #[test]
    fn validate_rejects_unsupported_dense_vector_types() {
        let schema = CollectionSchema::new("abc").with_field(
            FieldSchema::new("b", DataType::VectorBinary32)
                .with_dimension(8)
                .with_index(IndexParams::Flat(FlatIndexParams::new(MetricType::Hamming))),
        );
        let err = schema.validate().unwrap_err();
        assert!(err.message.contains("dense_vector's data type"));
    }

    #[test]
    fn validate_rejects_vector_quantize_unsupported_dtype() {
        let mut p = HnswIndexParams::new(MetricType::L2);
        p.quantize = QuantizeType::Int8;
        let schema = CollectionSchema::new("abc").with_field(
            FieldSchema::new("vec", DataType::VectorFp16)
                .with_dimension(8)
                .with_index(IndexParams::Hnsw(p)),
        );
        let err = schema.validate().unwrap_err();
        assert!(err.message.contains("do not support quantize"));
    }

    #[test]
    fn validate_allows_vectorfp32_quantize_int8() {
        let mut p = IvfIndexParams::new(MetricType::L2);
        p.quantize = QuantizeType::Int8;
        let schema = CollectionSchema::new("abc").with_field(
            FieldSchema::new("vec", DataType::VectorFp32)
                .with_dimension(8)
                .with_index(IndexParams::Ivf(p)),
        );
        schema.validate().unwrap();
    }

    #[test]
    fn validate_rejects_ivf_ip_non_fp16_fp32() {
        let p = IvfIndexParams::new(MetricType::InnerProduct);
        let schema = CollectionSchema::new("abc").with_field(
            FieldSchema::new("vec", DataType::VectorInt8)
                .with_dimension(8)
                .with_index(IndexParams::Ivf(p)),
        );
        let err = schema.validate().unwrap_err();
        assert!(err.message.contains("IVF index only support FP32/FP16"));
    }

    #[test]
    fn validate_rejects_scalar_with_vector_index_params() {
        let schema = CollectionSchema::new("abc")
            .with_field(FieldSchema::new("vec", DataType::VectorFp32).with_dimension(8))
            .with_field(
                FieldSchema::new("title", DataType::String)
                    .with_index(IndexParams::Flat(FlatIndexParams::new(MetricType::L2))),
            );
        let err = schema.validate().unwrap_err();
        assert!(err.message.contains("scalar_field"));
    }
}

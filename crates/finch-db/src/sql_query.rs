use std::cmp::Ordering;
use std::collections::HashSet;
use std::sync::Arc;

use finch_types::{
    CollectionSchema, DataType, Doc, Status, Value, VectorQuery, ZResult, SYS_GLOBAL_DOC_ID,
    SYS_LOCAL_ROW_ID, SYS_SCORE, SYS_USER_ID,
};

use crate::row_locator::RowLocator;
use crate::sqlengine::query::{
    CmpOp, LogicExpr, RelExpr, SelectItem as SqlSelectItem, SqlSelect, ValueExpr, VectorCond,
    VectorLiteral as SqlVectorLiteral,
};
use crate::system_projection::{resolve_row_id, SystemColumnOutputs};
use crate::vector_literal::{
    parse_vector_literal_f32, parse_vector_literal_u32, parse_vector_literal_u64,
};
use crate::vector_normalization::has_query_vector_payload;

const DEFAULT_SQL_TOPN: usize = 20;

pub(crate) struct SqlQueryPlan {
    query: VectorQuery,
    projection: SqlProjection,
}

impl SqlQueryPlan {
    pub(crate) fn prepare(sql: &str, schema: &CollectionSchema) -> ZResult<Self> {
        let select = SqlSelect::parse(sql)?;

        if select.table != schema.name {
            return Err(Status::invalid_argument(format!(
                "table not found: {}",
                select.table
            )));
        }

        let topk = match select.limit {
            None => DEFAULT_SQL_TOPN,
            Some(n) => usize::try_from(n).map_err(|_| {
                Status::invalid_argument(format!("LIMIT must not be negative: {n}"))
            })?,
        };

        let (vector_cond, scalar_expr) = match select.where_expr.clone() {
            None => (None, None),
            Some(expr) => split_vector_condition_with_schema(expr, schema)?,
        };
        let filter = scalar_expr
            .as_ref()
            .map(|e| e.to_filter_string())
            .and_then(|s| {
                let trimmed = s.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            });

        let order_by = prepare_order_by(schema, &select)?;
        let mut projection = SqlProjection::prepare(schema, &select, order_by)?;

        let mut query = VectorQuery {
            topk,
            field_name: String::new(),
            id: None,
            query_vector: Vec::new(),
            query_vector_u32: Vec::new(),
            query_vector_u64: Vec::new(),
            sparse_indices: Vec::new(),
            sparse_values: Vec::new(),
            filter,
            include_vector: projection.include_vector,
            include_doc_id: projection.include_doc_id,
            output_fields: projection.output_fields.clone(),
            query_params: finch_types::QueryParams::default(),
        };

        apply_vector_condition(schema, vector_cond, &mut query)?;
        if !projection.order_by.is_empty() && !has_query_vector_payload(&query) {
            projection.limit_after_sort = Some(topk);
        }

        Ok(Self { query, projection })
    }

    pub(crate) fn into_parts(self) -> (VectorQuery, SqlProjection) {
        (self.query, self.projection)
    }
}

/// Selected scalar or vector columns: sources, `(source, output)` renames, and
/// sources that stay under their own name.
#[derive(Default)]
struct ProjectedColumns {
    sources: Vec<String>,
    selects: Vec<(String, String)>,
    keep_sources: HashSet<String>,
}

impl ProjectedColumns {
    fn add(&mut self, name: &str, out_name: &str) {
        self.sources.push(name.to_string());
        self.selects.push((name.to_string(), out_name.to_string()));
        if out_name == name {
            self.keep_sources.insert(name.to_string());
        }
    }

    fn sort_sources(&mut self) {
        self.sources.sort();
        self.sources.dedup();
    }

    fn renames_any(&self) -> bool {
        self.selects.iter().any(|(source, out)| source != out)
    }

    #[inline]
    fn copy_to_outputs(&self, doc: &mut Doc, drop_renamed_sources: bool) {
        // Read all sources first so 'a AS b, b AS a' copies original values.
        let values: Vec<(String, Value)> = self
            .selects
            .iter()
            .filter_map(|(source, out_name)| {
                let value = doc.fields.get(source)?.clone();
                Some((out_name.clone(), value))
            })
            .collect();
        if drop_renamed_sources {
            for source in &self.sources {
                if !self.keep_sources.contains(source) {
                    doc.fields.remove(source);
                }
            }
        }
        doc.fields.extend(values);
    }
}

/// Columns named by the SELECT list, split by kind.
#[derive(Default)]
struct SelectedColumns {
    saw_asterisk: bool,
    scalar: ProjectedColumns,
    vector: ProjectedColumns,
    system_outputs: SystemColumnOutputs,
}

impl SelectedColumns {
    fn collect(schema: &CollectionSchema, items: &[SqlSelectItem]) -> ZResult<Self> {
        let mut columns = Self::default();
        for item in items {
            let SqlSelectItem::Field { name, alias } = item else {
                columns.saw_asterisk = true;
                continue;
            };
            let out_name = alias.clone().unwrap_or_else(|| name.clone());
            if let Some(names) = columns.system_outputs.output_names_mut(name) {
                names.push(out_name);
                continue;
            }
            let fs = schema
                .get_field(name)
                .ok_or_else(|| Status::invalid_argument(format!("{name} not defined in schema")))?;
            if fs.data_type.is_vector() {
                columns.vector.add(name, &out_name);
            } else {
                columns.scalar.add(name, &out_name);
            }
        }
        Ok(columns)
    }
}

pub(crate) struct SqlProjection {
    include_vector: bool,
    include_doc_id: bool,
    output_fields: Option<Vec<String>>,
    saw_asterisk: bool,
    scalar: ProjectedColumns,
    vector: ProjectedColumns,
    system_outputs: SystemColumnOutputs,
    expose_doc_id_member: bool,
    order_by: Vec<(OrderByField, bool)>,
    /// Set when a scalar ORDER BY must see every matching row before LIMIT applies.
    limit_after_sort: Option<usize>,
}

impl SqlProjection {
    fn prepare(
        schema: &CollectionSchema,
        select: &SqlSelect,
        order_by: OrderByPlan,
    ) -> ZResult<Self> {
        let SelectedColumns {
            saw_asterisk,
            mut scalar,
            mut vector,
            system_outputs,
        } = SelectedColumns::collect(schema, &select.selected)?;

        scalar.sort_sources();
        vector.sort_sources();
        for field in &order_by.scalar_fetch {
            if !scalar.sources.iter().any(|source| source == field) {
                scalar.sources.push(field.clone());
            }
        }
        scalar.sort_sources();

        let include_vector = !vector.sources.is_empty();
        let expose_doc_id_member = system_outputs.wants_global_doc_id();
        let include_doc_id = expose_doc_id_member
            || system_outputs.wants_row_id()
            || order_by.needs_row_id_base
            || order_by.needs_doc_id;

        let output_fields = if saw_asterisk {
            None
        } else if scalar.sources.is_empty() {
            Some(Vec::new())
        } else {
            Some(scalar.sources.clone())
        };

        Ok(Self {
            include_vector,
            include_doc_id,
            output_fields,
            saw_asterisk,
            scalar,
            vector,
            system_outputs,
            expose_doc_id_member,
            order_by: order_by.keys,
            limit_after_sort: None,
        })
    }

    pub(crate) fn sorts_all_matches(&self) -> bool {
        self.limit_after_sort.is_some()
    }

    pub(crate) fn needs_projection(&self) -> bool {
        self.include_vector
            || self.system_outputs.wants_any()
            || !self.order_by.is_empty()
            || self.scalar.renames_any()
            || self.vector.renames_any()
    }

    pub(crate) fn apply(
        self,
        mut docs: Vec<Arc<Doc>>,
        row_locator: &RowLocator,
    ) -> ZResult<Vec<Arc<Doc>>> {
        if !self.needs_projection() {
            return Ok(docs);
        }

        let vector_source_set: HashSet<&str> =
            self.vector.sources.iter().map(String::as_str).collect();

        let mut out: Vec<(Vec<OrderKey>, Arc<Doc>)> = Vec::with_capacity(docs.len());
        for doc in docs.drain(..) {
            let mut doc = (*doc).clone();
            let order_keys = self.order_keys_for_doc(&doc, row_locator)?;
            self.project_doc(&mut doc, &vector_source_set, row_locator)?;
            out.push((order_keys, Arc::new(doc)));
        }

        if !self.order_by.is_empty() {
            out.sort_by(|(left_keys, left_doc), (right_keys, right_doc)| {
                self.cmp_order_keys(left_keys, right_keys)
                    .then_with(|| left_doc.pk.cmp(&right_doc.pk))
            });
        }
        if let Some(limit) = self.limit_after_sort {
            out.truncate(limit);
        }

        Ok(out.into_iter().map(|(_, doc)| doc).collect())
    }

    #[inline]
    fn project_doc(
        &self,
        doc: &mut Doc,
        vector_source_set: &HashSet<&str>,
        row_locator: &RowLocator,
    ) -> ZResult<()> {
        if self.include_vector {
            doc.fields.retain(|key, value| {
                !value.is_vector() || vector_source_set.contains(key.as_str())
            });
            self.vector.copy_to_outputs(doc, true);
        }

        self.scalar.copy_to_outputs(doc, !self.saw_asterisk);

        self.system_outputs
            .insert_into_doc(doc, Some(row_locator))?;

        if !self.expose_doc_id_member {
            doc.doc_id = 0;
        }
        Ok(())
    }

    #[inline]
    fn cmp_order_keys(&self, left_keys: &[OrderKey], right_keys: &[OrderKey]) -> Ordering {
        for (idx, (_field, desc)) in self.order_by.iter().enumerate() {
            let left = left_keys.get(idx).unwrap_or(&OrderKey::Null);
            let right = right_keys.get(idx).unwrap_or(&OrderKey::Null);
            let ord = cmp_order_key(left, right, *desc);
            if ord != Ordering::Equal {
                return ord;
            }
        }
        Ordering::Equal
    }

    fn order_keys_for_doc(&self, doc: &Doc, row_locator: &RowLocator) -> ZResult<Vec<OrderKey>> {
        if self.order_by.is_empty() {
            return Ok(Vec::new());
        }

        let mut keys = Vec::with_capacity(self.order_by.len());
        for (field, _desc) in &self.order_by {
            let key = match field {
                OrderByField::System(sys) => sys.key_for(doc, row_locator)?,
                OrderByField::Scalar { name, dt } => scalar_order_key(doc.fields.get(name), *dt),
            };
            keys.push(key);
        }
        Ok(keys)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OrderBySystem {
    Uid,
    RowId,
    GlobalDocId,
    Score,
}

impl OrderBySystem {
    fn key_for(self, doc: &Doc, row_locator: &RowLocator) -> ZResult<OrderKey> {
        Ok(match self {
            OrderBySystem::Uid => OrderKey::String(doc.pk.clone()),
            OrderBySystem::GlobalDocId => OrderKey::U64(doc.doc_id),
            OrderBySystem::Score => OrderKey::F64(doc.score as f64),
            OrderBySystem::RowId => OrderKey::U64(resolve_row_id(Some(row_locator), doc.doc_id)?),
        })
    }
}

#[derive(Clone, Debug)]
enum OrderByField {
    System(OrderBySystem),
    Scalar { name: String, dt: DataType },
}

#[derive(Clone, Debug)]
enum OrderKey {
    Null,
    Bool(bool),
    I64(i64),
    U64(u64),
    F64(f64),
    String(String),
    Bytes(Vec<u8>),
}

struct OrderByPlan {
    /// Order-by fields with their `desc` flag.
    keys: Vec<(OrderByField, bool)>,
    /// Scalar fields that must be fetched to compute order keys.
    scalar_fetch: Vec<String>,
    needs_row_id_base: bool,
    needs_doc_id: bool,
}

fn prepare_order_by(schema: &CollectionSchema, select: &SqlSelect) -> ZResult<OrderByPlan> {
    let mut plan = OrderByPlan {
        keys: Vec::new(),
        scalar_fetch: Vec::new(),
        needs_row_id_base: false,
        needs_doc_id: false,
    };

    for item in &select.order_by {
        let field = item.field.as_str();
        let order_field = match field {
            SYS_USER_ID => OrderByField::System(OrderBySystem::Uid),
            SYS_LOCAL_ROW_ID => {
                plan.needs_row_id_base = true;
                plan.needs_doc_id = true;
                OrderByField::System(OrderBySystem::RowId)
            }
            SYS_GLOBAL_DOC_ID => {
                plan.needs_doc_id = true;
                OrderByField::System(OrderBySystem::GlobalDocId)
            }
            SYS_SCORE => OrderByField::System(OrderBySystem::Score),
            _ => {
                let dt = require_orderable_scalar_field(
                    schema,
                    field,
                    "order by fields should not be array data type",
                )?;
                plan.scalar_fetch.push(field.to_string());
                OrderByField::Scalar {
                    name: field.to_string(),
                    dt,
                }
            }
        };
        plan.keys.push((order_field, item.desc));
    }

    plan.scalar_fetch.sort();
    plan.scalar_fetch.dedup();
    Ok(plan)
}

/// Data type of a non-vector, non-array schema field used for ordering or grouping.
pub(crate) fn require_orderable_scalar_field(
    schema: &CollectionSchema,
    field: &str,
    array_error: &'static str,
) -> ZResult<DataType> {
    let not_defined = || Status::invalid_argument(format!("{field} not defined in schema"));
    let fs = schema.get_field(field).ok_or_else(not_defined)?;
    if fs.data_type.is_vector() {
        return Err(not_defined());
    }
    if fs.data_type.is_array() {
        return Err(Status::invalid_argument(array_error));
    }
    Ok(fs.data_type)
}

fn apply_vector_condition(
    schema: &CollectionSchema,
    vector_cond: Option<VectorCond>,
    query: &mut VectorQuery,
) -> ZResult<()> {
    let Some(vector_cond) = vector_cond else {
        return Ok(());
    };

    let vector_fs = schema
        .get_field(vector_cond.field.as_str())
        .ok_or_else(|| {
            Status::invalid_argument(format!("vector field not found: {}", vector_cond.field))
        })?;
    if !vector_fs.data_type.is_vector() {
        return Err(Status::invalid_argument(format!(
            "vector field not found: {}",
            vector_cond.field
        )));
    }

    query.field_name = vector_cond.field;
    match vector_cond.literal {
        SqlVectorLiteral::Vector(vector) => {
            apply_vector_literal(vector_fs.data_type, vector.as_str(), query)
        }
        SqlVectorLiteral::Matrix(rows) => {
            apply_matrix_vector_literal(vector_fs.data_type, &rows, query)
        }
    }
}

fn apply_vector_literal(
    data_type: DataType,
    literal: &str,
    query: &mut VectorQuery,
) -> ZResult<()> {
    match data_type {
        DataType::VectorBinary32 => {
            query.query_vector_u32 = parse_vector_literal_u32(literal)?;
        }
        DataType::VectorBinary64 => {
            query.query_vector_u64 = parse_vector_literal_u64(literal)?;
        }
        dt if dt.is_sparse() => {
            let values = parse_vector_literal_f32(literal)?;
            query.sparse_indices = (0..values.len() as u32).collect();
            query.sparse_values = values;
        }
        _ => {
            query.query_vector = parse_vector_literal_f32(literal)?;
        }
    }
    Ok(())
}

/// Concatenates the parsed rows of a matrix literal.
fn parse_matrix_rows<T>(rows: &[String], parse: fn(&str) -> ZResult<Vec<T>>) -> ZResult<Vec<T>> {
    let mut values = Vec::new();
    for row in rows {
        values.extend(parse(row.as_str())?);
    }
    Ok(values)
}

fn apply_matrix_vector_literal(
    data_type: DataType,
    rows: &[String],
    query: &mut VectorQuery,
) -> ZResult<()> {
    match data_type {
        DataType::VectorBinary32 => {
            query.query_vector_u32 = parse_matrix_rows(rows, parse_vector_literal_u32)?;
        }
        DataType::VectorBinary64 => {
            query.query_vector_u64 = parse_matrix_rows(rows, parse_vector_literal_u64)?;
        }
        dt if dt.is_sparse() => {
            let values = parse_matrix_rows(rows, parse_vector_literal_f32)?;
            query.sparse_indices = (0..values.len() as u32).collect();
            query.sparse_values = values;
        }
        _ => {
            query.query_vector = parse_matrix_rows(rows, parse_vector_literal_f32)?;
        }
    }
    Ok(())
}

/// Extracted vector condition and the remaining scalar expression.
type VectorSplit = (Option<VectorCond>, Option<LogicExpr>);

const VECTOR_UNDER_OR: &str = "vector condition must NOT be OR ancestor.";

fn split_vector_condition_with_schema(
    expr: LogicExpr,
    schema: &CollectionSchema,
) -> ZResult<VectorSplit> {
    let mut first_vector_text: Option<String> = None;
    remove_vector_condition(expr, schema, false, &mut first_vector_text)
}

fn remove_vector_condition(
    expr: LogicExpr,
    schema: &CollectionSchema,
    or_ancestor: bool,
    first_vector_text: &mut Option<String>,
) -> ZResult<VectorSplit> {
    match expr {
        LogicExpr::Or(left, right) => {
            let (left_vector, left_remaining) =
                remove_vector_condition(*left, schema, true, first_vector_text)?;
            let (right_vector, right_remaining) =
                remove_vector_condition(*right, schema, true, first_vector_text)?;

            if left_vector.is_some() || right_vector.is_some() {
                return Err(Status::invalid_argument(VECTOR_UNDER_OR));
            }

            let remaining =
                LogicExpr::join_remaining(left_remaining, right_remaining, LogicExpr::Or);
            Ok((None, remaining))
        }
        LogicExpr::And(left, right) => {
            let (left_vector, left_remaining) =
                remove_vector_condition(*left, schema, or_ancestor, first_vector_text)?;
            let (right_vector, right_remaining) =
                remove_vector_condition(*right, schema, or_ancestor, first_vector_text)?;

            if let (Some(first), Some(second)) = (left_vector.as_ref(), right_vector.as_ref()) {
                return Err(invalid_more_than_one_vector(
                    first_vector_text.as_deref().unwrap_or(first.field.as_str()),
                    second.field.as_str(),
                ));
            }

            let vector = left_vector.or(right_vector);
            let remaining =
                LogicExpr::join_remaining(left_remaining, right_remaining, LogicExpr::And);
            Ok((vector, remaining))
        }
        LogicExpr::Rel(rel) => remove_vector_relation(rel, schema, or_ancestor, first_vector_text),
    }
}

fn remove_vector_relation(
    rel: RelExpr,
    schema: &CollectionSchema,
    or_ancestor: bool,
    first_vector_text: &mut Option<String>,
) -> ZResult<VectorSplit> {
    let rel_text = rel.to_filter_string();
    match &rel {
        RelExpr::Compare {
            lhs: ValueExpr::Ident(field),
            op,
            rhs,
        } if field_is_vector(schema, field.as_str()) => {
            if or_ancestor || *op != CmpOp::Eq {
                return Err(vector_relation_error(or_ancestor, &rel_text));
            }
            return vector_compare_condition(field, rhs, rel_text, first_vector_text);
        }
        RelExpr::Compare { lhs, rhs, .. } => {
            reject_scalar_vector_compare(schema, lhs, rhs, &rel_text)?;
        }
        RelExpr::Like { field, pattern } => {
            if is_vector_lit(pattern) {
                return Err(Status::invalid_argument(
                    "like phrase only support string now.",
                ));
            }
            reject_vector_non_eq_relation(schema, field, or_ancestor, &rel_text)?;
        }
        RelExpr::InList { field, .. }
        | RelExpr::Contain { field, .. }
        | RelExpr::IsNull { field, .. } => {
            reject_vector_non_eq_relation(schema, field, or_ancestor, &rel_text)?;
        }
    }

    Ok((None, Some(LogicExpr::Rel(rel))))
}

/// A vector literal compared against a non-vector left side.
fn reject_scalar_vector_compare(
    schema: &CollectionSchema,
    lhs: &ValueExpr,
    rhs: &ValueExpr,
    rel_text: &str,
) -> ZResult<()> {
    if !is_vector_lit(rhs) {
        return Ok(());
    }
    let ValueExpr::Ident(field) = lhs else {
        return Err(Status::invalid_argument(format!(
            "left side in relation expr must be single field name or function call. {}",
            rel_text
        )));
    };
    if field_exists(schema, field.as_str()) {
        return Err(Status::invalid_argument(format!(
            "field type and value type not match in relation expr. {}",
            rel_text
        )));
    }
    Err(Status::invalid_argument(format!(
        "vector vector not supported for schema free field in relation expr: {}",
        rel_text
    )))
}

fn vector_compare_condition(
    field: &str,
    rhs: &ValueExpr,
    rel_text: String,
    first_vector_text: &mut Option<String>,
) -> ZResult<VectorSplit> {
    let literal = match rhs {
        ValueExpr::VectorLiteral(vector) => SqlVectorLiteral::Vector(vector.clone()),
        ValueExpr::MatrixLiteral(rows) => SqlVectorLiteral::Matrix(rows.clone()),
        _ => return Err(Status::invalid_argument("invalid vector value node.")),
    };
    first_vector_text.get_or_insert(rel_text);
    Ok((
        Some(VectorCond {
            field: field.to_string(),
            literal,
        }),
        None,
    ))
}

/// A vector field may only appear in a top-level `=` relation.
fn vector_relation_error(or_ancestor: bool, rel_text: &str) -> Status {
    if or_ancestor {
        return Status::invalid_argument(VECTOR_UNDER_OR);
    }
    Status::invalid_argument(format!("vector field only support EQ. {}", rel_text))
}

fn reject_vector_non_eq_relation(
    schema: &CollectionSchema,
    field: &ValueExpr,
    or_ancestor: bool,
    rel_text: &str,
) -> ZResult<()> {
    if let ValueExpr::Ident(name) = field {
        if field_is_vector(schema, name.as_str()) {
            return Err(vector_relation_error(or_ancestor, rel_text));
        }
    }
    Ok(())
}

fn field_is_vector(schema: &CollectionSchema, field: &str) -> bool {
    schema
        .get_field(field)
        .is_some_and(|field_schema| field_schema.data_type.is_vector())
}

fn field_exists(schema: &CollectionSchema, field: &str) -> bool {
    schema.get_field(field).is_some()
}

fn is_vector_lit(value: &ValueExpr) -> bool {
    matches!(
        value,
        ValueExpr::VectorLiteral(_) | ValueExpr::MatrixLiteral(_)
    )
}

fn invalid_more_than_one_vector(first: &str, second: &str) -> Status {
    Status::invalid_argument(format!(
        "more than one vector search is not supported. {first} {second}"
    ))
}

fn scalar_order_key(value: Option<&Value>, data_type: DataType) -> OrderKey {
    let Some(value) = value else {
        return OrderKey::Null;
    };
    if value.is_null() {
        return OrderKey::Null;
    }

    match data_type {
        DataType::Bool => match value {
            Value::Bool(x) => OrderKey::Bool(*x),
            _ => OrderKey::Null,
        },
        DataType::Int8 => match value {
            Value::I8(x) => OrderKey::I64(*x as i64),
            _ => OrderKey::Null,
        },
        DataType::Int16 => match value {
            Value::I16(x) => OrderKey::I64(*x as i64),
            _ => OrderKey::Null,
        },
        DataType::Int32 => match value {
            Value::I32(x) => OrderKey::I64(*x as i64),
            _ => OrderKey::Null,
        },
        DataType::Int64 => match value {
            Value::I64(x) => OrderKey::I64(*x),
            _ => OrderKey::Null,
        },
        DataType::Uint8 => match value {
            Value::U8(x) => OrderKey::U64(*x as u64),
            _ => OrderKey::Null,
        },
        DataType::Uint16 => match value {
            Value::U16(x) => OrderKey::U64(*x as u64),
            _ => OrderKey::Null,
        },
        DataType::Uint32 => match value {
            Value::U32(x) => OrderKey::U64(*x as u64),
            _ => OrderKey::Null,
        },
        DataType::Uint64 => match value {
            Value::U64(x) => OrderKey::U64(*x),
            _ => OrderKey::Null,
        },
        DataType::Float16 => match value {
            Value::F16(x) => OrderKey::F64(x.to_f32() as f64),
            _ => OrderKey::Null,
        },
        DataType::Float32 => match value {
            Value::F32(x) => OrderKey::F64(*x as f64),
            _ => OrderKey::Null,
        },
        DataType::Float64 => match value {
            Value::F64(x) => OrderKey::F64(*x),
            _ => OrderKey::Null,
        },
        DataType::String => match value {
            Value::String(s) => OrderKey::String(s.clone()),
            _ => OrderKey::Null,
        },
        DataType::Bytes | DataType::Binary => match value {
            Value::Bytes(bytes) => OrderKey::Bytes(bytes.clone()),
            _ => OrderKey::Null,
        },
        _ => OrderKey::Null,
    }
}

fn cmp_order_key(left: &OrderKey, right: &OrderKey, desc: bool) -> Ordering {
    let left_null = matches!(left, OrderKey::Null);
    let right_null = matches!(right, OrderKey::Null);
    if left_null && right_null {
        return Ordering::Equal;
    }
    if left_null {
        return Ordering::Greater;
    }
    if right_null {
        return Ordering::Less;
    }

    let ord = match (left, right) {
        (OrderKey::Bool(x), OrderKey::Bool(y)) => x.cmp(y),
        (OrderKey::I64(x), OrderKey::I64(y)) => x.cmp(y),
        (OrderKey::U64(x), OrderKey::U64(y)) => x.cmp(y),
        (OrderKey::F64(x), OrderKey::F64(y)) => x.partial_cmp(y).unwrap_or_else(|| {
            if x.is_nan() && y.is_nan() {
                Ordering::Equal
            } else if x.is_nan() {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        }),
        (OrderKey::String(x), OrderKey::String(y)) => x.cmp(y),
        (OrderKey::Bytes(x), OrderKey::Bytes(y)) => x.cmp(y),
        _ => order_key_rank(left).cmp(&order_key_rank(right)),
    };

    if desc {
        ord.reverse()
    } else {
        ord
    }
}

fn order_key_rank(key: &OrderKey) -> u8 {
    match key {
        OrderKey::Null => 0,
        OrderKey::Bool(_) => 1,
        OrderKey::I64(_) => 2,
        OrderKey::U64(_) => 3,
        OrderKey::F64(_) => 4,
        OrderKey::String(_) => 5,
        OrderKey::Bytes(_) => 6,
    }
}

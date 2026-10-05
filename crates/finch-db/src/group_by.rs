use std::collections::HashMap;
use std::sync::Arc;

use finch_types::{
    CollectionSchema, Doc, GroupResult, Value, VectorQuery, ZResult, SYS_GLOBAL_DOC_ID,
    SYS_LOCAL_ROW_ID, SYS_USER_ID,
};
use half::f16;

use crate::row_locator::RowLocator;
use crate::sql_query::require_orderable_scalar_field;
use crate::system_projection::resolve_row_id;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum GroupByKind {
    SchemaScalar,
    SystemUid,
    SystemRowId,
    SystemGlobalDocId,
}

impl GroupByKind {
    fn for_field(field: &str) -> Self {
        match field {
            SYS_USER_ID => Self::SystemUid,
            SYS_LOCAL_ROW_ID => Self::SystemRowId,
            SYS_GLOBAL_DOC_ID => Self::SystemGlobalDocId,
            _ => Self::SchemaScalar,
        }
    }

    fn needs_doc_id(self) -> bool {
        matches!(self, Self::SystemRowId | Self::SystemGlobalDocId)
    }
}

pub(crate) struct GroupByPlan {
    group_by_field: String,
    kind: GroupByKind,
    docs_per_group: usize,
    group_limit: usize,
    original_output_fields: Option<Vec<String>>,
    added_group_by_field: bool,
    expose_doc_id_member: bool,
}

impl GroupByPlan {
    pub(crate) fn prepare(
        schema: &CollectionSchema,
        mut base_query: VectorQuery,
        group_by_field: String,
        docs_per_group: usize,
        group_limit: usize,
    ) -> ZResult<(Self, VectorQuery)> {
        let kind = GroupByKind::for_field(&group_by_field);
        if matches!(kind, GroupByKind::SchemaScalar) {
            require_orderable_scalar_field(
                schema,
                &group_by_field,
                "group by fields should not be array data type",
            )?;
        }

        base_query.topk = group_limit
            .saturating_mul(docs_per_group)
            .saturating_mul(2)
            .max(base_query.topk);

        let original_output_fields = base_query.output_fields.clone();
        let selects_all_scalars = original_output_fields
            .as_ref()
            .map(|v| v.iter().any(|f| f == "*"))
            .unwrap_or(true);
        let mut added_group_by_field = false;
        if matches!(kind, GroupByKind::SchemaScalar) && !selects_all_scalars {
            if let Some(fields) = base_query.output_fields.as_mut() {
                if !fields.iter().any(|f| f == &group_by_field) {
                    fields.push(group_by_field.clone());
                    added_group_by_field = true;
                }
            }
        }

        let original_include_doc_id = base_query.include_doc_id;
        let user_wants_global_doc_id = original_output_fields
            .as_ref()
            .is_some_and(|v| v.iter().any(|f| f == SYS_GLOBAL_DOC_ID));
        let expose_doc_id_member = original_include_doc_id || user_wants_global_doc_id;
        if kind.needs_doc_id() {
            base_query.include_doc_id = true;
        }

        Ok((
            Self {
                group_by_field,
                kind,
                docs_per_group,
                group_limit,
                original_output_fields,
                added_group_by_field,
                expose_doc_id_member,
            },
            base_query,
        ))
    }

    pub(crate) fn needs_row_locator(&self) -> bool {
        matches!(self.kind, GroupByKind::SystemRowId)
    }

    pub(crate) fn build_results(
        &self,
        results: Vec<Arc<Doc>>,
        row_locator: Option<&RowLocator>,
    ) -> ZResult<Vec<GroupResult>> {
        let mut groups: HashMap<String, (Value, Vec<Doc>)> = HashMap::new();
        let mut order: Vec<String> = Vec::new();
        let mut full_groups = 0usize;
        let should_strip_group_by_field = self.should_strip_group_by_field();

        for doc in results.into_iter() {
            let mut d = (*doc).clone();
            let group_val = self.group_value(&d, row_locator)?;

            if group_val.is_null() {
                continue;
            }

            if should_strip_group_by_field {
                d.fields.remove(&self.group_by_field);
            }
            if !self.expose_doc_id_member {
                d.doc_id = 0;
            }

            // Signed zeros compare equal in filters, so they share one group key.
            let key = match group_val {
                Value::F16(v) if v.to_f32() == 0.0 => format!("{:?}", Value::F16(f16::ZERO)),
                Value::F32(0.0) => format!("{:?}", Value::F32(0.0)),
                Value::F64(0.0) => format!("{:?}", Value::F64(0.0)),
                _ => format!("{:?}", &group_val),
            };
            if !groups.contains_key(&key) {
                if order.len() >= self.group_limit {
                    continue;
                }
                order.push(key.clone());
                groups.insert(key.clone(), (group_val, Vec::new()));
            }

            let Some((_, docs)) = groups.get_mut(&key) else {
                continue;
            };
            if docs.len() < self.docs_per_group {
                docs.push(d);
                if docs.len() == self.docs_per_group {
                    full_groups += 1;
                }
            }

            if order.len() == self.group_limit && full_groups == self.group_limit {
                break;
            }
        }

        let mut output: Vec<GroupResult> = Vec::new();
        for key in order {
            if let Some((group_value, docs)) = groups.remove(&key) {
                output.push(GroupResult { group_value, docs });
            }
        }

        Ok(output)
    }

    pub(crate) fn is_filled(&self, groups: &[GroupResult]) -> bool {
        groups.len() == self.group_limit
            && groups.iter().all(|g| g.docs.len() == self.docs_per_group)
    }

    fn group_value(&self, doc: &Doc, row_locator: Option<&RowLocator>) -> ZResult<Value> {
        match self.kind {
            GroupByKind::SystemUid => Ok(Value::String(doc.pk.clone())),
            GroupByKind::SystemGlobalDocId => Ok(Value::U64(doc.doc_id)),
            GroupByKind::SystemRowId => Ok(Value::U64(resolve_row_id(row_locator, doc.doc_id)?)),
            GroupByKind::SchemaScalar => Ok(doc
                .fields
                .get(&self.group_by_field)
                .cloned()
                .unwrap_or(Value::Null)),
        }
    }

    fn should_strip_group_by_field(&self) -> bool {
        let user_selected_group_by_field = self
            .original_output_fields
            .as_ref()
            .is_some_and(|v| v.iter().any(|f| f == &self.group_by_field));
        self.added_group_by_field && !user_selected_group_by_field
    }
}

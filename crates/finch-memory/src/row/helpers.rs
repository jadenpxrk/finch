use super::*;

pub(super) fn enum_name<T: Serialize>(value: T) -> Result<String, serde_json::Error> {
    let value = serde_json::to_value(value)?;
    Ok(value.as_str().unwrap_or_default().to_string())
}

pub(super) fn json_string<T: Serialize>(value: &T) -> Result<Value, serde_json::Error> {
    Ok(Value::String(serde_json::to_string(value)?))
}

pub(super) fn opt_string(value: &Option<String>) -> Value {
    value
        .as_ref()
        .map(|v| Value::String(v.clone()))
        .unwrap_or(Value::Null)
}

pub(super) fn opt_i64(value: Option<i64>) -> Value {
    value.map(Value::I64).unwrap_or(Value::Null)
}

pub(super) fn opt_f32(value: Option<f32>) -> Value {
    value.map(Value::F32).unwrap_or(Value::Null)
}

pub(super) fn row_error(message: impl Into<String>) -> Status {
    Status::invalid_argument(format!("memory row decode failed: {}", message.into()))
}

pub(super) fn required<'a>(doc: &'a Doc, field: &str) -> ZResult<&'a Value> {
    doc.fields
        .get(field)
        .ok_or_else(|| row_error(format!("missing `{field}`")))
}

pub(super) fn required_string(doc: &Doc, field: &str) -> ZResult<String> {
    required(doc, field)?
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| row_error(format!("`{field}` is not a string")))
}

pub(super) fn optional_string(doc: &Doc, field: &str) -> ZResult<Option<String>> {
    match doc.fields.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_str()
            .map(|v| Some(v.to_string()))
            .ok_or_else(|| row_error(format!("`{field}` is not a string or null"))),
    }
}

pub(super) fn required_i64(doc: &Doc, field: &str) -> ZResult<i64> {
    required(doc, field)?
        .as_i64()
        .ok_or_else(|| row_error(format!("`{field}` is not an integer")))
}

pub(super) fn optional_i64(doc: &Doc, field: &str) -> ZResult<Option<i64>> {
    match doc.fields.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_i64()
            .map(Some)
            .ok_or_else(|| row_error(format!("`{field}` is not an integer or null"))),
    }
}

pub(super) fn optional_f32(doc: &Doc, field: &str) -> ZResult<Option<f32>> {
    match doc.fields.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_f32()
            .map(Some)
            .ok_or_else(|| row_error(format!("`{field}` is not a float or null"))),
    }
}

pub(super) fn json_vec<T: serde::de::DeserializeOwned>(doc: &Doc, field: &str) -> ZResult<Vec<T>> {
    let text = required_string(doc, field)?;
    serde_json::from_str(&text).map_err(|err| row_error(format!("`{field}` JSON: {err}")))
}

pub(super) fn parse_status(value: &str) -> ZResult<MemoryStatus> {
    match value {
        "active" => Ok(MemoryStatus::Active),
        "superseded" => Ok(MemoryStatus::Superseded),
        "retracted" => Ok(MemoryStatus::Retracted),
        "tombstoned" => Ok(MemoryStatus::Tombstoned),
        "expired" => Ok(MemoryStatus::Expired),
        "redacted" => Ok(MemoryStatus::Redacted),
        other => Err(row_error(format!("unknown status `{other}`"))),
    }
}

pub(super) fn parse_rule_action(value: &str) -> ZResult<RuleAction> {
    match value {
        "derive_value" => Ok(RuleAction::DeriveValue),
        "mark_unsupported" => Ok(RuleAction::MarkUnsupported),
        other => Err(row_error(format!("unknown rule action `{other}`"))),
    }
}

pub(super) fn parse_rule_activation(
    value: Option<String>,
    action: RuleAction,
) -> ZResult<RuleActivation> {
    match value.as_deref() {
        None if matches!(action, RuleAction::DeriveValue) => {
            Ok(RuleActivation::ContinuousProjection)
        }
        None => Ok(RuleActivation::OnChange),
        Some("continuous_projection") => Ok(RuleActivation::ContinuousProjection),
        Some("on_change") => Ok(RuleActivation::OnChange),
        Some(other) => Err(row_error(format!("unknown rule activation `{other}`"))),
    }
}

pub(super) fn parse_rule_target_match(value: Option<String>) -> ZResult<RuleTargetMatch> {
    match value.as_deref().unwrap_or("exact_slot") {
        "exact_slot" => Ok(RuleTargetMatch::ExactSlot),
        "any_active_slot_for_subject" => Ok(RuleTargetMatch::AnyActiveSlotForSubject),
        other => Err(row_error(format!("unknown rule target_match `{other}`"))),
    }
}

pub(super) fn parse_rule_binding_status(
    value: Option<String>,
    slot_id: Option<&str>,
) -> ZResult<RuleBindingStatus> {
    match value.as_deref() {
        Some("pending") => Ok(RuleBindingStatus::Pending),
        Some("bound") => Ok(RuleBindingStatus::Bound),
        Some("ambiguous") => Ok(RuleBindingStatus::Ambiguous),
        Some("invalid") => Ok(RuleBindingStatus::Invalid),
        None if slot_id.is_some() => Ok(RuleBindingStatus::Bound),
        None => Ok(RuleBindingStatus::Pending),
        Some(other) => Err(row_error(format!("unknown rule binding status `{other}`"))),
    }
}

pub(super) fn parse_state_record_kind(value: &str) -> ZResult<StateRecordKind> {
    match value {
        "current" => Ok(StateRecordKind::Current),
        "set" => Ok(StateRecordKind::Set),
        "tombstone" => Ok(StateRecordKind::Tombstone),
        "unsupported" => Ok(StateRecordKind::Unsupported),
        "derived" => Ok(StateRecordKind::Derived),
        "rule" => Ok(StateRecordKind::Rule),
        other => Err(row_error(format!("unknown state kind `{other}`"))),
    }
}

pub(super) fn parse_visibility(value: &str) -> ZResult<Visibility> {
    match value {
        "private" => Ok(Visibility::Private),
        "shared" => Ok(Visibility::Shared),
        "system" => Ok(Visibility::System),
        "tool" => Ok(Visibility::Tool),
        other => Err(row_error(format!("unknown visibility `{other}`"))),
    }
}

pub(super) fn parse_source_type(value: &str) -> ZResult<SourceType> {
    match value {
        "episode" => Ok(SourceType::Episode),
        "artifact" => Ok(SourceType::Artifact),
        other => Err(row_error(format!("unknown source_type `{other}`"))),
    }
}

pub(super) fn parse_enum<T: serde::de::DeserializeOwned>(field: &str, value: &str) -> ZResult<T> {
    serde_json::from_value(serde_json::Value::String(value.to_string()))
        .map_err(|err| row_error(format!("`{field}` enum: {err}")))
}

pub(super) fn set_scope(
    doc: Doc,
    scope: &MemoryScope,
    visibility: Visibility,
    policy_tags: &[String],
) -> Result<Doc, serde_json::Error> {
    Ok(doc
        .set("space_id", Value::String(scope.space_id.clone()))
        .set("tenant_id", opt_string(&scope.tenant_id))
        .set("user_id", opt_string(&scope.user_id))
        .set("agent_id", opt_string(&scope.agent_id))
        .set("project_id", opt_string(&scope.project_id))
        .set("thread_id", opt_string(&scope.thread_id))
        .set("visibility", Value::String(visibility.as_str().to_string()))
        .set("policy_tags_json", json_string(&policy_tags)?))
}

pub(super) fn set_status(doc: Doc, status: MemoryStatus) -> Doc {
    doc.set("status", Value::String(status.as_str().to_string()))
}

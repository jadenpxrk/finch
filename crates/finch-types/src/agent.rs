use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Stable identifier for an agent session.
pub type SessionId = String;

/// Shared runtime context for Finch-native agent features.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AgentContext {
    pub session_id: SessionId,
    pub working_dir: Option<String>,
    pub variables: HashMap<String, ContextValue>,
    pub permissions: AgentPermissions,
    pub budget: OperationBudget,
    pub tool_registry: Vec<ToolDefinition>,
}

/// Serializable value type for agent/session variables.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum ContextValue {
    String(String),
    Integer(i64),
    Float(f64),
    Bool(bool),
    List(Vec<ContextValue>),
    Object(HashMap<String, ContextValue>),
    #[default]
    Null,
}

/// Runtime permission envelope for local agent execution.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentPermissions {
    pub filesystem: FsPermissions,
    pub database: DbPermissions,
    pub network: NetworkPermissions,
    pub calculator: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FsPermissions {
    pub read: bool,
    pub write: bool,
    pub mkdir: bool,
    pub delete: bool,
    pub allowed_paths: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DbPermissions {
    pub read: bool,
    pub write: bool,
    pub create: bool,
    pub drop: bool,
    pub allowed_collections: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NetworkPermissions {
    pub http: bool,
    pub allowed_domains: Vec<String>,
}

/// Budget envelope for a session or operation.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OperationBudget {
    pub max_tokens: Option<u64>,
    pub max_operations: Option<u64>,
    pub max_cost_micros: Option<u64>,
    pub tokens_used: u64,
    pub operations_used: u64,
    pub cost_micros_used: u64,
}

impl OperationBudget {
    pub fn consume(&mut self, tokens: u64, cost_micros: u64) {
        self.tokens_used = self.tokens_used.saturating_add(tokens);
        self.operations_used = self.operations_used.saturating_add(1);
        self.cost_micros_used = self.cost_micros_used.saturating_add(cost_micros);
    }

    pub fn remaining_tokens(&self) -> Option<u64> {
        self.max_tokens
            .map(|max| max.saturating_sub(self.tokens_used))
    }
}

/// Tool descriptor for local Finch agent runtimes.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters_schema: Option<String>,
    pub requires_confirmation: bool,
    pub tags: Vec<String>,
}

/// Record of a tool invocation.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ToolCallRecord {
    pub call_id: String,
    pub session_id: SessionId,
    pub tool_name: String,
    pub arguments_json: String,
    pub result_json: Option<String>,
    pub error: Option<String>,
    pub started_at_unix_ms: u64,
    pub finished_at_unix_ms: Option<u64>,
}

/// Audit entry for agent-visible side effects.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AuditEntry {
    pub timestamp_unix_ms: u64,
    pub operation: AuditOperation,
    pub resource: String,
    pub result: AuditResult,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditOperation {
    FsRead,
    FsWrite,
    FsMkdir,
    FsDelete,
    FsList,
    DbQuery,
    DbInsert,
    DbUpdate,
    DbDelete,
    #[default]
    ToolCall,
    VariableGet,
    VariableSet,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", content = "detail", rename_all = "snake_case")]
pub enum AuditResult {
    #[default]
    Success,
    Error(String),
    Denied(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operation_budget_consumes_safely() {
        let mut budget = OperationBudget {
            max_tokens: Some(100),
            ..Default::default()
        };

        budget.consume(30, 500);
        budget.consume(50, 250);

        assert_eq!(budget.tokens_used, 80);
        assert_eq!(budget.operations_used, 2);
        assert_eq!(budget.cost_micros_used, 750);
        assert_eq!(budget.remaining_tokens(), Some(20));
    }

    #[test]
    fn context_value_object_round_trip_shape() {
        let mut inner = HashMap::new();
        inner.insert(
            "mode".to_string(),
            ContextValue::String("local".to_string()),
        );

        let value = ContextValue::Object(inner);
        let encoded = serde_json::to_string(&value).unwrap();
        let decoded: ContextValue = serde_json::from_str(&encoded).unwrap();

        assert_eq!(decoded, value);
    }
}

//! System/internal column names.
//!
//! Finch supports a small set of "system columns" that can be referenced in
//! filters/queries:
//! - `_finch_uid_`      : user primary key (string)
//! - `_finch_row_id_`   : segment-local row id (uint64)
//! - `_finch_g_doc_id_` : global doc id (uint64)
//!
//! Finch stores the underlying data in Arrow IPC with its own internal columns
//! (e.g. `__doc_id__`, `__pk__`). These names must never be user-defined fields.
//!
//! Finch also uses several internal columns within its SQL engine / planner
//! record batches. These are not part of the public schema surface, but must
//! still be reserved to avoid collisions:
//! - `_finch_vector`
//! - `_finch_sindices`
//! - `_finch_svalues`
//! - `_finch_is_valid`
//! - `_finch_group_id`

/// System: document primary key (string).
pub const SYS_USER_ID: &str = "_finch_uid_";
/// System: segment-local row id (uint64).
pub const SYS_LOCAL_ROW_ID: &str = "_finch_row_id_";
/// System: global document id (uint64).
pub const SYS_GLOBAL_DOC_ID: &str = "_finch_g_doc_id_";
/// SQL engine internal: vector distance/similarity score (float32).
///
/// This is not a forward-store column and is produced by vector recall.
pub const SYS_SCORE: &str = "_finch_score";

/// SQL engine internal: materialized vector payload column name.
pub(crate) const SYS_INTERNAL_VECTOR: &str = "_finch_vector";
/// SQL engine internal: sparse indices payload column name.
pub(crate) const SYS_INTERNAL_SPARSE_INDICES: &str = "_finch_sindices";
/// SQL engine internal: sparse values payload column name.
pub(crate) const SYS_INTERNAL_SPARSE_VALUES: &str = "_finch_svalues";
/// SQL engine internal: per-row validity marker column name.
pub(crate) const SYS_INTERNAL_IS_VALID: &str = "_finch_is_valid";
/// SQL engine internal: group-by group id column name.
pub(crate) const SYS_INTERNAL_GROUP_ID: &str = "_finch_group_id";

/// Finch Arrow IPC: global doc id column (uint64).
pub const FINCH_IPC_DOC_ID: &str = "__doc_id__";

pub(crate) fn is_system_reserved_column(name: &str) -> bool {
    matches!(
        name,
        SYS_USER_ID
            | SYS_LOCAL_ROW_ID
            | SYS_GLOBAL_DOC_ID
            | SYS_SCORE
            | SYS_INTERNAL_VECTOR
            | SYS_INTERNAL_SPARSE_INDICES
            | SYS_INTERNAL_SPARSE_VALUES
            | SYS_INTERNAL_IS_VALID
            | SYS_INTERNAL_GROUP_ID
    )
}

/// Whether `name` is the user id, local row id, or global doc id column.
pub fn is_system_column(name: &str) -> bool {
    matches!(name, SYS_USER_ID | SYS_LOCAL_ROW_ID | SYS_GLOBAL_DOC_ID)
}

pub(crate) fn is_reserved_field_name(name: &str) -> bool {
    // Finch reserves `__*` for internal Arrow storage columns and vector payload
    // columns (e.g. `__vec__{field}`).
    if name.starts_with("__") {
        return true;
    }
    is_system_reserved_column(name)
}

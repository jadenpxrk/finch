use crate::doc::{docobject_to_doc, DocObject, GroupResultObject, VectorQueryOptions};
use crate::schema::FieldSchemaOptions;
use crate::{finch_err, BlockingTask};
use finch_db::Collection as RustCollection;
use finch_types::ZResult;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::sync::Arc;

fn parse_file_format(s: &str) -> Result<finch_types::FileFormat> {
    let v = s.trim().to_ascii_lowercase();
    match v.as_str() {
        "arrow" | "arrow_ipc" | "ipc" => Ok(finch_types::FileFormat::ArrowIpc),
        "parquet" => Ok(finch_types::FileFormat::Parquet),
        "raw" => Ok(finch_types::FileFormat::Raw),
        "protobuf" | "proto" => Ok(finch_types::FileFormat::Protobuf),
        _ => Err(napi::Error::from_reason(format!(
            "invalid forward_file_format '{s}' (expected arrow_ipc|parquet|raw|protobuf)"
        ))),
    }
}

fn parse_storage_type(s: &str) -> Result<finch_types::StorageType> {
    let v = s.trim().to_ascii_lowercase();
    match v.as_str() {
        "none" => Ok(finch_types::StorageType::None),
        "mmap" => Ok(finch_types::StorageType::Mmap),
        "memory" => Ok(finch_types::StorageType::Memory),
        "buffer_pool" | "bufferpool" => Ok(finch_types::StorageType::BufferPool),
        _ => Err(napi::Error::from_reason(format!(
            "invalid storage type '{s}' (expected none|mmap|memory|buffer_pool)"
        ))),
    }
}

fn collection_options_from_node_args(
    read_only: Option<bool>,
    enable_mmap: Option<bool>,
    max_buffer_size: Option<u32>,
    forward_file_format: Option<String>,
    index_storage: Option<String>,
    forward_storage: Option<String>,
) -> Result<finch_types::CollectionOptions> {
    Ok(finch_types::CollectionOptions {
        read_only: read_only.unwrap_or(false),
        enable_mmap: enable_mmap.unwrap_or(true),
        index_storage: index_storage
            .as_deref()
            .map(parse_storage_type)
            .transpose()?,
        forward_storage: forward_storage
            .as_deref()
            .map(parse_storage_type)
            .transpose()?,
        max_buffer_size: max_buffer_size
            .unwrap_or(finch_types::CollectionOptions::DEFAULT_MAX_BUFFER_SIZE),
        forward_file_format: forward_file_format
            .as_deref()
            .map(parse_file_format)
            .transpose()?,
    })
}

#[napi]
pub struct Collection {
    inner: Option<Arc<RustCollection>>,
}

#[napi(object)]
pub struct StatusObject {
    pub ok: bool,
    pub code: u32,
    pub message: String,
}

impl From<finch_types::Status> for StatusObject {
    fn from(status: finch_types::Status) -> Self {
        Self {
            ok: status.is_ok(),
            code: status.code as u32,
            message: status.message,
        }
    }
}

impl Collection {
    fn inner_arc(&self) -> Result<&Arc<RustCollection>> {
        self.inner
            .as_ref()
            .ok_or_else(|| napi::Error::from_reason("Collection is closed/destroyed"))
    }

    /// Runs `work` off the JS thread; the returned promise settles with its result.
    fn spawn<T: ToNapiValue + TypeName + Send + 'static>(
        &self,
        work: impl FnOnce(&RustCollection) -> ZResult<T> + Send + 'static,
    ) -> Result<AsyncTask<BlockingTask<T>>> {
        let inner = self.inner_arc()?.clone();
        Ok(BlockingTask::spawn(move || work(&inner)))
    }

    fn to_docs(&self, docs: Vec<DocObject>) -> Result<Vec<finch_types::Doc>> {
        let schema = self.inner_arc()?.schema_info();
        docs.into_iter()
            .map(|d| docobject_to_doc(d, &schema))
            .collect()
    }
}

fn statuses(results: Vec<finch_types::Status>) -> Vec<StatusObject> {
    results.into_iter().map(StatusObject::from).collect()
}

#[napi]
impl Collection {
    #[napi]
    pub fn insert(
        &self,
        docs: Vec<DocObject>,
    ) -> Result<AsyncTask<BlockingTask<Vec<StatusObject>>>> {
        let docs = self.to_docs(docs)?;
        self.spawn(move |c| c.insert(docs).map(statuses))
    }

    #[napi]
    pub fn upsert(
        &self,
        docs: Vec<DocObject>,
    ) -> Result<AsyncTask<BlockingTask<Vec<StatusObject>>>> {
        let docs = self.to_docs(docs)?;
        self.spawn(move |c| c.upsert(docs).map(statuses))
    }

    #[napi]
    pub fn update(
        &self,
        docs: Vec<DocObject>,
    ) -> Result<AsyncTask<BlockingTask<Vec<StatusObject>>>> {
        let docs = self.to_docs(docs)?;
        self.spawn(move |c| c.update(docs).map(statuses))
    }

    #[napi]
    pub fn delete(&self, pks: Vec<String>) -> Result<AsyncTask<BlockingTask<Vec<StatusObject>>>> {
        self.spawn(move |c| c.delete(pks).map(statuses))
    }

    #[napi]
    pub fn delete_by_filter(
        &self,
        filter: String,
    ) -> Result<AsyncTask<BlockingTask<StatusObject>>> {
        self.spawn(move |c| c.delete_by_filter(&filter).map(StatusObject::from))
    }

    #[napi]
    pub fn query(&self, env: Env, opts: VectorQueryOptions) -> Result<Vec<DocObject>> {
        let query = finch_types::VectorQuery::try_from(opts)?;
        let results = self
            .inner_arc()?
            .query(query)
            .map_err(|e| finch_err(&env, e))?;
        Ok(results
            .into_iter()
            .map(|doc| DocObject::from((*doc).clone()))
            .collect())
    }

    #[napi]
    pub fn query_sql(&self, env: Env, sql: String) -> Result<Vec<DocObject>> {
        let results = self
            .inner_arc()?
            .query_sql(&sql)
            .map_err(|e| finch_err(&env, e))?;
        Ok(results
            .into_iter()
            .map(|doc| DocObject::from((*doc).clone()))
            .collect())
    }

    #[napi]
    pub fn fetch(
        &self,
        env: Env,
        pks: Vec<String>,
    ) -> Result<std::collections::HashMap<String, DocObject>> {
        let results = self
            .inner_arc()?
            .fetch(pks)
            .map_err(|e| finch_err(&env, e))?;
        Ok(results
            .into_iter()
            .map(|(pk, doc)| (pk, DocObject::from((*doc).clone())))
            .collect())
    }

    #[napi]
    pub fn group_by_query(
        &self,
        env: Env,
        opts: VectorQueryOptions,
        group_by_field: String,
        group_count: u32,
        group_topk: u32,
    ) -> Result<Vec<GroupResultObject>> {
        let base = finch_types::VectorQuery::try_from(opts)?;
        let gbq = finch_types::GroupByVectorQuery {
            base,
            group_by_field,
            group_count: group_count as usize,
            group_topk: group_topk as usize,
        };
        let results = self
            .inner_arc()?
            .group_by_query(gbq)
            .map_err(|e| finch_err(&env, e))?;
        Ok(results.into_iter().map(GroupResultObject::from).collect())
    }

    #[napi]
    pub fn create_index(
        &self,
        field: String,
        field_opts: FieldSchemaOptions,
        rebuild: Option<bool>,
        concurrency: Option<u32>,
    ) -> Result<AsyncTask<BlockingTask<()>>> {
        let field_schema = finch_types::FieldSchema::try_from(field_opts)?;
        let params = field_schema.index_params.ok_or_else(|| {
            napi::Error::from_reason("FieldSchemaOptions must carry an index param variant")
        })?;
        let options = finch_types::CreateIndexOptions {
            rebuild: rebuild.unwrap_or(false),
            concurrency: concurrency.map(|n| n as usize),
        };
        self.spawn(move |c| c.create_index(&field, params, options))
    }

    #[napi]
    pub fn drop_index(&self, field: String) -> Result<AsyncTask<BlockingTask<()>>> {
        self.spawn(move |c| c.drop_index(&field))
    }

    #[napi]
    pub fn add_column(
        &self,
        field_opts: FieldSchemaOptions,
        rebuild_index: Option<bool>,
        concurrency: Option<u32>,
        expression: Option<String>,
    ) -> Result<AsyncTask<BlockingTask<()>>> {
        let field_schema = finch_types::FieldSchema::try_from(field_opts)?;
        let options = finch_types::AddColumnOptions {
            rebuild_index: rebuild_index.unwrap_or(false),
            concurrency: concurrency.map(|n| n as usize),
        };
        self.spawn(move |c| {
            c.add_column_with_expression(field_schema, expression.as_deref(), options)
        })
    }

    #[napi]
    pub fn drop_column(&self, field: String) -> Result<AsyncTask<BlockingTask<()>>> {
        self.spawn(move |c| c.drop_column(&field))
    }

    #[napi]
    pub fn alter_column(
        &self,
        field: String,
        rename_to: Option<String>,
        field_schema: Option<FieldSchemaOptions>,
        rebuild_index: Option<bool>,
        concurrency: Option<u32>,
    ) -> Result<AsyncTask<BlockingTask<()>>> {
        let field_schema = field_schema
            .map(finch_types::FieldSchema::try_from)
            .transpose()?;
        let options = finch_types::AlterColumnOptions {
            rebuild_index: rebuild_index.unwrap_or(false),
            concurrency: concurrency.map(|n| n as usize),
        };
        self.spawn(move |c| c.alter_column(&field, rename_to.as_deref(), field_schema, options))
    }

    #[napi]
    pub fn optimize(
        &self,
        max_segments: Option<u32>,
        concurrency: Option<u32>,
    ) -> Result<AsyncTask<BlockingTask<()>>> {
        let options = finch_types::OptimizeOptions {
            max_segments: max_segments.map(|n| n as usize),
            concurrency: concurrency.map(|n| n as usize),
            ..Default::default()
        };
        self.spawn(move |c| c.optimize(options))
    }

    #[napi]
    pub fn flush(&self) -> Result<AsyncTask<BlockingTask<()>>> {
        self.spawn(|c| c.flush())
    }

    #[napi]
    pub fn stats(&self, env: Env) -> Result<std::collections::HashMap<String, f64>> {
        let stats = self.inner_arc()?.stats().map_err(|e| finch_err(&env, e))?;
        let mut m = std::collections::HashMap::new();
        m.insert("doc_count".to_string(), stats.doc_count as f64);
        m.insert("segment_count".to_string(), stats.segment_count as f64);
        Ok(m)
    }

    #[napi(js_name = "statsInfo")]
    pub fn stats_info(&self, env: Env) -> Result<serde_json::Value> {
        let stats = self.inner_arc()?.stats().map_err(|e| finch_err(&env, e))?;
        serde_json::to_value(&stats).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    #[napi]
    pub fn schema_info(&self) -> Result<serde_json::Value> {
        let schema = self.inner_arc()?.schema_info();
        serde_json::to_value(&schema).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    #[napi]
    pub fn path(&self) -> Result<String> {
        Ok(self.inner_arc()?.path_string())
    }

    #[napi]
    pub fn options(&self) -> Result<serde_json::Value> {
        let opts = self.inner_arc()?.options();
        serde_json::to_value(&opts).map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    #[napi]
    pub fn destroy(&mut self, env: Env) -> Result<()> {
        let arc = self
            .inner
            .take()
            .ok_or_else(|| napi::Error::from_reason("Collection is closed/destroyed"))?;
        match Arc::try_unwrap(arc) {
            Ok(col) => col.destroy().map_err(|e| finch_err(&env, e)),
            Err(arc) => {
                self.inner = Some(arc);
                Err(napi::Error::from_reason(
                    "Collection has other references; drop them before destroy",
                ))
            }
        }
    }
}

#[napi(js_name = "createAndOpen")]
#[cfg(not(test))]
pub fn create_and_open(
    path: String,
    schema: crate::schema::CollectionSchemaOptions,
    read_only: Option<bool>,
    enable_mmap: Option<bool>,
    max_buffer_size: Option<u32>,
    forward_file_format: Option<String>,
    index_storage: Option<String>,
    forward_storage: Option<String>,
) -> Result<AsyncTask<BlockingTask<Collection>>> {
    let options = collection_options_from_node_args(
        read_only,
        enable_mmap,
        max_buffer_size,
        forward_file_format,
        index_storage,
        forward_storage,
    )?;
    let rust_schema = finch_types::CollectionSchema::try_from(schema)?;
    Ok(BlockingTask::spawn(move || {
        let col =
            RustCollection::create_and_open(std::path::Path::new(&path), rust_schema, options)?;
        Ok(Collection { inner: Some(col) })
    }))
}

#[napi(js_name = "open")]
#[cfg(not(test))]
pub fn open(
    path: String,
    read_only: Option<bool>,
    enable_mmap: Option<bool>,
    max_buffer_size: Option<u32>,
    forward_file_format: Option<String>,
    index_storage: Option<String>,
    forward_storage: Option<String>,
) -> Result<AsyncTask<BlockingTask<Collection>>> {
    let options = collection_options_from_node_args(
        read_only,
        enable_mmap,
        max_buffer_size,
        forward_file_format,
        index_storage,
        forward_storage,
    )?;
    Ok(BlockingTask::spawn(move || {
        let col = RustCollection::open(std::path::Path::new(&path), options)?;
        Ok(Collection { inner: Some(col) })
    }))
}

#[napi(object)]
pub struct GlobalConfigOptions {
    pub memory_limit_bytes: Option<i64>,
    /// 0=Debug, 1=Info, 2=Warn, 3=Error, 4=Fatal
    pub log_level: Option<u32>,
    pub log_to_file: Option<bool>,
    pub log_dir: Option<String>,
    pub log_basename: Option<String>,
    pub log_file_size_mb: Option<u32>,
    pub log_overdue_days: Option<u32>,
    pub query_thread_count: Option<u32>,
    pub wal_flush_every_docs: Option<u32>,
    pub wal_fsync_every_docs: Option<u32>,
    pub invert_to_forward_scan_ratio: Option<f64>,
    pub brute_force_by_keys_ratio: Option<f64>,
    pub optimize_thread_count: Option<u32>,
}

fn global_config_from_node_options(
    opts: Option<GlobalConfigOptions>,
) -> Result<finch_types::GlobalConfigData> {
    let mut cfg = finch_types::GlobalConfigData::default();
    if let Some(opts) = opts {
        if let Some(v) = opts.memory_limit_bytes {
            if v < 0 {
                return Err(napi::Error::from_reason(
                    "memory_limit_bytes must be non-negative",
                ));
            }
            cfg.memory_limit_bytes = v as u64;
        }
        if let Some(v) = opts.log_level {
            cfg.log_level = match v {
                0 => finch_types::LogLevel::Debug,
                1 => finch_types::LogLevel::Info,
                2 => finch_types::LogLevel::Warn,
                3 => finch_types::LogLevel::Error,
                _ => finch_types::LogLevel::Fatal,
            };
        }
        if let Some(v) = opts.log_to_file {
            cfg.log_to_file = v;
        }
        if let Some(v) = opts.log_dir {
            cfg.log_dir = v;
        }
        if let Some(v) = opts.log_basename {
            cfg.log_basename = v;
        }
        if let Some(v) = opts.log_file_size_mb {
            cfg.log_file_size_mb = v;
        }
        if let Some(v) = opts.log_overdue_days {
            cfg.log_overdue_days = v;
        }
        if let Some(v) = opts.query_thread_count {
            cfg.query_thread_count = v;
        }
        if let Some(v) = opts.wal_flush_every_docs {
            cfg.wal_flush_every_docs = v;
        }
        if let Some(v) = opts.wal_fsync_every_docs {
            cfg.wal_fsync_every_docs = v;
        }
        if let Some(v) = opts.invert_to_forward_scan_ratio {
            cfg.invert_to_forward_scan_ratio = v as f32;
        }
        if let Some(v) = opts.brute_force_by_keys_ratio {
            cfg.brute_force_by_keys_ratio = v as f32;
        }
        if let Some(v) = opts.optimize_thread_count {
            cfg.optimize_thread_count = v;
        }
    }
    Ok(cfg)
}

#[napi(js_name = "initGlobalConfig")]
#[cfg(not(test))]
pub fn init_global_config(env: Env, opts: Option<GlobalConfigOptions>) -> Result<()> {
    let cfg = global_config_from_node_options(opts)?;
    finch_db::initialize_global_config(cfg).map_err(|e| finch_err(&env, e))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_object_preserves_status_shape() {
        let status = StatusObject::from(finch_types::Status::not_found("missing"));
        assert!(!status.ok);
        assert_eq!(status.code, finch_types::StatusCode::NotFound as u32);
        assert_eq!(status.message, "missing");
    }

    #[test]
    fn test_collection_options_from_node_args_parses_shared_options() {
        let options = collection_options_from_node_args(
            Some(true),
            Some(false),
            Some(123),
            Some("parquet".to_string()),
            Some("buffer_pool".to_string()),
            Some("memory".to_string()),
        )
        .unwrap();

        assert!(options.read_only);
        assert!(!options.enable_mmap);
        assert_eq!(options.max_buffer_size, 123);
        assert_eq!(
            options.forward_file_format,
            Some(finch_types::FileFormat::Parquet)
        );
        assert_eq!(
            options.index_storage,
            Some(finch_types::StorageType::BufferPool)
        );
        assert_eq!(
            options.forward_storage,
            Some(finch_types::StorageType::Memory)
        );
    }

    #[test]
    fn test_global_config_rejects_negative_memory_limit() {
        let err = global_config_from_node_options(Some(GlobalConfigOptions {
            memory_limit_bytes: Some(-1),
            log_level: None,
            log_to_file: None,
            log_dir: None,
            log_basename: None,
            log_file_size_mb: None,
            log_overdue_days: None,
            query_thread_count: None,
            wal_flush_every_docs: None,
            wal_fsync_every_docs: None,
            invert_to_forward_scan_ratio: None,
            brute_force_by_keys_ratio: None,
            optimize_thread_count: None,
        }))
        .unwrap_err();

        assert!(err.reason.contains("memory_limit_bytes"));
    }
}

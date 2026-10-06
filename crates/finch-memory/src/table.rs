//! The storage operations a memory store needs from each of its tables.

use finch_types::{Doc, HnswIndexParams, Status, VectorQuery, ZResult};
use std::collections::HashMap;
use std::sync::Arc;

pub(crate) trait MemoryTable: Send + Sync {
    /// One status per doc; a pk that already exists fails with AlreadyExists.
    fn insert(&self, docs: Vec<Doc>) -> ZResult<Vec<Status>>;
    /// Replaces each whole row, so later scans see it after every row written before it.
    fn upsert(&self, docs: Vec<Doc>) -> ZResult<Vec<Status>>;
    /// Every field of each existing pk, vectors included; missing pks are absent.
    fn fetch(&self, pks: Vec<String>) -> ZResult<HashMap<String, Arc<Doc>>>;
    /// Every row matching the filter, oldest write first; `topk` is not applied.
    fn scan_filter_only(&self, query: VectorQuery) -> ZResult<Vec<Arc<Doc>>>;
    /// The first `limit` rows `scan_filter_only` returns.
    fn scan_prefix(&self, query: VectorQuery, limit: usize) -> ZResult<Vec<Arc<Doc>>>;
    /// Nearest rows by cosine distance in `score`, or with no query vector the first `topk`
    /// matching rows, oldest write first.
    fn query(&self, query: VectorQuery) -> ZResult<Vec<Arc<Doc>>>;
    fn create_hnsw_index(
        &self,
        field: &str,
        params: HnswIndexParams,
        concurrency: Option<usize>,
    ) -> ZResult<()>;
}

impl<T: MemoryTable + ?Sized> MemoryTable for Arc<T> {
    fn insert(&self, docs: Vec<Doc>) -> ZResult<Vec<Status>> {
        (**self).insert(docs)
    }

    fn upsert(&self, docs: Vec<Doc>) -> ZResult<Vec<Status>> {
        (**self).upsert(docs)
    }

    fn fetch(&self, pks: Vec<String>) -> ZResult<HashMap<String, Arc<Doc>>> {
        (**self).fetch(pks)
    }

    fn scan_filter_only(&self, query: VectorQuery) -> ZResult<Vec<Arc<Doc>>> {
        (**self).scan_filter_only(query)
    }

    fn scan_prefix(&self, query: VectorQuery, limit: usize) -> ZResult<Vec<Arc<Doc>>> {
        (**self).scan_prefix(query, limit)
    }

    fn query(&self, query: VectorQuery) -> ZResult<Vec<Arc<Doc>>> {
        (**self).query(query)
    }

    fn create_hnsw_index(
        &self,
        field: &str,
        params: HnswIndexParams,
        concurrency: Option<usize>,
    ) -> ZResult<()> {
        (**self).create_hnsw_index(field, params, concurrency)
    }
}

thread_local! {
    // Rows of state-table reads made while this thread holds the outermost read of a store.
    static READ_CACHE: std::cell::RefCell<Option<HashMap<String, Vec<Arc<Doc>>>>> =
        const { std::cell::RefCell::new(None) };
}

/// Starts this thread's read cache; false when an outer read already started it.
pub(crate) fn begin_read_cache() -> bool {
    READ_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let starts = cache.is_none();
        if starts {
            *cache = Some(HashMap::new());
        }
        starts
    })
}

pub(crate) fn end_read_cache() {
    READ_CACHE.with(|cache| cache.borrow_mut().take());
}

/// A state table whose repeated identical reads during one store read come from memory. State
/// tables change only under the store's write lock, which a read excludes, and a write through
/// this table clears the cache.
pub(crate) struct ReadCachedTable {
    inner: Arc<dyn MemoryTable>,
    key: String,
}

impl ReadCachedTable {
    pub(crate) fn new(inner: Arc<dyn MemoryTable>, store_id: u64, name: &str) -> Self {
        Self {
            inner,
            key: format!("{store_id}/{name}"),
        }
    }

    fn cached(
        &self,
        operation: &str,
        request: &impl serde::Serialize,
        read: impl FnOnce() -> ZResult<Vec<Arc<Doc>>>,
    ) -> ZResult<Vec<Arc<Doc>>> {
        let key = match serde_json::to_string(request) {
            Ok(request) => format!("{}/{operation}/{request}", self.key),
            Err(_) => return read(),
        };
        if let Some(rows) = READ_CACHE.with(|cache| {
            cache
                .borrow()
                .as_ref()
                .and_then(|rows| rows.get(&key).cloned())
        }) {
            return Ok(rows);
        }
        let rows = read()?;
        READ_CACHE.with(|cache| {
            if let Some(cache) = cache.borrow_mut().as_mut() {
                cache.insert(key, rows.clone());
            }
        });
        Ok(rows)
    }

    fn clear() {
        READ_CACHE.with(|cache| {
            if let Some(cache) = cache.borrow_mut().as_mut() {
                cache.clear();
            }
        });
    }
}

#[cfg(test)]
thread_local! {
    // Test crash point: this many state writes succeed, then the next one panics before it runs.
    pub(crate) static STATE_WRITES_BEFORE_CRASH: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn count_state_write_toward_crash() {
    STATE_WRITES_BEFORE_CRASH.with(|remaining| match remaining.get() {
        Some(0) => {
            remaining.set(None);
            panic!("injected crash before a state write");
        }
        Some(writes) => remaining.set(Some(writes - 1)),
        None => {}
    });
}

impl MemoryTable for ReadCachedTable {
    fn insert(&self, docs: Vec<Doc>) -> ZResult<Vec<Status>> {
        #[cfg(test)]
        count_state_write_toward_crash();
        Self::clear();
        self.inner.insert(docs)
    }

    fn upsert(&self, docs: Vec<Doc>) -> ZResult<Vec<Status>> {
        #[cfg(test)]
        count_state_write_toward_crash();
        Self::clear();
        self.inner.upsert(docs)
    }

    fn fetch(&self, pks: Vec<String>) -> ZResult<HashMap<String, Arc<Doc>>> {
        let rows = self.cached("fetch", &pks, || {
            Ok(self.inner.fetch(pks.clone())?.into_values().collect())
        })?;
        Ok(rows.into_iter().map(|doc| (doc.pk.clone(), doc)).collect())
    }

    fn scan_filter_only(&self, query: VectorQuery) -> ZResult<Vec<Arc<Doc>>> {
        self.cached("scan", &query, || {
            self.inner.scan_filter_only(query.clone())
        })
    }

    fn scan_prefix(&self, query: VectorQuery, limit: usize) -> ZResult<Vec<Arc<Doc>>> {
        self.cached("prefix", &(&query, limit), || {
            self.inner.scan_prefix(query.clone(), limit)
        })
    }

    fn query(&self, query: VectorQuery) -> ZResult<Vec<Arc<Doc>>> {
        self.cached("query", &query, || self.inner.query(query.clone()))
    }

    fn create_hnsw_index(
        &self,
        field: &str,
        params: HnswIndexParams,
        concurrency: Option<usize>,
    ) -> ZResult<()> {
        self.inner.create_hnsw_index(field, params, concurrency)
    }
}

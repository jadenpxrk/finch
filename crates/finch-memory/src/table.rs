//! The storage operations a memory store needs from each of its tables.

use finch_db::Collection;
use finch_types::{
    CreateIndexOptions, Doc, HnswIndexParams, IndexParams, Status, VectorQuery, ZResult,
};
use std::collections::HashMap;
use std::sync::Arc;

pub trait MemoryTable: Send + Sync {
    /// One status per doc; a pk that already exists fails with AlreadyExists.
    fn insert(&self, docs: Vec<Doc>) -> ZResult<Vec<Status>>;
    /// Replaces each whole row, so later scans see it after every row written before it.
    fn upsert(&self, docs: Vec<Doc>) -> ZResult<Vec<Status>>;
    /// One status per pk; a missing pk is NotFound.
    fn delete(&self, pks: Vec<String>) -> ZResult<Vec<Status>>;
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
    fn read_only(&self) -> bool;
}

impl MemoryTable for Collection {
    fn insert(&self, docs: Vec<Doc>) -> ZResult<Vec<Status>> {
        Collection::insert(self, docs)
    }

    fn upsert(&self, docs: Vec<Doc>) -> ZResult<Vec<Status>> {
        Collection::upsert(self, docs)
    }

    fn delete(&self, pks: Vec<String>) -> ZResult<Vec<Status>> {
        Collection::delete(self, pks)
    }

    fn fetch(&self, pks: Vec<String>) -> ZResult<HashMap<String, Arc<Doc>>> {
        Collection::fetch(self, pks)
    }

    fn scan_filter_only(&self, query: VectorQuery) -> ZResult<Vec<Arc<Doc>>> {
        Collection::scan_filter_only(self, query)
    }

    // A collection cannot stop a filter scan early, so it scans and truncates.
    fn scan_prefix(&self, query: VectorQuery, limit: usize) -> ZResult<Vec<Arc<Doc>>> {
        let mut docs = Collection::scan_filter_only(self, query)?;
        docs.truncate(limit);
        Ok(docs)
    }

    fn query(&self, query: VectorQuery) -> ZResult<Vec<Arc<Doc>>> {
        Collection::query(self, query)
    }

    fn create_hnsw_index(
        &self,
        field: &str,
        params: HnswIndexParams,
        concurrency: Option<usize>,
    ) -> ZResult<()> {
        self.create_index(
            field,
            IndexParams::Hnsw(params),
            CreateIndexOptions {
                rebuild: false,
                concurrency,
            },
        )
    }

    fn read_only(&self) -> bool {
        self.options().read_only
    }
}

impl<T: MemoryTable + ?Sized> MemoryTable for Arc<T> {
    fn insert(&self, docs: Vec<Doc>) -> ZResult<Vec<Status>> {
        (**self).insert(docs)
    }

    fn upsert(&self, docs: Vec<Doc>) -> ZResult<Vec<Status>> {
        (**self).upsert(docs)
    }

    fn delete(&self, pks: Vec<String>) -> ZResult<Vec<Status>> {
        (**self).delete(pks)
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

    fn read_only(&self) -> bool {
        (**self).read_only()
    }
}

/// A store whose tables commit together runs each state mutation as one transaction.
pub trait MemoryTransactions: Send + Sync {
    /// Starts a transaction that the calling thread's table operations join.
    fn begin(&self) -> ZResult<()>;
    fn commit(&self) -> ZResult<()>;
    /// Discards the open transaction, if there is one.
    fn rollback(&self) -> ZResult<()>;
}

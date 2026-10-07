//! Memory stores in Postgres with pgvector: one schema per store, one table per collection, and
//! each state mutation in one transaction.

use crate::row::{span_from_doc, state_record_from_doc};
use crate::schema::{span_schema, CLAIMS_COLLECTION, SPANS_COLLECTION, STATE_RECORDS_COLLECTION};
use crate::store::state_records::state_lexical_terms;
use crate::store::{
    collection_schemas, span_term_counts, take_schema, MemoryStore, OPEN_JOBS_INDEX_DDL,
};
use crate::table::MemoryTable;
use finch_db::sqlengine::{parse_filter, FilterExpr};
use finch_types::{
    CollectionSchema, CompareOp, DataType, Doc, HnswIndexParams, Status, Value, VectorQuery,
    ZResult,
};
use parking_lot::{Condvar, Mutex};
use pgvector::Vector;
use postgres::types::ToSql;
use postgres::{Client, NoTls, Row};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;
use std::thread::{self, ThreadId};

type Param = Box<dyn ToSql + Sync>;
type Migration = fn(&Arc<PgSession>, usize) -> ZResult<()>;

/// Where a memory store lives in Postgres, and how many connections it may hold at once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PostgresStore {
    /// Connection URL of the server, for example `postgres://user:password@host:5432/db`.
    pub url: String,
    /// Schema that holds the store: 1 to 63 characters of a-z, 0-9, or _.
    pub name: String,
    /// Most connections the store holds at once. Each read or write holds one until it ends.
    pub max_connections: usize,
    /// When true, Postgres row-level security shows each operation only the rows of its scope's
    /// tenant, and an operation without a tenant fails. The role must not bypass row-level
    /// security.
    pub tenant_isolation: bool,
}

impl PostgresStore {
    /// The store `name` at `url`, holding at most 8 connections.
    pub fn new(url: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            name: name.into(),
            max_connections: 8,
            tenant_isolation: false,
        }
    }
}

/// Migration `i` changes a store of schema version `i` to version `i + 1`. The current version is
/// the number of migrations.
const MIGRATIONS: &[Migration] = &[migrate_to_1];
const SCHEMA_VERSION: usize = MIGRATIONS.len();

impl MemoryStore {
    /// Creates a store in the new schema `store.name`; fails if that schema exists.
    /// `vector_index` builds an HNSW index on each embedding column; without it searches are
    /// exact.
    pub fn create(
        store: &PostgresStore,
        embedding_dim: usize,
        vector_index: bool,
    ) -> ZResult<Self> {
        let name = &store.name;
        let session = Arc::new(PgSession::connect(store)?);
        let created = session.with_client(|client| {
            client.batch_execute(&format!(
                "CREATE SCHEMA {schema}; CREATE SEQUENCE {schema}.write_seq;",
                schema = quote(name)
            ))
        });
        if let Err(error) = created {
            if session.schema_exists()? {
                return Err(Status::already_exists(format!(
                    "postgres store {name} already exists"
                )));
            }
            return Err(error);
        }
        let mut schemas = collection_schemas(embedding_dim, span_schema(embedding_dim));
        let store = Self::from_tables(session.clone(), |table| {
            let table = PgTable::new(session.clone(), take_schema(&mut schemas, table)?)?;
            table.create(vector_index)?;
            Ok(Arc::new(table))
        })?;
        session.with_client(|client| client.batch_execute(OPEN_JOBS_INDEX_DDL))?;
        session.with_client(|client| write_schema_version(client, SCHEMA_VERSION))?;
        prepare_schema(&session, embedding_dim)?;
        Ok(store)
    }

    /// Opens the store in the schema `store.name`. A store of an older schema version changes
    /// to the current version in one transaction. A store of a newer version does not open.
    pub fn open(store: &PostgresStore) -> ZResult<Self> {
        let name = &store.name;
        let session = Arc::new(PgSession::connect(store)?);
        if !session.schema_exists()? {
            return Err(Status::not_found(format!(
                "postgres store {name} does not exist"
            )));
        }
        let embedding_dim = session.embedding_dim()?;
        prepare_schema(&session, embedding_dim)?;
        let mut schemas = collection_schemas(embedding_dim, span_schema(embedding_dim));
        Self::from_tables(session.clone(), |table| {
            let table = PgTable::new(session.clone(), take_schema(&mut schemas, table)?)?;
            table.load_hnsw_fields()?;
            Ok(Arc::new(table))
        })
    }
}

/// Runs the migrations from the store's schema version to `SCHEMA_VERSION`, and turns on tenant
/// isolation if the session has it. An advisory lock on the store makes concurrent opens wait.
fn prepare_schema(session: &Arc<PgSession>, embedding_dim: usize) -> ZResult<()> {
    let lock = crate::ingest::stable_hash_hex(&["finch_schema", &session.schema]);
    let lock = u64::from_str_radix(&lock, 16).map_or(0, |key| key as i64);
    session.begin_write(Some((&[], lock)), None)?;
    let prepared = (|| {
        let version = session.with_client(|client| {
            let table = client.query_opt(
                "SELECT 1 FROM pg_tables \
                 WHERE schemaname = current_schema() AND tablename = 'finch_meta'",
                &[],
            )?;
            match table {
                Some(_) => client
                    .query_opt("SELECT schema_version FROM finch_meta", &[])
                    .map(|row| row.map_or(0, |row| row.get::<_, i32>(0))),
                None => Ok(0),
            }
        })?;
        let version = usize::try_from(version).unwrap_or(0);
        if version > SCHEMA_VERSION {
            return Err(Status::unimplemented(format!(
                "postgres store {} has schema version {version}, and this Finch knows versions \
                 up to {SCHEMA_VERSION}",
                session.schema
            )));
        }
        if version < SCHEMA_VERSION {
            for migration in &MIGRATIONS[version..] {
                migration(session, embedding_dim)?;
            }
            session.with_client(|client| write_schema_version(client, SCHEMA_VERSION))?;
        }
        if session.tenant_isolation {
            isolate_tenants(session, embedding_dim)?;
        }
        Ok(())
    })();
    session.finish(prepared)
}

fn write_schema_version(client: &mut Client, version: usize) -> Result<(), postgres::Error> {
    let version = i32::try_from(version).unwrap_or(i32::MAX);
    client.batch_execute(
        "CREATE TABLE IF NOT EXISTS finch_meta (schema_version integer NOT NULL); \
         DELETE FROM finch_meta;",
    )?;
    client
        .execute("INSERT INTO finch_meta VALUES ($1)", &[&version])
        .map(drop)
}

/// Version 0 is every store without `finch_meta`. Its spans may keep keyword terms in the table
/// `memory_terms`, and it may lack columns that reads need.
fn migrate_to_1(session: &Arc<PgSession>, embedding_dim: usize) -> ZResult<()> {
    let tables = add_missing_tables_and_columns(session, embedding_dim)?;
    let had_terms = session.with_client(|client| {
        client
            .query_opt("SELECT to_regclass('memory_terms')::text", &[])
            .map(|row| {
                row.and_then(|row| row.get::<_, Option<String>>(0))
                    .is_some()
            })
    })?;
    // Only spans that the keyword-indexed write paths wrote have postings in memory_terms.
    if had_terms {
        let spans = &tables[SPANS_COLLECTION];
        let indexed = spans.pks_where(
            "lexical_terms IS NULL AND EXISTS \
             (SELECT 1 FROM memory_terms t WHERE t.span_id = memory_spans.pk)",
        )?;
        spans.backfill_terms("lexical_terms", indexed, |doc| {
            Ok(span_term_counts(&span_from_doc(doc)?)
                .0
                .into_keys()
                .collect())
        })?;
    }
    let state_records = &tables[STATE_RECORDS_COLLECTION];
    let pending = state_records.pks_where("lexical_terms IS NULL")?;
    state_records.backfill_terms("lexical_terms", pending, |doc| {
        Ok(state_lexical_terms(&state_record_from_doc(doc)?))
    })?;
    // claim_doc stores the sorted distinct ids of the claim's source spans and episodes.
    session.with_client(|client| {
        client.batch_execute(&format!(
            "UPDATE memory_spans SET lexical_terms = '{{}}' WHERE lexical_terms IS NULL; \
             DROP TABLE IF EXISTS memory_terms; \
             UPDATE {claims} SET evidence_ids = ARRAY(SELECT DISTINCT id COLLATE \"C\" \
               FROM jsonb_array_elements_text(COALESCE(source_spans_json, '[]')::jsonb \
                 || COALESCE(source_eps_json, '[]')::jsonb) AS e(id) ORDER BY 1) \
             WHERE evidence_ids IS NULL; \
             {OPEN_JOBS_INDEX_DDL};",
            claims = quote(CLAIMS_COLLECTION)
        ))
    })
}

/// Creates each collection table and column of the current schema that the store lacks, with
/// the indexes that `MemoryStore::create` builds. Returns the tables by name.
fn add_missing_tables_and_columns(
    session: &Arc<PgSession>,
    embedding_dim: usize,
) -> ZResult<BTreeMap<String, PgTable>> {
    let spans_hnsw = hnsw_index_name(SPANS_COLLECTION, "embedding");
    let vector_index = session.with_client(|client| {
        client
            .query_opt(
                "SELECT 1 FROM pg_indexes WHERE schemaname = current_schema() AND indexname = $1",
                &[&spans_hnsw],
            )
            .map(|row| row.is_some())
    })?;
    let mut tables = BTreeMap::new();
    for (name, schema) in collection_schemas(embedding_dim, span_schema(embedding_dim)) {
        let table = PgTable::new(session.clone(), schema)?;
        let present = session.with_client(|client| {
            client.query(
                "SELECT column_name::text FROM information_schema.columns \
                 WHERE table_schema = current_schema() AND table_name = $1",
                &[&name],
            )
        })?;
        let present = present
            .iter()
            .map(|row| row.get::<_, String>(0))
            .collect::<BTreeSet<_>>();
        if present.is_empty() {
            table.create(vector_index)?;
        } else {
            table.add_columns(&present, vector_index)?;
        }
        tables.insert(name, table);
    }
    Ok(tables)
}

/// Turns on row-level security for every collection table, with a policy that admits only rows
/// of the tenant that the transaction sets. The policy replaces any earlier one, so a second run
/// changes nothing.
fn isolate_tenants(session: &PgSession, embedding_dim: usize) -> ZResult<()> {
    let bypasses = session.with_client(|client| {
        client
            .query_one(
                "SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname = current_user",
                &[],
            )
            .map(|row| row.get::<_, bool>(0))
    })?;
    if bypasses {
        return Err(Status::permission_denied(
            "tenant isolation needs a postgres role that does not bypass row-level security",
        ));
    }
    let ddl = collection_schemas(embedding_dim, span_schema(embedding_dim))
        .keys()
        .map(|name| {
            let table = quote(name);
            format!(
                "ALTER TABLE {table} ENABLE ROW LEVEL SECURITY; \
                 ALTER TABLE {table} FORCE ROW LEVEL SECURITY; \
                 DROP POLICY IF EXISTS finch_tenant ON {table}; \
                 CREATE POLICY finch_tenant ON {table} \
                 USING (tenant_id = NULLIF(current_setting('finch.tenant_id', true), ''));"
            )
        })
        .collect::<String>();
    session.with_client(|client| client.batch_execute(&ddl))
}

/// Deletes the store in the schema `store.name` and every row it holds.
pub fn drop_store(store: &PostgresStore) -> ZResult<()> {
    PgSession::connect(store)?.with_client(|client| {
        client.batch_execute(&format!(
            "DROP SCHEMA IF EXISTS {} CASCADE",
            quote(&store.name)
        ))
    })
}

/// Connections to one store's schema. A thread inside a transaction, a write or a snapshot read,
/// uses the connection bound to it; every other operation autocommits on a pooled connection.
/// The pool opens connections on demand, up to `max_connections`, and a closed pooled
/// connection opens again before its next use.
pub(crate) struct PgSession {
    url: String,
    schema: String,
    idle: Mutex<Vec<Client>>,
    open: Mutex<usize>,
    released: Condvar,
    bound: Mutex<HashMap<ThreadId, Client>>,
    max_connections: usize,
    pub(crate) tenant_isolation: bool,
}

impl PgSession {
    fn connect(store: &PostgresStore) -> ZResult<Self> {
        let (url, schema) = (store.url.as_str(), store.name.as_str());
        if store.max_connections == 0 {
            return Err(Status::invalid_argument(
                "a postgres store needs at least one connection",
            ));
        }
        if schema.is_empty()
            || schema.len() > 63
            || !schema
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
        {
            return Err(Status::invalid_argument(format!(
                "postgres store name {schema:?} must be 1-63 characters of a-z, 0-9, or _"
            )));
        }
        let first = open_connection(url, schema)?;
        Ok(Self {
            url: url.to_string(),
            schema: schema.to_string(),
            idle: Mutex::new(vec![first]),
            open: Mutex::new(1),
            released: Condvar::new(),
            bound: Mutex::new(HashMap::new()),
            max_connections: store.max_connections,
            tenant_isolation: store.tenant_isolation,
        })
    }

    fn schema_exists(&self) -> ZResult<bool> {
        let schema = self.schema.clone();
        self.with_client(|client| {
            client
                .query_opt("SELECT 1 FROM pg_namespace WHERE nspname = $1", &[&schema])
                .map(|row| row.is_some())
        })
    }

    /// The dimension of the span embedding column, which pgvector keeps as the type modifier.
    fn embedding_dim(&self) -> ZResult<usize> {
        let dim: i32 = self
            .with_client(|client| {
                client.query_one(
                    "SELECT atttypmod FROM pg_attribute \
                     WHERE attrelid = to_regclass('memory_spans') AND attname = 'embedding'",
                    &[],
                )
            })?
            .get(0);
        usize::try_from(dim)
            .map_err(|_| Status::internal(format!("memory_spans.embedding has dimension {dim}")))
    }

    /// An idle connection, a new one while fewer than `max_connections` are open, or the next
    /// one another thread releases.
    fn checkout(&self) -> ZResult<Client> {
        let mut idle = self.idle.lock();
        loop {
            if let Some(client) = idle.pop() {
                return Ok(client);
            }
            let mut open = self.open.lock();
            if *open < self.max_connections {
                *open += 1;
                drop(open);
                drop(idle);
                return open_connection(&self.url, &self.schema).inspect_err(|_| {
                    *self.open.lock() -= 1;
                });
            }
            drop(open);
            self.released.wait(&mut idle);
        }
    }

    fn checkin(&self, client: Client) {
        if client.is_closed() {
            *self.open.lock() -= 1;
        } else {
            self.idle.lock().push(client);
        }
        self.released.notify_one();
    }

    pub(crate) fn with_client<T>(
        &self,
        mut run: impl FnMut(&mut Client) -> Result<T, postgres::Error>,
    ) -> ZResult<T> {
        let thread = thread::current().id();
        // A bound connection must not reconnect: its transaction's work would be gone. The lookup
        // is its own statement so the map's lock is free while the statement runs.
        let bound = self.bound.lock().remove(&thread);
        if let Some(mut client) = bound {
            let result = run(&mut client);
            self.bound.lock().insert(thread, client);
            return result.map_err(pg_error);
        }
        let mut client = self.checkout()?;
        if client.is_closed() {
            client = open_connection(&self.url, &self.schema)?;
        }
        let result = match run(&mut client) {
            // A closed or terminated connection never ran the statement, so it runs once more.
            Err(error) if connection_lost(&error) => {
                match open_connection(&self.url, &self.schema) {
                    Ok(fresh) => {
                        client = fresh;
                        run(&mut client).map_err(pg_error)
                    }
                    Err(status) => Err(status),
                }
            }
            result => result.map_err(pg_error),
        };
        self.checkin(client);
        result
    }

    /// Whether the calling thread is inside a transaction of this session.
    pub(crate) fn in_transaction(&self) -> bool {
        self.bound.lock().contains_key(&thread::current().id())
    }

    /// Starts a write transaction that the calling thread's operations join. With `locks`, it
    /// holds the shared advisory locks and then the exclusive one until the transaction ends, so
    /// it waits for every write that holds one of them in the conflicting mode, in any process.
    /// With `tenant`, row-level security shows the transaction only that tenant's rows.
    pub(crate) fn begin_write(
        &self,
        locks: Option<(&[i64], i64)>,
        tenant: Option<&str>,
    ) -> ZResult<()> {
        self.begin("BEGIN", locks, tenant)
    }

    /// Starts a read-only transaction whose reads all see one snapshot of `tenant`'s rows.
    pub(crate) fn begin_snapshot(&self, tenant: Option<&str>) -> ZResult<()> {
        self.begin(
            "BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY",
            None,
            tenant,
        )
    }

    fn begin(
        &self,
        statement: &str,
        locks: Option<(&[i64], i64)>,
        tenant: Option<&str>,
    ) -> ZResult<()> {
        if self.in_transaction() {
            return Err(Status::internal("a postgres transaction is already open"));
        }
        let mut client = self.checkout()?;
        if client.is_closed() {
            client = open_connection(&self.url, &self.schema)?;
        }
        let start = |client: &mut Client| {
            client.batch_execute(statement)?;
            if let Some((shared, exclusive)) = locks {
                for key in shared {
                    client.execute("SELECT pg_advisory_xact_lock_shared($1)", &[key])?;
                }
                client.execute("SELECT pg_advisory_xact_lock($1)", &[&exclusive])?;
            }
            if let Some(tenant) = tenant {
                client.execute("SELECT set_config('finch.tenant_id', $1, true)", &[&tenant])?;
            }
            Ok(())
        };
        let mut started = start(&mut client);
        // A lost connection started no transaction, so it starts once more on a new one.
        if started.as_ref().is_err_and(connection_lost) {
            match open_connection(&self.url, &self.schema) {
                Ok(fresh) => {
                    client = fresh;
                    started = start(&mut client);
                }
                Err(status) => {
                    self.checkin(client);
                    return Err(status);
                }
            }
        }
        if let Err(error) = started {
            self.checkin(client);
            return Err(pg_error(error));
        }
        self.bound.lock().insert(thread::current().id(), client);
        Ok(())
    }

    pub(crate) fn commit(&self) -> ZResult<()> {
        self.end("COMMIT")
    }

    /// Commits the calling thread's transaction when `result` is a success, and rolls it back
    /// otherwise.
    pub(crate) fn finish<T>(&self, result: ZResult<T>) -> ZResult<T> {
        match result {
            Ok(value) => {
                self.commit()?;
                Ok(value)
            }
            Err(error) => match self.rollback() {
                Ok(()) => Err(error),
                Err(rollback) => Err(Status::internal(format!(
                    "{error}; rolling it back failed: {rollback}"
                ))),
            },
        }
    }

    /// Fails when tenant isolation is on and the calling thread has no transaction, which is
    /// where an operation sets its tenant.
    fn check_tenant_bound(&self, table: &str) -> ZResult<()> {
        if self.tenant_isolation && !self.in_transaction() {
            return Err(Status::permission_denied(format!(
                "tenant isolation is on, so {table} is read and written only by an operation \
                 with a tenant scope"
            )));
        }
        Ok(())
    }

    /// Discards the calling thread's transaction, if it has one.
    pub(crate) fn rollback(&self) -> ZResult<()> {
        if !self.in_transaction() {
            return Ok(());
        }
        self.end("ROLLBACK")
    }

    fn end(&self, statement: &str) -> ZResult<()> {
        let mut client = self
            .bound
            .lock()
            .remove(&thread::current().id())
            .ok_or_else(|| Status::internal("no postgres transaction to end"))?;
        let ended = client.batch_execute(statement).map_err(pg_error);
        self.checkin(client);
        ended
    }

    #[cfg(test)]
    pub(crate) fn open_connections(&self) -> usize {
        *self.open.lock()
    }
}

/// A connection with the store's schema first on the search path.
fn open_connection(url: &str, schema: &str) -> ZResult<Client> {
    let mut client = Client::connect(url, NoTls).map_err(pg_error)?;
    // Filtered HNSW scans keep searching until they find `LIMIT` matches, in exact order. A larger
    // ef_search makes the planner cost the index above a full scan and skip it.
    client
        .batch_execute(&format!(
            "SET search_path TO {}, public; SET hnsw.ef_search = 100; \
             SET hnsw.iterative_scan = strict_order;",
            quote(schema)
        ))
        .map_err(pg_error)?;
    Ok(client)
}

#[derive(Clone, Copy, PartialEq)]
enum ColumnKind {
    Text,
    BigInt,
    Real,
    TextArray,
    Vector,
}

struct Column {
    name: String,
    kind: ColumnKind,
    dimension: usize,
    indexed: bool,
}

/// One collection as a table: `pk`, a `seq` that orders rows by their last write, and one column
/// per field.
struct PgTable {
    session: Arc<PgSession>,
    name: String,
    columns: Vec<Column>,
    // Vector fields with an HNSW index; their searches order by the indexed halfvec distance.
    hnsw_fields: Mutex<BTreeSet<String>>,
}

impl PgTable {
    fn new(session: Arc<PgSession>, schema: CollectionSchema) -> ZResult<Self> {
        let columns = schema
            .fields
            .iter()
            .map(|field| {
                let kind = match field.data_type {
                    DataType::String => ColumnKind::Text,
                    DataType::Int64 => ColumnKind::BigInt,
                    DataType::Float32 => ColumnKind::Real,
                    DataType::ArrayString => ColumnKind::TextArray,
                    DataType::VectorFp32 => ColumnKind::Vector,
                    other => {
                        return Err(Status::invalid_argument(format!(
                            "postgres tables do not store {other:?} field {}",
                            field.name
                        )))
                    }
                };
                Ok(Column {
                    name: field.name.clone(),
                    kind,
                    dimension: field.dimension.unwrap_or(0),
                    indexed: field.index_params.is_some(),
                })
            })
            .collect::<ZResult<Vec<_>>>()?;
        Ok(Self {
            session,
            name: schema.name,
            columns,
            hnsw_fields: Mutex::new(BTreeSet::new()),
        })
    }

    fn create(&self, vector_index: bool) -> ZResult<()> {
        let table = quote(&self.name);
        let mut ddl = format!(
            "CREATE TABLE {table} (pk text PRIMARY KEY, seq bigint NOT NULL{}); \
             CREATE INDEX ON {table} (seq);",
            self.columns
                .iter()
                .map(|column| format!(", {} {}", quote(&column.name), column_type(column)))
                .collect::<String>()
        );
        for column in &self.columns {
            ddl.push_str(&self.column_index_ddl(column, vector_index));
        }
        // Scope filters test space, then tenant and user, so one index serves a store of many users.
        ddl.push_str(&format!(
            "CREATE INDEX ON {table} (space_id, tenant_id, user_id);"
        ));
        self.session
            .with_client(|client| client.batch_execute(&ddl))
    }

    /// Adds each column that `present` does not name, with its index.
    fn add_columns(&self, present: &BTreeSet<String>, vector_index: bool) -> ZResult<()> {
        let ddl = self
            .columns
            .iter()
            .filter(|column| !present.contains(&column.name))
            .map(|column| {
                format!(
                    "ALTER TABLE {} ADD COLUMN {} {};{}",
                    quote(&self.name),
                    quote(&column.name),
                    column_type(column),
                    self.column_index_ddl(column, vector_index)
                )
            })
            .collect::<String>();
        self.session
            .with_client(|client| client.batch_execute(&ddl))
    }

    fn column_index_ddl(&self, column: &Column, vector_index: bool) -> String {
        let table = quote(&self.name);
        if !column.indexed {
            String::new()
        } else if column.kind == ColumnKind::Vector {
            if !vector_index {
                return String::new();
            }
            self.hnsw_fields.lock().insert(column.name.clone());
            hnsw_index_ddl(&self.name, column, None)
        } else if column.kind == ColumnKind::TextArray {
            format!(
                "CREATE INDEX ON {table} USING gin ({});",
                quote(&column.name)
            )
        } else if column.name != "space_id" {
            format!("CREATE INDEX ON {table} ({});", quote(&column.name))
        } else {
            String::new()
        }
    }

    /// The pks of the rows that match the SQL condition `condition`, in pk order.
    fn pks_where(&self, condition: &str) -> ZResult<Vec<String>> {
        let sql = format!(
            "SELECT pk FROM {} WHERE {condition} ORDER BY pk",
            quote(&self.name)
        );
        let rows = self.session.with_client(|client| client.query(&sql, &[]))?;
        Ok(rows.iter().map(|row| row.get::<_, String>(0)).collect())
    }

    /// Sets the text[] `column` of each row in `pks` to `terms` of the row.
    fn backfill_terms(
        &self,
        column: &str,
        pks: Vec<String>,
        terms: impl Fn(&Doc) -> ZResult<Vec<String>>,
    ) -> ZResult<()> {
        let sql = format!(
            "UPDATE {table} SET {column} = ARRAY(SELECT jsonb_array_elements_text(u.terms::jsonb)) \
             FROM unnest($1::text[], $2::text[]) AS u(pk, terms) WHERE {table}.pk = u.pk",
            table = quote(&self.name),
            column = quote(column)
        );
        for chunk in pks.chunks(1024) {
            let docs = self.fetch(chunk.to_vec())?;
            let mut keys = Vec::with_capacity(docs.len());
            let mut values = Vec::with_capacity(docs.len());
            for (pk, doc) in docs {
                let terms = serde_json::to_string(&terms(&doc)?)
                    .map_err(|e| Status::internal(e.to_string()))?;
                keys.push(pk);
                values.push(terms);
            }
            self.session
                .with_client(|client| client.execute(&sql, &[&keys, &values]))?;
        }
        Ok(())
    }

    fn load_hnsw_fields(&self) -> ZResult<()> {
        let indexes = self.session.with_client(|client| {
            client.query(
                "SELECT indexname FROM pg_indexes \
                 WHERE schemaname = current_schema() AND tablename = $1",
                &[&self.name],
            )
        })?;
        let names = indexes
            .iter()
            .map(|row| row.get::<_, String>(0))
            .collect::<BTreeSet<_>>();
        let mut fields = self.hnsw_fields.lock();
        for column in self.columns.iter().filter(|c| c.kind == ColumnKind::Vector) {
            if names.contains(&hnsw_index_name(&self.name, &column.name)) {
                fields.insert(column.name.clone());
            }
        }
        Ok(())
    }

    fn column(&self, name: &str) -> ZResult<&Column> {
        self.columns
            .iter()
            .find(|column| column.name == name)
            .ok_or_else(|| Status::invalid_argument(format!("{} has no field {name}", self.name)))
    }

    /// Writes each doc whole in one statement; a field the doc lacks is stored as null. Rows
    /// take sequence numbers in doc order.
    fn write(&self, docs: Vec<Doc>, replace: bool) -> ZResult<Vec<Status>> {
        self.session.check_tenant_bound(&self.name)?;
        let mut statuses = vec![Status::default(); docs.len()];
        let mut batch = WriteBatch::new(&self.columns);
        let mut position = HashMap::<&str, usize>::new();
        for (index, doc) in docs.iter().enumerate() {
            if let Some(&earlier) = position.get(doc.pk.as_str()) {
                if !replace {
                    statuses[index] =
                        Status::already_exists(format!("pk {} already exists", doc.pk));
                    continue;
                }
                // A later upsert of the same pk replaces the earlier one, as separate writes would.
                batch.remove(earlier);
            }
            match self.check_fields(doc).and_then(|()| batch.push(index, doc)) {
                Ok(slot) => {
                    position.insert(doc.pk.as_str(), slot);
                }
                Err(status) => statuses[index] = status,
            }
        }
        let (indexes, mut params) = batch.finish();
        if indexes.is_empty() {
            return Ok(statuses);
        }
        let sql = self.write_sql(replace);
        let written = self.session.with_client(|client| {
            let count = i64::try_from(indexes.len()).unwrap_or(i64::MAX);
            let seqs = client
                .query(
                    "SELECT nextval('write_seq') FROM generate_series(1, $1::bigint)",
                    &[&count],
                )?
                .iter()
                .map(|row| row.get::<_, i64>(0))
                .collect::<Vec<_>>();
            params.insert(1, Box::new(sorted(seqs)));
            let refs = params.iter().map(|p| p.as_ref()).collect::<Vec<_>>();
            client.query(&sql, &refs)
        })?;
        if !replace {
            let written = written
                .iter()
                .map(|row| row.get::<_, String>(0))
                .collect::<std::collections::HashSet<_>>();
            for &index in &indexes {
                if !written.contains(&docs[index].pk) {
                    statuses[index] =
                        Status::already_exists(format!("pk {} already exists", docs[index].pk));
                }
            }
        }
        Ok(statuses)
    }

    fn write_sql(&self, replace: bool) -> String {
        let names = self
            .columns
            .iter()
            .map(|column| quote(&column.name))
            .collect::<Vec<_>>();
        let arrays = self
            .columns
            .iter()
            .enumerate()
            .map(|(i, column)| format!("${}::{}", i + 3, array_param_type(column)))
            .collect::<Vec<_>>();
        let values = self
            .columns
            .iter()
            .map(|column| match column.kind {
                // text[] cells travel as JSON, since unnest flattens arrays of arrays.
                ColumnKind::TextArray => format!(
                    "CASE WHEN u.{name} IS NULL THEN NULL \
                     ELSE ARRAY(SELECT jsonb_array_elements_text(u.{name}::jsonb)) END",
                    name = quote(&column.name)
                ),
                _ => format!("u.{}", quote(&column.name)),
            })
            .collect::<Vec<_>>();
        let on_conflict = if replace {
            format!(
                "DO UPDATE SET seq = EXCLUDED.seq{}",
                names
                    .iter()
                    .map(|name| format!(", {name} = EXCLUDED.{name}"))
                    .collect::<String>()
            )
        } else {
            "DO NOTHING".to_string()
        };
        format!(
            "INSERT INTO {table} (pk, seq, {names}) \
             SELECT u.pk, u.seq, {values} FROM unnest($1::text[], $2::bigint[], {arrays}) \
             AS u(pk, seq, {names}) ON CONFLICT (pk) {on_conflict} RETURNING pk",
            table = quote(&self.name),
            names = names.join(", "),
            values = values.join(", "),
            arrays = arrays.join(", "),
        )
    }

    fn check_fields(&self, doc: &Doc) -> Result<(), Status> {
        match doc
            .fields
            .keys()
            .find(|field| self.columns.iter().all(|column| &column.name != *field))
        {
            Some(unknown) => Err(Status::invalid_argument(format!(
                "{} has no field {unknown}",
                self.name
            ))),
            None => Ok(()),
        }
    }

    fn select_list(&self, query: Option<&VectorQuery>) -> ZResult<Vec<&Column>> {
        match query.and_then(|query| query.output_fields.as_ref()) {
            Some(fields) => fields.iter().map(|field| self.column(field)).collect(),
            None => Ok(self
                .columns
                .iter()
                .filter(|column| {
                    column.kind != ColumnKind::Vector || query.is_none_or(|q| q.include_vector)
                })
                .collect()),
        }
    }

    /// Selects `pk` and `columns`, plus `distance` into `score` when it is given.
    fn select(
        &self,
        columns: &[&Column],
        distance: Option<&str>,
        tail: &str,
        params: Vec<Param>,
    ) -> ZResult<Vec<Arc<Doc>>> {
        self.session.check_tenant_bound(&self.name)?;
        let sql = format!(
            "SELECT pk{}{} FROM {} {tail}",
            columns
                .iter()
                .map(|column| format!(", {}", quote(&column.name)))
                .collect::<String>(),
            distance.map_or(String::new(), |distance| format!(
                ", {distance} AS distance"
            )),
            quote(&self.name)
        );
        let rows = self.session.with_client(|client| {
            let refs = params.iter().map(|p| p.as_ref()).collect::<Vec<_>>();
            client.query(&sql, &refs)
        })?;
        rows.iter()
            .map(|row| row_doc(row, columns, distance.is_some()).map(Arc::new))
            .collect()
    }

    fn filter_clause(&self, query: &VectorQuery, params: &mut Vec<Param>) -> ZResult<String> {
        let expr = parse_filter(query.filter.as_deref().unwrap_or(""))?;
        let mut sql = String::new();
        self.filter_sql(&expr, &mut sql, params)?;
        Ok(sql)
    }

    fn filter_sql(
        &self,
        expr: &FilterExpr,
        sql: &mut String,
        params: &mut Vec<Param>,
    ) -> ZResult<()> {
        match expr {
            FilterExpr::AlwaysTrue => sql.push_str("TRUE"),
            FilterExpr::AlwaysFalse => sql.push_str("FALSE"),
            FilterExpr::And(left, right) | FilterExpr::Or(left, right) => {
                let op = if matches!(expr, FilterExpr::And(..)) {
                    " AND "
                } else {
                    " OR "
                };
                sql.push('(');
                self.filter_sql(left, sql, params)?;
                sql.push_str(op);
                self.filter_sql(right, sql, params)?;
                sql.push(')');
            }
            // A comparison with null is unknown in SQL but false in the filter language.
            FilterExpr::Not(inner) => {
                sql.push_str("NOT COALESCE(");
                self.filter_sql(inner, sql, params)?;
                sql.push_str(", FALSE)");
            }
            FilterExpr::IsNull(field) => sql.push_str(&format!("{} IS NULL", self.quoted(field)?)),
            FilterExpr::IsNotNull(field) => {
                sql.push_str(&format!("{} IS NOT NULL", self.quoted(field)?))
            }
            FilterExpr::Compare { field, op, value } => {
                let column = self.column(field)?;
                let placeholder = push_param(params, scalar_param(column, value)?);
                sql.push_str(&format!(
                    "{} {} {placeholder}",
                    compared_column(column, value),
                    compare_sql(op)
                ));
            }
            FilterExpr::InList {
                field,
                values,
                negated,
            } => {
                let column = self.column(field)?;
                let placeholder = push_param(params, list_param(column, values)?);
                let test = format!("{} = ANY({placeholder})", quote(&column.name));
                sql.push_str(&if *negated {
                    format!("NOT ({test})")
                } else {
                    test
                });
            }
            FilterExpr::ContainAny { field, values } | FilterExpr::ContainAll { field, values } => {
                let column = self.column(field)?;
                let placeholder = push_param(params, Box::new(string_list(values)?));
                let op = if matches!(expr, FilterExpr::ContainAny { .. }) {
                    "&&"
                } else {
                    "@>"
                };
                sql.push_str(&format!("{} {op} {placeholder}", quote(&column.name)));
            }
            FilterExpr::LikePattern { field, pattern } => {
                let placeholder = push_param(params, Box::new(pattern.clone()));
                sql.push_str(&format!("{} LIKE {placeholder}", self.quoted(field)?));
            }
            FilterExpr::HasPrefix { field, prefix } => {
                let placeholder = push_param(params, Box::new(prefix.clone()));
                sql.push_str(&format!(
                    "starts_with({}, {placeholder})",
                    self.quoted(field)?
                ));
            }
            FilterExpr::HasSuffix { field, suffix } => {
                let placeholder = push_param(params, Box::new(suffix.clone()));
                sql.push_str(&format!(
                    "right({}, char_length({placeholder}::text)) = {placeholder}::text",
                    self.quoted(field)?
                ));
            }
            FilterExpr::ArrayLengthCompare { field, op, len } => sql.push_str(&format!(
                "cardinality({}) {} {len}",
                self.quoted(field)?,
                compare_sql(op)
            )),
        }
        Ok(())
    }

    fn quoted(&self, field: &str) -> ZResult<String> {
        self.column(field).map(|column| quote(&column.name))
    }
}

impl MemoryTable for PgTable {
    fn insert(&self, docs: Vec<Doc>) -> ZResult<Vec<Status>> {
        self.write(docs, false)
    }

    fn upsert(&self, docs: Vec<Doc>) -> ZResult<Vec<Status>> {
        self.write(docs, true)
    }

    fn fetch(&self, pks: Vec<String>) -> ZResult<HashMap<String, Arc<Doc>>> {
        let columns = self.columns.iter().collect::<Vec<_>>();
        let docs = self.select(&columns, None, "WHERE pk = ANY($1)", vec![Box::new(pks)])?;
        Ok(docs.into_iter().map(|doc| (doc.pk.clone(), doc)).collect())
    }

    fn scan_filter_only(&self, query: VectorQuery) -> ZResult<Vec<Arc<Doc>>> {
        let mut params = Vec::new();
        let filter = self.filter_clause(&query, &mut params)?;
        let columns = self.select_list(Some(&query))?;
        self.select(
            &columns,
            None,
            &format!("WHERE {filter} ORDER BY seq"),
            params,
        )
    }

    fn scan_prefix(&self, mut query: VectorQuery, limit: usize) -> ZResult<Vec<Arc<Doc>>> {
        query.query_vector.clear();
        query.topk = limit;
        self.query(query)
    }

    fn query(&self, query: VectorQuery) -> ZResult<Vec<Arc<Doc>>> {
        let columns = self.select_list(Some(&query))?;
        let limit = i64::try_from(query.topk).unwrap_or(i64::MAX);
        if query.query_vector.is_empty() {
            let mut params = Vec::new();
            let filter = self.filter_clause(&query, &mut params)?;
            let tail = format!("WHERE {filter} ORDER BY seq LIMIT {limit}");
            return self.select(&columns, None, &tail, params);
        }
        let vector = self.column(&query.field_name)?;
        if vector.kind != ColumnKind::Vector {
            return Err(Status::invalid_argument(format!(
                "{} is not a vector field",
                vector.name
            )));
        }
        let filter_params = || -> ZResult<(String, Vec<Param>)> {
            let mut params: Vec<Param> = vec![Box::new(Vector::from(query.query_vector.clone()))];
            let filter = self.filter_clause(&query, &mut params)?;
            Ok((filter, params))
        };
        let field = quote(&vector.name);
        if self.hnsw_fields.lock().contains(&vector.name) {
            // A second sort key would keep the planner from using the HNSW index.
            let dim = vector.dimension;
            let (filter, params) = filter_params()?;
            let tail =
                format!("WHERE {field} IS NOT NULL AND ({filter}) ORDER BY distance LIMIT {limit}");
            let distance = format!("({field})::halfvec({dim}) <=> $1::vector::halfvec({dim})");
            let hits = self.select(&columns, Some(&distance), &tail, params)?;
            // The index scan stops after hnsw.max_scan_tuples rows, so a narrow filter can leave
            // it short; the exact search below then finds every match.
            if hits.len() >= query.topk {
                return Ok(hits);
            }
        }
        let (filter, params) = filter_params()?;
        let tail = format!(
            "WHERE {field} IS NOT NULL AND ({filter}) ORDER BY distance, seq LIMIT {limit}"
        );
        self.select(&columns, Some(&format!("{field} <=> $1")), &tail, params)
    }

    fn delete(&self, filter: &str) -> ZResult<usize> {
        // An empty filter matches every row, and no caller means that.
        if filter.trim().is_empty() {
            return Err(Status::invalid_argument("a delete needs a filter"));
        }
        let mut params = Vec::new();
        let mut condition = String::new();
        self.filter_sql(&parse_filter(filter)?, &mut condition, &mut params)?;
        let sql = format!("DELETE FROM {} WHERE {condition}", quote(&self.name));
        let deleted = self.session.with_client(|client| {
            let refs = params.iter().map(|p| p.as_ref()).collect::<Vec<_>>();
            client.execute(&sql, &refs)
        })?;
        Ok(usize::try_from(deleted).unwrap_or(usize::MAX))
    }

    fn create_hnsw_index(
        &self,
        field: &str,
        params: HnswIndexParams,
        _concurrency: Option<usize>,
    ) -> ZResult<()> {
        let column = self.column(field)?;
        let ddl = hnsw_index_ddl(&self.name, column, Some(&params));
        self.session
            .with_client(|client| client.batch_execute(&ddl))?;
        self.hnsw_fields.lock().insert(column.name.clone());
        Ok(())
    }
}

/// An HNSW index on the halfvec form of `column`: pgvector indexes vector columns of at most
/// 2000 dimensions and halfvec columns of at most 4000, and halfvec halves the index size.
fn hnsw_index_ddl(table: &str, column: &Column, params: Option<&HnswIndexParams>) -> String {
    let with = params.map_or(String::new(), |params| {
        format!(
            " WITH (m = {}, ef_construction = {})",
            params.m, params.ef_construction
        )
    });
    format!(
        "CREATE INDEX IF NOT EXISTS {} ON {} USING hnsw (({}::halfvec({})) halfvec_cosine_ops){with};",
        quote(&hnsw_index_name(table, &column.name)),
        quote(table),
        quote(&column.name),
        column.dimension
    )
}

fn hnsw_index_name(table: &str, field: &str) -> String {
    format!("{table}_{field}_hnsw")
}

fn column_type(column: &Column) -> String {
    match column.kind {
        ColumnKind::Text => "text".to_string(),
        ColumnKind::BigInt => "bigint".to_string(),
        ColumnKind::Real => "real".to_string(),
        ColumnKind::TextArray => "text[]".to_string(),
        ColumnKind::Vector => format!("vector({})", column.dimension),
    }
}

/// Docs to write, one array per column, with each doc's index in the caller's list.
struct WriteBatch<'a> {
    columns: &'a [Column],
    indexes: Vec<Option<usize>>,
    pks: Vec<String>,
    cells: Vec<Vec<Cell>>,
}

enum Cell {
    Text(Option<String>),
    BigInt(Option<i64>),
    Real(Option<f32>),
    Vector(Option<Vector>),
}

// `cell` builds each column's variant from the column's kind, so another variant never appears.
impl Cell {
    fn into_text(self) -> Option<String> {
        match self {
            Cell::Text(v) => v,
            _ => None,
        }
    }

    fn into_i64(self) -> Option<i64> {
        match self {
            Cell::BigInt(v) => v,
            _ => None,
        }
    }

    fn into_f32(self) -> Option<f32> {
        match self {
            Cell::Real(v) => v,
            _ => None,
        }
    }

    fn into_vector(self) -> Option<Vector> {
        match self {
            Cell::Vector(v) => v,
            _ => None,
        }
    }
}

impl<'a> WriteBatch<'a> {
    fn new(columns: &'a [Column]) -> Self {
        Self {
            columns,
            indexes: Vec::new(),
            pks: Vec::new(),
            cells: columns.iter().map(|_| Vec::new()).collect(),
        }
    }

    /// Adds `doc` and returns its slot in the batch.
    fn push(&mut self, index: usize, doc: &Doc) -> Result<usize, Status> {
        let row = self
            .columns
            .iter()
            .map(|column| cell(column, doc.fields.get(&column.name)))
            .collect::<Result<Vec<_>, _>>()?;
        for (cells, cell) in self.cells.iter_mut().zip(row) {
            cells.push(cell);
        }
        self.indexes.push(Some(index));
        self.pks.push(doc.pk.clone());
        Ok(self.indexes.len() - 1)
    }

    fn remove(&mut self, slot: usize) {
        self.indexes[slot] = None;
    }

    /// The written docs' indexes, and the pk array followed by one array per column.
    fn finish(self) -> (Vec<usize>, Vec<Param>) {
        let keep = self.indexes.iter().map(Option::is_some).collect::<Vec<_>>();
        let mut params: Vec<Param> = vec![Box::new(kept(self.pks, &keep))];
        for (column, cells) in self.columns.iter().zip(self.cells) {
            let cells = kept(cells, &keep).into_iter();
            params.push(match column.kind {
                ColumnKind::BigInt => Box::new(cells.map(Cell::into_i64).collect::<Vec<_>>()),
                ColumnKind::Real => Box::new(cells.map(Cell::into_f32).collect::<Vec<_>>()),
                ColumnKind::Vector => Box::new(cells.map(Cell::into_vector).collect::<Vec<_>>()),
                ColumnKind::Text | ColumnKind::TextArray => {
                    Box::new(cells.map(Cell::into_text).collect::<Vec<_>>())
                }
            });
        }
        (self.indexes.into_iter().flatten().collect(), params)
    }
}

fn kept<T>(items: Vec<T>, keep: &[bool]) -> Vec<T> {
    items
        .into_iter()
        .zip(keep)
        .filter_map(|(item, keep)| keep.then_some(item))
        .collect()
}

fn sorted(mut values: Vec<i64>) -> Vec<i64> {
    values.sort_unstable();
    values
}

fn array_param_type(column: &Column) -> &'static str {
    match column.kind {
        ColumnKind::Text | ColumnKind::TextArray => "text[]",
        ColumnKind::BigInt => "bigint[]",
        ColumnKind::Real => "real[]",
        ColumnKind::Vector => "vector[]",
    }
}

fn cell(column: &Column, value: Option<&Value>) -> Result<Cell, Status> {
    let value = value.filter(|value| !value.is_null());
    let mismatch =
        || Status::invalid_argument(format!("field {} does not hold {:?}", column.name, value));
    Ok(match column.kind {
        ColumnKind::Text => Cell::Text(
            value
                .map(|v| v.as_str().map(str::to_string).ok_or_else(mismatch))
                .transpose()?,
        ),
        ColumnKind::BigInt => {
            Cell::BigInt(value.map(|v| v.as_i64().ok_or_else(mismatch)).transpose()?)
        }
        ColumnKind::Real => Cell::Real(value.map(|v| v.as_f32().ok_or_else(mismatch)).transpose()?),
        ColumnKind::TextArray => Cell::Text(
            value
                .map(|v| match v {
                    Value::ArrayString(items) => {
                        serde_json::to_string(items).map_err(|e| Status::internal(e.to_string()))
                    }
                    _ => Err(mismatch()),
                })
                .transpose()?,
        ),
        ColumnKind::Vector => Cell::Vector(
            value
                .map(|v| match v {
                    Value::VecF32(items) if items.len() == column.dimension => {
                        Ok(Vector::from(items.clone()))
                    }
                    _ => Err(mismatch()),
                })
                .transpose()?,
        ),
    })
}

/// A filter literal as a parameter of the column's type; numbers compare as double precision.
fn scalar_param(column: &Column, value: &Value) -> ZResult<Param> {
    let mismatch = || {
        Status::invalid_argument(format!(
            "invalid filter: {:?} cannot compare with field {}",
            value, column.name
        ))
    };
    match (column.kind, value) {
        (ColumnKind::Text, Value::String(text)) => Ok(Box::new(text.clone())),
        (ColumnKind::BigInt, value) if value.as_i64().is_some() && !is_float(value) => {
            Ok(Box::new(value.as_i64().ok_or_else(mismatch)?))
        }
        (ColumnKind::BigInt | ColumnKind::Real, value) => {
            Ok(Box::new(as_f64(value).ok_or_else(mismatch)?))
        }
        _ => Err(mismatch()),
    }
}

/// The column as the comparison needs it: a float literal against an integer column, or any
/// number against a real column, compares in double precision.
fn compared_column(column: &Column, value: &Value) -> String {
    let name = quote(&column.name);
    match column.kind {
        ColumnKind::Real => format!("{name}::float8"),
        ColumnKind::BigInt if is_float(value) => format!("{name}::float8"),
        _ => name,
    }
}

fn list_param(column: &Column, values: &[Value]) -> ZResult<Param> {
    match column.kind {
        ColumnKind::Text => Ok(Box::new(string_list(values)?)),
        ColumnKind::BigInt => Ok(Box::new(
            values
                .iter()
                .map(|value| {
                    value.as_i64().ok_or_else(|| {
                        Status::invalid_argument(format!(
                            "invalid filter: {value:?} is not an integer for {}",
                            column.name
                        ))
                    })
                })
                .collect::<ZResult<Vec<i64>>>()?,
        )),
        _ => Err(Status::invalid_argument(format!(
            "invalid filter: IN is not supported on {}",
            column.name
        ))),
    }
}

fn string_list(values: &[Value]) -> ZResult<Vec<String>> {
    values
        .iter()
        .map(|value| {
            value.as_str().map(str::to_string).ok_or_else(|| {
                Status::invalid_argument(format!("invalid filter: {value:?} is not a string"))
            })
        })
        .collect()
}

fn is_float(value: &Value) -> bool {
    matches!(value, Value::F16(_) | Value::F32(_) | Value::F64(_))
}

fn as_f64(value: &Value) -> Option<f64> {
    match value {
        Value::F64(v) => Some(*v),
        Value::I64(v) => Some(*v as f64),
        other => other.as_f32().map(f64::from),
    }
}

fn compare_sql(op: &CompareOp) -> &'static str {
    match op {
        CompareOp::Equal => "=",
        CompareOp::NotEqual => "<>",
        CompareOp::LessThan => "<",
        CompareOp::LessEqual => "<=",
        CompareOp::GreaterThan => ">",
        CompareOp::GreaterEqual => ">=",
    }
}

fn push_param(params: &mut Vec<Param>, param: Param) -> String {
    params.push(param);
    format!("${}", params.len())
}

fn row_doc(row: &Row, columns: &[&Column], scored: bool) -> ZResult<Doc> {
    let mut doc = Doc::new(row.try_get::<_, String>(0).map_err(pg_error)?);
    for (i, column) in columns.iter().enumerate() {
        let index = i + 1;
        let value = match column.kind {
            ColumnKind::Text => row
                .try_get::<_, Option<String>>(index)
                .map(|v| v.map(Value::String)),
            ColumnKind::BigInt => row
                .try_get::<_, Option<i64>>(index)
                .map(|v| v.map(Value::I64)),
            ColumnKind::Real => row
                .try_get::<_, Option<f32>>(index)
                .map(|v| v.map(Value::F32)),
            ColumnKind::TextArray => row
                .try_get::<_, Option<Vec<String>>>(index)
                .map(|v| v.map(Value::ArrayString)),
            ColumnKind::Vector => row
                .try_get::<_, Option<Vector>>(index)
                .map(|v| v.map(|v| Value::VecF32(v.to_vec()))),
        }
        .map_err(pg_error)?;
        if let Some(value) = value {
            doc.fields.insert(column.name.clone(), value);
        }
    }
    if scored {
        doc.score = row.try_get::<_, f64>(columns.len() + 1).map_err(pg_error)? as f32;
    }
    Ok(doc)
}

fn quote(identifier: &str) -> String {
    format!("\"{}\"", identifier.replace('"', "\"\""))
}

pub(crate) fn pg_error(error: postgres::Error) -> Status {
    match error.as_db_error() {
        Some(db) => Status::io_error(format!("postgres: {} ({})", db.message(), db.code().code())),
        None => Status::io_error(format!("postgres: {error}")),
    }
}

fn connection_lost(error: &postgres::Error) -> bool {
    use postgres::error::SqlState;
    error.is_closed()
        || error
            .code()
            .is_some_and(|code| [SqlState::ADMIN_SHUTDOWN, SqlState::CRASH_SHUTDOWN].contains(code))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ActorKind, ChunkOptions, ClaimKind, ClaimPolarity, EpisodeInput, ManualClaimInput,
        MemoryScope, SourceKind, Visibility,
    };

    fn episode(scope: &MemoryScope, id: &str, text: &str) -> EpisodeInput {
        EpisodeInput {
            id: Some(id.to_string()),
            scope: scope.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            source_kind: SourceKind::UserMessage,
            actor: ActorKind::User,
            sequence_no: 1,
            event_time_ms: Some(10),
            valid_from_ms: None,
            valid_to_ms: None,
            raw_text: text.to_string(),
            blob_ref: None,
            mime_type: None,
            causal_parent_ids: Vec::new(),
            metadata_json: None,
        }
    }

    fn chunks() -> ChunkOptions {
        ChunkOptions {
            max_chars: 24,
            overlap_chars: 0,
            chunker_version: "test".to_string(),
        }
    }

    /// Writes keyword-indexed spans, one span without keyword terms, and a claim with its state.
    fn seed(store: &MemoryStore, scope: &MemoryScope) {
        let ingested = store
            .ingest_episode(
                episode(
                    scope,
                    "ep1",
                    "alpha beta gamma alpha delta epsilon zeta gamma",
                ),
                11,
                &chunks(),
            )
            .unwrap();
        let mut plain = ingested.spans[0].clone();
        plain.id = "plain_span".to_string();
        store
            .append_vector_spans(&[(plain, vec![1.0, 0.0, 0.0])])
            .unwrap();
        store
            .add_manual_claim(
                ManualClaimInput {
                    id: Some("c1".to_string()),
                    scope: scope.clone(),
                    visibility: Visibility::Private,
                    policy_tags: Vec::new(),
                    claim_text: "Alpha status is green".to_string(),
                    subject: Some("alpha".to_string()),
                    predicate: Some("status".to_string()),
                    object_value: Some("green".to_string()),
                    claim_kind: ClaimKind::Fact,
                    polarity: ClaimPolarity::Affirmative,
                    source_span_ids: vec![ingested.spans[0].id.clone()],
                    source_episode_ids: vec!["ep1".to_string()],
                    asserted_by: "user".to_string(),
                    confidence: None,
                    observed_at_ms: 10,
                    valid_from_ms: Some(10),
                    valid_to_ms: None,
                },
                None,
            )
            .unwrap();
    }

    fn column(
        admin: &mut Client,
        schema: &str,
        table: &str,
        column: &str,
    ) -> Vec<(String, Vec<String>)> {
        admin
            .query(
                &format!("SELECT pk, {column} FROM {schema}.{table} ORDER BY pk"),
                &[],
            )
            .unwrap()
            .iter()
            .map(|row| (row.get(0), row.get(1)))
            .collect()
    }

    #[test]
    fn open_migrates_a_version_0_store_in_place() {
        let url = crate::test_postgres_url();
        let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
        let old = PostgresStore::new(&url, "t_migrate_old");
        let fresh = PostgresStore::new(&url, "t_migrate_fresh");
        let scope = MemoryScope::new("s");
        for location in [&old, &fresh] {
            drop_store(location).unwrap();
            seed(&MemoryStore::create(location, 3, false).unwrap(), &scope);
        }
        let mut admin = Client::connect(&url, NoTls).unwrap();
        let written = [
            ("memory_spans", "lexical_terms"),
            ("memory_claims", "evidence_ids"),
            ("memory_state_records", "lexical_terms"),
        ]
        .map(|(table, name)| column(&mut admin, "t_migrate_old", table, name));
        assert!(written
            .iter()
            .all(|rows| rows.iter().any(|(_, terms)| !terms.is_empty())));
        // A store from before finch_meta: postings in memory_terms and none of the newer columns.
        admin
            .batch_execute(
                "SET search_path TO t_migrate_old, public; \
                 DROP TABLE finch_meta; \
                 CREATE TABLE memory_terms (pk text PRIMARY KEY, seq bigint NOT NULL, \
                   space_id text, tenant_id text, user_id text, agent_id text, project_id text, \
                   thread_id text, visibility text, policy_tags_json text, id text, term text, \
                   span_id text, tf bigint, doc_len bigint); \
                 INSERT INTO memory_terms (pk, seq, space_id, id, term, span_id, tf, doc_len) \
                   SELECT pk || ':' || term, 0, space_id, pk || ':' || term, term, pk, 1, 1 \
                   FROM memory_spans, unnest(lexical_terms) AS term; \
                 ALTER TABLE memory_spans DROP COLUMN lexical_terms; \
                 ALTER TABLE memory_claims DROP COLUMN evidence_ids; \
                 ALTER TABLE memory_state_records DROP COLUMN lexical_terms; \
                 RESET search_path;",
            )
            .unwrap();

        let migrated = MemoryStore::open(&old).unwrap();
        let version = |admin: &mut Client| -> (i32, String) {
            let row = admin
                .query_one(
                    "SELECT schema_version, xmin::text FROM t_migrate_old.finch_meta",
                    &[],
                )
                .unwrap();
            (row.get(0), row.get(1))
        };
        let (number, written_by) = version(&mut admin);
        assert_eq!(number as usize, SCHEMA_VERSION);
        let backfilled = [
            ("memory_spans", "lexical_terms"),
            ("memory_claims", "evidence_ids"),
            ("memory_state_records", "lexical_terms"),
        ]
        .map(|(table, name)| column(&mut admin, "t_migrate_old", table, name));
        assert_eq!(backfilled, written);
        let terms_table = admin
            .query_one(
                "SELECT to_regclass('t_migrate_old.memory_terms')::text",
                &[],
            )
            .unwrap()
            .get::<_, Option<String>>(0);
        assert_eq!(terms_table, None);
        let search = |store: &MemoryStore| {
            store
                .keyword_search_spans(&scope, "alpha gamma zeta", 10, 100, None)
                .unwrap()
                .into_iter()
                .map(|hit| (hit.span.id, hit.score))
                .collect::<Vec<_>>()
        };
        let hits = search(&migrated);
        assert!(!hits.is_empty());
        assert_eq!(hits, search(&MemoryStore::open(&fresh).unwrap()));
        drop(migrated);

        MemoryStore::open(&old).unwrap();
        assert_eq!(version(&mut admin), (number, written_by));
    }

    #[test]
    fn a_store_of_a_newer_schema_version_does_not_open() {
        let url = crate::test_postgres_url();
        let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
        let location = PostgresStore::new(&url, "t_newer_version");
        drop_store(&location).unwrap();
        drop(MemoryStore::create(&location, 3, false).unwrap());
        Client::connect(&url, NoTls)
            .unwrap()
            .batch_execute("UPDATE t_newer_version.finch_meta SET schema_version = 99")
            .unwrap();
        let error = MemoryStore::open(&location).err().unwrap();
        assert!(error.message().contains("version 99"), "{error}");
        assert!(
            error.message().contains(&format!("up to {SCHEMA_VERSION}")),
            "{error}"
        );
    }

    /// The test server's URL with a role that does not bypass row-level security.
    fn tenant_role_url(url: &str) -> String {
        let mut admin = Client::connect(url, NoTls).unwrap();
        let database = admin
            .query_one("SELECT current_database()", &[])
            .unwrap()
            .get::<_, String>(0);
        let role = format!("{database}_tenant_role");
        admin
            .batch_execute(&format!(
                "DO $$ BEGIN \
                   IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{role}') THEN \
                     CREATE ROLE {role} LOGIN PASSWORD '{role}' NOSUPERUSER NOBYPASSRLS; \
                   END IF; \
                 END $$; \
                 GRANT CREATE ON DATABASE {database} TO {role};"
            ))
            .unwrap();
        let config = url.parse::<postgres::Config>().unwrap();
        let host = match &config.get_hosts()[0] {
            postgres::config::Host::Tcp(host) => host.clone(),
            postgres::config::Host::Unix(path) => path.display().to_string(),
        };
        format!(
            "host={host} port={} dbname={database} user={role} password={role}",
            config.get_ports().first().copied().unwrap_or(5432)
        )
    }

    fn tenant_scope(tenant: &str) -> MemoryScope {
        let mut scope = MemoryScope::new("s");
        scope.tenant_id = Some(tenant.to_string());
        scope
    }

    fn space_scan() -> VectorQuery {
        VectorQuery::new("", Vec::new(), 100).with_filter("space_id = 's'")
    }

    #[test]
    fn tenant_isolation_shows_each_operation_only_its_tenants_rows() {
        let url = crate::test_postgres_url();
        let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
        drop_store(&PostgresStore::new(&url, "t_tenants")).unwrap();
        let mut location = PostgresStore::new(tenant_role_url(&url), "t_tenants");
        location.tenant_isolation = true;
        let store = MemoryStore::create(&location, 3, false).unwrap();
        let (a, b) = (tenant_scope("a"), tenant_scope("b"));
        store
            .ingest_episode(episode(&a, "ep_a", "alpha beta"), 11, &chunks())
            .unwrap();
        let b_spans = store
            .ingest_episode(episode(&b, "ep_b", "alpha beta"), 11, &chunks())
            .unwrap()
            .spans
            .into_iter()
            .map(|span| span.id)
            .collect::<Vec<_>>();

        let reopened = MemoryStore::open(&location).unwrap();
        for store in [&store, &reopened] {
            let guard = store.lock_state_read(&a).unwrap();
            // The filter matches both tenants' rows; row-level security keeps B's out.
            let rows = store.spans.scan_filter_only(space_scan()).unwrap();
            assert!(!rows.is_empty());
            assert!(rows
                .iter()
                .all(|doc| doc.fields.get("tenant_id").and_then(|v| v.as_str()) == Some("a")));
            assert!(store.spans.fetch(b_spans.clone()).unwrap().is_empty());
            drop(guard);
            assert!(store
                .fetch_spans_by_ids(&a, &b_spans, None)
                .unwrap()
                .is_empty());
            assert_eq!(
                store.fetch_spans_by_ids(&b, &b_spans, None).unwrap().len(),
                b_spans.len()
            );
        }

        let no_tenant = MemoryScope::new("s");
        let write = store.ingest_episode(episode(&no_tenant, "ep_none", "gamma"), 11, &chunks());
        assert!(write.is_err_and(|status| status.message().contains("tenant_id")));
        assert!(store.scan_spans(&no_tenant, 10, None).is_err());
        assert!(store.spans.fetch(b_spans).is_err());
    }

    #[test]
    fn tenant_isolation_refuses_a_role_that_bypasses_row_level_security() {
        let url = crate::test_postgres_url();
        let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
        let mut location = PostgresStore::new(&url, "t_tenants_superuser");
        drop_store(&location).unwrap();
        let bypasses = Client::connect(&url, NoTls)
            .unwrap()
            .query_one(
                "SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname = current_user",
                &[],
            )
            .unwrap()
            .get::<_, bool>(0);
        location.tenant_isolation = true;
        assert_eq!(MemoryStore::create(&location, 3, false).is_err(), bypasses);
    }

    #[test]
    fn without_tenant_isolation_a_store_holds_rows_without_a_tenant() {
        let url = crate::test_postgres_url();
        let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
        drop_store(&PostgresStore::new(&url, "t_tenants_off")).unwrap();
        let location = PostgresStore::new(tenant_role_url(&url), "t_tenants_off");
        let store = MemoryStore::create(&location, 3, false).unwrap();
        for (id, scope) in [
            ("ep_none", MemoryScope::new("s")),
            ("ep_a", tenant_scope("a")),
            ("ep_b", tenant_scope("b")),
        ] {
            store
                .ingest_episode(episode(&scope, id, "alpha"), 11, &chunks())
                .unwrap();
        }
        let tenants = store
            .spans
            .scan_filter_only(space_scan())
            .unwrap()
            .iter()
            .map(|doc| {
                doc.fields
                    .get("tenant_id")
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(tenants.len(), 3);
    }

    #[test]
    fn create_refuses_an_existing_store_and_keeps_its_rows() {
        let url = crate::test_postgres_url();
        let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
        let location = PostgresStore::new(&url, "t_existing");
        drop_store(&location).unwrap();
        let store = MemoryStore::create(&location, 3, false).unwrap();
        store.claims.insert(vec![Doc::new("kept")]).unwrap();
        drop(store);
        let again = MemoryStore::create(&location, 3, false);
        assert!(again.is_err_and(|status| status.is_already_exists()));
        let reopened = MemoryStore::open(&location).unwrap();
        assert!(reopened
            .claims
            .fetch(vec!["kept".to_string()])
            .unwrap()
            .contains_key("kept"));
    }

    #[test]
    fn an_indexed_search_with_a_narrow_filter_returns_every_match() {
        let url = crate::test_postgres_url();
        let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
        let location = PostgresStore::new(&url, "t_narrow_hnsw");
        drop_store(&location).unwrap();
        let store = MemoryStore::create(&location, 3, true).unwrap();
        let span = |pk: String, status: &str, embedding: Vec<f32>| {
            Doc::new(pk)
                .set("space_id", "s")
                .set("status", status)
                .set("embedding", embedding)
        };
        let mut docs: Vec<Doc> = (0..300)
            .map(|i| {
                span(
                    format!("near{i}"),
                    "common",
                    vec![1.0, i as f32 / 1000.0, 0.0],
                )
            })
            .collect();
        docs.extend((0..3).map(|i| span(format!("far{i}"), "rare", vec![-1.0, 0.0, i as f32])));
        store.spans.insert(docs).unwrap();
        // The planner then takes the index, and the index scan stops after its first candidate
        // list, which holds only "common" rows.
        store
            .session
            .with_client(|client| {
                client.batch_execute("SET enable_seqscan = off; SET hnsw.max_scan_tuples = 1")
            })
            .unwrap();
        let hits = store
            .spans
            .query(
                VectorQuery::new("embedding", vec![1.0, 0.0, 0.0], 3)
                    .with_filter("status = 'rare'"),
            )
            .unwrap();
        assert_eq!(hits.len(), 3);
    }

    #[test]
    fn reads_reconnect_after_the_server_drops_every_pooled_connection() {
        let url = crate::test_postgres_url();
        let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
        let location = PostgresStore::new(&url, "t_reconnect");
        drop_store(&location).unwrap();
        let store = MemoryStore::create(&location, 3, false).unwrap();
        let mut admin = Client::connect(&url, NoTls).unwrap();
        admin
            .execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity \
                 WHERE datname = current_database() AND pid <> pg_backend_pid()",
                &[],
            )
            .unwrap();
        for _ in 0..location.max_connections + 1 {
            assert!(store
                .scan_claims(&MemoryScope::new("s"), 10, None)
                .unwrap()
                .is_empty());
        }
    }
}

//! A memory store in Postgres with pgvector: one schema per store, one table per collection, and
//! each state mutation in one transaction.

use crate::schema::span_schema;
use crate::store::{collection_schemas, take_schema, MemoryStore};
use crate::table::{MemoryTable, MemoryTransactions};
use finch_db::sqlengine::{parse_filter, FilterExpr};
use finch_types::{
    CollectionSchema, CompareOp, DataType, Doc, HnswIndexParams, Status, Value, VectorQuery,
    ZResult,
};
use parking_lot::Mutex;
use pgvector::Vector;
use postgres::types::ToSql;
use postgres::{Client, NoTls, Row};
use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::{self, ThreadId};

type Param = Box<dyn ToSql + Sync>;

impl MemoryStore {
    /// Creates a store in the Postgres schema `name`, replacing any store already there.
    /// `vector_index` builds an HNSW index on each embedding column; without it searches are
    /// exact.
    pub fn create_postgres(
        url: &str,
        name: &str,
        embedding_dim: usize,
        vector_index: bool,
    ) -> ZResult<Self> {
        let session = Arc::new(PgSession::connect(url, name)?);
        session.with_client(|client| {
            client.batch_execute(&format!(
                "DROP SCHEMA IF EXISTS {schema} CASCADE; CREATE SCHEMA {schema}; \
                 CREATE SEQUENCE {schema}.write_seq;",
                schema = quote(name)
            ))
        })?;
        let mut schemas = collection_schemas(embedding_dim, span_schema(embedding_dim));
        Self::from_tables(postgres_path(name), Some(session.clone()), |table| {
            let table = PgTable::new(session.clone(), take_schema(&mut schemas, table)?)?;
            table.create(vector_index)?;
            Ok(Arc::new(table))
        })
    }

    /// Opens the store in the Postgres schema `name`.
    pub fn open_postgres(url: &str, name: &str) -> ZResult<Self> {
        let session = Arc::new(PgSession::connect(url, name)?);
        let embedding_dim = session.embedding_dim()?;
        let mut schemas = collection_schemas(embedding_dim, span_schema(embedding_dim));
        Self::from_tables(postgres_path(name), Some(session.clone()), |table| {
            let table = PgTable::new(session.clone(), take_schema(&mut schemas, table)?)?;
            table.load_hnsw_fields()?;
            Ok(Arc::new(table))
        })
    }
}

fn postgres_path(name: &str) -> PathBuf {
    PathBuf::from(format!("postgres:{name}"))
}

/// The thread running a state mutation uses the connection that holds its transaction; every
/// other operation autocommits on a free connection of the pool. A closed connection reconnects
/// before its next use.
struct PgSession {
    url: String,
    schema: String,
    pool: Vec<Mutex<Client>>,
    transaction: Mutex<Client>,
    transaction_owner: Mutex<Option<ThreadId>>,
}

const POOL_CONNECTIONS: usize = 8;

impl PgSession {
    fn connect(url: &str, schema: &str) -> ZResult<Self> {
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
        let pool = (0..POOL_CONNECTIONS)
            .map(|_| open_connection(url, schema).map(Mutex::new))
            .collect::<ZResult<Vec<_>>>()?;
        Ok(Self {
            url: url.to_string(),
            schema: schema.to_string(),
            pool,
            transaction: Mutex::new(open_connection(url, schema)?),
            transaction_owner: Mutex::new(None),
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

    fn with_client<T>(
        &self,
        mut run: impl FnMut(&mut Client) -> Result<T, postgres::Error>,
    ) -> ZResult<T> {
        let in_transaction = *self.transaction_owner.lock() == Some(thread::current().id());
        let mut client = if in_transaction {
            self.transaction.lock()
        } else {
            self.pool
                .iter()
                .find_map(|client| client.try_lock())
                .unwrap_or_else(|| self.pool[pool_slot(self.pool.len())].lock())
        };
        // A transaction's connection must not reconnect: its uncommitted writes are gone.
        if in_transaction {
            return run(&mut client).map_err(pg_error);
        }
        if client.is_closed() {
            *client = open_connection(&self.url, &self.schema)?;
        }
        match run(&mut client) {
            // A closed or terminated connection never ran the statement, so it runs once more.
            Err(error) if connection_lost(&error) => {
                *client = open_connection(&self.url, &self.schema)?;
                run(&mut client).map_err(pg_error)
            }
            result => result.map_err(pg_error),
        }
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

/// A pool slot for a caller that found every connection busy, spread by thread.
fn pool_slot(len: usize) -> usize {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    thread::current().id().hash(&mut hasher);
    (hasher.finish() % len as u64) as usize
}

impl MemoryTransactions for PgSession {
    fn begin(&self) -> ZResult<()> {
        let mut owner = self.transaction_owner.lock();
        if owner.is_some() {
            return Err(Status::internal("a postgres transaction is already open"));
        }
        let mut client = self.transaction.lock();
        if client.is_closed() {
            *client = open_connection(&self.url, &self.schema)?;
        }
        client.batch_execute("BEGIN").map_err(pg_error)?;
        *owner = Some(thread::current().id());
        Ok(())
    }

    fn commit(&self) -> ZResult<()> {
        if self.transaction_owner.lock().take().is_none() {
            return Err(Status::internal("no postgres transaction to commit"));
        }
        self.transaction
            .lock()
            .batch_execute("COMMIT")
            .map_err(pg_error)
    }

    fn rollback(&self) -> ZResult<()> {
        if self.transaction_owner.lock().take().is_none() {
            return Ok(());
        }
        self.transaction
            .lock()
            .batch_execute("ROLLBACK")
            .map_err(pg_error)
    }
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
        for column in self.columns.iter().filter(|column| column.indexed) {
            if column.kind == ColumnKind::Vector {
                if vector_index {
                    ddl.push_str(&hnsw_index_ddl(&self.name, column, None));
                    self.hnsw_fields.lock().insert(column.name.clone());
                }
            } else if column.kind == ColumnKind::TextArray {
                ddl.push_str(&format!(
                    "CREATE INDEX ON {table} USING gin ({});",
                    quote(&column.name)
                ));
            } else if column.name != "space_id" {
                ddl.push_str(&format!(
                    "CREATE INDEX ON {table} ({});",
                    quote(&column.name)
                ));
            }
        }
        // Scope filters test space, then tenant and user, so one index serves a store of many users.
        ddl.push_str(&format!(
            "CREATE INDEX ON {table} (space_id, tenant_id, user_id);"
        ));
        self.session
            .with_client(|client| client.batch_execute(&ddl))
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

    fn delete(&self, pks: Vec<String>) -> ZResult<Vec<Status>> {
        let sql = format!(
            "DELETE FROM {} WHERE pk = ANY($1) RETURNING pk",
            quote(&self.name)
        );
        let deleted = self
            .session
            .with_client(|client| client.query(&sql, &[&pks]))?
            .iter()
            .map(|row| row.get::<_, String>(0))
            .collect::<std::collections::HashSet<_>>();
        Ok(pks
            .iter()
            .map(|pk| match deleted.contains(pk) {
                true => Status::default(),
                false => Status::not_found(format!("pk {pk} not found")),
            })
            .collect())
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
        let mut params: Vec<Param> = vec![Box::new(Vector::from(query.query_vector.clone()))];
        let filter = self.filter_clause(&query, &mut params)?;
        let field = quote(&vector.name);
        if self.hnsw_fields.lock().contains(&vector.name) {
            // A second sort key would keep the planner from using the HNSW index.
            let dim = vector.dimension;
            let tail =
                format!("WHERE {field} IS NOT NULL AND ({filter}) ORDER BY distance LIMIT {limit}");
            let distance = format!("({field})::halfvec({dim}) <=> $1::vector::halfvec({dim})");
            return self.select(&columns, Some(&distance), &tail, params);
        }
        let tail = format!(
            "WHERE {field} IS NOT NULL AND ({filter}) ORDER BY distance, seq LIMIT {limit}"
        );
        self.select(&columns, Some(&format!("{field} <=> $1")), &tail, params)
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

    fn read_only(&self) -> bool {
        false
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

fn pg_error(error: postgres::Error) -> Status {
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
pub(crate) fn test_url() -> Option<String> {
    std::env::var("FINCH_MEMORY_TEST_POSTGRES_URL").ok()
}

/// Each test store path maps to its own schema, so a reopen by path finds the same tables.
#[cfg(test)]
pub(crate) fn test_store_name(path: &std::path::Path) -> String {
    let hash = crate::ingest::stable_hash_hex(&[&path.to_string_lossy()]);
    format!("t_{}", &hash[..hash.len().min(32)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryScope;

    #[test]
    fn reads_reconnect_after_the_server_drops_every_pooled_connection() {
        let Some(url) = test_url() else {
            return;
        };
        let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
        let store = MemoryStore::create_postgres(&url, "t_reconnect", 3, false).unwrap();
        let mut admin = Client::connect(&url, NoTls).unwrap();
        admin
            .execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity \
                 WHERE datname = current_database() AND pid <> pg_backend_pid()",
                &[],
            )
            .unwrap();
        for _ in 0..POOL_CONNECTIONS + 1 {
            assert!(store
                .scan_claims(&MemoryScope::new("s"), 10, None)
                .unwrap()
                .is_empty());
        }
    }
}

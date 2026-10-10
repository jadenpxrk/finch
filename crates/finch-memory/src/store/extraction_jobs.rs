//! The queue of episodes that wait for the caller's extractor. Every episode write adds one job
//! in the same transaction. Workers lease jobs, run the extractor, and complete or fail them.

use super::state_records::system_time_ms;
use super::MemoryStore;
use crate::postgres::pg_error;
use crate::types::{MemoryId, MemoryScope};
use finch_types::{Status, ZResult};
use postgres::types::ToSql;
use postgres::Row;
use serde::{Deserialize, Serialize};

/// Where an extraction job is in its life.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtractionJobStatus {
    /// The job waits for a worker.
    Pending,
    /// A worker holds the job until its lease expires.
    Leased,
    /// A worker extracted the episode. Completion deletes the job, so only its return value
    /// holds this status.
    Done,
    /// The job failed its maximum number of attempts. It blocks the later jobs of its scope.
    Failed,
    /// A caller skipped the failed job, so the later jobs of its scope can run.
    Skipped,
}

/// One episode that waits for extraction, or that a worker extracted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractionJob {
    /// Id of the job.
    pub id: MemoryId,
    /// Scope of the episode.
    pub scope: MemoryScope,
    /// Id of the episode to extract.
    pub episode_id: MemoryId,
    /// Sequence number of the episode. Jobs of one scope run in this order, then by id.
    pub sequence_no: i64,
    /// Event time of the episode in Unix ms. `None` means unknown.
    pub event_time_ms: Option<i64>,
    /// Where the job is in its life.
    pub status: ExtractionJobStatus,
    /// Number of failed attempts and expired leases.
    pub attempt_count: u32,
    /// Id of the worker that holds or completed the job.
    pub lease_owner: Option<String>,
    /// End of the current lease in Unix ms.
    pub lease_expires_at_ms: Option<i64>,
    /// Error of the last failed attempt.
    pub last_error: Option<String>,
    /// Time the store wrote the episode, in Unix ms.
    pub created_at_ms: i64,
    /// Time of the last change to the job, in Unix ms.
    pub updated_at_ms: i64,
}

/// A worker's request for jobs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractionLeaseRequest {
    /// Id of the worker. Only this worker can complete or fail the leased jobs.
    pub worker_id: String,
    /// Maximum number of jobs to lease.
    pub limit: usize,
    /// Length of the lease in ms. After it, another worker can lease the job.
    pub lease_ms: i64,
    /// Current time in Unix ms. `None` means the system clock.
    #[serde(default)]
    pub now_ms: Option<i64>,
    /// Number of attempts after which a job with an expired lease stops at `Failed`. Each
    /// expired lease counts as one attempt. Must be 1 or more.
    pub max_attempts: u32,
}

/// A worker's report that the extraction of a leased job failed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractionFailure {
    /// Id of the job.
    pub job_id: MemoryId,
    /// Id of the worker that holds the lease.
    pub worker_id: String,
    /// Error text to keep on the job.
    pub error: String,
    /// Number of failed attempts after which the job stops at `Failed`. Must be 1 or more.
    pub max_attempts: u32,
}

/// A worker's leased job that a state mutation batch completes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractionJobCompletion {
    /// Id of the job.
    pub job_id: MemoryId,
    /// Id of the worker that holds the lease.
    pub worker_id: String,
}

/// The jobs of a scope and its narrower scopes that are not complete or skipped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractionBacklog {
    /// Number of pending, leased, and failed jobs.
    pub jobs: usize,
    /// Number of failed jobs.
    pub failed_jobs: usize,
    /// Write time of the oldest of these jobs, in Unix ms. `None` means no job waits.
    pub oldest_created_at_ms: Option<i64>,
}

// Array equality treats two nulls as equal, so it matches an exact scope, and a btree index can
// serve it, unlike `IS NOT DISTINCT FROM`.
const SCOPE_IDS_E: &str = "ARRAY[e.tenant_id, e.user_id, e.agent_id, e.project_id, e.thread_id]";
const SCOPE_IDS_J: &str = "ARRAY[j.tenant_id, j.user_id, j.agent_id, j.project_id, j.thread_id]";

/// The index that the scope check of a lease uses: open jobs by exact scope and order.
pub(crate) const OPEN_JOBS_INDEX_DDL: &str =
    "CREATE INDEX IF NOT EXISTS memory_extraction_jobs_open ON \
     memory_extraction_jobs (space_id, (ARRAY[tenant_id, user_id, agent_id, project_id, \
     thread_id]), sequence_no, pk) WHERE status IN ('pending', 'leased', 'failed')";

const JOB_COLUMNS: &str = "pk, space_id, tenant_id, user_id, agent_id, project_id, thread_id, \
     episode_id, sequence_no, event_time_ms, status, attempt_count, lease_owner, \
     lease_expires_at_ms, last_error, created_at_ms, updated_at_ms";

impl MemoryStore {
    /// Leases up to `limit` jobs to the worker, in sequence order. A job is free when it is
    /// pending or its lease expired, and no earlier job of the same exact scope is pending,
    /// leased, or failed. Jobs of different scopes lease in parallel. Two workers never lease
    /// one job at the same time. An expired lease counts as one attempt, and a job whose expired
    /// leases reach `max_attempts` becomes failed.
    pub fn lease_extraction_jobs(
        &self,
        request: &ExtractionLeaseRequest,
    ) -> ZResult<Vec<ExtractionJob>> {
        if request.worker_id.is_empty() {
            return Err(Status::invalid_argument("worker_id must not be empty"));
        }
        if request.lease_ms <= 0 {
            return Err(Status::invalid_argument("lease_ms must be positive"));
        }
        if request.max_attempts == 0 {
            return Err(Status::invalid_argument("max_attempts must be 1 or more"));
        }
        if request.limit == 0 {
            return Ok(Vec::new());
        }
        let now = request.now_ms.unwrap_or_else(system_time_ms);
        let expires = now.saturating_add(request.lease_ms);
        let limit = i64::try_from(request.limit).unwrap_or(i64::MAX);
        let max_attempts = i64::from(request.max_attempts);
        // A worker that dies holding a lease never calls fail, so its expiry is the attempt.
        let fail_expired = "WITH expired AS ( \
               SELECT pk AS job_pk FROM memory_extraction_jobs \
               WHERE status = 'leased' AND lease_expires_at_ms <= $1 AND attempt_count + 1 >= $2 \
               FOR UPDATE SKIP LOCKED) \
             UPDATE memory_extraction_jobs SET status = 'failed', \
               attempt_count = attempt_count + 1, lease_owner = NULL, lease_expires_at_ms = NULL, \
               last_error = 'the lease expired before the worker completed the job', \
               updated_at_ms = $1, seq = nextval('write_seq') \
             FROM expired WHERE pk = expired.job_pk";
        // An earlier job holds the extractor state that the later jobs of its scope read.
        let lease = format!(
            "WITH free AS ( \
               SELECT j.pk AS job_pk FROM memory_extraction_jobs j \
               WHERE (j.status = 'pending' OR (j.status = 'leased' \
                   AND j.lease_expires_at_ms <= $1 AND j.attempt_count + 1 < $5)) \
                 AND NOT EXISTS ( \
                   SELECT 1 FROM memory_extraction_jobs e \
                   WHERE e.status IN ('pending', 'leased', 'failed') \
                     AND e.space_id = j.space_id AND {SCOPE_IDS_E} = {SCOPE_IDS_J} \
                     AND (e.sequence_no, e.pk) < (j.sequence_no, j.pk)) \
               ORDER BY j.sequence_no, j.pk LIMIT $2 \
               FOR UPDATE OF j SKIP LOCKED) \
             UPDATE memory_extraction_jobs SET status = 'leased', \
               attempt_count = attempt_count + CASE WHEN status = 'leased' THEN 1 ELSE 0 END, \
               lease_owner = $3, lease_expires_at_ms = $4, updated_at_ms = $1, \
               seq = nextval('write_seq') \
             FROM free WHERE pk = free.job_pk RETURNING {JOB_COLUMNS}"
        );
        let rows = self.session.with_client(|client| {
            client.execute(fail_expired, &[&now, &max_attempts])?;
            let params: [&(dyn ToSql + Sync); 5] =
                [&now, &limit, &request.worker_id, &expires, &max_attempts];
            client.query(&lease, &params)
        })?;
        let mut jobs = rows.iter().map(job_from_row).collect::<ZResult<Vec<_>>>()?;
        jobs.sort_by(|a, b| (a.sequence_no, &a.id).cmp(&(b.sequence_no, &b.id)));
        Ok(jobs)
    }

    /// Deletes a completed job and returns it with status `Done`. Only the worker that holds
    /// its lease can complete it.
    pub fn complete_extraction_job(&self, job_id: &str, worker_id: &str) -> ZResult<ExtractionJob> {
        let mut job = self.transition_extraction_job(
            job_id,
            &format!("leased by {worker_id}"),
            "DELETE FROM memory_extraction_jobs \
             WHERE pk = $1 AND status = 'leased' AND lease_owner = $2",
            &[&worker_id],
        )?;
        job.status = ExtractionJobStatus::Done;
        job.lease_expires_at_ms = None;
        job.updated_at_ms = system_time_ms();
        Ok(job)
    }

    /// Records a failed attempt. The job returns to pending, or stops at failed when it reaches
    /// `max_attempts`. Only the worker that holds its lease can fail it.
    pub fn fail_extraction_job(&self, failure: &ExtractionFailure) -> ZResult<ExtractionJob> {
        if failure.max_attempts == 0 {
            return Err(Status::invalid_argument("max_attempts must be 1 or more"));
        }
        let max_attempts = i64::from(failure.max_attempts);
        self.transition_extraction_job(
            &failure.job_id,
            &format!("leased by {}", failure.worker_id),
            "UPDATE memory_extraction_jobs SET attempt_count = attempt_count + 1, \
               status = CASE WHEN attempt_count + 1 >= $5 THEN 'failed' ELSE 'pending' END, \
               lease_owner = NULL, lease_expires_at_ms = NULL, last_error = $4, updated_at_ms = $2 \
             WHERE pk = $1 AND status = 'leased' AND lease_owner = $3",
            &[
                &system_time_ms(),
                &failure.worker_id,
                &failure.error,
                &max_attempts,
            ],
        )
    }

    /// Returns a failed job to pending with no failed attempts, so a worker can lease it again.
    pub fn retry_extraction_job(&self, job_id: &str) -> ZResult<ExtractionJob> {
        self.transition_extraction_job(
            job_id,
            "failed",
            "UPDATE memory_extraction_jobs SET status = 'pending', attempt_count = 0, \
               updated_at_ms = $2 WHERE pk = $1 AND status = 'failed'",
            &[&system_time_ms()],
        )
    }

    /// Marks a failed job skipped, so the later jobs of its scope can run.
    pub fn skip_extraction_job(&self, job_id: &str) -> ZResult<ExtractionJob> {
        self.transition_extraction_job(
            job_id,
            "failed",
            "UPDATE memory_extraction_jobs SET status = 'skipped', updated_at_ms = $2 \
             WHERE pk = $1 AND status = 'failed'",
            &[&system_time_ms()],
        )
    }

    /// Counts the jobs of `scope` and its narrower scopes that are not complete or skipped. A
    /// nonzero count tells a reader that the extracted state is behind the stored episodes.
    pub fn extraction_backlog(&self, scope: &MemoryScope) -> ZResult<ExtractionBacklog> {
        let row = self.session.with_client(|client| {
            client.query_one(
                "SELECT count(*), count(*) FILTER (WHERE status = 'failed'), min(created_at_ms) \
                 FROM memory_extraction_jobs \
                 WHERE status IN ('pending', 'leased', 'failed') AND space_id = $1 \
                   AND ($2::text IS NULL OR tenant_id = $2) AND ($3::text IS NULL OR user_id = $3) \
                   AND ($4::text IS NULL OR agent_id = $4) AND ($5::text IS NULL OR project_id = $5) \
                   AND ($6::text IS NULL OR thread_id = $6)",
                &[
                    &scope.space_id as &(dyn ToSql + Sync),
                    &scope.tenant_id,
                    &scope.user_id,
                    &scope.agent_id,
                    &scope.project_id,
                    &scope.thread_id,
                ],
            )
        })?;
        let count = |index: usize| -> ZResult<usize> {
            let count = row.try_get::<_, i64>(index).map_err(pg_error)?;
            usize::try_from(count).map_err(|_| Status::internal(format!("job count {count}")))
        };
        Ok(ExtractionBacklog {
            jobs: count(0)?,
            failed_jobs: count(1)?,
            oldest_created_at_ms: row.try_get(2).map_err(pg_error)?,
        })
    }

    /// Runs `statement`, an update or delete of one job that binds $1 to the job id and then
    /// `params`. When it changes no row, the error says whether the job is missing or not
    /// `expected`.
    fn transition_extraction_job(
        &self,
        job_id: &str,
        expected: &str,
        statement: &str,
        params: &[&(dyn ToSql + Sync)],
    ) -> ZResult<ExtractionJob> {
        let sql = format!("{statement} RETURNING {JOB_COLUMNS}");
        let bound = std::iter::once(&job_id as &(dyn ToSql + Sync))
            .chain(params.iter().copied())
            .collect::<Vec<_>>();
        let row = self
            .session
            .with_client(|client| client.query_opt(&sql, &bound))?;
        if let Some(row) = row {
            return job_from_row(&row);
        }
        let current = self.session.with_client(|client| {
            client.query_opt(
                &format!("SELECT {JOB_COLUMNS} FROM memory_extraction_jobs WHERE pk = $1"),
                &[&job_id],
            )
        })?;
        match current {
            None => Err(Status::not_found(format!("no extraction job {job_id}"))),
            Some(row) => {
                let job = job_from_row(&row)?;
                Err(Status::invalid_argument(format!(
                    "extraction job {job_id} is {:?} with lease owner {:?}, not {expected}",
                    job.status, job.lease_owner
                )))
            }
        }
    }
}

fn job_from_row(row: &Row) -> ZResult<ExtractionJob> {
    let status = row.try_get::<_, String>("status").map_err(pg_error)?;
    let attempt_count = row.try_get::<_, i64>("attempt_count").map_err(pg_error)?;
    Ok(ExtractionJob {
        id: row.try_get("pk").map_err(pg_error)?,
        scope: MemoryScope {
            space_id: row.try_get("space_id").map_err(pg_error)?,
            tenant_id: row.try_get("tenant_id").map_err(pg_error)?,
            user_id: row.try_get("user_id").map_err(pg_error)?,
            agent_id: row.try_get("agent_id").map_err(pg_error)?,
            project_id: row.try_get("project_id").map_err(pg_error)?,
            thread_id: row.try_get("thread_id").map_err(pg_error)?,
        },
        episode_id: row.try_get("episode_id").map_err(pg_error)?,
        sequence_no: row.try_get("sequence_no").map_err(pg_error)?,
        event_time_ms: row.try_get("event_time_ms").map_err(pg_error)?,
        status: serde_json::from_value(serde_json::Value::String(status))
            .map_err(|e| Status::internal(format!("extraction job status: {e}")))?,
        attempt_count: u32::try_from(attempt_count).map_err(|_| {
            Status::internal(format!("extraction job attempt count {attempt_count}"))
        })?,
        lease_owner: row.try_get("lease_owner").map_err(pg_error)?,
        lease_expires_at_ms: row.try_get("lease_expires_at_ms").map_err(pg_error)?,
        last_error: row.try_get("last_error").map_err(pg_error)?,
        created_at_ms: row.try_get("created_at_ms").map_err(pg_error)?,
        updated_at_ms: row.try_get("updated_at_ms").map_err(pg_error)?,
    })
}

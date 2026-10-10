use super::*;
use crate::{
    ExtractionFailure, ExtractionJobCompletion, ExtractionJobStatus, ExtractionLeaseRequest,
};
use std::sync::Barrier;

fn thread(id: &str) -> MemoryScope {
    let mut scope = scope();
    scope.thread_id = Some(id.to_string());
    scope
}

fn ingest(store: &MemoryStore, scope: &MemoryScope, sequence_no: i64) -> MemoryId {
    ingested(store, scope, sequence_no).episode.id
}

fn ingested(store: &MemoryStore, scope: &MemoryScope, sequence_no: i64) -> crate::IngestedEpisode {
    let input = EpisodeInput {
        id: None,
        scope: scope.clone(),
        visibility: Visibility::Private,
        policy_tags: Vec::new(),
        source_kind: SourceKind::UserMessage,
        actor: ActorKind::User,
        sequence_no,
        event_time_ms: Some(sequence_no),
        valid_from_ms: None,
        valid_to_ms: None,
        raw_text: format!("message {sequence_no}"),
        blob_ref: None,
        mime_type: None,
        causal_parent_ids: Vec::new(),
        metadata_json: None,
    };
    store
        .ingest_episode(input, 1_000 + sequence_no, &ChunkOptions::default())
        .unwrap()
}

fn lease(worker: &str, limit: usize, now_ms: i64) -> ExtractionLeaseRequest {
    ExtractionLeaseRequest {
        worker_id: worker.to_string(),
        limit,
        lease_ms: 100,
        now_ms: Some(now_ms),
        max_attempts: 3,
    }
}

fn episode_ids(jobs: &[crate::ExtractionJob]) -> Vec<MemoryId> {
    jobs.iter().map(|job| job.episode_id.clone()).collect()
}

fn failure(job_id: &str, worker: &str, max_attempts: u32) -> ExtractionFailure {
    ExtractionFailure {
        job_id: job_id.to_string(),
        worker_id: worker.to_string(),
        error: "extractor timed out".to_string(),
        max_attempts,
    }
}

#[test]
fn episode_writes_create_pending_jobs_that_lease_in_sequence_order() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let store = crate::test_store(&temp_dir("extraction_jobs_order"), 3).unwrap();
    let a = thread("a");
    let third = ingest(&store, &a, 3);
    let first = ingest(&store, &a, 1);
    let second = ingest(&store, &a, 2);
    let batch = ingest_episode(
        EpisodeInput {
            id: Some("batch".to_string()),
            scope: a.clone(),
            visibility: Visibility::Private,
            policy_tags: Vec::new(),
            source_kind: SourceKind::UserMessage,
            actor: ActorKind::User,
            sequence_no: 4,
            event_time_ms: None,
            valid_from_ms: None,
            valid_to_ms: None,
            raw_text: "batch message".to_string(),
            blob_ref: None,
            mime_type: None,
            causal_parent_ids: Vec::new(),
            metadata_json: None,
        },
        1_004,
        &ChunkOptions::default(),
    );
    let embeddings = vec![vec![1.0, 0.0, 0.0]; batch.spans.len()];
    store
        .append_ingested_episodes_with_embeddings(&[(batch, embeddings)])
        .unwrap();

    let backlog = store.extraction_backlog(&scope()).unwrap();
    assert_eq!(backlog.jobs, 4);
    assert_eq!(backlog.oldest_created_at_ms, Some(1_001));

    let mut order = Vec::new();
    for now in 0..5 {
        let jobs = store.lease_extraction_jobs(&lease("w", 10, now)).unwrap();
        assert!(jobs.len() <= 1, "one scope leases one job at a time");
        for job in jobs {
            assert_eq!(job.status, ExtractionJobStatus::Leased);
            store.complete_extraction_job(&job.id, "w").unwrap();
            order.push(job.episode_id);
        }
    }
    assert_eq!(order, vec![first, second, third, "batch".to_string()]);
}

#[test]
fn concurrent_workers_never_lease_the_same_job() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let store = crate::test_store(&temp_dir("extraction_jobs_concurrent"), 3).unwrap();
    let workers = 8;
    let docs = (0..400)
        .map(|index| {
            let episode = EpisodeRecord {
                id: format!("ep{index}"),
                scope: thread(&format!("t{index}")),
                status: MemoryStatus::Active,
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                source_kind: SourceKind::UserMessage,
                actor: ActorKind::User,
                sequence_no: 1,
                temporal: crate::types::TemporalFields {
                    created_at_ms: 0,
                    ingested_at_ms: 0,
                    event_time_ms: None,
                    valid_from_ms: None,
                    valid_to_ms: None,
                },
                raw_text: String::new(),
                blob_ref: None,
                mime_type: None,
                content_hash: String::new(),
                causal_parent_ids: Vec::new(),
                metadata_json: None,
            };
            (
                IngestedEpisode {
                    episode,
                    spans: Vec::new(),
                },
                Vec::new(),
            )
        })
        .collect::<Vec<_>>();
    store
        .append_ingested_episodes_with_embeddings(&docs)
        .unwrap();
    let barrier = Barrier::new(workers);
    let leased = std::thread::scope(|threads| {
        let handles = (0..workers)
            .map(|worker| {
                let (store, barrier) = (&store, &barrier);
                threads.spawn(move || {
                    let worker = format!("w{worker}");
                    barrier.wait();
                    let mut ids = Vec::new();
                    loop {
                        let jobs = store.lease_extraction_jobs(&lease(&worker, 5, 0)).unwrap();
                        if jobs.is_empty() {
                            return ids;
                        }
                        ids.extend(jobs.into_iter().map(|job| job.id));
                    }
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    let all = leased.concat();
    assert_eq!(all.iter().collect::<BTreeSet<_>>().len(), 400);
    assert_eq!(all.len(), 400, "a job leased to two workers");
}

#[test]
fn a_leased_or_failed_job_blocks_later_jobs_of_its_scope_only() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let store = crate::test_store(&temp_dir("extraction_jobs_blocking"), 3).unwrap();
    let (a, b) = (thread("a"), thread("b"));
    let a1 = ingest(&store, &a, 1);
    let a2 = ingest(&store, &a, 2);
    let b1 = ingest(&store, &b, 1);
    let b2 = ingest(&store, &b, 2);

    let jobs = store.lease_extraction_jobs(&lease("w", 10, 0)).unwrap();
    let job_of = |episode: &MemoryId| {
        jobs.iter()
            .find(|job| &job.episode_id == episode)
            .map(|job| job.id.clone())
            .unwrap()
    };
    assert_eq!(jobs.len(), 2);
    let (a1_job, b1_job) = (job_of(&a1), job_of(&b1));
    assert!(store
        .lease_extraction_jobs(&lease("w", 10, 0))
        .unwrap()
        .is_empty());

    let failed = store
        .fail_extraction_job(&failure(&a1_job, "w", 1))
        .unwrap();
    assert_eq!(failed.status, ExtractionJobStatus::Failed);
    assert_eq!(failed.attempt_count, 1);
    store.complete_extraction_job(&b1_job, "w").unwrap();
    let jobs = store.lease_extraction_jobs(&lease("w", 10, 0)).unwrap();
    assert_eq!(episode_ids(&jobs), vec![b2]);
    assert_eq!(store.extraction_backlog(&a).unwrap().failed_jobs, 1);

    store.retry_extraction_job(&a1_job).unwrap();
    let jobs = store.lease_extraction_jobs(&lease("w", 10, 0)).unwrap();
    assert_eq!(episode_ids(&jobs), vec![a1]);
    let retried = store
        .fail_extraction_job(&failure(&a1_job, "w", 3))
        .unwrap();
    assert_eq!(retried.status, ExtractionJobStatus::Pending);
    assert_eq!(retried.last_error.as_deref(), Some("extractor timed out"));
    store
        .lease_extraction_jobs(&lease("w", 10, 0))
        .unwrap()
        .iter()
        .for_each(|job| {
            store
                .fail_extraction_job(&failure(&job.id, "w", 1))
                .unwrap();
        });

    store.skip_extraction_job(&a1_job).unwrap();
    let jobs = store.lease_extraction_jobs(&lease("w", 10, 0)).unwrap();
    assert_eq!(episode_ids(&jobs), vec![a2]);
}

#[test]
fn an_expired_lease_leases_again_and_only_the_owner_finishes_a_job() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let store = crate::test_store(&temp_dir("extraction_jobs_expiry"), 3).unwrap();
    ingest(&store, &scope(), 1);

    let job = store.lease_extraction_jobs(&lease("w1", 1, 1_000)).unwrap()[0].clone();
    assert!(store
        .lease_extraction_jobs(&lease("w2", 1, 1_099))
        .unwrap()
        .is_empty());
    let again = store.lease_extraction_jobs(&lease("w2", 1, 1_100)).unwrap();
    assert_eq!(again.len(), 1);
    assert_eq!(again[0].id, job.id);
    assert_eq!(again[0].lease_owner.as_deref(), Some("w2"));

    assert!(store.complete_extraction_job(&job.id, "w1").is_err());
    assert!(store
        .fail_extraction_job(&failure(&job.id, "w1", 1))
        .is_err());
    assert!(store
        .complete_extraction_job("xjob_missing", "w2")
        .is_err_and(|status| status.is_not_found()));
    let done = store.complete_extraction_job(&job.id, "w2").unwrap();
    assert_eq!(done.status, ExtractionJobStatus::Done);
    assert!(store.complete_extraction_job(&job.id, "w2").is_err());
}

fn extraction_batch(
    claim_id: &str,
    evidence: &crate::IngestedEpisode,
    job: &crate::ExtractionJob,
    worker: &str,
) -> StateMutationBatch {
    let mut claim = make_claim(
        &scope(),
        claim_id,
        "user",
        "city",
        Some("Denver"),
        10,
        (ClaimKind::Fact, ClaimPolarity::Affirmative),
    );
    claim.source_span_ids = vec![evidence.spans[0].id.clone()];
    claim.source_episode_ids = vec![evidence.episode.id.clone()];
    StateMutationBatch {
        extraction_job: Some(ExtractionJobCompletion {
            job_id: job.id.clone(),
            worker_id: worker.to_string(),
        }),
        scope: scope(),
        transaction_time_ms: 20,
        propagation_seed: claim_id.to_string(),
        max_rule_hops: 8,
        claims: vec![claim],
        entities: Vec::new(),
        rules: Vec::new(),
        slot_aliases: Vec::new(),
        corrections: Vec::new(),
    }
}

fn claim_ids(store: &MemoryStore) -> Vec<MemoryId> {
    let claims = store.scan_claims(&scope(), usize::MAX, None).unwrap();
    claims.into_iter().map(|claim| claim.id).collect()
}

#[test]
fn a_batch_completes_its_job_or_writes_nothing() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let store = crate::test_store(&temp_dir("extraction_jobs_batch"), 3).unwrap();
    let evidence = ingested(&store, &scope(), 1);
    let job = store.lease_extraction_jobs(&lease("w1", 1, 1_000)).unwrap()[0].clone();
    store.lease_extraction_jobs(&lease("w2", 1, 1_100)).unwrap();

    // w1 lost its lease, so its late batch must not write.
    assert!(store
        .apply_state_mutation_batch(extraction_batch("claim_w1", &evidence, &job, "w1"))
        .is_err());
    assert_eq!(claim_ids(&store), Vec::<MemoryId>::new());

    store
        .apply_state_mutation_batch(extraction_batch("claim_w2", &evidence, &job, "w2"))
        .unwrap();
    assert_eq!(store.extraction_backlog(&scope()).unwrap().jobs, 0);
    // A retry after a crash finds the job complete and writes nothing twice.
    assert!(store
        .apply_state_mutation_batch(extraction_batch("claim_w2_retry", &evidence, &job, "w2"))
        .is_err());
    assert_eq!(claim_ids(&store), vec!["claim_w2".to_string()]);
}

#[test]
fn backlog_counts_drop_after_completion() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let store = crate::test_store(&temp_dir("extraction_jobs_backlog"), 3).unwrap();
    let (a, b) = (thread("a"), thread("b"));
    ingest(&store, &a, 1);
    ingest(&store, &b, 2);
    assert_eq!(store.extraction_backlog(&scope()).unwrap().jobs, 2);
    assert_eq!(store.extraction_backlog(&a).unwrap().jobs, 1);

    let jobs = store.lease_extraction_jobs(&lease("w", 10, 0)).unwrap();
    assert_eq!(store.extraction_backlog(&scope()).unwrap().jobs, 2);
    store.complete_extraction_job(&jobs[0].id, "w").unwrap();
    assert_eq!(store.extraction_backlog(&scope()).unwrap().jobs, 1);
    store.complete_extraction_job(&jobs[1].id, "w").unwrap();
    let backlog = store.extraction_backlog(&scope()).unwrap();
    assert_eq!(backlog.jobs, 0);
    assert_eq!(backlog.oldest_created_at_ms, None);
}

#[test]
fn expired_leases_count_as_attempts_until_the_job_fails() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let store = crate::test_store(&temp_dir("extraction_jobs_expired_attempts"), 3).unwrap();
    let first = ingest(&store, &scope(), 1);
    ingest(&store, &scope(), 2);

    // Each lease lasts 100 ms and no worker completes it, as when the worker dies.
    let mut leased = Vec::new();
    for now in [0, 100, 200] {
        let jobs = store.lease_extraction_jobs(&lease("w", 10, now)).unwrap();
        assert_eq!(episode_ids(&jobs), vec![first.clone()]);
        leased.push(jobs[0].attempt_count);
    }
    assert_eq!(leased, vec![0, 1, 2]);

    assert!(store
        .lease_extraction_jobs(&lease("w", 10, 300))
        .unwrap()
        .is_empty());
    let backlog = store.extraction_backlog(&scope()).unwrap();
    assert_eq!((backlog.jobs, backlog.failed_jobs), (2, 1));
    let job_id = format!("xjob_{first}");
    let skipped = store.skip_extraction_job(&job_id).unwrap();
    assert_eq!(skipped.attempt_count, 3);
    assert_eq!(
        skipped.last_error.as_deref(),
        Some("the lease expired before the worker completed the job")
    );
    assert_eq!(
        store
            .lease_extraction_jobs(&lease("w", 10, 300))
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn completion_deletes_the_job() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let store = crate::test_store(&temp_dir("extraction_jobs_complete_deletes"), 3).unwrap();
    ingest(&store, &scope(), 1);
    let job = store.lease_extraction_jobs(&lease("w", 1, 0)).unwrap()[0].clone();

    let done = store.complete_extraction_job(&job.id, "w").unwrap();
    assert_eq!(done.status, ExtractionJobStatus::Done);
    assert_eq!(done.episode_id, job.episode_id);
    let rows = store
        .session
        .with_client(|client| client.query_one("SELECT count(*) FROM memory_extraction_jobs", &[]))
        .unwrap()
        .get::<_, i64>(0);
    assert_eq!(rows, 0);
    assert!(store
        .complete_extraction_job(&job.id, "w")
        .is_err_and(|status| status.is_not_found()));
}

#[test]
fn the_lease_scope_check_reads_the_open_jobs_index() {
    let _guard = crate::TEST_STORE_MUTEX.lock().unwrap();
    let store = crate::test_store(&temp_dir("extraction_jobs_open_index"), 3).unwrap();
    let plan = store
        .session
        .with_client(|client| {
            client.batch_execute("SET enable_seqscan = off")?;
            let plan = client.query(
                "EXPLAIN SELECT 1 FROM memory_extraction_jobs e \
                 WHERE e.status IN ('pending', 'leased', 'failed') AND e.space_id = 's' \
                   AND ARRAY[e.tenant_id, e.user_id, e.agent_id, e.project_id, e.thread_id] \
                       = ARRAY[NULL, NULL, NULL, NULL, 't']::text[] \
                   AND (e.sequence_no, e.pk) < (5, 'x')",
                &[],
            );
            client.batch_execute("RESET enable_seqscan")?;
            plan
        })
        .unwrap()
        .iter()
        .map(|row| row.get::<_, String>(0))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(plan.contains("memory_extraction_jobs_open"), "{plan}");
    assert!(plan.contains("Index Cond"), "{plan}");
}

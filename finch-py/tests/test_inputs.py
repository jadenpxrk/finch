"""Timestamp, scope, and tenant inputs of the Python binding."""

import os
import time
import unittest
from datetime import datetime, timezone

from finch.memory import MemoryScope, _ms, _ms_or_now, _scope


class TimestampTest(unittest.TestCase):
    def setUp(self):
        # A zone that is not UTC shows a string that is read as local time.
        self._tz = os.environ.get("TZ")
        os.environ["TZ"] = "America/Los_Angeles"
        time.tzset()

    def tearDown(self):
        if self._tz is None:
            del os.environ["TZ"]
        else:
            os.environ["TZ"] = self._tz
        time.tzset()

    def test_a_time_without_a_zone_is_utc(self):
        self.assertEqual(_ms("1970-01-01T00:00:01"), 1000)
        self.assertEqual(_ms(datetime(1970, 1, 1, 0, 0, 1)), 1000)
        self.assertEqual(_ms("1970-01-01T00:00:01Z"), 1000)
        self.assertEqual(_ms(datetime(1970, 1, 1, 0, 0, 1, tzinfo=timezone.utc)), 1000)
        self.assertEqual(_ms("1970-01-01T01:00:01+01:00"), 1000)

    def test_epoch_zero_is_a_time_and_not_a_missing_one(self):
        self.assertEqual(_ms_or_now(0), 0)
        self.assertEqual(_ms_or_now("1970-01-01T00:00:00Z"), 0)
        self.assertGreater(_ms_or_now(None), 0)


class ScopeTest(unittest.TestCase):
    def test_a_scope_keeps_its_ids(self):
        want = {
            "space_id": "s",
            "tenant_id": None,
            "user_id": "u1",
            "agent_id": None,
            "project_id": None,
            "thread_id": None,
        }
        self.assertEqual(_scope(MemoryScope(space="s", user_id="u1")), want)
        self.assertEqual(_scope({"space_id": "s", "user_id": "u1"}), want)

    def test_a_misspelled_key_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "unknown keys: userId"):
            _scope({"space": "s", "userId": "u1"})


# Memory stores live in Postgres; the test needs a server with pgvector.
POSTGRES_URL = os.environ.get("FINCH_MEMORY_TEST_POSTGRES_URL")


@unittest.skipUnless(POSTGRES_URL, "set FINCH_MEMORY_TEST_POSTGRES_URL")
class ExtractionQueueTest(unittest.TestCase):
    def test_a_queue_call_with_a_tenant_reaches_only_the_jobs_of_that_tenant(self):
        from finch.memory import FinchMemory

        memory = FinchMemory.create(POSTGRES_URL, f"py_queue_{os.getpid()}_{int(time.time())}", 4)
        for tenant in ("a", "b"):
            scope = {"space": "s", "tenant_id": tenant}
            memory.ingest_episode({"scope": scope, "text": f"hello from {tenant}"})

        def lease_and_fail():
            jobs = memory.lease_extraction_jobs({"worker_id": "w", "tenant_id": "a"})["jobs"]
            self.assertEqual([job["scope"]["tenant_id"] for job in jobs], ["a"])
            failure = {"job_id": jobs[0]["id"], "worker_id": "w", "error": "timeout"}
            failed = memory.fail_extraction_job({**failure, "max_attempts": 1, "tenant_id": "a"})
            self.assertEqual(failed["status"], "failed")
            return jobs[0]["id"]

        job_id = lease_and_fail()
        with self.assertRaisesRegex(Exception, "no extraction job"):
            memory.retry_extraction_job(job_id, "b")
        self.assertEqual(memory.retry_extraction_job(job_id, "a")["status"], "pending")
        self.assertEqual(lease_and_fail(), job_id)
        self.assertEqual(memory.skip_extraction_job(job_id, "a")["status"], "skipped")

    def test_get_claim_takes_a_limit_that_has_no_effect(self):
        from finch.memory import FinchMemory

        memory = FinchMemory.create(POSTGRES_URL, f"py_claim_{os.getpid()}_{int(time.time())}", 4)
        self.assertIsNone(memory.get_claim("claim_missing", {"space": "s"}, limit=1))


@unittest.skipUnless(POSTGRES_URL, "set FINCH_MEMORY_TEST_POSTGRES_URL")
class SubQueryTest(unittest.TestCase):
    def test_a_sub_query_brings_in_the_hits_of_its_event(self):
        from finch.memory import FinchMemory

        memory = FinchMemory.create(POSTGRES_URL, f"py_subq_{os.getpid()}_{int(time.time())}", 4)
        scope = {"space": "s"}
        for text in ("The invoice was paid.", "The invoice draft went out.", "The dentist visit is set."):
            memory.ingest_episode({"scope": scope, "text": text})
        search = {"scope": scope, "query": "invoice", "top_k": 2, "query_embedding": [1.0, 0.0, 0.0, 0.0]}

        def texts(request):
            return [hit["span"]["text"] for hit in memory.search(request)["hits"]]

        self.assertNotIn("The dentist visit is set.", texts(search))
        dentist = {"query": "dentist", "query_embedding": [0.0, 1.0, 0.0, 0.0]}
        self.assertIn("The dentist visit is set.", texts({**search, "sub_queries": [dentist]}))
        with self.assertRaisesRegex(ValueError, "hybrid"):
            memory.search({**search, "mode": "keyword", "sub_queries": [dentist]})


if __name__ == "__main__":
    unittest.main()

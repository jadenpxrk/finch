from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest


class WalCrashRecoveryTest(unittest.TestCase):
    def test_wal_replay_recovers_after_abrupt_exit(self) -> None:
        import finch

        base = tempfile.mkdtemp(prefix="finch_wal_crash_py_")
        self.addCleanup(lambda: shutil.rmtree(base, ignore_errors=True))
        path = os.path.join(base, "col")

        helper = r"""
import os
import finch

path = os.environ["FINCH_CRASH_PATH"]

# Ensure WAL writes are durable in this test.
finch.init(
    query_threads=1,
    optimize_threads=1,
    wal_flush_every_docs=1,
    wal_fsync_every_docs=1,
)

schema = finch.CollectionSchema(
    name="crash",
    fields=[
        finch.FieldSchema(
            "id",
            finch.DataType.Int64,
            nullable=False,
            index_param=finch.InvertIndexParam(enable_range_optimization=True),
        ),
    ],
    vectors=[
        finch.VectorSchema(
            "emb",
            finch.DataType.VectorFp32,
            dimension=4,
        ),
    ],
)

col = finch.create_and_open(path, schema)
st = col.insert(
    [
        finch.Doc(id="a", fields={"id": 0}, vectors={"emb": [1.0, 0.0, 0.0, 0.0]}),
        finch.Doc(id="b", fields={"id": 1}, vectors={"emb": [0.0, 1.0, 0.0, 0.0]}),
        finch.Doc(id="c", fields={"id": 2}, vectors={"emb": [0.0, 0.0, 1.0, 0.0]}),
    ]
)
assert all(s.is_ok() for s in st), st

col.delete("c")

# Simulate an abrupt crash: skip destructors/finalizers.
os._exit(0)
"""

        proc = subprocess.run(
            [sys.executable, "-c", helper],
            env={**os.environ, "FINCH_CRASH_PATH": path},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            raise AssertionError(
                f"helper failed (rc={proc.returncode})\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )

        col = finch.open(path)
        self.assertEqual(col.stats.doc_count, 2)

        results = col.query(
            vectors=finch.VectorQuery(field_name="emb", vector=[1.0, 0.0, 0.0, 0.0]),
            topk=10,
            output_fields=[],
        )
        pks = {doc.id for doc in results}
        self.assertIn("a", pks)
        self.assertIn("b", pks)
        self.assertNotIn("c", pks)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


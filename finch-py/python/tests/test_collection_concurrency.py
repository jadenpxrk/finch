from __future__ import annotations

import shutil
import tempfile
import threading
import unittest

import finch


def _make_collection(path: str) -> finch.Collection:
    schema = finch.CollectionSchema(
        name="concurrency",
        fields=[
            finch.FieldSchema(
                "id",
                finch.DataType.INT64,
                nullable=False,
                index_param=finch.InvertIndexParam(enable_range_optimization=True),
            ),
            finch.FieldSchema(
                "name",
                finch.DataType.STRING,
                nullable=False,
                index_param=finch.InvertIndexParam(),
            ),
            finch.FieldSchema("weight", finch.DataType.FLOAT, nullable=True),
        ],
        vectors=[
            finch.VectorSchema(
                "dense",
                finch.DataType.VECTOR_FP32,
                16,
                index_param=finch.HnswIndexParam(),
            ),
            finch.VectorSchema(
                "sparse",
                finch.DataType.SPARSE_VECTOR_FP32,
                0,
                index_param=finch.HnswIndexParam(),
            ),
        ],
    )
    return finch.create_and_open(path, schema, finch.CollectionOption(read_only=False, enable_mmap=True))


class CollectionConcurrencyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            finch.init()
        except RuntimeError:
            pass

    def test_concurrent_query(self) -> None:
        try:
            import numpy as np
        except Exception:  # pragma: no cover
            self.skipTest("numpy is not available")

        root = tempfile.mkdtemp(prefix="finch_test_concurrency_")
        path = f"{root}/collection"
        try:
            col = _make_collection(path)

            docs = []
            for i in range(30):
                docs.append(
                    finch.Doc(
                        id=str(i),
                        fields={"id": i, "name": f"test_{i}", "weight": float(i)},
                        vectors={
                            "dense": np.random.random(16).tolist(),
                            "sparse": {1: float(i), 2: float(i * 2)},
                        },
                    )
                )
            res = col.insert(docs)
            self.assertEqual(len(res), 30)
            self.assertTrue(all(s.is_ok() for s in res))

            results: list[tuple[int, str, object]] = []

            def query_operation(thread_id: int) -> None:
                try:
                    out = col.query(filter=f"id > {thread_id}", topk=5)
                    results.append((thread_id, "query", len(out)))
                except Exception as e:  # pragma: no cover
                    results.append((thread_id, "exception", str(e)))

            threads = [threading.Thread(target=query_operation, args=(i,)) for i in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            exc = [r for r in results if r[1] == "exception"]
            self.assertEqual(exc, [])
            self.assertEqual(len([r for r in results if r[1] == "query"]), 5)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_concurrent_read_write_smoke(self) -> None:
        try:
            import numpy as np
        except Exception:  # pragma: no cover
            self.skipTest("numpy is not available")

        root = tempfile.mkdtemp(prefix="finch_test_concurrency_rw_")
        path = f"{root}/collection"
        try:
            col = _make_collection(path)
            results: list[tuple[int, str, object]] = []

            def insert_docs(thread_id: int) -> None:
                try:
                    docs = []
                    for i in range(5):
                        docs.append(
                            finch.Doc(
                                id=f"{thread_id}_{i}",
                                fields={
                                    "id": int(f"{thread_id}{i}"),
                                    "name": f"thread_{thread_id}_doc_{i}",
                                    "weight": float(i),
                                },
                                vectors={
                                    "dense": np.random.random(16).tolist(),
                                    "sparse": {1: float(i), 2: float(i * 2)},
                                },
                            )
                        )
                    out = col.insert(docs)
                    results.append((thread_id, "insert", len(out)))
                except Exception as e:  # pragma: no cover
                    results.append((thread_id, "insert_exception", str(e)))

            def query_docs(thread_id: int) -> None:
                try:
                    out = col.query(filter="id > 0", topk=10)
                    results.append((thread_id, "query", len(out)))
                except Exception as e:  # pragma: no cover
                    results.append((thread_id, "query_exception", str(e)))

            threads: list[threading.Thread] = []
            for i in range(3):
                threads.append(threading.Thread(target=insert_docs, args=(i,)))
            for i in range(3):
                threads.append(threading.Thread(target=query_docs, args=(i,)))
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            ok_ops = [r for r in results if r[1] in ("insert", "query")]
            self.assertGreater(len(ok_ops), 0)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)


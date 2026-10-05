from __future__ import annotations

import math
import shutil
import tempfile
import unittest

import finch


def _rrf_scores(results: dict[str, list[finch.Doc]], k: int = 60) -> dict[str, float]:
    out: dict[str, float] = {}
    for docs in results.values():
        for rank, doc in enumerate(docs):
            out[doc.id] = out.get(doc.id, 0.0) + (1.0 / (k + rank + 1))
    return out


def _normalize_score(score: float, metric: finch.MetricType) -> float:
    if metric == finch.MetricType.L2:
        return 1.0 - 2 * math.atan(score) / math.pi
    if metric == finch.MetricType.IP:
        return 0.5 + math.atan(score) / math.pi
    if metric == finch.MetricType.COSINE:
        return 1.0 - score / 2.0
    raise ValueError("unsupported metric")


def _weighted_scores(
    results: dict[str, list[finch.Doc]],
    weights: dict[str, float],
    metric: finch.MetricType,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for vector_name, docs in results.items():
        w = weights.get(vector_name, 1.0)
        for doc in docs:
            out[doc.id] = out.get(doc.id, 0.0) + (_normalize_score(doc.score, metric) * w)
    return out


class CollectionMultiVectorRerankTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            finch.init()
        except RuntimeError:
            pass

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="finch_test_multi_vec_")
        self.path = f"{self._tmp}/collection"

        schema = finch.CollectionSchema(
            name="multi_vector",
            fields=[
                finch.FieldSchema(
                    "id",
                    finch.DataType.INT64,
                    nullable=False,
                    index_param=finch.InvertIndexParam(enable_range_optimization=True),
                )
            ],
            vectors=[
                finch.VectorSchema("v1", finch.DataType.VECTOR_FP32, 4, index_param=finch.FlatIndexParam(metric_type=finch.MetricType.IP)),
                finch.VectorSchema("v2", finch.DataType.VECTOR_FP32, 4, index_param=finch.FlatIndexParam(metric_type=finch.MetricType.IP)),
            ],
        )
        self.col = finch.create_and_open(self.path, schema, finch.CollectionOption(read_only=False, enable_mmap=True))

        docs: list[finch.Doc] = []
        for i in range(10):
            docs.append(
                finch.Doc(
                    id=str(i),
                    fields={"id": i},
                    vectors={
                        "v1": [float(i + 1), 0.0, 0.0, 0.0],
                        "v2": [0.0, float(i + 1), 0.0, 0.0],
                    },
                )
            )
        res = self.col.insert(docs)
        self.assertTrue(all(s.ok() for s in res))

    def tearDown(self) -> None:
        try:
            self.col.destroy()
        finally:
            shutil.rmtree(self._tmp, ignore_errors=True)

    def test_query_multivector_rrf_integration(self) -> None:
        q1 = finch.VectorQuery(field_name="v1", vector=[1.0, 0.0, 0.0, 0.0])
        q2 = finch.VectorQuery(field_name="v2", vector=[0.0, 1.0, 0.0, 0.0])

        single = {
            "v1": self.col.query(q1, topk=10),
            "v2": self.col.query(q2, topk=10),
        }
        expected = _rrf_scores(single, k=60)

        reranker = finch.RrfReRanker(topn=5, rank_constant=60)
        out = self.col.query(vectors=[q1, q2], topk=10, reranker=reranker)
        self.assertEqual(len(out), 5)

        prev = float("inf")
        for doc in out:
            self.assertIn(doc.id, expected)
            self.assertAlmostEqual(doc.score, expected[doc.id], delta=1e-10)
            self.assertLessEqual(doc.score, prev)
            prev = doc.score
            self.assertEqual(doc.vector_names(), [])

    def test_query_multivector_weighted_integration(self) -> None:
        q1 = finch.VectorQuery(field_name="v1", vector=[1.0, 0.0, 0.0, 0.0])
        q2 = finch.VectorQuery(field_name="v2", vector=[0.0, 1.0, 0.0, 0.0])

        single = {
            "v1": self.col.query(q1, topk=10),
            "v2": self.col.query(q2, topk=10),
        }
        weights = {"v1": 0.7, "v2": 0.3}
        expected = _weighted_scores(single, weights, finch.MetricType.IP)

        reranker = finch.WeightedReRanker(topn=5, metric=finch.MetricType.IP, weights=weights)
        out = self.col.query(vectors=[q1, q2], topk=10, reranker=reranker)
        self.assertEqual(len(out), 5)

        prev = float("inf")
        for doc in out:
            self.assertIn(doc.id, expected)
            self.assertAlmostEqual(doc.score, expected[doc.id], delta=1e-10)
            self.assertLessEqual(doc.score, prev)
            prev = doc.score
            self.assertEqual(doc.vector_names(), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)


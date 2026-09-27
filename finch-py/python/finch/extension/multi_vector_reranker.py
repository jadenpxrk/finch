from __future__ import annotations

import heapq
import math
from collections import defaultdict
from typing import Optional

from ..model.doc import Doc
from ..typing import MetricType
from .rerank_function import RerankFunction


class RrfReRanker(RerankFunction):
    def __init__(
        self,
        topn: int = 10,
        rerank_field: Optional[str] = None,
        rank_constant: int = 60,
    ):
        super().__init__(topn=topn, rerank_field=rerank_field)
        self._rank_constant = int(rank_constant)

    @property
    def rank_constant(self) -> int:
        return self._rank_constant

    def _rrf_score(self, rank: int) -> float:
        return 1.0 / (self._rank_constant + rank + 1)

    def rerank(self, query_results: dict[str, list[Doc]]) -> list[Doc]:
        rrf_scores: dict[str, float] = defaultdict(float)
        id_to_doc: dict[str, Doc] = {}

        for _, query_result in query_results.items():
            for rank, doc in enumerate(query_result):
                doc_id = doc.id
                rrf_scores[doc_id] += self._rrf_score(rank)
                if doc_id not in id_to_doc:
                    id_to_doc[doc_id] = doc

        top_docs = heapq.nlargest(self.topn, rrf_scores.items(), key=lambda x: x[1])
        return [id_to_doc[doc_id]._replace(score=score) for doc_id, score in top_docs]


class WeightedReRanker(RerankFunction):
    def __init__(
        self,
        topn: int = 10,
        rerank_field: Optional[str] = None,
        metric: MetricType = MetricType.L2,
        weights: Optional[dict[str, float]] = None,
    ):
        super().__init__(topn=topn, rerank_field=rerank_field)
        self._weights = weights or {}
        self._metric = metric

    @property
    def weights(self) -> dict[str, float]:
        return self._weights

    @property
    def metric(self) -> MetricType:
        return self._metric

    def _normalize_score(self, score: float, metric: MetricType) -> float:
        if metric == MetricType.L2:
            return 1.0 - 2 * math.atan(score) / math.pi
        if metric == MetricType.IP:
            return 0.5 + math.atan(score) / math.pi
        if metric == MetricType.COSINE:
            return 1.0 - score / 2.0
        raise ValueError("Unsupported metric type")

    def rerank(self, query_results: dict[str, list[Doc]]) -> list[Doc]:
        weighted_scores: dict[str, float] = defaultdict(float)
        id_to_doc: dict[str, Doc] = {}

        for vector_name, query_result in query_results.items():
            for doc in query_result:
                doc_id = doc.id
                weighted_scores[doc_id] += self._normalize_score(doc.score, self.metric) * self.weights.get(vector_name, 1.0)
                if doc_id not in id_to_doc:
                    id_to_doc[doc_id] = doc

        top_docs = heapq.nlargest(self.topn, weighted_scores.items(), key=lambda x: x[1])
        return [id_to_doc[doc_id]._replace(score=score) for doc_id, score in top_docs]


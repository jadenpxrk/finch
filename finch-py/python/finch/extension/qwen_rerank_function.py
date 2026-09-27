from __future__ import annotations

from typing import Optional

from ..model.doc import Doc
from .qwen_function import QwenFunctionBase
from .rerank_function import RerankFunction


class QwenReRanker(QwenFunctionBase, RerankFunction):
    def __init__(
        self,
        query: Optional[str] = None,
        topn: int = 10,
        rerank_field: Optional[str] = None,
        model: str = "gte-rerank-v2",
        api_key: Optional[str] = None,
    ):
        QwenFunctionBase.__init__(self, model=model, api_key=api_key)
        RerankFunction.__init__(self, topn=topn, rerank_field=rerank_field)
        if not query:
            raise ValueError("Query is required for QwenReRanker")
        if not rerank_field:
            raise ValueError("rerank_field is required for QwenReRanker")
        self._query = query

    @property
    def query(self) -> str:
        return self._query

    def rerank(self, query_results: dict[str, list[Doc]]) -> list[Doc]:
        if not query_results:
            return []

        id_to_doc: dict[str, Doc] = {}
        doc_ids: list[str] = []
        contents: list[str] = []

        for _, query_result in query_results.items():
            for doc in query_result:
                if doc.id in id_to_doc:
                    continue
                field_value = doc.field(self.rerank_field)
                rank_content = str(field_value).strip() if field_value else ""
                if not rank_content:
                    continue
                id_to_doc[doc.id] = doc
                doc_ids.append(doc.id)
                contents.append(rank_content)

        if not contents:
            raise ValueError("No documents to rerank")

        output = self._call_rerank_api(query=self.query, documents=contents, top_n=self.topn)
        results = []
        for item in output.get("results", []):
            idx = item.get("index")
            score = item.get("relevance_score")
            if idx is None or score is None:
                continue
            doc_id = doc_ids[int(idx)]
            results.append(id_to_doc[doc_id]._replace(score=float(score)))
        return results


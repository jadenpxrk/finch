from __future__ import annotations

from typing import Literal, Optional

from ..model.doc import Doc
from ..tool import require_module
from .rerank_function import RerankFunction
from .sentence_transformer_function import SentenceTransformerFunctionBase


class DefaultLocalReRanker(SentenceTransformerFunctionBase, RerankFunction):
    def __init__(
        self,
        query: Optional[str] = None,
        topn: int = 10,
        rerank_field: Optional[str] = None,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        model_source: Literal["huggingface", "modelscope"] = "huggingface",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        SentenceTransformerFunctionBase.__init__(self, model_name=model_name, model_source=model_source, device=device)
        RerankFunction.__init__(self, topn=topn, rerank_field=rerank_field)
        if not query:
            raise ValueError("Query is required for DefaultLocalReRanker")
        if not rerank_field:
            raise ValueError("rerank_field is required for DefaultLocalReRanker")
        self._query = query
        self._batch_size = int(batch_size)
        self._model = None
        model = self._get_model()
        if not hasattr(model, "predict"):
            raise ValueError(
                f"Model '{model_name}' does not appear to be a cross-encoder model. "
                "Cross-encoder models should have a 'predict' method."
            )
        self._model = model

    def _get_model(self):
        if self._model is not None:
            return self._model

        try:
            sentence_transformers = require_module("sentence_transformers")
            if self.model_source == "modelscope":
                require_module("modelscope")
                from modelscope.hub.snapshot_download import snapshot_download

                model_dir = snapshot_download(self.model_name)
                model = sentence_transformers.CrossEncoder(model_dir, device=self._device)
            else:
                model = sentence_transformers.CrossEncoder(self.model_name, device=self._device)
            return model
        except ImportError as e:
            if "modelscope" in str(e) and self.model_source == "modelscope":
                raise ImportError(
                    "ModelScope support requires the 'modelscope' package. "
                    "Please install it with: pip install modelscope"
                ) from e
            raise
        except Exception as e:
            raise ValueError(
                f"Failed to load CrossEncoder model '{self.model_name}' "
                f"from {self.model_source}: {e!s}"
            ) from e

    @property
    def query(self) -> str:
        return self._query

    @property
    def batch_size(self) -> int:
        return self._batch_size

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

        try:
            pairs = [[self.query, c] for c in contents]
            scores = self._model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            scores = [float(s) for s in scores]
        except Exception as e:
            raise RuntimeError(f"Failed to compute rerank scores: {e!s}") from e

        scored = [(doc_ids[i], scores[i]) for i in range(len(doc_ids))]
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[: self.topn]
        return [id_to_doc[doc_id]._replace(score=score) for doc_id, score in scored]

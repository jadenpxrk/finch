from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..model.doc import Doc


class RerankFunction(ABC):
    def __init__(self, topn: int = 10, rerank_field: Optional[str] = None):
        self._topn = int(topn)
        self._rerank_field = rerank_field

    @property
    def topn(self) -> int:
        return self._topn

    @property
    def rerank_field(self) -> Optional[str]:
        return self._rerank_field

    @abstractmethod
    def rerank(self, query_results: dict[str, list[Doc]]) -> list[Doc]: ...


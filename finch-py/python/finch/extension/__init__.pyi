from __future__ import annotations

from typing import Optional, Protocol

from ..model.doc import Doc
from ..typing import MetricType

class ReRanker(Protocol):
    def rerank(self, query_results: dict[str, list[Doc]]) -> list[Doc]: ...

class RrfReRanker:
    topn: int
    rerank_field: Optional[str]
    rank_constant: int
    def __init__(self, topn: int = ..., rerank_field: Optional[str] = ..., rank_constant: int = ...) -> None: ...
    def rerank(self, query_results: dict[str, list[Doc]]) -> list[Doc]: ...

class WeightedReRanker:
    topn: int
    rerank_field: Optional[str]
    metric: MetricType
    weights: dict[str, float]
    def __init__(
        self,
        topn: int = ...,
        rerank_field: Optional[str] = ...,
        metric: MetricType = ...,
        weights: Optional[dict[str, float]] = ...,
    ) -> None: ...
    def rerank(self, query_results: dict[str, list[Doc]]) -> list[Doc]: ...

class DenseEmbeddingFunction(Protocol):
    def embed(self, input: object) -> object: ...

class SparseEmbeddingFunction(Protocol):
    def embed(self, input: object) -> object: ...

class OpenAIFunctionBase: ...
class QwenFunctionBase: ...
class SentenceTransformerFunctionBase: ...

class OpenAIDenseEmbedding:
    model: str
    dimension: int
    extra_params: dict
    def __init__(
        self,
        model: str = ...,
        dimension: Optional[int] = ...,
        api_key: Optional[str] = ...,
        base_url: Optional[str] = ...,
        **kwargs: object,
    ) -> None: ...
    def embed(self, input: str) -> list[float]: ...

class QwenDenseEmbedding:
    model: str
    dimension: int
    extra_params: dict
    def __init__(
        self,
        dimension: int,
        model: str = ...,
        api_key: Optional[str] = ...,
        **kwargs: object,
    ) -> None: ...
    def embed(self, input: str) -> list[float]: ...

class QwenSparseEmbedding:
    model: str
    dimension: int
    extra_params: dict
    def __init__(
        self,
        dimension: int,
        model: str = ...,
        api_key: Optional[str] = ...,
        **kwargs: object,
    ) -> None: ...
    def embed(self, input: str) -> dict[int, float]: ...

class DefaultLocalDenseEmbedding:
    dimension: int
    extra_params: dict
    model_name: str
    model_source: str
    device: str
    def __init__(
        self,
        model_source: str = ...,
        device: Optional[str] = ...,
        normalize_embeddings: bool = ...,
        batch_size: int = ...,
        **kwargs: object,
    ) -> None: ...
    def embed(self, input: str) -> list[float]: ...

class DefaultLocalSparseEmbedding:
    def __init__(self, encoding_type: str = ..., model_source: str = ..., device: Optional[str] = ..., **kwargs: object) -> None: ...
    def embed(self, input: str) -> dict[int, float]: ...
    @classmethod
    def clear_cache(cls) -> None: ...
    @classmethod
    def get_cache_info(cls) -> dict: ...
    @classmethod
    def remove_from_cache(cls, model_source: str = ..., device: Optional[str] = ...) -> bool: ...

class BM25EmbeddingFunction:
    corpus_size: int
    encoding_type: str
    language: str
    extra_params: dict
    def __init__(
        self,
        corpus: Optional[list[str]] = ...,
        encoding_type: str = ...,
        language: str = ...,
        b: float = ...,
        k1: float = ...,
        **kwargs: object,
    ) -> None: ...
    def embed(self, input: str) -> dict[int, float]: ...

class DefaultLocalReRanker:
    query: str
    topn: int
    rerank_field: str
    def __init__(self, query: Optional[str] = ..., topn: int = ..., rerank_field: Optional[str] = ..., **kwargs: object) -> None: ...
    def rerank(self, query_results: dict[str, list[Doc]]) -> list[Doc]: ...

class QwenReRanker:
    query: str
    topn: int
    rerank_field: str
    model: str
    def __init__(
        self,
        query: Optional[str] = ...,
        topn: int = ...,
        rerank_field: Optional[str] = ...,
        model: str = ...,
        api_key: Optional[str] = ...,
    ) -> None: ...
    def rerank(self, query_results: dict[str, list[Doc]]) -> list[Doc]: ...

__all__: list[str]

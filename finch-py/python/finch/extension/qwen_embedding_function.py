from __future__ import annotations

from functools import lru_cache
from typing import Optional

from ..common.constants import TEXT, DenseVectorType, SparseVectorType
from .embedding_function import DenseEmbeddingFunction, SparseEmbeddingFunction
from .qwen_function import QwenFunctionBase


class QwenDenseEmbedding(QwenFunctionBase, DenseEmbeddingFunction[TEXT]):
    def __init__(
        self,
        dimension: int,
        model: str = "text-embedding-v4",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        QwenFunctionBase.__init__(self, model=model, api_key=api_key)
        self._dimension = int(dimension)
        self._extra_params = kwargs

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def extra_params(self) -> dict:
        return self._extra_params

    def __call__(self, input: TEXT) -> DenseVectorType:
        return self.embed(input)

    @lru_cache(maxsize=10)
    def embed(self, input: TEXT) -> DenseVectorType:
        if not isinstance(input, TEXT):
            raise TypeError(f"Expected 'input' to be str, got {type(input).__name__}")
        s = input.strip()
        if not s:
            raise ValueError("Input text cannot be empty or whitespace only")

        output = self._call_text_embedding_api(
            input=s,
            dimension=self.dimension,
            output_type="dense",
            text_type=self._extra_params.get("text_type"),
        )
        embeddings = output.get("embeddings")
        if not isinstance(embeddings, list):
            raise ValueError("Invalid API response: 'embeddings' field is missing or not a list")
        if len(embeddings) != 1:
            raise ValueError(f"Expected exactly 1 embedding in response, got {len(embeddings)}")

        first = embeddings[0]
        if not isinstance(first, dict):
            raise ValueError("Invalid API response: embedding item is not a dictionary")

        dense = first.get("embedding")
        if not isinstance(dense, list):
            raise ValueError("Invalid API response: 'embedding' field is missing or not a list")

        if len(dense) != self.dimension:
            raise ValueError(f"Dimension mismatch: expected {self.dimension}, got {len(dense)}")
        return list(dense)


class QwenSparseEmbedding(QwenFunctionBase, SparseEmbeddingFunction[TEXT]):
    def __init__(
        self,
        dimension: int,
        model: str = "text-embedding-v4",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        QwenFunctionBase.__init__(self, model=model, api_key=api_key)
        self._dimension = int(dimension)
        self._extra_params = kwargs

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def extra_params(self) -> dict:
        return self._extra_params

    def __call__(self, input: TEXT) -> SparseVectorType:
        return self.embed(input)

    @lru_cache(maxsize=10)
    def embed(self, input: TEXT) -> SparseVectorType:
        if not isinstance(input, TEXT):
            raise TypeError(f"Expected 'input' to be str, got {type(input).__name__}")
        s = input.strip()
        if not s:
            raise ValueError("Input text cannot be empty or whitespace only")

        output = self._call_text_embedding_api(
            input=s,
            dimension=self.dimension,
            output_type="sparse",
            text_type=self._extra_params.get("encoding_type", "query"),
        )
        embeddings = output.get("embeddings")
        if not isinstance(embeddings, list):
            raise ValueError("Invalid API response: 'embeddings' field is missing or not a list")
        if len(embeddings) != 1:
            raise ValueError(f"Expected exactly 1 embedding in response, got {len(embeddings)}")

        first = embeddings[0]
        if not isinstance(first, dict):
            raise ValueError("Invalid API response: embedding item is not a dictionary")

        sparse_embedding = first.get("sparse_embedding")
        if not isinstance(sparse_embedding, list):
            raise ValueError("Invalid API response: 'sparse_embedding' field is missing or not a list")

        out: dict[int, float] = {}
        for item in sparse_embedding:
            if not isinstance(item, dict):
                raise ValueError("Invalid API response: sparse_embedding item is not a dictionary")
            idx = item.get("index")
            val = item.get("value")
            if idx is None or val is None:
                raise ValueError("Invalid API response: sparse_embedding item missing 'index' or 'value'")
            f = float(val)
            if f > 0.0:
                out[int(idx)] = f
        return dict(sorted(out.items(), key=lambda kv: kv[0]))

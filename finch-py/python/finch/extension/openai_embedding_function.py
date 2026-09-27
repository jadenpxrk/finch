from __future__ import annotations

from functools import lru_cache
from typing import Optional

from ..common.constants import TEXT, DenseVectorType
from .embedding_function import DenseEmbeddingFunction
from .openai_function import OpenAIFunctionBase


class OpenAIDenseEmbedding(OpenAIFunctionBase, DenseEmbeddingFunction[TEXT]):
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimension: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        OpenAIFunctionBase.__init__(self, model=model, api_key=api_key, base_url=base_url)
        self._custom_dimension = dimension
        self._dimension = dimension if dimension is not None else self._MODEL_DIMENSIONS.get(model, 1536)
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
        vec = self._call_text_embedding_api(input=s, dimension=self._custom_dimension)
        if len(vec) != self.dimension:
            raise ValueError(f"Dimension mismatch: expected {self.dimension}, got {len(vec)}")
        return vec

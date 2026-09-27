from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from ..common.constants import TEXT, SparseVectorType
from ..tool import require_module
from .embedding_function import SparseEmbeddingFunction


class BM25EmbeddingFunction(SparseEmbeddingFunction[TEXT]):
    def __init__(
        self,
        corpus: Optional[list[str]] = None,
        encoding_type: Literal["query", "document"] = "query",
        language: Literal["zh", "en"] = "zh",
        b: float = 0.75,
        k1: float = 1.2,
        **kwargs,
    ):
        if corpus is not None:
            if not corpus or not isinstance(corpus, list):
                raise ValueError("Corpus must be a non-empty list of strings")
            if not all(isinstance(doc, str) for doc in corpus):
                raise ValueError("All corpus documents must be strings")

        try:
            self._dashtext = require_module("dashtext")
        except ImportError as e:
            raise ImportError(
                "dashtext package is required for BM25EmbeddingFunction. "
                "Install it with: pip install dashtext"
            ) from e
        self._corpus = corpus
        self._encoding_type = encoding_type
        self._language = language
        self._b = float(b)
        self._k1 = float(k1)
        self._extra_params = kwargs
        self._build_encoder()

    @property
    def corpus_size(self) -> int:
        return 0 if self._corpus is None else len(self._corpus)

    @property
    def encoding_type(self) -> str:
        return str(self._encoding_type)

    @property
    def language(self) -> str:
        return str(self._language)

    @property
    def extra_params(self) -> dict:
        return self._extra_params

    def _build_encoder(self):
        try:
            if self._corpus is None:
                self._encoder = self._dashtext.SparseVectorEncoder.default(name=self._language)
            else:
                self._encoder = self._dashtext.SparseVectorEncoder(b=self._b, k1=self._k1, **self._extra_params)
                self._encoder.train(self._corpus)
        except ImportError as e:
            raise ImportError(
                "dashtext package is required for BM25EmbeddingFunction. "
                "Install it with: pip install dashtext"
            ) from e
        except Exception as e:
            if isinstance(e, (ValueError, RuntimeError)):
                raise
            raise RuntimeError(f"Failed to build BM25 encoder: {e!s}") from e

    def __call__(self, input: TEXT) -> SparseVectorType:
        return self.embed(input)

    @lru_cache(maxsize=10)
    def embed(self, input: TEXT) -> SparseVectorType:
        if not isinstance(input, str):
            raise TypeError(f"Expected 'input' to be str, got {type(input).__name__}")
        s = input.strip()
        if not s:
            raise ValueError("Input text cannot be empty or whitespace only")

        if self._encoding_type == "document" and hasattr(self._encoder, "encode_documents"):
            out = self._encoder.encode_documents(s)
        elif hasattr(self._encoder, "encode_queries"):
            out = self._encoder.encode_queries(s)
        else:
            out = self._encoder.encode(s, self._encoding_type)

        if not isinstance(out, dict):
            raise ValueError("dashtext encoder returned invalid sparse vector")
        sparse = {int(k): float(v) for k, v in out.items() if float(v) > 0.0}
        return dict(sorted(sparse.items(), key=lambda kv: kv[0]))

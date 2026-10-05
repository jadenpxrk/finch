from __future__ import annotations

from typing import ClassVar, Literal, Optional

import numpy as np

from ..common.constants import TEXT, DenseVectorType, SparseVectorType
from .embedding_function import DenseEmbeddingFunction, SparseEmbeddingFunction
from .sentence_transformer_function import SentenceTransformerFunctionBase


class DefaultLocalDenseEmbedding(SentenceTransformerFunctionBase, DenseEmbeddingFunction[TEXT]):
    # model ids
    _DEFAULT_HF_MODEL: ClassVar[str] = "all-MiniLM-L6-v2"
    _DEFAULT_MS_MODEL: ClassVar[str] = "iic/nlp_gte_sentence-embedding_chinese-small"

    def __init__(
        self,
        model_source: Literal["huggingface", "modelscope"] = "huggingface",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        **kwargs,
    ):
        model_name = self._DEFAULT_HF_MODEL if model_source == "huggingface" else self._DEFAULT_MS_MODEL
        SentenceTransformerFunctionBase.__init__(self, model_name=model_name, model_source=model_source, device=device)
        self._normalize = bool(normalize_embeddings)
        self._batch_size = int(batch_size)
        self._extra_params = kwargs

        # load model during init to derive embedding dimension.
        model = self._get_model()
        self._dimension = int(model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def extra_params(self) -> dict:
        return self._extra_params

    def __call__(self, input: TEXT) -> DenseVectorType:
        return self.embed(input)

    def embed(self, input: TEXT) -> DenseVectorType:
        if not isinstance(input, str):
            raise TypeError(f"Expected 'input' to be str, got {type(input).__name__}")
        s = input.strip()
        if not s:
            raise ValueError("Input text cannot be empty or whitespace only")
        try:
            model = self._get_model()
            vec = model.encode(
                s,
                normalize_embeddings=self._normalize,
                batch_size=self._batch_size,
                convert_to_numpy=True,
            )
            if hasattr(vec, "tolist"):
                vec = vec.tolist()
            else:
                vec = list(vec)
            if len(vec) != self.dimension:
                raise ValueError(f"Dimension mismatch: expected {self.dimension}, got {len(vec)}")
            return vec
        except Exception as e:
            if isinstance(e, (TypeError, ValueError)):
                raise
            raise RuntimeError(f"Failed to generate embedding: {e!s}") from e


class DefaultLocalSparseEmbedding(SentenceTransformerFunctionBase, SparseEmbeddingFunction[TEXT]):
    _DEFAULT_MODEL: ClassVar[str] = "naver/splade-cocondenser-ensembledistil"
    _model_cache: ClassVar[dict] = {}

    @classmethod
    def clear_cache(cls) -> None:
        cls._model_cache.clear()

    @classmethod
    def get_cache_info(cls) -> dict:
        return {
            "cached_models": len(cls._model_cache),
            "cache_keys": list(cls._model_cache.keys()),
        }

    @classmethod
    def remove_from_cache(
        cls, model_source: str = "huggingface", device: Optional[str] = None
    ) -> bool:
        key = (cls._DEFAULT_MODEL, model_source, device)
        return cls._model_cache.pop(key, None) is not None

    def __init__(
        self,
        encoding_type: Literal["query", "document"] = "query",
        model_source: Literal["huggingface", "modelscope"] = "huggingface",
        device: Optional[str] = None,
        **kwargs,
    ):
        SentenceTransformerFunctionBase.__init__(
            self, model_name=self._DEFAULT_MODEL, model_source=model_source, device=device
        )
        self._encoding_type = encoding_type
        self._extra_params = kwargs

    def _get_sparse_model(self):
        key = (self.model_name, self.model_source, self._device)
        if key in self._model_cache:
            return self._model_cache[key]
        model = self._get_model()
        self._model_cache[key] = model
        return model

    def __call__(self, input: TEXT) -> SparseVectorType:
        return self.embed(input)

    def embed(self, input: TEXT) -> SparseVectorType:
        if not isinstance(input, str):
            raise TypeError(f"Expected 'input' to be str, got {type(input).__name__}")
        s = input.strip()
        if not s:
            raise ValueError("Input text cannot be empty or whitespace only")

        try:
            model = self._get_sparse_model()

            # sentence-transformers sparse encoders commonly accept
            # list inputs and return a matrix-like structure.
            if self._encoding_type == "document" and hasattr(model, "encode_document"):
                out = model.encode_document([s])
            elif hasattr(model, "encode_query"):
                out = model.encode_query([s])
            else:
                try:
                    out = model.encode([s])
                except Exception:
                    out = model.encode(s)

            if isinstance(out, dict):
                sparse = {int(k): float(v) for k, v in out.items() if float(v) > 0.0}
                return dict(sorted(sparse.items(), key=lambda kv: kv[0]))

            # Sparse matrix (CSR/CSC/etc.) duck typing.
            if hasattr(out, "toarray"):
                arr = np.asarray(out[0].toarray()).flatten()
            else:
                arr = out
                if hasattr(arr, "tolist"):
                    arr = arr.tolist()
                arr = np.asarray(arr)
                if arr.ndim > 1:
                    arr = arr[0]
                arr = arr.flatten()

            sparse = {int(i): float(v) for i, v in enumerate(arr.tolist()) if float(v) > 0.0}
            return dict(sorted(sparse.items(), key=lambda kv: kv[0]))
        except Exception as e:
            if isinstance(e, (TypeError, ValueError)):
                raise
            raise RuntimeError(f"Failed to generate sparse embedding: {e!s}") from e

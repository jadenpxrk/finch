from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, runtime_checkable

from ..common.constants import MD, DenseVectorType, SparseVectorType


@runtime_checkable
class DenseEmbeddingFunction(Protocol[MD]):
    @abstractmethod
    def embed(self, input: MD) -> DenseVectorType: ...


@runtime_checkable
class SparseEmbeddingFunction(Protocol[MD]):
    @abstractmethod
    def embed(self, input: MD) -> SparseVectorType: ...


from __future__ import annotations

from typing import Any, Mapping, Sequence, TypeAlias, Union

VectorType: TypeAlias = Union[Sequence[float], Mapping[int, float], Any]

from .constants import (
    AUDIO,
    IMAGE,
    TEXT,
    DenseVectorType,
    Embeddable,
    MD,
    SparseVectorType,
)

__all__ = [
    "AUDIO",
    "IMAGE",
    "TEXT",
    "DenseVectorType",
    "Embeddable",
    "MD",
    "SparseVectorType",
    "VectorType",
]

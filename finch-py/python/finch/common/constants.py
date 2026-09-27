from __future__ import annotations

from typing import Optional, TypeVar, Union

import numpy as np

# VectorType: DenseVectorType | SparseVectorType
DenseVectorType = Union[list[float], list[int], np.ndarray]
SparseVectorType = dict[int, float]
VectorType = Optional[Union[DenseVectorType, SparseVectorType]]

# Embeddable: Text | Image | Audio
TEXT = str
IMAGE = Union[str, bytes, np.ndarray]
AUDIO = Union[str, bytes, np.ndarray]

Embeddable = Optional[Union[TEXT, IMAGE, AUDIO]]

# Multimodal Embeddable
MD = TypeVar("MD", bound=Embeddable, contravariant=True)


from __future__ import annotations

from typing import Any, ClassVar, Optional, Sequence

from ...typing import IndexType, MetricType, QuantizeType

__all__: list[str]


class AddColumnOption:
    concurrency: int
    def __init__(self, concurrency: int = ...) -> None: ...


class AlterColumnOption:
    concurrency: int
    def __init__(self, concurrency: int = ...) -> None: ...


class CollectionOption:
    read_only: bool
    enable_mmap: bool
    max_buffer_size: int
    def __init__(self, read_only: bool = ..., enable_mmap: bool = ..., max_buffer_size: int = ...) -> None: ...


class IndexOption:
    concurrency: int
    def __init__(self, concurrency: int = ...) -> None: ...


class OptimizeOption:
    concurrency: int
    def __init__(self, concurrency: int = ...) -> None: ...


class InvertIndexParam:
    type: IndexType
    enable_range_optimization: bool
    enable_extended_wildcard: bool
    def __init__(self, enable_range_optimization: bool = ..., enable_extended_wildcard: bool = ...) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...


class HnswIndexParam:
    type: IndexType
    metric_type: MetricType
    m: int
    ef_construction: int
    quantize_type: QuantizeType
    def __init__(
        self,
        metric: Optional[MetricType] = ...,
        metric_type: Optional[MetricType] = ...,
        m: int = ...,
        ef_construction: int = ...,
        quantize: Optional[QuantizeType] = ...,
        quantize_type: Optional[QuantizeType] = ...,
    ) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...


class FlatIndexParam:
    type: IndexType
    metric_type: MetricType
    quantize_type: QuantizeType
    def __init__(
        self,
        metric: Optional[MetricType] = ...,
        metric_type: Optional[MetricType] = ...,
        quantize: Optional[QuantizeType] = ...,
        quantize_type: Optional[QuantizeType] = ...,
    ) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...


class IVFIndexParam:
    type: IndexType
    metric_type: MetricType
    n_list: int
    n_iters: int
    use_soar: bool
    quantize_type: QuantizeType
    def __init__(
        self,
        metric: Optional[MetricType] = ...,
        metric_type: Optional[MetricType] = ...,
        n_list: int = ...,
        n_iters: int = ...,
        use_soar: bool = ...,
        quantize: Optional[QuantizeType] = ...,
        quantize_type: Optional[QuantizeType] = ...,
    ) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...


class QueryParam:
    ef: int
    n_probe: int
    nprobe: int
    radius: float
    is_linear: bool
    is_using_refiner: bool
    def __init__(
        self,
        *,
        ef: Optional[int] = ...,
        n_probe: Optional[int] = ...,
        concurrency: Optional[int] = ...,
        bf_pks: Optional[Sequence[str]] = ...,
        radius: Optional[float] = ...,
        is_linear: Optional[bool] = ...,
        use_refiner: bool = ...,
        refiner_k: Optional[int] = ...,
        refiner_scale_factor: Optional[float] = ...,
    ) -> None: ...


def HnswQueryParam(
    ef: int = ...,
    radius: float = ...,
    is_linear: bool = ...,
    is_using_refiner: bool = ...,
) -> QueryParam: ...


def IVFQueryParam(nprobe: int = ...) -> QueryParam: ...


# reference API surface: SegmentOption is currently equivalent to CollectionOption.
SegmentOption: ClassVar[type[CollectionOption]]

# Backward/bench compatibility alias.
IvfIndexParam: ClassVar[type[IVFIndexParam]]

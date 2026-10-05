from __future__ import annotations

from typing import Any, ClassVar, Optional, SupportsInt

__all__: list[str]


class _EnumLike:
    __members__: ClassVar[dict[str, Any]]
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...
    def __int__(self) -> int: ...


class DataType(_EnumLike):
    UNDEFINED: ClassVar[DataType]
    BINARY: ClassVar[DataType]
    STRING: ClassVar[DataType]
    BOOL: ClassVar[DataType]
    INT32: ClassVar[DataType]
    INT64: ClassVar[DataType]
    UINT32: ClassVar[DataType]
    UINT64: ClassVar[DataType]
    FLOAT: ClassVar[DataType]
    DOUBLE: ClassVar[DataType]
    FLOAT32: ClassVar[DataType]
    FLOAT64: ClassVar[DataType]
    VECTOR_FP16: ClassVar[DataType]
    VECTOR_FP32: ClassVar[DataType]
    VECTOR_FP64: ClassVar[DataType]
    VECTOR_INT8: ClassVar[DataType]
    SPARSE_VECTOR_FP16: ClassVar[DataType]
    SPARSE_VECTOR_FP32: ClassVar[DataType]
    ARRAY_STRING: ClassVar[DataType]
    ARRAY_BOOL: ClassVar[DataType]
    ARRAY_INT32: ClassVar[DataType]
    ARRAY_INT64: ClassVar[DataType]
    ARRAY_UINT32: ClassVar[DataType]
    ARRAY_UINT64: ClassVar[DataType]
    ARRAY_FLOAT: ClassVar[DataType]
    ARRAY_DOUBLE: ClassVar[DataType]


class IndexType(_EnumLike):
    UNDEFINED: ClassVar[IndexType]
    HNSW: ClassVar[IndexType]
    IVF: ClassVar[IndexType]
    FLAT: ClassVar[IndexType]
    INVERT: ClassVar[IndexType]


class MetricType(_EnumLike):
    UNDEFINED: ClassVar[MetricType]
    L2: ClassVar[MetricType]
    IP: ClassVar[MetricType]
    COSINE: ClassVar[MetricType]


class QuantizeType(_EnumLike):
    UNDEFINED: ClassVar[QuantizeType]
    FP16: ClassVar[QuantizeType]
    INT8: ClassVar[QuantizeType]
    INT4: ClassVar[QuantizeType]


class StatusCode(_EnumLike):
    OK: ClassVar[StatusCode]
    NOT_FOUND: ClassVar[StatusCode]
    ALREADY_EXISTS: ClassVar[StatusCode]
    INVALID_ARGUMENT: ClassVar[StatusCode]
    PERMISSION_DENIED: ClassVar[StatusCode]
    FAILED_PRECONDITION: ClassVar[StatusCode]
    RESOURCE_EXHAUSTED: ClassVar[StatusCode]
    UNAVAILABLE: ClassVar[StatusCode]
    INTERNAL_ERROR: ClassVar[StatusCode]
    NOT_SUPPORTED: ClassVar[StatusCode]
    UNKNOWN: ClassVar[StatusCode]


class Status:
    @staticmethod
    def OK() -> Status: ...
    @staticmethod
    def NotFound(message: str) -> Status: ...
    @staticmethod
    def AlreadyExists(message: str) -> Status: ...
    @staticmethod
    def InvalidArgument(message: str) -> Status: ...
    @staticmethod
    def PermissionDenied(message: str) -> Status: ...
    @staticmethod
    def InternalError(message: str) -> Status: ...

    def __init__(self, code: StatusCode = ..., message: str = ...) -> None: ...
    def ok(self) -> bool: ...
    def is_ok(self) -> bool: ...
    def is_err(self) -> bool: ...
    def code(self) -> StatusCode: ...
    def message(self) -> str: ...

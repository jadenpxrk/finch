from __future__ import annotations

from .._finch import PyDataType as DataType
from .._finch import PyIndexType as IndexType
from .._finch import PyMetricType as MetricType
from .._finch import PyQuantizeType as QuantizeType
from .._finch import PyStatus as Status
from .._finch import PyStatusCode as StatusCode

__all__ = [
    "DataType",
    "IndexType",
    "MetricType",
    "QuantizeType",
    "Status",
    "StatusCode",
]

def _install_enum_surface(enum_cls, members: dict[str, object]) -> None:
    for n, v in members.items():
        if not hasattr(enum_cls, n):
            setattr(enum_cls, n, v)

    enum_cls.__members__ = dict(members)

    name_by_value: dict[int, str] = {}
    for n, v in members.items():
        name_by_value.setdefault(int(v), n)

    def _name(self) -> str:
        return name_by_value.get(int(self), str(self).split(".", 1)[-1])

    def _value(self) -> int:
        return int(self)

    enum_cls.name = property(_name)
    enum_cls.value = property(_value)


_install_enum_surface(
    DataType,
    {
        "UNDEFINED": DataType.Undefined,
        "BINARY": DataType.Binary,
        "STRING": DataType.String,
        "BOOL": DataType.Bool,
        "INT32": DataType.Int32,
        "INT64": DataType.Int64,
        "UINT32": DataType.Uint32,
        "UINT64": DataType.Uint64,
        "FLOAT": DataType.Float32,
        "DOUBLE": DataType.Float64,
        # Common aliases.
        "FLOAT32": DataType.Float32,
        "FLOAT64": DataType.Float64,
        "VECTOR_BINARY32": DataType.VectorBinary32,
        "VECTOR_BINARY64": DataType.VectorBinary64,
        "VECTOR_FP16": DataType.VectorFp16,
        "VECTOR_FP32": DataType.VectorFp32,
        "VECTOR_FP64": DataType.VectorFp64,
        "VECTOR_INT4": DataType.VectorInt4,
        "VECTOR_INT8": DataType.VectorInt8,
        "VECTOR_INT16": DataType.VectorInt16,
        "SPARSE_VECTOR_FP16": DataType.SparseFp16,
        "SPARSE_VECTOR_FP32": DataType.SparseFp32,
        "SPARSE_FP16": DataType.SparseFp16,
        "SPARSE_FP32": DataType.SparseFp32,
        "ARRAY_BINARY": DataType.ArrayBinary,
        "ARRAY_STRING": DataType.ArrayString,
        "ARRAY_BOOL": DataType.ArrayBool,
        "ARRAY_INT32": DataType.ArrayInt32,
        "ARRAY_INT64": DataType.ArrayInt64,
        "ARRAY_UINT32": DataType.ArrayUint32,
        "ARRAY_UINT64": DataType.ArrayUint64,
        "ARRAY_FLOAT": DataType.ArrayFp32,
        "ARRAY_DOUBLE": DataType.ArrayFp64,
        "ARRAY_FP32": DataType.ArrayFp32,
        "ARRAY_FP64": DataType.ArrayFp64,
    },
)

_install_enum_surface(
    IndexType,
    {
        "UNDEFINED": IndexType.Undefined,
        "HNSW": IndexType.Hnsw,
        "IVF": IndexType.Ivf,
        "FLAT": IndexType.Flat,
        "INVERT": IndexType.Invert,
        # Finch-only types (best-effort).
        "HNSW_SPARSE": IndexType.HnswSparse,
        "FLAT_SPARSE": IndexType.FlatSparse,
    },
)

_install_enum_surface(
    MetricType,
    {
        "UNDEFINED": MetricType.Undefined,
        "L2": MetricType.L2,
        "IP": MetricType.InnerProduct,
        "COSINE": MetricType.Cosine,
        "MIPS_L2": MetricType.MipsL2,
        "HAMMING": MetricType.Hamming,
    },
)

_install_enum_surface(
    QuantizeType,
    {
        "UNDEFINED": QuantizeType.Undefined,
        "FP16": QuantizeType.Fp16,
        "INT8": QuantizeType.Int8,
        "INT4": QuantizeType.Int4,
    },
)

_install_enum_surface(
    StatusCode,
    {
        "OK": StatusCode.Ok,
        "NOT_FOUND": StatusCode.NotFound,
        "ALREADY_EXISTS": StatusCode.AlreadyExists,
        "INVALID_ARGUMENT": StatusCode.InvalidArgument,
        "PERMISSION_DENIED": StatusCode.PermissionDenied,
        # Finch matches some but not all status codes; map missing ones to closest equivalents.
        "FAILED_PRECONDITION": StatusCode.Internal,
        "RESOURCE_EXHAUSTED": StatusCode.ResourceExhausted,
        "UNAVAILABLE": StatusCode.Unknown,
        "INTERNAL_ERROR": StatusCode.Internal,
        "NOT_SUPPORTED": StatusCode.Unimplemented,
        "UNKNOWN": StatusCode.Unknown,
        # Finch-only.
        "IO_ERROR": StatusCode.IoError,
        "INTERNAL": StatusCode.Internal,
        "UNIMPLEMENTED": StatusCode.Unimplemented,
        "OUT_OF_RANGE": StatusCode.OutOfRange,
        "CANCELLED": StatusCode.Cancelled,
    },
)


def _install_status_surface() -> None:
    # Adds Python-side accessors and constructors to the native Status class.
    if not hasattr(Status, "ok"):
        Status.ok = Status.is_ok  # type: ignore[attr-defined]

    _code_desc = Status.code
    _message_desc = Status.message

    def code(self):
        return _code_desc.__get__(self, type(self))

    def message(self):
        return _message_desc.__get__(self, type(self))

    Status.code = code  # type: ignore[assignment]
    Status.message = message  # type: ignore[assignment]

    # Static constructors.
    Status.OK = staticmethod(lambda: Status(StatusCode.OK))  # type: ignore[attr-defined]
    Status.NotFound = staticmethod(lambda msg: Status(StatusCode.NOT_FOUND, msg))  # type: ignore[attr-defined]
    Status.AlreadyExists = staticmethod(lambda msg: Status(StatusCode.ALREADY_EXISTS, msg))  # type: ignore[attr-defined]
    Status.InvalidArgument = staticmethod(lambda msg: Status(StatusCode.INVALID_ARGUMENT, msg))  # type: ignore[attr-defined]
    Status.PermissionDenied = staticmethod(lambda msg: Status(StatusCode.PERMISSION_DENIED, msg))  # type: ignore[attr-defined]
    Status.FailedPrecondition = staticmethod(lambda msg: Status(StatusCode.FAILED_PRECONDITION, msg))  # type: ignore[attr-defined]
    Status.ResourceExhausted = staticmethod(lambda msg: Status(StatusCode.RESOURCE_EXHAUSTED, msg))  # type: ignore[attr-defined]
    Status.Unavailable = staticmethod(lambda msg: Status(StatusCode.UNAVAILABLE, msg))  # type: ignore[attr-defined]
    Status.InternalError = staticmethod(lambda msg: Status(StatusCode.INTERNAL_ERROR, msg))  # type: ignore[attr-defined]
    Status.NotSupported = staticmethod(lambda msg: Status(StatusCode.NOT_SUPPORTED, msg))  # type: ignore[attr-defined]
    Status.Unknown = staticmethod(lambda msg: Status(StatusCode.UNKNOWN, msg))  # type: ignore[attr-defined]


_install_status_surface()

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Optional

from ..._finch import PyFieldSchema as _CoreFieldSchema
from ..._finch import (
    PyFlatIndexParam as _CoreFlatIndexParam,
    PyHnswIndexParam as _CoreHnswIndexParam,
    PyInvertIndexParam as _CoreInvertIndexParam,
    PyIvfIndexParam as _CoreIvfIndexParam,
)
from ...typing import DataType, IndexType, MetricType, QuantizeType
from ..param import FlatIndexParam, HnswIndexParam, InvertIndexParam, IVFIndexParam

__all__ = ["FieldSchema", "VectorSchema"]

_VECTOR_TYPES = (
    DataType.VectorBinary32,
    DataType.VectorBinary64,
    DataType.VectorFp16,
    DataType.VectorFp32,
    DataType.VectorFp64,
    DataType.VectorInt4,
    DataType.VectorInt8,
    DataType.VectorInt16,
    DataType.SparseFp16,
    DataType.SparseFp32,
)

_DATA_TYPE_MEMBER_T = type(DataType.VectorFp32)

_METRIC_BY_VALUE = {
    int(MetricType.Undefined): MetricType.Undefined,
    int(MetricType.L2): MetricType.L2,
    int(MetricType.InnerProduct): MetricType.InnerProduct,
    int(MetricType.Cosine): MetricType.Cosine,
    int(MetricType.MipsL2): MetricType.MipsL2,
    int(MetricType.Hamming): MetricType.Hamming,
}
_QUANTIZE_BY_VALUE = {
    int(QuantizeType.Undefined): QuantizeType.Undefined,
    int(QuantizeType.Fp16): QuantizeType.Fp16,
    int(QuantizeType.Int8): QuantizeType.Int8,
    int(QuantizeType.Int4): QuantizeType.Int4,
}


def _metric_from_int(v: int) -> MetricType:
    return _METRIC_BY_VALUE.get(int(v), MetricType.InnerProduct)


def _quantize_from_int(v: int) -> QuantizeType:
    return _QUANTIZE_BY_VALUE.get(int(v), QuantizeType.Undefined)


@dataclass
class FieldSchema:
    _core: _CoreFieldSchema

    def __setattr__(self, name: str, value: object) -> None:
        # FieldSchema is immutable once built.
        raise AttributeError("can't set attribute")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("can't delete attribute")

    def __repr__(self) -> str:  # pragma: no cover
        try:
            return json.dumps(self.__dict__(), indent=2, ensure_ascii=False)
        except Exception as e:
            return f"<FieldSchema error during repr: {e}>"

    def __str__(self) -> str:  # pragma: no cover
        return self.__repr__()

    def _eq_key(self) -> tuple:
        idx_type = self._core.index_type()
        idx_params = self._core.index_params_dict() or None
        # Canonicalize dict for stable comparisons.
        idx_params_s = None
        if isinstance(idx_params, dict):
            idx_params_s = json.dumps(idx_params, sort_keys=True, ensure_ascii=False)
        return (
            self.name,
            int(self.data_type),
            bool(self.nullable),
            int(self.dimension or 0),
            None if idx_type is None else int(idx_type),
            idx_params_s,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FieldSchema):
            return False
        return self._eq_key() == other._eq_key()

    def __hash__(self) -> int:
        return hash(self._eq_key())

    def __init__(
        self,
        name: str,
        data_type: Optional[DataType] = None,
        nullable: bool = False,
        dimension: Optional[int] = None,
        index_param: Optional[object] = None,
    ):
        if name is None or not isinstance(name, str):
            raise ValueError(
                f"schema validate failed: field name must be str, got {type(name).__name__}"
            )

        if data_type is None:
            raise TypeError("missing required argument: 'data_type'")

        if data_type in _VECTOR_TYPES:
            raise ValueError("FieldSchema is for scalar fields; use VectorSchema for vectors")
        core = _CoreFieldSchema(name, data_type, bool(nullable), dimension)
        if index_param is not None:
            if not isinstance(index_param, _CoreInvertIndexParam):
                raise TypeError("index_param must be InvertIndexParam")
            core = core.with_invert_index(index_param)
        object.__setattr__(self, "_core", core)

    @classmethod
    def _from_core(cls, core: _CoreFieldSchema) -> "FieldSchema":
        inst = cls.__new__(cls)
        object.__setattr__(inst, "_core", core)
        return inst

    @property
    def name(self) -> str:
        return self._core.name

    @property
    def dtype(self) -> DataType:
        return self._core.data_type

    @property
    def data_type(self) -> DataType:
        return self._core.data_type

    @property
    def nullable(self) -> bool:
        return bool(self._core.nullable)

    @property
    def dimension(self) -> Optional[int]:
        return self._core.dimension

    @property
    def index_param(self):
        idx_type = self._core.index_type()
        if idx_type is None:
            return None
        if int(idx_type) != int(IndexType.Invert):
            return None

        d = self._core.index_params_dict() or {}
        return InvertIndexParam(
            enable_range_optimization=bool(d.get("enable_range_optimization", False)),
            enable_extended_wildcard=bool(d.get("enable_extended_wildcard", False)),
        )

    @property
    def is_vector(self) -> bool:
        return bool(self._core.is_vector())

    @property
    def is_scalar(self) -> bool:
        return bool(self._core.is_scalar())

    def __dict__(self) -> dict:
        return {
            "name": self.name,
            "data_type": str(self.data_type).split(".", 1)[-1],
            "nullable": self.nullable,
            "index_param": (self._core.index_params_dict() or None),
        }

    def _get_object(self) -> _CoreFieldSchema:
        return self._core

    def with_hnsw_index(self, params):
        return FieldSchema._from_core(self._core.with_hnsw_index(params))

    def with_ivf_index(self, params):
        return FieldSchema._from_core(self._core.with_ivf_index(params))

    def with_flat_index(self, params):
        return FieldSchema._from_core(self._core.with_flat_index(params))

    def with_invert_index(self, params):
        return FieldSchema._from_core(self._core.with_invert_index(params))

    def __repr__(self) -> str:  # pragma: no cover
        return repr(self._core)


class VectorSchema(FieldSchema):
    def __hash__(self) -> int:
        return super().__hash__()

    def __init__(self, name: str, *args, **kwargs):
        data_type = kwargs.pop("data_type", None)
        dimension = kwargs.pop("dimension", None)
        dim = kwargs.pop("dim", None)
        nullable = kwargs.pop("nullable", False)
        index_param = kwargs.pop("index_param", None)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {', '.join(sorted(kwargs.keys()))}")

        if len(args) == 0:
            pass
        elif len(args) == 1:
            if isinstance(args[0], int):
                dimension = args[0]
            elif isinstance(args[0], _DATA_TYPE_MEMBER_T):
                data_type = args[0]
            else:
                raise TypeError("VectorSchema second positional arg must be DataType or int dimension")
        elif len(args) == 2:
            if isinstance(args[0], _DATA_TYPE_MEMBER_T) and isinstance(args[1], int):
                data_type = args[0]
                dimension = args[1]
            else:
                raise TypeError("VectorSchema positional args must be (DataType, int dimension)")
        else:
            raise TypeError("VectorSchema accepts at most 2 positional arguments after name")

        if dim is not None:
            if dimension is not None:
                raise ValueError("dimension and dim cannot both be set")
            dimension = dim

        if data_type is None:
            data_type = DataType.VectorFp32
        if dimension is None:
            dimension = 0

        if name is None or not isinstance(name, str):
            raise ValueError(
                f"schema validate failed: field name must be str, got {type(name).__name__}"
            )
        if not isinstance(dimension, int) or int(dimension) < 0:
            raise ValueError("schema validate failed: vector's dimension must be >= 0")
        if data_type not in _VECTOR_TYPES:
            raise ValueError("VectorSchema requires a vector DataType")

        core = _CoreFieldSchema(name, data_type, bool(nullable), int(dimension))
        if index_param is None:
            index_param = FlatIndexParam()

        if isinstance(index_param, _CoreHnswIndexParam):
            core = core.with_hnsw_index(index_param)
        elif isinstance(index_param, _CoreIvfIndexParam):
            core = core.with_ivf_index(index_param)
        elif isinstance(index_param, _CoreFlatIndexParam):
            core = core.with_flat_index(index_param)
        elif isinstance(index_param, _CoreInvertIndexParam):
            # allow constructing invalid schemas (e.g. vector + INVERT)
            # and defer validation to create/open.
            core = core.with_invert_index(index_param)
        else:
            raise TypeError("index_param must be HnswIndexParam | IVFIndexParam | FlatIndexParam | InvertIndexParam | None")

        object.__setattr__(self, "_core", core)

    @property
    def index_param(self):
        idx_type = self._core.index_type()
        if idx_type is None:
            return None
        d = self._core.index_params_dict() or {}
        if int(idx_type) in (int(IndexType.Flat), int(IndexType.FlatSparse)):
            return FlatIndexParam(
                metric_type=_metric_from_int(int(d.get("metric", int(MetricType.InnerProduct)))),
                quantize_type=_quantize_from_int(int(d.get("quantize", int(QuantizeType.Undefined)))),
            )
        if int(idx_type) in (int(IndexType.Hnsw), int(IndexType.HnswSparse)):
            return HnswIndexParam(
                metric_type=_metric_from_int(int(d.get("metric", int(MetricType.InnerProduct)))),
                m=int(d.get("m", 50)),
                ef_construction=int(d.get("ef_construction", 500)),
                quantize_type=_quantize_from_int(int(d.get("quantize", int(QuantizeType.Undefined)))),
            )
        if int(idx_type) == int(IndexType.Ivf):
            return IVFIndexParam(
                metric_type=_metric_from_int(int(d.get("metric", int(MetricType.InnerProduct)))),
                n_list=int(d.get("n_list", 0)),
                n_iters=int(d.get("n_iters", 10)),
                use_soar=bool(d.get("use_soar", False)),
                quantize_type=_quantize_from_int(int(d.get("quantize", int(QuantizeType.Undefined)))),
            )
        return None

    def __dict__(self) -> dict:
        d = super().__dict__()
        d["dimension"] = self.dimension
        return d

from __future__ import annotations

import operator

from ..._finch import (
    PyAddColumnOption as AddColumnOption,
    PyAlterColumnOption as AlterColumnOption,
    PyCollectionOption as CollectionOption,
    PyFlatIndexParam as _CoreFlatIndexParam,
    PyHnswIndexParam as _CoreHnswIndexParam,
    PyIndexOption as IndexOption,
    PyInvertIndexParam as _CoreInvertIndexParam,
    PyIvfIndexParam as _CoreIVFIndexParam,
    PyOptimizeOption as OptimizeOption,
    PyQueryParam as QueryParam,
)
from ...typing import MetricType, QuantizeType
from ...typing import IndexType

__all__ = [
    "AddColumnOption",
    "AlterColumnOption",
    "CollectionOption",
    "SegmentOption",
    "FlatIndexParam",
    "HnswIndexParam",
    "IVFIndexParam",
    "IndexOption",
    "InvertIndexParam",
    "OptimizeOption",
    "QueryParam",
    "HnswQueryParam",
    "IVFQueryParam",
    "IvfIndexParam",
]

class HnswIndexParam:
    def __new__(
        cls,
        metric: MetricType | None = None,
        metric_type: MetricType | None = None,
        m: int = 50,
        ef_construction: int = 500,
        quantize: QuantizeType | None = None,
        quantize_type: QuantizeType | None = None,
    ):
        return _CoreHnswIndexParam(
            metric=metric if metric is not None else (metric_type if metric_type is not None else MetricType.IP),
            m=int(m),
            ef_construction=int(ef_construction),
            quantize=quantize if quantize is not None else (quantize_type if quantize_type is not None else QuantizeType.UNDEFINED),
        )


class IVFIndexParam:
    def __new__(
        cls,
        metric: MetricType | None = None,
        metric_type: MetricType | None = None,
        n_list: int = 0,
        n_iters: int = 10,
        use_soar: bool = False,
        quantize: QuantizeType | None = None,
        quantize_type: QuantizeType | None = None,
    ):
        return _CoreIVFIndexParam(
            metric=metric if metric is not None else (metric_type if metric_type is not None else MetricType.IP),
            n_list=int(n_list),
            n_iters=int(n_iters),
            use_soar=bool(use_soar),
            quantize=quantize if quantize is not None else (quantize_type if quantize_type is not None else QuantizeType.UNDEFINED),
        )


class FlatIndexParam:
    def __new__(
        cls,
        metric: MetricType | None = None,
        metric_type: MetricType | None = None,
        quantize: QuantizeType | None = None,
        quantize_type: QuantizeType | None = None,
    ):
        return _CoreFlatIndexParam(
            metric=metric if metric is not None else (metric_type if metric_type is not None else MetricType.IP),
            quantize=quantize if quantize is not None else (quantize_type if quantize_type is not None else QuantizeType.UNDEFINED),
        )


class InvertIndexParam:
    def __new__(
        cls,
        enable_range_optimization: bool = False,
        enable_extended_wildcard: bool = False,
    ):
        return _CoreInvertIndexParam(
            enable_range_optimization=bool(enable_range_optimization),
            enable_extended_wildcard=bool(enable_extended_wildcard),
        )


class HnswQueryParam:
    def __new__(
        cls,
        ef: int = 300,
        radius: float = 0.0,
        is_linear: bool = False,
        is_using_refiner: bool = False,
    ) -> QueryParam:
        # invalid argument types should raise a pybind11-like
        # "incompatible constructor arguments" error (tests assert substring).
        try:
            ef_i = operator.index(ef)
            radius_f = None if (radius is None or float(radius) <= 0.0) else float(radius)
            is_linear_b = bool(is_linear)
            is_using_refiner_b = bool(is_using_refiner)
        except Exception as e:  # pragma: no cover
            raise TypeError("incompatible constructor arguments") from e

        return QueryParam(
            ef=int(ef_i),
            radius=radius_f,
            is_linear=is_linear_b,
            use_refiner=is_using_refiner_b,
        )


class IVFQueryParam:
    def __new__(cls, nprobe: int = 10) -> QueryParam:
        try:
            nprobe_i = operator.index(nprobe)
        except Exception as e:  # pragma: no cover
            raise TypeError("incompatible constructor arguments") from e
        return QueryParam(n_probe=int(nprobe_i))


# Backward/bench compatibility alias.
IvfIndexParam = IVFIndexParam

# SegmentOption is currently equivalent to CollectionOption.
SegmentOption = CollectionOption


def _install_readonly_prop(cls, name: str, fget):
    if hasattr(cls, name):
        return
    setattr(cls, name, property(fget))


def _install_readonly_setattr(cls) -> None:
    # Force stable readonly-mutation error messages across platforms/interpreters.
    marker = "__finch_readonly_setattr__"
    if getattr(cls, marker, False):
        return

    def _readonly_setattr(_self, _name: str, _value: object) -> None:
        raise AttributeError("can't set attribute")

    def _readonly_delattr(_self, _name: str) -> None:
        raise AttributeError("can't delete attribute")

    try:
        setattr(cls, "__setattr__", _readonly_setattr)
        setattr(cls, "__delattr__", _readonly_delattr)
        setattr(cls, marker, True)
    except Exception:
        # Some extension types may not allow patching; in that case, keep the
        # default behavior.
        return


def _metric_from_str(s: str):
    if hasattr(MetricType, "__members__") and s in MetricType.__members__:
        return MetricType.__members__[s]
    return getattr(MetricType, "IP", MetricType.InnerProduct)


def _quantize_from_str(s: str):
    if hasattr(QuantizeType, "__members__") and s in QuantizeType.__members__:
        return QuantizeType.__members__[s]
    return getattr(QuantizeType, "UNDEFINED", QuantizeType.Undefined)


def _patch_param_surfaces() -> None:
    # Index params are core objects; provide ergonomic attribute accessors.
    _install_readonly_prop(_CoreInvertIndexParam, "type", lambda _self: IndexType.INVERT)
    _install_readonly_setattr(_CoreInvertIndexParam)

    def _hnsw_dict(self):
        return self.to_dict() if hasattr(self, "to_dict") else {}

    _install_readonly_prop(_CoreHnswIndexParam, "type", lambda _self: IndexType.HNSW)
    _install_readonly_prop(_CoreHnswIndexParam, "metric_type", lambda self: _metric_from_str(_hnsw_dict(self).get("metric_type", "IP")))
    _install_readonly_prop(_CoreHnswIndexParam, "m", lambda self: int(_hnsw_dict(self).get("m", 50)))
    _install_readonly_prop(_CoreHnswIndexParam, "ef_construction", lambda self: int(_hnsw_dict(self).get("ef_construction", 500)))
    _install_readonly_prop(_CoreHnswIndexParam, "quantize_type", lambda self: _quantize_from_str(_hnsw_dict(self).get("quantize_type", "UNDEFINED")))
    _install_readonly_setattr(_CoreHnswIndexParam)

    def _flat_dict(self):
        return self.to_dict() if hasattr(self, "to_dict") else {}

    _install_readonly_prop(_CoreFlatIndexParam, "type", lambda _self: IndexType.FLAT)
    _install_readonly_prop(_CoreFlatIndexParam, "metric_type", lambda self: _metric_from_str(_flat_dict(self).get("metric_type", "IP")))
    _install_readonly_prop(_CoreFlatIndexParam, "quantize_type", lambda self: _quantize_from_str(_flat_dict(self).get("quantize_type", "UNDEFINED")))
    _install_readonly_setattr(_CoreFlatIndexParam)

    def _ivf_dict(self):
        return self.to_dict() if hasattr(self, "to_dict") else {}

    _install_readonly_prop(_CoreIVFIndexParam, "type", lambda _self: IndexType.IVF)
    _install_readonly_prop(_CoreIVFIndexParam, "metric_type", lambda self: _metric_from_str(_ivf_dict(self).get("metric_type", "IP")))
    _install_readonly_prop(_CoreIVFIndexParam, "n_list", lambda self: int(_ivf_dict(self).get("n_list", 0)))
    _install_readonly_prop(_CoreIVFIndexParam, "n_iters", lambda self: int(_ivf_dict(self).get("n_iters", 10)))
    _install_readonly_prop(_CoreIVFIndexParam, "use_soar", lambda self: bool(_ivf_dict(self).get("use_soar", False)))
    _install_readonly_prop(_CoreIVFIndexParam, "quantize_type", lambda self: _quantize_from_str(_ivf_dict(self).get("quantize_type", "UNDEFINED")))
    _install_readonly_setattr(_CoreIVFIndexParam)

    # These option/param core objects are immutable.
    _install_readonly_setattr(CollectionOption)
    _install_readonly_setattr(IndexOption)
    _install_readonly_setattr(AddColumnOption)
    _install_readonly_setattr(AlterColumnOption)
    _install_readonly_setattr(OptimizeOption)
    _install_readonly_setattr(QueryParam)


_patch_param_surfaces()

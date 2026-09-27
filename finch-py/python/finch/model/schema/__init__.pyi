from __future__ import annotations

from typing import Any, Optional, Union, overload

from ...typing import DataType
from ..param import (
    FlatIndexParam,
    HnswIndexParam,
    IVFIndexParam,
    InvertIndexParam,
)

__all__: list[str]


class FieldSchema:
    name: str
    data_type: DataType
    nullable: bool
    dimension: Optional[int]
    index_param: Optional[InvertIndexParam]

    def __init__(
        self,
        name: str,
        data_type: Optional[DataType] = ...,
        nullable: bool = ...,
        dimension: Optional[int] = ...,
        index_param: Optional[InvertIndexParam] = ...,
        *,
        dtype: Optional[DataType] = ...,
        is_primary: bool = ...,
    ) -> None: ...

    def with_invert_index(self, params: InvertIndexParam) -> FieldSchema: ...
    def __dict__(self) -> dict: ...


class VectorSchema(FieldSchema):
    index_param: Union[HnswIndexParam, IVFIndexParam, FlatIndexParam, InvertIndexParam, None]

    @overload
    def __init__(
        self,
        name: str,
        data_type: DataType = ...,
        dimension: int = ...,
        index_param: Optional[Union[HnswIndexParam, IVFIndexParam, FlatIndexParam, InvertIndexParam]] = ...,
        *,
        dim: Optional[int] = ...,
        nullable: bool = ...,
    ) -> None: ...

    @overload
    def __init__(
        self,
        name: str,
        dimension: int,
        *,
        data_type: DataType = ...,
        index_param: Optional[Union[HnswIndexParam, IVFIndexParam, FlatIndexParam, InvertIndexParam]] = ...,
        nullable: bool = ...,
    ) -> None: ...


class CollectionSchema:
    name: str

    def __init__(
        self,
        name: str,
        fields: Optional[Union[FieldSchema, list[FieldSchema]]] = ...,
        vectors: Optional[Union[VectorSchema, list[VectorSchema]]] = ...,
        *,
        max_doc_count_per_segment: Optional[int] = ...,
    ) -> None: ...

    def add_field(self, field: FieldSchema) -> None: ...
    def field(self, name: str) -> Optional[FieldSchema]: ...
    def vector(self, name: str) -> Optional[VectorSchema]: ...
    @property
    def fields(self) -> list[FieldSchema]: ...
    @property
    def vectors(self) -> list[VectorSchema]: ...


class CollectionStats: ...

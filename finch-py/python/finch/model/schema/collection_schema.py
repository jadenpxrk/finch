from __future__ import annotations

import json
from typing import Optional, Union

from ..._finch import PyCollectionSchema as _CoreCollectionSchema
from .field_schema import FieldSchema, VectorSchema

__all__ = ["CollectionSchema"]


class CollectionSchema:
    def __init__(
        self,
        name: str,
        fields: Optional[Union[FieldSchema, list[FieldSchema]]] = None,
        vectors: Optional[Union[VectorSchema, list[VectorSchema]]] = None,
        *,
        max_doc_count_per_segment: Optional[int] = None,
    ):
        max_docs = max_doc_count_per_segment if max_doc_count_per_segment is not None else _CoreCollectionSchema(name).max_doc_count_per_segment
        self._core = _CoreCollectionSchema(name, max_doc_count_per_segment=int(max_docs))

        field_items: list[FieldSchema] = []
        if isinstance(fields, FieldSchema):
            field_items = [fields]
        elif isinstance(fields, list):
            field_items = fields
        elif fields is None:
            field_items = []
        else:
            raise TypeError("fields must be FieldSchema | list[FieldSchema] | None")

        vector_items: list[VectorSchema] = []
        if isinstance(vectors, VectorSchema):
            vector_items = [vectors]
        elif isinstance(vectors, list):
            vector_items = vectors
        elif vectors is None:
            vector_items = []
        else:
            raise TypeError("vectors must be VectorSchema | list[VectorSchema] | None")

        for f in field_items + vector_items:
            self.add_field(f)

        # validate uniqueness early.
        all_names = [f.name for f in self.fields] + [v.name for v in self.vectors]
        if len(all_names) != len(set(all_names)):
            raise ValueError("schema validate failed: duplicate field name: field names must be unique")

    @classmethod
    def _from_core(cls, core: _CoreCollectionSchema) -> "CollectionSchema":
        inst = cls.__new__(cls)
        inst._core = core
        return inst

    @property
    def name(self) -> str:
        return self._core.name

    @property
    def max_doc_count_per_segment(self) -> int:
        return int(self._core.max_doc_count_per_segment)

    def add_field(self, field: FieldSchema) -> None:
        if not isinstance(field, FieldSchema):
            raise TypeError("field must be FieldSchema")
        name = field.name
        if self._core.get_field(name):
            raise ValueError(
                f"schema validate failed: duplicate field name '{name}': field names must be unique"
            )
        self._core.add_field(field._get_object())

    def field(self, name: str) -> Optional[FieldSchema]:
        f = self._core.get_field(name)
        if not f:
            return None
        fs = FieldSchema._from_core(f)
        return fs if fs.is_scalar else None

    def vector(self, name: str) -> Optional[VectorSchema]:
        f = self._core.get_field(name)
        if not f:
            return None
        fs = FieldSchema._from_core(f)
        return VectorSchema._from_core(f) if fs.is_vector else None  # type: ignore[arg-type]

    @property
    def fields(self) -> list[FieldSchema]:
        return [FieldSchema._from_core(f) for f in self._core.scalar_fields()]

    @property
    def vectors(self) -> list[VectorSchema]:
        return [VectorSchema._from_core(f) for f in self._core.vector_fields()]  # type: ignore[arg-type]

    def _get_object(self) -> _CoreCollectionSchema:
        return self._core

    def __repr__(self) -> str:  # pragma: no cover
        try:
            schema = {
                "name": self.name,
                "fields": {field.name: field.__dict__() for field in self.fields},
                "vectors": {vector.name: vector.__dict__() for vector in self.vectors},
            }
            return json.dumps(schema, indent=2, ensure_ascii=False)
        except Exception as e:
            return f"<CollectionSchema error during repr: {e}>"

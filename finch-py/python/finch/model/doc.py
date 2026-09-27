from __future__ import annotations

import json
from typing import Any, Optional

__all__ = ["Doc"]


class Doc:
    __slots__ = ("id", "score", "vectors", "fields")

    def __init__(
        self,
        id: str,
        score: Optional[float] = None,
        vectors: Optional[dict[str, Any]] = None,
        fields: Optional[dict[str, Any]] = None,
    ):
        self.id = id
        self.score = score
        if vectors:
            self.vectors = {
                name: (vec.tolist() if hasattr(vec, "tolist") else vec)
                for name, vec in vectors.items()
            }
        else:
            self.vectors = {}
        self.fields = fields or {}

    def has_field(self, name: str) -> bool:
        return name in self.fields

    def has_vector(self, name: str) -> bool:
        return name in self.vectors

    def vector(self, name: str):
        # when no vectors are present, return `{}` (not `None`).
        return self.vectors and self.vectors.get(name)

    def field(self, name: str):
        # when no scalar fields are present, return `{}` (not `None`).
        return self.fields and self.fields.get(name)

    def vector_names(self) -> list[str]:
        return [] if not self.vectors else list(self.vectors.keys())

    def field_names(self) -> list[str]:
        return [] if not self.fields else list(self.fields.keys())

    def __repr__(self) -> str:
        try:
            return json.dumps(
                {"id": self.id, "score": self.score, "fields": self.fields, "vectors": self.vectors},
                indent=2,
                ensure_ascii=False,
            )
        except Exception as e:  # pragma: no cover
            return f"<Doc error during repr: {e}>"

    def _replace(self, **changes):
        new_tuple = (
            changes.get("id", self.id),
            changes.get("score", self.score),
            changes.get("fields", self.fields.copy() if self.fields else None),
            changes.get("vectors", self.vectors.copy() if self.vectors else None),
        )
        return type(self)._from_tuple(new_tuple)

    @classmethod
    def _from_tuple(
        cls,
        data_tuple: tuple[
            str,
            Optional[float],
            Optional[dict[str, Any]],
            Optional[dict[str, Any]],
        ],
    ):
        obj = object.__new__(cls)
        obj.id = data_tuple[0]
        obj.score = data_tuple[1]
        obj.fields = data_tuple[2] or {}

        vectors = data_tuple[3]
        if vectors is not None:
            obj.vectors = {
                name: (vec.tolist() if hasattr(vec, "tolist") else vec)
                for name, vec in vectors.items()
            }
        else:
            obj.vectors = {}
        return obj

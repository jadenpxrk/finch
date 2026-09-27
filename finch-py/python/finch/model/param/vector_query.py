from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..._finch import PyQueryParam as QueryParam

__all__ = ["VectorQuery"]


@dataclass(frozen=True)
class VectorQuery:
    field_name: str
    id: str | None = None
    vector: Any | None = None
    param: Optional[QueryParam] = None

    def has_id(self) -> bool:
        return self.id is not None

    def has_vector(self) -> bool:
        v = self.vector
        if v is None:
            return False
        if isinstance(v, dict):
            return len(v) > 0
        # numpy arrays
        if hasattr(v, "size"):
            try:
                return int(v.size) > 0
            except Exception:
                return True
        try:
            return len(v) > 0  # type: ignore[arg-type]
        except Exception:
            return True

    def _validate(self) -> None:
        # allow "placeholder" queries (field_name only) to exist;
        # higher layers may fill vectors later. Only reject None field_name and
        # the "both id and vector" case.
        if self.field_name is None:
            raise ValueError("Field name cannot be empty")
        if self.has_vector() and self.id is not None:
            raise ValueError("Cannot provide both id and vector")
        # passing the wrong object type as `param` should surface
        # as a function-argument mismatch.
        if self.param is not None and not isinstance(self.param, QueryParam):
            raise TypeError("incompatible function arguments")

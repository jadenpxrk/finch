from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .doc import Doc

__all__ = ["GroupResult"]


@dataclass(frozen=True)
class GroupResult:
    group_value: Any
    docs: list[Doc]


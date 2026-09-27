from __future__ import annotations

from enum import IntEnum

class LogLevel(IntEnum):
    DEBUG: int
    INFO: int
    WARN: int
    WARNING: int
    ERROR: int
    FATAL: int

class LogType(IntEnum):
    CONSOLE: int
    FILE: int

__all__: list[str]


from __future__ import annotations

from enum import IntEnum

__all__ = ["LogLevel", "LogType"]


class LogLevel(IntEnum):
    DEBUG = 0
    INFO = 1
    WARN = 2
    WARNING = 2
    ERROR = 3
    FATAL = 4


class LogType(IntEnum):
    CONSOLE = 0
    FILE = 1


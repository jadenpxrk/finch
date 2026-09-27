from __future__ import annotations

from typing import Optional

from ._finch import create_and_open as _create_and_open
from ._finch import init_global_config as _init_global_config
from ._finch import open as _open

from .model.collection import Collection
from .model.param import CollectionOption
from .model.schema import CollectionSchema
from .typing.enum import LogLevel, LogType

__all__ = ["create_and_open", "init", "open"]

_MIN_MEMORY_LIMIT_MB = 100
_INIT_DONE = False


def _is_int_like(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_float_like(v: object) -> bool:
    return isinstance(v, (float, int)) and not isinstance(v, bool)


def _validate_positive_int(name: str, v: object | None) -> int | None:
    if v is None:
        return None
    if not _is_int_like(v):
        raise TypeError(f"{name} must be int")
    if int(v) <= 0:
        raise ValueError(f"{name} must be > 0")
    return int(v)


def _validate_ratio(name: str, v: object | None) -> float | None:
    if v is None:
        return None
    if not _is_float_like(v):
        raise TypeError(f"{name} must be float")
    fv = float(v)
    if fv < 0.0 or fv > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return fv


def _log_to_file(log_type: object) -> bool | None:
    if log_type is None:
        return None
    if not isinstance(log_type, LogType):
        raise TypeError("log_type must be LogType")
    return log_type == LogType.FILE


def init(
    *,
    log_type: Optional[LogType] = LogType.CONSOLE,
    log_level: Optional[LogLevel] = LogLevel.WARN,
    log_dir: Optional[str] = "./logs",
    log_basename: Optional[str] = "finch.log",
    log_file_size: Optional[int] = 2048,
    log_overdue_days: Optional[int] = 7,
    query_threads: Optional[int] = None,
    optimize_threads: Optional[int] = None,
    invert_to_forward_scan_ratio: Optional[float] = None,
    brute_force_by_keys_ratio: Optional[float] = None,
    memory_limit_mb: Optional[int] = None,
    wal_flush_every_docs: Optional[int] = None,
    wal_fsync_every_docs: Optional[int] = None,
) -> None:
    global _INIT_DONE
    if _init_global_config is None:  # pragma: no cover
        raise RuntimeError("finch init_global_config is not available in this build")

    # Finch init is best-effort idempotent after the first success.
    if _INIT_DONE:
        return None

    memory_limit_mb = _validate_positive_int("memory_limit_mb", memory_limit_mb)
    if memory_limit_mb is not None and memory_limit_mb < _MIN_MEMORY_LIMIT_MB:
        raise RuntimeError(f"memory_limit_mb must be >= {_MIN_MEMORY_LIMIT_MB}")

    memory_limit_bytes = (
        None if memory_limit_mb is None else int(memory_limit_mb) * 1024 * 1024
    )

    query_threads = _validate_positive_int("query_threads", query_threads)
    optimize_threads = _validate_positive_int("optimize_threads", optimize_threads)
    invert_to_forward_scan_ratio = _validate_ratio(
        "invert_to_forward_scan_ratio", invert_to_forward_scan_ratio
    )
    brute_force_by_keys_ratio = _validate_ratio(
        "brute_force_by_keys_ratio", brute_force_by_keys_ratio
    )
    wal_flush_every_docs = _validate_positive_int("wal_flush_every_docs", wal_flush_every_docs)
    wal_fsync_every_docs = _validate_positive_int("wal_fsync_every_docs", wal_fsync_every_docs)

    if log_dir is not None and not isinstance(log_dir, str):
        raise TypeError("log_dir must be str")
    if log_basename is not None and not isinstance(log_basename, str):
        raise TypeError("log_basename must be str")
    log_file_size = _validate_positive_int("log_file_size", log_file_size)
    log_overdue_days = _validate_positive_int("log_overdue_days", log_overdue_days)

    log_to_file = _log_to_file(log_type)

    if log_level is not None and not isinstance(log_level, LogLevel):
        raise TypeError("log_level must be LogLevel")

    try:
        _init_global_config(
            memory_limit_bytes=memory_limit_bytes,
            log_level=None if log_level is None else int(log_level),
            log_to_file=log_to_file,
            log_dir=log_dir,
            log_basename=log_basename,
            log_file_size_mb=None if log_file_size is None else int(log_file_size),
            log_overdue_days=None if log_overdue_days is None else int(log_overdue_days),
            query_thread_count=query_threads,
            optimize_thread_count=optimize_threads,
            invert_to_forward_scan_ratio=invert_to_forward_scan_ratio,
            brute_force_by_keys_ratio=brute_force_by_keys_ratio,
            wal_flush_every_docs=wal_flush_every_docs,
            wal_fsync_every_docs=wal_fsync_every_docs,
        )
    except RuntimeError as e:
        # Finch init is best-effort idempotent.
        if str(e).startswith("AlreadyExists:"):
            _INIT_DONE = True
            return None
        raise
    _INIT_DONE = True


def create_and_open(
    path: str,
    schema: CollectionSchema,
    option: Optional[CollectionOption] = None,
) -> Collection:
    if not isinstance(path, str):
        raise TypeError("path must be a string")
    if not path or path.strip() != path:
        raise ValueError("path must be a non-empty string without leading/trailing whitespace")
    if not isinstance(schema, CollectionSchema):
        raise TypeError("schema must be a CollectionSchema")
    option = option or CollectionOption()
    if not isinstance(option, CollectionOption):
        raise TypeError("option must be a CollectionOption")
    if bool(getattr(option, "read_only", False)):
        raise ValueError("cannot create_and_open with read_only=True")
    core = _create_and_open(path, schema._get_object(), option)
    return Collection._from_core(core)


_DEFAULT_OPEN_OPTION = object()


def open(path: str, option: CollectionOption | object = _DEFAULT_OPEN_OPTION) -> Collection:
    if not isinstance(path, str):
        raise TypeError("path must be a string")
    if not path or path.strip() != path:
        raise ValueError("path must be a non-empty string without leading/trailing whitespace")
    # Allow `open(path)` with a default option, but passing `None`
    # should surface the underlying binding error ("incompatible function arguments").
    if option is _DEFAULT_OPEN_OPTION:
        option = CollectionOption()
    if option is None:  # type: ignore[redundant-expr]
        raise TypeError("incompatible function arguments")
    core = _open(path, option)
    return Collection._from_core(core)

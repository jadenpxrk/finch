from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest


def _run_in_subprocess(init_src: str, expect_exc: str | None = None) -> None:
    init_src = textwrap.dedent(init_src).strip("\n")
    indented = textwrap.indent(init_src, "    ")
    code = f"""
import sys
import finch
from finch import LogLevel, LogType

expected = {expect_exc!r}
try:
{indented}
except Exception as e:
    if expected is None:
        raise
    if type(e).__name__ != expected:
        raise AssertionError(f"expected {{expected}}, got {{type(e).__name__}}: {{e}}")
else:
    if expected is not None:
        raise AssertionError(f"expected exception {{expected}}")
"""
    env = os.environ.copy()
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"subprocess failed (rc={proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


class GlobalConfigTest(unittest.TestCase):
    def test_init_default(self):
        _run_in_subprocess("finch.init()")

    def test_repeated_initialization_is_noop(self):
        _run_in_subprocess("finch.init(); finch.init()")

    def test_memory_limit_validation(self):
        _run_in_subprocess("finch.init(memory_limit_mb=0)", expect_exc="ValueError")
        _run_in_subprocess("finch.init(memory_limit_mb=-1)", expect_exc="ValueError")
        _run_in_subprocess("finch.init(memory_limit_mb='512')", expect_exc="TypeError")
        _run_in_subprocess("finch.init(memory_limit_mb=512.5)", expect_exc="TypeError")
        _run_in_subprocess("finch.init(memory_limit_mb=99)", expect_exc="RuntimeError")
        _run_in_subprocess("finch.init(memory_limit_mb=100)")

    def test_query_threads_validation(self):
        _run_in_subprocess("finch.init(query_threads=1)")
        _run_in_subprocess("finch.init(query_threads=0)", expect_exc="ValueError")
        _run_in_subprocess("finch.init(query_threads=-1)", expect_exc="ValueError")
        _run_in_subprocess("finch.init(query_threads='1')", expect_exc="TypeError")
        _run_in_subprocess("finch.init(query_threads=1.5)", expect_exc="TypeError")

    def test_optimize_threads_validation(self):
        _run_in_subprocess("finch.init(optimize_threads=1)")
        _run_in_subprocess("finch.init(optimize_threads=0)", expect_exc="ValueError")
        _run_in_subprocess("finch.init(optimize_threads=-1)", expect_exc="ValueError")
        _run_in_subprocess("finch.init(optimize_threads='1')", expect_exc="TypeError")
        _run_in_subprocess("finch.init(optimize_threads=1.5)", expect_exc="TypeError")

    def test_ratio_validation(self):
        _run_in_subprocess("finch.init(invert_to_forward_scan_ratio=0.8)")
        _run_in_subprocess("finch.init(invert_to_forward_scan_ratio=1.1)", expect_exc="ValueError")
        _run_in_subprocess("finch.init(invert_to_forward_scan_ratio=-0.1)", expect_exc="ValueError")
        _run_in_subprocess("finch.init(invert_to_forward_scan_ratio='0.8')", expect_exc="TypeError")

        _run_in_subprocess("finch.init(brute_force_by_keys_ratio=0.8)")
        _run_in_subprocess("finch.init(brute_force_by_keys_ratio=1.1)", expect_exc="ValueError")
        _run_in_subprocess("finch.init(brute_force_by_keys_ratio=-0.1)", expect_exc="ValueError")
        _run_in_subprocess("finch.init(brute_force_by_keys_ratio='0.8')", expect_exc="TypeError")

    def test_log_type_and_level_validation(self):
        _run_in_subprocess("finch.init(log_type=LogType.CONSOLE)")
        _run_in_subprocess("finch.init(log_level=LogLevel.ERROR)")
        _run_in_subprocess("finch.init(log_type='FILE')", expect_exc="TypeError")
        _run_in_subprocess("finch.init(log_level='WARN')", expect_exc="TypeError")

    def test_log_file_size_validation(self):
        _run_in_subprocess("finch.init(log_type=LogType.FILE, log_file_size='df')", expect_exc="TypeError")
        _run_in_subprocess("finch.init(log_type=LogType.FILE, log_file_size=0)", expect_exc="ValueError")
        _run_in_subprocess("finch.init(log_type=LogType.FILE, log_file_size=-1)", expect_exc="ValueError")

    def test_log_overdue_days_validation(self):
        _run_in_subprocess("finch.init(log_type=LogType.FILE, log_overdue_days='df')", expect_exc="TypeError")
        _run_in_subprocess("finch.init(log_type=LogType.FILE, log_overdue_days=0)", expect_exc="ValueError")
        _run_in_subprocess("finch.init(log_type=LogType.FILE, log_overdue_days=-1)", expect_exc="ValueError")

    def test_file_logger_writes(self):
        with tempfile.TemporaryDirectory() as td:
            init_src = f"""
import pathlib

finch.init(
    log_type=LogType.FILE,
    log_level=LogLevel.DEBUG,
    log_dir={td!r},
    log_basename="finch_test.log",
)
from finch import CollectionSchema, VectorSchema, DataType, Doc

col = finch.create_and_open(
    path=str(pathlib.Path({td!r}) / "col"),
    schema=CollectionSchema(
        name="test",
        vectors=VectorSchema("image", DataType.VECTOR_FP32, 4),
    ),
)
col.insert(Doc(id="1", vectors={{"image": [1.0, 2.0, 3.0, 4.0]}}))

log_dir = pathlib.Path({td!r})
if not any(log_dir.glob("finch_test.log.*")):
    raise AssertionError(
        "expected log files in {td!r}, got: " + ", ".join(p.name for p in log_dir.iterdir())
    )
col.destroy()
"""
            _run_in_subprocess(init_src)

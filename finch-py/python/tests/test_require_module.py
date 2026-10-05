from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import finch


class RequireModuleTest(unittest.TestCase):
    def test_require_module_success(self):
        m = finch.require_module("os")
        self.assertIsNotNone(m)
        self.assertTrue(hasattr(m, "path"))

    def test_require_module_with_submodule_success(self):
        m = finch.require_module("os.path")
        self.assertIsNotNone(m)
        self.assertTrue(hasattr(m, "join"))

    def test_require_module_import_error(self):
        with self.assertRaises(ImportError) as ctx:
            finch.require_module("nonexistent_module_for_finch_tests_12345")
        self.assertIn(
            "Required package 'nonexistent_module_for_finch_tests_12345' is not installed.",
            str(ctx.exception),
        )

    @patch("importlib.import_module")
    def test_require_module_wraps_original_exception(self, mock_import_module):
        original = ImportError("Original error")
        mock_import_module.side_effect = original
        with self.assertRaises(ImportError) as ctx:
            finch.require_module("some_module")
        self.assertIs(ctx.exception.__cause__, original)

    @patch("importlib.import_module")
    def test_require_module_calls_importlib(self, mock_import_module):
        mod = MagicMock()
        mock_import_module.return_value = mod
        out = finch.require_module("test_module")
        mock_import_module.assert_called_once_with("test_module")
        self.assertIs(out, mod)

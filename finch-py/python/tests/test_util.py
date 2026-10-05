from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import finch


class UtilTest(unittest.TestCase):
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
        self.assertIn("Required package 'nonexistent_module_for_finch_tests_12345' is not installed.", str(ctx.exception))

    def test_require_module_with_mitigation_import_error(self):
        with self.assertRaises(ImportError) as ctx:
            finch.require_module("nonexistent_module_for_finch_tests_12345.submodule", mitigation="custom_package")
        msg = str(ctx.exception)
        self.assertIn("Required package 'custom_package' is not installed.", msg)
        self.assertIn("Module 'nonexistent_module_for_finch_tests_12345.submodule' is part of 'nonexistent_module_for_finch_tests_12345'", msg)
        self.assertIn("please pip install 'custom_package'.", msg)

    def test_require_module_submodule_import_error(self):
        with self.assertRaises(ImportError) as ctx:
            finch.require_module("os.nonexistent_submodule_for_finch_tests_12345")
        msg = str(ctx.exception)
        self.assertIn("Required package 'os.nonexistent_submodule_for_finch_tests_12345' is not installed.", msg)
        self.assertIn("Module 'os.nonexistent_submodule_for_finch_tests_12345' is part of 'os'", msg)
        self.assertIn("please pip install 'os'.", msg)

    @patch("importlib.import_module")
    def test_require_module_wraps_original_exception(self, mock_import_module):
        original = ImportError("Original error")
        mock_import_module.side_effect = original
        with self.assertRaises(ImportError) as ctx:
            finch.require_module("some_module")
        self.assertIs(ctx.exception.__cause__, original)

    @patch("importlib.import_module")
    def test_require_module_calls_importlib(self, mock_import_module):
        mock_module = MagicMock()
        mock_import_module.return_value = mock_module
        out = finch.require_module("test_module")
        mock_import_module.assert_called_once_with("test_module")
        self.assertIs(out, mock_module)

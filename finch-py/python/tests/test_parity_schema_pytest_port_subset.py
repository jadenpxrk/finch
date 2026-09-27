from __future__ import annotations

import re
import sys
import unittest

import finch


def _readonly_match_pattern() -> str:
    if sys.version_info >= (3, 11):
        return r"(can't set attribute|has no setter|readonly attribute)"
    return r"can't set attribute"


class ReferenceSchemaPytestPortSubsetTest(unittest.TestCase):
    def test_field_schema_default_and_readonly(self):
        field = finch.FieldSchema("field", data_type=finch.DataType.FLOAT)
        self.assertEqual(field.name, "field")
        self.assertEqual(field.data_type, finch.DataType.FLOAT)
        self.assertFalse(field.nullable)
        self.assertIsNone(field.index_param)

        field2 = finch.FieldSchema(
            name="float",
            data_type=finch.DataType.FLOAT,
            nullable=True,
            index_param=finch.InvertIndexParam(),
        )
        with self.assertRaises(AttributeError) as ctx:
            field2.index_param = finch.InvertIndexParam(enable_range_optimization=True)  # type: ignore[misc]
        self.assertRegex(str(ctx.exception), re.compile(_readonly_match_pattern()))

    def test_vector_schema_default_and_readonly(self):
        v = finch.VectorSchema("vector", data_type=finch.DataType.VECTOR_FP32, dimension=128)
        self.assertEqual(v.dimension, 128)
        with self.assertRaises(AttributeError) as ctx:
            v.dimension = 4  # type: ignore[misc]
        self.assertRegex(str(ctx.exception), re.compile(_readonly_match_pattern()))

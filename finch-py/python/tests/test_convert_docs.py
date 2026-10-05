from __future__ import annotations

import math
import unittest

import finch
from finch.model.convert import convert_to_core_doc, convert_to_py_doc


class ConvertDocsTest(unittest.TestCase):
    def test_convert_to_core_doc_default(self):
        doc = finch.Doc(id="1")
        schema = finch.CollectionSchema(
            name="test_collection",
            fields=finch.FieldSchema("name", finch.DataType.STRING),
        )
        core_doc = convert_to_core_doc(doc, schema=schema)
        self.assertIsNotNone(core_doc)
        self.assertEqual(core_doc.pk(), "1")

    def test_convert_to_core_doc_field_not_in_schema_message(self):
        doc = finch.Doc(id="1", fields={"name": "Tom"})
        schema = finch.CollectionSchema(
            name="test_collection",
            fields=[finch.FieldSchema("id", finch.DataType.UINT64)],
        )
        with self.assertRaises(ValueError) as ctx:
            convert_to_core_doc(doc, schema=schema)
        self.assertIn("schema validate failed:", str(ctx.exception))

    def test_convert_to_core_doc_scalars_and_arrays(self):
        schema = finch.CollectionSchema(
            name="test_collection",
            fields=[
                finch.FieldSchema("id", finch.DataType.UINT64),
                finch.FieldSchema("salary", finch.DataType.UINT32),
                finch.FieldSchema("age", finch.DataType.INT32),
                finch.FieldSchema("create_at", finch.DataType.INT64),
                finch.FieldSchema("author", finch.DataType.STRING),
                finch.FieldSchema("weight", finch.DataType.FLOAT),
                finch.FieldSchema("bmi", finch.DataType.DOUBLE),
                finch.FieldSchema("is_male", finch.DataType.BOOL),
                finch.FieldSchema("tags", finch.DataType.ARRAY_STRING),
                finch.FieldSchema("ids", finch.DataType.ARRAY_UINT64),
            ],
        )
        doc = finch.Doc(
            id="1",
            fields={
                "id": 1,
                "salary": 1000,
                "age": 18,
                "create_at": 1640995200,
                "author": "Tom",
                "weight": 80.0,
                "bmi": 0.4,
                "is_male": True,
                "tags": ["tag1", "tag2"],
                "ids": [1, 2, 3],
            },
        )
        core = convert_to_core_doc(doc, schema=schema)
        self.assertEqual(core.get_any("id", finch.DataType.UINT64), 1)
        self.assertEqual(core.get_any("tags", finch.DataType.ARRAY_STRING), ["tag1", "tag2"])
        got_weight = core.get_any("weight", finch.DataType.FLOAT)
        self.assertTrue(math.isclose(got_weight, 80.0, rel_tol=1e-6))

    def test_convert_to_py_doc_uses_get_all(self):
        schema = finch.CollectionSchema(
            name="c",
            fields=[finch.FieldSchema("x", finch.DataType.INT32)],
            vectors=[finch.VectorSchema("v", finch.DataType.VECTOR_FP32, 2)],
        )
        core = convert_to_core_doc(
            finch.Doc(id="1", fields={"x": 7}, vectors={"v": [0.1, 0.2]}),
            schema=schema,
        )
        py = convert_to_py_doc(core, schema)
        self.assertEqual(py.id, "1")
        self.assertEqual(py.field("x"), 7)
        self.assertEqual(len(py.vector("v")), 2)

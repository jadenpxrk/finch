from __future__ import annotations

import math
import unittest

import finch


class FinchReferenceDocConvertParityTest(unittest.TestCase):
    def test_py_doc_vectors_and_numpy(self):
        doc = finch.Doc(id="1", vectors={"dense": [1, 2, 3]})
        self.assertEqual(doc.id, "1")
        self.assertEqual(doc.vector("dense"), [1, 2, 3])

        doc2 = finch.Doc(
            id="1",
            vectors={"dense": [1, 2, 3], "sparse": {1: 1.0, 2: 2.0, 3: 3.0}},
        )
        self.assertEqual(doc2.vector("dense"), [1, 2, 3])
        self.assertEqual(doc2.vector("sparse"), {1: 1.0, 2: 2.0, 3: 3.0})

        try:
            import numpy as np
        except Exception:  # pragma: no cover
            self.skipTest("numpy is not available")

        doc3 = finch.Doc._from_tuple(
            (
                "1",
                0.0,
                None,
                {"image": np.array([1, 2, 3]), "keys": {1: 1.0, 2: 2.0}},
            )
        )
        self.assertEqual(doc3.vector("image"), [1, 2, 3])
        self.assertEqual(doc3.vector("keys"), {1: 1.0, 2: 2.0})

    def test_core_doc_set_pk_score(self):
        from finch._finch import PyDoc as _Doc

        d = _Doc()
        d.set_pk("1")
        self.assertEqual(d.pk(), "1")
        d.set_score(0.9)
        self.assertTrue(math.isclose(d.score(), 0.9, rel_tol=1e-6))

    def test_core_doc_set_any_scalar_and_array(self):
        from finch._finch import PyDoc as _Doc

        d = _Doc()

        s = finch.FieldSchema("author", finch.DataType.STRING, nullable=True)
        d.set_any("author", s._get_object(), None)
        self.assertTrue(d.has_field("author"))
        self.assertIsNone(d.get_any("author", finch.DataType.STRING))

        s2 = finch.FieldSchema("author2", finch.DataType.STRING, nullable=False)
        with self.assertRaises(ValueError):
            d.set_any("author2", s2._get_object(), None)

        d.set_any("is_male", finch.FieldSchema("is_male", finch.DataType.BOOL)._get_object(), True)
        self.assertEqual(d.get_any("is_male", finch.DataType.BOOL), True)

        d.set_any("age", finch.FieldSchema("age", finch.DataType.INT32)._get_object(), 19)
        self.assertEqual(d.get_any("age", finch.DataType.INT32), 19)

        d.set_any("id", finch.FieldSchema("id", finch.DataType.INT64)._get_object(), 1111111111111111111)
        self.assertEqual(d.get_any("id", finch.DataType.INT64), 1111111111111111111)

        d.set_any("weight", finch.FieldSchema("weight", finch.DataType.FLOAT)._get_object(), 60.5)
        self.assertTrue(math.isclose(d.get_any("weight", finch.DataType.FLOAT), 60.5, rel_tol=1e-6))

        d.set_any("height", finch.FieldSchema("height", finch.DataType.DOUBLE)._get_object(), 1.77777777777)
        self.assertTrue(math.isclose(d.get_any("height", finch.DataType.DOUBLE), 1.77777777777, rel_tol=1e-9))

        d.set_any("u32", finch.FieldSchema("u32", finch.DataType.UINT32)._get_object(), 4294967295)
        self.assertEqual(d.get_any("u32", finch.DataType.UINT32), 4294967295)

        d.set_any("u64", finch.FieldSchema("u64", finch.DataType.UINT64)._get_object(), 18446744073709551615)
        self.assertEqual(d.get_any("u64", finch.DataType.UINT64), 18446744073709551615)

        d.set_any("tags", finch.FieldSchema("tags", finch.DataType.ARRAY_STRING)._get_object(), ["tag1", "tag2"])
        self.assertEqual(d.get_any("tags", finch.DataType.ARRAY_STRING), ["tag1", "tag2"])

        d.set_any("ids", finch.FieldSchema("ids", finch.DataType.ARRAY_INT32)._get_object(), [1, 2, 3])
        self.assertEqual(d.get_any("ids", finch.DataType.ARRAY_INT32), [1, 2, 3])

        d.set_any("ids64", finch.FieldSchema("ids64", finch.DataType.ARRAY_INT64)._get_object(), [1, 2, 3])
        self.assertEqual(d.get_any("ids64", finch.DataType.ARRAY_INT64), [1, 2, 3])

        d.set_any("weights", finch.FieldSchema("weights", finch.DataType.ARRAY_FLOAT)._get_object(), [1.0, 2.0])
        self.assertEqual(d.get_any("weights", finch.DataType.ARRAY_FLOAT), [1.0, 2.0])

        d.set_any("heights", finch.FieldSchema("heights", finch.DataType.ARRAY_DOUBLE)._get_object(), [1.0, 2.0, 3.0])
        self.assertEqual(d.get_any("heights", finch.DataType.ARRAY_DOUBLE), [1.0, 2.0, 3.0])

    def test_core_doc_set_any_vectors(self):
        from finch._finch import PyDoc as _Doc

        d = _Doc()

        v16 = finch.VectorSchema("image", finch.DataType.VECTOR_FP16, 0)
        d.set_any("image", v16._get_object(), [1.0, 2.0, 3.0])
        vec = d.get_any("image", finch.DataType.VECTOR_FP16)
        self.assertEqual(len(vec), 3)
        for i in range(3):
            self.assertTrue(math.isclose(vec[i], [1.0, 2.0, 3.0][i], rel_tol=1e-1))

        v32 = finch.VectorSchema("image2", finch.DataType.VECTOR_FP32, 0)
        d.set_any("image2", v32._get_object(), [1.1, 2.2, 3.3])
        vec2 = d.get_any("image2", finch.DataType.VECTOR_FP32)
        for i in range(3):
            self.assertTrue(math.isclose(vec2[i], [1.1, 2.2, 3.3][i], rel_tol=1e-6))

        vint8 = finch.VectorSchema("text", finch.DataType.VECTOR_INT8, 0)
        d.set_any("text", vint8._get_object(), [1, 2, 3])
        self.assertEqual(d.get_any("text", finch.DataType.VECTOR_INT8), [1, 2, 3])

        sparse = {1: 1.111111, 2: 2.222222, 3: 3.333333}
        vs = finch.VectorSchema("key", finch.DataType.SPARSE_VECTOR_FP32, 0)
        d.set_any("key", vs._get_object(), sparse)
        got = d.get_any("key", finch.DataType.SPARSE_VECTOR_FP32)
        self.assertIsInstance(got, dict)
        for k, v in sparse.items():
            self.assertTrue(math.isclose(got[k], v, rel_tol=1e-6))

    def test_convert_to_cpp_and_py_doc(self):
        from finch.model.convert import convert_to_cpp_doc, convert_to_py_doc
        from finch._finch import PyDoc as _Doc

        schema = finch.CollectionSchema(
            name="test_collection",
            fields=finch.FieldSchema("name", finch.DataType.STRING),
        )
        cpp = convert_to_cpp_doc(finch.Doc(id="1"), collection_schema=schema)
        self.assertEqual(cpp.pk(), "1")

        schema2 = finch.CollectionSchema(
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
            },
        )
        cpp2 = convert_to_cpp_doc(doc, collection_schema=schema2)
        self.assertEqual(cpp2.get_any("author", finch.DataType.STRING), "Tom")

        # Field not in schema => ValueError.
        bad_schema = finch.CollectionSchema(
            name="test_collection",
            fields=[finch.FieldSchema("id", finch.DataType.UINT64)],
        )
        with self.assertRaises(ValueError):
            convert_to_cpp_doc(finch.Doc(id="1", fields={"name": "Tom"}), collection_schema=bad_schema)

        schema3 = finch.CollectionSchema(
            name="test_collection",
            fields=[
                finch.FieldSchema("ids_u64", finch.DataType.ARRAY_UINT64),
                finch.FieldSchema("ids_u32", finch.DataType.ARRAY_UINT32),
            ],
        )
        cpp3 = convert_to_cpp_doc(
            finch.Doc(id="1", fields={"ids_u64": [1, 2, 3], "ids_u32": [4, 5]}),
            collection_schema=schema3,
        )
        self.assertEqual(cpp3.get_any("ids_u64", finch.DataType.ARRAY_UINT64), [1, 2, 3])
        self.assertEqual(cpp3.get_any("ids_u32", finch.DataType.ARRAY_UINT32), [4, 5])

        # Convert core->py.
        core = _Doc()
        core.set_pk("1")
        core.set_score(1.0)
        core.set_any("author", schema2.field("author")._get_object(), "Tom")
        py = convert_to_py_doc(core, schema2)
        self.assertEqual(py.id, "1")
        self.assertEqual(py.score, 1.0)
        self.assertEqual(py.field("author"), "Tom")

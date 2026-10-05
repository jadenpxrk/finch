from __future__ import annotations

import math
import unittest

import finch


class DocTest(unittest.TestCase):
    def test_py_doc_default(self):
        doc = finch.Doc(id="1")
        # vector()/field() return `{}` when no vectors/fields exist.
        self.assertEqual(doc.vector("missing"), {})
        self.assertEqual(doc.field("missing"), {})

    def test_py_doc_vectors_and_fields(self):
        doc = finch.Doc(id="1", vectors={"dense": [1, 2, 3]})
        self.assertEqual(doc.id, "1")
        self.assertEqual(doc.vector("dense"), [1, 2, 3])

        doc = finch.Doc(
            id="1",
            vectors={"dense": [1, 2, 3], "sparse": {1: 1.0, 2: 2.0, 3: 3.0}},
        )
        self.assertEqual(doc.vector("dense"), [1, 2, 3])
        self.assertEqual(doc.vector("sparse"), {1: 1.0, 2: 2.0, 3: 3.0})

        doc = finch.Doc(
            id="1",
            vectors={
                "image": [1, 2, 3],
                "description": [4, 5, 6],
                "keys": {1: 1.0, 2: 2.0, 3: 3.0},
            },
            fields={"author": "Tom", "age": 19, "is_male": True, "weight": 60.5},
        )
        self.assertEqual(doc.vector("image"), [1, 2, 3])
        self.assertEqual(doc.vector("description"), [4, 5, 6])
        self.assertEqual(doc.vector("keys"), {1: 1.0, 2: 2.0, 3: 3.0})
        self.assertEqual(doc.field("author"), "Tom")
        self.assertEqual(doc.field("age"), 19)
        self.assertEqual(doc.field("is_male"), True)
        self.assertTrue(math.isclose(doc.field("weight"), 60.5, rel_tol=1e-6))

    def test_py_doc_from_tuple_numpy(self):
        try:
            import numpy as np
        except Exception:  # pragma: no cover
            self.skipTest("numpy is not available")

        doc = finch.Doc._from_tuple(
            (
                "1",
                0.0,
                None,
                {
                    "image": np.array([1, 2, 3]),
                    "description": np.random.random(8),
                    "keys": {1: 1.0, 2: 2.0, 3: 3.0},
                },
            )
        )
        self.assertEqual(doc.id, "1")
        self.assertEqual(doc.vector("image"), [1, 2, 3])
        self.assertEqual(doc.vector("keys"), {1: 1.0, 2: 2.0, 3: 3.0})

    def test_py_doc_replace_tuple_order(self):
        doc = finch.Doc(id="1", fields={"author": "Tom"}, vectors={"dense": [1, 2, 3]})
        doc2 = doc._replace(id="2")
        self.assertEqual(doc2.id, "2")
        self.assertEqual(doc2.field("author"), "Tom")
        self.assertEqual(doc2.vector("dense"), [1, 2, 3])


class CoreDocTest(unittest.TestCase):
    def test_core_doc_default_and_pk_score(self):
        from finch._finch import PyDoc as _Doc

        d = _Doc()
        self.assertIsNotNone(d)
        d.set_pk("1")
        self.assertEqual(d.pk(), "1")
        d.set_score(0.9)
        self.assertTrue(math.isclose(d.score(), 0.9, rel_tol=1e-6))

    def test_core_doc_set_any_nullability(self):
        from finch._finch import PyDoc as _Doc

        d = _Doc()
        schema = finch.FieldSchema("author", finch.DataType.STRING, nullable=True)
        d.set_any("author", schema._get_object(), None)
        self.assertTrue(d.has_field("author"))
        self.assertIsNone(d.get_any("author", finch.DataType.STRING))

        schema2 = finch.FieldSchema("author2", finch.DataType.STRING, nullable=False)
        with self.assertRaises(ValueError):
            d.set_any("author2", schema2._get_object(), None)

    def test_core_doc_scalar_and_array_types(self):
        from finch._finch import PyDoc as _Doc

        d = _Doc()
        d.set_any("author", finch.FieldSchema("author", finch.DataType.STRING)._get_object(), "Tom")
        self.assertEqual(d.get_any("author", finch.DataType.STRING), "Tom")

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

        d.set_any("tags", finch.FieldSchema("tags", finch.DataType.ARRAY_STRING)._get_object(), ["tag1", "tag2", "tag3"])
        self.assertEqual(d.get_any("tags", finch.DataType.ARRAY_STRING), ["tag1", "tag2", "tag3"])

        d.set_any("ids", finch.FieldSchema("ids", finch.DataType.ARRAY_INT32)._get_object(), [1, 2, 3])
        self.assertEqual(d.get_any("ids", finch.DataType.ARRAY_INT32), [1, 2, 3])

        d.set_any("weights", finch.FieldSchema("weights", finch.DataType.ARRAY_FLOAT)._get_object(), [1.0, 2.0, 3.0])
        self.assertEqual(d.get_any("weights", finch.DataType.ARRAY_FLOAT), [1.0, 2.0, 3.0])

        d.set_any("bools", finch.FieldSchema("bools", finch.DataType.ARRAY_BOOL)._get_object(), [True, False, True])
        self.assertEqual(d.get_any("bools", finch.DataType.ARRAY_BOOL), [True, False, True])

    def test_core_doc_vector_types(self):
        from finch._finch import PyDoc as _Doc

        d = _Doc()
        v16 = finch.VectorSchema("image", finch.DataType.VECTOR_FP16)
        d.set_any("image", v16._get_object(), [1.0, 2.0, 3.0])
        image_vector = d.get_any("image", finch.DataType.VECTOR_FP16)
        self.assertIsNotNone(image_vector)
        for i in range(len(image_vector)):
            self.assertTrue(math.isclose(image_vector[i], [1.0, 2.0, 3.0][i], rel_tol=1e-1))

        v32 = finch.VectorSchema("image2", finch.DataType.VECTOR_FP32)
        d.set_any("image2", v32._get_object(), [1.111111, 2.222222, 3.333333])
        vec = d.get_any("image2", finch.DataType.VECTOR_FP32)
        for i in range(len(vec)):
            self.assertTrue(math.isclose(vec[i], [1.111111, 2.222222, 3.333333][i], rel_tol=1e-6))

        vint8 = finch.VectorSchema("image3", finch.DataType.VECTOR_INT8)
        d.set_any("image3", vint8._get_object(), [1, 2, 3])
        self.assertEqual(d.get_any("image3", finch.DataType.VECTOR_INT8), [1, 2, 3])

        sparse = {1: 1.111111, 2: 2.222222, 3: 3.333333}
        vs32 = finch.VectorSchema("key", finch.DataType.SPARSE_VECTOR_FP32)
        d.set_any("key", vs32._get_object(), sparse)
        got = d.get_any("key", finch.DataType.SPARSE_VECTOR_FP32)
        self.assertIsInstance(got, dict)
        for key, value in sparse.items():
            self.assertTrue(math.isclose(got[key], value, rel_tol=1e-6))

        sparse16 = {1: 1.1, 2: 2.2, 3: 3.3}
        vs16 = finch.VectorSchema("key2", finch.DataType.SPARSE_VECTOR_FP16)
        d.set_any("key2", vs16._get_object(), sparse16)
        got16 = d.get_any("key2", finch.DataType.SPARSE_VECTOR_FP16)
        self.assertIsInstance(got16, dict)
        for key, value in sparse16.items():
            self.assertTrue(math.isclose(got16[key], value, rel_tol=1e-1))

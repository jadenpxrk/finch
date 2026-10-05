from __future__ import annotations

import pickle
import unittest

import finch


class PickleVectorQueryTest(unittest.TestCase):
    def test_core_vector_query_pickle_roundtrip(self):
        from finch._finch import PyVectorQuery as _VectorQuery

        v = _VectorQuery()
        v.field_name = "image"
        v.topk = 7
        v.filter = "id > 0"
        v.include_vector = True
        v.include_doc_id = True
        v.output_fields = ["id", "title"]
        v.set_vector(
            finch.VectorSchema("image", finch.DataType.VECTOR_FP32, 0)._get_object(),
            [1.0, 2.0, 3.0],
        )
        v.query_params = finch.QueryParam(ef=10, n_probe=2, radius=0.0, is_linear=False)

        blob = pickle.dumps(v)
        v2 = pickle.loads(blob)

        self.assertEqual(v2.field_name, "image")
        self.assertEqual(int(v2.topk), 7)
        self.assertEqual(v2.filter, "id > 0")
        self.assertTrue(v2.include_vector)
        self.assertTrue(v2.include_doc_id)
        self.assertEqual(v2.output_fields, ["id", "title"])
        self.assertEqual(
            v2.get_vector(finch.VectorSchema("image", finch.DataType.VECTOR_FP32, 0)._get_object()),
            [1.0, 2.0, 3.0],
        )

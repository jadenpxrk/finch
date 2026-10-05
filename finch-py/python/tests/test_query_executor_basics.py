from __future__ import annotations

import math
import unittest

import finch


class QueryExecutorBasicsTest(unittest.TestCase):
    def test_vector_query_has_id_has_vector(self):
        q = finch.VectorQuery(field_name="test_field")
        self.assertFalse(q.has_id())
        self.assertFalse(q.has_vector())

        q2 = finch.VectorQuery(field_name="test_field", id="x")
        self.assertTrue(q2.has_id())

        q3 = finch.VectorQuery(field_name="test_field", vector=[])
        self.assertFalse(q3.has_vector())

        q4 = finch.VectorQuery(field_name="test_field", vector=[1, 2, 3])
        self.assertTrue(q4.has_vector())

    def test_private_vector_query_set_get_vector_dense_and_sparse(self):
        try:
            import numpy as np
        except Exception:  # pragma: no cover
            self.skipTest("numpy is not available")

        from finch._finch import PyVectorQuery as _VectorQuery

        v = _VectorQuery()

        s16 = finch.VectorSchema(name="test", data_type=finch.DataType.VECTOR_FP16)
        vec16 = np.array([1.1, 2.1, 3.1], dtype=np.float16)
        v.set_vector(s16._get_object(), vec16)
        ret16 = v.get_vector(s16._get_object())
        self.assertTrue(np.array_equal(vec16, ret16))

        sp32 = finch.VectorSchema(name="test", data_type=finch.DataType.SPARSE_VECTOR_FP32)
        sparse = {1: 1.1, 2: 2.2, 3: 3.3}
        v.set_vector(sp32._get_object(), sparse)
        ret_sp32 = v.get_vector(sp32._get_object())
        for k, val in sparse.items():
            self.assertTrue(math.isclose(ret_sp32[k], val, abs_tol=1e-6))

    def test_query_executor_factory(self):
        from finch.executor.query_executor import (
            MultiVectorQueryExecutor,
            NoVectorQueryExecutor,
            QueryExecutorFactory,
            SingleVectorQueryExecutor,
        )

        s0 = finch.CollectionSchema(name="c0")
        ex0 = QueryExecutorFactory.create(s0)
        self.assertIsInstance(ex0, NoVectorQueryExecutor)

        s1 = finch.CollectionSchema(
            name="c1",
            vectors=finch.VectorSchema("v", finch.DataType.VECTOR_FP32, 4),
        )
        ex1 = QueryExecutorFactory.create(s1)
        self.assertIsInstance(ex1, SingleVectorQueryExecutor)

        s2 = finch.CollectionSchema(
            name="c2",
            vectors=[
                finch.VectorSchema("v1", finch.DataType.VECTOR_FP32, 4),
                finch.VectorSchema("v2", finch.DataType.VECTOR_FP32, 4),
            ],
        )
        ex2 = QueryExecutorFactory.create(s2)
        self.assertIsInstance(ex2, MultiVectorQueryExecutor)

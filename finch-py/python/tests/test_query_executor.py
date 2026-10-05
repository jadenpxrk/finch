from __future__ import annotations

import math
import unittest

import finch


class QueryExecutorTest(unittest.TestCase):
    def test_vector_query_has_id_has_vector(self):
        q = finch.VectorQuery(field_name="test_field")
        self.assertFalse(q.has_id())
        self.assertFalse(q.has_vector())

        q2 = finch.VectorQuery(field_name="test_field", id="x")
        self.assertTrue(q2.has_id())
        self.assertFalse(q2.has_vector())

        q3 = finch.VectorQuery(field_name="test_field", vector=[])
        self.assertFalse(q3.has_vector())

        q4 = finch.VectorQuery(field_name="test_field", vector=[1, 2, 3])
        self.assertTrue(q4.has_vector())

    def test_core_vector_query_set_get_vector_dense_and_sparse(self):
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

        s32 = finch.VectorSchema(name="test", data_type=finch.DataType.VECTOR_FP32)
        vec32 = np.array([1.1, 2.1, 3.1], dtype=np.float32)
        v.set_vector(s32._get_object(), vec32)
        ret32 = v.get_vector(s32._get_object())
        self.assertTrue(np.array_equal(vec32, ret32))

        s64 = finch.VectorSchema(name="test", data_type=finch.DataType.VECTOR_FP64)
        vec64 = np.array([1.1, 2.1, 3.1], dtype=np.float64)
        v.set_vector(s64._get_object(), vec64)
        ret64 = v.get_vector(s64._get_object())
        self.assertTrue(np.array_equal(vec64, ret64))

        s8 = finch.VectorSchema(name="test", data_type=finch.DataType.VECTOR_INT8)
        vec8 = np.array([1, 2, 3], dtype=np.int8)
        v.set_vector(s8._get_object(), vec8)
        ret8 = v.get_vector(s8._get_object())
        self.assertTrue(np.array_equal(vec8, ret8))

        # set_vector/get_vector reject unsupported dense vector dtypes.
        s16 = finch.VectorSchema(name="test", data_type=finch.DataType.VECTOR_INT16)
        vec16i = np.array([1, 2, 3], dtype=np.int16)
        with self.assertRaises(TypeError) as cm:
            v.set_vector(s16._get_object(), vec16i)
        self.assertIn("Unsupported dense vector type for ndarray input", str(cm.exception))

        sb32 = finch.VectorSchema(name="test", data_type=finch.DataType.VECTOR_BINARY32)
        vb32 = np.array([1, 2], dtype=np.uint32)
        with self.assertRaises(TypeError) as cm:
            v.set_vector(sb32._get_object(), vb32)
        self.assertIn("Unsupported dense vector type for ndarray input", str(cm.exception))

        sp32 = finch.VectorSchema(name="test", data_type=finch.DataType.SPARSE_VECTOR_FP32)
        sparse = {1: 1.1, 2: 2.2, 3: 3.3}
        v.set_vector(sp32._get_object(), sparse)
        ret_sp32 = v.get_vector(sp32._get_object())
        for k, val in sparse.items():
            self.assertTrue(math.isclose(ret_sp32[k], val, abs_tol=1e-6))

        sp16 = finch.VectorSchema(name="test", data_type=finch.DataType.SPARSE_VECTOR_FP16)
        v.set_vector(sp16._get_object(), sparse)
        ret_sp16 = v.get_vector(sp16._get_object())
        for k, val in sparse.items():
            self.assertTrue(math.isclose(ret_sp16[k], float(np.float16(val)), abs_tol=1e-6))

    def test_query_context_properties(self):
        from finch.executor.query_executor import QueryContext
        from finch import RrfReRanker

        ctx = QueryContext(topk=10)
        self.assertEqual(ctx.topk, 10)
        self.assertEqual(ctx.queries, [])
        self.assertIsNone(ctx.filter)
        self.assertIsNone(ctx.reranker)
        self.assertIsNone(ctx.output_fields)
        self.assertFalse(ctx.include_vector)
        self.assertEqual(ctx.core_vectors, [])

        queries = [finch.VectorQuery(field_name="dense", id="1")]
        reranker = RrfReRanker()
        out_fields = ["field1", "field2"]
        ctx2 = QueryContext(
            topk=5,
            filter="test_filter",
            include_vector=True,
            queries=queries,
            output_fields=out_fields,
            reranker=reranker,
        )
        self.assertEqual(ctx2.topk, 5)
        self.assertEqual(ctx2.queries, queries)
        self.assertEqual(ctx2.filter, "test_filter")
        self.assertIs(ctx2.reranker, reranker)
        self.assertEqual(ctx2.output_fields, out_fields)
        self.assertTrue(ctx2.include_vector)

    def test_query_executor_factory(self):
        from finch.executor.query_executor import (
            QueryExecutorFactory,
            NoVectorQueryExecutor,
            SingleVectorQueryExecutor,
            MultiVectorQueryExecutor,
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

    def test_no_vector_query_executor_build(self):
        from finch.executor.query_executor import NoVectorQueryExecutor, QueryContext

        schema = finch.CollectionSchema(name="c0")
        ex = NoVectorQueryExecutor(schema)

        ctx = QueryContext(topk=5, filter="test_filter")
        built = ex._do_build(ctx, object())
        self.assertEqual(len(built), 1)
        core = built[0]
        self.assertEqual(core.topk, 5)
        self.assertEqual(core.filter, "test_filter")

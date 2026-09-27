from __future__ import annotations

import unittest

import finch


class FinchReferenceSchemaParityTest(unittest.TestCase):
    def test_field_schema_default_and_custom(self):
        field = finch.FieldSchema("field", data_type=finch.DataType.FLOAT)
        self.assertEqual(field.name, "field")
        self.assertEqual(field.data_type, finch.DataType.FLOAT)
        self.assertFalse(field.nullable)
        self.assertIsNone(field.index_param)

        field2 = finch.FieldSchema(
            name="str",
            data_type=finch.DataType.STRING,
            nullable=True,
            index_param=finch.InvertIndexParam(enable_range_optimization=True),
        )
        self.assertEqual(field2.name, "str")
        self.assertEqual(field2.data_type, finch.DataType.STRING)
        self.assertTrue(field2.nullable)
        self.assertIsNotNone(field2.index_param)
        self.assertTrue(field2.index_param.enable_range_optimization)
        self.assertEqual(field2.index_param.type, finch.IndexType.INVERT)

        with self.assertRaises(AttributeError):
            field2.index_param = finch.InvertIndexParam()  # type: ignore[misc]

    def test_vector_schema_default_and_custom(self):
        vec = finch.VectorSchema("vector", data_type=finch.DataType.VECTOR_FP32, dimension=128)
        self.assertEqual(vec.name, "vector")
        self.assertEqual(vec.data_type, finch.DataType.VECTOR_FP32)
        self.assertEqual(vec.dimension, 128)
        self.assertIsNotNone(vec.index_param)
        self.assertEqual(vec.index_param.type, finch.IndexType.FLAT)
        self.assertEqual(vec.index_param.metric_type, finch.MetricType.IP)

        vec2 = finch.VectorSchema(
            name="vector2",
            data_type=finch.DataType.VECTOR_INT8,
            dimension=512,
            index_param=finch.HnswIndexParam(metric_type=finch.MetricType.COSINE, m=15, ef_construction=300),
        )
        self.assertEqual(vec2.index_param.type, finch.IndexType.HNSW)
        self.assertEqual(vec2.index_param.metric_type, finch.MetricType.COSINE)
        self.assertEqual(vec2.index_param.m, 15)
        self.assertEqual(vec2.index_param.ef_construction, 300)

        with self.assertRaises(AttributeError):
            vec2.dimension = 4  # type: ignore[misc]

    def test_collection_schema_and_stats_constructible(self):
        schema = finch.CollectionSchema(
            name="test_collection",
            fields=finch.FieldSchema("id", finch.DataType.INT64, nullable=False),
            vectors=finch.VectorSchema("vector", finch.DataType.VECTOR_FP32, 4),
        )
        self.assertEqual(schema.name, "test_collection")
        self.assertIsNotNone(schema.field("id"))
        self.assertIsNotNone(schema.vector("vector"))

        stats = finch.CollectionStats()
        self.assertIsNotNone(stats)

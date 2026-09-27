from __future__ import annotations

import unittest

import finch
from finch.model.convert import convert_to_cpp_doc


class FinchReferenceConvertErrorParityTest(unittest.TestCase):
    def test_scalar_fields_type_errors(self):
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
            ],
        )
        with self.assertRaises(TypeError):
            convert_to_cpp_doc(finch.Doc(id="1", fields={"id": "1"}), collection_schema=schema)
        with self.assertRaises(TypeError):
            convert_to_cpp_doc(finch.Doc(id="1", fields={"salary": "1000"}), collection_schema=schema)
        with self.assertRaises(TypeError):
            convert_to_cpp_doc(finch.Doc(id="1", fields={"age": "18"}), collection_schema=schema)
        with self.assertRaises(TypeError):
            convert_to_cpp_doc(finch.Doc(id="1", fields={"create_at": "2021-01-01"}), collection_schema=schema)
        with self.assertRaises(TypeError):
            convert_to_cpp_doc(finch.Doc(id="1", fields={"author": 1}), collection_schema=schema)
        with self.assertRaises(TypeError):
            convert_to_cpp_doc(finch.Doc(id="1", fields={"weight": "80.5"}), collection_schema=schema)
        with self.assertRaises(TypeError):
            convert_to_cpp_doc(finch.Doc(id="1", fields={"bmi": "25.0"}), collection_schema=schema)
        with self.assertRaises(TypeError):
            convert_to_cpp_doc(finch.Doc(id="1", fields={"is_male": "true"}), collection_schema=schema)

    def test_array_fields_type_errors(self):
        schema = finch.CollectionSchema(
            name="test_collection",
            fields=[
                finch.FieldSchema("tags", finch.DataType.ARRAY_STRING),
                finch.FieldSchema("ids", finch.DataType.ARRAY_UINT64),
                finch.FieldSchema("marks", finch.DataType.ARRAY_UINT32),
                finch.FieldSchema("x", finch.DataType.ARRAY_INT32),
                finch.FieldSchema("y", finch.DataType.ARRAY_INT64),
                finch.FieldSchema("scores", finch.DataType.ARRAY_FLOAT),
                finch.FieldSchema("ratios", finch.DataType.ARRAY_DOUBLE),
                finch.FieldSchema("results", finch.DataType.ARRAY_BOOL),
            ],
        )
        with self.assertRaises(TypeError):
            convert_to_cpp_doc(finch.Doc(id="1", fields={"tags": [1, 2, 3]}), collection_schema=schema)
        with self.assertRaises(TypeError):
            convert_to_cpp_doc(finch.Doc(id="1", fields={"ids": ["1", "2", "3"]}), collection_schema=schema)
        with self.assertRaises(TypeError):
            convert_to_cpp_doc(finch.Doc(id="1", fields={"marks": [1.1, 2.2, 3.3]}), collection_schema=schema)
        with self.assertRaises(TypeError):
            convert_to_cpp_doc(finch.Doc(id="1", fields={"x": [1.1, 2.2, 3.3]}), collection_schema=schema)
        with self.assertRaises(TypeError):
            convert_to_cpp_doc(finch.Doc(id="1", fields={"y": [1.1, 2.2, 3.3]}), collection_schema=schema)
        with self.assertRaises(TypeError):
            convert_to_cpp_doc(finch.Doc(id="1", fields={"scores": ["1", "2", "3"]}), collection_schema=schema)
        with self.assertRaises(TypeError):
            convert_to_cpp_doc(finch.Doc(id="1", fields={"ratios": ["1", "2", "3"]}), collection_schema=schema)
        with self.assertRaises(TypeError):
            convert_to_cpp_doc(finch.Doc(id="1", fields={"results": ["1", "2", "3"]}), collection_schema=schema)

    def test_vector_fields_type_errors(self):
        schema = finch.CollectionSchema(
            name="test_collection",
            vectors=[
                finch.VectorSchema(name="embedding", data_type=finch.DataType.VECTOR_FP16, dimension=4),
                finch.VectorSchema(name="image", data_type=finch.DataType.VECTOR_FP32, dimension=8),
                finch.VectorSchema(name="text", data_type=finch.DataType.VECTOR_INT8, dimension=32),
            ],
        )
        with self.assertRaises(TypeError):
            convert_to_cpp_doc(finch.Doc(id="1", vectors={"image": ["1.1"] * 4}), collection_schema=schema)
        with self.assertRaises(TypeError):
            convert_to_cpp_doc(finch.Doc(id="1", vectors={"text": ["1"] * 4}), collection_schema=schema)
        with self.assertRaises(TypeError):
            convert_to_cpp_doc(finch.Doc(id="1", vectors={"embedding": ["1"] * 4}), collection_schema=schema)

    def test_sparse_vector_type_errors(self):
        schema = finch.CollectionSchema(
            name="test_collection",
            vectors=[
                finch.VectorSchema(name="author", data_type=finch.DataType.SPARSE_VECTOR_FP32),
                finch.VectorSchema(name="content", data_type=finch.DataType.SPARSE_VECTOR_FP16),
            ],
        )
        with self.assertRaises(TypeError):
            convert_to_cpp_doc(finch.Doc(id="1", vectors={"author": {"1": 1.1}}), collection_schema=schema)
        with self.assertRaises(TypeError):
            convert_to_cpp_doc(finch.Doc(id="1", vectors={"content": {"1": 1.1}}), collection_schema=schema)
        with self.assertRaises(TypeError):
            convert_to_cpp_doc(finch.Doc(id="1", vectors={"author": {1: "1"}}), collection_schema=schema)


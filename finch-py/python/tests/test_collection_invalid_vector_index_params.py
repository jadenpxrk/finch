from __future__ import annotations

import tempfile
import unittest

import finch


class CollectionInvalidVectorIndexParamsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            finch.init()
        except RuntimeError:
            pass

    def test_invalid_vector_index_params_in_schema(self) -> None:
        # Pairs of vector data type and index parameters that schema validation rejects.
        cases: list[tuple[finch.DataType, object]] = [
            (finch.DataType.VECTOR_FP32, finch.InvertIndexParam()),
            (finch.DataType.VECTOR_FP16, finch.InvertIndexParam()),
            (finch.DataType.VECTOR_INT8, finch.InvertIndexParam()),
            (finch.DataType.SPARSE_VECTOR_FP32, finch.HnswIndexParam(metric_type=finch.MetricType.L2)),
            (finch.DataType.SPARSE_VECTOR_FP32, finch.FlatIndexParam(metric_type=finch.MetricType.COSINE)),
            (finch.DataType.SPARSE_VECTOR_FP32, finch.IVFIndexParam()),
            (finch.DataType.SPARSE_VECTOR_FP32, finch.InvertIndexParam()),
            (finch.DataType.SPARSE_VECTOR_FP16, finch.HnswIndexParam(metric_type=finch.MetricType.L2)),
            (finch.DataType.SPARSE_VECTOR_FP16, finch.FlatIndexParam(metric_type=finch.MetricType.COSINE)),
            (finch.DataType.SPARSE_VECTOR_FP16, finch.IVFIndexParam()),
            (finch.DataType.SPARSE_VECTOR_FP16, finch.InvertIndexParam()),
        ]

        for data_type, index_param in cases:
            with self.subTest(data_type=data_type.name, index_param=type(index_param).__name__):
                dim = 0 if data_type in (finch.DataType.SPARSE_VECTOR_FP16, finch.DataType.SPARSE_VECTOR_FP32) else 8
                schema = finch.CollectionSchema(
                    name="invalid_vec_param",
                    fields=[finch.FieldSchema("id", finch.DataType.INT64, nullable=False)],
                    vectors=[finch.VectorSchema("v", data_type=data_type, dimension=dim, index_param=index_param)],
                )
                with tempfile.TemporaryDirectory(prefix="finch_test_invalid_vec_") as td:
                    path = f"{td}/collection"
                    with self.assertRaises(ValueError):
                        finch.create_and_open(path, schema, finch.CollectionOption(read_only=False, enable_mmap=True))


if __name__ == "__main__":
    unittest.main(verbosity=2)


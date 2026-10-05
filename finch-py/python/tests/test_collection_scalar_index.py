from __future__ import annotations

import shutil
import tempfile
import unittest

import finch


def _schema() -> finch.CollectionSchema:
    # One field per scalar type that takes an invert index.
    return finch.CollectionSchema(
        name="scalar_index",
        fields=[
            finch.FieldSchema("bool_field", finch.DataType.BOOL, nullable=False),
            finch.FieldSchema("float_field", finch.DataType.FLOAT, nullable=False),
            finch.FieldSchema("double_field", finch.DataType.DOUBLE, nullable=False),
            finch.FieldSchema("int32_field", finch.DataType.INT32, nullable=False),
            finch.FieldSchema("int64_field", finch.DataType.INT64, nullable=False),
            finch.FieldSchema("uint32_field", finch.DataType.UINT32, nullable=False),
            finch.FieldSchema("uint64_field", finch.DataType.UINT64, nullable=False),
            finch.FieldSchema("string_field", finch.DataType.STRING, nullable=False),
            finch.FieldSchema("array_bool_field", finch.DataType.ARRAY_BOOL, nullable=False),
            finch.FieldSchema("array_float_field", finch.DataType.ARRAY_FLOAT, nullable=False),
            finch.FieldSchema("array_double_field", finch.DataType.ARRAY_DOUBLE, nullable=False),
            finch.FieldSchema("array_int32_field", finch.DataType.ARRAY_INT32, nullable=False),
            finch.FieldSchema("array_int64_field", finch.DataType.ARRAY_INT64, nullable=False),
            finch.FieldSchema("array_uint32_field", finch.DataType.ARRAY_UINT32, nullable=False),
            finch.FieldSchema("array_uint64_field", finch.DataType.ARRAY_UINT64, nullable=False),
            finch.FieldSchema("array_string_field", finch.DataType.ARRAY_STRING, nullable=False),
        ],
        vectors=[
            finch.VectorSchema("dense", finch.DataType.VECTOR_FP32, 4),
        ],
    )


def _docs(n: int) -> list[finch.Doc]:
    out: list[finch.Doc] = []
    for i in range(n):
        out.append(
            finch.Doc(
                id=str(i),
                fields={
                    "bool_field": bool(i % 2 == 0),
                    "float_field": float(i),
                    "double_field": float(i),
                    "int32_field": int(i * 10),
                    "int64_field": int(i * 10),
                    "uint32_field": int(i * 10),
                    "uint64_field": int(i * 10),
                    "string_field": f"test_{i}",
                    "array_bool_field": [bool(i % 2 == 0), False],
                    "array_float_field": [float(i), float(i + 1)],
                    "array_double_field": [float(i), float(i + 1)],
                    "array_int32_field": [int(i), int(i + 1)],
                    "array_int64_field": [int(i), int(i + 1)],
                    "array_uint32_field": [int(i), int(i + 1)],
                    "array_uint64_field": [int(i), int(i + 1)],
                    "array_string_field": [f"test_{i}", f"test_{i + 1}"],
                },
                vectors={"dense": [float(i), 0.0, 0.0, 0.0]},
            )
        )
    return out


class CollectionScalarIndexTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            finch.init()
        except RuntimeError:
            pass

    def test_scalar_index_operation_preserves_filter_results(self) -> None:
        root = tempfile.mkdtemp(prefix="finch_test_scalar_index_")
        path = f"{root}/collection"
        try:
            col = finch.create_and_open(path, _schema(), finch.CollectionOption(read_only=False, enable_mmap=True))
            res = col.insert(_docs(5))
            self.assertTrue(all(s.is_ok() for s in res))

            cases = [
                ("bool_field", "bool_field = true"),
                ("float_field", "float_field >= 3.0"),
                ("double_field", "double_field >= 3.0"),
                ("int32_field", "int32_field >= 30"),
                ("int64_field", "int64_field >= 30"),
                ("uint32_field", "uint32_field >= 30"),
                ("uint64_field", "uint64_field >= 30"),
                ("string_field", "string_field >= 'test_3'"),
                ("array_bool_field", "array_bool_field contain_any (false)"),
                ("array_float_field", "array_float_field contain_any (3.0, 4.0)"),
                ("array_double_field", "array_double_field contain_any (3.0, 4.0)"),
                ("array_int32_field", "array_int32_field contain_any (3, 4)"),
                ("array_int64_field", "array_int64_field contain_any (3, 4)"),
                ("array_uint32_field", "array_uint32_field contain_any (3, 4)"),
                ("array_uint64_field", "array_uint64_field contain_any (3, 4)"),
                ("array_string_field", "array_string_field contain_any ('test_3', 'test_4')"),
            ]

            for field_name, query_filter in cases:
                with self.subTest(field=field_name):
                    before = [d.id for d in col.query(filter=query_filter, topk=100)]

                    col.create_index(
                        field_name=field_name,
                        index_param=finch.InvertIndexParam(enable_range_optimization=True),
                        option=finch.IndexOption(),
                    )
                    after_create = [d.id for d in col.query(filter=query_filter, topk=100)]
                    self.assertEqual(before, after_create)

                    col.drop_index(field_name)
                    after_drop = [d.id for d in col.query(filter=query_filter, topk=100)]
                    self.assertEqual(before, after_drop)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)


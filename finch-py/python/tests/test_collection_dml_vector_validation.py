from __future__ import annotations

import math
import shutil
import tempfile
import unittest

import finch


def _make_schema(dim: int) -> finch.CollectionSchema:
    # One field per dense and sparse vector type that DML validation checks.
    return finch.CollectionSchema(
        name="dml_vec_validation",
        fields=[finch.FieldSchema("id", finch.DataType.INT32, nullable=False)],
        vectors=[
            finch.VectorSchema("vector_fp32_field", finch.DataType.VECTOR_FP32, dim),
            finch.VectorSchema("vector_fp16_field", finch.DataType.VECTOR_FP16, dim),
            finch.VectorSchema("vector_int8_field", finch.DataType.VECTOR_INT8, dim),
            finch.VectorSchema("sparse_vector_fp32_field", finch.DataType.SPARSE_VECTOR_FP32, 0),
            finch.VectorSchema("sparse_vector_fp16_field", finch.DataType.SPARSE_VECTOR_FP16, 0),
        ],
    )


def _baseline_vectors(dim: int) -> dict[str, object]:
    return {
        "vector_fp32_field": [0.0] * dim,
        "vector_fp16_field": [0.0] * dim,
        "vector_int8_field": [0] * dim,
        "sparse_vector_fp32_field": {0: 1.0},
        "sparse_vector_fp16_field": {0: 1.0},
    }


class CollectionDmlVectorValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            finch.init()
        except RuntimeError:
            pass

    def test_insert_vector_valid_and_invalid(self) -> None:
        dim = 8
        schema = _make_schema(dim)

        valid_cases: list[tuple[str, list[object]]] = [
            ("vector_fp32_field", [[0.0] * dim, [1.0] * dim, [-1.0] * dim]),
            ("vector_fp16_field", [[0.0] * dim, [1.0] * dim, [-1.0] * dim]),
            ("vector_int8_field", [[100] * dim, [0] * dim, [-100] * dim]),
            ("sparse_vector_fp32_field", [{0: 1.0}, {0: 0.0, 1: 1.0, 2: -1.0}]),
            ("sparse_vector_fp16_field", [{0: 1.0}, {0: 0.0, 1: 1.0, 2: -1.0}]),
        ]

        invalid_cases: list[tuple[str, list[object]]] = [
            ("vector_fp32_field", [None, [], [0.0] * (dim - 1), [0.0] * (dim + 1), ["invalid"], [None] * dim]),
            ("vector_fp16_field", [None, [], [0.0] * (dim - 1), [0.0] * (dim + 1), ["invalid"], [None] * dim]),
            ("vector_int8_field", [None, [], [1] * (dim - 1), [10] * (dim + 1), ["invalid"], [None] * dim]),
            ("sparse_vector_fp32_field", [None, "invalid", {None: 1.0}, {"0": 1.0}, {0: "invalid"}, {0: None}, {-1: 1.0}]),
            ("sparse_vector_fp16_field", [None, "invalid", {None: 1.0}, {"0": 1.0}, {0: "invalid"}, {0: None}, {-1: 1.0}]),
        ]

        root = tempfile.mkdtemp(prefix="finch_test_vec_val_")
        path = f"{root}/collection"
        try:
            col = finch.create_and_open(path, schema, finch.CollectionOption(read_only=False, enable_mmap=True))

            for vector_field, values in valid_cases:
                for i, v in enumerate(values):
                    with self.subTest(vector_field=vector_field, value=v):
                        vectors = _baseline_vectors(dim)
                        vectors[vector_field] = v
                        doc = finch.Doc(id=str(i), fields={"id": i}, vectors=vectors)
                        s = col.insert(doc)
                        self.assertTrue(s.ok(), str(s))
                        col.delete(str(i))

            for vector_field, values in invalid_cases:
                for i, v in enumerate(values):
                    with self.subTest(vector_field=vector_field, value=v):
                        vectors = _baseline_vectors(dim)
                        vectors[vector_field] = v
                        doc = finch.Doc(id=str(i), fields={"id": i}, vectors=vectors)
                        with self.assertRaises(Exception):
                            col.insert(doc)
                        self.assertEqual(col.stats.doc_count, 0)

            col.destroy()
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_update_vector_only_patch(self) -> None:
        # An update may set vectors without setting any field.
        dim = 8
        schema = _make_schema(dim)

        root = tempfile.mkdtemp(prefix="finch_test_update_vec_only_")
        path = f"{root}/collection"
        try:
            col = finch.create_and_open(path, schema, finch.CollectionOption(read_only=False, enable_mmap=True))

            base = finch.Doc(
                id="1",
                fields={"id": 1},
                vectors=_baseline_vectors(dim),
            )
            self.assertTrue(col.insert(base).ok())

            patch = finch.Doc(id="1", vectors={"vector_fp32_field": [0.3] * dim})
            self.assertTrue(col.update(patch).ok())
            got = col.fetch("1")["1"]
            v = got.vector("vector_fp32_field")
            self.assertEqual(len(v), dim)
            self.assertTrue(all(math.isclose(float(x), 0.3, rel_tol=1e-6, abs_tol=1e-6) for x in v))
            self.assertEqual(got.vector("vector_int8_field"), [0] * dim)
            self.assertEqual(got.vector("sparse_vector_fp32_field"), {0: 1.0})

            patch2 = finch.Doc(id="1", vectors={"sparse_vector_fp32_field": {1: 2.0, 2: 3.0}})
            self.assertTrue(col.update(patch2).ok())
            got2 = col.fetch("1")["1"]
            self.assertEqual(got2.vector("sparse_vector_fp32_field"), {1: 2.0, 2: 3.0})

            col.destroy()
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)

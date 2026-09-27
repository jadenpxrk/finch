from __future__ import annotations

import shutil
import tempfile
import unittest

import finch


DOCID_VALID_LIST = [
    "1valid_Id",
    "123.45",
    "123abc",
    "-!@#$%+=.123abc_+",
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ123456789012",
]

DOCID_INVALID_LIST = [
    None,
    "",
    "()qsd123",
    " ",
    "/&AS12",
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890121",
]


def _make_schema(nullable: bool) -> finch.CollectionSchema:
    # Subset of reference `full_schema_new` coverage, focused on scalar type validation.
    fields = [
        finch.FieldSchema("bool_field", finch.DataType.BOOL, nullable=nullable),
        finch.FieldSchema("float_field", finch.DataType.FLOAT, nullable=nullable),
        finch.FieldSchema("double_field", finch.DataType.DOUBLE, nullable=nullable),
        finch.FieldSchema("int32_field", finch.DataType.INT32, nullable=nullable),
        finch.FieldSchema("int64_field", finch.DataType.INT64, nullable=nullable),
        finch.FieldSchema("uint32_field", finch.DataType.UINT32, nullable=nullable),
        finch.FieldSchema("uint64_field", finch.DataType.UINT64, nullable=nullable),
        finch.FieldSchema("string_field", finch.DataType.STRING, nullable=nullable),
        finch.FieldSchema("array_bool_field", finch.DataType.ARRAY_BOOL, nullable=nullable),
        finch.FieldSchema("array_float_field", finch.DataType.ARRAY_FLOAT, nullable=nullable),
        finch.FieldSchema("array_double_field", finch.DataType.ARRAY_DOUBLE, nullable=nullable),
        finch.FieldSchema("array_int32_field", finch.DataType.ARRAY_INT32, nullable=nullable),
        finch.FieldSchema("array_int64_field", finch.DataType.ARRAY_INT64, nullable=nullable),
        finch.FieldSchema("array_uint32_field", finch.DataType.ARRAY_UINT32, nullable=nullable),
        finch.FieldSchema("array_uint64_field", finch.DataType.ARRAY_UINT64, nullable=nullable),
        finch.FieldSchema("array_string_field", finch.DataType.ARRAY_STRING, nullable=nullable),
    ]
    return finch.CollectionSchema(
        name="dml_validation",
        fields=fields,
        vectors=finch.VectorSchema("dense", finch.DataType.VECTOR_FP32, 4, index_param=finch.HnswIndexParam()),
    )


def _baseline_fields() -> dict[str, object]:
    return {
        "bool_field": True,
        "float_field": 1.0,
        "double_field": 1.0,
        "int32_field": 1,
        "int64_field": 1,
        "uint32_field": 1,
        "uint64_field": 1,
        "string_field": "test",
        "array_bool_field": [True, False],
        "array_float_field": [1.0, 2.0],
        "array_double_field": [1.0, 2.0],
        "array_int32_field": [1, 2],
        "array_int64_field": [1, 2],
        "array_uint32_field": [1, 2],
        "array_uint64_field": [1, 2],
        "array_string_field": ["a", "b"],
    }


class FinchReferenceCollectionDmlValidationParitySubsetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            finch.init()
        except RuntimeError:
            pass

    def test_insert_docid_valid(self) -> None:
        root = tempfile.mkdtemp(prefix="finch_parity_docid_valid_")
        path = f"{root}/collection"
        try:
            col = finch.create_and_open(path, _make_schema(nullable=True), finch.CollectionOption(read_only=False, enable_mmap=True))
            for doc_id in DOCID_VALID_LIST:
                with self.subTest(doc_id=doc_id):
                    doc = finch.Doc(id=doc_id, fields=_baseline_fields(), vectors={"dense": [1.0, 2.0, 3.0, 4.0]})
                    s = col.insert(doc)
                    self.assertTrue(s.ok(), str(s))
                    col.delete(doc_id)
            col.destroy()
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_insert_docid_invalid_raises(self) -> None:
        root = tempfile.mkdtemp(prefix="finch_parity_docid_invalid_")
        path = f"{root}/collection"
        try:
            col = finch.create_and_open(path, _make_schema(nullable=True), finch.CollectionOption(read_only=False, enable_mmap=True))
            for doc_id in DOCID_INVALID_LIST:
                with self.subTest(doc_id=doc_id):
                    doc = finch.Doc(id=doc_id, fields=_baseline_fields(), vectors={"dense": [1.0, 2.0, 3.0, 4.0]})  # type: ignore[arg-type]
                    with self.assertRaises(Exception):
                        col.insert(doc)
                    self.assertEqual(col.stats.doc_count, 0)
            col.destroy()
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_upsert_docid_valid_and_invalid_subset(self) -> None:
        root = tempfile.mkdtemp(prefix="finch_parity_upsert_docid_")
        path = f"{root}/collection"
        try:
            col = finch.create_and_open(path, _make_schema(nullable=True), finch.CollectionOption(read_only=False, enable_mmap=True))

            for doc_id in DOCID_VALID_LIST:
                with self.subTest(doc_id=doc_id):
                    doc = finch.Doc(id=doc_id, fields=_baseline_fields(), vectors={"dense": [1.0, 2.0, 3.0, 4.0]})
                    s = col.upsert(doc)
                    self.assertTrue(s.ok(), str(s))
                    self.assertEqual(col.stats.doc_count, 1)
                    col.delete(doc_id)
                    self.assertEqual(col.stats.doc_count, 0)

            for doc_id in DOCID_INVALID_LIST:
                with self.subTest(doc_id=doc_id):
                    doc = finch.Doc(id=doc_id, fields=_baseline_fields(), vectors={"dense": [1.0, 2.0, 3.0, 4.0]})  # type: ignore[arg-type]
                    with self.assertRaises(Exception):
                        col.upsert(doc)
                    self.assertEqual(col.stats.doc_count, 0)

            col.destroy()
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_update_docid_invalid_raises(self) -> None:
        root = tempfile.mkdtemp(prefix="finch_parity_update_docid_")
        path = f"{root}/collection"
        try:
            col = finch.create_and_open(path, _make_schema(nullable=True), finch.CollectionOption(read_only=False, enable_mmap=True))

            for doc_id in DOCID_INVALID_LIST:
                with self.subTest(doc_id=doc_id):
                    doc = finch.Doc(id=doc_id, fields={"int32_field": 1})  # type: ignore[arg-type]
                    with self.assertRaises(Exception):
                        col.update(doc)
                    self.assertEqual(col.stats.doc_count, 0)

            col.destroy()
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_insert_fields_valid_and_invalid_subset(self) -> None:
        # Mirrors reference field value validation behavior (subset).
        valid_cases: list[tuple[str, list[object]]] = [
            ("int32_field", [None, 0, 1, -1, 2147483647, -2147483648]),
            ("uint32_field", [None, 0, 1, 4294967295]),
            ("string_field", [None, "", "a", "test_name", "这是一个中文名称测试"]),
            ("array_int32_field", [None, [], [0], [1, 2, 3]]),
        ]
        invalid_cases: list[tuple[str, list[object]]] = [
            ("int32_field", ["invalid", [1], {"value": 1}, 2147483648, -2147483649]),
            ("uint32_field", ["invalid", [1], {"value": 1}, 4294967296, -1]),
            ("bool_field", ["True", "False", ""]),
            ("array_int32_field", ["invalid", 1, {"x": 1}, [1.0]]),
        ]

        for nullable in (True, False):
            with self.subTest(nullable=nullable):
                root = tempfile.mkdtemp(prefix="finch_parity_field_val_")
                path = f"{root}/collection"
                try:
                    col = finch.create_and_open(path, _make_schema(nullable=nullable), finch.CollectionOption(read_only=False, enable_mmap=True))
                    for field_name, values in valid_cases:
                        for i, v in enumerate(values):
                            with self.subTest(field_name=field_name, value=v):
                                fields = _baseline_fields()
                                fields[field_name] = v
                                doc = finch.Doc(id=str(i), fields=fields, vectors={"dense": [1.0, 2.0, 3.0, 4.0]})
                                if v is None and not nullable:
                                    with self.assertRaises(Exception):
                                        col.insert(doc)
                                    self.assertEqual(col.stats.doc_count, 0)
                                else:
                                    s = col.insert(doc)
                                    self.assertTrue(s.ok(), str(s))
                                    col.delete(str(i))

                    for field_name, values in invalid_cases:
                        for i, v in enumerate(values):
                            with self.subTest(field_name=field_name, value=v):
                                fields = _baseline_fields()
                                fields[field_name] = v
                                doc = finch.Doc(id=str(i), fields=fields, vectors={"dense": [1.0, 2.0, 3.0, 4.0]})
                                with self.assertRaises(Exception):
                                    col.insert(doc)
                                self.assertEqual(col.stats.doc_count, 0)
                    col.destroy()
                finally:
                    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)

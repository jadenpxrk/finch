from __future__ import annotations

import pickle
import shutil
import tempfile
import unittest

import finch


def _typed_fields() -> list[tuple[str, object, object]]:
    return [
        ("author", finch.FieldSchema("author", finch.DataType.STRING), "Tom"),
        ("blob", finch.FieldSchema("blob", finch.DataType.BINARY), b"\x00\xff"),
        ("tags", finch.FieldSchema("tags", finch.DataType.ARRAY_STRING), ["tag1", "tag2"]),
        ("weights", finch.FieldSchema("weights", finch.DataType.ARRAY_FLOAT), [1.5, 2.5]),
        ("chunks", finch.FieldSchema("chunks", finch.DataType.ARRAY_BINARY), [b"a", b"bc"]),
        ("dense16", finch.VectorSchema("dense16", dim=3, data_type=finch.DataType.VECTOR_FP16), [1.0, 2.0, 3.0]),
        ("dense32", finch.VectorSchema("dense32", dim=3, data_type=finch.DataType.VECTOR_FP32), [1.25, 2.25, 3.25]),
        ("s16", finch.VectorSchema("s16", data_type=finch.DataType.SPARSE_VECTOR_FP16), {1: 1.1, 2: 2.2}),
        ("s32", finch.VectorSchema("s32", data_type=finch.DataType.SPARSE_VECTOR_FP32), {1: 1.125, 2: 2.25}),
    ]


class FinchReferencePickleCoreDocParitySubsetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            finch.init()
        except RuntimeError:
            pass

    def test_core_doc_pickle_roundtrip_preserves_types_for_insert(self) -> None:
        from finch._finch import PyDoc as _Doc
        from finch._finch import create_and_open as _create_and_open

        root = tempfile.mkdtemp(prefix="finch_parity_pickle_doc_")
        path = f"{root}/collection"
        try:
            schema = finch.CollectionSchema("pickle_doc")
            for _, field, _ in _typed_fields():
                schema.add_field(field)

            col = _create_and_open(
                path,
                schema._get_object(),
                finch.CollectionOption(read_only=False, enable_mmap=True),
            )

            d = _Doc()
            d.set_pk("p1")
            for name, field, value in _typed_fields():
                d.set_any(name, field._get_object(), value)

            d2 = pickle.loads(pickle.dumps(d))
            self.assertEqual(d2.pk(), "p1")
            self.assertEqual(d2.get_any("author", finch.DataType.STRING), "Tom")
            self.assertEqual(d2.get_any("blob", finch.DataType.BINARY), b"\x00\xff")
            self.assertEqual(d2.get_any("tags", finch.DataType.ARRAY_STRING), ["tag1", "tag2"])
            self.assertEqual(d2.get_any("chunks", finch.DataType.ARRAY_BINARY), [b"a", b"bc"])

            results = col.insert([d2])
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].is_ok())

            got = col.fetch(["p1"])["p1"]
            self.assertEqual(got.get_any("author", finch.DataType.STRING), "Tom")
            self.assertEqual(got.get_any("blob", finch.DataType.BINARY), b"\x00\xff")
            self.assertEqual(got.get_any("tags", finch.DataType.ARRAY_STRING), ["tag1", "tag2"])
            self.assertEqual(got.get_any("weights", finch.DataType.ARRAY_FLOAT), [1.5, 2.5])
            self.assertEqual(got.get_any("chunks", finch.DataType.ARRAY_BINARY), [b"a", b"bc"])
            self.assertEqual(got.get_any("dense16", finch.DataType.VECTOR_FP16), [1.0, 2.0, 3.0])
            self.assertEqual(got.get_any("dense32", finch.DataType.VECTOR_FP32), [1.25, 2.25, 3.25])
            self.assertIsInstance(got.get_any("s16", finch.DataType.SPARSE_VECTOR_FP16), dict)
            self.assertIsInstance(got.get_any("s32", finch.DataType.SPARSE_VECTOR_FP32), dict)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)


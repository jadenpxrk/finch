from __future__ import annotations

import tempfile
import unittest

import finch


def _schema() -> finch.CollectionSchema:
    return finch.CollectionSchema(
        name="dql_subset",
        fields=[
            finch.FieldSchema("int32_field", finch.DataType.INT32, nullable=True),
            finch.FieldSchema("bool_field", finch.DataType.BOOL, nullable=True),
        ],
        vectors=[
            finch.VectorSchema(
                "vector_fp32_field",
                finch.DataType.VECTOR_FP32,
                dimension=8,
                index_param=finch.FlatIndexParam(),
            )
        ],
    )


class FinchReferenceCollectionDqlMissingParitySubsetTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="finch_parity_dql_subset_")
        self.path = f"{self._tmp.name}/col"
        self.col = finch.create_and_open(self.path, _schema(), finch.CollectionOption())

        docs = [
            finch.Doc(
                id=str(i),
                fields={"int32_field": i, "bool_field": (i % 2 == 0)},
                vectors={"vector_fp32_field": [float(i)] * 8},
            )
            for i in range(10)
        ]
        res = self.col.insert(docs)
        self.assertEqual(len(res), len(docs))
        for s in res:
            self.assertTrue(s.ok())

    def tearDown(self) -> None:
        try:
            self.col.destroy()
        finally:
            self._tmp.cleanup()

    def test_query_filter_none_and_empty_string_are_equivalent(self) -> None:
        out_default = self.col.query(topk=100)
        out_empty = self.col.query(filter="", topk=100)
        out_none = self.col.query(filter=None, topk=100)

        self.assertEqual({d.id for d in out_default}, {d.id for d in out_empty})
        self.assertEqual({d.id for d in out_default}, {d.id for d in out_none})

    def test_query_invalid_filter_raises(self) -> None:
        for bad in [
            "int32_field >",
            "int32_field > > 5",
            "int32_field = 'string'",
            "nonexistent_field = 5",
            "int32_field > 5 and",
        ]:
            with self.subTest(filter=bad):
                with self.assertRaises(Exception):
                    self.col.query(filter=bad, topk=10)

    def test_output_fields_preserves_order(self) -> None:
        out = self.col.query(output_fields=["bool_field", "int32_field"], topk=1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].field_names(), ["bool_field", "int32_field"])


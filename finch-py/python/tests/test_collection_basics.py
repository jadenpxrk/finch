from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import finch


def _schema() -> finch.CollectionSchema:
    return finch.CollectionSchema(
        name="test_collection",
        fields=[
            finch.FieldSchema(
                "id",
                finch.DataType.INT64,
                nullable=False,
                index_param=finch.InvertIndexParam(enable_range_optimization=True),
            ),
            finch.FieldSchema(
                "name",
                finch.DataType.STRING,
                nullable=False,
                index_param=finch.InvertIndexParam(),
            ),
            finch.FieldSchema("weight", finch.DataType.FLOAT, nullable=True),
            finch.FieldSchema("height", finch.DataType.INT32, nullable=True),
        ],
        vectors=[
            finch.VectorSchema(
                "dense",
                finch.DataType.VECTOR_FP32,
                dimension=128,
                index_param=finch.HnswIndexParam(),
            ),
            finch.VectorSchema(
                "sparse",
                finch.DataType.SPARSE_VECTOR_FP32,
                index_param=finch.HnswIndexParam(),
            ),
        ],
    )


def _single_doc(doc_id: str = "0") -> finch.Doc:
    i = int(doc_id)
    return finch.Doc(
        id=doc_id,
        fields={"id": i, "name": "test", "weight": 80.0, "height": i + 140},
        vectors={"dense": [i + 0.1] * 128, "sparse": {1: 1.0, 2: 2.0, 3: 3.0}},
    )


class CollectionBasicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        finch.init(log_type=finch.LogType.CONSOLE, log_level=finch.LogLevel.ERROR, log_dir="./log")

    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self._path = str(Path(self._td.name) / "test_collection")
        self._schema = _schema()
        self._option = finch.CollectionOption(read_only=False, enable_mmap=True)
        self.col = finch.create_and_open(path=self._path, schema=self._schema, option=self._option)

    def tearDown(self) -> None:
        try:
            if getattr(self, "col", None) is not None and hasattr(self.col, "destroy"):
                try:
                    self.col.destroy()
                except Exception:
                    pass
        finally:
            self._td.cleanup()

    def test_collection_stats_default(self):
        stats = self.col.stats
        self.assertEqual(stats.doc_count, 0)
        self.assertEqual(len(stats.index_completeness), 2)
        self.assertEqual(stats.index_completeness["dense"], 1)
        self.assertEqual(stats.index_completeness["sparse"], 1)

    def test_create_and_drop_scalar_index(self):
        # Before: no index on weight.
        field_schema = self.col.schema.field("weight")
        self.assertIsNotNone(field_schema)
        self.assertIsNone(field_schema.index_param)

        self.col.create_index(
            field_name="weight",
            index_param=finch.InvertIndexParam(),
            option=finch.IndexOption(),
        )
        field_schema2 = self.col.schema.field("weight")
        self.assertIsNotNone(field_schema2)
        idx = field_schema2.index_param
        self.assertIsNotNone(idx)
        self.assertEqual(idx.type, finch.IndexType.INVERT)
        self.assertFalse(idx.enable_range_optimization)
        self.assertFalse(idx.enable_extended_wildcard)

        self.col.drop_index("weight")
        field_schema3 = self.col.schema.field("weight")
        self.assertIsNotNone(field_schema3)
        self.assertIsNone(field_schema3.index_param)

    def test_insert_and_duplicate(self):
        doc = _single_doc("0")
        r1 = self.col.insert(doc)
        self.assertTrue(bool(r1))
        self.assertTrue(r1.ok())
        self.assertEqual(self.col.stats.doc_count, 1)

        r2 = self.col.insert(doc)
        self.assertTrue(bool(r2))
        self.assertEqual(r2.code(), finch.StatusCode.ALREADY_EXISTS)
        self.assertEqual(self.col.stats.doc_count, 1)

    def test_insert_missing_required_fields_raises(self):
        # Missing required scalar fields (id + name).
        doc = finch.Doc(id="0", vectors={"dense": [0.1] * 128, "sparse": {1: 1.0}})
        with self.assertRaises(ValueError) as ctx:
            self.col.insert(doc)
        self.assertIn("field[id] is configured not nullable", str(ctx.exception))

        # Missing required `name`.
        doc2 = finch.Doc(id="0", fields={"id": 1}, vectors={"dense": [0.1] * 128, "sparse": {1: 1.0}})
        with self.assertRaises(ValueError) as ctx2:
            self.col.insert(doc2)
        self.assertIn("field[name] is configured not nullable", str(ctx2.exception))

    def test_fetch_omits_missing_ids(self):
        docs = [_single_doc(str(i)) for i in range(3)]
        rs = self.col.insert(docs)
        self.assertEqual(len(rs), 3)
        out = self.col.fetch(ids=["x", "0", "1", "2"])
        self.assertEqual(len(out), 3)
        self.assertNotIn("x", out)

    def test_update_not_found(self):
        doc = _single_doc("0")
        r = self.col.update(doc)
        self.assertTrue(bool(r))
        self.assertEqual(r.code(), finch.StatusCode.NOT_FOUND)
        self.assertEqual(self.col.stats.doc_count, 0)

    def test_query_smoke(self):
        doc = _single_doc("0")
        r = self.col.insert(doc)
        self.assertTrue(r.ok())
        out = self.col.query()
        self.assertEqual(len(out), 1)

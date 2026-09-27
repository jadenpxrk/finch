from __future__ import annotations

import tempfile
import unittest

import finch


def _make_schema() -> finch.CollectionSchema:
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
                data_type=finch.DataType.VECTOR_FP32,
                dimension=16,
                index_param=finch.HnswIndexParam(),
            ),
            finch.VectorSchema(
                "sparse",
                data_type=finch.DataType.SPARSE_VECTOR_FP32,
                dimension=0,
                index_param=finch.HnswIndexParam(),
            ),
        ],
    )


class FinchReferenceCollectionMoreParityTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="finch_parity_collection_more_")
        self.path = f"{self._tmp.name}/test_collection"
        self.schema = _make_schema()
        self.option = finch.CollectionOption(read_only=False, enable_mmap=True)
        self.coll = finch.create_and_open(path=self.path, schema=self.schema, option=self.option)

    def tearDown(self):
        try:
            self.coll.destroy()
        finally:
            self._tmp.cleanup()

    def test_init_api_exists(self):
        finch.init(log_type=finch.LogType.CONSOLE, log_level=finch.LogLevel.ERROR, log_dir="./log")

    def test_add_column_nullable_false_rejected(self):
        with self.assertRaises(ValueError):
            self.coll.add_column(finch.FieldSchema("age", finch.DataType.INT32, nullable=False))

    def test_fetch_skips_missing_ids(self):
        docs = [
            finch.Doc(
                id=f"{i}",
                fields={"id": i, "name": "t"},
                vectors={"dense": [0.1] * 16, "sparse": {1: 1.0}},
            )
            for i in range(1, 6)
        ]
        self.coll.insert(docs)
        out = self.coll.fetch(["x"] + [d.id for d in docs])
        self.assertEqual(len(out), 5)
        self.assertNotIn("x", out)

    def test_query_empty_collection_returns_empty(self):
        self.assertEqual(self.coll.stats.doc_count, 0)
        out = self.coll.query()
        self.assertEqual(len(out), 0)

    def test_query_by_id_vector(self):
        docs = [
            finch.Doc(
                id=f"{i}",
                fields={"id": i, "name": "t"},
                vectors={"dense": [i + 0.1] * 16, "sparse": {1: 1.0}},
            )
            for i in range(1, 31)
        ]
        self.coll.insert(docs)
        out = self.coll.query(finch.VectorQuery(field_name="dense", id=docs[0].id))
        self.assertEqual(len(out), 10)

    def test_delete_existing_doc(self):
        doc = finch.Doc(
            id="0",
            fields={"id": 0, "name": "t"},
            vectors={"dense": [0.1] * 16, "sparse": {1: 1.0}},
        )
        self.assertTrue(self.coll.insert(doc).ok())
        self.assertEqual(self.coll.stats.doc_count, 1)
        s = self.coll.delete(doc.id)
        self.assertTrue(s.ok())
        self.assertEqual(self.coll.stats.doc_count, 0)

    def test_optimize_does_not_throw(self):
        self.coll.optimize(option=finch.OptimizeOption())


from __future__ import annotations

import shutil
import tempfile
import unittest


class TestSqlQueryAndSystemColumns(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import finch

        # init is a one-shot global initializer.
        try:
            finch.init()
        except RuntimeError:
            pass

    def test_system_columns_are_preserved_in_python_docs_when_requested(self) -> None:
        import finch

        root = tempfile.mkdtemp(prefix="finch_py_sql_syscols_")
        path = f"{root}/collection"
        try:
            schema = finch.CollectionSchema("test")
            schema.add_field(finch.VectorSchema("emb", dim=4, data_type=finch.DataType.VectorFp32))
            schema.add_field(finch.FieldSchema("label", finch.DataType.String, nullable=True))

            col = finch.create_and_open(path, schema, finch.CollectionOption(read_only=False, enable_mmap=True))
            results = col.insert(
                [
                    finch.Doc(id="d0", vectors={"emb": [1.0, 0.0, 0.0, 0.0]}, fields={"label": "d0"}),
                    finch.Doc(id="d1", vectors={"emb": [0.0, 1.0, 0.0, 0.0]}, fields={"label": "d1"}),
                ]
            )
            self.assertTrue(all(s.is_ok() for s in results))
            col.flush()

            res = col.query(
                vectors=finch.VectorQuery("emb", vector=[1.0, 0.0, 0.0, 0.0]),
                topk=1,
                output_fields=["_finch_uid_", "_finch_g_doc_id_", "_finch_row_id_", "_finch_score"],
            )
            self.assertEqual(len(res), 1)
            doc = res[0]
            self.assertEqual(doc.fields.get("_finch_uid_"), "d0")
            self.assertIsInstance(doc.fields.get("_finch_g_doc_id_"), int)
            self.assertIsInstance(doc.fields.get("_finch_row_id_"), int)
            self.assertIsInstance(doc.fields.get("_finch_score"), float)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_query_sql_preserves_aliases_and_system_columns(self) -> None:
        import finch

        root = tempfile.mkdtemp(prefix="finch_py_sql_query_sql_")
        path = f"{root}/collection"
        try:
            schema = finch.CollectionSchema("test")
            schema.add_field(finch.VectorSchema("emb", dim=4, data_type=finch.DataType.VectorFp32))
            schema.add_field(finch.FieldSchema("label", finch.DataType.String, nullable=True))

            col = finch.create_and_open(path, schema, finch.CollectionOption(read_only=False, enable_mmap=True))
            results = col.insert(
                [
                    finch.Doc(id="d0", vectors={"emb": [1.0, 0.0, 0.0, 0.0]}, fields={"label": "d0"}),
                    finch.Doc(id="d1", vectors={"emb": [0.0, 1.0, 0.0, 0.0]}, fields={"label": "d1"}),
                ]
            )
            self.assertTrue(all(s.is_ok() for s in results))
            col.flush()

            if not hasattr(col, "query_sql"):
                self.skipTest("query_sql is not available in this build")

            res = col.query_sql(
                "SELECT _finch_uid_ AS u, _finch_score AS s, label AS l FROM test WHERE emb = [1,0,0,0] LIMIT 1"
            )
            self.assertEqual(len(res), 1)
            doc = res[0]
            self.assertEqual(doc.fields.get("u"), "d0")
            self.assertIsInstance(doc.fields.get("s"), float)
            self.assertEqual(doc.fields.get("l"), "d0")
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)


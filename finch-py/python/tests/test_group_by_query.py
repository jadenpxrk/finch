from __future__ import annotations

import shutil
import tempfile
import unittest


class TestGroupByQuery(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import finch

        try:
            finch.init()
        except RuntimeError:
            pass

    def test_group_by_query_semantics(self) -> None:
        import finch

        root = tempfile.mkdtemp(prefix="finch_py_group_by_")
        path = f"{root}/collection"
        try:
            schema = finch.CollectionSchema("test")
            schema.add_field(finch.VectorSchema("emb", dim=4, data_type=finch.DataType.VectorFp32))
            schema.add_field(finch.FieldSchema("category", finch.DataType.String, nullable=True))

            col = finch.create_and_open(path, schema, finch.CollectionOption(read_only=False, enable_mmap=True))
            results = col.insert(
                [
                    finch.Doc(id="a1", vectors={"emb": [1.0, 0.0, 0.0, 0.0]}, fields={"category": "A"}),
                    finch.Doc(id="a2", vectors={"emb": [0.9, 0.1, 0.0, 0.0]}, fields={"category": "A"}),
                    finch.Doc(id="b1", vectors={"emb": [0.0, 1.0, 0.0, 0.0]}, fields={"category": "B"}),
                    finch.Doc(id="b2", vectors={"emb": [0.0, 0.9, 0.1, 0.0]}, fields={"category": "B"}),
                    # NULL group key is ignored (no NULL group).
                    finch.Doc(id="n1", vectors={"emb": [0.8, 0.2, 0.0, 0.0]}),
                ]
            )
            self.assertTrue(all(s.is_ok() for s in results))
            col.flush()

            groups = col.group_by_query(
                finch.VectorQuery("emb", vector=[1.0, 0.0, 0.0, 0.0]),
                group_by_field="category",
                # group_topk = groups, group_count = docs per group
                group_topk=1,
                group_count=2,
                topk=10,
                output_fields=[],
            )
            self.assertEqual(len(groups), 1)
            self.assertEqual(groups[0].group_value, "A")
            self.assertEqual([d.id for d in groups[0].docs], ["a1", "a2"])
            # The group-by field was not selected and must not leak into docs.
            self.assertTrue(all("category" not in d.fields for d in groups[0].docs))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_group_by_query_supports_system_uid_column(self) -> None:
        import finch

        root = tempfile.mkdtemp(prefix="finch_py_group_by_sysuid_")
        path = f"{root}/collection"
        try:
            schema = finch.CollectionSchema("test")
            schema.add_field(finch.VectorSchema("emb", dim=4, data_type=finch.DataType.VectorFp32))

            col = finch.create_and_open(path, schema, finch.CollectionOption(read_only=False, enable_mmap=True))
            results = col.insert(
                [
                    finch.Doc(id="d1", vectors={"emb": [1.0, 0.0, 0.0, 0.0]}),
                    finch.Doc(id="d2", vectors={"emb": [0.9, 0.1, 0.0, 0.0]}),
                    finch.Doc(id="d3", vectors={"emb": [0.0, 1.0, 0.0, 0.0]}),
                ]
            )
            self.assertTrue(all(s.is_ok() for s in results))
            col.flush()

            groups = col.group_by_query(
                finch.VectorQuery("emb", vector=[1.0, 0.0, 0.0, 0.0]),
                group_by_field="_finch_uid_",
                group_topk=2,
                group_count=1,
                topk=10,
                output_fields=[],
            )
            self.assertEqual(len(groups), 2)
            self.assertEqual([g.group_value for g in groups], ["d1", "d2"])
            self.assertEqual([g.docs[0].id for g in groups], ["d1", "d2"])
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)


from __future__ import annotations

import shutil
import tempfile
import unittest

import finch


class FinchReferenceCollectionDestroyParitySubsetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            finch.init()
        except RuntimeError:
            pass

    def test_destroy_makes_collection_unusable_and_removes_path(self) -> None:
        try:
            import numpy as np
        except Exception:  # pragma: no cover
            self.skipTest("numpy is not available")

        root = tempfile.mkdtemp(prefix="finch_parity_destroy_")
        path = f"{root}/collection"
        opt = finch.CollectionOption(read_only=False, enable_mmap=True)
        try:
            schema = finch.CollectionSchema(
                name="destroy",
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
                ],
                vectors=[
                    finch.VectorSchema(
                        "dense",
                        finch.DataType.VECTOR_FP32,
                        8,
                        index_param=finch.HnswIndexParam(),
                    ),
                ],
            )
            col = finch.create_and_open(path=path, schema=schema, option=opt)
            st = col.insert(
                finch.Doc(
                    id="1",
                    fields={"id": 1, "name": "x"},
                    vectors={"dense": np.random.random(8).tolist()},
                )
            )
            self.assertTrue(st.ok())
            self.assertEqual(col.stats.doc_count, 1)

            col.destroy()

            with self.assertRaises(Exception):
                _ = col.stats
            with self.assertRaises(Exception):
                _ = col.schema
            with self.assertRaises(Exception):
                finch.open(path=path, option=opt)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)

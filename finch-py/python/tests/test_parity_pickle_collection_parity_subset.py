from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path
import gc

import finch


class FinchReferencePickleCollectionParitySubsetTest(unittest.TestCase):
    def test_core_collection_pickle_reopens_by_path(self):
        schema = finch.CollectionSchema(
            name="test_collection",
            vectors=finch.VectorSchema("image", finch.DataType.VECTOR_FP32, 4),
        )

        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "col")
            from finch._finch import create_and_open as core_create
            from finch.model.convert import convert_to_core_doc

            core = core_create(path, schema._get_object(), finch.CollectionOption())
            core.insert([convert_to_core_doc(finch.Doc(id="1", vectors={"image": [1.0, 2.0, 3.0, 4.0]}), schema)])
            core.flush()

            blob = pickle.dumps(core)
            del core
            gc.collect()

            core2 = pickle.loads(blob)

            self.assertEqual(core2.path(), path)
            self.assertEqual(core2.stats().doc_count, 1)

            core2.destroy()
            del core2
            gc.collect()

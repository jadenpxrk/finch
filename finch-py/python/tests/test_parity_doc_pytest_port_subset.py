from __future__ import annotations

import unittest

import finch


class ReferenceDocPytestPortSubsetTest(unittest.TestCase):
    def test_doc_vectors_and_fields(self):
        doc = finch.Doc(id="1", vectors={"dense": [1, 2, 3]})
        self.assertEqual(doc.id, "1")
        self.assertEqual(doc.vector("dense"), [1, 2, 3])

        doc2 = finch.Doc(
            id="1",
            vectors={"dense": [1, 2, 3], "sparse": {1: 1.0, 2: 2.0, 3: 3.0}},
        )
        self.assertEqual(doc2.vector("dense"), [1, 2, 3])
        self.assertEqual(doc2.vector("sparse"), {1: 1.0, 2: 2.0, 3: 3.0})

        doc3 = finch.Doc(
            id="1",
            vectors={"image": [1, 2, 3], "keys": {1: 1.0}},
            fields={"author": "Tom", "age": 19, "is_male": True, "weight": 60.5},
        )
        self.assertEqual(doc3.field("author"), "Tom")
        self.assertEqual(doc3.field("age"), 19)
        self.assertEqual(doc3.field("is_male"), True)
        self.assertEqual(doc3.field("weight"), 60.5)

    def test_doc_from_tuple_converts_numpy_vectors(self):
        try:
            import numpy as np
        except Exception:  # pragma: no cover
            self.skipTest("numpy is not available")

        d = finch.Doc._from_tuple(
            (
                "1",
                0.0,
                None,
                {
                    "image": np.array([1, 2, 3]),
                    "keys": {1: 1.0, 2: 2.0, 3: 3.0},
                },
            )
        )
        self.assertEqual(d.vector("image"), [1, 2, 3])
        self.assertEqual(d.vector("keys"), {1: 1.0, 2: 2.0, 3: 3.0})

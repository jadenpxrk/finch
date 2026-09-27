from __future__ import annotations

import unittest


class FinchReferenceModelExportsParitySubsetTest(unittest.TestCase):
    def test_finch_model_reexports_match_reference_expectations(self) -> None:
        import finch

        self.assertTrue(hasattr(finch, "model"))

        from finch.model import Collection, Doc
        from finch.model import CollectionSchema, FieldSchema, VectorSchema, VectorQuery

        self.assertIsNotNone(Collection)
        self.assertIsNotNone(Doc)
        self.assertIsNotNone(CollectionSchema)
        self.assertIsNotNone(FieldSchema)
        self.assertIsNotNone(VectorSchema)
        self.assertIsNotNone(VectorQuery)


if __name__ == "__main__":
    unittest.main(verbosity=2)


from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import finch


class DefaultLocalSparseEmbeddingTest(unittest.TestCase):
    @patch("finch.extension.sentence_transformer_function.require_module")
    def test_filters_positive_values_and_wraps_runtime_errors(self, mock_require_module) -> None:
        # Stub sentence_transformers.SentenceTransformer used by base class.
        mock_st = Mock()

        class _SparseMatrix:
            def __getitem__(self, idx):
                return self

            def toarray(self):
                return [[1.0, 0.0, -0.5, 2.0]]

        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.encode_query = Mock(return_value=_SparseMatrix())
        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        finch.DefaultLocalSparseEmbedding.clear_cache()
        emb = finch.DefaultLocalSparseEmbedding(encoding_type="query")
        out = emb.embed("hello")
        self.assertEqual(out, {0: 1.0, 3: 2.0})

        # An inference error surfaces as a RuntimeError.
        mock_model.encode_query = Mock(side_effect=Exception("boom"))
        finch.DefaultLocalSparseEmbedding.clear_cache()
        emb2 = finch.DefaultLocalSparseEmbedding(encoding_type="query")
        with self.assertRaises(RuntimeError) as cm:
            emb2.embed("hello")
        self.assertIn("Failed to generate sparse embedding", str(cm.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)

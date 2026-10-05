from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import finch


class Bm25EmbeddingTest(unittest.TestCase):
    def test_init_with_built_in_encoder(self):
        with patch("finch.extension.bm25_embedding_function.require_module") as mock_require:
            mock_dashtext = Mock()
            mock_encoder = Mock()
            mock_dashtext.SparseVectorEncoder.default.return_value = mock_encoder
            mock_require.return_value = mock_dashtext

            bm25 = finch.BM25EmbeddingFunction()
            self.assertEqual(bm25.corpus_size, 0)
            self.assertEqual(bm25.encoding_type, "query")
            self.assertEqual(bm25.language, "zh")
            mock_dashtext.SparseVectorEncoder.default.assert_called_once_with(name="zh")

    def test_init_with_custom_encoder(self):
        corpus = ["a", "b", "c"]
        with patch("finch.extension.bm25_embedding_function.require_module") as mock_require:
            mock_dashtext = Mock()
            mock_encoder = Mock()
            mock_dashtext.SparseVectorEncoder.return_value = mock_encoder
            mock_require.return_value = mock_dashtext

            bm25 = finch.BM25EmbeddingFunction(corpus=corpus, b=0.75, k1=1.2)
            self.assertEqual(bm25.corpus_size, 3)
            self.assertEqual(bm25.encoding_type, "query")
            mock_dashtext.SparseVectorEncoder.assert_called_once_with(b=0.75, k1=1.2)
            mock_encoder.train.assert_called_once_with(corpus)

    def test_embed_query_and_document_paths(self):
        with patch("finch.extension.bm25_embedding_function.require_module") as mock_require:
            mock_dashtext = Mock()
            mock_encoder = Mock()
            mock_encoder.encode_queries.return_value = {5: 0.89, 12: 1.45, 1: 0.0, 2: -0.5}
            mock_encoder.encode_documents.return_value = {10: 1.5, 20: 2.3}
            mock_dashtext.SparseVectorEncoder.default.return_value = mock_encoder
            mock_require.return_value = mock_dashtext

            bm25_q = finch.BM25EmbeddingFunction(encoding_type="query")
            bm25_q.embed.cache_clear()
            out = bm25_q.embed("cat purr loud")
            self.assertEqual(out, {5: 0.89, 12: 1.45})
            mock_encoder.encode_queries.assert_called_once_with("cat purr loud")

            bm25_d = finch.BM25EmbeddingFunction(encoding_type="document")
            bm25_d.embed.cache_clear()
            out2 = bm25_d.embed("document text")
            self.assertEqual(out2, {10: 1.5, 20: 2.3})
            mock_encoder.encode_documents.assert_called_once_with("document text")

    def test_properties_extra_params(self):
        corpus = ["doc1", "doc2", "doc3"]
        with patch("finch.extension.bm25_embedding_function.require_module") as mock_require:
            mock_dashtext = Mock()
            mock_encoder = Mock()
            mock_dashtext.SparseVectorEncoder.return_value = mock_encoder
            mock_require.return_value = mock_dashtext

            bm25 = finch.BM25EmbeddingFunction(
                corpus=corpus,
                encoding_type="document",
                language="en",
                b=0.8,
                k1=1.5,
                custom_param="test",
            )
            self.assertEqual(bm25.corpus_size, 3)
            self.assertEqual(bm25.encoding_type, "document")
            self.assertEqual(bm25.language, "en")
            self.assertEqual(bm25.extra_params, {"custom_param": "test"})


from __future__ import annotations

import unittest
from http import HTTPStatus
from unittest.mock import MagicMock, patch


class FinchReferenceQwenSparseEmbeddingMessageParitySubsetTest(unittest.TestCase):
    @patch("finch.extension.qwen_function.require_module")
    def test_filters_positive_values_only(self, mock_require_module) -> None:
        from finch.extension import QwenSparseEmbedding

        mock_dashscope = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = HTTPStatus.OK
        mock_resp.output = {
            "embeddings": [
                {
                    "sparse_embedding": [
                        {"index": 10, "value": 0.5, "token": "a"},
                        {"index": 20, "value": 0.0, "token": "b"},
                        {"index": 30, "value": -1.0, "token": "c"},
                        {"index": 40, "value": 0.25, "token": "d"},
                    ]
                }
            ]
        }
        mock_dashscope.TextEmbedding.call.return_value = mock_resp
        mock_require_module.return_value = mock_dashscope

        emb = QwenSparseEmbedding(dimension=1024, api_key="test_key")
        emb.embed.cache_clear()
        out = emb.embed("hello")
        self.assertEqual(list(out.keys()), [10, 40])
        self.assertTrue(all(v > 0 for v in out.values()))

    @patch("finch.extension.qwen_function.require_module")
    def test_expected_exactly_one_embedding_message(self, mock_require_module) -> None:
        from finch.extension import QwenSparseEmbedding

        mock_dashscope = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = HTTPStatus.OK
        mock_resp.output = {"embeddings": []}
        mock_dashscope.TextEmbedding.call.return_value = mock_resp
        mock_require_module.return_value = mock_dashscope

        emb = QwenSparseEmbedding(dimension=1024, api_key="test_key")
        emb.embed.cache_clear()
        with self.assertRaises(ValueError) as cm:
            emb.embed("hello")
        self.assertIn("Expected exactly 1 embedding", str(cm.exception))

    @patch("finch.extension.qwen_function.require_module")
    def test_sparse_embedding_not_list_message(self, mock_require_module) -> None:
        from finch.extension import QwenSparseEmbedding

        mock_dashscope = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = HTTPStatus.OK
        mock_resp.output = {"embeddings": [{"sparse_embedding": {"index": 10, "value": 0.5}}]}
        mock_dashscope.TextEmbedding.call.return_value = mock_resp
        mock_require_module.return_value = mock_dashscope

        emb = QwenSparseEmbedding(dimension=1024, api_key="test_key")
        emb.embed.cache_clear()
        with self.assertRaises(ValueError) as cm:
            emb.embed("hello")
        self.assertIn("'sparse_embedding' field is missing or not a list", str(cm.exception))

    @patch("finch.extension.qwen_function.require_module")
    def test_sparse_item_missing_index_value_message(self, mock_require_module) -> None:
        from finch.extension import QwenSparseEmbedding

        mock_dashscope = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = HTTPStatus.OK
        mock_resp.output = {
            "embeddings": [{"sparse_embedding": [{"index": 10, "token": "a"}]}]
        }
        mock_dashscope.TextEmbedding.call.return_value = mock_resp
        mock_require_module.return_value = mock_dashscope

        emb = QwenSparseEmbedding(dimension=1024, api_key="test_key")
        emb.embed.cache_clear()
        with self.assertRaises(ValueError) as cm:
            emb.embed("hello")
        self.assertIn("missing 'index' or 'value'", str(cm.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)

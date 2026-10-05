from __future__ import annotations

import os
import unittest
from http import HTTPStatus
from unittest.mock import MagicMock, Mock, patch

import finch


class QwenEmbeddingTest(unittest.TestCase):
    def test_qwen_dense_init_with_api_key(self):
        emb = finch.QwenDenseEmbedding(dimension=128, api_key="test_key")
        self.assertEqual(emb.dimension, 128)
        self.assertEqual(emb.model, "text-embedding-v4")
        self.assertEqual(emb._api_key, "test_key")

    @patch.dict(os.environ, {"DASHSCOPE_API_KEY": ""})
    def test_qwen_dense_init_with_empty_env_key(self):
        with self.assertRaisesRegex(ValueError, "DashScope API key is required"):
            finch.QwenDenseEmbedding(dimension=128)

    @patch("finch.extension.qwen_function.require_module")
    def test_qwen_dense_embed_empty_text(self, mock_require_module):
        mock_require_module.return_value = MagicMock()
        emb = finch.QwenDenseEmbedding(dimension=8, api_key="test_key")
        emb.embed.cache_clear()
        with self.assertRaisesRegex(ValueError, "Input text cannot be empty"):
            emb.embed("")
        with self.assertRaises(TypeError):
            emb.embed(None)  # type: ignore[arg-type]

    @patch("finch.extension.qwen_function.require_module")
    def test_qwen_dense_embed_http_error(self, mock_require_module):
        mock_dashscope = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.BAD_REQUEST
        mock_response.message = "Bad Request"
        mock_dashscope.TextEmbedding.call.return_value = mock_response
        mock_require_module.return_value = mock_dashscope

        emb = finch.QwenDenseEmbedding(dimension=8, api_key="test_key")
        emb.embed.cache_clear()
        with self.assertRaises(ValueError):
            emb.embed("hello")

    @patch("finch.extension.qwen_function.require_module")
    def test_qwen_dense_embed_invalid_response(self, mock_require_module):
        mock_dashscope = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.output = {"embeddings": []}
        mock_dashscope.TextEmbedding.call.return_value = mock_response
        mock_require_module.return_value = mock_dashscope

        emb = finch.QwenDenseEmbedding(dimension=8, api_key="test_key")
        emb.embed.cache_clear()
        with self.assertRaises(ValueError):
            emb.embed("hello")

    @patch("finch.extension.qwen_function.require_module")
    def test_qwen_dense_embed_cached(self, mock_require_module):
        mock_dashscope = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.output = {"embeddings": [{"embedding": [0.0, 0.0, 0.0]}]}
        mock_dashscope.TextEmbedding.call.return_value = mock_response
        mock_require_module.return_value = mock_dashscope

        emb = finch.QwenDenseEmbedding(dimension=3, api_key="test_key")
        emb.embed.cache_clear()
        _ = emb.embed("hello")
        _ = emb.embed("hello")
        mock_dashscope.TextEmbedding.call.assert_called_once()

    def test_qwen_sparse_init_custom_encoding_type(self):
        emb = finch.QwenSparseEmbedding(dimension=1024, encoding_type="document", api_key="test_key")
        self.assertEqual(emb.extra_params.get("encoding_type"), "document")

    @patch("finch.extension.qwen_function.require_module")
    def test_qwen_sparse_embed_http_error(self, mock_require_module):
        mock_dashscope = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.BAD_REQUEST
        mock_response.message = "Bad Request"
        mock_dashscope.TextEmbedding.call.return_value = mock_response
        mock_require_module.return_value = mock_dashscope

        emb = finch.QwenSparseEmbedding(dimension=1024, api_key="test_key")
        emb.embed.cache_clear()
        with self.assertRaises(ValueError):
            emb.embed("hello")

    @patch("finch.extension.openai_function.require_module")
    def test_openai_dense_embed_invalid_response(self, mock_require_module):
        class _Embeddings:
            def create(self, **kwargs):
                return type("_Resp", (), {"data": []})()

        class _Client:
            def __init__(self):
                self.embeddings = _Embeddings()

        openai_mod = MagicMock()
        openai_mod.OpenAI = lambda api_key=None, base_url=None: _Client()
        mock_require_module.return_value = openai_mod

        emb = finch.OpenAIDenseEmbedding(api_key="sk-test", dimension=3)
        emb.embed.cache_clear()
        with self.assertRaises(ValueError):
            emb.embed("hello")

    @patch("finch.extension.openai_function.require_module")
    def test_openai_dense_embed_runtime_error(self, mock_require_module):
        class _Embeddings:
            def create(self, **kwargs):
                raise Exception("boom")

        class _Client:
            def __init__(self):
                self.embeddings = _Embeddings()

        openai_mod = MagicMock()
        openai_mod.OpenAI = lambda api_key=None, base_url=None: _Client()
        mock_require_module.return_value = openai_mod

        emb = finch.OpenAIDenseEmbedding(api_key="sk-test", dimension=3)
        emb.embed.cache_clear()
        # Only APIError and APIConnectionError map to "Failed to call OpenAI API"; other errors are unexpected.
        with self.assertRaisesRegex(RuntimeError, "Unexpected error during API call"):
            emb.embed("hello")

    @patch("finch.extension.openai_function.require_module")
    def test_openai_dense_base_url_is_forwarded(self, mock_require_module):
        client = MagicMock()
        openai_mod = MagicMock()
        openai_mod.OpenAI = MagicMock(return_value=client)
        mock_require_module.return_value = openai_mod

        emb = finch.OpenAIDenseEmbedding(api_key="sk-test", base_url="https://example.invalid", dimension=3)
        emb.embed.cache_clear()

        class _EmbItem:
            def __init__(self):
                self.embedding = [0.0, 0.0, 0.0]

        client.embeddings.create.return_value = type("_Resp", (), {"data": [_EmbItem()]})()
        _ = emb.embed("hello")

        openai_mod.OpenAI.assert_called_once_with(api_key="sk-test", base_url="https://example.invalid")

    @patch("finch.extension.sentence_transformer_function.require_module")
    def test_default_local_dense_embedding_empty_text(self, mock_require_module):
        mock_st = Mock()
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_model.device = "cpu"
        mock_model.encode = Mock(return_value=[0.0, 0.0, 0.0])
        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        emb = finch.DefaultLocalDenseEmbedding()
        with self.assertRaisesRegex(ValueError, "Input text cannot be empty"):
            emb.embed("")
        with self.assertRaises(TypeError):
            emb.embed(None)  # type: ignore[arg-type]

    @patch("finch.extension.sentence_transformer_function.require_module")
    def test_default_local_sparse_embedding_uses_encode_document(self, mock_require_module):
        mock_st = Mock()
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.encode_document = Mock(return_value={1: 1.0, 2: 0.0})
        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        finch.DefaultLocalSparseEmbedding.clear_cache()
        emb = finch.DefaultLocalSparseEmbedding(encoding_type="document")
        out = emb.embed("hello")
        self.assertEqual(out, {1: 1.0})
        mock_model.encode_document.assert_called_once()

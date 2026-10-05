from __future__ import annotations

import os
import unittest
from http import HTTPStatus
from unittest.mock import MagicMock, Mock, patch

import finch


class EmbeddingTest(unittest.TestCase):
    @patch.dict(os.environ, {"DASHSCOPE_API_KEY": "env_key"})
    def test_qwen_dense_init_env_key(self):
        emb = finch.QwenDenseEmbedding(dimension=128)
        self.assertEqual(emb._api_key, "env_key")
        self.assertEqual(emb.extra_params, {})

    def test_qwen_dense_init_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "DashScope API key is required"):
                finch.QwenDenseEmbedding(dimension=128)

    @patch("finch.extension.qwen_function.require_module")
    def test_qwen_dense_embed_success(self, mock_require_module):
        mock_dashscope = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.output = {"embeddings": [{"embedding": [0.1, 0.2, 0.3]}]}
        mock_dashscope.TextEmbedding.call.return_value = mock_response
        mock_require_module.return_value = mock_dashscope

        embedding_func = finch.QwenDenseEmbedding(dimension=3, api_key="test_key")
        embedding_func.embed.cache_clear()
        result = embedding_func.embed("test text")
        self.assertEqual(result, [0.1, 0.2, 0.3])
        self.assertEqual(embedding_func.extra_params, {})
        mock_dashscope.TextEmbedding.call.assert_called_once_with(
            model="text-embedding-v4",
            input="test text",
            dimension=3,
            output_type="dense",
        )

    @patch("finch.extension.qwen_function.require_module")
    def test_qwen_sparse_embed_success(self, mock_require_module):
        mock_dashscope = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.output = {
            "embeddings": [
                {
                    "sparse_embedding": [
                        {"index": 10, "value": 0.5, "token": "a"},
                        {"index": 2, "value": 1.0, "token": "b"},
                    ]
                }
            ]
        }
        mock_dashscope.TextEmbedding.call.return_value = mock_response
        mock_require_module.return_value = mock_dashscope

        embedding_func = finch.QwenSparseEmbedding(dimension=1024, api_key="test_key")
        embedding_func.embed.cache_clear()
        result = embedding_func.embed("test text")
        self.assertEqual(result, {2: 1.0, 10: 0.5})
        self.assertEqual(embedding_func.extra_params.get("encoding_type", "query"), "query")

    @patch("finch.extension.openai_function.require_module")
    def test_openai_dense_embed_success(self, mock_require_module):
        class _EmbItem:
            def __init__(self, embedding):
                self.embedding = embedding

        class _Resp:
            def __init__(self, embedding):
                self.data = [_EmbItem(embedding)]

        class _Embeddings:
            def create(self, **kwargs):
                self.kwargs = kwargs
                return _Resp([0.0, 1.0, 2.0])

        class _Client:
            def __init__(self):
                self.embeddings = _Embeddings()

        openai_mod = MagicMock()
        openai_mod.OpenAI = lambda api_key=None, base_url=None: _Client()
        mock_require_module.return_value = openai_mod

        emb = finch.OpenAIDenseEmbedding(model="text-embedding-3-small", api_key="sk", dimension=3)
        emb.embed.cache_clear()
        out = emb.embed("hello")
        self.assertEqual(out, [0.0, 1.0, 2.0])
        self.assertEqual(emb.extra_params, {})

    @patch("finch.extension.sentence_transformer_function.require_module")
    def test_default_local_dense_embedding_init_and_embed(self, mock_require_module):
        import numpy as np

        mock_st = Mock()
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.device = "cpu"
        fake_embedding = np.random.rand(384).astype(np.float32)
        mock_model.encode = Mock(return_value=fake_embedding)
        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        emb_func = finch.DefaultLocalDenseEmbedding()
        self.assertEqual(emb_func.dimension, 384)
        self.assertEqual(emb_func.model_name, "all-MiniLM-L6-v2")
        self.assertEqual(emb_func.model_source, "huggingface")
        self.assertEqual(emb_func.device, "cpu")

        out = emb_func.embed("Hello, world!")
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 384)
        mock_model.encode.assert_called_once_with(
            "Hello, world!",
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
        )

    def test_default_local_dense_embedding_invalid_model_source(self):
        with self.assertRaisesRegex(ValueError, "Invalid model_source"):
            finch.DefaultLocalDenseEmbedding(model_source="invalid_source")  # type: ignore[arg-type]

    @patch("finch.extension.sentence_transformer_function.require_module")
    def test_default_local_sparse_embedding_cache_controls(self, mock_require_module):
        mock_st = Mock()
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_st.SentenceTransformer.return_value = mock_model
        mock_require_module.return_value = mock_st

        finch.DefaultLocalSparseEmbedding.clear_cache()
        info = finch.DefaultLocalSparseEmbedding.get_cache_info()
        self.assertEqual(info["cached_models"], 0)

        sparse_emb = finch.DefaultLocalSparseEmbedding()
        _ = sparse_emb._get_sparse_model()
        info2 = finch.DefaultLocalSparseEmbedding.get_cache_info()
        self.assertEqual(info2["cached_models"], 1)

        removed = finch.DefaultLocalSparseEmbedding.remove_from_cache(model_source="huggingface", device=None)
        self.assertTrue(removed)
        info3 = finch.DefaultLocalSparseEmbedding.get_cache_info()
        self.assertEqual(info3["cached_models"], 0)

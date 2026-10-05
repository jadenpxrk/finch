import os
import unittest
from http import HTTPStatus
from unittest.mock import MagicMock, patch

import finch


class ExtensionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            finch.init()
        except RuntimeError:
            pass

    def test_qwen_dense_init_env_key(self) -> None:
        from finch.extension import QwenDenseEmbedding

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "env_key"}):
            emb = QwenDenseEmbedding(dimension=128)
            self.assertEqual(emb._api_key, "env_key")

    def test_qwen_dense_init_missing_key(self) -> None:
        from finch.extension import QwenDenseEmbedding

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                QwenDenseEmbedding(dimension=128)

    @patch("finch.extension.qwen_function.require_module")
    def test_qwen_dense_embed_success(self, mock_require_module) -> None:
        from finch.extension import QwenDenseEmbedding

        mock_dashscope = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = HTTPStatus.OK
        mock_resp.output = {"embeddings": [{"embedding": [0.1, 0.2, 0.3]}]}
        mock_dashscope.TextEmbedding.call.return_value = mock_resp
        mock_require_module.return_value = mock_dashscope

        emb = QwenDenseEmbedding(dimension=3, api_key="test_key")
        emb.embed.cache_clear()
        out = emb.embed("test text")
        self.assertEqual(out, [0.1, 0.2, 0.3])

        mock_dashscope.TextEmbedding.call.assert_called_once_with(
            model="text-embedding-v4",
            input="test text",
            dimension=3,
            output_type="dense",
        )

    @patch("finch.extension.openai_function.require_module")
    def test_openai_dense_embed_success_uses_require_module(self, mock_require_module) -> None:
        from finch.extension import OpenAIDenseEmbedding

        class _EmbItem:
            def __init__(self, embedding):
                self.embedding = embedding

        class _Resp:
            def __init__(self, embedding):
                self.data = [_EmbItem(embedding)]

        class _Embeddings:
            def create(self, **_kwargs):
                return _Resp([0.0, 1.0, 2.0])

        class _Client:
            def __init__(self):
                self.embeddings = _Embeddings()

        openai_mod = MagicMock()
        openai_mod.OpenAI = lambda api_key=None, base_url=None: _Client()
        mock_require_module.return_value = openai_mod

        emb = OpenAIDenseEmbedding(model="text-embedding-3-small", api_key="sk", dimension=3)
        emb.embed.cache_clear()
        out = emb.embed("hello")
        self.assertEqual(out, [0.0, 1.0, 2.0])

        mock_require_module.assert_called_once_with("openai")

    @patch("finch.extension.openai_function.require_module")
    def test_openai_invalid_response_message(self, mock_require_module) -> None:
        from finch.extension import OpenAIDenseEmbedding

        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = []

        mock_client.embeddings.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client
        mock_openai.APIError = type("APIError", (Exception,), {})
        mock_openai.APIConnectionError = type("APIConnectionError", (Exception,), {})
        mock_require_module.return_value = mock_openai

        emb = OpenAIDenseEmbedding(api_key="sk-test")
        emb.embed.cache_clear()
        with self.assertRaises(ValueError) as cm:
            emb.embed("test text")
        self.assertIn("no embedding data returned", str(cm.exception))

    @patch("finch.extension.openai_function.require_module")
    def test_openai_api_error_message(self, mock_require_module) -> None:
        from finch.extension import OpenAIDenseEmbedding

        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.APIError = type("APIError", (Exception,), {})
        mock_openai.APIConnectionError = type("APIConnectionError", (Exception,), {})

        mock_client.embeddings.create.side_effect = mock_openai.APIError("Rate limit exceeded")
        mock_openai.OpenAI.return_value = mock_client
        mock_require_module.return_value = mock_openai

        emb = OpenAIDenseEmbedding(api_key="sk-test")
        emb.embed.cache_clear()
        with self.assertRaises(RuntimeError) as cm:
            emb.embed("test text")
        self.assertIn("Failed to call OpenAI API", str(cm.exception))

    @patch("finch.extension.sentence_transformer_function.require_module")
    def test_sentence_transformer_model_loading_error_message(self, mock_require_module) -> None:
        from finch.extension import DefaultLocalDenseEmbedding

        mock_st = MagicMock()
        mock_st.SentenceTransformer.side_effect = Exception("Model not found")
        mock_require_module.return_value = mock_st

        with self.assertRaises(ValueError) as cm:
            DefaultLocalDenseEmbedding()
        self.assertIn("Failed to load Sentence Transformer model", str(cm.exception))

    @patch("finch.extension.sentence_transformer_function.require_module")
    def test_sentence_transformer_modelscope_import_error_message(self, mock_require_module) -> None:
        from finch.extension import DefaultLocalDenseEmbedding

        mock_st = MagicMock()

        def _side_effect(module_name: str):
            if module_name == "sentence_transformers":
                return mock_st
            if module_name == "modelscope":
                raise ImportError("No module named 'modelscope'")
            raise ImportError(f"No module named '{module_name}'")

        mock_require_module.side_effect = _side_effect

        with self.assertRaises(ImportError) as cm:
            DefaultLocalDenseEmbedding(model_source="modelscope")
        self.assertIn("ModelScope support requires the 'modelscope' package", str(cm.exception))

    @patch("finch.extension.bm25_embedding_function.require_module")
    def test_bm25_missing_dashtext_message(self, mock_require_module) -> None:
        from finch.extension import BM25EmbeddingFunction

        mock_require_module.side_effect = ImportError("No module named 'dashtext'")
        with self.assertRaises(ImportError) as cm:
            BM25EmbeddingFunction(language="en")
        self.assertIn("dashtext package is required for BM25EmbeddingFunction", str(cm.exception))

    @patch("finch.extension.qwen_function.require_module")
    def test_qwen_dense_embed_empty_text(self, _mock_require_module) -> None:
        from finch.extension import QwenDenseEmbedding

        emb = QwenDenseEmbedding(dimension=3, api_key="test_key")
        with self.assertRaises(ValueError):
            emb.embed("")

        with self.assertRaises(TypeError):
            emb.embed(None)  # type: ignore[arg-type]

    def test_qwen_rerank_requires_query_and_field(self) -> None:
        from finch.extension import QwenReRanker

        with self.assertRaises(ValueError):
            QwenReRanker(api_key="test_key")  # type: ignore[call-arg]

        with self.assertRaises(ValueError):
            QwenReRanker(query="q", api_key="test_key")  # type: ignore[call-arg]

    @patch("finch.extension.qwen_function.require_module")
    def test_qwen_rerank_success(self, mock_require_module) -> None:
        from finch import Doc
        from finch.extension import QwenReRanker

        mock_dashscope = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = HTTPStatus.OK
        mock_resp.output = {"results": [{"index": 1, "relevance_score": 0.9}]}
        mock_dashscope.TextReRank.call.return_value = mock_resp
        mock_require_module.return_value = mock_dashscope

        rr = QwenReRanker(query="test", api_key="k", rerank_field="content", topn=1)
        out = rr.rerank(
            {"v": [Doc(id="1", fields={"content": "a"}), Doc(id="2", fields={"content": "b"})]}
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].id, "2")
        self.assertAlmostEqual(float(out[0].score), 0.9, places=6)

    @patch("finch.extension.openai_function.require_module")
    def test_openai_dense_embed_success(self, mock_require_module) -> None:
        from finch.extension import OpenAIDenseEmbedding

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

        # openai.OpenAI() returns a client instance.
        openai_mod = MagicMock()
        openai_mod.OpenAI = lambda api_key=None, base_url=None: _Client()
        mock_require_module.return_value = openai_mod

        emb = OpenAIDenseEmbedding(model="text-embedding-3-small", api_key="sk", dimension=3)
        emb.embed.cache_clear()
        out = emb.embed("hello")
        self.assertEqual(out, [0.0, 1.0, 2.0])

    @patch("finch.extension.sentence_transformer_function.require_module")
    def test_default_local_dense_embedding_stubbed(self, mock_require_module) -> None:
        from finch.extension import DefaultLocalDenseEmbedding

        class _ST:
            def __init__(self, *args, **kwargs):
                self.device = kwargs.get("device") or "cpu"

            def get_sentence_embedding_dimension(self):
                return 384

            def encode(self, *args, **kwargs):
                return [0.0] * 384

        st_mod = MagicMock()
        st_mod.SentenceTransformer = _ST
        mock_require_module.return_value = st_mod

        emb = DefaultLocalDenseEmbedding()
        out = emb.embed("hi")
        self.assertEqual(len(out), 384)

    @patch("finch.extension.sentence_transformer_function.require_module")
    def test_default_local_dense_embedding_wraps_runtime_errors(self, mock_require_module) -> None:
        from finch.extension import DefaultLocalDenseEmbedding

        st_mod = MagicMock()
        model = MagicMock()
        model.get_sentence_embedding_dimension.return_value = 384
        model.encode.side_effect = Exception("boom")
        st_mod.SentenceTransformer.return_value = model
        mock_require_module.return_value = st_mod

        emb = DefaultLocalDenseEmbedding()
        with self.assertRaises(RuntimeError) as cm:
            emb.embed("hello")
        self.assertIn("Failed to generate embedding", str(cm.exception))

    @patch("finch.extension.sentence_transformer_rerank_function.require_module")
    def test_default_local_reranker_stubbed(self, mock_require_module) -> None:
        from finch import Doc
        from finch.extension import DefaultLocalReRanker

        class _CE:
            def __init__(self, *args, **kwargs):
                pass

            def predict(self, pairs, **kwargs):
                # Return higher score for second doc.
                return [0.1, 0.9]

        st_mod = MagicMock()
        st_mod.CrossEncoder = _CE
        mock_require_module.return_value = st_mod

        rr = DefaultLocalReRanker(query="q", rerank_field="content", topn=1)
        out = rr.rerank(
            {
                "v": [
                    Doc(id="1", fields={"content": "a"}),
                    Doc(id="2", fields={"content": "b"}),
                ]
            }
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].id, "2")

    def test_default_local_reranker_wraps_predict_errors(self) -> None:
        from finch.extension.sentence_transformer_rerank_function import DefaultLocalReRanker
        from finch import Doc

        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception("Model inference error")

        mock_st = MagicMock()
        mock_st.CrossEncoder.return_value = mock_model

        with patch(
            "finch.extension.sentence_transformer_rerank_function.require_module",
            return_value=mock_st,
        ):
            rr = DefaultLocalReRanker(query="test", rerank_field="content")
            with self.assertRaises(RuntimeError) as cm:
                rr.rerank({"v": [Doc(id="1", fields={"content": "Document 1"})]})
            self.assertIn("Failed to compute rerank scores", str(cm.exception))

    @patch("finch.extension.bm25_embedding_function.require_module")
    def test_bm25_embedding_stubbed(self, mock_require_module) -> None:
        from finch.extension import BM25EmbeddingFunction

        class _Encoder:
            def encode(self, s, mode):
                return {10: 0.5, 2: 1.0}

        class _SVE:
            @staticmethod
            def default(name="zh"):
                return _Encoder()

            def __init__(self, *args, **kwargs):
                self._enc = _Encoder()

            def train(self, corpus):
                return None

        dashtext_mod = MagicMock()
        dashtext_mod.SparseVectorEncoder = _SVE
        mock_require_module.return_value = dashtext_mod

        emb = BM25EmbeddingFunction(language="en")
        out = emb.embed("hello world")
        self.assertEqual(list(out.keys()), [2, 10])


if __name__ == "__main__":
    unittest.main(verbosity=2)

from __future__ import annotations

import os
from typing import ClassVar, Optional

from ..common.constants import TEXT
from ..tool import require_module


class OpenAIFunctionBase:
    _MODEL_DIMENSIONS: ClassVar[dict[str, int]] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = base_url
        if not self._api_key:
            raise ValueError(
                "OpenAI API key is required. Provide api_key= or set OPENAI_API_KEY."
            )

    @property
    def model(self) -> str:
        return self._model

    def _get_client(self):
        openai = require_module("openai")
        if self._base_url:
            return openai.OpenAI(api_key=self._api_key, base_url=self._base_url)
        return openai.OpenAI(api_key=self._api_key)

    def _call_text_embedding_api(self, input: TEXT, dimension: Optional[int] = None) -> list:
        try:
            client = self._get_client()
            params = {"model": self.model, "input": input}
            if dimension is not None:
                params["dimensions"] = dimension
            response = client.embeddings.create(**params)
        except Exception as e:
            # distinguish OpenAI API errors from unexpected ones.
            openai = require_module("openai")
            api_error = getattr(openai, "APIError", None)
            api_conn_error = getattr(openai, "APIConnectionError", None)
            api_error_types = tuple(t for t in (api_error, api_conn_error) if isinstance(t, type))
            if api_error_types and isinstance(e, api_error_types):
                raise RuntimeError(f"Failed to call OpenAI API: {e!s}") from e
            raise RuntimeError(f"Unexpected error during API call: {e!s}") from e

        try:
            data = getattr(response, "data", None)
            if not data:
                raise ValueError("Invalid API response: no embedding data returned")

            embedding_vector = data[0].embedding
            if not isinstance(embedding_vector, list):
                raise ValueError("Invalid API response: embedding is not a list of numbers")
            return embedding_vector
        except ValueError:
            raise
        except (AttributeError, IndexError, TypeError) as e:
            raise ValueError(f"Failed to parse API response: {e!s}") from e

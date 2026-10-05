from __future__ import annotations

import os
from http import HTTPStatus
from typing import Optional

from ..common.constants import TEXT
from ..tool import require_module


class QwenFunctionBase:
    def __init__(self, model: str, api_key: Optional[str] = None):
        self._model = model
        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "DashScope API key is required. Provide api_key= or set DASHSCOPE_API_KEY."
            )

    @property
    def model(self) -> str:
        return self._model

    def _get_connection(self):
        dashscope = require_module("dashscope")
        dashscope.api_key = self._api_key
        return dashscope

    def _call_text_embedding_api(
        self,
        input: TEXT,
        dimension: int,
        output_type: str,
        text_type: Optional[str] = None,
    ) -> dict:
        call_params = {
            "model": self.model,
            "input": input,
            "dimension": dimension,
            "output_type": output_type,
        }
        if text_type is not None:
            call_params["text_type"] = text_type
        try:
            resp = self._get_connection().TextEmbedding.call(**call_params)
        except Exception as e:
            raise RuntimeError(f"Failed to call DashScope API: {e!s}") from e

        if getattr(resp, "status_code", None) != HTTPStatus.OK:
            raise ValueError(
                f"DashScope API error: [Code={getattr(resp, 'code', 'N/A')}, "
                f"Status={getattr(resp, 'status_code', 'N/A')}] {getattr(resp, 'message', 'Unknown error')}"
            )

        output = getattr(resp, "output", None)
        if not isinstance(output, dict):
            raise ValueError("Invalid DashScope response: missing output")
        return output

    def _call_rerank_api(self, query: str, documents: list[str], top_n: int) -> dict:
        try:
            resp = self._get_connection().TextReRank.call(
                model=self.model,
                query=query,
                documents=documents,
                top_n=top_n,
                return_documents=False,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to call DashScope API: {e!s}") from e

        if getattr(resp, "status_code", None) != HTTPStatus.OK:
            raise ValueError(
                f"DashScope API error: [Code={getattr(resp, 'code', 'N/A')}, "
                f"Status={getattr(resp, 'status_code', 'N/A')}] {getattr(resp, 'message', 'Unknown error')}"
            )

        output = getattr(resp, "output", None)
        if not isinstance(output, dict):
            raise ValueError("Invalid DashScope response: missing output")
        return output


from __future__ import annotations

from typing import Literal, Optional

from ..tool import require_module


class SentenceTransformerFunctionBase:
    def __init__(
        self,
        model_name: str,
        model_source: Literal["huggingface", "modelscope"] = "huggingface",
        device: Optional[str] = None,
    ):
        if model_source not in ("huggingface", "modelscope"):
            raise ValueError(
                f"Invalid model_source: '{model_source}'. Must be 'huggingface' or 'modelscope'."
            )
        self._model_name = model_name
        self._model_source = model_source
        self._device = device
        self._model = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_source(self) -> str:
        return self._model_source

    @property
    def device(self) -> str:
        model = self._get_model()
        if model is not None:
            return str(getattr(model, "device", self._device or "cpu"))
        return self._device or "cpu"

    def _get_model(self):
        if self._model is not None:
            return self._model

        try:
            sentence_transformers = require_module("sentence_transformers")

            if self._model_source == "modelscope":
                require_module("modelscope")
                from modelscope.hub.snapshot_download import snapshot_download

                model_dir = snapshot_download(self._model_name)
                self._model = sentence_transformers.SentenceTransformer(
                    model_dir, device=self._device, trust_remote_code=True
                )
            else:
                self._model = sentence_transformers.SentenceTransformer(
                    self._model_name, device=self._device, trust_remote_code=True
                )
            return self._model
        except ImportError as e:
            if "modelscope" in str(e) and self._model_source == "modelscope":
                raise ImportError(
                    "ModelScope support requires the 'modelscope' package. "
                    "Please install it with: pip install modelscope"
                ) from e
            raise
        except Exception as e:
            raise ValueError(
                f"Failed to load Sentence Transformer model '{self._model_name}' "
                f"from {self._model_source}: {e!s}"
            ) from e

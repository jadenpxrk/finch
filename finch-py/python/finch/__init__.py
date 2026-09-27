from __future__ import annotations

from . import model as model
from . import typing as typing
from . import extension as extension
from . import common as common
from . import tool as tool
from .model import schema as schema

# Extensions
from .extension import (
    ReRanker,
    RrfReRanker,
    WeightedReRanker,
    DenseEmbeddingFunction,
    SparseEmbeddingFunction,
    OpenAIFunctionBase,
    QwenFunctionBase,
    SentenceTransformerFunctionBase,
    OpenAIDenseEmbedding,
    QwenDenseEmbedding,
    QwenSparseEmbedding,
    DefaultLocalDenseEmbedding,
    DefaultLocalSparseEmbedding,
    BM25EmbeddingFunction,
    DefaultLocalReRanker,
    QwenReRanker,
)

# Typing
from .typing import (
    DataType,
    IndexType,
    MetricType,
    QuantizeType,
    Status,
    StatusCode,
)
from .typing.enum import LogLevel, LogType

# Core data structures
from .model.collection import Collection
from .model.doc import Doc
from .model.group_result import GroupResult

# Query & index parameters
from .model import param as param
from .model.param import (
    AddColumnOption,
    AlterColumnOption,
    CollectionOption,
    FlatIndexParam,
    HnswIndexParam,
    HnswQueryParam,
    IndexOption,
    InvertIndexParam,
    IvfIndexParam,
    IVFIndexParam,
    IVFQueryParam,
    OptimizeOption,
    QueryParam,
)
from .model.param.vector_query import VectorQuery

# Schema & field definitions
from .model.schema import CollectionSchema, CollectionStats, FieldSchema, VectorSchema
from .memory import (
    AnswerEmission,
    AnswerEvidence,
    ArtifactInput,
    ClaimInput,
    CorrectionInput,
    EpisodeInput,
    FinchMemory,
    GroundedAnswerClaim,
    MemoryScope,
    PackContextRequest,
    SearchRequest,
    SlotAliasInput,
    StateReadSelection,
    TargetRef,
    ValidateAnswerRequest,
)

# tools
from .tool import require_module

# lifecycle
from .finch import create_and_open, init, open

__all__ = [
    # Finch functions
    "create_and_open",
    "init",
    "open",
    # Core classes
    "Collection",
    "Doc",
    "GroupResult",
    # Schema
    "CollectionSchema",
    "FieldSchema",
    "VectorSchema",
    "CollectionStats",
    # Memory
    "FinchMemory",
    "MemoryScope",
    "EpisodeInput",
    "ArtifactInput",
    "ClaimInput",
    "TargetRef",
    "CorrectionInput",
    "SearchRequest",
    "StateReadSelection",
    "PackContextRequest",
    "AnswerEvidence",
    "GroundedAnswerClaim",
    "AnswerEmission",
    "ValidateAnswerRequest",
    "SlotAliasInput",
    # Parameters
    "VectorQuery",
    "InvertIndexParam",
    "HnswIndexParam",
    "FlatIndexParam",
    "IVFIndexParam",
    "IvfIndexParam",
    "CollectionOption",
    "IndexOption",
    "OptimizeOption",
    "AddColumnOption",
    "AlterColumnOption",
    "HnswQueryParam",
    "IVFQueryParam",
    "QueryParam",
    # Typing
    "DataType",
    "MetricType",
    "QuantizeType",
    "IndexType",
    "LogLevel",
    "LogType",
    "Status",
    "StatusCode",
    # Tools
    "require_module",
    # Modules
    "model",
    "param",
    "schema",
    "typing",
    "extension",
    "common",
    "tool",
    # Extensions
    "DenseEmbeddingFunction",
    "SparseEmbeddingFunction",
    "QwenFunctionBase",
    "OpenAIFunctionBase",
    "SentenceTransformerFunctionBase",
    "ReRanker",
    "DefaultLocalDenseEmbedding",
    "DefaultLocalSparseEmbedding",
    "BM25EmbeddingFunction",
    "OpenAIDenseEmbedding",
    "QwenDenseEmbedding",
    "QwenSparseEmbedding",
    "RrfReRanker",
    "WeightedReRanker",
    "DefaultLocalReRanker",
    "QwenReRanker",
]

# Version handling
__version__: str
try:
    from importlib.metadata import version
except Exception:  # pragma: no cover
    version = None  # type: ignore[assignment]

try:
    __version__ = "unknown" if version is None else version("finch")
except Exception:  # pragma: no cover
    __version__ = "unknown"

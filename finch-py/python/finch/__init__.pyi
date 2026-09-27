from __future__ import annotations

from typing import Any, Optional

from .typing import DataType, IndexType, MetricType, QuantizeType, Status, StatusCode
from .typing.enum import LogLevel, LogType
from .model.collection import Collection
from .model.doc import Doc
from .model.group_result import GroupResult
from .model.schema import CollectionSchema, FieldSchema, VectorSchema, CollectionStats
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
from .model.param import (
    AddColumnOption,
    AlterColumnOption,
    CollectionOption,
    FlatIndexParam,
    HnswIndexParam,
    IVFIndexParam,
    IndexOption,
    InvertIndexParam,
    OptimizeOption,
    QueryParam,
    HnswQueryParam,
    IVFQueryParam,
)
from .model.param.vector_query import VectorQuery
from . import common
from . import tool
from .tool import require_module
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

def init(
    *,
    log_type: Optional[LogType] = ...,
    log_level: Optional[LogLevel] = ...,
    log_dir: Optional[str] = ...,
    log_basename: Optional[str] = ...,
    log_file_size: Optional[int] = ...,
    log_overdue_days: Optional[int] = ...,
    query_threads: Optional[int] = ...,
    optimize_threads: Optional[int] = ...,
    invert_to_forward_scan_ratio: Optional[float] = ...,
    brute_force_by_keys_ratio: Optional[float] = ...,
    memory_limit_mb: Optional[int] = ...,
    wal_flush_every_docs: Optional[int] = ...,
    wal_fsync_every_docs: Optional[int] = ...,
) -> None: ...

def create_and_open(path: str, schema: CollectionSchema, option: Optional[CollectionOption] = ...) -> Collection: ...
def open(path: str, option: CollectionOption) -> Collection: ...

__version__: str
__all__: list[str]

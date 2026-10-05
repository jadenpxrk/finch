from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from .model.param import CollectionOption

Json = Union[None, bool, int, float, str, List["Json"], Dict[str, "Json"]]


def _now_ms() -> int:
    return int(time.time() * 1000)


def _ms(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("timestamp must not be bool")
    if isinstance(value, int):
        return value
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return int(value.timestamp() * 1000)
    if isinstance(value, str):
        text = value[:-1] + "+00:00" if value.endswith("Z") else value
        return int(datetime.fromisoformat(text).timestamp() * 1000)
    raise TypeError("timestamp must be datetime, RFC3339 string, epoch ms, or None")


def _raw(value: object) -> Dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("expected dataclass or dict input")


def _json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"))


def _scope(value: Union["MemoryScope", Mapping[str, Any]]) -> Dict[str, Any]:
    raw = _raw(value)
    space = raw.pop("space", None) or raw.pop("space_id", None)
    if not space:
        raise ValueError("scope.space is required")
    return {
        "space_id": space,
        "tenant_id": raw.get("tenant_id"),
        "user_id": raw.get("user_id"),
        "agent_id": raw.get("agent_id"),
        "project_id": raw.get("project_id"),
        "thread_id": raw.get("thread_id"),
    }


def _parse(raw: str) -> Any:
    return json.loads(raw)


def _state_selections(values: Sequence[object]) -> List[Dict[str, str]]:
    selections = []
    for value in values:
        raw = _raw(value)
        slot_id = raw.get("slot_id")
        if not slot_id:
            raise ValueError("state selection slot_id is required")
        selections.append({"slot_id": str(slot_id), "view": str(raw.get("view", "current"))})
    return selections


def _native_memory_store() -> Any:
    from . import _finch

    store_type = getattr(_finch, "MemoryStore", None) or getattr(_finch, "PyMemoryStore", None)
    if store_type is None:
        raise RuntimeError("finch memory bindings are not available in this build")
    return store_type


@dataclass(frozen=True)
class MemoryScope:
    space: str
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    project_id: Optional[str] = None
    thread_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeInput:
    scope: Union[MemoryScope, Mapping[str, Any]]
    text: str
    source_kind: str = "user_message"
    actor: str = "user"
    event_time: object = None
    metadata: Dict[str, Json] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactInput:
    scope: Union[MemoryScope, Mapping[str, Any]]
    text: Optional[str] = None
    uri: Optional[str] = None
    title: Optional[str] = None
    mime_type: Optional[str] = None
    event_time: object = None
    metadata: Dict[str, Json] = field(default_factory=dict)


@dataclass(frozen=True)
class ClaimInput:
    scope: Union[MemoryScope, Mapping[str, Any]]
    text: str
    source_ids: Sequence[str] = field(default_factory=list)
    valid_from: object = None
    valid_to: object = None
    metadata: Dict[str, Json] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetRef:
    type: str
    id: str


@dataclass(frozen=True)
class CorrectionInput:
    scope: Union[MemoryScope, Mapping[str, Any]]
    operation: str
    target: Optional[Union[TargetRef, Mapping[str, Any]]] = None
    selector: Optional[Union[str, Mapping[str, Json]]] = None
    text: Optional[str] = None
    reason: Optional[str] = None
    effective_at: object = None
    metadata: Dict[str, Json] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchRequest:
    scope: Union[MemoryScope, Mapping[str, Any]]
    query: str
    top_k: int = 10
    mode: str = "hybrid"
    at: object = None
    query_embedding: Sequence[float] = field(default_factory=list)
    include_neighbors: bool = False


@dataclass(frozen=True)
class StateReadSelection:
    slot_id: str
    view: str = "current"


@dataclass(frozen=True)
class PackContextRequest:
    scope: Union[MemoryScope, Mapping[str, Any]]
    query: str
    max_tokens: int
    mode: str = "hybrid"
    at: object = None
    query_embedding: Sequence[float] = field(default_factory=list)
    include_provenance: bool = True
    complete_evidence: bool = True
    preferred_slot_ids: Sequence[str] = field(default_factory=list)
    state_selections: Sequence[Union[StateReadSelection, Mapping[str, Any]]] = field(
        default_factory=list
    )


@dataclass(frozen=True)
class AnswerEvidence:
    span_id: str
    quote: str


@dataclass(frozen=True)
class GroundedAnswerClaim:
    text: str
    evidence: Sequence[Union[AnswerEvidence, Mapping[str, Any]]]
    slot_id: Optional[str] = None


@dataclass(frozen=True)
class AnswerEmission:
    disposition: str
    claims: Sequence[Union[GroundedAnswerClaim, Mapping[str, Any]]] = field(default_factory=list)


@dataclass(frozen=True)
class ValidateAnswerRequest:
    scope: Union[MemoryScope, Mapping[str, Any]]
    emission: Union[AnswerEmission, Mapping[str, Any]]
    context: Mapping[str, Any]


@dataclass(frozen=True)
class SlotAliasInput:
    scope: Union[MemoryScope, Mapping[str, Any]]
    alias_subject: str
    alias_predicate: str
    source_claim_ids: Sequence[str]
    canonical_slot_id: Optional[str] = None
    target_claim_id: Optional[str] = None
    id: Optional[str] = None
    visibility: str = "private"
    policy_tags: Sequence[str] = field(default_factory=list)
    valid_from: object = None
    valid_to: object = None


class _EvalApi:
    def __init__(self, memory: "FinchMemory") -> None:
        self._memory = memory

    def run_retrieval(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return self.compare_search_modes(payload)

    def compare_search_modes(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return self._memory.compare_search_modes(payload)


class _RawApi:
    def __init__(self, store: Any) -> None:
        self._store = store

    def call(self, operation: str, payload: Mapping[str, Any]) -> Any:
        method = getattr(self._store, operation)
        args = payload.get("args", [])
        kwargs = payload.get("kwargs", {})
        return _parse(method(*args, **kwargs))


class FinchMemory:
    def __init__(self, store: Any) -> None:
        self._store = store
        self.raw = _RawApi(store)
        self.experimental = self.raw
        self.evals = _EvalApi(self)

    @classmethod
    def create(cls, path: str, embedding_dim: int, options: Optional[CollectionOption] = None) -> "FinchMemory":
        store_type = _native_memory_store()
        return cls(store_type.create(path, embedding_dim, options))

    @classmethod
    def open(cls, path: str, options: Optional[CollectionOption] = None) -> "FinchMemory":
        store_type = _native_memory_store()
        return cls(store_type.open(path, options))

    def apply_state_mutation_batch(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        return _parse(self._store.apply_state_mutation_batch_json(_json(dict(batch))))

    def project_answer_ready_state(
        self,
        *,
        projection_seed: str,
        scope: Union[MemoryScope, Mapping[str, Any]],
        query: str,
        evidence_hits: Sequence[Mapping[str, Any]] = (),
        claim_limit: int = 256,
        entity_limit: int = 256,
        valid_at: object = None,
        transaction_at: object = None,
        state_selections: Sequence[Union[StateReadSelection, Mapping[str, Any]]] = (),
    ) -> Dict[str, Any]:
        temporal = {
            "valid_at_ms": _ms(valid_at),
            "transaction_at_ms": _ms(transaction_at),
        }
        result = self._store.project_answer_ready_state_json(_json({
            "projection_seed": projection_seed,
            "scope": _scope(scope),
            "query_text": query,
            "evidence_hits": list(evidence_hits),
            "selections": _state_selections(state_selections),
            "claim_limit": claim_limit,
            "entity_limit": entity_limit,
            "temporal": temporal,
        }))
        return _parse(result)

    def ingest_episode(self, input: Union[EpisodeInput, Mapping[str, Any]]) -> Dict[str, Any]:
        raw = _raw(input)
        at_ms = _ms(raw.get("event_time")) or _now_ms()
        payload = {
            "id": raw.get("id"),
            "scope": _scope(raw["scope"]),
            "visibility": raw.get("visibility", "private"),
            "policy_tags": raw.get("policy_tags", []),
            "source_kind": raw.get("source_kind", "user_message"),
            "actor": raw.get("actor", "user"),
            "sequence_no": raw.get("sequence_no", at_ms),
            "event_time_ms": at_ms,
            "valid_from_ms": _ms(raw.get("valid_from")),
            "valid_to_ms": _ms(raw.get("valid_to")),
            "raw_text": raw["text"],
            "blob_ref": raw.get("blob_ref"),
            "mime_type": raw.get("mime_type", "text/plain"),
            "causal_parent_ids": raw.get("causal_parent_ids", []),
            "metadata_json": _json(raw.get("metadata", {})),
        }
        return _parse(self._store.ingest_episode_json(_json(payload), at_ms, None))

    def ingest_artifact(self, input: Union[ArtifactInput, Mapping[str, Any]]) -> Dict[str, Any]:
        raw = _raw(input)
        text = raw.get("text") or ""
        at_ms = _ms(raw.get("event_time")) or _now_ms()
        title = raw.get("title") or raw.get("uri") or "artifact"
        payload = {
            "id": raw.get("id") or "artifact_" + hashlib.sha256((title + text).encode()).hexdigest()[:16],
            "scope": _scope(raw["scope"]),
            "status": "active",
            "visibility": raw.get("visibility", "private"),
            "policy_tags": raw.get("policy_tags", []),
            "artifact_kind": raw.get("artifact_kind") or ("url" if raw.get("uri") else "document"),
            "title": title,
            "uri": raw.get("uri"),
            "blob_ref": raw.get("blob_ref"),
            "mime_type": raw.get("mime_type"),
            "content_hash": hashlib.sha256(text.encode()).hexdigest(),
            "source_created_at_ms": _ms(raw.get("source_created_at")),
            "source_modified_at_ms": _ms(raw.get("source_modified_at")),
            "created_at_ms": at_ms,
            "ingested_at_ms": at_ms,
            "valid_from_ms": _ms(raw.get("valid_from")),
            "valid_to_ms": _ms(raw.get("valid_to")),
            "extracted_text_ref": raw.get("extracted_text_ref"),
            "metadata_json": _json(raw.get("metadata", {})),
        }
        return _parse(self._store.ingest_artifact_text_json(_json(payload), text, None))

    def add_claim(self, input: Union[ClaimInput, Mapping[str, Any]]) -> Dict[str, Any]:
        raw = _raw(input)
        now = _now_ms()
        payload = {
            "id": raw.get("id"),
            "scope": _scope(raw["scope"]),
            "visibility": raw.get("visibility", "private"),
            "policy_tags": raw.get("policy_tags", []),
            "claim_text": raw["text"],
            "subject": raw.get("subject"),
            "predicate": raw.get("predicate"),
            "object_value": raw.get("object_value"),
            "claim_kind": raw.get("claim_kind", "fact"),
            "polarity": raw.get("polarity", "affirmative"),
            "source_span_ids": raw.get("source_span_ids", raw.get("source_ids", [])),
            "source_episode_ids": raw.get("source_episode_ids", []),
            "asserted_by": raw.get("asserted_by", "user"),
            "confidence": raw.get("confidence"),
            "observed_at_ms": _ms(raw.get("observed_at")) or now,
            "valid_from_ms": _ms(raw.get("valid_from")),
            "valid_to_ms": _ms(raw.get("valid_to")),
        }
        return _parse(self._store.add_manual_claim_json(_json(payload), raw.get("embedding")))

    def list_claims(self, input: Mapping[str, Any]) -> Dict[str, Any]:
        scope = _json(_scope(input["scope"]))
        claims = _parse(self._store.scan_current_claims_json(scope, int(input.get("limit", 1000)), _ms(input.get("at"))))
        return {"claims": claims}

    def get_claim(self, id: str, scope: Union[MemoryScope, Mapping[str, Any]], limit: int = 1000) -> Optional[Dict[str, Any]]:
        for claim in _parse(self._store.scan_claims_json(_json(_scope(scope)), limit, None)):
            if claim.get("id") == id:
                return claim
        return None

    def add_correction(self, input: Union[CorrectionInput, Mapping[str, Any]]) -> Dict[str, Any]:
        raw = _raw(input)
        target = _raw(raw["target"]) if raw.get("target") is not None else {}
        selector = raw.get("selector")
        payload = {
            "id": raw.get("id"),
            "scope": _scope(raw["scope"]),
            "visibility": raw.get("visibility", "private"),
            "policy_tags": raw.get("policy_tags", []),
            "operation": raw["operation"],
            "target_type": target.get("type", raw.get("target_type", "span")),
            "target_ids": [target["id"]] if target.get("id") else raw.get("target_ids", []),
            "target_selector": selector if isinstance(selector, str) else (_json(selector) if selector else None),
            "new_value": raw.get("text"),
            "reason": raw.get("reason"),
            "actor": raw.get("actor", "user"),
            "authority": raw.get("authority", "user"),
            "effective_at_ms": _ms(raw.get("effective_at")),
            "applies_valid_from_ms": _ms(raw.get("applies_valid_from")),
            "applies_valid_to_ms": _ms(raw.get("applies_valid_to")),
            "cascade_policy": raw.get("cascade_policy"),
            "metadata_json": _json(raw.get("metadata", {})),
        }
        return _parse(self._store.add_correction_json(_json(payload), _now_ms()))

    def tombstone(self, input: Union[CorrectionInput, Mapping[str, Any]]) -> Dict[str, Any]:
        raw = _raw(input)
        raw["operation"] = "tombstone"
        return self.add_correction(raw)

    def restore(self, input: Union[CorrectionInput, Mapping[str, Any]]) -> Dict[str, Any]:
        raw = _raw(input)
        raw["operation"] = "restore"
        return self.add_correction(raw)

    def add_slot_alias(self, input: Union[SlotAliasInput, Mapping[str, Any]]) -> Dict[str, Any]:
        raw = _raw(input)
        payload = {
            "id": raw.get("id"),
            "scope": _scope(raw["scope"]),
            "visibility": raw.get("visibility", "private"),
            "policy_tags": raw.get("policy_tags", []),
            "alias_subject": raw["alias_subject"],
            "alias_predicate": raw["alias_predicate"],
            "canonical_slot_id": raw.get("canonical_slot_id"),
            "target_claim_id": raw.get("target_claim_id"),
            "source_claim_ids": list(raw["source_claim_ids"]),
            "valid_from_ms": _ms(raw.get("valid_from")),
            "valid_to_ms": _ms(raw.get("valid_to")),
        }
        return _parse(self._store.add_slot_alias_json(_json(payload), _now_ms()))

    def search(self, input: Union[SearchRequest, Mapping[str, Any]]) -> Dict[str, Any]:
        raw = _raw(input)
        scope = _json(_scope(raw["scope"]))
        k = int(raw.get("top_k", 10))
        at_ms = _ms(raw.get("at"))
        mode = raw.get("mode", "hybrid")
        embedding = list(raw.get("query_embedding", []))
        query = raw["query"]
        if mode == "keyword":
            hits = self._store.keyword_search_spans_json(scope, query, k, 1000, at_ms)
        elif mode == "vector":
            hits = self._store.query_spans_json(scope, embedding, k, at_ms)
        else:
            hits = self._store.hybrid_search_spans_json(scope, embedding, query, k, 1000, at_ms)
        items = _parse(hits)
        if raw.get("include_neighbors"):
            items = _parse(self._store.complete_span_evidence_json(scope, _json(items), 1, 1, 1000, at_ms))
        return {"hits": items}

    def pack_context(self, input: Union[PackContextRequest, Mapping[str, Any]]) -> Dict[str, Any]:
        raw = _raw(input)
        search = self.search({
            "scope": raw["scope"],
            "query": raw["query"],
            "top_k": raw.get("top_k", 10),
            "mode": raw.get("mode", "hybrid"),
            "at": raw.get("at"),
            "query_embedding": raw.get("query_embedding", []),
            "include_neighbors": raw.get("complete_evidence", True),
        })
        preferred_slot_ids = list(raw.get("preferred_slot_ids", []))
        state_selections = _state_selections(raw.get("state_selections", []))
        if preferred_slot_ids and state_selections:
            raise ValueError("state_selections and preferred_slot_ids are mutually exclusive")
        if preferred_slot_ids:
            state_selections = [{"slot_id": str(slot_id)} for slot_id in preferred_slot_ids]
        packed = self._store.build_answer_ready_context_json(_json({
            "scope": _scope(raw["scope"]),
            "query_text": raw["query"],
            "hits": search["hits"],
            "selections": state_selections,
            "profile_limit": int(raw.get("profile_limit", 10)),
            "claim_limit": int(raw.get("claim_limit", 20)),
            "artifact_limit": int(raw.get("artifact_limit", 10)),
            "at_ms": _ms(raw.get("at")),
            "token_budget": int(raw["max_tokens"]),
            "include_provenance": bool(raw.get("include_provenance", True)),
        }))
        return _parse(packed)

    def validate_answer(
        self,
        input: Union[ValidateAnswerRequest, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        raw = _raw(input)
        emission = raw["emission"]
        if is_dataclass(emission):
            emission = asdict(emission)
        scope_json = _json(_scope(raw["scope"]))
        if raw.get("context") is None:
            raise ValueError("validate_answer requires the packed context returned by pack_context")
        return _parse(self._store.validate_answer_context_json(
            scope_json,
            _json(raw["context"]),
            _json(emission),
        ))

    def finalize_answer(
        self,
        input: Union[ValidateAnswerRequest, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        raw = _raw(input)
        emission = raw["emission"]
        if is_dataclass(emission):
            emission = asdict(emission)
        scope_json = _json(_scope(raw["scope"]))
        if raw.get("context") is None:
            raise ValueError("finalize_answer requires the packed context returned by pack_context")
        return _parse(self._store.finalize_answer_context_json(
            scope_json,
            _json(raw["context"]),
            _json(emission),
        ))

    def compare_search_modes(self, input: Mapping[str, Any]) -> Dict[str, Any]:
        scope = _json(_scope(input["scope"]))
        query = str(input["query"])
        k = int(input.get("top_k", 10))
        at_ms = _ms(input.get("at"))
        embedding = list(input.get("query_embedding", []))
        vector = self._store.query_spans_json(scope, embedding, k, at_ms)
        keyword = self._store.keyword_search_spans_json(scope, query, k, 1000, at_ms)
        hybrid = self._store.hybrid_search_spans_json(scope, embedding, query, k, 1000, at_ms)
        report = self._store.evaluate_retrieval_baseline_json(_json({
            "vector_hits": _parse(vector),
            "keyword_hits": _parse(keyword),
            "hybrid_hits": _parse(hybrid),
            "relevant_span_ids": list(input.get("relevant_span_ids", [])),
            "required_exact_tokens": list(input.get("required_exact_tokens", [])),
            "stale_or_inadmissible_span_ids": list(input.get("stale_or_inadmissible_span_ids", [])),
            "k": k,
        }))
        return _parse(report)

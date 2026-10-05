from __future__ import annotations

from typing import Any

from .._finch import PyDoc as _CoreDoc
from .doc import Doc
from .schema import CollectionSchema

__all__ = [
    "convert_to_core_doc",
    "convert_to_py_doc",
    "convert_to_py_doc_sql",
]


def _maybe_numpy_to_list(v: Any) -> Any:
    if hasattr(v, "tolist"):
        try:
            return v.tolist()
        except Exception:
            return v
    return v


def convert_to_core_doc(doc: Doc, schema: CollectionSchema) -> _CoreDoc:
    # Do not coerce doc IDs to strings (e.g. `None` should not become `"None"`).
    # Invalid IDs will fail later with a validate error.
    pk = doc.id if isinstance(doc.id, str) else ""
    core = _CoreDoc(pk, float(doc.score or 0.0))

    # Scalar fields.
    for k, v in (doc.fields or {}).items():
        v = _maybe_numpy_to_list(v)
        fs = schema._core.get_field(k)  # type: ignore[attr-defined]
        if fs is None:
            raise ValueError(f"schema validate failed: {k} not found in collection schema")
        core.set_any(k, fs, v)

    # Vector fields.
    for k, v in (doc.vectors or {}).items():
        v = _maybe_numpy_to_list(v)
        fs = schema._core.get_field(k)  # type: ignore[attr-defined]
        if fs is None:
            raise ValueError(f"schema validate failed: {k} not found in collection schema")
        core.set_any(k, fs, v)

    return core


def convert_to_py_doc(core_doc: _CoreDoc, schema: CollectionSchema) -> Doc:
    fields = {}
    vectors = {}
    raw = core_doc.fields_dict()
    for name, value in raw.items():
        vs = schema.vector(name)
        if vs is not None:
            vectors[name] = value
        elif schema.field(name) is not None:
            fields[name] = value
        else:
            # Preserve system/internal columns even though they are not part of
            # the user schema.
            if name in ("_finch_uid_", "_finch_row_id_", "_finch_g_doc_id_", "_finch_score"):
                fields[name] = value
                continue
            # Dropped/unknown columns are not surfaced through docs.
            continue
    pk = core_doc.pk() if hasattr(core_doc, "pk") and callable(core_doc.pk) else getattr(core_doc, "pk")  # type: ignore[truthy-function]
    score = core_doc.score() if hasattr(core_doc, "score") and callable(core_doc.score) else getattr(core_doc, "score")  # type: ignore[truthy-function]
    return Doc(id=str(pk), score=float(score), vectors=vectors, fields=fields)


def convert_to_py_doc_sql(core_doc: _CoreDoc, schema: CollectionSchema) -> Doc:
    """
    Convert a core doc for SQL SELECT results.

    Unlike `convert_to_py_doc`, SQL SELECT can project aliases and virtual/system
    columns that are not present in the collection schema; those must be
    preserved.
    """
    fields: dict[str, Any] = {}
    vectors: dict[str, Any] = {}
    raw = core_doc.fields_dict()
    for name, value in raw.items():
        vs = schema.vector(name)
        if vs is not None:
            vectors[name] = value
        else:
            fields[name] = value
    pk = core_doc.pk() if hasattr(core_doc, "pk") and callable(core_doc.pk) else getattr(core_doc, "pk")  # type: ignore[truthy-function]
    score = core_doc.score() if hasattr(core_doc, "score") and callable(core_doc.score) else getattr(core_doc, "score")  # type: ignore[truthy-function]
    return Doc(id=str(pk), score=float(score), vectors=vectors, fields=fields)

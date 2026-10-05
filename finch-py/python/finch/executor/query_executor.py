from __future__ import annotations

import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, final

from .._finch import PyVectorQuery as _CoreVectorQuery
from ..model.convert import convert_to_py_doc
from ..model.doc import Doc
from ..model.param.vector_query import VectorQuery
from ..model.schema import CollectionSchema
from ..typing import DataType
from ..extension import ReRanker, RrfReRanker, WeightedReRanker

__all__ = [
    "QueryContext",
    "QueryExecutor",
    "QueryExecutorFactory",
    "NoVectorQueryExecutor",
    "SingleVectorQueryExecutor",
    "MultiVectorQueryExecutor",
    "VectorQuery",
]


def _convert_to_list(v):
    if hasattr(v, "tolist"):
        return v.tolist()
    return v


class QueryContext:
    def __init__(
        self,
        topk: int,
        filter: Optional[str] = None,
        include_vector: bool = False,
        queries: Optional[list[VectorQuery]] = None,
        output_fields: Optional[list[str]] = None,
        reranker: Optional[ReRanker] = None,
    ):
        self._filter = filter
        self._queries = queries or []
        self._topk = int(topk)
        self._include_vector = bool(include_vector)
        self._output_fields = output_fields
        self._reranker = reranker
        self._core_vectors: list[_CoreVectorQuery] = []

    @property
    def topk(self) -> int:
        return self._topk

    @property
    def queries(self) -> list[VectorQuery]:
        return self._queries

    @property
    def filter(self) -> Optional[str]:
        return self._filter

    @property
    def reranker(self) -> Optional[ReRanker]:
        return self._reranker

    @property
    def output_fields(self) -> Optional[list[str]]:
        return self._output_fields

    @property
    def include_vector(self) -> bool:
        return self._include_vector

    @property
    def core_vectors(self) -> list[_CoreVectorQuery]:
        return self._core_vectors

    @core_vectors.setter
    def core_vectors(self, core_vectors: list[_CoreVectorQuery]):
        self._core_vectors = core_vectors


class QueryExecutor(ABC):
    def __init__(self, schema: CollectionSchema):
        self._schema = schema
        self._concurrency = max(1, int(os.getenv("FINCH_QUERY_CONCURRENCY", "1")))

    @abstractmethod
    def _do_validate(self, ctx: QueryContext) -> None:
        raise NotImplementedError

    @abstractmethod
    def _do_build(self, ctx: QueryContext, collection) -> list[_CoreVectorQuery]:
        raise NotImplementedError

    def _build_query_wo_vector(self, ctx: QueryContext) -> _CoreVectorQuery:
        q = _CoreVectorQuery("", [], ctx.topk, filter=ctx.filter)
        q.set_include_vector(ctx.include_vector)
        if ctx.output_fields is not None:
            q.set_output_fields(ctx.output_fields)
        return q

    def _build_query_with_vector(self, ctx: QueryContext, query: VectorQuery, collection) -> _CoreVectorQuery:
        vector_schema = self._schema.vector(query.field_name) if query else None
        if vector_schema is None:
            raise ValueError(f"No vector field found for '{query.field_name}'")

        vec_data = None
        if query.has_vector():
            vec_data = query.vector
        else:
            fetched = collection.fetch([query.id])
            if query.id not in fetched:
                raise ValueError(f"Doc '{query.id}' not found")
            vec_data = fetched[query.id].vectors.get(vector_schema.name)

        dt = vector_schema.data_type
        if dt in (DataType.SparseFp16, DataType.SparseFp32):
            indices = []
            values = []
            if isinstance(vec_data, dict):
                for k, v in vec_data.items():
                    indices.append(int(k))
                    values.append(float(v))
            else:
                raise TypeError("Sparse vector query expects dict[int, float]")
            pairs = sorted(zip(indices, values), key=lambda x: x[0])
            indices = [i for i, _ in pairs]
            values = [v for _, v in pairs]
            core = _CoreVectorQuery(query.field_name, [], ctx.topk, filter=ctx.filter)
            core.set_sparse(indices, values)
        elif dt in (DataType.VectorBinary32, DataType.VectorBinary64):
            if dt == DataType.VectorBinary32:
                core = _CoreVectorQuery.binary32(query.field_name, [int(x) for x in _convert_to_list(vec_data)], ctx.topk, filter=ctx.filter)
            else:
                core = _CoreVectorQuery.binary64(query.field_name, [int(x) for x in _convert_to_list(vec_data)], ctx.topk, filter=ctx.filter)
        else:
            core = _CoreVectorQuery(query.field_name, [float(x) for x in _convert_to_list(vec_data)], ctx.topk, filter=ctx.filter)

        core.set_include_vector(ctx.include_vector)
        if ctx.output_fields is not None:
            core.set_output_fields(ctx.output_fields)

        if query.param is not None:
            core.set_query_param(query.param)
        return core

    @staticmethod
    def _apply_output_fields_order(doc: Doc, output_fields: list[str]) -> None:
        if not output_fields:
            return
        if not doc.fields:
            return
        ordered: dict[str, object] = {}
        for name in output_fields:
            if name in doc.fields and name not in ordered:
                ordered[name] = doc.fields[name]
        for name, value in doc.fields.items():
            if name not in ordered:
                ordered[name] = value
        doc.fields = ordered

    def _do_execute(self, ctx: QueryContext, vectors: list[_CoreVectorQuery], collection) -> dict[str, list[Doc]]:
        if not vectors:
            raise ValueError("No query to execute")
        if len(vectors) == 1 or self._concurrency == 1:
            results: dict[str, list[Doc]] = {}
            for q in vectors:
                docs = collection._obj.query(q)
                py_docs = [convert_to_py_doc(d, self._schema) for d in docs]
                if ctx.output_fields is not None:
                    for doc in py_docs:
                        self._apply_output_fields_order(doc, ctx.output_fields)
                results[q.field_name] = py_docs
            return results

        results: dict[str, list[Doc]] = {}
        with ThreadPoolExecutor(max_workers=self._concurrency) as executor:
            future_to_query = {
                executor.submit(collection._obj.query, q): q.field_name for q in vectors
            }
            for future in as_completed(future_to_query):
                field_name = future_to_query[future]
                docs = future.result()
                py_docs = [convert_to_py_doc(d, self._schema) for d in docs]
                if ctx.output_fields is not None:
                    for doc in py_docs:
                        self._apply_output_fields_order(doc, ctx.output_fields)
                results[field_name] = py_docs
        return results

    def _do_merge_rerank_results(self, ctx: QueryContext, docs_map: dict[str, list[Doc]]) -> list[Doc]:
        if not docs_map:
            raise ValueError("Query results is none and does not to rerank")
        if len(docs_map) == 1:
            if not ctx.reranker or isinstance(ctx.reranker, (RrfReRanker, WeightedReRanker)):
                return next(iter(docs_map.values()))
            return ctx.reranker.rerank(docs_map)
        return ctx.reranker.rerank(docs_map)

    @final
    def execute(self, ctx: QueryContext, collection) -> list[Doc]:
        vectors = self.build_core_queries(ctx, collection)
        docs_map = self._do_execute(ctx, vectors, collection)
        return self._do_merge_rerank_results(ctx, docs_map)

    @final
    def build_core_queries(self, ctx: QueryContext, collection) -> list[_CoreVectorQuery]:
        self._do_validate(ctx)
        vectors = self._do_build(ctx, collection)
        ctx.core_vectors = vectors
        return vectors


class NoVectorQueryExecutor(QueryExecutor):
    def _do_validate(self, ctx: QueryContext) -> None:
        if len(ctx.queries) > 0:
            raise ValueError("Collection does not support query with vector or id")

    def _do_build(self, ctx: QueryContext, _collection) -> list[_CoreVectorQuery]:
        return [self._build_query_wo_vector(ctx)]


class SingleVectorQueryExecutor(NoVectorQueryExecutor):
    def _do_validate(self, ctx: QueryContext) -> None:
        if len(ctx.queries) > 1:
            raise ValueError("Collection has only one vector field, cannot query with multiple vectors")
        for q in ctx.queries:
            q._validate()

    def _do_build(self, ctx: QueryContext, collection) -> list[_CoreVectorQuery]:
        if len(ctx.queries) == 0:
            return [self._build_query_wo_vector(ctx)]
        return [self._build_query_with_vector(ctx, q, collection) for q in ctx.queries]


class MultiVectorQueryExecutor(SingleVectorQueryExecutor):
    def _do_validate(self, ctx: QueryContext) -> None:
        if len(ctx.queries) > 1 and ctx.reranker is None:
            raise ValueError("Reranker is required for multi-vector query")
        seen_fields = set()
        for q in ctx.queries:
            q._validate()
            if q.field_name in seen_fields:
                raise ValueError(f"Query field name '{q.field_name}' appears more than once")
            seen_fields.add(q.field_name)


class QueryExecutorFactory:
    @staticmethod
    def create(schema: CollectionSchema) -> QueryExecutor:
        vectors = schema.vectors
        if len(vectors) == 0:
            return NoVectorQueryExecutor(schema)
        if len(vectors) == 1:
            return SingleVectorQueryExecutor(schema)
        return MultiVectorQueryExecutor(schema)

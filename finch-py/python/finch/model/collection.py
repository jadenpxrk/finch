from __future__ import annotations

from typing import Optional, Union

from .._finch import PyCollection as _CoreCollection
from .._finch import (
    PyFlatIndexParam as _CoreFlatIndexParam,
    PyHnswIndexParam as _CoreHnswIndexParam,
    PyInvertIndexParam as _CoreInvertIndexParam,
    PyIvfIndexParam as _CoreIvfIndexParam,
)
from ..extension import ReRanker
from ..executor import QueryContext, QueryExecutorFactory
from ..typing import DataType, Status
from ..typing import StatusCode
from .convert import convert_to_core_doc, convert_to_py_doc, convert_to_py_doc_sql
from .doc import Doc
from .group_result import GroupResult
from .param import (
    AddColumnOption,
    AlterColumnOption,
    FlatIndexParam,
    HnswIndexParam,
    IndexOption,
    InvertIndexParam,
    IVFIndexParam,
    OptimizeOption,
)
from .param.vector_query import VectorQuery
from .schema import CollectionSchema, CollectionStats, FieldSchema

__all__ = ["Collection"]


def _write(call, items: list, is_single: bool) -> list[Status]:
    """Runs a batch write; a collection-level error raises, except for a single item."""
    try:
        return call(items)
    except Exception as e:
        # A single-item write reports the error as that item's status.
        code = getattr(e, "code", None)
        if not is_single or code is None:
            raise
        return [Status(code, str(e))]


class Collection:
    def __init__(self, obj: _CoreCollection):
        self._obj = obj
        self._schema: CollectionSchema | None = None
        self._querier = None

    @classmethod
    def _from_core(cls, core_collection: _CoreCollection) -> "Collection":
        if not core_collection:
            raise ValueError("Collection is None")
        inst = cls.__new__(cls)
        inst._obj = core_collection
        schema = CollectionSchema._from_core(core_collection.schema())
        inst._schema = schema
        inst._querier = QueryExecutorFactory.create(schema)
        return inst

    @property
    def path(self) -> str:
        return self._obj.path()

    @property
    def option(self):
        return self._obj.options()

    @property
    def schema(self) -> CollectionSchema:
        assert self._schema is not None
        return self._schema

    @property
    def stats(self) -> CollectionStats:
        return self._obj.stats()

    def destroy(self) -> None:
        self._obj.destroy()
        # after destroy(), the collection handle is unusable and
        # accessors like `schema` should fail (core object is destroyed).
        self._schema = None
        self._querier = None

    def close(self) -> None:
        self._obj.close()
        self._schema = None
        self._querier = None

    def flush(self) -> None:
        self._obj.flush()

    def create_index(
        self,
        field_name: str,
        index_param: Union[HnswIndexParam, IVFIndexParam, FlatIndexParam, InvertIndexParam],
        option: IndexOption = IndexOption(),
    ) -> None:
        # wrong-typed `field_name` should look like a binding
        # signature mismatch.
        if not isinstance(field_name, str):
            raise TypeError("incompatible function arguments")

        existing_field = self.schema._core.get_field(field_name)  # type: ignore[attr-defined]
        existing_index = existing_field.index_type() if existing_field else None
        # create_index has no explicit rebuild flag; treat it as
        # "replace existing index" by default.
        rebuild = existing_index is not None

        vector_schema = self.schema.vector(field_name)
        is_sparse_vector = vector_schema is not None and vector_schema.data_type in (
            DataType.SparseFp16,
            DataType.SparseFp32,
            DataType.SPARSE_VECTOR_FP16,
            DataType.SPARSE_VECTOR_FP32,
        )

        if isinstance(index_param, _CoreHnswIndexParam):
            if is_sparse_vector:
                self.create_hnsw_sparse_index(field_name, index_param, rebuild=rebuild, option=option)
            else:
                self.create_hnsw_index(field_name, index_param, rebuild=rebuild, option=option)
        elif isinstance(index_param, _CoreIvfIndexParam):
            self.create_ivf_index(field_name, index_param, rebuild=rebuild, option=option)
        elif isinstance(index_param, _CoreFlatIndexParam):
            if is_sparse_vector:
                self.create_flat_sparse_index(field_name, index_param, rebuild=rebuild, option=option)
            else:
                self.create_flat_index(field_name, index_param, rebuild=rebuild, option=option)
        elif isinstance(index_param, _CoreInvertIndexParam):
            self.create_invert_index(field_name, index_param, rebuild=rebuild, option=option)
        else:  # pragma: no cover
            raise TypeError("Unsupported index_param type")
        self._schema = CollectionSchema._from_core(self._obj.schema())

    def create_hnsw_index(
        self,
        field_name: str,
        params: HnswIndexParam,
        *,
        rebuild: bool = False,
        concurrency: Optional[int] = None,
        option: Optional[IndexOption] = None,
    ) -> None:
        self._obj.create_hnsw_index(
            field_name,
            params,
            rebuild=bool(rebuild),
            concurrency=concurrency,
            option=option,
        )
        self._schema = CollectionSchema._from_core(self._obj.schema())

    def create_ivf_index(
        self,
        field_name: str,
        params: IVFIndexParam,
        *,
        rebuild: bool = False,
        concurrency: Optional[int] = None,
        option: Optional[IndexOption] = None,
    ) -> None:
        self._obj.create_ivf_index(
            field_name,
            params,
            rebuild=bool(rebuild),
            concurrency=concurrency,
            option=option,
        )
        self._schema = CollectionSchema._from_core(self._obj.schema())

    def create_hnsw_sparse_index(
        self,
        field_name: str,
        params: HnswIndexParam,
        *,
        rebuild: bool = False,
        concurrency: Optional[int] = None,
        option: Optional[IndexOption] = None,
    ) -> None:
        self._obj.create_hnsw_sparse_index(
            field_name,
            params,
            rebuild=bool(rebuild),
            concurrency=concurrency,
            option=option,
        )
        self._schema = CollectionSchema._from_core(self._obj.schema())

    def create_flat_index(
        self,
        field_name: str,
        params: FlatIndexParam,
        *,
        rebuild: bool = False,
        concurrency: Optional[int] = None,
        option: Optional[IndexOption] = None,
    ) -> None:
        self._obj.create_flat_index(
            field_name,
            params,
            rebuild=bool(rebuild),
            concurrency=concurrency,
            option=option,
        )
        self._schema = CollectionSchema._from_core(self._obj.schema())

    def create_flat_sparse_index(
        self,
        field_name: str,
        params: FlatIndexParam,
        *,
        rebuild: bool = False,
        concurrency: Optional[int] = None,
        option: Optional[IndexOption] = None,
    ) -> None:
        self._obj.create_flat_sparse_index(
            field_name,
            params,
            rebuild=bool(rebuild),
            concurrency=concurrency,
            option=option,
        )
        self._schema = CollectionSchema._from_core(self._obj.schema())

    def create_invert_index(
        self,
        field_name: str,
        params: InvertIndexParam,
        *,
        rebuild: bool = False,
        concurrency: Optional[int] = None,
        option: Optional[IndexOption] = None,
    ) -> None:
        self._obj.create_invert_index(
            field_name,
            params,
            rebuild=bool(rebuild),
            concurrency=concurrency,
            option=option,
        )
        self._schema = CollectionSchema._from_core(self._obj.schema())

    def drop_index(self, field_name: str) -> None:
        if not isinstance(field_name, str):
            raise TypeError("incompatible function arguments")
        self._obj.drop_index(field_name)
        self._schema = CollectionSchema._from_core(self._obj.schema())

    def optimize(self, option: OptimizeOption = OptimizeOption()) -> None:
        self._obj.optimize(option=option)

    def add_column(
        self,
        field_schema: FieldSchema,
        expression: str = "",
        option: AddColumnOption = AddColumnOption(),
    ) -> None:
        self._obj.add_column(field_schema._get_object(), option=option, expression=expression)
        self._schema = CollectionSchema._from_core(self._obj.schema())

    def drop_column(self, field_name: str) -> None:
        self._obj.drop_column(field_name)
        self._schema = CollectionSchema._from_core(self._obj.schema())

    def alter_column(
        self,
        old_name: str,
        new_name: Optional[str] = None,
        field_schema: Optional[FieldSchema] = None,
        option: AlterColumnOption = AlterColumnOption(),
    ) -> None:
        if field_schema is not None and new_name is not None and new_name != field_schema.name:
            raise ValueError("new_name must match field_schema.name when both are provided")
        self._obj.alter_column(
            old_name,
            rename_to=new_name,
            field_schema=field_schema._get_object() if field_schema else None,
            option=option,
        )
        self._schema = CollectionSchema._from_core(self._obj.schema())

    def insert(self, docs: Union[Doc, list[Doc]]) -> Union[Status, list[Status]]:
        is_single = isinstance(docs, Doc)
        doc_list = [docs] if is_single else docs
        core_docs = [convert_to_core_doc(doc, self.schema) for doc in doc_list]
        results = _write(self._obj.insert, core_docs, is_single)
        if is_single and results and not results[0].ok() and results[0].code() == StatusCode.INVALID_ARGUMENT:
            raise ValueError(results[0].message())
        return results[0] if is_single else results

    def upsert(self, docs: Union[Doc, list[Doc]]) -> Union[Status, list[Status]]:
        required_scalars = [f.name for f in self.schema.fields if not f.nullable]
        required_vectors = [v.name for v in self.schema.vectors if not v.nullable]

        def _is_full_doc(doc: Doc) -> bool:
            fields = doc.fields or {}
            vectors = doc.vectors or {}
            for name in required_scalars:
                if name not in fields:
                    return False
            for name in required_vectors:
                if name not in vectors:
                    return False
            return True

        is_single = isinstance(docs, Doc)
        doc_list = [docs] if is_single else docs

        # treat upsert as "insert-or-update".
        # - If a doc is missing required (non-nullable) fields/vectors, interpret it as a patch update.
        # - For non-existent docs, patch updates will return NOT_FOUND.
        full_idxs = [i for i, d in enumerate(doc_list) if _is_full_doc(d)]
        full_set = set(full_idxs)
        patch_idxs = [i for i in range(len(doc_list)) if i not in full_set]

        results: list[Status] = [Status.OK() for _ in doc_list]  # placeholder

        if full_idxs:
            full_docs = [doc_list[i] for i in full_idxs]
            full_core = [convert_to_core_doc(d, self.schema) for d in full_docs]
            full_res = _write(self._obj.upsert, full_core, is_single)
            for j, i in enumerate(full_idxs):
                results[i] = full_res[j]

        if patch_idxs:
            patch_docs = [doc_list[i] for i in patch_idxs]
            patch_core = [convert_to_core_doc(d, self.schema) for d in patch_docs]
            patch_res = _write(self._obj.update, patch_core, is_single)
            for j, i in enumerate(patch_idxs):
                results[i] = patch_res[j]

        if is_single and results and not results[0].ok() and results[0].code() == StatusCode.INVALID_ARGUMENT:
            raise ValueError(results[0].message())
        return results[0] if is_single else results

    def update(self, docs: Union[Doc, list[Doc]]) -> Union[Status, list[Status]]:
        is_single = isinstance(docs, Doc)
        doc_list = [docs] if is_single else docs
        core_docs = [convert_to_core_doc(doc, self.schema) for doc in doc_list]
        results = _write(self._obj.update, core_docs, is_single)
        if is_single and results and not results[0].ok() and results[0].code() == StatusCode.INVALID_ARGUMENT:
            raise ValueError(results[0].message())
        return results[0] if is_single else results

    def delete(self, ids: Union[str, list[str]]) -> Union[Status, list[Status]]:
        is_single = isinstance(ids, str)
        id_list = [ids] if is_single else ids
        results = _write(self._obj.delete, id_list, is_single)
        return results[0] if is_single else results

    def delete_by_filter(self, filter: str) -> None:
        s = self._obj.delete_by_filter(filter)
        if not s.is_ok():
            raise RuntimeError(str(s))
        return None

    def fetch(self, ids: Union[str, list[str]]) -> dict[str, Doc]:
        id_list = [ids] if isinstance(ids, str) else ids
        docs = self._obj.fetch(id_list)
        out: dict[str, Doc] = {}
        for doc_id, core_doc in docs.items():
            out[doc_id] = convert_to_py_doc(core_doc, self.schema)
        return out

    def query(
        self,
        vectors: Optional[Union[VectorQuery, list[VectorQuery]]] = None,
        *,
        topk: int = 10,
        filter: Optional[str] = None,
        include_vector: bool = False,
        output_fields: Optional[list[str]] = None,
        reranker: Optional[ReRanker] = None,
    ) -> list[Doc]:
        ctx = QueryContext(
            topk=topk,
            filter=filter,
            queries=[vectors] if isinstance(vectors, VectorQuery) else (vectors or []),
            include_vector=include_vector,
            output_fields=output_fields,
            reranker=reranker,
        )
        return self._querier.execute(ctx, self)

    def query_ids(
        self,
        vectors: Optional[Union[VectorQuery, list[VectorQuery]]] = None,
        *,
        topk: int = 10,
        filter: Optional[str] = None,
    ) -> list[int]:
        ctx = QueryContext(
            topk=topk,
            filter=filter,
            queries=[vectors] if isinstance(vectors, VectorQuery) else (vectors or []),
            include_vector=False,
            output_fields=[],
        )
        core_vectors = self._querier.build_core_queries(ctx, self)
        if len(core_vectors) != 1:
            raise ValueError("query_ids requires exactly one vector query")
        return self._obj.query_ids(core_vectors[0])

    def query_sql(self, sql: str) -> list[Doc]:
        """Execute a SQL SELECT statement.

        Finch supports a strict subset:
        `SELECT <fields|*> FROM <table> [WHERE ...] [ORDER BY ...] [LIMIT n]`
        """
        if not isinstance(sql, str):
            raise TypeError("sql must be str")
        if not hasattr(self._obj, "query_sql"):
            raise RuntimeError("query_sql is not available in this build")
        core_docs = self._obj.query_sql(sql)
        return [convert_to_py_doc_sql(d, self.schema) for d in core_docs]

    def group_by_query(
        self,
        vector: VectorQuery,
        *,
        group_by_field: str,
        group_count: int = 2,
        group_topk: int = 3,
        topk: int = 10,
        filter: str | None = None,
        include_vector: bool = False,
        output_fields: list[str] | None = None,
    ) -> list[GroupResult]:
        """
        Execute a group-by vector query.

        - `group_topk` controls the number of groups to return (top groups)
        - `group_count` controls the max docs per group
        """
        if not isinstance(vector, VectorQuery):
            raise TypeError("vector must be VectorQuery")
        if not isinstance(group_by_field, str) or not group_by_field:
            raise TypeError("group_by_field must be non-empty str")

        ctx = QueryContext(
            topk=topk,
            filter=filter,
            include_vector=include_vector,
            queries=[vector],
            output_fields=output_fields,
            reranker=None,
        )
        # Reuse the existing schema-aware builder to resolve vectors-by-id,
        # dtype conversions (dense/binary/sparse), and per-query params.
        self._querier._do_validate(ctx)  # type: ignore[attr-defined]
        core_queries = self._querier._do_build(ctx, self)  # type: ignore[attr-defined]
        if len(core_queries) != 1:
            raise RuntimeError("group_by_query expects exactly one built query")
        core_q = core_queries[0]

        core_groups = self._obj.group_by_query(
            core_q,
            group_by_field,
            int(group_count),
            int(group_topk),
        )
        return [
            GroupResult(
                group_value=g.group_value,
                docs=[convert_to_py_doc(d, self.schema) for d in g.docs],
            )
            for g in core_groups
        ]

from __future__ import annotations

import shutil
import tempfile
import unittest

import finch


def _make_schema(vector_name: str, vector_type: finch.DataType) -> finch.CollectionSchema:
    dim = 8 if vector_type not in (finch.DataType.SPARSE_VECTOR_FP16, finch.DataType.SPARSE_VECTOR_FP32) else 0
    return finch.CollectionSchema(
        name="vector_index_ddl",
        fields=[
            finch.FieldSchema(
                "id",
                finch.DataType.INT64,
                nullable=False,
                index_param=finch.InvertIndexParam(enable_range_optimization=True),
            )
        ],
        vectors=[
            finch.VectorSchema(
                vector_name,
                vector_type,
                dim,
                index_param=None,
            )
        ],
    )


def _doc(i: int, vector_name: str, vector_type: finch.DataType) -> finch.Doc:
    if vector_type in (finch.DataType.SPARSE_VECTOR_FP16, finch.DataType.SPARSE_VECTOR_FP32):
        vec = {1: float(i + 1), 2: float(i + 2), 3: float(i + 3)}
    elif vector_type == finch.DataType.VECTOR_INT8:
        vec = [int(i)] * 8
    else:
        vec = [float(i)] * 8
    return finch.Doc(
        id=str(i),
        fields={"id": i},
        vectors={vector_name: vec},
    )


class FinchReferenceCollectionVectorIndexDdlParitySubsetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            finch.init()
        except RuntimeError:
            pass

    def test_vector_index_operation_subset(self) -> None:
        # Mirror reference support_helper.py: SUPPORT_VECTOR_DATA_TYPE_INDEX_MAP
        cases: list[tuple[finch.DataType, finch.IndexType]] = [
            (finch.DataType.VECTOR_FP16, finch.IndexType.FLAT),
            (finch.DataType.VECTOR_FP16, finch.IndexType.HNSW),
            (finch.DataType.VECTOR_FP16, finch.IndexType.IVF),
            (finch.DataType.VECTOR_FP32, finch.IndexType.FLAT),
            (finch.DataType.VECTOR_FP32, finch.IndexType.HNSW),
            (finch.DataType.VECTOR_FP32, finch.IndexType.IVF),
            (finch.DataType.VECTOR_INT8, finch.IndexType.FLAT),
            (finch.DataType.VECTOR_INT8, finch.IndexType.HNSW),
            (finch.DataType.SPARSE_VECTOR_FP32, finch.IndexType.FLAT),
            (finch.DataType.SPARSE_VECTOR_FP32, finch.IndexType.HNSW),
            (finch.DataType.SPARSE_VECTOR_FP16, finch.IndexType.FLAT),
            (finch.DataType.SPARSE_VECTOR_FP16, finch.IndexType.HNSW),
        ]

        for vector_type, index_type in cases:
            with self.subTest(vector_type=vector_type.name, index_type=index_type.name):
                root = tempfile.mkdtemp(prefix="finch_parity_vec_index_")
                path = f"{root}/collection"
                try:
                    vector_name = "v"
                    schema = _make_schema(vector_name, vector_type)
                    col = finch.create_and_open(path, schema, finch.CollectionOption(read_only=False, enable_mmap=True))

                    res = col.insert([_doc(i, vector_name, vector_type) for i in range(5)])
                    self.assertTrue(all(s.is_ok() for s in res))
                    self.assertEqual(col.stats.doc_count, 5)

                    if index_type == finch.IndexType.FLAT:
                        index_param = finch.FlatIndexParam()
                    elif index_type == finch.IndexType.HNSW:
                        index_param = finch.HnswIndexParam()
                    elif index_type == finch.IndexType.IVF:
                        index_param = finch.IvfIndexParam()
                    else:  # pragma: no cover
                        raise AssertionError(f"unsupported index type: {index_type}")

                    col.create_index(vector_name, index_param, finch.IndexOption())

                    res2 = col.insert([_doc(i, vector_name, vector_type) for i in range(5, 8)])
                    self.assertTrue(all(s.is_ok() for s in res2))
                    self.assertEqual(col.stats.doc_count, 8)
                    fetched = col.fetch([str(i) for i in range(5, 8)])
                    self.assertEqual(len(fetched), 3)

                    col.drop_index(vector_name)

                    res3 = col.insert([_doc(i, vector_name, vector_type) for i in range(8, 10)])
                    self.assertTrue(all(s.is_ok() for s in res3))
                    self.assertEqual(col.stats.doc_count, 10)
                    fetched2 = col.fetch([str(i) for i in range(8, 10)])
                    self.assertEqual(len(fetched2), 2)
                finally:
                    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)


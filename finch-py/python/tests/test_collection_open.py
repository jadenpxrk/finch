from __future__ import annotations

import gc
import shutil
import tempfile
import threading
import unittest

import finch


def _schema_for_open_test() -> finch.CollectionSchema:
    schema = finch.CollectionSchema(
        name="test_collection",
        fields=[
            finch.FieldSchema(
                "id",
                finch.DataType.INT64,
                nullable=False,
                index_param=finch.InvertIndexParam(enable_range_optimization=True),
            ),
            finch.FieldSchema(
                "name",
                finch.DataType.STRING,
                nullable=False,
                index_param=finch.InvertIndexParam(),
            ),
            finch.FieldSchema(
                "weight",
                finch.DataType.FLOAT,
                nullable=False,
                index_param=finch.InvertIndexParam(),
            ),
        ],
        vectors=[
            finch.VectorSchema(
                "dense",
                finch.DataType.VECTOR_FP32,
                16,
                index_param=finch.HnswIndexParam(),
            ),
            finch.VectorSchema(
                "sparse",
                finch.DataType.SPARSE_VECTOR_FP32,
                0,
                index_param=finch.HnswIndexParam(),
            ),
        ],
    )
    return schema


class CollectionOpenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            finch.init()
        except RuntimeError:
            pass

    def test_open_persists_data_and_schema(self) -> None:
        try:
            import numpy as np
        except Exception:  # pragma: no cover
            self.skipTest("numpy is not available")

        root = tempfile.mkdtemp(prefix="finch_test_open_")
        path = f"{root}/test_collection"
        try:
            schema = _schema_for_open_test()
            option = finch.CollectionOption(read_only=False, enable_mmap=True)

            created = finch.create_and_open(path=path, schema=schema, option=option)
            self.assertEqual(created.path, path)
            self.assertEqual(created.schema.name, schema.name)
            self.assertEqual(list(created.schema.fields), list(schema.fields))
            self.assertEqual(list(created.schema.vectors), list(schema.vectors))
            self.assertEqual(bool(created.option.read_only), bool(option.read_only))
            self.assertEqual(bool(created.option.enable_mmap), bool(option.enable_mmap))

            docs = []
            for i in range(3):
                docs.append(
                    finch.Doc(
                        id=str(i),
                        fields={"id": i, "name": f"test_{i}", "weight": float(i * 10)},
                        vectors={
                            "dense": np.arange(16, dtype=np.float32).tolist(),
                            "sparse": {j: float(j + i) for j in range(5)},
                        },
                    )
                )
            res = created.insert(docs)
            self.assertEqual(len(res), 3)
            self.assertTrue(all(s.is_ok() for s in res))
            created.flush()
            self.assertEqual(created.stats.doc_count, 3)

            created.close()
            del created
            gc.collect()

            opened = finch.open(path=path, option=option)
            self.assertEqual(opened.path, path)
            self.assertEqual(opened.schema.name, schema.name)
            self.assertEqual(list(opened.schema.fields), list(schema.fields))
            self.assertEqual(list(opened.schema.vectors), list(schema.vectors))

            fetched = opened.fetch(["0", "1", "2"])
            self.assertEqual(set(fetched.keys()), {"0", "1", "2"})
            for i in range(3):
                d = fetched[str(i)]
                self.assertEqual(d.id, str(i))
                self.assertEqual(d.field("id"), i)
                self.assertEqual(d.field("name"), f"test_{i}")
                self.assertEqual(d.field("weight"), float(i * 10))
                self.assertIsInstance(d.vector("dense"), list)
                self.assertIsInstance(d.vector("sparse"), dict)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_open_invalid_paths_raise(self) -> None:
        opt = finch.CollectionOption(read_only=False, enable_mmap=True)
        invalid = [
            "/nonexistent/directory/test_collection",
            "invalid:path",
            "",
        ]
        for p in invalid:
            with self.subTest(path=p):
                with self.assertRaises(Exception):
                    finch.open(path=p, option=opt)

    def test_read_only_open_rejects_writes(self) -> None:
        root = tempfile.mkdtemp(prefix="finch_test_open_ro_")
        path = f"{root}/test_collection"
        try:
            schema = _schema_for_open_test()
            rw = finch.CollectionOption(read_only=False, enable_mmap=True)
            ro = finch.CollectionOption(read_only=True, enable_mmap=True)

            col = finch.create_and_open(path=path, schema=schema, option=rw)
            col.flush()
            col.close()
            del col
            gc.collect()

            opened = finch.open(path=path, option=ro)

            # DML on a read-only collection returns a non-OK status instead of raising.
            d = finch.Doc(id="1", fields={"id": 1, "name": "x", "weight": 1.0}, vectors={"dense": [0.0] * 16})
            st = opened.insert(d)
            self.assertFalse(st.ok())
            self.assertEqual(int(st.code()), int(finch.StatusCode.PERMISSION_DENIED))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_open_persists_enable_mmap_from_create(self) -> None:
        root = tempfile.mkdtemp(prefix="finch_test_open_mmap_")
        try:
            schema = _schema_for_open_test()

            for created_enable_mmap in (True, False):
                path = f"{root}/test_collection_{int(created_enable_mmap)}"
                created_opt = finch.CollectionOption(
                    read_only=False, enable_mmap=created_enable_mmap
                )
                created = finch.create_and_open(path=path, schema=schema, option=created_opt)
                created.flush()
                created.close()
                del created
                gc.collect()

                for read_only in (False, True):
                    # Open ignores enable_mmap and returns the value stored at create time.
                    open_opt = finch.CollectionOption(
                        read_only=read_only, enable_mmap=(not created_enable_mmap)
                    )
                    opened = finch.open(path=path, option=open_opt)
                    self.assertEqual(bool(opened.option.read_only), bool(read_only))
                    self.assertEqual(bool(opened.option.enable_mmap), bool(created_enable_mmap))
                    del opened
                    gc.collect()
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_open_with_none_option_raises_binding_error(self) -> None:
        root = tempfile.mkdtemp(prefix="finch_test_open_none_opt_")
        path = f"{root}/test_collection"
        try:
            schema = _schema_for_open_test()
            rw = finch.CollectionOption(read_only=False, enable_mmap=True)
            col = finch.create_and_open(path=path, schema=schema, option=rw)
            col.flush()
            col.close()
            del col
            gc.collect()

            with self.assertRaises(Exception) as ctx:
                finch.open(path=path, option=None)  # type: ignore[arg-type]
            self.assertIn("incompatible function arguments", str(ctx.exception))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_open_concurrent_same_path_exclusive_lock(self) -> None:
        root = tempfile.mkdtemp(prefix="finch_test_open_concurrent_")
        path = f"{root}/test_collection"
        try:
            schema = _schema_for_open_test()
            opt = finch.CollectionOption(read_only=False, enable_mmap=True)
            created = finch.create_and_open(path=path, schema=schema, option=opt)
            created.flush()
            created.close()
            del created
            gc.collect()

            barrier = threading.Barrier(5)
            lock = threading.Lock()
            opened: list[finch.Collection] = []
            errors: list[str] = []

            def worker() -> None:
                try:
                    barrier.wait(timeout=5.0)
                    c = finch.open(path=path, option=opt)
                    with lock:
                        opened.append(c)
                except Exception as e:
                    with lock:
                        errors.append(str(e))

            threads = [threading.Thread(target=worker) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(len(opened), 1)
            self.assertEqual(len(errors), 4)

            # Release the lock-holding collection before cleanup.
            opened.clear()
            gc.collect()
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)

import gc
import random
import shutil
import tempfile
import threading
import time
import unittest

import finch
from finch._finch import PyDoc


def _schema() -> finch.CollectionSchema:
    return finch.CollectionSchema(
        name="gil_test",
        fields=[],
        vectors=[
            finch.VectorSchema(
                "v",
                data_type=finch.DataType.VECTOR_FP32,
                dimension=16,
                index_param=finch.HnswIndexParam(),
            )
        ],
    )


def _docs(rnd: random.Random, start: int, n: int) -> list:
    return [
        finch.Doc(id=str(start + i), vectors={"v": [rnd.random() for _ in range(16)]})
        for i in range(n)
    ]


class BindingGilAndErrorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp(prefix="finch_bindings_")
        self.path = f"{self.root}/c"
        self.rw = finch.CollectionOption(read_only=False, enable_mmap=True)

    def tearDown(self) -> None:
        gc.collect()
        shutil.rmtree(self.root, ignore_errors=True)

    def test_optimize_lets_other_threads_run(self) -> None:
        col = finch.create_and_open(path=self.path, schema=_schema(), option=self.rw)
        rnd = random.Random(0)
        for batch in range(2):
            col.insert(_docs(rnd, batch * 150, 150))
            col.flush()

        ticks: list[float] = []
        stop = threading.Event()

        def count() -> None:
            while not stop.is_set():
                ticks.append(time.monotonic())
                time.sleep(0.001)

        counter = threading.Thread(target=count)
        counter.start()
        try:
            start = time.monotonic()
            col.optimize()
            end = time.monotonic()
        finally:
            stop.set()
            counter.join()

        # Holding the GIL allows ticks only within one switch interval (5 ms) of either edge.
        margin = 0.02
        self.assertGreater(end - start, 4 * margin, "optimize finished too fast to observe")
        inside = [t for t in ticks if start + margin < t < end - margin]
        self.assertTrue(inside, f"no ticks during a {end - start:.3f}s optimize")

    def test_batch_write_to_read_only_collection_raises_with_status_code(self) -> None:
        col = finch.create_and_open(path=self.path, schema=_schema(), option=self.rw)
        col.flush()
        col.close()
        del col
        gc.collect()

        ro = finch.open(path=self.path, option=finch.CollectionOption(read_only=True, enable_mmap=True))
        docs = _docs(random.Random(1), 0, 2)
        for write in (ro.insert, ro.upsert, ro.update):
            with self.assertRaises(PermissionError) as ctx:
                write(docs)
            self.assertEqual(ctx.exception.code, finch.StatusCode.PERMISSION_DENIED)
        with self.assertRaises(PermissionError):
            ro.delete(["0", "1"])

    def test_untyped_set_field_infers_exact_types(self) -> None:
        doc = PyDoc("1", 0.0)

        doc.set_field("mixed", [1, 0.1])
        self.assertEqual(doc.get_field("mixed"), [1.0, 0.1])

        for bad in ([], [0.5, 2**53 + 1], ["a", 1], [True, 1]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                doc.set_field("bad", bad)


if __name__ == "__main__":
    unittest.main()

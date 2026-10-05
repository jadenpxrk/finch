from __future__ import annotations

import pickle
import unittest

import finch


class PickleParamTest(unittest.TestCase):
    def test_query_param_pickle_roundtrip(self):
        p = finch.QueryParam(ef=10, n_probe=2, radius=1.0, is_linear=True, use_refiner=True, refiner_k=20)
        blob = pickle.dumps(p)
        p2 = pickle.loads(blob)
        self.assertEqual(p2.ef, 10)
        self.assertEqual(p2.n_probe, 2)
        self.assertEqual(p2.radius, 1.0)
        self.assertTrue(p2.is_linear)
        self.assertTrue(p2.is_using_refiner)

    def test_collection_option_pickle_roundtrip(self):
        opt = finch.CollectionOption(read_only=False, enable_mmap=True, max_buffer_size=123)
        blob = pickle.dumps(opt)
        opt2 = pickle.loads(blob)
        self.assertFalse(opt2.read_only)
        self.assertTrue(opt2.enable_mmap)
        self.assertEqual(opt2.max_buffer_size, 123)

    def test_option_objects_pickle_roundtrip(self):
        for obj in (
            finch.IndexOption(concurrency=3),
            finch.OptimizeOption(concurrency=4),
            finch.AddColumnOption(concurrency=5),
            finch.AlterColumnOption(concurrency=6),
        ):
            blob = pickle.dumps(obj)
            obj2 = pickle.loads(blob)
            self.assertEqual(obj2.concurrency, obj.concurrency)


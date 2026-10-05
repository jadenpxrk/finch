from __future__ import annotations

import unittest

import finch


class ParamTest(unittest.TestCase):
    def test_core_param_submodule_aliases(self):
        from finch._finch import _Doc  # type: ignore[attr-defined]
        from finch._finch.param import _VectorQuery  # type: ignore[attr-defined]

        d = _Doc()
        self.assertIsNotNone(d)

        v = _VectorQuery()
        self.assertIsNotNone(v)
        self.assertEqual(int(v.topk), 10)

    def test_invert_index_param(self):
        param = finch.InvertIndexParam()
        self.assertFalse(param.enable_range_optimization)
        self.assertFalse(param.enable_extended_wildcard)
        self.assertEqual(param.type, finch.IndexType.INVERT)

        param2 = finch.InvertIndexParam(enable_range_optimization=True, enable_extended_wildcard=True)
        self.assertTrue(param2.enable_range_optimization)
        self.assertTrue(param2.enable_extended_wildcard)

        with self.assertRaises(AttributeError):
            param.enable_range_optimization = True  # type: ignore[misc]

    def test_hnsw_index_param(self):
        param = finch.HnswIndexParam()
        self.assertEqual(param.metric_type, finch.MetricType.IP)
        self.assertEqual(param.m, 50)
        self.assertEqual(param.ef_construction, 500)
        self.assertEqual(param.quantize_type, finch.QuantizeType.UNDEFINED)
        self.assertEqual(param.type, finch.IndexType.HNSW)
        self.assertFalse(hasattr(param, "scaling_factor"))

        param2 = finch.HnswIndexParam(metric_type=finch.MetricType.L2, m=10, ef_construction=1000, quantize_type=finch.QuantizeType.FP16)
        self.assertEqual(param2.metric_type, finch.MetricType.L2)
        self.assertEqual(param2.m, 10)
        self.assertEqual(param2.ef_construction, 1000)
        self.assertEqual(param2.quantize_type, finch.QuantizeType.FP16)

        with self.assertRaises(TypeError):
            finch.HnswIndexParam(scaling_factor=50)  # type: ignore[call-arg]

        with self.assertRaises(AttributeError):
            param.m = 1  # type: ignore[misc]

    def test_flat_index_param(self):
        param = finch.FlatIndexParam()
        self.assertEqual(param.type, finch.IndexType.FLAT)
        self.assertEqual(param.quantize_type, finch.QuantizeType.UNDEFINED)
        self.assertEqual(param.metric_type, finch.MetricType.IP)
        self.assertFalse(hasattr(param, "column_major"))

        param2 = finch.FlatIndexParam(metric_type=finch.MetricType.L2, quantize_type=finch.QuantizeType.INT8)
        self.assertEqual(param2.metric_type, finch.MetricType.L2)
        self.assertEqual(param2.quantize_type, finch.QuantizeType.INT8)

        with self.assertRaises(AttributeError):
            param.quantize_type = finch.QuantizeType.INT4  # type: ignore[misc]

    def test_ivf_index_param(self):
        param = finch.IVFIndexParam()
        self.assertEqual(param.metric_type, finch.MetricType.IP)
        self.assertEqual(param.n_list, 0)
        self.assertEqual(param.quantize_type, finch.QuantizeType.UNDEFINED)
        self.assertEqual(param.type, finch.IndexType.IVF)
        self.assertFalse(hasattr(param, "l1_index"))

        param2 = finch.IVFIndexParam(metric_type=finch.MetricType.L2, n_list=1000, quantize_type=finch.QuantizeType.FP16)
        self.assertEqual(param2.metric_type, finch.MetricType.L2)
        self.assertEqual(param2.n_list, 1000)
        self.assertEqual(param2.quantize_type, finch.QuantizeType.FP16)
        self.assertEqual(param2.type, finch.IndexType.IVF)

        with self.assertRaises(TypeError):
            finch.IVFIndexParam(l1_index=finch.IVFIndexParam())  # type: ignore[call-arg]

        with self.assertRaises(AttributeError):
            param.n_list = 10  # type: ignore[misc]

    def test_options(self):
        option = finch.CollectionOption()
        self.assertFalse(option.read_only)
        self.assertTrue(option.enable_mmap)
        with self.assertRaises(AttributeError):
            option.read_only = True  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            option.enable_mmap = False  # type: ignore[misc]

        index_opt = finch.IndexOption()
        self.assertEqual(index_opt.concurrency, 0)
        with self.assertRaises(AttributeError):
            index_opt.concurrency = 1  # type: ignore[misc]

        opt_opt = finch.OptimizeOption()
        self.assertEqual(opt_opt.concurrency, 0)
        with self.assertRaises(AttributeError):
            opt_opt.concurrency = 1  # type: ignore[misc]

        add_opt = finch.AddColumnOption()
        self.assertEqual(add_opt.concurrency, 0)
        with self.assertRaises(AttributeError):
            add_opt.concurrency = 1  # type: ignore[misc]

        alter_opt = finch.AlterColumnOption()
        self.assertEqual(alter_opt.concurrency, 0)
        with self.assertRaises(AttributeError):
            alter_opt.concurrency = 1  # type: ignore[misc]

    def test_hnsw_query_param(self):
        p = finch.HnswQueryParam()
        self.assertEqual(p.ef, 300)
        self.assertEqual(p.radius, 0.0)
        self.assertFalse(p.is_using_refiner)
        self.assertFalse(p.is_linear)

        p2 = finch.HnswQueryParam(ef=10, is_using_refiner=True, radius=30, is_linear=True)
        self.assertEqual(p2.ef, 10)
        self.assertEqual(p2.radius, 30.0)
        self.assertTrue(p2.is_using_refiner)
        self.assertTrue(p2.is_linear)

        with self.assertRaises(AttributeError):
            p.ef = 10  # type: ignore[misc]

    def test_ivf_query_param(self):
        p = finch.IVFQueryParam()
        self.assertEqual(p.nprobe, 10)
        self.assertEqual(p.n_probe, 10)

        with self.assertRaises(AttributeError):
            p.nprobe = 5  # type: ignore[misc]

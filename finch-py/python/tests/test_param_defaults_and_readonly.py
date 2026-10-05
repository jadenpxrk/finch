from __future__ import annotations

import re
import sys
import unittest

import finch


def _readonly_match_pattern() -> str:
    if sys.version_info >= (3, 11):
        return r"(can't set attribute|has no setter|readonly attribute)"
    return r"can't set attribute"


class ParamDefaultsAndReadonlyTest(unittest.TestCase):
    def test_invert_index_param_default_custom_and_readonly(self):
        p = finch.InvertIndexParam()
        self.assertFalse(p.enable_range_optimization)
        self.assertFalse(p.enable_extended_wildcard)
        self.assertEqual(p.type, finch.IndexType.INVERT)

        p2 = finch.InvertIndexParam(enable_range_optimization=True, enable_extended_wildcard=True)
        self.assertTrue(p2.enable_range_optimization)
        self.assertTrue(p2.enable_extended_wildcard)

        with self.assertRaises(AttributeError) as ctx:
            p.enable_range_optimization = False  # type: ignore[misc]
        self.assertRegex(str(ctx.exception), re.compile(_readonly_match_pattern()))

    def test_hnsw_index_param_default_custom_and_readonly(self):
        p = finch.HnswIndexParam()
        self.assertEqual(p.metric_type, finch.MetricType.IP)
        self.assertEqual(p.m, 50)
        self.assertEqual(p.ef_construction, 500)
        self.assertEqual(p.quantize_type, finch.QuantizeType.UNDEFINED)
        self.assertEqual(p.type, finch.IndexType.HNSW)

        p2 = finch.HnswIndexParam(
            metric_type=finch.MetricType.L2,
            m=10,
            ef_construction=1000,
            quantize_type=finch.QuantizeType.FP16,
        )
        self.assertEqual(p2.metric_type, finch.MetricType.L2)
        self.assertEqual(p2.m, 10)
        self.assertEqual(p2.ef_construction, 1000)
        self.assertEqual(p2.quantize_type, finch.QuantizeType.FP16)

        for attr in ("metric_type", "m", "ef_construction", "quantize_type"):
            with self.subTest(attr=attr):
                with self.assertRaises(AttributeError) as ctx:
                    setattr(p, attr, getattr(p, attr))
                self.assertRegex(str(ctx.exception), re.compile(_readonly_match_pattern()))

    def test_flat_index_param_default_custom_and_readonly(self):
        p = finch.FlatIndexParam()
        self.assertEqual(p.type, finch.IndexType.FLAT)
        self.assertEqual(p.quantize_type, finch.QuantizeType.UNDEFINED)
        self.assertEqual(p.metric_type, finch.MetricType.IP)

        p2 = finch.FlatIndexParam(metric_type=finch.MetricType.L2, quantize_type=finch.QuantizeType.INT8)
        self.assertEqual(p2.metric_type, finch.MetricType.L2)
        self.assertEqual(p2.quantize_type, finch.QuantizeType.INT8)

        for attr in ("metric_type", "quantize_type"):
            with self.subTest(attr=attr):
                with self.assertRaises(AttributeError) as ctx:
                    setattr(p, attr, getattr(p, attr))
                self.assertRegex(str(ctx.exception), re.compile(_readonly_match_pattern()))

    def test_ivf_index_param_default_custom_and_readonly(self):
        p = finch.IVFIndexParam()
        self.assertEqual(p.metric_type, finch.MetricType.IP)
        self.assertEqual(p.n_list, 0)
        self.assertEqual(p.quantize_type, finch.QuantizeType.UNDEFINED)
        self.assertEqual(p.type, finch.IndexType.IVF)

        p2 = finch.IVFIndexParam(metric_type=finch.MetricType.L2, n_list=1000, quantize_type=finch.QuantizeType.FP16)
        self.assertEqual(p2.metric_type, finch.MetricType.L2)
        self.assertEqual(p2.n_list, 1000)
        self.assertEqual(p2.quantize_type, finch.QuantizeType.FP16)
        self.assertEqual(p2.type, finch.IndexType.IVF)

        for attr in ("metric_type", "n_list", "quantize_type"):
            with self.subTest(attr=attr):
                with self.assertRaises(AttributeError) as ctx:
                    setattr(p, attr, getattr(p, attr))
                self.assertRegex(str(ctx.exception), re.compile(_readonly_match_pattern()))

    def test_option_objects_default_custom_and_readonly(self):
        co = finch.CollectionOption()
        self.assertFalse(co.read_only)
        self.assertTrue(co.enable_mmap)

        co2 = finch.CollectionOption(read_only=True, enable_mmap=False)
        self.assertTrue(co2.read_only)
        self.assertFalse(co2.enable_mmap)

        with self.assertRaises(AttributeError) as ctx:
            co.read_only = True  # type: ignore[misc]
        self.assertRegex(str(ctx.exception), re.compile(_readonly_match_pattern()))

        io = finch.IndexOption()
        self.assertEqual(io.concurrency, 0)
        io2 = finch.IndexOption(concurrency=10)
        self.assertEqual(io2.concurrency, 10)
        with self.assertRaises(AttributeError) as ctx:
            io.concurrency = 1  # type: ignore[misc]
        self.assertRegex(str(ctx.exception), re.compile(_readonly_match_pattern()))

        for cls in (finch.AddColumnOption, finch.AlterColumnOption, finch.OptimizeOption):
            opt = cls()
            self.assertEqual(opt.concurrency, 0)
            opt2 = cls(concurrency=10)
            self.assertEqual(opt2.concurrency, 10)
            with self.subTest(cls=cls.__name__):
                with self.assertRaises(AttributeError) as ctx:
                    opt.concurrency = 1  # type: ignore[misc]
                self.assertRegex(str(ctx.exception), re.compile(_readonly_match_pattern()))

    def test_hnsw_query_param_default_custom_and_readonly(self):
        qp = finch.HnswQueryParam()
        self.assertEqual(qp.ef, 300)
        self.assertFalse(qp.is_using_refiner)
        self.assertEqual(qp.radius, 0)
        self.assertFalse(qp.is_linear)

        qp2 = finch.HnswQueryParam(ef=10, is_using_refiner=True, radius=30, is_linear=True)
        self.assertEqual(qp2.ef, 10)
        self.assertTrue(qp2.is_using_refiner)
        self.assertEqual(qp2.radius, 30)
        self.assertTrue(qp2.is_linear)

        with self.assertRaises(AttributeError) as ctx:
            qp.ef = 10  # type: ignore[misc]
        self.assertRegex(str(ctx.exception), re.compile(_readonly_match_pattern()))

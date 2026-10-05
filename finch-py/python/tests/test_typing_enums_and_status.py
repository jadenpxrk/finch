from __future__ import annotations

import unittest

import finch


class TypingEnumsAndStatusTest(unittest.TestCase):
    def test_enum_names_and_values(self):
        self.assertEqual(finch.DataType.FLOAT.name, "FLOAT")
        self.assertEqual(finch.DataType.FLOAT.value, 8)
        self.assertEqual(finch.IndexType.HNSW.name, "HNSW")
        self.assertEqual(finch.IndexType.HNSW.value, 1)
        self.assertEqual(finch.MetricType.COSINE.name, "COSINE")
        self.assertEqual(finch.MetricType.COSINE.value, 3)
        self.assertEqual(finch.QuantizeType.INT8.name, "INT8")
        self.assertEqual(finch.QuantizeType.INT8.value, 2)
        self.assertEqual(finch.StatusCode.OK.name, "OK")
        self.assertEqual(finch.StatusCode.OK.value, 0)

        for m in ["L2", "IP", "COSINE"]:
            self.assertIn(m, finch.MetricType.__members__)

        for m in ["HNSW", "IVF", "FLAT", "INVERT"]:
            self.assertIn(m, finch.IndexType.__members__)

        for m in ["FP16", "INT8", "INT4", "UNDEFINED"]:
            self.assertIn(m, finch.QuantizeType.__members__)

    def test_status_api(self):
        s = finch.Status(finch.StatusCode.OK)
        self.assertEqual(s.code(), finch.StatusCode.OK)
        self.assertTrue(s.ok())

        s2 = finch.Status(finch.StatusCode.NOT_FOUND, "Not Found")
        self.assertEqual(s2.message(), "Not Found")
        self.assertFalse(s2.ok())

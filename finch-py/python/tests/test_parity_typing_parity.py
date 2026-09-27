from __future__ import annotations

import unittest

import finch


class FinchReferenceTypingParityTest(unittest.TestCase):
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

    def test_members_exist(self):
        for m in [
            "STRING",
            "BOOL",
            "INT32",
            "INT64",
            "FLOAT",
            "DOUBLE",
            "UINT32",
            "UINT64",
            "VECTOR_FP16",
            "VECTOR_FP32",
            "VECTOR_FP64",
            "VECTOR_INT8",
            "SPARSE_VECTOR_FP16",
            "SPARSE_VECTOR_FP32",
            "ARRAY_STRING",
            "ARRAY_INT32",
            "ARRAY_INT64",
            "ARRAY_FLOAT",
            "ARRAY_DOUBLE",
            "ARRAY_BOOL",
            "ARRAY_UINT32",
            "ARRAY_UINT64",
        ]:
            self.assertIn(m, finch.DataType.__members__)

        for m in ["HNSW", "IVF", "FLAT", "INVERT"]:
            self.assertIn(m, finch.IndexType.__members__)

        for m in ["L2", "IP", "COSINE", "MIPS_L2", "HAMMING"]:
            self.assertIn(m, finch.MetricType.__members__)

        for m in ["FP16", "INT8", "INT4", "UNDEFINED"]:
            self.assertIn(m, finch.QuantizeType.__members__)

        for m in [
            "OK",
            "UNKNOWN",
            "NOT_FOUND",
            "ALREADY_EXISTS",
            "INVALID_ARGUMENT",
            "PERMISSION_DENIED",
            "FAILED_PRECONDITION",
            "RESOURCE_EXHAUSTED",
            "UNAVAILABLE",
            "INTERNAL_ERROR",
            "NOT_SUPPORTED",
        ]:
            self.assertIn(m, finch.StatusCode.__members__)

    def test_status_api(self):
        s = finch.Status()
        self.assertTrue(s.ok())
        self.assertEqual(s.code(), finch.StatusCode.OK)
        self.assertEqual(s.message(), "")
        self.assertTrue(bool(s))

        s2 = finch.Status(finch.StatusCode.INVALID_ARGUMENT, "bad")
        self.assertFalse(s2.ok())
        self.assertEqual(s2.code(), finch.StatusCode.INVALID_ARGUMENT)
        self.assertEqual(s2.message(), "bad")
        # reference parity: Status objects are truthy regardless of code (tests use
        # `assert bool(result)` as a non-None check even for NOT_FOUND, etc).
        self.assertTrue(bool(s2))

        self.assertTrue(finch.Status.OK().ok())
        self.assertEqual(finch.Status.NotFound("x").code(), finch.StatusCode.NOT_FOUND)

        # reference-style factories should exist (some codes map to closest Finch equivalents).
        self.assertEqual(finch.Status.AlreadyExists("x").code(), finch.StatusCode.ALREADY_EXISTS)
        self.assertEqual(finch.Status.InvalidArgument("x").code(), finch.StatusCode.INVALID_ARGUMENT)
        self.assertEqual(finch.Status.PermissionDenied("x").code(), finch.StatusCode.PERMISSION_DENIED)
        self.assertIsNotNone(finch.Status.FailedPrecondition("x"))
        self.assertIsNotNone(finch.Status.ResourceExhausted("x"))
        self.assertIsNotNone(finch.Status.Unavailable("x"))
        self.assertIsNotNone(finch.Status.InternalError("x"))
        self.assertIsNotNone(finch.Status.NotSupported("x"))
        self.assertIsNotNone(finch.Status.Unknown("x"))

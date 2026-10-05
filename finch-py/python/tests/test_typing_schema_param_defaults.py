import unittest


class TypingSchemaParamDefaultsTest(unittest.TestCase):
    def test_enum_upper_snake_aliases_exist(self) -> None:
        import finch

        self.assertIs(finch.DataType.INT64, finch.DataType.Int64)
        self.assertIs(finch.MetricType.IP, finch.MetricType.InnerProduct)
        self.assertIs(finch.QuantizeType.FP16, finch.QuantizeType.Fp16)

    def test_invert_index_param_defaults(self) -> None:
        import finch

        p = finch.InvertIndexParam()
        self.assertFalse(bool(p.enable_range_optimization))
        self.assertFalse(bool(p.enable_extended_wildcard))

    def test_field_schema_nullable_default_false(self) -> None:
        import finch

        f = finch.FieldSchema("id", finch.DataType.INT64)
        self.assertFalse(bool(f.nullable))

    def test_collection_schema_duplicate_names_rejected(self) -> None:
        import finch

        schema = finch.CollectionSchema("dup")
        schema.add_field(finch.FieldSchema("x", finch.DataType.INT32))
        schema.add_field(finch.VectorSchema("v", dim=4, data_type=finch.DataType.VECTOR_FP32))
        with self.assertRaises(Exception):
            schema.add_field(finch.FieldSchema("x", finch.DataType.INT32))


if __name__ == "__main__":
    unittest.main(verbosity=2)


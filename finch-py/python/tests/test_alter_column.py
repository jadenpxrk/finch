import shutil
import tempfile
import unittest


class FinchAlterColumnTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import finch

        try:
            finch.init()
        except RuntimeError:
            pass

    def test_alter_column_modify_schema_and_rename(self) -> None:
        import finch

        root = tempfile.mkdtemp(prefix="finch_py_alter_")
        path = f"{root}/collection"
        try:
            schema = finch.CollectionSchema("alter")
            inv = finch.InvertIndexParam(enable_range_optimization=True)
            schema.add_field(finch.FieldSchema("id", finch.DataType.INT64, nullable=False).with_invert_index(inv))
            schema.add_field(finch.VectorSchema("emb", dim=4, data_type=finch.DataType.VECTOR_FP32))

            col = finch.create_and_open(path, schema, finch.CollectionOption())
            s = col.insert(
                finch.Doc(
                    id="1",
                    fields={"id": 1},
                    vectors={"emb": [1.0, 0.0, 0.0, 0.0]},
                )
            )
            self.assertTrue(s.is_ok())
            col.flush()

            col.alter_column(
                old_name="id",
                field_schema=finch.FieldSchema("doc_id", finch.DataType.UINT64, nullable=True),
            )

            self.assertIsNone(col.schema.field("id"))
            fs = col.schema.field("doc_id")
            self.assertIsNotNone(fs)
            self.assertEqual(fs.name, "doc_id")
            self.assertEqual(fs.data_type, finch.DataType.UINT64)
            self.assertTrue(fs.nullable)

            fetched = col.fetch("1")
            self.assertIn("1", fetched)
            d = fetched["1"]
            self.assertIn("doc_id", d.fields)
            self.assertNotIn("id", d.fields)
            self.assertEqual(int(d.fields["doc_id"]), 1)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_alter_column_rename_rewrites_persisted_forward(self) -> None:
        import finch

        root = tempfile.mkdtemp(prefix="finch_py_alter_rename_")
        path = f"{root}/collection"
        try:
            schema = finch.CollectionSchema("alter_rename")
            # reference parity: collections must have at least one vector field, even
            # if the test is scalar-only.
            schema.add_field(
                finch.VectorSchema(
                    "dummy_vec",
                    dim=1,
                    data_type=finch.DataType.VECTOR_FP32,
                    nullable=True,
                )
            )
            schema.add_field(finch.FieldSchema("weight", finch.DataType.FLOAT64, nullable=True))
            col = finch.create_and_open(path, schema, finch.CollectionOption())
            col.insert(finch.Doc(id="1", fields={"weight": 80.5}))
            col.flush()

            col.alter_column(old_name="weight", new_name="mass")

            fetched = col.fetch("1")
            self.assertIn("mass", fetched["1"].fields)
            self.assertNotIn("weight", fetched["1"].fields)
            self.assertAlmostEqual(float(fetched["1"].fields["mass"]), 80.5, places=6)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)

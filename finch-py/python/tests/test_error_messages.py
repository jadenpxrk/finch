import shutil
import tempfile
import unittest


class ErrorMessagesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import finch

        try:
            finch.init()
        except RuntimeError:
            pass

    def test_alter_column_missing_column_message(self) -> None:
        import finch

        root = tempfile.mkdtemp(prefix="finch_py_errmsg_")
        path = f"{root}/collection"
        try:
            schema = finch.CollectionSchema("errmsg")
            # A collection needs at least one vector field.
            schema.add_field(
                finch.VectorSchema(
                    "dummy_vec",
                    dim=1,
                    data_type=finch.DataType.VECTOR_FP32,
                    nullable=True,
                )
            )
            schema.add_field(finch.FieldSchema("x", finch.DataType.INT32, nullable=True))
            col = finch.create_and_open(path, schema, finch.CollectionOption())
            with self.assertRaises(Exception) as ctx:
                col.alter_column(old_name="non_existing", new_name="y")
            self.assertIn("column non_existing not found", str(ctx.exception))
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)

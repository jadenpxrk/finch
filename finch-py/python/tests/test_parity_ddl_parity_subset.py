import shutil
import tempfile
import unittest


class FinchReferenceDdlParitySubsetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import finch

        try:
            finch.init()
        except RuntimeError:
            pass

    def test_column_ddl_only_allows_basic_numeric(self) -> None:
        import finch

        root = tempfile.mkdtemp(prefix="finch_py_ddl_")
        path = f"{root}/collection"
        try:
            schema = finch.CollectionSchema("ddl")
            schema.add_field(
                finch.VectorSchema(
                    "dummy_vec",
                    dim=1,
                    data_type=finch.DataType.VECTOR_FP32,
                    nullable=True,
                )
            )
            schema.add_field(finch.FieldSchema("id", finch.DataType.INT64, nullable=False))
            schema.add_field(finch.FieldSchema("name", finch.DataType.STRING, nullable=True))
            col = finch.create_and_open(path, schema, finch.CollectionOption())

            col.add_column(finch.FieldSchema("age", finch.DataType.INT32, nullable=True), expression="100")

            with self.assertRaises(ValueError):
                col.add_column(finch.FieldSchema("full_name", finch.DataType.STRING, nullable=True), expression="'x'")

            with self.assertRaises(ValueError):
                col.drop_column("name")

            with self.assertRaises(ValueError):
                col.alter_column(old_name="name", new_name="full_name")
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)

import shutil
import tempfile
import unittest


class FinchFieldNameValidationSubsetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import finch

        try:
            finch.init()
        except RuntimeError:
            pass

    def test_add_column_rejects_invalid_names(self) -> None:
        import finch

        root = tempfile.mkdtemp(prefix="finch_py_name_")
        path = f"{root}/collection"
        try:
            schema = finch.CollectionSchema("name_test")
            schema.add_field(
                finch.VectorSchema(
                    "dummy_vec",
                    dim=1,
                    data_type=finch.DataType.VECTOR_FP32,
                    nullable=True,
                )
            )
            schema.add_field(finch.FieldSchema("id", finch.DataType.INT64, nullable=False))
            col = finch.create_and_open(path, schema, finch.CollectionOption())

            invalid = ["", " ", "a" * 33, "field name", "field.name", "field@name", "field/name"]
            for name in invalid:
                with self.assertRaises(Exception):
                    col.add_column(finch.FieldSchema(name, finch.DataType.INT32), expression="100")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_alter_column_rejects_invalid_new_names(self) -> None:
        import finch

        root = tempfile.mkdtemp(prefix="finch_py_name_")
        path = f"{root}/collection"
        try:
            schema = finch.CollectionSchema("name_test2")
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
            col.insert(finch.Doc(id="1", fields={"x": 1}))
            col.flush()

            invalid = ["", " ", "a" * 33, "field name", "field.name", "field@name", "field/name"]
            for name in invalid:
                with self.assertRaises(Exception):
                    col.alter_column(old_name="x", new_name=name)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)

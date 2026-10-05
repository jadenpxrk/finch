from __future__ import annotations

import shutil
import tempfile
import unittest

import finch


class CollectionInvalidFieldAndVectorNameTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            finch.init()
        except RuntimeError:
            pass

    def _create_collection(self, path: str) -> finch.Collection:
        schema = finch.CollectionSchema(
            name="invalid_name",
            fields=[
                finch.FieldSchema(
                    "id",
                    finch.DataType.INT64,
                    nullable=False,
                    index_param=finch.InvertIndexParam(enable_range_optimization=True),
                ),
                finch.FieldSchema(
                    "name",
                    finch.DataType.STRING,
                    nullable=True,
                    index_param=finch.InvertIndexParam(),
                ),
            ],
            vectors=[
                finch.VectorSchema(
                    "dense",
                    finch.DataType.VECTOR_FP32,
                    dimension=8,
                    index_param=finch.HnswIndexParam(),
                ),
            ],
        )
        return finch.create_and_open(path, schema, finch.CollectionOption(read_only=False, enable_mmap=True))

    def test_invalid_field_and_vector_name_create_drop_index(self) -> None:
        # The error message depends on the kind of invalid name:
        # - invalid strings => "not found in schema"
        # - invalid non-strings => "incompatible function arguments"
        invalid: list[object] = [
            "",
            " ",
            "v" * 33,
            "vector name",
            "vector@name",
            "vector/name",
            "vector\\name",
            "vector.name",
            "vector$data",
            "vector+name",
            "vector=name",
            None,
            1,
            1.1,
        ]

        root = tempfile.mkdtemp(prefix="finch_test_invalid_name_")
        path = f"{root}/collection"
        try:
            col = self._create_collection(path)

            for name in invalid:
                with self.subTest(op="create_vector_index", name=name):
                    try:
                        col.create_index(field_name=name, index_param=finch.HnswIndexParam(), option=finch.IndexOption())  # type: ignore[arg-type]
                    except Exception as e:
                        msg = str(e)
                        if isinstance(name, str):
                            self.assertIn("not found in schema", msg)
                        else:
                            self.assertIn("incompatible function arguments", msg)
                    else:  # pragma: no cover
                        self.fail("expected create_index to fail")

                with self.subTest(op="create_scalar_index", name=name):
                    try:
                        col.create_index(field_name=name, index_param=finch.InvertIndexParam(), option=finch.IndexOption())  # type: ignore[arg-type]
                    except Exception as e:
                        msg = str(e)
                        if isinstance(name, str):
                            self.assertIn("not found in schema", msg)
                        else:
                            self.assertIn("incompatible function arguments", msg)
                    else:  # pragma: no cover
                        self.fail("expected create_index to fail")

            col.destroy()

            # Recreate for drop_index checks.
            col2 = self._create_collection(path)
            for name in invalid:
                with self.subTest(op="drop_index", name=name):
                    try:
                        col2.drop_index(field_name=name)  # type: ignore[arg-type]
                    except Exception as e:
                        msg = str(e)
                        if isinstance(name, str):
                            self.assertIn("not found in schema", msg)
                        else:
                            self.assertIn("incompatible function arguments", msg)
                    else:  # pragma: no cover
                        self.fail("expected drop_index to fail")
            col2.destroy()
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)


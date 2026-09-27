from __future__ import annotations

import shutil
import tempfile
import unittest

import finch


def _schema() -> finch.CollectionSchema:
    return finch.CollectionSchema(
        name="column_ddl",
        fields=[
            finch.FieldSchema("id", finch.DataType.INT64, nullable=False),
            finch.FieldSchema("name", finch.DataType.STRING, nullable=True),
            finch.FieldSchema("weight", finch.DataType.FLOAT, nullable=True),
        ],
        vectors=[
            finch.VectorSchema("dense", finch.DataType.VECTOR_FP32, 4),
        ],
    )


class FinchReferenceCollectionColumnDdlParitySubsetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            finch.init()
        except RuntimeError:
            pass

    def test_drop_column_removes_field_from_schema_and_fetch(self) -> None:
        root = tempfile.mkdtemp(prefix="finch_parity_column_ddl_")
        path = f"{root}/collection"
        try:
            col = finch.create_and_open(path, _schema(), finch.CollectionOption(read_only=False, enable_mmap=True))

            col.add_column(
                field_schema=finch.FieldSchema("temp_field", finch.DataType.INT32, nullable=True),
                expression="100",
            )

            d1 = finch.Doc(
                id="1",
                fields={"id": 1, "name": "x", "weight": 1.0, "temp_field": 123},
                vectors={"dense": [0.1, 0.2, 0.3, 0.4]},
            )
            s = col.insert(d1)
            self.assertTrue(s.ok())
            col.flush()

            got = col.fetch("1")["1"]
            self.assertIn("temp_field", got.fields)

            col.drop_column("temp_field")

            got2 = col.fetch("1")["1"]
            self.assertNotIn("temp_field", got2.fields)

            # Inserting a dropped field should fail schema validation.
            with self.assertRaises(Exception):
                col.insert(
                    finch.Doc(
                        id="2",
                        fields={"id": 2, "name": "y", "weight": 2.0, "temp_field": 1},
                        vectors={"dense": [0.1, 0.2, 0.3, 0.4]},
                    )
                )
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_drop_column_missing_raises(self) -> None:
        root = tempfile.mkdtemp(prefix="finch_parity_drop_missing_")
        path = f"{root}/collection"
        try:
            col = finch.create_and_open(path, _schema(), finch.CollectionOption(read_only=False, enable_mmap=True))
            with self.assertRaises(Exception) as ctx:
                col.drop_column("non_existing_column")
            self.assertIn("not found", str(ctx.exception).lower())
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)


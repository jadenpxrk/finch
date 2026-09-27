import shutil
import tempfile
import unittest


class FinchSmallScalarRoundtripTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import finch

        try:
            finch.init()
        except RuntimeError:
            pass

    def test_small_scalars_survive_flush_and_fetch(self) -> None:
        import finch

        values = {
            "i8": (finch.DataType.Int8, 7),
            "i16": (finch.DataType.Int16, -300),
            "u8": (finch.DataType.Uint8, 200),
            "u16": (finch.DataType.Uint16, 60000),
            "f16": (finch.DataType.Float16, 1.5),
        }
        root = tempfile.mkdtemp(prefix="finch_py_small_scalar_")
        try:
            schema = finch.CollectionSchema("small")
            for name, (data_type, _) in values.items():
                schema.add_field(finch.FieldSchema(name, data_type, nullable=True))
            schema.add_field(finch.VectorSchema("emb", dim=2, data_type=finch.DataType.VectorFp32))
            col = finch.create_and_open(f"{root}/collection", schema, finch.CollectionOption())
            status = col.insert(
                finch.Doc(
                    id="1",
                    fields={name: value for name, (_, value) in values.items()},
                    vectors={"emb": [1.0, 0.0]},
                )
            )
            self.assertTrue(status.is_ok(), status.message)
            col.flush()

            fields = col.fetch("1")["1"].fields
            for name, (_, value) in values.items():
                self.assertEqual(fields[name], value, name)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_small_scalar_out_of_range_is_rejected(self) -> None:
        import finch

        schema = finch.CollectionSchema("small")
        schema.add_field(finch.FieldSchema("i8", finch.DataType.Int8, nullable=True))
        schema.add_field(finch.VectorSchema("emb", dim=2, data_type=finch.DataType.VectorFp32))
        root = tempfile.mkdtemp(prefix="finch_py_small_scalar_")
        try:
            col = finch.create_and_open(f"{root}/collection", schema, finch.CollectionOption())
            with self.assertRaises((ValueError, OverflowError)):
                col.insert(finch.Doc(id="1", fields={"i8": 128}, vectors={"emb": [1.0, 0.0]}))
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

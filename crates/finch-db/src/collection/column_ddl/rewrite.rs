use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{new_null_array, Array};
use arrow::datatypes::{Field, SchemaRef};
use arrow::ipc::reader::FileReader;
use arrow::ipc::writer::FileWriter;
use arrow::record_batch::RecordBatch;
use finch_types::{CollectionSchema, DataType, Status, ZResult};

use crate::collection_files::is_parquet_file;

mod add;
mod alter;

pub(super) use add::rewrite_forward_add_numeric_column;
pub(super) use alter::rewrite_forward_alter_numeric_column;

pub(super) fn is_basic_numeric(dt: DataType) -> bool {
    matches!(
        dt,
        DataType::Int32
            | DataType::Int64
            | DataType::Uint32
            | DataType::Uint64
            | DataType::Float32
            | DataType::Float64
    )
}

fn io_error(e: impl ToString) -> Status {
    Status::io_error(e.to_string())
}

// Produces the rewritten column of each batch; every other column is copied through.
trait ColumnRewrite {
    // Runs once per batch before any output column is assembled.
    fn start_batch(&mut self, _batch: &RecordBatch) -> ZResult<()> {
        Ok(())
    }

    fn column(&mut self, batch: &RecordBatch, field: &Field) -> ZResult<Arc<dyn Array>>;
}

// Copy-on-write rewrite of one forward store under a new schema.
struct ForwardRewrite<'a> {
    src_forward_path: &'a Path,
    dst_forward_path: &'a Path,
    tmp_path: PathBuf,
    parent: &'a Path,
    out_schema: SchemaRef,
}

impl<'a> ForwardRewrite<'a> {
    // Creates the destination directory; `run` writes the new forward store.
    fn begin(
        src_forward_path: &'a Path,
        dst_forward_path: &'a Path,
        new_schema: &CollectionSchema,
    ) -> ZResult<Self> {
        let tmp_path = if is_parquet_file(dst_forward_path) {
            dst_forward_path.with_extension("parquet.tmp")
        } else {
            dst_forward_path.with_extension("arrow.tmp")
        };
        let parent = dst_forward_path.parent().unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent).map_err(io_error)?;

        Ok(Self {
            src_forward_path,
            dst_forward_path,
            tmp_path,
            parent,
            out_schema: finch_storage::ipc_schema_for_collection(new_schema),
        })
    }

    fn run(self, target: &str, rewrite: &mut impl ColumnRewrite) -> ZResult<()> {
        if is_parquet_file(self.src_forward_path) {
            self.write_parquet(target, rewrite)?;
        } else {
            self.write_ipc(target, rewrite)?;
        }
        self.publish()
    }

    fn rewrite_batch(
        &self,
        batch: &RecordBatch,
        target: &str,
        rewrite: &mut impl ColumnRewrite,
    ) -> ZResult<RecordBatch> {
        rewrite.start_batch(batch)?;
        let mut cols: Vec<Arc<dyn Array>> = Vec::with_capacity(self.out_schema.fields().len());
        for f in self.out_schema.fields() {
            let name = f.name();
            if name == target {
                cols.push(rewrite.column(batch, f)?);
                continue;
            }
            if let Some(col) = batch.column_by_name(name) {
                cols.push(col.clone());
            } else {
                cols.push(new_null_array(f.data_type(), batch.num_rows()));
            }
        }
        RecordBatch::try_new(self.out_schema.clone(), cols).map_err(io_error)
    }

    fn write_parquet(&self, target: &str, rewrite: &mut impl ColumnRewrite) -> ZResult<()> {
        let out_file = std::fs::File::create(&self.tmp_path).map_err(io_error)?;
        let props = parquet::file::properties::WriterProperties::builder().build();
        let mut writer = parquet::arrow::arrow_writer::ArrowWriter::try_new(
            out_file,
            self.out_schema.clone(),
            Some(props),
        )
        .map_err(io_error)?;

        let in_file = std::fs::File::open(self.src_forward_path).map_err(io_error)?;
        let mut builder =
            parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(in_file)
                .map_err(io_error)?;
        builder = builder.with_batch_size(4096);
        let reader = builder.build().map_err(io_error)?;

        for batch in reader {
            let batch = batch.map_err(io_error)?;
            let out_batch = self.rewrite_batch(&batch, target, rewrite)?;
            writer.write(&out_batch).map_err(io_error)?;
        }

        writer.close().map_err(io_error)?;
        Ok(())
    }

    fn write_ipc(&self, target: &str, rewrite: &mut impl ColumnRewrite) -> ZResult<()> {
        let out_file = std::fs::File::create(&self.tmp_path).map_err(io_error)?;
        let mut writer = FileWriter::try_new(out_file, &self.out_schema).map_err(io_error)?;

        let in_file = std::fs::File::open(self.src_forward_path).map_err(io_error)?;
        let reader = FileReader::try_new(in_file, None).map_err(io_error)?;

        for batch_result in reader {
            let batch = batch_result.map_err(io_error)?;
            let out_batch = self.rewrite_batch(&batch, target, rewrite)?;
            writer.write(&out_batch).map_err(io_error)?;
        }

        writer.finish().map_err(io_error)
    }

    fn publish(self) -> ZResult<()> {
        // Best-effort durability: sync file contents before rename, then sync dir.
        if let Ok(f) = std::fs::File::open(&self.tmp_path) {
            let _ = f.sync_all();
        }

        // Replace any existing file (Windows-compatible).
        let _ = std::fs::remove_file(self.dst_forward_path);
        std::fs::rename(&self.tmp_path, self.dst_forward_path).map_err(io_error)?;

        if let Ok(dir) = std::fs::File::open(self.parent) {
            let _ = dir.sync_all();
        }
        Ok(())
    }
}

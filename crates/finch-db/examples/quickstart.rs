use finch_db::Collection;
use finch_types::{CollectionOptions, CollectionSchema, DataType, Doc, FieldSchema, VectorQuery};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir().join("finch-quickstart");
    let _ = std::fs::remove_dir_all(&dir);

    let schema = CollectionSchema::new("notes")
        .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4));
    let col = Collection::create_and_open(&dir, schema, CollectionOptions::default())?;

    col.insert(vec![Doc::new("d1").set("emb", vec![1.0f32, 0.0, 0.0, 0.0])])?;
    col.flush()?;

    let hits = col.query(VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 1))?;
    println!("top hit: {}", hits[0].pk);
    Ok(())
}

# Finch

Finch is a vector database written in Rust. It runs as a library inside the calling process.
Finch also has a memory layer for language-model agents, built on top of the database.

The database does these things:

- It stores documents with dense or sparse vectors and scalar fields in a directory on disk.
- It answers nearest-neighbor queries with HNSW, IVF, or flat indexes.
- It filters on scalar fields with a SQL-like syntax.
- After a crash, it recovers its last committed state from a write-ahead log and a manifest
  file.

The memory layer does these things:

- It stores conversation episodes and the claims extracted from them in Finch collections.
- It records when each claim held and when each correction or replacement of it happened.
- It packs the current claims and their source text into a context for a model to answer
  from.

## Layout

| Path | What it is |
|---|---|
| `crates/finch-types` | Schema, query, and error types |
| `crates/finch-proto` | Manifest encoding |
| `crates/finch-storage` | Columnar forward store (Arrow IPC or Parquet, optional mmap) |
| `crates/finch-core` | Index algorithms and quantization |
| `crates/finch-db` | Collections, segments, WAL, SQL-like filtering |
| `crates/finch-memory` | The memory layer |
| `finch-py`, `finch-node` | Python and Node.js bindings |

## Rust

```toml
[dependencies]
finch-db = { path = "crates/finch-db" }
finch-types = { path = "crates/finch-types" }
```

```rust
use std::path::Path;

use finch_db::Collection;
use finch_types::{CollectionOptions, CollectionSchema, DataType, Doc, FieldSchema, VectorQuery};

let schema = CollectionSchema::new("notes")
    .with_field(FieldSchema::new("emb", DataType::VectorFp32).with_dimension(4));
let col = Collection::create_and_open(Path::new("./notes"), schema, CollectionOptions::default())?;

// `insert` returns one `Status` per document; a failed document does not fail the call.
col.insert(vec![Doc::new("d1").set("emb", vec![1.0f32, 0.0, 0.0, 0.0])])?;
col.flush()?;

let hits = col.query(VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 1))?;
assert_eq!(hits[0].pk, "d1");
```

A field is required by default: `FieldSchema::new` in Rust, `FieldSchema` in Python, and
`FieldSchemaOptions.nullable` in Node all default to not nullable. Call `.nullable()` in Rust or
pass `nullable=True` to let a document omit the field or store null in it.

Run the tests:

```bash
cargo test --workspace
```

## Python

```bash
cd finch-py
python -m maturin develop --release
```

```python
import finch
col = finch.create_and_open("./notes", schema)
```

The Python API follows the Rust one. Its definitions are in `finch-py/python/finch`.

## Documentation

- `docs/architecture.md`: the crates, the write and read paths, crash recovery, and how
  the memory layer stores, corrects, and packs claims.
- `docs/vectordbbench.md`: how to run VectorDBBench against Finch.
- `docs/global-config.md`: process-wide settings.

## Building for one machine

On x86_64, building with `RUSTFLAGS="-C target-cpu=native"` lets the compiler use every
instruction set the build machine has. Use that flag only for a build that runs on the same
machine. A binary built with the flag can stop with an illegal-instruction error on a CPU
that lacks one of those instruction sets. Without the flag, the distance functions in
`finch-core` still choose AVX-512, AVX2, or SSE4.1 at runtime. The flag therefore changes
only code that the compiler vectorizes by itself.

## Benchmarks

These numbers come from runs of the code in this repository on 2026-09-26, through a benchmark
harness that is not part of this repository. Each run used the benchmark's own evaluator. No
product code reads a benchmark name, question id, label, or gold answer.

### MEME

MEME tests whether an agent's memory keeps up with facts that change. Each of its 100 episodes
is a conversation with 32k tokens of filler text, followed by questions in five task types.
Finch extracted memories with `gpt-4.1-mini` and answered with `gpt-4.1-mini` under the
official answer protocol. The official MEME evaluator judged the answers with `gpt-4o`.

| Task | Correct | Share | What the task checks |
|---|---:|---:|---|
| All | 481 / 694 | 69.3% | |
| Absence | 104 / 130 | 80.0% | The agent says a fact was never stated. |
| Deletion | 73 / 100 | 73.0% | The agent drops a fact the user retracted. |
| Cascade | 117 / 164 | 71.3% | A changed fact updates the facts that depend on it. |
| Tracking | 69 / 100 | 69.0% | The agent reports the current value after many updates. |
| Aggregation | 18 / 100 | 18.0% | The agent combines several remembered items into one answer. |

### EnterpriseRAG-Bench

The repository contains the EnterpriseRAG pipeline in its general form. Retrieval breadth comes
from the `k` and scan-limit settings, not from constants fitted to the benchmark. A run of this
form on 2026-09-26 scored as follows.

| Metric | Score |
|---|---|
| Combined (headline) | 69.05 |
| Correct | 76.8% |
| Completeness | 76.0% |
| Recall | 69.9% |

## Known limitations

Dropping a collection can hang. Each collection keeps its primary-key map in fjall, and fjall
versions 3.0 to 3.1.10 can block forever while a database shuts down. The report is
[fjall issue 260](https://github.com/fjall-rs/fjall/issues/260) and the fix is
[fjall pull request 321](https://github.com/fjall-rs/fjall/pull/321). The hang is rare. In a
test that opened and dropped about 7,500 collections, it happened once. When the fix ships,
Finch will move to that version.

## Status

The database crates are `finch-types`, `finch-proto`, `finch-storage`, `finch-core`, and
`finch-db`. They have 391 tests. One test carries the ignore attribute. The memory layer,
`finch-memory`, has 286 tests. All of them use synthetic data. The repository contains no
cached benchmark replays.

The project has not had a public release.

## License

Apache-2.0. See `LICENSE`.

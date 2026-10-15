# Finch

Finch is a memory layer for language-model agents, written in Rust. It does these things:

- It stores conversation episodes and the claims extracted from them in Postgres with pgvector.
- It records when each claim held and when each correction or replacement of it happened.
- It packs the current claims and their source text into a context for a model to answer
  from.

## Layout

| Path | What it is |
|---|---|
| `crates/finch-types` | Value and error types |
| `crates/finch-memory` | The memory layer, stored in Postgres with pgvector |
| `finch-py`, `finch-node` | Python and Node.js bindings |

## Postgres

Finch keeps its stores in Postgres with the pgvector extension. Start a server:

```bash
docker run -d --name finch-pg --shm-size=1g -p 55432:5432 -e POSTGRES_USER=finch \
  -e POSTGRES_PASSWORD=finch -e POSTGRES_DB=finch pgvector/pgvector:pg17
```

Run the tests against that server:

```bash
FINCH_MEMORY_TEST_POSTGRES_URL=postgres://finch:finch@localhost:55432/finch cargo test --workspace
```

## Python

```bash
cd finch-py
python -m maturin develop --release
```

```python
import finch

memory = finch.FinchMemory.create(
    url="postgres://finch:finch@localhost:55432/finch",
    name="notes",
    embedding_dim=4,
)
memory.ingest_episode({"scope": {"space": "team"}, "text": "Payments go through Stripe."})

result = memory.search({"scope": {"space": "team"}, "query": "payments", "mode": "keyword"})
print(result["hits"][0])
```

The Python API follows the Rust one. Its definitions are in `finch-py/python/finch/memory.py`.

## Documentation

`docs/architecture.md` describes how the memory layer stores, corrects, and packs claims.

## Benchmarks

These numbers come from Finch runs through a benchmark harness that is not part of this
repository. Each run used the benchmark's own evaluator. No product code reads a benchmark
name, question id, label, or gold answer.

### MEME

MEME tests whether an agent's memory keeps up with facts that change. Each of its 100 episodes
is a conversation with 32k tokens of filler text, followed by questions in five task types.
Finch extracted and selected memories with `gpt-6-luna` and answered with `gpt-4.1-mini` under
the official answer protocol. The official MEME evaluator judged the answers with `gpt-4o`.

| Task | Correct | Share | What the task checks |
|---|---:|---:|---|
| All | 481 / 694 | 69.3% | |
| Deletion | 84 / 100 | 84.0% | The agent drops a fact the user retracted. |
| Absence | 103 / 130 | 79.2% | The agent says a fact was never stated. |
| Tracking | 78 / 100 | 78.0% | The agent reports the current value after many updates. |
| Cascade | 99 / 164 | 60.4% | A changed fact updates the facts that depend on it. |
| Aggregation | 18 / 100 | 18.0% | The agent combines several remembered items into one answer. |

### EnterpriseRAG-Bench

Finch retrieved and packed the context for each question. `gpt-5.6-luna` expanded the
queries, reranked the results, and wrote the answers.

| Metric | Score |
|---|---|
| Combined (headline) | 69.05 |
| Correct | 76.8% |
| Completeness | 76.0% |
| Recall | 69.9% |

## Related

[finch-db](https://github.com/jadenpxrk/finch-db) is an in-process vector database in Rust.

## Status

The crates have 405 tests. The memory tests run against a Postgres server with pgvector. All
tests use synthetic data. The repository contains no cached benchmark replays.

The project has not had a public release.

## License

AGPL-3.0-only. See `LICENSE`.

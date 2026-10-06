# Working on Finch

This file is for people and for coding agents. Read it before the first change.

## What is here

| Path | Crate | Depends on |
|---|---|---|
| `crates/finch-types` | schema, query, document, and error types | nothing in the workspace |
| `crates/finch-proto` | manifest encoding | finch-types |
| `crates/finch-storage` | columnar forward store | finch-types |
| `crates/finch-core` | vector index algorithms and quantization | finch-types |
| `crates/finch-db` | collections, segments, WAL, SQL-like filters | all of the above |
| `crates/finch-memory` | the agent memory layer | finch-db |
| `finch-py`, `finch-node` | Python and Node.js bindings | finch-db, finch-memory |

`docs/architecture.md` describes the database crates and the memory layer. Read it before you
change `finch-db`, `finch-core`, or `finch-memory`. The benchmark harness that produced the numbers in the README is not in this
repository.

## Commands

`.github/workflows/ci.yml` names the Rust toolchain version. Use the same version locally.

```bash
cargo test                                   # all default workspace members
cargo test -p finch-db                       # one crate
cargo test -p finch-db --test wal_crash_recovery_test   # one test binary
cargo test -p finch-db -- wal_replay          # tests whose name contains a word
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cargo run -q --example quickstart -p finch-db
```

The `finch-memory` tests keep their stores in Postgres with pgvector. Start a server and name it
in `FINCH_MEMORY_TEST_POSTGRES_URL`:

```bash
docker run -d --name finch-pg -p 55432:5432 -e POSTGRES_USER=finch -e POSTGRES_PASSWORD=finch \
  -e POSTGRES_DB=finch pgvector/pgvector:pg17
export FINCH_MEMORY_TEST_POSTGRES_URL=postgres://finch:finch@localhost:55432/finch
```

Code under `cfg(target_arch = "x86_64")` does not compile on an ARM machine. Before you touch
`crates/finch-core/src/simd`, run clippy for that target too:

```bash
cargo clippy -p finch-core --all-targets --target x86_64-unknown-linux-gnu -- -D warnings
```

## Rules, with the reason for each

- Extend an existing function instead of adding a variant of it. A new argument goes into the
  function's request struct. Reason: variants multiply when each change adds one instead of
  touching the shared function.
- A function does one thing at one level of abstraction. A reader can narrate it from the
  names it calls. Length is a signal, not a limit. A function over about 80 lines needs one
  sentence in the review that says why one piece is clearer than a split. Inline a helper
  that has one caller and no name from the domain. Reason: functions built by accretion hide
  bugs. Long functions that walk one algorithm are fine.
- Library code returns errors. It does not call `unwrap()` on input it did not create. Reason:
  a panic in library code turns bad input into a crashed process.
- Every `unsafe` block has a one-line `SAFETY:` comment that states the invariant it relies on.
  Reason: a reviewer checks this first, and writing the invariant down is how unsound code
  gets caught.
- Iteration order never reaches an output. Use `BTreeMap` or `BTreeSet`, or sort first.
  Reason: hash-order iteration makes results differ between two runs of the same code.
- `cargo clippy --workspace --all-targets -- -D warnings` passes. Fix the code. Do not add
  `#[allow]`. Reason: allow attributes hide dead code and lints that point at real bugs.
- A comment states a reason, in one line. It does not restate the code.
- Product crates never branch on a benchmark name, question id, label, or gold answer.

## Changes

- A bug fix includes a test that fails before the fix and passes after it.
- A behavior change to `finch-memory` includes a note in the pull request that says what a
  packed context looks like before and after.
- Commit subject in the imperative, under 72 characters. The body says why, not what.
- CI must be green: tests, clippy with warnings denied, the bindings build, and the Python
  tests.

Before you open a change:

```bash
cargo test
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
git diff --check
```

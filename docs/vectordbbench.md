# Finch and VectorDBBench

VectorDBBench (https://github.com/zilliztech/VectorDBBench) is a benchmark harness for
vector databases. It reaches Finch through the Python bindings. The Finch client, the
benchmark script, and the conformance and parity tests below live in a VectorDBBench
checkout, not in this repository. The commands assume that checkout sits next to this
repository in a directory named VectorDBBench.

## Requirements

- A stable Rust toolchain.
- Python 3.11 or later.
- `maturin` and `numpy` in the Python environment for the benchmark.
- The packages that VectorDBBench itself requires.

## Build the Python bindings

From the root of this repository:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip maturin
cd finch-py
python -m maturin develop --release
```

VectorDBBench imports the installed `finch` package. After a change to Rust code, run
`maturin develop` again before benchmarking. Nothing rebuilds the package automatically.

To build into the repository's .venv directory without activating it:

```bash
cd finch-py
VIRTUAL_ENV="$(pwd)/../.venv" \
  ../.venv/bin/python -m maturin develop -q --release
```

On x86_64, `RUSTFLAGS="-C target-cpu=native"` builds for the local CPU. Use it only when the
benchmark runs on the machine that built the package. See "Building for one machine" in
`README.md`.

## Smoke benchmark without downloads

From the VectorDBBench checkout:

```bash
python run_finch_bench.py --smoke --index hnsw
```

The script generates a small dataset locally and runs load, index build, and search.

The Finch client puts the requested index parameters in the collection schema when it
creates the collection. `optimize()` is therefore the step that builds and compacts indexes. To
change index type or parameters between runs, pass a different `--path` or delete the old
collection directory.

## HNSW build speed

HNSW neighbor selection during a build compares vectors on a prefix of their dimensions. By
default the prefix is the full dimension. Setting `FINCH_HNSW_HEURISTIC_DIM` to a smaller
number shortens the prefix. A shorter prefix makes builds of high-dimensional vectors faster,
for example vectors with 1,536 dimensions. A shorter prefix can also lower recall. Search
always uses full vectors.

```bash
export FINCH_HNSW_HEURISTIC_DIM=64
```

Finch reads the variable each time it creates an HNSW builder.

## Conformance tests

From the VectorDBBench checkout:

```bash
python -m unittest tests.test_finch_conformance -q
```

These tests check the Finch client and its SQL filter handling for flat, HNSW, and IVF
indexes. They need no other database.

The `vectordbbench-conformance` job in `.github/workflows/ci.yml` runs these tests against
VectorDBBench commit `f35f648c81da3e25de530675bd8133fa8159f777`. If a newer VectorDBBench
passes locally, update that commit in the workflow.

## Parity tests

The parity tests compare results from two installations of Finch-compatible packages, each
in its own Python interpreter. From the VectorDBBench checkout:

```bash
export FINCH_PYTHON="/abs/path/to/finch-venv/bin/python"
export REFERENCE_PYTHON="/abs/path/to/other-venv/bin/python"
python -m unittest tests.test_finch_parity -q
```

## Larger datasets

This case downloads its dataset first:

```bash
python run_finch_bench.py --index hnsw --case 50k_1536
```

## VectorDBBench command line

The `vectordbbench` command exists only in an active Python environment that has
VectorDBBench installed. For a checkout, run `python -m pip install -e .` in it first.

```bash
vectordbbench finch --path /tmp/finch_bench --db-label local
```

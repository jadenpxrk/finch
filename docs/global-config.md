# Finch Global Config

`GlobalConfigData` (`crates/finch-types/src/config.rs`) holds the settings that apply to every
collection in a process:

- thread counts.
- filter planning ratios.
- logging.
- how often Finch flushes and syncs the write-ahead log (WAL).
- a memory budget.

## Initialization

Call `initialize_global_config` (Rust), `finch.init` (Python), or `initGlobalConfig` (Node)
once, before opening or creating any collection.

- The first call validates the config and installs it. Later calls return success and
  change nothing. They do not validate their argument.
- If no code makes the call, Finch uses `GlobalConfigData::default()` the first time it needs
  a value.
- Finch creates the query and optimize thread pools when it creates or opens the first
  collection. After that, an init call fails with `InvalidArgument` if its thread counts
  differ from the pool sizes.

## Fields

| Field | Type | Default |
|---|---|---|
| `query_thread_count` | u32 | the cgroup CPU quota, else the number of available CPUs, else 4 |
| `optimize_thread_count` | u32 | same as `query_thread_count` |
| `invert_to_forward_scan_ratio` | f32 | 0.9 |
| `brute_force_by_keys_ratio` | f32 | 0.1 |
| `wal_flush_every_docs` | u32 | 1 |
| `wal_fsync_every_docs` | u32 | 0 |
| `memory_limit_bytes` | u64 | 80% of the cgroup memory limit or physical memory, else 100 MiB |
| `log_level` | `LogLevel` | `Warn` |
| `log_to_file` | bool | false |
| `log_dir` | string | `./logs` |
| `log_basename` | string | `finch.log` |
| `log_file_size_mb` | u32 | 2048 |
| `log_overdue_days` | u32 | 7 |

Both thread counts must be greater than 0. Both ratios must be in [0, 1].
`docs/architecture.md` explains how the planner uses the two ratios. `log_level` applies
only when the `RUST_LOG` environment variable is unset.

## Threads

`query_thread_count` and `optimize_thread_count` size two Rayon thread pools that Finch
creates for itself. The pools are `query_pool` and `optimize_pool` in
`crates/finch-db/src/config.rs`.
Other crates that configure Rayon's global pool do not change them.

Some work runs outside these two pools:

- A dense query over at most one persisted segment runs on the calling thread. In that
  query, a flat index scan of 8,192 or more vectors without a filter runs on Rayon's global
  pool.
- `query_int_ids` (Python `query_ids`) uses a two-thread pool when the call does not set a
  concurrency.
- A parallel HNSW build creates a pool sized by the call's concurrency setting.

## HNSW tuning

HNSW build heuristics are not global settings. They live in `HnswIndexParams::build_tuning`
and are stored with the index. The search-time knobs `hnsw_upper_ef` and `hnsw_l0_seeds` are
fields of `QueryParams`.

## Write-ahead log

- `wal_flush_every_docs`: after this many WAL records, Finch flushes its write buffer to the
  operating system. `0` turns this off.
- `wal_fsync_every_docs`: after this many WAL records, Finch flushes and calls
  `File::sync_all`, so the records up to that point survive a power loss. Up to N - 1 later
  records can still be lost. `0`, the default, turns this off, and then a power loss or an
  operating system crash can lose any write since the collection's last flush. Lower values
  make writes slower.

Each insert, upsert, update, or delete of one document writes one WAL record.

## Memory limit

`memory_limit_bytes` must be at least 104,857,600 (100 MiB) and no larger than the detected
cgroup or physical memory. Finch validates the value but does not enforce it anywhere.

## Python

```py
import finch

finch.init(
    query_threads=8,
    optimize_threads=8,
    invert_to_forward_scan_ratio=0.9,
    brute_force_by_keys_ratio=0.1,
    log_type=finch.LogType.FILE,
    log_dir="./logs",
    log_basename="finch.log",
    wal_flush_every_docs=1,
    wal_fsync_every_docs=100,
)
```

The Python function takes `memory_limit_mb` instead of `memory_limit_bytes`. Every integer
argument of the Python function must be greater than 0. The Python function therefore cannot
set `wal_fsync_every_docs` or `wal_flush_every_docs` to `0`. Leave an argument out to keep
its default.

## Node

```ts
import { initGlobalConfig } from "@finch/finch";

initGlobalConfig({
  queryThreadCount: 8,
  optimizeThreadCount: 8,
  invertToForwardScanRatio: 0.9,
  bruteForceByKeysRatio: 0.1,
  logToFile: true,
  logDir: "./logs",
  logBasename: "finch.log",
  walFlushEveryDocs: 1,
  walFsyncEveryDocs: 0,
});
```

The Node options use the Rust field names in camelCase, including `memoryLimitBytes`,
`logLevel` (0 to 4), `logFileSizeMb`, and `logOverdueDays`.

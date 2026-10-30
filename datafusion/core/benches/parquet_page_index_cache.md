<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Wide Parquet page-index cache benchmark

This portable Criterion benchmark exercises DataFusion SQL planning and scans of
one wide Parquet file. It compares the selective decoded-index prototype with
repair-only whole-index caching. It is not a general query-engine benchmark.

## Workloads

Three queries each run with cold index metadata, warm identical projection, and
warm overlapping projection:

- **prune_only:** sorted integer-ID band within one page, page pruning enabled,
  row-filter pushdown disabled.
- **prune_and_pushdown:** the same band plus a non-prunable modulo residual,
  pushed-down filtering enabled.
- **filter_only_control:** modulo residual without a page-prunable predicate.

Projection A reads four string columns; B shares two and adds two. The integer
predicate column is deliberately separate from projected string columns.
The fixture contains long varied string min/max bounds across many row groups
and pages, with actual page indexes, Snappy compression, and no dictionaries.
All results are checked against deterministic ID and string-value oracles.
Pruning/filtering counters must prove those paths executed.

Fixture generation, registration, footer seeding, cache reset, warm seeding,
validation and metric snapshots are outside timed regions. Timing includes SQL
planning, physical-plan creation, execution and collection. "Cold" means only
page-index metadata is cold: filesystem/OS, footer, listing and statistics are
warm. Overlap resets and seeds A before every measured B query so it remains a
miss/overlap case, not an identical hit after its first iteration.

## Run on another system

From the repository root:

```sh
mkdir -p results/page-index

# Smoke/correctness check; also generates a persistent fixture if absent.
PAGE_INDEX_BENCH_COLUMNS=256 \
PAGE_INDEX_BENCH_FILE="$PWD/results/page-index/wide-256.parquet" \
PAGE_INDEX_BENCH_REPORT="$PWD/results/page-index/smoke-256.json" \
  cargo bench --profile release-nonlto -p datafusion \
    --bench parquet_page_index_cache -- --test

# A measured run; use a NEW report destination on each invocation.
PAGE_INDEX_BENCH_COLUMNS=256 \
PAGE_INDEX_BENCH_FILE="$PWD/results/page-index/wide-256.parquet" \
PAGE_INDEX_BENCH_REPORT="$PWD/results/page-index/selective-256-a.json" \
  cargo bench --profile release-nonlto -p datafusion \
    --bench parquet_page_index_cache -- \
    --sample-size 40 --warm-up-time 1 --measurement-time 5 \
    --save-baseline selective-256-a
```

Repeat with 64 columns and a different fixture path. Prefer a quiet, dedicated
machine and run each variant at least twice, reversing run order. Use the same
profile, configuration, and physical fixture for both variants. No dataset or
machine-specific paths are hardcoded in the benchmark.

| Variable | Default | Meaning |
| --- | ---: | --- |
| `PAGE_INDEX_BENCH_COLUMNS` | 256 | String columns, in addition to `id` and `residual` |
| `PAGE_INDEX_BENCH_ROWS` | 32768 | Total rows |
| `PAGE_INDEX_BENCH_ROW_GROUP_ROWS` | 4096 | Rows per row group |
| `PAGE_INDEX_BENCH_PAGE_ROWS` | 64 | Rows per data page |
| `PAGE_INDEX_BENCH_VALUE_BYTES` | 128 | String payload and page-bound length |
| `PAGE_INDEX_BENCH_FILE` | temporary file | Persistent fixture to create/reuse |
| `PAGE_INDEX_BENCH_REPORT` | stdout only | Fresh JSON diagnostics destination |

Rows must divide into at least two whole row groups; row groups must divide into
at least four pages; page rows must be at least eight; string columns at least
six; value bytes at least 32. Existing fixtures must match all dimensions and
embedded format metadata. They are validated and never overwritten. Reports
require a new pathname, protecting fixtures even through hard-link aliases.
All dimensions appear in Criterion identifiers, avoiding baseline collisions.

## Build the repair-only comparison

The candidate is `c264d5653`; repair-only is `c9fb12bdd`. Both must use exactly
Arrow-rs #11475 pin `e1d49438407325f0ff0057ae9c141c3016a92921` so differences
are not attributed to Arrow or its writer. The benchmark uses public APIs shared
by both DataFusion revisions.

After committing the benchmark, record its commit as `BENCH_COMMIT`. For example:

```sh
BENCH_COMMIT=$(git rev-parse HEAD)
git worktree add --detach ../page-index-repair-bench c9fb12bdd

# Same Arrow dependencies and the pin's FixedSizeList compatibility adjustment.
git diff c9fb12bdd c264d5653 -- Cargo.toml Cargo.lock \
  datafusion/physical-plan/src/unnest.rs > /tmp/page-index-arrow-pin.patch
git -C ../page-index-repair-bench apply /tmp/page-index-arrow-pin.patch

# The same benchmark source and registration in the older implementation.
git show "$BENCH_COMMIT:datafusion/core/benches/parquet_page_index_cache.rs" \
  > ../page-index-repair-bench/datafusion/core/benches/parquet_page_index_cache.rs
git diff c264d5653 "$BENCH_COMMIT" -- datafusion/core/Cargo.toml \
  > /tmp/page-index-bench-registration.patch
git -C ../page-index-repair-bench apply /tmp/page-index-bench-registration.patch
```

Run there with the identical fixture/dimensions and `--save-baseline repair-256-a`.
Use a separate build target directory for each checkout. This avoids cross-worktree
Cargo fingerprint reuse; changing `CARGO_TARGET_DIR` also changes where Criterion
saves results. If separate target directories are used, copy their named Criterion
baseline directories under one comparison root, retaining group/case paths.

## Inspect results

Criterion stores latency estimates, confidence intervals and raw samples under
`$CARGO_TARGET_DIR/criterion` (or `target/criterion`). The JSON diagnostics report
contains query rows, SQL/configuration, cached metadata byte estimates, populated
column/offset index counts, bytes scanned, and execution/pruning counters.
It is one untimed observation per case, not a distribution of per-sample counters.

```sh
python3 datafusion/core/benches/compare_page_index_cache.py \
  results/page-index/repair-256-a.json \
  results/page-index/selective-256-a.json \
  --criterion-root target/criterion \
  --base-name repair-256-a --candidate-name selective-256-a \
  --output results/page-index/comparison-256-a.json
```

The comparison helper rejects mismatched dimensions, fixture size/index layout,
queries, result counts and pruning/filtering counters. It prints candidate/base
latency ratios and cache statistics; ratios are descriptive, not a significance
test. Criterion reports are the source for sampling uncertainty.

## Interpretation and limits

- Compare cold, identical and overlap cases separately; do not hide a warm
  regression behind a cold improvement or one suite-wide average.
- Cache metadata bytes are implementation estimates, not RSS or exact allocator
  bytes. The selective implementation charges custom-provider payloads; private
  spare capacities and allocator overhead remain outside those estimates.
- The 1 GiB cache budget deliberately permits the full baseline entry: this
  measures selective decode/sharing, not an eviction-heavy workload.
- Arrow currently reads covering ranges for selected indexes. Fewer decoded
  entries need not mean proportionally fewer network bytes.
- This is local storage without added latency. Object-store request/latency
  tradeoffs need separate remote testing.
- The filter-only control should not gain from an index optimization.
- The pinned Arrow dependency is prototype glue; an upstream benchmark/feature
  should use an appropriate released dependency.
- `ci-bench` (optimization level 1) is suitable for a quick pilot, not a substitute
  for production-profile measurements on a dedicated system.

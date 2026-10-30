#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

"""Compare two saved page-index reports and optional named Criterion baselines."""

import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def estimates(root, dimensions, case, baseline):
    group = (
        f"parquet_page_index_cache_{dimensions['columns']}cols_"
        f"{dimensions['rows']}rows_rg{dimensions['row_group_rows']}_"
        f"page{dimensions['page_rows']}_bytes{dimensions['value_bytes']}"
    )
    key = f"{case['query']}_{case['cache_state']}"
    estimate = load(Path(root) / group / key / baseline / "estimates.json")
    # Criterion uses slope for iter() and mean for iter_batched(). Keep its
    # own chosen estimator, not the printed/rounded console interval.
    return estimate.get("slope") or estimate["mean"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_report")
    parser.add_argument("candidate_report")
    parser.add_argument("--criterion-root", help="Directory containing Criterion groups")
    parser.add_argument("--base-name")
    parser.add_argument("--candidate-name")
    parser.add_argument("--output", help="Optional fresh JSON destination")
    args = parser.parse_args()
    if any((args.criterion_root, args.base_name, args.candidate_name)) and not all(
        (args.criterion_root, args.base_name, args.candidate_name)
    ):
        parser.error("Criterion comparison needs root and both baseline names")
    base, candidate = load(args.base_report), load(args.candidate_report)
    for key in ("dimensions", "fixture", "target_partitions", "metadata_cache_limit_bytes", "timing", "cold_scope"):
        if base[key] != candidate[key]:
            parser.error(f"Reports differ in {key}; do not compare different workloads")
    cases = {(case["query"], case["cache_state"]): case for case in candidate["cases"]}
    if set(cases) != {(case["query"], case["cache_state"]) for case in base["cases"]}:
        parser.error("Reports have different case sets")
    output = {"dimensions": base["dimensions"], "fixture": base["fixture"], "cases": []}
    print("case | base ms | candidate ms | candidate/base | metadata base→candidate | index entries base→candidate")
    for before in base["cases"]:
        key = before["query"], before["cache_state"]
        after = cases[key]
        for field in ("sql", "seed_sql", "pushdown_filters", "query_rows", "page_index_rows_pruned", "pushdown_rows_pruned"):
            if before[field] != after[field]:
                parser.error(f"Case {key} differs in {field}; verify correctness/methodology")
        row = {"query": key[0], "cache_state": key[1], "query_rows": before["query_rows"],
               "base_cache": before["cache_after"], "candidate_cache": after["cache_after"],
               "base_bytes_scanned": before["bytes_scanned"], "candidate_bytes_scanned": after["bytes_scanned"]}
        timing = "— | — | —"
        if args.criterion_root:
            left = estimates(args.criterion_root, base["dimensions"], before, args.base_name)
            right = estimates(args.criterion_root, candidate["dimensions"], after, args.candidate_name)
            row["base_estimate_ns"] = left
            row["candidate_estimate_ns"] = right
            row["candidate_base_ratio"] = right["point_estimate"] / left["point_estimate"]
            timing = f"{left['point_estimate']/1e6:.3f} | {right['point_estimate']/1e6:.3f} | {row['candidate_base_ratio']:.3f}"
        old, new = before["cache_after"], after["cache_after"]
        print(f"{key[0]}/{key[1]} | {timing} | {old['metadata_bytes']}→{new['metadata_bytes']} | "
              f"{old['column_index_entries']}+{old['offset_index_entries']}→"
              f"{new['column_index_entries']}+{new['offset_index_entries']}")
        output["cases"].append(row)
    if args.output:
        # Never truncate an existing report/fixture, including hard-link aliases.
        with open(args.output, "x") as file:
            json.dump(output, file, indent=2)
            file.write("\n")
    print("Ratios are descriptive paired runs, not a statistical significance test; lower is faster.")


if __name__ == "__main__":
    main()

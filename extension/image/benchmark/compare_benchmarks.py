#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compare two image_processor_benchmark result files.

Each input is the output of `image_processor_benchmark --out=PATH` (or its
stdout). Rows are matched by (API section, input->target cell, variant label)
and the per-row speedup base/new is reported.

The summary buckets rows by execution path (CPU / GPU / default). Cross-run and
thermal drift shift all rows together, so compare the buckets against each other
rather than reading any single ratio absolutely.

Usage:
  compare_benchmarks.py BASE.txt NEW.txt [--metric=median|mean]
"""

import argparse
import re
import statistics
import sys

ROW_RE = re.compile(
    r"^(?P<label>.*?)\s+mean=\s*(?P<mean>[\d.]+) ms\s+"
    r"median=\s*(?P<median>[\d.]+) ms"
)
CELL_RE = re.compile(r"^\[(?P<cell>.+?)\]\s*$")


def path_bucket(label):
    """Bucket a variant by execution path for the summary, or None to skip."""
    if "GPU" in label:
        return "GPU"
    if "def" in label:
        return "Default"
    if "CPU" in label:
        return "CPU"
    return None


def parse(path, metric):
    """Return {(section, cell, label): value} for the chosen metric."""
    rows = {}
    section = None
    cell = None
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if "ImageProcessor::process_yuv_into" in stripped:
                section = "process_yuv_into"
                continue
            if "ImageProcessor::process_into" in stripped:
                section = "process_into"
                continue
            cell_m = CELL_RE.match(stripped)
            if cell_m and "->" in stripped:
                cell = cell_m.group("cell")
                continue
            row_m = ROW_RE.match(line)
            if row_m:
                key = (section, cell, row_m.group("label").strip())
                rows[key] = float(row_m.group(metric))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base", help="baseline results file")
    ap.add_argument("new", help="new results file")
    ap.add_argument("--metric", choices=["median", "mean"], default="median")
    args = ap.parse_args()

    base = parse(args.base, args.metric)
    new = parse(args.new, args.metric)

    keys = [k for k in base if k in new]
    if not keys:
        print("no matching rows between the two files", file=sys.stderr)
        return 1
    only = set(base) ^ set(new)
    if only:
        print(f"note: {len(only)} row(s) present in only one file (ignored)\n")

    buckets = {"CPU": [], "GPU": [], "Default": []}
    for section in ("process_into", "process_yuv_into"):
        sect_keys = [k for k in keys if k[0] == section]
        if not sect_keys:
            continue
        print(f"=== {section} ({args.metric}, speedup = base / new) ===")
        print(f"{'cell':<26}{'variant':<24}{'base':>9}{'new':>9}{'speedup':>9}")
        print("-" * 77)
        for k in sect_keys:
            _, cell, label = k
            b, n = base[k], new[k]
            sp = b / n if n else float("nan")
            bucket = path_bucket(label)
            if bucket is not None:
                buckets[bucket].append(sp)
            print(f"{cell:<26}{label:<24}{b:>9.3f}{n:>9.3f}{sp:>8.2f}x")
        print()

    def summary(name, xs):
        if not xs:
            return
        print(
            f"{name:<14} n={len(xs):<4} "
            f"median={statistics.median(xs):.2f}x  "
            f"min={min(xs):.2f}x  max={max(xs):.2f}x"
        )

    print("=== summary (speedup = base / new, by execution path) ===")
    for name in ("CPU", "GPU", "Default"):
        summary(f"{name} rows", buckets[name])
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Measure size of an exported Gemma 4 .pte and report a per-method breakdown.

Helps verify quantization wins and catch unintentional size regressions.
Output is a markdown table that can be pasted into TEST_RESULTS.md.

Usage:
    python -m executorch.examples.models.gemma4.tools.measure_pte \
        /tmp/gemma4_multimodal_v11.pte \
        /tmp/gemma4_multimodal_v11_q.pte
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List


def _format_size(bytes_: int) -> str:
    if bytes_ >= 1 << 30:
        return f"{bytes_ / (1 << 30):.2f} GB"
    if bytes_ >= 1 << 20:
        return f"{bytes_ / (1 << 20):.1f} MB"
    if bytes_ >= 1 << 10:
        return f"{bytes_ / (1 << 10):.1f} KB"
    return f"{bytes_} B"


# Method names that are constant-value metadata (`get_max_seq_len` etc.) — not
# real compute methods. We hide these from the per-method listing.
_METADATA_METHOD_PREFIXES = ("get_", "use_", "enable_", "n_", "num_")


def _real_method_names(pte_path: Path) -> List[str]:
    """Return the list of compute methods in a pte, excluding constant metadata.

    Returns an empty list if the runtime can't introspect the pte.
    """
    try:
        from executorch.runtime import Runtime, Verification
    except ImportError:
        return []
    try:
        runtime = Runtime.get()
        program = runtime.load_program(
            str(pte_path), verification=Verification.Minimal
        )
        names = list(program.method_names)
        return sorted(
            n for n in names
            if not any(n.startswith(p) for p in _METADATA_METHOD_PREFIXES)
        )
    except Exception:
        return []


def measure(paths: List[Path]) -> None:
    print()
    print("| .pte | size | delta vs first | methods |")
    print("|---|---|---|---|")
    baseline_size: int | None = None
    for p in paths:
        if not p.exists():
            print(f"| {p.name} | (missing) | — | — |")
            continue
        size = p.stat().st_size
        size_str = _format_size(size)
        if baseline_size is None:
            baseline_size = size
            delta = "—"
        else:
            pct = 100.0 * (size - baseline_size) / baseline_size
            delta = f"{pct:+.1f}%"
        names = _real_method_names(p)
        method_list = ", ".join(names) if names else "(unknown)"
        print(f"| {p.name} | {size_str} | {delta} | {method_list} |")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ptes", nargs="+", type=Path,
                        help="One or more .pte files to measure. The first "
                             "is treated as the baseline for percent-delta.")
    args = parser.parse_args()
    measure(args.ptes)


if __name__ == "__main__":
    main()

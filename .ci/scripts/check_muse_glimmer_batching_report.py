# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checks the reports Muse Glimmer's run_solo_batching writes, for the
solo-text-batching CI mode (test_model_e2e.sh).

Three things must hold:

1. Every generation stands on its own: it finishes normally, produces text,
   and contains its expected answer; and the single-prompt runs with and
   without the captured decode graph generate the same tokens. Tokens are not
   compared across batches: a prompt prefilled in a wider forward runs GEMMs of
   another shape, whose bf16 rounding can steer greedy decoding elsewhere.
2. The model's weights are loaded once: loading costs about the weights, not
   one copy per method, and does not grow with the sessions reserved for.
3. GPU memory behaves like an off-graph KV cache: the pool holds what the
   sessions used, grows geometrically rather than reserving every session's
   context up front, and accounts for what generation added.

    python .ci/scripts/check_muse_glimmer_batching_report.py \\
        --single eager.json --single graph.json --batch batch.json \\
        --expect Paris Tokyo Rome Paris Berlin
"""

import argparse
import json
import sys
from typing import Dict, List, Optional, Sequence

MIB = 1024 * 1024
OK_REASONS = ("stop_token", "token_limit")


class ReportError(AssertionError):
    pass


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ReportError(message)


def check_generations(report: Dict, expect: Sequence[str], label: str) -> None:
    generations = report["generations"]
    _check(
        len(generations) == len(expect),
        f"{label}: {len(generations)} generations, expected {len(expect)}",
    )
    for index, (generation, answer) in enumerate(zip(generations, expect)):
        name = f"{label}[{index}]"
        _check(
            generation["finish_reason"] in OK_REASONS,
            f"{name}: finished {generation['finish_reason']}: {generation['error']}",
        )
        _check(len(generation["tokens"]) > 0, f"{name}: generated nothing")
        _check(
            answer.lower() in generation["text"].lower(),
            f"{name}: expected '{answer}' in {generation['text']!r}",
        )


def check_same_tokens(first: Dict, second: Dict, message: str) -> None:
    _check(first["tokens"] == second["tokens"], message)


def load_cost(report: Dict) -> int:
    gpu = report["gpu"]
    return gpu["used_after_load_bytes"] - gpu["used_before_load_bytes"]


def check_weights_once(
    reports: Sequence[Dict], max_weight_ratio: float, max_load_spread_mib: float
) -> None:
    for index, report in enumerate(reports):
        weights = report["weights_bytes"]
        cost = load_cost(report)
        _check(weights > 0, f"report {index}: no weights size")
        _check(
            cost <= max_weight_ratio * weights,
            f"report {index}: loading took {cost / MIB:.0f} MiB for "
            f"{weights / MIB:.0f} MiB of weights (limit {max_weight_ratio}x): "
            "are they loaded once per method?",
        )
    costs = [load_cost(report) for report in reports]
    _check(
        max(costs) - min(costs) <= max_load_spread_mib * MIB,
        f"loading cost varies by {(max(costs) - min(costs)) / MIB:.0f} MiB "
        f"across runs reserving different session counts (limit "
        f"{max_load_spread_mib} MiB)",
    )


def check_kv(report: Dict, max_generate_slack_mib: float, label: str) -> Dict:
    kv = report["kv"]
    config = report["config"]
    rows = kv["rows"]
    used = kv["cells_in_use"]
    _check(rows >= used, f"{label}: {rows} rows hold {used} cells in use")
    _check(
        kv["allocated_bytes"] == rows * kv["bytes_per_cell"],
        f"{label}: {kv['allocated_bytes']} bytes allocated for {rows} rows of "
        f"{kv['bytes_per_cell']}",
    )
    # Geometric growth: never more than twice what the widest moment needed.
    _check(
        rows <= max(kv["initial_capacity"], 2 * (used + config["step_width"])),
        f"{label}: {rows} rows for {used} cells in use: the pool is not "
        "growing on demand",
    )
    reserved = config["max_sessions"] * config["max_session_tokens"]
    _check(
        rows < reserved,
        f"{label}: {rows} rows is every session's full context ({reserved})",
    )
    gpu = report["gpu"]
    added = gpu["used_after_generate_bytes"] - gpu["used_after_load_bytes"]
    _check(
        added <= kv["allocated_bytes"] + max_generate_slack_mib * MIB,
        f"{label}: generation added {added / MIB:.0f} MiB, the KV pool is "
        f"{kv['allocated_bytes'] / MIB:.0f} MiB (slack {max_generate_slack_mib} "
        "MiB)",
    )
    return {
        "rows": rows,
        "cells_in_use": used,
        "kv_mib": kv["allocated_bytes"] / MIB,
        "in_graph_mib": reserved * kv["bytes_per_cell"] / MIB,
        "generate_added_mib": added / MIB,
    }


def check_reports(
    singles: Sequence[Dict],
    batch: Dict,
    expect: Sequence[str],
    single_expect: Optional[str] = None,
    max_weight_ratio: float = 1.25,
    max_load_spread_mib: float = 256,
    max_generate_slack_mib: float = 3072,
) -> List[str]:
    """Raises ReportError on the first violation; returns a summary."""
    single_expect = single_expect or expect[0]
    for index, report in enumerate(singles):
        check_generations(report, [single_expect], f"single{index}")
    for first, second in zip(singles, singles[1:]):
        check_same_tokens(
            first["generations"][0],
            second["generations"][0],
            "single-prompt runs disagree: the captured decode graph generated "
            "different tokens from eager decode",
        )
    check_generations(batch, expect, "batch")
    check_weights_once(list(singles) + [batch], max_weight_ratio, max_load_spread_mib)
    summary = []
    for label, report in [(f"single{i}", r) for i, r in enumerate(singles)] + [
        ("batch", batch)
    ]:
        stats = check_kv(report, max_generate_slack_mib, label)
        summary.append(
            f"{label}: KV {stats['rows']} rows for {stats['cells_in_use']} cells, "
            f"{stats['kv_mib']:.0f} MiB (in-graph reservation "
            f"{stats['in_graph_mib']:.0f} MiB); load "
            f"{load_cost(report) / MIB:.0f} MiB for "
            f"{report['weights_bytes'] / MIB:.0f} MiB of weights; generation "
            f"added {stats['generate_added_mib']:.0f} MiB"
        )
    engine = batch["engine"]
    summary.append(
        f"batch: {engine['steps']} forwards for {engine['decode_tokens_total']} "
        f"decode and {engine['prefill_tokens_total']} prefill tokens"
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--single", action="append", default=[], required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--expect", nargs="+", required=True)
    parser.add_argument("--max-weight-ratio", type=float, default=1.25)
    parser.add_argument("--max-load-spread-mib", type=float, default=256)
    parser.add_argument("--max-generate-slack-mib", type=float, default=3072)
    args = parser.parse_args(argv)

    def load(path):
        with open(path) as f:
            return json.load(f)

    try:
        summary = check_reports(
            [load(path) for path in args.single],
            load(args.batch),
            args.expect,
            max_weight_ratio=args.max_weight_ratio,
            max_load_spread_mib=args.max_load_spread_mib,
            max_generate_slack_mib=args.max_generate_slack_mib,
        )
    except ReportError as error:
        print(f"FAIL: {error}")
        return 1
    for line in summary:
        print(line)
    print("Success: batching reports check out")
    return 0


if __name__ == "__main__":
    sys.exit(main())

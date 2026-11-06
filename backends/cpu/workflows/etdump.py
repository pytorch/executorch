# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import statistics
import subprocess
from pathlib import Path

from executorch.devtools.inspector import Inspector

logger: logging.Logger = logging.getLogger(__name__)
PLAN_CHUNK = re.compile(r"CPU_PLAN_CHUNK (\d+) (\d+) ([0-9a-f]+)")


def decode_plans(log: str) -> list[dict]:
    plans = []
    chunks = []
    total = 0
    for match in PLAN_CHUNK.finditer(log):
        index, count = int(match[1]), int(match[2])
        if index != len(chunks) or (chunks and count != total):
            raise RuntimeError("Runner log has an incomplete CPU plan trace")
        total = count
        chunks.append(bytes.fromhex(match[3]))
        if len(chunks) == total:
            plans.append(json.loads(b"".join(chunks)))
            chunks = []
    if chunks or not plans:
        raise RuntimeError("Runner log has no complete CPU plan trace")
    return plans


def step_name(region: dict) -> str:
    nodes = region["nodes"]
    name = f"{region['provider']}/{region['implementation']}:{nodes[0]}"
    return name + (f"+{len(nodes) - 1}" if len(nodes) > 1 else "")


def run(args: argparse.Namespace) -> tuple[Path, list[dict]]:
    etdump = args.output / "model.etdump"
    env = dict(
        os.environ,
        ET_CPU_ETDUMP=str(etdump),
        ET_CPU_WARMUP_RUNS=str(args.warmup),
        ET_CPU_TIMED_RUNS=str(args.iterations),
    )
    result = subprocess.run(
        [args.runner, args.model, args.golden, *args.preferences],
        env=env,
        capture_output=True,
        text=True,
    )
    log = result.stdout + result.stderr
    (args.output / "runner.log").write_text(log)
    if result.returncode:
        raise RuntimeError(f"Runner failed; see {args.output / 'runner.log'}")
    return etdump, decode_plans(log)


def shapes(event) -> dict | None:
    raw = next((m for m in event.raw_delegate_debug_metadatas if m), None)
    if not raw:
        return None
    try:
        return json.loads(bytes(raw))
    except ValueError:
        logger.warning("Ignoring non-JSON delegate metadata on %s", event.name)
        return None


def summarize(etdump: Path, plans: list[dict], warmup: int, iterations: int) -> dict:
    regions = [region for plan in plans for region in plan["regions"]]
    steps = [step_name(region) for region in regions]
    step_names = set(steps)
    rows = []
    for block in Inspector(etdump_path=str(etdump)).event_blocks:
        for event in block.events:
            if event.perf_data is None or not event.perf_data.raw:
                continue
            samples = event.perf_data.raw
            if len(samples) == warmup + iterations:
                samples = samples[warmup:]
            if len(samples) != iterations:
                if event.is_delegated_op:
                    logger.warning(
                        "%s has %d samples, expected %d or %d; reporting it as init",
                        event.name,
                        len(samples),
                        iterations,
                        warmup + iterations,
                    )
                kind = "init"
            elif not event.is_delegated_op:
                kind = "runtime"
            elif event.name in step_names:
                kind = "step"
            else:
                kind = "provider"
            rows.append(
                {
                    "name": event.name,
                    "kind": kind,
                    "samples": len(samples),
                    "median_ms": statistics.median(samples),
                    "min_ms": min(samples),
                    "max_ms": max(samples),
                    "shapes": shapes(event),
                }
            )
    missing = step_names - {row["name"] for row in rows}
    if missing:
        raise RuntimeError(f"ETDump lacks CPU plan steps: {sorted(missing)}")
    if any(region["provider"] == "XNNPACK" for region in regions) and not any(
        row["kind"] == "provider" for row in rows
    ):
        raise RuntimeError("ETDump lacks XNNPACK operator events")
    return {
        "steps": steps,
        "totals_ms": {
            kind: sum(row["median_ms"] for row in rows if row["kind"] == kind)
            for kind in ("step", "provider")
        },
        "events": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a CPU delegate model with ETDump and summarize latency."
    )
    parser.add_argument("--runner", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--golden", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "preferences", nargs="*", help="provider[/implementation] preferences"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.output.mkdir(parents=True, exist_ok=True)
    etdump, plans = run(args)
    profile = summarize(etdump, plans, args.warmup, args.iterations)
    (args.output / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    for row in profile["events"]:
        logger.info(
            "%10.4f ms  %-8s  %s%s",
            row["median_ms"],
            row["kind"],
            row["name"],
            f"  {row['shapes']}" if row["shapes"] else "",
        )
    logger.info(
        "%d CPU steps, step total %.4f ms, provider operator total %.4f ms",
        len(profile["steps"]),
        profile["totals_ms"]["step"],
        profile["totals_ms"]["provider"],
    )


if __name__ == "__main__":
    main()

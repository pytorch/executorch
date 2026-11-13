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
        raise RuntimeError(
            "Runner log has no complete CPU plan trace; the runner must log at Info"
        )
    return plans


def plan_steps(plan: dict) -> list[str]:
    steps = [region["trace_name"] for region in plan["regions"]]
    if not all(steps):
        raise RuntimeError("CPU plan has no step trace names; enable the event tracer")
    return steps


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


def event_row(event, kind: str, plan: int | None, samples: list[float]) -> dict:
    return {
        "name": event.name,
        "kind": kind,
        "plan": plan,
        "samples": len(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "shapes": shapes(event),
    }


def read_events(
    etdump: Path, warmup: int, iterations: int
) -> tuple[list[dict], list[tuple]]:
    """Returns init rows and (event, timed samples) pairs for executed events."""
    init = []
    executed = []
    for block in Inspector(etdump_path=str(etdump)).event_blocks:
        for event in block.events:
            if event.perf_data is None or not event.perf_data.raw:
                continue
            samples = event.perf_data.raw
            if block.name != "Execute":
                init.append(event_row(event, "init", None, samples))
            elif len(samples) != warmup + iterations:
                raise RuntimeError(
                    f"{event.name} has {len(samples)} samples, "
                    f"expected {warmup + iterations}"
                )
            else:
                executed.append((event, samples[warmup:]))
    return init, executed


def assign_plans(executed: list[tuple], steps: list[list[str]]) -> dict[int, int]:
    """Maps each CPU delegate call's instruction id to the plan whose steps it ran."""
    delegates = {}
    for event, _ in executed:
        if not event.is_delegated_op or event._instruction_id in delegates:
            continue
        for index, names in enumerate(steps):
            if event.name in names and index not in delegates.values():
                delegates[event._instruction_id] = index
                break
    return delegates


def summarize(etdump: Path, plans: list[dict], warmup: int, iterations: int) -> dict:
    steps = [plan_steps(plan) for plan in plans]
    rows, executed = read_events(etdump, warmup, iterations)
    delegates = assign_plans(executed, steps)
    xnnpack_events = set()
    for event, samples in executed:
        plan = delegates.get(event._instruction_id)
        # Method-level events use the unset instruction id 0.
        if plan is None or event.name in ("DELEGATE_CALL", "Method::execute"):
            plan, kind = None, "runtime"
        elif event.is_delegated_op and event.name in steps[plan]:
            kind = "step"
        else:
            kind = "provider"
            if event.is_delegated_op:
                xnnpack_events.add(plan)
        rows.append(event_row(event, kind, plan, samples))
    for index, plan in enumerate(plans):
        missing = set(steps[index]) - {
            row["name"]
            for row in rows
            if row["plan"] == index and row["kind"] == "step"
        }
        if missing:
            raise RuntimeError(
                f"ETDump lacks CPU plan {index} steps: {sorted(missing)}"
            )
        if index not in xnnpack_events and any(
            region["provider"] == "XNNPACK" for region in plan["regions"]
        ):
            raise RuntimeError(f"ETDump lacks CPU plan {index} XNNPACK operator events")
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
    if args.warmup < 0 or args.iterations < 1:
        parser.error("--warmup must be >= 0 and --iterations >= 1")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.output.mkdir(parents=True, exist_ok=True)
    etdump, plans = run(args)
    profile = summarize(etdump, plans, args.warmup, args.iterations)
    (args.output / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    for event in profile["events"]:
        logger.info(
            "%10.4f ms  %-8s  %s%s%s",
            event["median_ms"],
            event["kind"],
            "" if event["plan"] is None else f"[{event['plan']}] ",
            event["name"],
            f"  {event['shapes']}" if event["shapes"] else "",
        )
    logger.info(
        "%d CPU steps, step total %.4f ms, provider operator total %.4f ms",
        sum(len(steps) for steps in profile["steps"]),
        profile["totals_ms"]["step"],
        profile["totals_ms"]["provider"],
    )


if __name__ == "__main__":
    main()

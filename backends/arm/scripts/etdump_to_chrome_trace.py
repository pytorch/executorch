#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# The script reads profiling events from an ETDump file using the ExecuTorch
# Inspector API, optionally enriches them with ETRecord metadata, and writes a
# JSON trace that can be loaded in chrome://tracing or Perfetto. Each ExecuTorch
# event block is represented as a Chrome trace thread, and each profiling sample
# is emitted as a complete-duration event with timestamps and durations in
# microseconds.
#
# Example:
#   python backends/arm/scripts/etdump_to_chrome_trace.py \
#     --etdump_path ./etdumps/vgf_timestamps.etdp \
#     --output ./traces/vgf_timestamps_trace.json

import argparse
import json
import math

from executorch.devtools import Inspector
from executorch.devtools.inspector import TimeScale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--etdump_path", required=True)
    parser.add_argument("--etrecord_path", required=False, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--source_time_scale",
        default="ns",
        choices=[ts.value for ts in TimeScale],
    )
    parser.add_argument(
        "--clock_frequency_hz",
        type=float,
        help="Clock frequency used for timestamps when --source_time_scale=cycles.",
    )
    args = parser.parse_args()

    cycle_timestamps = args.source_time_scale == "cycles"
    if cycle_timestamps and (
        args.clock_frequency_hz is None
        or not math.isfinite(args.clock_frequency_hz)
        or args.clock_frequency_hz <= 0
    ):
        parser.error("Cycle timestamps require a positive finite --clock_frequency_hz")

    inspector = Inspector(
        etdump_path=args.etdump_path,
        etrecord=args.etrecord_path,
        source_time_scale=TimeScale(args.source_time_scale),
        target_time_scale=TimeScale.CYCLES if cycle_timestamps else TimeScale.US,
    )

    trace_events = []

    # Chrome trace uses microseconds for "ts" and "dur".
    source_to_us = {
        "ns": 1.0 / 1000.0,
        "us": 1.0,
        "ms": 1000.0,
        "s": 1000_000.0,
        "cycles": 1_000_000.0 / args.clock_frequency_hz if cycle_timestamps else 1.0,
    }[args.source_time_scale]

    duration_to_us = source_to_us if cycle_timestamps else 1.0

    for block_idx, event_block in enumerate(inspector.event_blocks):
        tid_name = event_block.name

        trace_events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": 1,
                "tid": block_idx,
                "args": {"name": tid_name},
            }
        )

        for event in event_block.events:
            if event.perf_data is None or event.start_time is None:
                continue

            durations = event.perf_data.raw
            start_times = event.start_time

            for iter_idx, (start_time, duration) in enumerate(
                zip(start_times, durations)
            ):
                trace_events.append(
                    {
                        "name": event.name,
                        "cat": event_block.name,
                        "ph": "X",
                        "ts": float(start_time) * source_to_us,
                        "dur": float(duration) * duration_to_us,
                        "pid": 1,
                        "tid": block_idx,
                        "args": {
                            "event_block": event_block.name,
                            "iteration": iter_idx,
                            "is_delegated_op": event.is_delegated_op,
                            "delegate_backend_name": event.delegate_backend_name,
                            "op_types": event.op_types,
                        },
                    }
                )

    with open(args.output, "w") as f:
        json.dump({"traceEvents": trace_events}, f)

    print(f"Wrote Chrome trace JSON: {args.output}")
    print(f"Events: {len(trace_events)}")


if __name__ == "__main__":
    main()

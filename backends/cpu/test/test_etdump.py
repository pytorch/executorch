# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from executorch.backends.cpu.workflows import etdump
from executorch.devtools.etdump import schema_flatcc as flatcc
from executorch.devtools.etdump.serialize import (
    deserialize_from_etdump_flatcc,
    serialize_to_etdump_flatcc,
)
from executorch.devtools.inspector import _inspector

XNNPACK_STEP = "XNNPACK/subgraph:conv+1"
REFERENCE_STEP = "Reference/registry:gelu"
XNNPACK_PLAN = {
    "regions": [
        {
            "provider": "XNNPACK",
            "implementation": "subgraph",
            "trace_name": XNNPACK_STEP,
            "nodes": ["conv", "norm"],
        }
    ]
}
REFERENCE_PLAN = {
    "regions": [
        {
            "provider": "Reference",
            "implementation": "registry",
            "trace_name": REFERENCE_STEP,
            "nodes": ["gelu"],
        }
    ]
}
SHAPES = {"inputs": [[1, 3]], "outputs": [[1, 3]]}


def chunk_lines(plan: dict) -> list[str]:
    payload = json.dumps(plan).encode()
    chunks = [payload[start : start + 16] for start in range(0, len(payload), 16)]
    return [
        f"I CPUBackend.cpp:259] CPU_PLAN_CHUNK {index} {len(chunks)} {chunk.hex()}"
        for index, chunk in enumerate(chunks)
    ]


def event(name, instruction, ms, delegated=False, metadata=None, chain=0):
    return flatcc.Event(
        profile_event=flatcc.ProfileEvent(
            name="" if delegated else name,
            chain_index=chain,
            instruction_id=instruction,
            delegate_debug_id_int=-1,
            delegate_debug_id_str=name if delegated else "",
            delegate_debug_metadata=metadata,
            start_time=0,
            end_time=int(ms * 1e6),
        ),
        allocation_event=None,
        debug_event=None,
    )


def run_data(name, events):
    return flatcc.RunData(
        name=name, bundled_input_index=-1, allocators=[], events=events
    )


def forward(scale, xnnpack_ops=True):
    """One execution: an XNNPACK delegate at instruction 0, a Reference one at 1."""
    events = [
        event(XNNPACK_STEP, 0, 4 * scale, delegated=True),
        event(
            "Convolution #1",
            0,
            3 * scale,
            delegated=True,
            metadata=json.dumps(SHAPES).encode(),
        ),
        event("DELEGATE_CALL", 0, 5 * scale),
        event("native_call_gelu.out", 1, 1 * scale),
        event(REFERENCE_STEP, 1, 2 * scale, delegated=True),
        event("DELEGATE_CALL", 1, 3 * scale),
        # Method-level events carry the unset chain -1 and instruction id 0.
        event("Method::execute", 0, 9 * scale, chain=-1),
    ]
    return events if xnnpack_ops else events[:1] + events[2:]


class ETDumpTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "model.etdump"

    def summarize(self, runs, plans=(XNNPACK_PLAN, REFERENCE_PLAN), iterations=2):
        init = run_data(
            "Default",
            [
                event("Method::init", 0, 7, chain=-1),
                event("Program::load_method", 0, 8, chain=-1),
            ],
        )
        executions = [run_data("Execute", events) for events in runs]
        data = serialize_to_etdump_flatcc(
            flatcc.ETDumpFlatCC(version=0, run_data=[init, *executions])
        )
        # The runtime writes size-prefixed ETDumps; the Python serializer does not.
        dump = deserialize_from_etdump_flatcc(data, size_prefixed=False)
        with patch.object(_inspector, "gen_etdump_object", return_value=dump):
            return etdump.summarize(self.path, list(plans), 1, iterations)

    def test_decode_plans_reassembles_each_delegate(self):
        log = "\n".join(
            chunk_lines(XNNPACK_PLAN) + ["unrelated"] + chunk_lines(REFERENCE_PLAN)
        )
        self.assertEqual(etdump.decode_plans(log), [XNNPACK_PLAN, REFERENCE_PLAN])

    def test_decode_plans_rejects_incomplete_or_inconsistent_traces(self):
        lines = chunk_lines(XNNPACK_PLAN)
        total = f" {len(lines)} "
        mixed = lines[:1] + [
            line.replace(total, f" {len(lines) + 1} ") for line in lines[1:]
        ]
        for invalid in (lines[:-1], lines[1:], list(reversed(lines)), mixed, []):
            with self.subTest(invalid=invalid), self.assertRaises(RuntimeError):
                etdump.decode_plans("\n".join(invalid))

    def test_summarize_attributes_events_to_delegates_and_drops_warmup(self):
        profile = self.summarize([forward(9), forward(1), forward(2)])
        rows = {(row["plan"], row["name"]): row for row in profile["events"]}
        kinds = {key: row["kind"] for key, row in rows.items()}
        self.assertEqual(
            kinds,
            {
                (None, "Method::init"): "init",
                (None, "Program::load_method"): "init",
                (0, XNNPACK_STEP): "step",
                (0, "Convolution #1"): "provider",
                (None, "DELEGATE_CALL"): "runtime",
                (1, "native_call_gelu.out"): "provider",
                (1, REFERENCE_STEP): "step",
                (None, "Method::execute"): "runtime",
            },
        )
        self.assertEqual(rows[(0, XNNPACK_STEP)]["median_ms"], 6.0)
        self.assertEqual(rows[(0, "Convolution #1")]["shapes"], SHAPES)
        self.assertEqual(profile["steps"], [[XNNPACK_STEP], [REFERENCE_STEP]])
        self.assertEqual(profile["totals_ms"], {"step": 9.0, "provider": 6.0})

    def test_summarize_classifies_init_by_block_with_one_iteration(self):
        profile = self.summarize([forward(9), forward(1)], iterations=1)
        kinds = {row["name"]: row["kind"] for row in profile["events"]}
        self.assertEqual(kinds["Method::init"], "init")
        self.assertEqual(kinds[XNNPACK_STEP], "step")

    def test_summarize_rejects_unexpected_sample_counts(self):
        with self.assertRaisesRegex(RuntimeError, "has 2 samples, expected 3"):
            self.summarize([forward(9), forward(1)])

    def test_summarize_rejects_missing_steps_per_plan(self):
        runs = [
            [e for e in forward(s) if e.profile_event.instruction_id == 0]
            for s in (9, 1, 2)
        ]
        with self.assertRaisesRegex(RuntimeError, r"plan 1 steps: \['Reference"):
            self.summarize(runs)

    def test_summarize_rejects_missing_xnnpack_operator_events(self):
        runs = [forward(scale, xnnpack_ops=False) for scale in (9, 1, 2)]
        with self.assertRaisesRegex(RuntimeError, "plan 0 XNNPACK operator events"):
            self.summarize(runs)

    def test_summarize_rejects_plans_without_trace_names(self):
        untraced = {"regions": [{**XNNPACK_PLAN["regions"][0], "trace_name": ""}]}
        with self.assertRaisesRegex(RuntimeError, "no step trace names"):
            self.summarize([forward(1)] * 3, plans=[untraced])

    def test_summarize_ignores_malformed_metadata(self):
        runs = [forward(scale) for scale in (9, 1, 2)]
        for events in runs:
            events[1].profile_event.delegate_debug_metadata = b"not json"
        with self.assertLogs(etdump.logger, "WARNING"):
            profile = self.summarize(runs)
        rows = {row["name"]: row for row in profile["events"]}
        self.assertIsNone(rows["Convolution #1"]["shapes"])

    def test_run_passes_counts_and_decodes_the_runner_log(self):
        args = Namespace(
            runner="runner",
            model="model.pte",
            golden="golden.bin",
            preferences=["XNNPACK"],
            output=self.path.parent,
            warmup=2,
            iterations=5,
        )
        result = subprocess.CompletedProcess(
            [], 0, stdout="\n".join(chunk_lines(XNNPACK_PLAN)), stderr=""
        )
        with patch.object(etdump.subprocess, "run", return_value=result) as run:
            path, plans = etdump.run(args)
        command, env = run.call_args.args[0], run.call_args.kwargs["env"]
        self.assertEqual(command, ["runner", "model.pte", "golden.bin", "XNNPACK"])
        self.assertEqual(env["ET_CPU_ETDUMP"], str(path))
        self.assertEqual(env["ET_CPU_WARMUP_RUNS"], "2")
        self.assertEqual(env["ET_CPU_TIMED_RUNS"], "5")
        self.assertEqual(plans, [XNNPACK_PLAN])
        self.assertTrue((self.path.parent / "runner.log").exists())

    def test_main_rejects_invalid_counts(self):
        for counts in (["--warmup", "-1"], ["--iterations", "0"]):
            argv = ["etdump_cpu", "--runner", "r", "--model", "m", "--golden", "g"]
            argv += ["--output", str(self.path.parent), *counts]
            with self.subTest(counts=counts), patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit), patch("sys.stderr"):
                    etdump.main()

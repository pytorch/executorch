# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from executorch.backends.cpu.workflows import etdump

PLAN = {
    "regions": [
        {"provider": "XNNPACK", "implementation": "subgraph", "nodes": ["conv", "norm"]}
    ]
}
STEP = "XNNPACK/subgraph:conv+1"


def chunk_lines(plan: dict) -> list[str]:
    payload = json.dumps(plan).encode()
    chunks = [payload[start : start + 16] for start in range(0, len(payload), 16)]
    return [
        f"I CPUBackend.cpp:259] CPU_PLAN_CHUNK {index} {len(chunks)} {chunk.hex()}"
        for index, chunk in enumerate(chunks)
    ]


def event(name, samples, delegated=True, metadata=b""):
    return SimpleNamespace(
        name=name,
        perf_data=SimpleNamespace(raw=samples),
        is_delegated_op=delegated,
        raw_delegate_debug_metadatas=[metadata],
    )


class ETDumpTest(unittest.TestCase):
    def summarize(self, events):
        inspector = SimpleNamespace(event_blocks=[SimpleNamespace(events=events)])
        with patch.object(etdump, "Inspector", return_value=inspector):
            return etdump.summarize(
                Path("model.etdump"), [PLAN], warmup=1, iterations=2
            )

    def test_decode_plans_reassembles_each_delegate(self):
        other = {
            "regions": [
                {
                    "provider": "Reference",
                    "implementation": "registry",
                    "nodes": ["gelu"],
                }
            ]
        }
        log = "\n".join(chunk_lines(PLAN) + ["unrelated"] + chunk_lines(other))
        self.assertEqual(etdump.decode_plans(log), [PLAN, other])

    def test_decode_plans_rejects_incomplete_or_inconsistent_traces(self):
        lines = chunk_lines(PLAN)
        total = f" {len(lines)} "
        mixed = lines[:1] + [
            line.replace(total, f" {len(lines) + 1} ") for line in lines[1:]
        ]
        for invalid in (lines[:-1], lines[1:], list(reversed(lines)), mixed, []):
            with self.subTest(invalid=invalid), self.assertRaises(RuntimeError):
                etdump.decode_plans("\n".join(invalid))

    def test_summarize_drops_warmup_and_reports_shapes(self):
        shapes = {"inputs": [[1, 3]], "outputs": [[1, 3]]}
        profile = self.summarize(
            [
                event(STEP, [9.0, 2.0, 4.0]),
                event(
                    "Convolution #1",
                    [8.0, 1.0, 3.0],
                    metadata=json.dumps(shapes).encode(),
                ),
                event("Method::init", [5.0], delegated=False),
            ]
        )
        rows = {row["name"]: row for row in profile["events"]}
        self.assertEqual(rows[STEP]["kind"], "step")
        self.assertEqual(rows[STEP]["median_ms"], 3.0)
        self.assertEqual(rows["Convolution #1"]["shapes"], shapes)
        self.assertEqual(rows["Method::init"]["kind"], "init")
        self.assertEqual(profile["totals_ms"], {"step": 3.0, "provider": 2.0})

    def test_summarize_skips_empty_samples_and_ignores_malformed_metadata(self):
        with self.assertLogs(etdump.logger, "WARNING"):
            profile = self.summarize(
                [
                    event(STEP, [9.0, 2.0, 4.0]),
                    event("Convolution #1", [8.0, 1.0, 3.0], metadata=b"not json"),
                    event("Empty #1", []),
                ]
            )
        rows = {row["name"]: row for row in profile["events"]}
        self.assertNotIn("Empty #1", rows)
        self.assertIsNone(rows["Convolution #1"]["shapes"])

    def test_summarize_warns_about_unexpected_delegate_sample_counts(self):
        with self.assertLogs(etdump.logger, "WARNING") as logs:
            profile = self.summarize(
                [
                    event(STEP, [9.0, 2.0, 4.0]),
                    event("Convolution #1", [8.0, 1.0, 3.0]),
                    event("Transpose #1", [1.0] * 5),
                ]
            )
        self.assertIn("Transpose #1 has 5 samples", logs.output[0])
        rows = {row["name"]: row for row in profile["events"]}
        self.assertEqual(rows["Transpose #1"]["kind"], "init")

    def test_summarize_rejects_missing_steps(self):
        with self.assertRaisesRegex(RuntimeError, "lacks CPU plan steps"):
            self.summarize([event("Convolution #1", [8.0, 1.0, 3.0])])

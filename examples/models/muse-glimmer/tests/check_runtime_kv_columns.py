# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check CUDA column growth against preallocated storage using the same PTE.

The reference admits and closes all owners before its first forward. Both
processes then run identical queries through row growth, wrapped-history cloning,
sparse owners, and CUDA graph recapture. Closing an owner does not shrink storage.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from executorch.examples.models.muse_glimmer.export.export_solo import (
    export_cuda_batching,
)
from executorch.examples.models.muse_glimmer.model.model import MuseGlimmerModel
from executorch.examples.models.muse_glimmer.tests.test_pipeline import TINY_CONFIG


def commands() -> list[dict]:
    requests = [
        {"op": "new"},
        {
            "op": "step",
            "tokens": list(range(32)),
            "seq_ids": [0] * 32,
            "select": [31],
            "tag": "prefill",
            "check_reentrancy": True,
        },
        {
            "op": "step",
            "tokens": list(range(32, 48)),
            "seq_ids": [0] * 16,
            "select": [15],
            "tag": "wrap",
        },
        {"op": "clone", "seq_id": 0, "upto": 48},
        {"op": "new"},
        {
            "op": "step",
            "tokens": [13, 13, 19],
            "seq_ids": [0, 1, 2],
            "select": [0, 1, 2],
            "tag": "grow",
        },
        {"op": "remove", "seq_id": 0},
    ]
    for repeat in range(5):
        requests.append(
            {
                "op": "step",
                "tokens": [20 + repeat, 30 + repeat],
                "seq_ids": [1, 2],
                "select": [0, 1],
                "tag": f"decode-before-{repeat}",
            }
        )
    requests.extend([{"op": "remove", "seq_id": 1}, {"op": "remove", "seq_id": 2}])
    requests.extend(
        [
            {"op": "new"},
            {
                "op": "step",
                "tokens": [5, 6, 7, 8],
                "seq_ids": [0] * 4,
                "select": [3],
                "tag": "reuse",
            },
        ]
    )
    requests.extend({"op": "new"} for _ in range(4))
    requests.append(
        {
            "op": "step",
            "tokens": [9, 10, 11, 12, 13],
            "seq_ids": list(range(5)),
            "select": list(range(5)),
            "tag": "burst",
        }
    )
    for repeat in range(5):
        requests.append(
            {
                "op": "step",
                "tokens": [20 + repeat, 30 + repeat],
                "seq_ids": [1, 2],
                "select": [0, 1],
                "tag": f"decode-after-{repeat}",
            }
        )
    requests.extend([{"op": "remove", "seq_id": 3}, {"op": "remove", "seq_id": 4}])
    requests.extend([{"op": "new"}, {"op": "new"}])
    requests.append(
        {
            "op": "step",
            "tokens": [14, 15, 16, 17, 18],
            "seq_ids": list(range(5)),
            "select": list(range(5)),
            "tag": "burst-reused",
        }
    )
    requests.extend({"op": "remove", "seq_id": owner} for owner in (4, 3, 2, 1, 0))
    for request in requests:
        if request["op"] == "step":
            request["save_logits"] = True
    return requests


def run(
    probe: Path,
    artifact: Path,
    preallocate: bool,
    custom_requests=None,
    compaction_delay_ms=500,
    label=None,
) -> tuple[dict, list[dict]]:
    mode = label or ("preallocated" if preallocate else "growing")
    output = Path(tempfile.mkdtemp(prefix=mode + "-", dir=artifact))
    prefix = []
    if preallocate:
        prefix.extend({"op": "new"} for _ in range(8))
        prefix.extend({"op": "remove", "seq_id": owner} for owner in range(8))
    requests = prefix + (commands() if custom_requests is None else custom_requests)
    completed = subprocess.run(
        [
            str(probe),
            f"--pte={artifact / 'model.pte'}",
            f"--data_path={artifact / 'aoti_cuda_blob.ptd'}",
            f"--output_dir={output}",
            "--max_sessions=8",
            "--max_context=64",
            "--initial_kv_capacity=4",
            f"--kv_compaction_delay_ms={compaction_delay_ms}",
        ],
        input="".join(json.dumps(r) + "\n" for r in requests),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
    )
    (output / "probe.stderr.log").write_text(completed.stderr)
    (output / "probe.jsonl").write_text(completed.stdout)
    if completed.returncode:
        print(completed.stderr, file=sys.stderr)
        print(completed.stdout, file=sys.stderr)
    completed.check_returncode()
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    assert responses[0]["ready"] and len(responses) == len(requests) + 1
    logits = {}
    for request, response in zip(requests, responses[1:], strict=True):
        assert response["op"] == request["op"] and not response.get("failed")
        if request["op"] == "step":
            arrays = [
                np.fromfile(item["file"], dtype=np.float32).reshape(item["shape"])
                for item in response["logits"]
            ]
            logits[request["tag"]] = np.concatenate(arrays)
    return logits, responses[1 + len(prefix) :]


def check_compaction(probe: Path, artifact: Path) -> None:
    # A clone holds a wrapped history while every other owner is explicitly
    # closed. Exercise both manual compaction and automatic idle policy.
    requests = [
        {"op": "new"},
        {
            "op": "step",
            "tokens": list(range(32)),
            "seq_ids": [0] * 32,
            "select": [31],
            "tag": "compact-prefill",
        },
        {
            "op": "step",
            "tokens": list(range(32, 48)),
            "seq_ids": [0] * 16,
            "select": [15],
            "tag": "compact-wrap",
        },
        {"op": "clone", "seq_id": 0, "upto": 48},
    ]
    requests.extend({"op": "new"} for _ in range(6))
    requests.append(
        {
            "op": "step",
            "tokens": [13] * 8,
            "seq_ids": list(range(8)),
            "select": list(range(8)),
            "tag": "compact-burst",
        }
    )
    requests.append({"op": "idle", "tag": "all-open"})
    requests.extend({"op": "remove", "seq_id": i} for i in (0, 2, 3, 4, 5, 6, 7))
    requests.append({"op": "compact", "spare_columns": 0, "tag": "manual"})
    for repeat in range(5):
        requests.append(
            {
                "op": "step",
                "tokens": [20 + repeat],
                "seq_ids": [1],
                "select": [0],
                "tag": f"compact-before-{repeat}",
            }
        )
    requests.extend({"op": "new"} for _ in range(7))
    requests.append(
        {
            "op": "step",
            "tokens": [14] * 7,
            "seq_ids": [0, 2, 3, 4, 5, 6, 7],
            "select": list(range(7)),
            "tag": "compact-regrow",
        }
    )
    requests.extend({"op": "remove", "seq_id": i} for i in (0, 2, 3, 4, 5, 6, 7))
    requests.append({"op": "idle", "tag": "idle-before"})
    requests.append({"op": "wait", "milliseconds": 600})
    requests.append({"op": "idle", "tag": "idle-shrink"})
    for repeat in range(5):
        requests.append(
            {
                "op": "step",
                "tokens": [30 + repeat],
                "seq_ids": [1],
                "select": [0],
                "tag": f"compact-after-{repeat}",
            }
        )
    for request in requests:
        if request["op"] == "step":
            request["save_logits"] = True
    reference_requests = [
        {"op": "metrics"} if r["op"] in ("compact", "idle") else r for r in requests
    ]
    fixed, _ = run(probe, artifact, True, reference_requests, label="compact-fixed")
    for delay, before, expected, label in (
        (0, 3, 3, "compact-immediate"),
        (500, 9, 3, "compact-expiring"),
        (60000, 9, 9, "compact-delayed"),
    ):
        actual, trace = run(probe, artifact, False, requests, delay, label)
        assert fixed.keys() == actual.keys()
        for tag in fixed:
            assert np.isfinite(fixed[tag]).all() and np.isfinite(actual[tag]).all(), tag
            np.testing.assert_array_equal(actual[tag], fixed[tag], err_msg=tag)
        snapshots = {
            r.get("tag"): response["snapshot"]
            for r, response in zip(requests, trace, strict=True)
            if "tag" in r
        }
        assert snapshots["all-open"]["live_sequences"] == 8
        assert snapshots["all-open"]["physical_columns"] == 9
        assert snapshots["manual"]["live_sequences"] == 1
        assert snapshots["manual"]["physical_columns"] == 2
        assert snapshots["idle-before"]["physical_columns"] == before
        assert snapshots["idle-shrink"]["live_sequences"] == 1
        assert snapshots["idle-shrink"]["physical_columns"] == expected
        assert (
            snapshots["manual"]["live_KV_bytes"]
            < snapshots["all-open"]["live_KV_bytes"]
        )
        if expected == 3:
            assert (
                snapshots["idle-shrink"]["live_KV_bytes"]
                < snapshots["compact-regrow"]["live_KV_bytes"]
            )
        graph = next(
            m
            for m in trace[-1]["snapshot"]["method_calls"]
            if m["method"] == "forward_1"
        )
        assert graph["captures"] >= 2 and graph["replays"] > graph["captures"], graph
        assert graph["capture_failures"] == 0, graph
    print(
        "Native compaction, open-owner retention, idle delay, identical logits and graph recapture passed"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--check-compaction", action="store_true")
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    artifact = Path(tempfile.mkdtemp(prefix="run-", dir=root))
    print(f"Runtime KV column validation artifacts: {artifact}", flush=True)
    torch.manual_seed(123)
    config = replace(
        TINY_CONFIG,
        dim=128,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        global_attn_cfg="[16,0]",
    )
    model = MuseGlimmerModel(config).to(torch.bfloat16).eval()
    model.activation_dtype = torch.bfloat16
    export_cuda_batching(
        model,
        config,
        str(artifact),
        max_step_tokens=32,
        max_cells=512,
        forward_sizes=(1, 2, 4, 8, 16),
        kv_layout="sequence",
        max_kv_seqs=8,
    )
    fixed, fixed_responses = run(args.probe.resolve(), artifact, True)
    dynamic, responses = run(args.probe.resolve(), artifact, False)
    assert fixed.keys() == dynamic.keys()
    for tag in fixed:
        assert np.isfinite(fixed[tag]).all(), tag
        assert np.isfinite(dynamic[tag]).all(), tag
        np.testing.assert_array_equal(dynamic[tag], fixed[tag], err_msg=tag)
    np.testing.assert_array_equal(dynamic["grow"][0], dynamic["grow"][1])
    steps = [r for r in responses if r["op"] == "step"]
    fixed_steps = [r for r in fixed_responses if r["op"] == "step"]
    for reference, candidate in zip(fixed_steps, steps, strict=True):
        assert (
            candidate["snapshot"]["live_KV_bytes"]
            <= reference["snapshot"]["live_KV_bytes"]
        )
    assert (
        steps[0]["snapshot"]["live_KV_bytes"]
        < fixed_steps[0]["snapshot"]["live_KV_bytes"]
    )
    assert steps[0]["snapshot"]["physical_columns"] == 2
    assert steps[2]["snapshot"]["physical_columns"] == 5
    assert steps[2]["snapshot"]["declared_columns"] == 9
    assert responses[-1]["snapshot"]["physical_columns"] == 9
    assert responses[-1]["snapshot"]["live_sequences"] == 0
    assert fixed_responses[-1]["snapshot"]["column_resize_count"] == 0
    assert responses[-1]["snapshot"]["column_resize_count"] == 3
    for request, response in zip(commands(), responses, strict=True):
        if request["op"] == "remove":
            assert response["snapshot"]["physical_columns"] >= 5
    for trace, captures in ((fixed_responses, 1), (responses, 2)):
        graph = next(
            method
            for method in trace[-1]["snapshot"]["method_calls"]
            if method["method"] == "forward_2"
        )
        assert graph["captures"] == captures, graph
        assert graph["replays"] > graph["captures"], graph
        assert graph["capture_failures"] == 0, graph
    print(
        "Native column growth, identical logits, ring cloning and graph recapture passed"
    )

    if args.check_compaction:
        check_compaction(args.probe.resolve(), artifact)


if __name__ == "__main__":
    main()

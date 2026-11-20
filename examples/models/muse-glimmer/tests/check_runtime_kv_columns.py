# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compare actual ABI-2 CUDA allocations and logits with the fixed-column ABI.

Exports the same tiny flat/ring model twice and drives the native probe through
row growth, wrapped-history cloning, sparse-owner compaction and graph replay.
"""

import argparse
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from executorch.examples.models.muse_glimmer.export.export_solo import (
    export_cuda_batching,
)
from executorch.examples.models.muse_glimmer.model.model import MuseGlimmerModel
from executorch.examples.models.muse_glimmer.tests.test_pipeline import TINY_CONFIG


def commands(runtime_columns: bool) -> list[dict]:
    """Cross row/column resize boundaries with deterministic inputs."""
    requests = [
        {"op": "new"},
        {
            "op": "step",
            "tokens": list(range(32)),
            "seq_ids": [0] * 32,
            "select": [31],
            "tag": "prefill",
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
    for stage in range(2):
        if stage == 1 and runtime_columns:
            requests.append({"op": "compact", "spare_columns": 0})
        for repeat in range(5):
            requests.append(
                {
                    "op": "step",
                    "tokens": [20 + repeat, 30 + repeat],
                    "seq_ids": [1, 2],
                    "select": [0, 1],
                    "tag": f"decode-{stage}-{repeat}",
                }
            )
    requests.extend([{"op": "remove", "seq_id": 1}, {"op": "remove", "seq_id": 2}])
    if runtime_columns:
        requests.append({"op": "compact", "spare_columns": 0})
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
    for request in requests:
        if request["op"] == "step":
            request["save_logits"] = True
    return requests


def run(probe: Path, root: Path, runtime_columns: bool) -> tuple[dict, list[dict]]:
    """Export and drive one ABI using the same model weights and queries."""
    artifact = root / ("abi2" if runtime_columns else "abi1")
    artifact.mkdir()
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
        runtime_kv_columns=runtime_columns,
    )
    output = artifact / "logits"
    output.mkdir()
    requests = commands(runtime_columns)
    completed = subprocess.run(
        [
            str(probe),
            f"--pte={artifact / 'model.pte'}",
            f"--data_path={artifact / 'aoti_cuda_blob.ptd'}",
            f"--output_dir={output}",
            "--max_sessions=8",
            "--max_context=64",
            "--initial_kv_capacity=4",
        ],
        input="".join(json.dumps(r) + "\n" for r in requests),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
    )
    (artifact / "probe.stderr.log").write_text(completed.stderr)
    (artifact / "probe.jsonl").write_text(completed.stdout)
    completed.check_returncode()
    capture = "CUDA graph: captured and instantiated for 'forward_2"
    assert completed.stderr.count(capture) >= (2 if runtime_columns else 1), (
        "forward_2 must capture, replay, and recapture after compaction; "
        "see probe.stderr.log"
    )
    for failure in (
        "CUDA graph: capture failed for 'forward_2",
        "CUDA graph: disabled for 'forward_2",
    ):
        assert (
            failure not in completed.stderr
        ), "eager fallback cannot qualify graph replay"
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
    return logits, responses


def main() -> None:
    """Require native ABI-2 numerical agreement and memory contraction."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed, _ = run(args.probe.resolve(), args.output_dir, False)
    dynamic, responses = run(args.probe.resolve(), args.output_dir, True)
    assert fixed.keys() == dynamic.keys()
    for tag in fixed:
        np.testing.assert_allclose(dynamic[tag], fixed[tag], atol=5e-2, rtol=0)
        np.testing.assert_array_equal(dynamic[tag].argmax(-1), fixed[tag].argmax(-1))
    np.testing.assert_allclose(
        dynamic["grow"][0], dynamic["grow"][1], atol=5e-2, rtol=0
    )
    steps = [r for r in responses if r.get("op") == "step"]
    assert steps[0]["snapshot"]["physical_columns"] == 2
    assert steps[2]["snapshot"]["physical_columns"] == 5
    assert steps[2]["snapshot"]["declared_columns"] == 9
    compactions = [r["snapshot"] for r in responses if r.get("op") == "compact"]
    assert [m["physical_columns"] for m in compactions] == [3, 2]
    assert compactions[0]["live_KV_bytes"] < steps[2]["snapshot"]["live_KV_bytes"]
    print("ABI-2 native logits, ring cloning, compaction and graph replay passed")


if __name__ == "__main__":
    main()

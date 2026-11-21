# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native ABI-2/ABI-3 equality, allocation and same-work timing qualification.

Tests MG- and Gemma-style head dimensions on the same tiny model weights.
Timing includes shared-stage undo transfers; it excludes logits file writes.
"""

import argparse
import json
import statistics
import subprocess
from dataclasses import replace
from pathlib import Path


def step(owners, tag, *, save=True):
    return {
        "op": "step",
        "seq_ids": owners,
        "tokens": [13] * len(owners),
        "select": list(range(len(owners))),
        "tag": tag,
        "save_logits": save,
        "argmax": save,
    }


def commands():
    requests = [{"op": "new"} for _ in range(64)]
    requests += [step(list(range(64)), "decode64-first")]
    long = step([0] * 96, "long")
    long["select"] = [95]
    requests += [long, step(list(range(64)), "decode64-mixed")]
    requests += [
        {"op": "remove", "seq_id": 63},
        {"op": "clone", "seq_id": 0, "upto": 95},
        {"op": "rewind", "seq_id": 63, "position": 94},
    ]
    requests += [step(list(range(64)), f"before-{i}") for i in range(7)]
    requests += [
        {"op": "remove", "seq_id": 0},
        {"op": "remove", "seq_id": 63},
        {"op": "compact", "spare_columns": 0},
    ]
    requests += [step(list(range(1, 63)), f"after-{i}") for i in range(5)]
    requests += [
        {"op": "clear"},
        {"op": "compact", "spare_columns": 0},
        {"op": "new"},
    ]
    for i in range(2):
        request = step([0] * 128, f"floor-seed-{i}")
        request["select"] = [127]
        requests.append(request)
    requests += [
        {"op": "clone", "seq_id": 0, "upto": 128},
        {"op": "rewind", "seq_id": 0, "position": 128},
        step([0, 1], "floor-replay"),
    ]
    for concurrency in (1, 2, 4, 8, 16, 32, 64):
        requests += [{"op": "clear"}, {"op": "compact", "spare_columns": 0}]
        requests += [{"op": "new"} for _ in range(concurrency)]
        for first in range(0, concurrency, 2):
            owners = [
                owner
                for owner in range(first, min(first + 2, concurrency))
                for _ in range(64)
            ]
            prefill = step(owners, f"seed-c{concurrency}-{first}", save=False)
            prefill["select"] = list(range(63, len(owners), 64))
            requests.append(prefill)
        requests += [
            step(list(range(concurrency)), f"warm-c{concurrency}-{i}", save=False)
            for i in range(4)
        ]
        requests += [
            step(list(range(concurrency)), f"bench-c{concurrency}-{i}", save=False)
            for i in range(20)
        ]
    return requests


def run(probe, root, head_dim, abi, repeat=0):
    import numpy as np

    artifact = root / f"d{head_dim}-abi{abi}"
    if repeat == 0:
        import torch
        from executorch.examples.models.muse_glimmer.export.export_solo import (
            export_cuda_batching,
        )
        from executorch.examples.models.muse_glimmer.model.model import MuseGlimmerModel
        from executorch.examples.models.muse_glimmer.tests.test_pipeline import (
            TINY_CONFIG,
        )

        artifact.mkdir()
        torch.manual_seed(123)
        config = replace(
            TINY_CONFIG,
            dim=2 * head_dim,
            head_dim=head_dim,
            n_layers=2,
            n_heads=2,
            n_kv_heads=1,
            max_seq_len=256,
            global_attn_cfg="[16,0]",
        )
        model = MuseGlimmerModel(config).to(torch.bfloat16).eval()
        model.activation_dtype = torch.bfloat16
        export_cuda_batching(
            model,
            config,
            str(artifact),
            max_step_tokens=128,
            max_cells=16384,
            forward_sizes=(1, 2, 4, 8, 16, 32, 64),
            kv_layout="sequence",
            max_kv_seqs=64,
            runtime_kv_columns=abi == 2,
            private_kv_rows=abi == 3,
        )
    run_dir = artifact / f"run-{repeat}"
    run_dir.mkdir()
    logits_dir = run_dir / "logits"
    logits_dir.mkdir()
    requests = commands()
    completed = subprocess.run(
        [
            str(probe),
            f"--pte={artifact / 'model.pte'}",
            f"--data_path={artifact / 'aoti_cuda_blob.ptd'}",
            f"--output_dir={logits_dir}",
            "--max_sessions=64",
            "--max_context=256",
            "--initial_kv_capacity=2",
        ],
        input="".join(json.dumps(r) + "\n" for r in requests),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=900,
    )
    (run_dir / "probe.stderr.log").write_text(completed.stderr)
    (run_dir / "probe.jsonl").write_text(completed.stdout)
    completed.check_returncode()
    replies = [json.loads(line) for line in completed.stdout.splitlines()]
    assert replies[0]["ready"] and len(replies) == len(requests) + 1
    pairs = list(zip(requests, replies[1:], strict=True))
    logits, steps = {}, {}
    for request, response in pairs:
        assert not response.get("failed") and request["op"] == response["op"]
        if request["op"] == "step":
            steps[request["tag"]] = response
            if request["save_logits"]:
                arrays = [
                    np.fromfile(x["file"], dtype=np.float32).reshape(x["shape"])
                    for x in response["logits"]
                ]
                logits[request["tag"]] = np.concatenate(arrays)
    return logits, steps, pairs


def graph(snapshot, name):
    return next(x for x in snapshot["method_calls"] if x["method"] == name)


def require_performance(evidence):
    """Block qualification on a median of three same-work throughput regressions."""
    regressions = [
        f"d{dim}/c{c}"
        for dim, result in evidence.items()
        for c, timing in result["same_work_timings"].items()
        if timing["throughput_ratio"] < 0.98
    ]
    if regressions:
        raise RuntimeError("Over 2% throughput regression: " + ", ".join(regressions))


def main():
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence = {}
    for dim in (128, 256):
        baseline, old_steps, _ = run(args.probe.resolve(), args.output_dir, dim, 2)
        candidate, new_steps, pairs = run(args.probe.resolve(), args.output_dir, dim, 3)
        assert baseline.keys() == candidate.keys()
        for tag in baseline:
            np.testing.assert_array_equal(
                candidate[tag], baseline[tag], err_msg=f"d{dim}/{tag}"
            )
        memory = {}
        for tag in ("decode64-first", "decode64-mixed", "before-6", "after-4"):
            old = old_steps[tag]["snapshot"]
            new = new_steps[tag]["snapshot"]
            combined = new["live_KV_bytes"] + new["device_staging_bytes"]
            memory[tag] = {
                "abi2_live_KV_bytes": old["live_KV_bytes"],
                "abi3_live_KV_bytes": new["live_KV_bytes"],
                "abi3_device_staging_bytes": new["device_staging_bytes"],
                "abi3_host_undo_bytes": new["host_undo_bytes"],
            }
            # Before either ring reaches its window there is little duplicated
            # headroom to remove; shared staging can cost more at this point.
            if tag != "decode64-first":
                assert combined < old["live_KV_bytes"], (tag, old, new)
            assert new["host_undo_bytes"] > 0
        compact = next(
            reply["snapshot"] for request, reply in pairs if request["op"] == "compact"
        )
        before = new_steps["before-6"]["snapshot"]
        after = new_steps["after-4"]["snapshot"]
        assert compact["live_KV_bytes"] < before["live_KV_bytes"]
        assert compact["allocator_used_bytes"] < before["allocator_used_bytes"]
        assert compact["allocator_reserved_bytes"] <= before["allocator_reserved_bytes"]
        old_graph, new_graph = graph(before, "forward_64"), graph(after, "forward_64")
        assert new_graph["captures"] - old_graph["captures"] == 1
        assert new_graph["replays"] - old_graph["replays"] > 1
        assert new_graph["capture_failures"] == 0
        repeats = [(old_steps, new_steps)]
        for repeat in (1, 2):
            # Reverse the order for the middle round without re-exporting.
            order = (3, 2) if repeat == 1 else (2, 3)
            results = {
                abi: run(args.probe.resolve(), args.output_dir, dim, abi, repeat)
                for abi in order
            }
            repeats.append((results[2][1], results[3][1]))
        timings = {}
        for c in (1, 2, 4, 8, 16, 32, 64):
            rounds = []
            for old_steps, new_steps in repeats:
                old = [
                    old_steps[f"bench-c{c}-{i}"]["elapsed_seconds"] for i in range(20)
                ]
                new = [
                    new_steps[f"bench-c{c}-{i}"]["elapsed_seconds"] for i in range(20)
                ]
                old_median, new_median = statistics.median(old), statistics.median(new)
                rounds.append(
                    {
                        "abi2_median_s": old_median,
                        "abi3_median_s": new_median,
                        "throughput_ratio": old_median / new_median,
                    }
                )
            ratio = statistics.median(r["throughput_ratio"] for r in rounds)
            timings[c] = {
                "rounds": rounds,
                "throughput_ratio": ratio,
                "investigate_over_2_percent_regression": ratio < 0.98,
            }
        evidence[dim] = {
            "exact_logits": True,
            "memory": memory,
            "same_work_timings": timings,
        }
    (args.output_dir / "qualification.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )
    print(json.dumps(evidence, indent=2))
    require_performance(evidence)
    print(
        "ABI-3 native logits, private memory, shared staging and per-stage replay passed"
    )


if __name__ == "__main__":
    main()

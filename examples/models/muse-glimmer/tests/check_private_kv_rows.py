# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native ABI-2/ABI-3 equality, allocation and same-work timing qualification.

Tests MG- and Gemma-style head dimensions on the same tiny model weights.
The mg-ring profile retains production MG attention/cache geometry, not its
30B parameter count or quantized GEMM performance.
Timing covers undo transfers, staggered growth and clone/rewind cycles;
it excludes logits file writes.
"""

import argparse
import json
import statistics
import subprocess
import sys
from dataclasses import replace
from pathlib import Path


def report_export_resources(tag):
    snapshot = {"tag": tag}
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        snapshot["cpu_peak_rss_bytes"] = int(peak) * (
            1 if sys.platform == "darwin" else 1024
        )
    except ImportError:
        pass
    if sys.platform.startswith("linux"):
        for name in ("memory.current", "memory.max", "memory.events"):
            try:
                snapshot["cgroup_" + name] = (
                    (Path("/sys/fs/cgroup") / name).read_text().strip()
                )
            except OSError:
                pass
    print("PRIVATE_KV_EXPORT_RESOURCE " + json.dumps(snapshot), flush=True)
    return snapshot


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


def qualification_config(profile, head_dim):
    if profile == "mini":
        return {
            "max_step_tokens": 128,
            "max_context": 4096,
            "seed_length": 64,
            "window": 16,
            "ring_layers": 1,
            "model": {
                "dim": 2 * head_dim,
                "head_dim": head_dim,
                "n_layers": 2,
                "n_heads": 2,
                "n_kv_heads": 1,
                "max_seq_len": 4096,
                "global_attn_cfg": "[16,0]",
            },
        }
    if profile != "mg-ring" or head_dim != 128:
        raise ValueError("mg-ring requires head dimension 128")
    # Production attention/cache geometry, with reduced embedding and MLP
    # weights. This stresses the cache lifecycle without claiming 30B perf.
    return {
        "max_step_tokens": 4096,
        "max_context": 131072,
        "seed_length": 2048,
        "window": 2048,
        "ring_layers": 39,
        "model": {
            "dim": 256,
            "head_dim": 128,
            "n_layers": 52,
            "n_heads": 32,
            "n_kv_heads": 2,
            "max_seq_len": 131072,
            "global_attn_cfg": "[2048,2048,2048,0]",
        },
    }


def seed_commands(concurrency, geometry, profile):
    seed_length = geometry["seed_length"]
    chunk_size = min(seed_length, 512)
    owners_per_chunk = geometry["max_step_tokens"] // chunk_size
    requests = []
    for first in range(0, concurrency, owners_per_chunk):
        for start in range(0, seed_length, chunk_size):
            owners = [
                owner
                for owner in range(first, min(first + owners_per_chunk, concurrency))
                for _ in range(chunk_size)
            ]
            tag = f"seed-c{concurrency}-{first}"
            if profile != "mini":
                tag += f"-{start}"
            prefill = step(owners, tag, save=False)
            prefill["select"] = list(range(chunk_size - 1, len(owners), chunk_size))
            requests.append(prefill)
    return requests


def diversify_tokens(requests):
    lengths, seeds = {}, {}
    generation = 0
    for request in requests:
        op = request["op"]
        if op == "clear":
            lengths.clear()
            seeds.clear()
        elif op in ("new", "clone"):
            owner = next(i for i in range(64) if i not in lengths)
            generation += 1
            seeds[owner] = generation
            lengths[owner] = 0 if op == "new" else request["upto"]
        elif op == "remove":
            del lengths[request["seq_id"]]
            del seeds[request["seq_id"]]
        elif op == "rewind":
            lengths[request["seq_id"]] = request["position"]
        elif op == "step":
            tokens = []
            for owner in request["seq_ids"]:
                position = lengths[owner]
                tokens.append(
                    1
                    + (
                        seeds[owner] * 67
                        + owner * 29
                        + position * 17
                        + (position // 251) * 13
                    )
                    % 251
                )
                lengths[owner] += 1
            request["tokens"] = tokens
    return requests


def graph_commands():
    # Exercise every exported captured stage across the same contraction.
    requests = [{"op": "clear"}, {"op": "compact", "spare_columns": 0}]
    requests += [{"op": "new"} for _ in range(64)]
    for first in range(0, 64, 2):
        seed = step(
            [owner for owner in (first, first + 1) for _ in range(64)],
            f"graph-seed-{first}",
            save=False,
        )
        seed["select"] = [63, 127]
        requests.append(seed)
    for phase, owners in (("before", 64), ("after", 48)):
        if phase == "after":
            requests += [{"op": "remove", "seq_id": owner} for owner in range(48, 64)]
            requests.append({"op": "compact", "spare_columns": 0})
        for repeat in range(2):
            for width in (1, 2, 4, 8, 16, 32, 64):
                batch = sorted(row % owners for row in range(width))
                requests.append(step(batch, f"graph-{phase}-{repeat}-{width}"))
    return requests


def commands(profile="mini"):
    geometry = qualification_config(profile, 128)
    seed_length = geometry["seed_length"]
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
    requests += graph_commands()
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
    if profile == "mg-ring":
        requests += [
            {"op": "clear"},
            {"op": "compact", "spare_columns": 0},
            {"op": "new"},
        ]
        width = geometry["max_step_tokens"]
        for chunk in range(2):
            seed = step([0] * width, f"production-ring-seed-{chunk}", save=False)
            seed["select"] = [width - 1]
            seed["tokens"] = [13 + (chunk * width + i) % 23 for i in range(width)]
            requests.append(seed)
        requests += [
            {"op": "clone", "seq_id": 0, "upto": width},
            {"op": "rewind", "seq_id": 0, "position": width},
            step([0, 1], "production-ring-floor-replay"),
        ]
    for concurrency in (1, 2, 4, 8, 16, 32, 64):
        requests += [{"op": "clear"}, {"op": "compact", "spare_columns": 0}]
        requests += [{"op": "new"} for _ in range(concurrency)]
        requests += seed_commands(concurrency, geometry, profile)
        requests += [
            step(list(range(concurrency)), f"warm-c{concurrency}-{i}", save=False)
            for i in range(4)
        ]
        requests += [
            step(list(range(concurrency)), f"bench-c{concurrency}-{i}", save=False)
            for i in range(20)
        ]
        # Leave one free slot at c64 for a fork.
        destination = max(1, concurrency - 1)
        if concurrency > 1:
            requests.append({"op": "remove", "seq_id": destination})
        for i in range(8):
            tag = f"restore-c{concurrency}-{i}"
            requests += [
                {
                    "op": "clone",
                    "seq_id": 0,
                    "upto": seed_length + 23,
                    "tag": tag + "-clone",
                },
                {
                    "op": "rewind",
                    "seq_id": destination,
                    "position": seed_length + 22,
                    "tag": tag + "-rewind",
                },
                step([destination], tag + "-step", save=False),
                {"op": "remove", "seq_id": destination, "tag": tag + "-remove"},
            ]
        requests += [{"op": "clear"}, {"op": "compact", "spare_columns": 0}]
        for owner in range(concurrency):
            tag = f"growth-c{concurrency}-{owner}"
            seed = step([owner] * 8, tag + "-seed", save=False)
            seed["select"] = [7]
            requests += [
                {"op": "new", "tag": tag + "-new"},
                seed,
                step(list(range(owner + 1)), tag + "-decode", save=False),
            ]
        requests.append(step(list(range(concurrency)), f"growth-verify-c{concurrency}"))
    # Free enough private history to exceed CUDA pool page granularity. This
    # phase is separate from timing and keeps 48 short owners alive throughout.
    requests += [{"op": "clear"}, {"op": "compact", "spare_columns": 0}]
    requests += [{"op": "new"} for _ in range(64)]
    requests.append(step(list(range(64)), "driver-seed", save=False))
    for owner in range(16):
        for chunk in range(16):
            prefill = step([owner] * 128, f"driver-prefill-{owner}-{chunk}", save=False)
            prefill["select"] = [127]
            requests.append(prefill)
    requests.append(step(list(range(64)), "driver-before", save=False))
    requests += [{"op": "remove", "seq_id": owner} for owner in range(16)]
    requests.append({"op": "compact", "spare_columns": 0, "tag": "driver-compact"})
    requests.append(step(list(range(16, 64)), "driver-after"))
    # Reuse a long owner's ID while a short owner stays live, then compact.
    requests += [
        {"op": "clear"},
        {"op": "compact", "spare_columns": 0},
        {"op": "new"},
        {"op": "new"},
    ]
    long = step([0] * 128, "reuse-long", save=False)
    long["select"] = [127]
    requests += [
        long,
        step([1], "reuse-short", save=False),
        {"op": "remove", "seq_id": 0},
        {"op": "new"},
        step([0, 1], "reuse-before"),
        {"op": "compact", "spare_columns": 0, "tag": "reuse-compact"},
        step([0, 1], "reuse-after"),
    ]
    return diversify_tokens(requests)


def run(probe, root, head_dim, abi, repeat=0, profile="mini"):
    import numpy as np

    geometry = qualification_config(profile, head_dim)
    prefix = "" if profile == "mini" else profile + "-"
    artifact = root / f"{prefix}d{head_dim}-abi{abi}"
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
        config = replace(TINY_CONFIG, **geometry["model"])
        report_export_resources(f"{profile}/d{head_dim}/abi{abi}/before-model")
        model = MuseGlimmerModel(config).to(torch.bfloat16).eval()
        model.activation_dtype = torch.bfloat16
        report_export_resources(f"{profile}/d{head_dim}/abi{abi}/before-export")
        export_cuda_batching(
            model,
            config,
            str(artifact),
            max_step_tokens=geometry["max_step_tokens"],
            max_cells=64 * geometry["max_context"],
            forward_sizes=(1, 2, 4, 8, 16, 32, 64),
            kv_layout="sequence",
            max_kv_seqs=64,
            runtime_kv_columns=abi == 2,
            private_kv_rows=abi == 3,
        )
        report_export_resources(f"{profile}/d{head_dim}/abi{abi}/after-export")
    run_dir = artifact / f"run-{repeat}"
    run_dir.mkdir()
    logits_dir = run_dir / "logits"
    logits_dir.mkdir()
    requests = commands(profile)
    completed = subprocess.run(
        [
            str(probe),
            f"--pte={artifact / 'model.pte'}",
            f"--data_path={artifact / 'aoti_cuda_blob.ptd'}",
            f"--output_dir={logits_dir}",
            "--max_sessions=64",
            f"--max_context={geometry['max_context']}",
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
    if completed.returncode != 0:
        replies = completed.stdout.splitlines()
        print(
            f"PRIVATE_KV_PROBE_FAILURE {profile}/d{head_dim}/abi{abi}: "
            f"{replies[-1] if replies else 'no protocol response'}",
            flush=True,
        )
        print(completed.stderr[-2000:], file=sys.stderr, flush=True)
    completed.check_returncode()
    replies = [json.loads(line) for line in completed.stdout.splitlines()]
    assert replies[0]["ready"] and len(replies) == len(requests) + 1
    pairs = list(zip(requests, replies[1:], strict=True))
    logits, steps = {}, {}
    previous_snapshot = replies[0]["snapshot"]
    for request, response in pairs:
        assert not response.get("failed") and request["op"] == response["op"]
        response["cache_phase_seconds"] = {
            name.removesuffix("_ns")
            + "_s": (response["snapshot"][name] - previous_snapshot[name]) * 1e-9
            for name in (
                "cache_prepare_time_ns",
                "cache_commit_time_ns",
                "archive_transfer_time_ns",
                "archive_append_time_ns",
            )
        }
        assert all(t >= 0 for t in response["cache_phase_seconds"].values())
        previous_snapshot = response["snapshot"]
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


def require_graph_contraction(before, after):
    for width in (1, 2, 4, 8, 16, 32, 64):
        name = f"forward_{width}"
        old_graph, new_graph = graph(before, name), graph(after, name)
        assert old_graph["captures"] > 0, name
        assert new_graph["captures"] - old_graph["captures"] == 1, name
        assert new_graph["replays"] - old_graph["replays"] > 0, name
        assert new_graph["capture_failures"] == 0, name


def host_undo_budget(geometry):
    # BF16 K/V, 64 tagged rows per page, including pinned archive staging.
    undo_rows = min(
        geometry["max_context"],
        geometry["window"] + geometry["max_step_tokens"] - 1,
    )
    undo_pages = (undo_rows + 63) // 64
    row_bytes = geometry["model"]["n_kv_heads"] * geometry["model"]["head_dim"] * 2
    return geometry["ring_layers"] * (
        64 * undo_pages * (2 * 64 * row_bytes + 64 * 4)
        + 2 * geometry["max_step_tokens"] * row_bytes
    )


def require_exact_logits(candidate, baseline, tag):
    import numpy as np

    for name, logits in (("ABI 2", baseline), ("ABI 3", candidate)):
        if not np.isfinite(logits).all():
            raise RuntimeError(f"{tag}: {name} returned non-finite logits")
    np.testing.assert_array_equal(candidate, baseline, err_msg=tag)


def require_performance(evidence):
    """Block qualification on decode or lifecycle median regressions."""
    regressions = [
        f"d{dim}/{'c' if str(c).isdigit() else ''}{c}"
        for dim, result in evidence.items()
        for c, timing in {
            **result["same_work_timings"],
            **result.get("lifecycle_timings", {}),
        }.items()
        if timing["throughput_ratio"] < 0.98
    ]
    if regressions:
        raise RuntimeError("Over 2% throughput regression: " + ", ".join(regressions))


def require_driver_contraction(before, compact):
    released = before["live_KV_bytes"] - compact["live_KV_bytes"]
    reserved_drop = (
        before["allocator_reserved_bytes"] - compact["allocator_reserved_bytes"]
    )
    driver_drop = before["device_used_bytes"] - compact["device_used_bytes"]
    # Require a material fixture and actual release, with room for CUDA pool
    # page granularity and graph bookkeeping; lower live bytes alone do not pass.
    if released < 16 * 1024**2:
        raise RuntimeError("Driver contraction fixture released less than 16 MiB")
    if reserved_drop < 8 * 1024**2 or driver_drop < 8 * 1024**2:
        raise RuntimeError("Compaction did not return at least 8 MiB to the driver")
    return {
        "live_KV_drop_bytes": released,
        "allocator_reserved_drop_bytes": reserved_drop,
        "driver_used_drop_bytes": driver_drop,
    }


def paired_timings(repeats):
    timings = {}
    for c in (1, 2, 4, 8, 16, 32, 64):
        rounds = []
        for old_steps, new_steps, _, _ in repeats:
            old = [old_steps[f"bench-c{c}-{i}"]["elapsed_seconds"] for i in range(20)]
            new = [new_steps[f"bench-c{c}-{i}"]["elapsed_seconds"] for i in range(20)]
            old_median, new_median = statistics.median(old), statistics.median(new)
            rounds.append(
                {
                    "abi2_median_s": old_median,
                    "abi3_median_s": new_median,
                    "throughput_ratio": old_median / new_median,
                    "cache_cpu_wall_s": {
                        f"abi{abi}": {
                            phase: statistics.median(
                                steps[f"bench-c{c}-{i}"]["cache_phase_seconds"][phase]
                                for i in range(20)
                            )
                            for phase in steps[f"bench-c{c}-0"].get(
                                "cache_phase_seconds", {}
                            )
                        }
                        for abi, steps in ((2, old_steps), (3, new_steps))
                    },
                }
            )
        ratio = statistics.median(r["throughput_ratio"] for r in rounds)
        timings[c] = {
            "rounds": rounds,
            "throughput_ratio": ratio,
            "investigate_over_2_percent_regression": ratio < 0.98,
        }
    lifecycle = {}
    for phase in ("growth", "restore"):
        for c in (1, 2, 4, 8, 16, 32, 64):
            prefix = f"{phase}-c{c}-"
            rounds = []
            for _, _, old_pairs, new_pairs in repeats:
                totals = [
                    sum(
                        response["elapsed_seconds"]
                        for request, response in pairs
                        if request.get("tag", "").startswith(prefix)
                    )
                    for pairs in (old_pairs, new_pairs)
                ]
                assert all(t > 0 for t in totals)
                rounds.append(
                    {
                        "abi2_total_s": totals[0],
                        "abi3_total_s": totals[1],
                        "throughput_ratio": totals[0] / totals[1],
                    }
                )
            ratio = statistics.median(r["throughput_ratio"] for r in rounds)
            lifecycle[f"{phase}-c{c}"] = {
                "rounds": rounds,
                "throughput_ratio": ratio,
                "investigate_over_2_percent_regression": ratio < 0.98,
            }
    return timings, lifecycle


def write_qualification(output_dir, evidence, report_performance):
    failure = None
    try:
        require_performance(evidence)
    except RuntimeError as error:
        failure = error
    result = {
        "timing_mode": "report-only" if report_performance else "enforced",
        "performance_qualified": not report_performance and failure is None,
        "timing_threshold_met": failure is None,
        "performance_error": None if failure is None else str(failure),
        "results": evidence,
    }
    (output_dir / "qualification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if report_performance:
        print(
            "PERFORMANCE_REPORT_ONLY: "
            + (str(failure) if failure is not None else "all median ratios >= 0.98")
            + "; performance not qualified"
        )
    elif failure is not None:
        raise failure
    else:
        print("PERFORMANCE_QUALIFIED: all median ratios >= 0.98")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--profile",
        choices=("mini", "mg-ring"),
        default="mini",
        help="mg-ring uses production MG attention/cache geometry with small weights",
    )
    parser.add_argument(
        "--report-performance",
        action="store_true",
        help="Report timing regressions without failing PR CI; does not qualify perf",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence = {}
    for dim in (128, 256) if args.profile == "mini" else (128,):
        geometry = qualification_config(args.profile, dim)
        baseline, old_steps, old_pairs = run(
            args.probe.resolve(), args.output_dir, dim, 2, profile=args.profile
        )
        candidate, new_steps, pairs = run(
            args.probe.resolve(), args.output_dir, dim, 3, profile=args.profile
        )
        assert baseline.keys() == candidate.keys()
        for tag in baseline:
            require_exact_logits(candidate[tag], baseline[tag], f"d{dim}/{tag}")
        memory = {}
        undo_budget = host_undo_budget(geometry)
        memory_tags = ("decode64-first", "decode64-mixed", "before-6", "after-4")
        savings_tags = set(memory_tags[1:])
        if args.profile == "mg-ring":
            # The first heterogeneous phase is shorter than the 2K window.
            # Also measure saturated Ring history, where headroom is removed.
            memory_tags += ("bench-c64-19", "driver-before")
            savings_tags = {"bench-c64-19", "driver-before"}
        for tag in memory_tags:
            old = old_steps[tag]["snapshot"]
            new = new_steps[tag]["snapshot"]
            combined = new["live_KV_bytes"] + new["device_staging_bytes"]
            memory[tag] = {
                "abi2_live_KV_bytes": old["live_KV_bytes"],
                "abi3_live_KV_bytes": new["live_KV_bytes"],
                "abi3_device_staging_bytes": new["device_staging_bytes"],
                "abi3_host_undo_bytes": new["host_undo_bytes"],
                "abi3_host_undo_budget_bytes": undo_budget,
                "abi2_allocator_used_bytes": old["allocator_used_bytes"],
                "abi3_allocator_used_bytes": new["allocator_used_bytes"],
            }
            # Before either ring reaches its window there is little duplicated
            # headroom to remove; shared staging can cost more at this point.
            if tag in savings_tags:
                assert combined < old["live_KV_bytes"], (tag, old, new)
                assert new["allocator_used_bytes"] < old["allocator_used_bytes"], (
                    tag,
                    old,
                    new,
                )
            assert new["host_undo_bytes"] > 0
            assert new["host_undo_bytes"] <= undo_budget, (tag, new, undo_budget)
        peak_host_undo = max(reply["snapshot"]["host_undo_bytes"] for _, reply in pairs)
        assert peak_host_undo <= undo_budget, (peak_host_undo, undo_budget)
        compact = next(
            reply["snapshot"] for request, reply in pairs if request["op"] == "compact"
        )
        before = new_steps["before-6"]["snapshot"]
        assert compact["live_KV_bytes"] < before["live_KV_bytes"]
        assert compact["allocator_used_bytes"] < before["allocator_used_bytes"]
        assert compact["allocator_reserved_bytes"] <= before["allocator_reserved_bytes"]
        require_graph_contraction(
            new_steps["graph-before-1-64"]["snapshot"],
            new_steps["graph-after-1-64"]["snapshot"],
        )
        driver_before = new_steps["driver-before"]["snapshot"]
        driver_compact = next(
            reply["snapshot"]
            for request, reply in pairs
            if request.get("tag") == "driver-compact"
        )
        driver = require_driver_contraction(driver_before, driver_compact)
        driver["before"] = driver_before
        driver["compact"] = driver_compact
        driver["after"] = new_steps["driver-after"]["snapshot"]
        reuse_before = new_steps["reuse-before"]["snapshot"]
        reuse_compact = next(
            reply["snapshot"]
            for request, reply in pairs
            if request.get("tag") == "reuse-compact"
        )
        assert reuse_compact["flat_capacity"] <= 4, reuse_compact
        assert reuse_compact["live_KV_bytes"] < reuse_before["live_KV_bytes"]
        repeats = [(old_steps, new_steps, old_pairs, pairs)]
        for repeat in (1, 2, 3):
            # Four rounds balance AB/BA order without re-exporting.
            order = (3, 2) if repeat % 2 else (2, 3)
            results = {
                abi: run(
                    args.probe.resolve(),
                    args.output_dir,
                    dim,
                    abi,
                    repeat,
                    profile=args.profile,
                )
                for abi in order
            }
            for tag in results[2][0]:
                require_exact_logits(
                    results[3][0][tag], results[2][0][tag], f"d{dim}/{tag}"
                )
            repeats.append((results[2][1], results[3][1], results[2][2], results[3][2]))
        timings, lifecycle = paired_timings(repeats)
        evidence[dim] = {
            "profile": args.profile,
            "geometry": geometry,
            "exact_logits": True,
            "memory": memory,
            "peak_host_undo_bytes": peak_host_undo,
            "host_undo_budget_bytes": undo_budget,
            "driver_contraction": driver,
            "reused_owner_compaction": {
                "before": reuse_before,
                "compact": reuse_compact,
                "after": new_steps["reuse-after"]["snapshot"],
            },
            "same_work_timings": timings,
            "lifecycle_timings": lifecycle,
        }
    write_qualification(args.output_dir, evidence, args.report_performance)
    print(
        "ABI-3 native logits, private memory, shared staging and per-stage replay passed"
    )


if __name__ == "__main__":
    main()

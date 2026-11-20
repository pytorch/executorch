#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark triton::linear on strict FP32 ``x @ weight.T + bias`` against what
the CUDA backend lowers it to otherwise, Inductor's Triton GEMM templates,
and cuBLAS.

Inductor compiles the graph each lowering API gives the backend: ``linear``
(the ``aten.linear`` call to_edge_transform_and_lower keeps) or ``core`` (the
``view``, ``permute_copy`` of the weight and ``addmm`` or ``mm`` that
to_edge().to_backend() gives, which includes copying the weight on every
call). triton::linear reads the weight as it is.

Inductor gets the choices the backend's AOT compile has (no decompose-K).
Timing as in ``benchmarks/conv``: cold L2, CUDA graph replay, median and
10th-90th percentile spread; "significantly faster" means a median speedup of
at least ``--threshold`` with our 90th percentile below Inductor's 10th.
triton::linear runs with the schedule its compile would pick. Accuracy gate,
against an FP64 reference: with ``scale = max(1, max|reference|)``, our largest
absolute error must stay within ``max(1e-6 * scale, 3 * cuBLAS's)``.

Usage:
    python benchmark_linear.py --graph core
    python benchmark_linear.py --shard 0/4 --json out.jsonl
"""

import argparse
import json
import statistics
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from executorch.backends.cuda.autotune.launch_params import (
    autotune_launch_params,
    clear_launch_param_cache,
)
from executorch.backends.cuda.triton.kernels.gemm import routing
from executorch.backends.cuda.triton.kernels.gemm.linear import (
    force_linear_candidate,
    linear,
)


@dataclass(frozen=True)
class Case:
    label: str
    lead: tuple[int, ...]
    n: int
    k: int
    bias: bool
    graph: str
    # "contiguous", "transposed" (x stored as the transpose of a [K, M]
    # tensor) or "sliced" (x a slice of wider rows, at an offset).
    x_layout: str = "contiguous"


# Calls across the row classes: a few rows (a matrix-vector product), short
# rows and many rows, with short and long reductions.
_SHAPES = (
    ("vector", (1,), 256, 2048, False),
    ("vector", (1,), 2048, 2048, False),
    ("vector", (1,), 2048, 16384, False),
    ("vector", (1,), 3072, 2828, True),
    ("vector", (4,), 1024, 4096, False),
    ("short rows", (32,), 512, 512, True),
    ("short rows", (32,), 2048, 512, True),
    ("short rows", (40,), 768, 3072, True),
    ("short rows", (50,), 1024, 4096, False),
    ("short rows", (16,), 2048, 4096, True),
    ("many rows", (302,), 512, 3200, True),
    ("many rows", (563,), 512, 512, True),
    ("many rows", (768,), 1152, 1152, True),
    ("many rows", (2, 131), 1024, 4096, True),
)


def shape_cases(graph):
    return [Case(label, lead, n, k, bias, graph) for label, lead, n, k, bias in _SHAPES]


def envelope_cases(graph):
    """Calls of at most four rows across the weight sizes."""
    return [
        Case("envelope", (m,), n, k, False, graph)
        for m in (1, 4)
        for n in (64, 256, 512, 1024, 2048, 4096, 16384, 65536)
        for k in (64, 256, 1024, 4096, 16384)
    ]


def gemv_boundary_cases(graph):
    """Every corner of the class the router takes to the matrix-vector
    schedule without measurement, with and without bias and with an input
    sliced from wider rows at an offset (which measurement does not take)."""
    return [
        Case("gemv boundary", (m,), n, k, bias, graph, layout)
        for m, n, k in routing.gemv_boundary_calls()
        for bias, layout in (
            (False, "contiguous"),
            (True, "contiguous"),
            (False, "sliced"),
        )
    ]


_CALLS_PER_GRAPH = 10
_REPLAYS = 20


def time_us(fn, flush):
    """Per-call device times (us) of ``fn`` replayed in a CUDA graph."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    events = [
        (
            torch.cuda.Event(enable_timing=True, external=True),
            torch.cuda.Event(enable_timing=True, external=True),
        )
        for _ in range(_CALLS_PER_GRAPH)
    ]
    with torch.cuda.stream(stream):
        fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for start, end in events:
                if flush is not None:
                    flush.zero_()
                start.record()
                fn()
                end.record()
        graph.replay()
        times = []
        for _ in range(_REPLAYS):
            graph.replay()
            events[-1][1].synchronize()
            times += [s.elapsed_time(e) * 1000.0 for s, e in events]
    torch.cuda.current_stream().wait_stream(stream)
    return times


def summarize(times):
    deciles = statistics.quantiles(times, n=10)
    return statistics.median(times), deciles[0], deciles[-1]


def triton_runner(x, w, b):
    # Times the schedules again, as the compile does for a new shape.
    clear_launch_param_cache()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), autotune_launch_params() as stats:
        linear(x, w, b)
    stream.synchronize()
    (pick,) = stats.picks
    schedule = pick.choice

    def run():
        with force_linear_candidate(schedule):
            return linear(x, w, b)

    return run, schedule


def _core(x, w, b):
    """What to_edge().to_backend() gives the backend for ``F.linear``."""
    rows = x.reshape(-1, x.shape[-1])
    wt = torch.ops.aten.permute_copy.default(w, [1, 0])
    out = torch.mm(rows, wt) if b is None else torch.addmm(b, rows, wt)
    return out.reshape(*x.shape[:-1], w.shape[0])


def inductor_runner(x, w, b, case):
    torch._dynamo.reset()
    fn = F.linear if case.graph == "linear" else _core
    # Without decompose-K, which the backend's AOT compile does not offer.
    with torch._inductor.config.patch(
        {
            "max_autotune": True,
            "max_autotune_gemm_backends": "TRITON",
            "triton.num_decompose_k_splits": 0,
        }
    ):
        compiled = torch.compile(fn, dynamic=False, fullgraph=True)
        compiled(x, w, b)
    return lambda: compiled(x, w, b)


def _tensors(case):
    torch.manual_seed(0)
    x = torch.randn(*case.lead, case.k, device="cuda")
    if case.x_layout == "transposed":
        x = x.reshape(-1, case.k).t().contiguous().t().reshape(x.shape)
    elif case.x_layout == "sliced":
        wide = torch.randn(*case.lead, case.k + 8, device="cuda")
        x = wide[..., 4 : 4 + case.k]
    w = torch.randn(case.n, case.k, device="cuda") / case.k**0.5
    b = torch.randn(case.n, device="cuda") if case.bias else None
    return x, w, b


def run_case(case, flush, threshold):
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    x, w, b = _tensors(case)
    ours, schedule = triton_runner(x, w, b)
    inductor = inductor_runner(x, w, b, case)
    reference = F.linear(x.double(), w.double(), None if b is None else b.double())
    ours_abs = (ours().double() - reference).abs().max().item()
    cublas_abs = (F.linear(x, w, b).double() - reference).abs().max().item()
    scale = max(1.0, reference.abs().max().item())
    gate = max(1e-6 * scale, 3 * cublas_abs)
    results = {
        name: summarize(time_us(fn, flush))
        for name, fn in (
            ("ours", ours),
            ("inductor", inductor),
            ("cublas", lambda: F.linear(x, w, b)),
        )
    }
    m = x.numel() // case.k
    speedup = results["inductor"][0] / results["ours"][0]
    return {
        **asdict(case),
        "schedule": schedule,
        "error": {
            "ours_abs": ours_abs,
            "cublas_abs": cublas_abs,
            "ours_normalized": ours_abs / scale,
            "threshold": gate,
            "passes": ours_abs <= gate,
        },
        "flops": 2 * m * case.n * case.k,
        "speedup_vs_inductor": speedup,
        "speedup_vs_cublas": results["cublas"][0] / results["ours"][0],
        "significantly_faster": speedup >= threshold
        and results["ours"][2] < results["inductor"][1],
        **{
            f"{name}_us": dict(zip(("median", "p10", "p90"), r))
            for name, r in results.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--suite", default="shapes", choices=["shapes", "envelope", "gemv_boundary"]
    )
    parser.add_argument("--graph", default="linear", choices=["linear", "core"])
    parser.add_argument("--shard", default="0/1", help="run cases i::n of the suite")
    parser.add_argument("--warm", action="store_true", help="do not flush L2")
    parser.add_argument("--threshold", type=float, default=1.25)
    parser.add_argument("--json", help="append one JSON object per case")
    args = parser.parse_args()

    index, count = map(int, args.shard.split("/"))
    l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
    flush = (
        None
        if args.warm
        else torch.empty(2 * l2_bytes, dtype=torch.uint8, device="cuda")
    )
    print(
        f"{torch.cuda.get_device_name()} {'warm' if args.warm else 'cold'} L2, "
        "median [p10, p90] us: ours, Inductor, cuBLAS"
    )
    cases = {
        "shapes": shape_cases,
        "envelope": envelope_cases,
        "gemv_boundary": gemv_boundary_cases,
    }[args.suite](args.graph)
    for case in cases[index::count]:
        r = run_case(case, flush, args.threshold)
        times = " ".join(
            f"{t['median']:8.1f} [{t['p10']:7.1f},{t['p90']:8.1f}]"
            for t in (r["ours_us"], r["inductor_us"], r["cublas_us"])
        )
        print(
            f"{case.label:13s} {case.graph:6s} {str(case.lead):10s} {case.n:6d} "
            f"{case.k:6d} {'bias' if case.bias else '    '} "
            f"{case.x_layout[:5]:5s} {times} "
            f"{r['speedup_vs_inductor']:6.2f}x {r['speedup_vs_cublas']:6.2f}x "
            f"err {r['error']['ours_abs']:.1e}/{r['error']['cublas_abs']:.1e} "
            f"{'pass' if r['error']['passes'] else 'FAIL'} {r['schedule']:10s} "
            f"{'yes' if r['significantly_faster'] else 'no'}",
            flush=True,
        )
        if args.json:
            with open(args.json, "a") as f:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()

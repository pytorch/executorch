#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark triton::convolution on depthwise and few-channel grouped shapes
against Inductor's Triton conv template (what the CUDA backend lowers a
convolution to otherwise) and cuDNN; grouped shapes also against a sum over
kernel taps, a decomposition of grouped convolutions into elementwise products
and sums that Inductor compiles.

Each call is timed inside a CUDA graph, preceded by a write larger than L2
(cold L2), with timing events around the call only; the graph is replayed
several times and the median and the 10th-90th percentile spread of the
per-call times are reported. ``--warm`` times without the flush.

triton::convolution runs with the schedule and config the backend's compile
would pick: the schedules are timed once under ``autotune_launch_params`` and
Triton's autotuner picks the config. Inductor compiles the convolution with
the backend's options (``max_autotune``, Triton-only conv and GEMM backends).

A shape is "significantly faster" when the median speedup over Inductor is at
least ``--threshold`` (default 1.25x) and our 90th percentile is below
Inductor's 10th. GB/s counts the compulsory traffic: one read of the input,
weight and bias, one write of the output.

Suites: ``representative`` (model shapes, one dtype and layout per run),
``envelope`` (each axis of what the template supports varied around base
shapes, every dtype and layout), ``extent`` (the largest shapes and the
remaining weight layouts), ``layouts`` (those shapes with channels-last
inputs or weights and the contiguous output a graph exported on CUDA records,
which triton::convolution produces through ``output_strides`` and the other
paths through a ``contiguous()`` that Inductor may fuse) and ``grouped`` (1 to
8 input channels per group, channel multipliers 1 to 4). The router's
``direct.worth_routing`` takes the range these suites cover. For grouped shapes
a case is significantly faster when it is against both the template and the
tap sum.

Usage:
    python benchmark_depthwise.py
    python benchmark_depthwise.py --dtype float16 --layout channels_last
    python benchmark_depthwise.py --suite envelope --shard 0/8 --json results.jsonl
    python benchmark_depthwise.py --suite grouped --json grouped.jsonl
"""

import argparse
import json
import statistics
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from executorch.backends.cuda.autotune.launch_params import autotune_launch_params
from executorch.backends.cuda.triton.kernels.conv.conv import (
    convolution,
    force_conv_candidate,
)

# (label, (N, C, H, W), kernel, stride, padding)
SHAPES = [
    ("3x3 x3", (3, 256, 56, 56), 3, 1, 1),
    ("3x3 x3", (3, 512, 28, 28), 3, 1, 1),
    ("3x3 x3", (3, 1024, 14, 14), 3, 1, 1),
    ("3x3 x3", (3, 2048, 7, 7), 3, 1, 1),
    ("large 7x7", (1, 128, 128, 160), 7, 1, 3),
    ("7x7", (2, 160, 32, 40), 7, 1, 3),
    ("9x9", (2, 304, 16, 20), 9, 1, 4),
    ("5x5", (2, 96, 64, 80), 5, 1, 2),
    ("3x3", (2, 48, 128, 160), 3, 1, 1),
    ("3x3", (2, 48, 64, 80), 3, 1, 1),
    ("3x3", (2, 54, 32, 40), 3, 1, 1),
    ("3x3", (2, 76, 16, 20), 3, 1, 1),
    ("7x7 x8", (8, 96, 56, 56), 7, 1, 3),
    ("7x7 x8", (8, 192, 28, 28), 7, 1, 3),
    ("7x7 x8", (8, 384, 14, 14), 7, 1, 3),
    ("7x7 x8", (8, 768, 7, 7), 7, 1, 3),
    ("3x3 x8", (8, 32, 112, 112), 3, 1, 1),
    ("3x3 s2 x8", (8, 96, 112, 112), 3, 2, 1),
    ("3x3 x8", (8, 144, 56, 56), 3, 1, 1),
    ("3x3 s2 x8", (8, 144, 56, 56), 3, 2, 1),
    ("3x3 x8", (8, 192, 28, 28), 3, 1, 1),
    ("3x3 x8", (8, 384, 14, 14), 3, 1, 1),
    ("3x3 x8", (8, 576, 14, 14), 3, 1, 1),
    ("3x3 x8", (8, 960, 7, 7), 3, 1, 1),
    ("batch 1 3x3", (1, 32, 8, 8), 3, 1, 1),
]


@dataclass(frozen=True)
class Case:
    label: str
    shape: tuple[int, int, int, int]
    kernel: tuple[int, int]
    stride: tuple[int, int]
    before: tuple[int, int]
    after: tuple[int, int]
    dtype: str = "float32"
    layout: str = "nchw"
    # Input channels per group and output channels per group; (1, 1) is a
    # depthwise convolution.
    group_channels: int = 1
    multiplier: int = 1

    @property
    def groups(self):
        return self.shape[1] // self.group_channels


def representative_cases(dtype, layout):
    return [
        Case(label, shape, (k, k), (s, s), (p, p), (p, p), dtype, layout)
        for label, shape, k, s, p in SHAPES
    ]


def _same(kernel):
    before = tuple((k - 1) // 2 for k in kernel)
    return before, tuple(k - 1 - b for k, b in zip(kernel, before))


def envelope_cases():
    """Shapes spanning what ``direct.supports()`` accepts: each axis (kernel,
    stride, padding, channels, spatial size, batch) is varied around a few
    base shapes, in every dtype with NCHW and channels-last inputs, plus
    channels-last weights in fp32."""
    geometries = []
    for base in ((1, 16, 16, 20), (2, 64, 56, 56), (8, 256, 28, 28)):
        for kernel in (
            (1, 1),
            (3, 3),
            (5, 5),
            (7, 7),
            (9, 9),
            (1, 9),
            (9, 1),
            (3, 5),
            (7, 3),
            (4, 4),
            (2, 6),
        ):
            geometries.append(("kernel", base, kernel, (1, 1), *_same(kernel)))
        for stride in ((2, 2), (3, 3), (1, 2), (2, 1), (3, 1)):
            geometries.append(("stride", base, (3, 3), stride, (1, 1), (1, 1)))
        for before, after in (
            ((0, 0), (0, 0)),
            ((4, 4), (4, 4)),
            ((0, 0), (4, 4)),
            ((4, 4), (0, 0)),
            ((1, 3), (3, 0)),
        ):
            geometries.append(("padding", base, (5, 5), (1, 1), before, after))
    for channels in (1, 2, 3, 8, 32, 96, 512, 2048):
        geometries.append(
            ("channels", (2, channels, 28, 28), (3, 3), (1, 1), (1, 1), (1, 1))
        )
    for spatial in (
        (1, 1),
        (2, 2),
        (7, 7),
        (1, 1024),
        (1024, 1),
        (256, 256),
        (512, 512),
    ):
        geometries.append(
            ("spatial", (1, 32, *spatial), (3, 3), (1, 1), (1, 1), (1, 1))
        )
    for batch in (4, 32, 128):
        geometries.append(
            ("batch", (batch, 64, 28, 28), (3, 3), (1, 1), (1, 1), (1, 1))
        )
    cases = []
    for label, shape, kernel, stride, before, after in geometries:
        for dtype in ("float32", "float16", "bfloat16"):
            for layout in ("nchw", "channels_last"):
                cases.append(
                    Case(label, shape, kernel, stride, before, after, dtype, layout)
                )
        cases.append(
            Case(label, shape, kernel, stride, before, after, "float32", "weight_cl")
        )
    return cases


def layout_cases():
    """The envelope and extent shapes whose input or weight is channels-last,
    with the contiguous output a graph exported on CUDA records for them."""
    follow = {
        "channels_last": "cl_to_nchw",
        "weight_cl": "weight_cl_to_nchw",
        "both_cl": "both_cl_to_nchw",
    }
    return [
        Case(**{**asdict(case), "layout": follow[case.layout]})
        for case in envelope_cases() + extent_cases()
        if case.layout in follow
    ]


def extent_cases():
    """The largest shapes and the remaining weight layouts the router takes:
    4096 channels, 2048x2048 images, batch 256, ~50M-element inputs, and
    channels-last weights with NCHW or channels-last inputs in every dtype."""
    cases = []
    for shape, kernel in (
        ((1, 4096, 32, 32), (3, 3)),
        ((2, 4096, 14, 14), (7, 7)),
        ((1, 16, 1024, 1024), (3, 3)),
        ((1, 4, 2048, 2048), (5, 5)),
        ((256, 32, 28, 28), (3, 3)),
        ((8, 96, 224, 224), (3, 3)),
        ((32, 128, 112, 112), (3, 3)),
        ((4, 512, 96, 96), (9, 9)),
    ):
        for dtype in ("float32", "float16"):
            for layout in ("nchw", "channels_last"):
                cases.append(
                    Case("extent", shape, kernel, (1, 1), *_same(kernel), dtype, layout)
                )
    for kernel in ((3, 3), (7, 7), (1, 9), (4, 4)):
        for dtype in ("float32", "float16", "bfloat16"):
            for layout in ("weight_cl", "both_cl"):
                if (dtype, layout) != ("float32", "weight_cl"):
                    cases.append(
                        Case(
                            "weights",
                            (2, 64, 56, 56),
                            kernel,
                            (1, 1),
                            *_same(kernel),
                            dtype,
                            layout,
                        )
                    )
    return cases


def grouped_cases():
    """Grouped convolutions with 1 to 8 input channels per group and channel
    multipliers 1 to 4 at one size: every combination with 3x3, 5x5 and 7x7
    kernels and strides 1 and 2 in fp32 and bf16, odd group widths, and
    channels-last inputs; then four combinations from the smallest to the
    largest sizes."""
    shape = (8, 192, 28, 28)
    combinations = [(g, m) for g in (1, 2, 4, 8) for m in (1, 2, 4) if (g, m) != (1, 1)]
    geometries = [
        (g, m, k, s, "nchw")
        for g, m in combinations
        for k in (3, 5, 7)
        for s in (1, 2)
    ]
    geometries += [(g, m, 3, s, "nchw") for g in (3, 6) for m in (1, 2) for s in (1, 2)]
    geometries += [(g, m, 3, 1, "channels_last") for g, m in combinations]
    sized = [(shape, *geometry) for geometry in geometries]
    for size in (
        (1, 24, 8, 8),
        (32, 192, 28, 28),
        (1, 96, 224, 224),
        (2, 1536, 14, 14),
        (8, 384, 112, 112),
    ):
        sized += [
            (size, g, m, k, 1, "nchw")
            for g, m in ((1, 2), (2, 1), (4, 2), (8, 2))
            for k in (3, 7)
        ]
    return [
        Case(
            f"g{g} m{m}",
            size,
            (k, k),
            (s, s),
            (k // 2, k // 2),
            (k // 2, k // 2),
            dtype,
            layout,
            g,
            m,
        )
        for size, g, m, k, s, layout in sized
        for dtype in ("float32", "bfloat16")
    ]


_MAX_TAP_SUM_GROUP_CHANNELS = 8


def tap_sum_conv2d(input, weight, bias, stride, padding, groups):
    """A grouped convolution with few channels per group as a sum over kernel
    taps of elementwise products, accumulated in fp32 (symmetric padding, no
    dilation)."""
    if groups == 1 or weight.shape[1] > _MAX_TAP_SUM_GROUP_CHANNELS:
        return F.conv2d(input, weight, bias, stride, padding, 1, groups)
    dtype = input.dtype
    x = input.float()
    batch = x.shape[0]
    out_channels, group_channels, kernel_h, kernel_w = weight.shape
    out_h, out_w = (
        (size + 2 * pad - kernel) // step + 1
        for size, pad, kernel, step in zip(
            x.shape[2:], padding, (kernel_h, kernel_w), stride
        )
    )
    if any(padding):
        x = F.pad(x, [padding[1], padding[1], padding[0], padding[0]])
    x = x.reshape(batch, groups, 1, group_channels, x.shape[2], x.shape[3])
    w = weight.float().reshape(
        1, groups, out_channels // groups, group_channels, kernel_h, kernel_w
    )
    total = None
    for i in range(kernel_h):
        for j in range(kernel_w):
            window = x[
                ...,
                i : i + stride[0] * (out_h - 1) + 1 : stride[0],
                j : j + stride[1] * (out_w - 1) + 1 : stride[1],
            ]
            term = (window * w[..., i, j, None, None]).sum(3)
            total = term if total is None else total + term
    out = total.reshape(batch, out_channels, out_h, out_w)
    if bias is not None:
        out = out + bias.float().reshape(1, out_channels, 1, 1)
    return out.to(dtype)


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


def _pad(x, case):
    return F.pad(x, [case.before[1], case.after[1], case.before[0], case.after[0]])


def _contiguous_output(case):
    return case.layout.endswith("_to_nchw")


def triton_runner(x, w, b, case, output_strides=None):
    args = (
        list(case.stride),
        list(case.before),
        list(case.after),
        [1, 1],
        False,
        [0, 0],
        case.groups,
        output_strides,
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), autotune_launch_params() as stats:
        convolution(x, w, b, *args)
    stream.synchronize()
    schedule = stats.picks[0].choice if stats.picks else "dw_flat"

    def run():
        with force_conv_candidate(schedule):
            return convolution(x, w, b, *args)

    return run, schedule


def aten_conv(x, w, b, case):
    """What the exported graph holds: a separate pad for asymmetric padding, and
    a contiguous output where the graph records one."""
    if case.before == case.after:
        out = F.conv2d(x, w, b, case.stride, case.before, 1, case.groups)
    else:
        out = F.conv2d(_pad(x, case), w, b, case.stride, 0, 1, case.groups)
    return out.contiguous() if _contiguous_output(case) else out


def tap_sum(x, w, b, case):
    return tap_sum_conv2d(x, w, b, case.stride, case.before, case.groups)


# The Inductor options of the backend's AOTI compile that affect kernels.
INDUCTOR_OPTIONS = {
    "loop_ordering_after_fusion": False,
    "emulate_precision_casts": True,
    "max_autotune": True,
    "max_autotune_conv_backends": "TRITON",
    "max_autotune_gemm_backends": "TRITON",
}


def inductor_runner(x, w, b, case, conv=aten_conv):
    torch._dynamo.reset()
    with torch._inductor.config.patch(INDUCTOR_OPTIONS):
        compiled = torch.compile(
            lambda x, w, b: conv(x, w, b, case), dynamic=False, fullgraph=True
        )
        compiled(x, w, b)
    return lambda: compiled(x, w, b)


def run_case(case, flush, threshold):
    torch.manual_seed(0)
    dtype = getattr(torch, case.dtype)
    out_channels = case.groups * case.multiplier
    x = torch.randn(case.shape, device="cuda", dtype=dtype)
    w = torch.randn(
        out_channels, case.group_channels, *case.kernel, device="cuda", dtype=dtype
    )
    w = w / (case.group_channels * case.kernel[0] * case.kernel[1]) ** 0.5
    b = torch.randn(out_channels, device="cuda", dtype=dtype)
    if case.layout in ("channels_last", "both_cl", "cl_to_nchw", "both_cl_to_nchw"):
        x = x.to(memory_format=torch.channels_last)
    if case.layout in ("weight_cl", "both_cl", "weight_cl_to_nchw", "both_cl_to_nchw"):
        w = w.to(memory_format=torch.channels_last)
    output_strides = None
    if _contiguous_output(case):
        shape = aten_conv(x[:1].double(), w.double(), None, case).shape
        output_strides = list(torch.empty((case.shape[0], *shape[1:])).stride())
    ours, schedule = triton_runner(x, w, b, case, output_strides)
    runners = [
        ("ours", ours),
        ("inductor", inductor_runner(x, w, b, case)),
        ("cudnn", lambda: aten_conv(x, w, b, case)),
    ]
    grouped = case.groups != case.shape[1] or case.multiplier != 1
    if grouped:
        runners.append(("tap_sum", inductor_runner(x, w, b, case, tap_sum)))
    reference = aten_conv(x.double(), w.double(), b.double(), case)
    error = (ours().double() - reference).abs().max().item()
    results = {name: summarize(time_us(fn, flush)) for name, fn in runners}
    nbytes = sum(t.numel() * t.element_size() for t in (x, w, b, ours()))
    speedup = results["inductor"][0] / results["ours"][0]
    significant = all(
        results[name][0] / results["ours"][0] >= threshold
        and results["ours"][2] < results[name][1]
        for name in ("inductor", "tap_sum") if name in results
    )
    extra = {}
    if grouped:
        extra["speedup_vs_tap_sum"] = results["tap_sum"][0] / results["ours"][0]
    return {
        **asdict(case),
        "schedule": schedule,
        "max_abs_error": error,
        "bytes": nbytes,
        "speedup_vs_inductor": speedup,
        "speedup_vs_cudnn": results["cudnn"][0] / results["ours"][0],
        **extra,
        "significantly_faster": significant,
        **{
            f"{name}_us": dict(zip(("median", "p10", "p90"), r))
            for name, r in results.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--suite",
        default="representative",
        choices=["representative", "envelope", "extent", "layouts", "grouped"],
    )
    parser.add_argument(
        "--dtype", default="float32", choices=["float32", "float16", "bfloat16"]
    )
    parser.add_argument("--layout", default="nchw", choices=["nchw", "channels_last"])
    parser.add_argument("--shard", default="0/1", help="run cases i::n of the suite")
    parser.add_argument("--warm", action="store_true", help="do not flush L2")
    parser.add_argument("--threshold", type=float, default=1.25)
    parser.add_argument("--json", help="append one JSON object per case")
    args = parser.parse_args()

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    cases = {
        "representative": lambda: representative_cases(args.dtype, args.layout),
        "envelope": envelope_cases,
        "extent": extent_cases,
        "layouts": layout_cases,
        "grouped": grouped_cases,
    }[args.suite]()
    index, count = map(int, args.shard.split("/"))
    l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
    flush = (
        None
        if args.warm
        else torch.empty(2 * l2_bytes, dtype=torch.uint8, device="cuda")
    )
    print(
        f"{torch.cuda.get_device_name()} {'warm' if args.warm else 'cold'} L2, "
        "median [p10, p90] us"
    )
    print(
        f"{'case':10s} {'dtype':9s} {'layout':13s} {'input':20s} {'k':6s} {'s':6s} "
        f"{'pad (before/after)':18s} {'ours':>22s} {'inductor':>22s} {'cudnn':>22s} "
        f"{'vs ind':>8s} {'vs cudnn':>8s} {'GB/s':>6s} schedule  significant "
        "[tap sum, vs tap sum]"
    )
    for case in cases[index::count]:
        r = run_case(case, flush, args.threshold)
        times = " ".join(
            f"{t['median']:7.1f} [{t['p10']:5.1f},{t['p90']:6.1f}]"
            for t in (r["ours_us"], r["inductor_us"], r["cudnn_us"])
        )
        geometry = (
            f"{case.kernel[0]}x{case.kernel[1]:<4d} {case.stride[0]}x{case.stride[1]:<4d} "
            f"{str(case.before) + '/' + str(case.after):18s}"
        )
        print(
            f"{case.label:10s} {case.dtype:9s} {case.layout:13s} {str(case.shape):20s} "
            f"{geometry} {times} {r['speedup_vs_inductor']:7.2f}x "
            f"{r['speedup_vs_cudnn']:7.2f}x {r['bytes'] / r['ours_us']['median'] / 1e3:6.0f} "
            f"{r['schedule']:9s} {'yes' if r['significantly_faster'] else 'no ':11s}"
            + (
                f" {r['tap_sum_us']['median']:7.1f} {r['speedup_vs_tap_sum']:7.2f}x"
                if "tap_sum_us" in r
                else ""
            ),
            flush=True,
        )
        if args.json:
            with open(args.json, "a") as f:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()

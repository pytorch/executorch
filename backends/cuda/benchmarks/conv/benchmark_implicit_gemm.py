#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark the implicit-GEMM template of triton::convolution on dense
convolutions against Inductor's Triton conv template (what the CUDA backend
lowers a convolution to otherwise) and cuDNN.

Timing, the compile options of the two Triton paths and "significantly
faster" are those of ``benchmark_depthwise.py``: cold L2, CUDA graph replay,
median and 10th-90th percentile spread, a median speedup of at least
``--threshold`` with our 90th percentile below Inductor's 10th. triton::convolution
runs with the schedule its compile would pick, for the targets of this GPU and
the TF32 setting the backend reads (``conv_allows_tf32``). Each case also
records the kernels Inductor launched, which shows the path it took (e.g. a
1x1 convolution as a matrix multiplication).

Precisions: ``fp32`` (TF32 off everywhere), ``tf32`` (TF32 allowed; needs a
PyTorch whose Inductor conv templates can use TF32), ``fp16`` and ``bf16``.

Suites: ``shapes`` (dense and kernel == stride convolutions at representative
shapes), ``envelope`` (each axis of what the
template supports varied around base shapes, in every precision and input
layout) and ``boundary`` (every geometry the router takes at every corner of
its limits, ``implicit_gemm.boundary_calls()``). The router's
``implicit_gemm.worth_routing`` takes the range these suites show to be
significantly faster.

Usage:
    python benchmark_implicit_gemm.py --precision bf16
    python benchmark_implicit_gemm.py --suite envelope --shard 0/8 --json out.jsonl
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
from executorch.backends.cuda.triton.kernels.conv.core import implicit_gemm
from executorch.backends.cuda.triton.kernels.conv.core.common import conv_allows_tf32


@dataclass(frozen=True)
class Case:
    label: str
    shape: tuple[int, ...]
    out_channels: int
    kernel: tuple[int, ...]
    stride: tuple[int, ...]
    before: tuple[int, ...]
    after: tuple[int, ...]
    bias: bool
    precision: str = "fp32"
    layout: str = "nchw"


def _case(label, shape, out_channels, kernel, stride, padding, bias, precision):
    rank = len(shape) - 2

    def expand(v):
        return tuple(v) if isinstance(v, tuple) else (v,) * rank

    padding = expand(padding)
    return Case(
        label,
        shape,
        out_channels,
        expand(kernel),
        expand(stride),
        padding,
        padding,
        bias,
        precision,
    )


# (group, label, input shape, C_out, kernel, stride, padding, bias)
SHAPES = [
    (
        "kernel_stride_3d",
        "2x14x14 x256",
        (256, 3, 2, 14, 14),
        1280,
        (2, 14, 14),
        (2, 14, 14),
        0,
        False,
    ),
    (
        "kernel_stride_3d",
        "2x14x14 x324",
        (324, 3, 2, 14, 14),
        1280,
        (2, 14, 14),
        (2, 14, 14),
        0,
        False,
    ),
    (
        "kernel_stride_3d",
        "2x14x14 x2048",
        (2048, 3, 2, 14, 14),
        1280,
        (2, 14, 14),
        (2, 14, 14),
        0,
        False,
    ),
    (
        "kernel_stride_3d",
        "2x16x16 x256",
        (256, 3, 2, 16, 16),
        1024,
        (2, 16, 16),
        (2, 16, 16),
        0,
        True,
    ),
    (
        "kernel_stride_3d",
        "2x16x16 x392",
        (392, 3, 2, 16, 16),
        1024,
        (2, 16, 16),
        (2, 16, 16),
        0,
        True,
    ),
    (
        "kernel_stride_3d",
        "2x16x16 x512",
        (512, 3, 2, 16, 16),
        1024,
        (2, 16, 16),
        (2, 16, 16),
        0,
        True,
    ),
    (
        "kernel_stride_3d",
        "2x16x16 x768",
        (768, 3, 2, 16, 16),
        1024,
        (2, 16, 16),
        (2, 16, 16),
        0,
        True,
    ),
    (
        "kernel_stride_3d",
        "1x2x2, 48 ch",
        (1, 48, 1, 14, 28),
        3072,
        (1, 2, 2),
        (1, 2, 2),
        0,
        True,
    ),
    (
        "kernel_stride_3d",
        "1x4x4, 2x544x736",
        (1, 3, 2, 544, 736),
        256,
        (1, 4, 4),
        (1, 4, 4),
        0,
        True,
    ),
    ("kernel_stride_2d", "14x14 224x224", (1, 3, 224, 224), 1152, 14, 14, 0, True),
    ("kernel_stride_2d", "14x14 448x448", (1, 3, 448, 448), 1152, 14, 14, 0, True),
    ("kernel_stride_2d", "14x14 448x448 x3", (3, 3, 448, 448), 1024, 14, 14, 0, True),
    ("kernel_stride_2d", "16x16 512x512", (1, 3, 512, 512), 768, 16, 16, 0, True),
    ("kernel_stride_2d", "16x16 224x224 x2", (2, 3, 224, 224), 768, 16, 16, 0, False),
    ("kernel_stride_2d", "16x16 256x256", (1, 3, 256, 256), 768, 16, 16, 0, True),
    ("3ch_7x7", "7x7 s4 224x224 x3", (3, 3, 224, 224), 256, 7, 4, 3, True),
    ("3ch_7x7", "7x7 s2 480x640", (1, 3, 480, 640), 64, 7, 2, 3, False),
    ("3ch_7x7", "7x7 s2 84x84 x2", (2, 3, 84, 84), 64, 7, 2, 3, False),
    ("3ch_7x7", "7x7 s2 84x84 x5", (5, 3, 84, 84), 64, 7, 2, 3, False),
    ("3x3", "64 ch 120x160", (1, 64, 120, 160), 64, 3, 1, 1, False),
    ("3x3", "128 ch 60x80", (1, 128, 60, 80), 128, 3, 1, 1, False),
    ("3x3", "256 ch 30x40", (1, 256, 30, 40), 256, 3, 1, 1, False),
    ("3x3", "512 ch 15x20", (1, 512, 15, 20), 512, 3, 1, 1, False),
    ("3x3", "64 ch 21x21 x2", (2, 64, 21, 21), 64, 3, 1, 1, False),
    ("3x3", "128 ch 11x11 x2", (2, 128, 11, 11), 128, 3, 1, 1, False),
    ("3x3", "256 ch 6x6 x2", (2, 256, 6, 6), 256, 3, 1, 1, False),
    ("3x3", "512 ch 3x3 x2", (2, 512, 3, 3), 512, 3, 1, 1, False),
    ("3x3", "64 ch 21x21 x5", (5, 64, 21, 21), 64, 3, 1, 1, False),
    ("3x3", "512 ch 3x3 x5", (5, 512, 3, 3), 512, 3, 1, 1, False),
    ("3x3_s2", "64-128 ch 120x160", (1, 64, 120, 160), 128, 3, 2, 1, False),
    ("3x3_s2", "128-256 ch 60x80", (1, 128, 60, 80), 256, 3, 2, 1, False),
    ("3x3_s2", "256-512 ch 30x40", (1, 256, 30, 40), 512, 3, 2, 1, False),
    ("3x3_s2", "64-128 ch 21x21 x2", (2, 64, 21, 21), 128, 3, 2, 1, False),
    ("3x3_s2", "128-256 ch 11x11 x2", (2, 128, 11, 11), 256, 3, 2, 1, False),
    ("3x3_s2", "256-512 ch 6x6 x2", (2, 256, 6, 6), 512, 3, 2, 1, False),
    ("3x3_s2", "256-512 ch 56x56 x3", (3, 256, 56, 56), 512, 3, 2, 1, True),
    ("3x3_s2", "512-1024 ch 28x28 x3", (3, 512, 28, 28), 1024, 3, 2, 1, True),
    ("3x3_s2", "1024-2048 ch 14x14 x3", (3, 1024, 14, 14), 2048, 3, 2, 1, True),
    ("1x1", "512 ch 15x20", (1, 512, 15, 20), 512, 1, 1, 0, True),
    ("1x1", "512-32 ch 3x3 x2", (2, 512, 3, 3), 32, 1, 1, 0, True),
    ("1x1", "512-32 ch 3x3 x5", (5, 512, 3, 3), 32, 1, 1, 0, True),
    ("1x1", "640-1920 ch 14x28", (1, 640, 14, 28), 1920, 1, 1, 0, True),
    ("1x1", "640 ch 14x28", (1, 640, 14, 28), 640, 1, 1, 0, True),
    ("1x1", "640-1920 ch 8x8 x2", (2, 640, 8, 8), 1920, 1, 1, 0, True),
    ("1x1_s2", "64-128 ch 120x160", (1, 64, 120, 160), 128, 1, 2, 0, False),
    ("1x1_s2", "128-256 ch 60x80", (1, 128, 60, 80), 256, 1, 2, 0, False),
    ("1x1_s2", "256-512 ch 30x40", (1, 256, 30, 40), 512, 1, 2, 0, False),
    ("1x1_s2", "64-128 ch 21x21 x2", (2, 64, 21, 21), 128, 1, 2, 0, False),
    ("1x1_s2", "128-256 ch 11x11 x2", (2, 128, 11, 11), 256, 1, 2, 0, False),
    ("1x1_s2", "256-512 ch 6x6 x2", (2, 256, 6, 6), 512, 1, 2, 0, False),
    ("32ch_s2", "3-32 ch 7x7 84x84", (1, 3, 84, 84), 32, 7, 2, 0, True),
    ("32ch_s2", "32 ch 5x5 39x39", (1, 32, 39, 39), 32, 5, 2, 0, True),
    ("32ch_s2", "32 ch 3x3 18x18", (1, 32, 18, 18), 32, 3, 2, 0, True),
    ("32ch_s2", "32 ch 3x3 8x8", (1, 32, 8, 8), 32, 3, 2, 0, True),
    ("32ch_s2", "3-32 ch 7x7 256x256 x2", (2, 3, 256, 256), 32, 7, 2, 0, True),
    ("32ch_s2", "32 ch 5x5 125x125 x2", (2, 32, 125, 125), 32, 5, 2, 0, True),
    ("32ch_s2", "32 ch 3x3 61x61 x2", (2, 32, 61, 61), 32, 3, 2, 0, True),
    ("32ch_s2", "32 ch 3x3 30x30 x2", (2, 32, 30, 30), 32, 3, 2, 0, True),
]


def shape_cases(precision, layout):
    return [
        Case(
            **{**asdict(_case(f"{group} {label}", *rest, precision)), "layout": layout}
        )
        for group, label, *rest in SHAPES
    ]


def _envelope_geometries():
    """(axis, input shape, C_out, kernel, stride, padding) around the routed
    range (fewer than 16 input channels), 1x1 convolutions, and a few dense
    convolutions with more input channels."""
    geometries = []
    for channels in (1, 2, 3, 4, 6, 8, 12, 15, 16, 32):
        geometries.append(("channels", (2, channels, 56, 56), 64, 3, 1, 1))
        geometries.append(("channels", (1, channels, 224, 224), 128, 7, 2, 3))
    for kernel, stride, padding in (
        (2, 2, 0),
        (3, 2, 0),
        (3, 1, 2),
        (4, 4, 0),
        (5, 2, 2),
        (7, 2, 3),
        (7, 4, 3),
        (8, 8, 0),
        (11, 4, 2),
        (14, 14, 0),
        (16, 16, 0),
        (32, 32, 0),
        ((1, 7), 1, (0, 3)),
        ((3, 5), (2, 1), (1, 2)),
    ):
        for channels in (3, 12):
            geometries.append(
                ("kernel", (2, channels, 224, 224), 128, kernel, stride, padding)
            )
    for out_channels in (1, 8, 16, 32, 48, 96, 192, 384, 768, 1152, 1536, 3072):
        geometries.append(("out_channels", (1, 3, 112, 112), out_channels, 3, 2, 1))
        geometries.append(("out_channels", (1, 3, 224, 224), out_channels, 16, 16, 0))
    for spatial in (
        (1, 1),
        (3, 3),
        (7, 9),
        (33, 47),
        (1, 1024),
        (1024, 1),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    ):
        geometries.append(("spatial", (1, 3, *spatial), 64, 3, 1, 1))
    for batch in (1, 4, 16, 64, 256):
        geometries.append(("batch", (batch, 3, 56, 56), 64, 3, 1, 1))
    for shape, out_channels, kernel, stride, padding in (
        ((1, 3, 2, 14, 14), 1280, (2, 14, 14), (2, 14, 14), 0),
        ((64, 3, 2, 14, 14), 1280, (2, 14, 14), (2, 14, 14), 0),
        ((4096, 3, 2, 16, 16), 1024, (2, 16, 16), (2, 16, 16), 0),
        ((1, 3, 8, 256, 256), 256, (1, 4, 4), (1, 4, 4), 0),
        ((1, 12, 4, 64, 64), 1024, (1, 2, 2), (1, 2, 2), 0),
        ((2, 3, 9, 64, 64), 64, 3, 1, 1),
        ((1, 3, 16, 112, 112), 64, (3, 7, 7), (1, 2, 2), (1, 3, 3)),
        ((1, 8, 5, 40, 40), 32, (3, 3, 3), (2, 2, 2), 1),
    ):
        geometries.append(("3-D", shape, out_channels, kernel, stride, padding))
    for channels, out_channels in ((64, 32), (64, 512), (640, 1920)):
        for spatial in ((3, 3), (15, 20), (60, 80)):
            for stride in (1, 2):
                geometries.append(
                    ("1x1", (2, channels, *spatial), out_channels, 1, stride, 0)
                )
    for channels in (16, 64, 256):
        geometries.append(("dense", (2, channels, 56, 56), channels, 3, 1, 1))
    return geometries


def envelope_cases(precisions):
    """``_envelope_geometries`` in each precision, with NCHW and channels-last
    inputs, and in FP32 also with a channels-last weight."""
    cases = []
    for label, shape, out_channels, kernel, stride, padding in _envelope_geometries():
        for precision in precisions:
            layouts = ["nchw", "channels_last"]
            if precision == "fp32":
                layouts.append("weight_cl")
            for layout in layouts:
                case = _case(
                    label, shape, out_channels, kernel, stride, padding, True, precision
                )
                cases.append(Case(**{**asdict(case), "layout": layout}))
    return cases


def boundary_cases(precision, layout):
    """``implicit_gemm.boundary_calls()``: every routed geometry at every corner
    of its limits."""
    return [
        Case(
            "boundary",
            shape,
            out_channels,
            kernel,
            stride,
            padding,
            padding,
            True,
            precision,
            layout,
        )
        for shape, out_channels, kernel, stride, padding in (
            implicit_gemm.boundary_calls()
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


def _args(case):
    rank = len(case.kernel)
    return (
        list(case.stride),
        list(case.before),
        list(case.after),
        [1] * rank,
        False,
        [0] * rank,
        1,
    )


def triton_runner(x, w, b, case):
    args = _args(case)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), autotune_launch_params() as stats:
        convolution(x, w, b, *args)
    stream.synchronize()
    schedule = stats.picks[0].choice if stats.picks else "igemm_ffma"

    def run():
        with force_conv_candidate(schedule):
            return convolution(x, w, b, *args)

    return run, schedule


def aten_conv(x, w, b, case):
    """What the exported graph holds: a separate pad for asymmetric padding."""
    conv = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[len(case.kernel)]
    if case.before == case.after:
        return conv(x, w, b, case.stride, case.before)
    pad = [p for pair in zip(reversed(case.before), reversed(case.after)) for p in pair]
    return conv(F.pad(x, pad), w, b, case.stride)


def inductor_runner(x, w, b, case):
    torch._dynamo.reset()
    with torch._inductor.config.patch(
        max_autotune=True,
        max_autotune_conv_backends="TRITON",
        max_autotune_gemm_backends="TRITON",
    ):
        compiled = torch.compile(
            lambda x, w, b: aten_conv(x, w, b, case), dynamic=False, fullgraph=True
        )
        compiled(x, w, b)
    return lambda: compiled(x, w, b)


def kernel_names(fn):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        fn()
        torch.cuda.synchronize()
    return sorted(
        {
            e.name
            for e in prof.events()
            if e.device_type == torch.autograd.DeviceType.CUDA
        }
    )


def _tensors(case):
    torch.manual_seed(0)
    dtype = {"fp32": torch.float32, "tf32": torch.float32, "fp16": torch.float16}.get(
        case.precision, torch.bfloat16
    )
    channels = case.shape[1]
    fan_in = channels * torch.Size(case.kernel).numel()
    x = torch.randn(case.shape, device="cuda", dtype=dtype)
    w = (
        torch.randn(
            case.out_channels, channels, *case.kernel, device="cuda", dtype=dtype
        )
        / fan_in**0.5
    )
    b = (
        torch.randn(case.out_channels, device="cuda", dtype=dtype)
        if case.bias
        else None
    )
    memory_format = torch.channels_last if x.dim() == 4 else torch.channels_last_3d
    if case.layout == "channels_last":
        x = x.to(memory_format=memory_format)
    if case.layout == "weight_cl":
        w = w.to(memory_format=memory_format)
    return x, w, b


def run_case(case, flush, threshold):
    tf32 = case.precision == "tf32"
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    if conv_allows_tf32() != tf32:
        raise RuntimeError(
            f"{case.precision}: this PyTorch's Inductor conv templates "
            f"{'cannot' if tf32 else 'must not'} use TF32"
        )
    x, w, b = _tensors(case)
    ours, schedule = triton_runner(x, w, b, case)
    inductor = inductor_runner(x, w, b, case)
    reference = aten_conv(
        x.double(), w.double(), None if b is None else b.double(), case
    )
    error = (ours().double() - reference).abs().max().item() / max(
        1.0, reference.abs().max().item()
    )
    results = {
        name: summarize(time_us(fn, flush))
        for name, fn in (
            ("ours", ours),
            ("inductor", inductor),
            ("cudnn", lambda: aten_conv(x, w, b, case)),
        )
    }
    out = ours()
    nbytes = sum(t.numel() * t.element_size() for t in (x, w, b, out) if t is not None)
    flops = 2 * out.numel() * w[0].numel()
    speedup = results["inductor"][0] / results["ours"][0]
    significant = speedup >= threshold and results["ours"][2] < results["inductor"][1]
    return {
        **asdict(case),
        "schedule": schedule,
        "inductor_kernels": kernel_names(inductor),
        "max_rel_error": error,
        "bytes": nbytes,
        "flops": flops,
        "speedup_vs_inductor": speedup,
        "speedup_vs_cudnn": results["cudnn"][0] / results["ours"][0],
        "significantly_faster": significant,
        **{
            f"{name}_us": dict(zip(("median", "p10", "p90"), r))
            for name, r in results.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--suite", default="shapes", choices=["shapes", "envelope", "boundary"]
    )
    parser.add_argument(
        "--precision",
        default="fp32",
        help="fp32, tf32, fp16 or bf16; the envelope suite takes a comma-separated list",
    )
    parser.add_argument("--layout", default="nchw", choices=["nchw", "channels_last"])
    parser.add_argument("--shard", default="0/1", help="run cases i::n of the suite")
    parser.add_argument("--warm", action="store_true", help="do not flush L2")
    parser.add_argument("--threshold", type=float, default=1.25)
    parser.add_argument("--json", help="append one JSON object per case")
    args = parser.parse_args()

    cases = {
        "shapes": lambda: shape_cases(args.precision, args.layout),
        "envelope": lambda: envelope_cases(args.precision.split(",")),
        "boundary": lambda: boundary_cases(args.precision, args.layout),
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
    for case in cases[index::count]:
        r = run_case(case, flush, args.threshold)
        times = " ".join(
            f"{t['median']:8.1f} [{t['p10']:7.1f},{t['p90']:8.1f}]"
            for t in (r["ours_us"], r["inductor_us"], r["cudnn_us"])
        )
        print(
            f"{case.label:32s} {case.precision:4s} {case.layout:13s} "
            f"{str(case.shape):22s} {case.out_channels:5d} {str(case.kernel):12s} "
            f"{str(case.stride):12s} {str(case.before) + '/' + str(case.after):20s} "
            f"{times} {r['speedup_vs_inductor']:6.2f}x {r['speedup_vs_cudnn']:6.2f}x "
            f"{r['flops'] / r['ours_us']['median'] / 1e6:7.2f} {r['schedule']:10s} "
            f"{'yes' if r['significantly_faster'] else 'no'}",
            flush=True,
        )
        if args.json:
            with open(args.json, "a") as f:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()

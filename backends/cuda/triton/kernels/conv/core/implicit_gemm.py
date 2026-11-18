# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Implicit-GEMM template: dense (groups 1) 1-D, 2-D and 3-D forward and
transposed convolution.

The convolution is a GEMM of the output pixels (``M = N * OD * OH * OW``) by the
output channels, reducing over ``K = C_in * KD * KH * KW``; input tiles are
gathered from the input as the GEMM walks K. The kernel is written once for
(D, H, W): lower ranks have unit leading dims, which, like every shape, stride
and geometry argument, are constexpr and compile away; so does every bounds
check the geometry cannot fail (all of them when kernel == stride and
there is no padding). Tiles are autotuned. K is walked in
one of two orders:

- tap-major: for each kernel tap, blocks of input channels, sharing the bounds
  check of the tap. Used when the input has at least ``TAP_MAJOR_MIN_CHANNELS``
  channels.
- flat: one index over (channel, tap), decoded per element, which wastes no
  lanes on inputs with few channels, such as 3.

A transposed convolution is the same GEMM over its output pixels: output index
``o`` reads, at tap ``k``, input index ``(o + padding_before - k) / stride``
when that is an integer inside the input, and its ``[C_in, C_out, ...]``
weight is read with its first two strides swapped.

Four schedules write the same output layout, accumulating in FP32 and adding
the bias in FP32 before the single store:

- ``igemm_ffma``: FP32 FMA on operands converted to FP32. The base schedule,
  correct on every NVIDIA GPU.
- ``igemm_tc``: tensor cores on FP16/BF16 operands, or FP32 operands rounded to
  TF32 when TF32 is allowed (``conv_allows_tf32``). Needs sm_80 on the export
  GPU. For a channel-first output it accumulates the
  transposed tile (output channels by pixels), whose rows are contiguous in
  memory, so its store is coalesced.
- ``igemm_ffma_splitk``: ``igemm_ffma`` with the reduction split over a second
  grid axis into FP32 partial sums, which ``_reduce_split_k`` adds in a fixed
  order before the store; for outputs with too few tiles to fill the GPU.
- ``igemm_tc_packed``: ``igemm_tc`` on copies of the operands with the
  channels innermost: the input channels-last and the weight as
  ``[taps, C_in, C_out]``, so each tap's input and weight tiles are contiguous
  along K and along the output channels. For tap-major K orders, where the
  native weight layout strides both tile dimensions by the tap count.

Strict-FP32 3x3 stride-1 calls also time ``winograd_f4`` (``winograd.py``),
which writes the same output layout from a Winograd F(4x4, 3x3) computation.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl
from executorch.backends.cuda.autotune.launch_params import (
    autotune_launch_param,
    InvalidLaunchParam,
)
from executorch.backends.cuda.triton.kernels.conv.core import winograd
from executorch.backends.cuda.triton.kernels.conv.core.common import (
    conv_allows_tf32,
    ConvCall,
    ConvParams,
    max_offset,
    Schedule,
    SUPPORTED_DTYPES,
    unbenchmarked_layout,
)
from executorch.backends.cuda.triton.routing import device_supports
from torch.library import wrap_triton

SCHEDULES = (
    Schedule("igemm_ffma", 0),
    Schedule("igemm_tc", 80),
    Schedule("igemm_ffma_splitk", 0),
    Schedule("winograd_f4", 0),
    Schedule("igemm_tc_packed", 80),
)
TENSOR_CORE_SCHEDULES = frozenset({"igemm_tc", "igemm_tc_packed"})
TAP_MAJOR_MIN_CHANNELS = 16
# Output pixels per image from which an output with dense spatial dims is
# tiled per image (always for one image): the partial last tile of each image
# wastes at most 1/8 of a 128-row tile.
PIXEL_TILES_MIN_PIXELS = 1024
_INT32_MAX = 2**31 - 1
# Split-K: enough splits to give every SM this many tiles of 64x64 outputs,
# at most MAX_SPLIT_K, each with at least SPLIT_K_MIN_BLOCKS blocks of 32 of K.
SPLIT_K_WAVES = 2
MAX_SPLIT_K = 32
SPLIT_K_MIN_BLOCKS = 4


def _output_size(call: ConvCall) -> list[int]:
    x, w, p = call.input, call.weight, call.params
    if p.transposed:
        return [
            (i - 1) * s - b - a + k + o
            for i, k, s, b, a, o in zip(
                x.shape[2:],
                w.shape[2:],
                p.stride,
                p.padding_before,
                p.padding_after,
                p.output_padding,
            )
        ]
    return [
        (i + b + a - k) // s + 1
        for i, k, s, b, a in zip(
            x.shape[2:], w.shape[2:], p.stride, p.padding_before, p.padding_after
        )
    ]


def _out_channels(call: ConvCall) -> int:
    return call.weight.shape[1 if call.params.transposed else 0]


def _geometry_reason(call: ConvCall) -> Optional[str]:
    x, w, p = call.input, call.weight, call.params
    if any(d != 1 for d in p.dilation):
        return f"implicit GEMM: dilation {list(p.dilation)}"
    in_channels = w.shape[0] if p.transposed else w.shape[1]
    if p.groups != 1 or in_channels != x.shape[1]:
        return (
            f"implicit GEMM: grouped convolution (groups {p.groups}, input "
            f"{list(x.shape)}, weight {list(w.shape)})"
        )
    if min(*x.shape, *w.shape, *_output_size(call)) <= 0:
        return f"implicit GEMM: empty input or output (input {list(x.shape)})"
    return None


def supports(call: ConvCall) -> Optional[str]:
    """Why this template cannot run ``call``, or None. Assumes valid canonical
    arguments (``check_params``)."""
    reason = _geometry_reason(call)
    if reason is not None:
        return reason
    x, w, b = call.input, call.weight, call.bias
    if x.dtype not in SUPPORTED_DTYPES:
        return f"implicit GEMM: dtype {x.dtype}"
    if w.dtype != x.dtype or (b is not None and b.dtype != x.dtype):
        return "implicit GEMM: input, weight and bias dtypes differ"
    if b is not None and (
        tuple(b.shape) != (_out_channels(call),) or not b.is_contiguous()
    ):
        return "implicit GEMM: bias is not a contiguous [C_out] tensor"
    return None


@dataclass(frozen=True)
class RoutedFamily:
    """Geometries benchmarked together: every (kernel, stride, padding) listed,
    with symmetric padding, up to ``max_batch`` images of at most
    ``max_spatial`` pixels per dimension."""

    name: str
    geometries: tuple[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]], ...]
    max_batch: int
    max_spatial: tuple[int, ...]


# The range where benchmarks/conv/benchmark_implicit_gemm.py finds every
# measured call significantly faster than Inductor's Triton conv template:
# 3-channel inputs, which the flat K order walks without the empty
# lanes of the template's channel blocks, in the geometry families below, with
# the (input, weight, output) memory formats in ROUTED_LAYOUTS. Its
# ``boundary`` suite measures ``boundary_calls()``: each geometry at every
# corner of these limits.
ROUTED_CHANNELS = 3
MIN_ROUTED_OUT_CHANNELS = 32
MAX_ROUTED_OUT_CHANNELS = 1280
MIN_ROUTED_OUTPUT_PIXELS = 256
MAX_ROUTED_INPUT_PIXELS = 1024 * 1024
MAX_ROUTED_OUTPUT_ELEMENTS = 1 << 24
ROUTED_LAYOUTS = frozenset(
    {
        ("contiguous", "contiguous", "contiguous"),
        ("channels_last", "contiguous", "channels_last"),
        ("channels_last", "contiguous", "contiguous"),
    }
)
ROUTED_FAMILIES = (
    RoutedFamily(
        "window",
        tuple(
            ((k, k), (s, s), (p, p))
            for k in (3, 5, 7)
            for s in (1, 2, 4)
            if s <= k
            for p in (0, k // 2)
        ),
        256,
        (1024, 1024),
    ),
    RoutedFamily(
        "kernel_stride_2d",
        tuple(((k, k), (k, k), (0, 0)) for k in (2, 4, 8, 14, 16, 32)),
        256,
        (1024, 1024),
    ),
    RoutedFamily(
        "kernel_stride_3d",
        tuple((k, k, (0, 0, 0)) for k in ((1, 4, 4), (2, 14, 14), (2, 16, 16))),
        2048,
        (4, 1024, 1024),
    ),
)


# Dense 3x3 stage convolutions of convolutional backbones in strict FP32:
# (kernel, stride, padding, C_out / C_in, smallest and largest C_in). The
# benchmark's ``stage_boundary`` suite measures ``stage_boundary_calls()``,
# every corner of these limits.
STAGE_GEOMETRIES = (
    ((3, 3), (1, 1), (1, 1), 1, 64, 512),
    ((3, 3), (2, 2), (1, 1), 2, 64, 256),
)
MAX_STAGE_BATCH = 8
MAX_STAGE_EXTENT = 160
MAX_STAGE_INPUT_PIXELS = 120 * 160
# Larger outputs fill the GPU without splitting K, where the FFMA schedules are
# no faster than Inductor's template.
MAX_STAGE_OUTPUT_ELEMENTS = 160 * 1024
STAGE_LAYOUTS = frozenset({("contiguous", "contiguous", "contiguous")})


def _within_stage_limits(x: torch.Tensor) -> bool:
    n, _, *extent = x.shape
    return (
        n <= MAX_STAGE_BATCH
        and max(extent) <= MAX_STAGE_EXTENT
        and n * math.prod(extent) <= MAX_STAGE_INPUT_PIXELS
    )


def _stage_reason(call: ConvCall) -> Optional[str]:
    """Why ``call`` is outside the benchmarked stage range, or None."""
    x, w, p = call.input, call.weight, call.params
    channels = x.shape[1]
    geometry = next(
        (
            g
            for g in STAGE_GEOMETRIES
            if (tuple(w.shape[2:]), p.stride, p.padding_before, p.padding_after)
            == g[:3] + g[2:3]
            and w.shape[0] == g[3] * channels
        ),
        None,
    )
    if geometry is None or not geometry[4] <= channels <= geometry[5]:
        return f"implicit GEMM: {channels} input channels are not benchmarked faster"
    if x.dtype != torch.float32 or conv_allows_tf32():
        return f"implicit GEMM: {x.dtype} stage convolutions are not benchmarked"
    if (
        not _within_stage_limits(x)
        or x.shape[0] * math.prod(_output_size(call)) * w.shape[0]
        > MAX_STAGE_OUTPUT_ELEMENTS
    ):
        return (
            f"implicit GEMM: input {list(x.shape)} is outside the benchmarked "
            "stage range"
        )
    reason = unbenchmarked_layout(call, STAGE_LAYOUTS)
    return None if reason is None else f"implicit GEMM: {reason}"


def _stage_outputs(extent, kernel, stride, padding, out_channels) -> int:
    return _outputs(1, extent, kernel, stride, padding) * out_channels


def stage_boundary_calls() -> (
    list[tuple[tuple[int, ...], int, tuple[int, ...], tuple[int, ...], tuple[int, ...]]]
):
    """(input shape, C_out, kernel, stride, padding) of every stage geometry at
    every corner of its limits: the fewest and most input channels; each
    spatial dimension at its smallest (3) or largest extent, the largest
    reduced until one image fits ``MAX_STAGE_INPUT_PIXELS`` and
    ``MAX_STAGE_OUTPUT_ELEMENTS``; one image and the most images that fit."""
    calls = []
    for kernel, stride, padding, multiplier, *channel_range in STAGE_GEOMETRIES:
        for channels, extent in itertools.product(
            channel_range, itertools.product((3, MAX_STAGE_EXTENT), repeat=2)
        ):
            out_channels = multiplier * channels
            geometry = (kernel, stride, padding, out_channels)
            extent = list(extent)
            while (
                math.prod(extent) > MAX_STAGE_INPUT_PIXELS
                or _stage_outputs(extent, *geometry) > MAX_STAGE_OUTPUT_ELEMENTS
            ):
                i = max(range(2), key=lambda d: extent[d])
                extent[i] -= 1
            most = min(
                MAX_STAGE_BATCH,
                MAX_STAGE_INPUT_PIXELS // math.prod(extent),
                MAX_STAGE_OUTPUT_ELEMENTS // _stage_outputs(extent, *geometry),
            )
            for batch in sorted({1, most}):
                calls.append(
                    ((batch, channels, *extent), out_channels, kernel, stride, padding)
                )
    return calls


# Strict-FP32 3x3 stride-1 convolutions with input and output channels in
# WINOGRAD_CHANNELS, within the stage batch, extent and input-pixel limits, at
# any output size: ``winograd_f4`` does a quarter of the multiplies of a direct
# convolution such as Inductor's template. The benchmark's
# ``winograd_boundary`` suite measures ``winograd_boundary_calls()``, every
# corner of these limits.
WINOGRAD_CHANNELS = (64, 512)


def _winograd_reason(call: ConvCall) -> Optional[str]:
    """Why ``call`` is outside the benchmarked Winograd range, or None."""
    x, w = call.input, call.weight
    reason = winograd.reason(call)
    if reason is not None:
        return f"implicit GEMM: {reason}"
    fewest, most = WINOGRAD_CHANNELS
    if not (fewest <= x.shape[1] <= most and fewest <= w.shape[0] <= most):
        return (
            f"implicit GEMM: {x.shape[1]} to {w.shape[0]} channels are not "
            "benchmarked faster with winograd_f4"
        )
    if conv_allows_tf32():
        return "implicit GEMM: winograd_f4 is not benchmarked against TF32"
    if not _within_stage_limits(x):
        return (
            f"implicit GEMM: input {list(x.shape)} is outside the benchmarked "
            "winograd_f4 range"
        )
    reason = unbenchmarked_layout(call, STAGE_LAYOUTS)
    return None if reason is None else f"implicit GEMM: {reason}"


def winograd_boundary_calls() -> (
    list[tuple[tuple[int, ...], int, tuple[int, ...], tuple[int, ...], tuple[int, ...]]]
):
    """(input shape, C_out, kernel, stride, padding) at every corner of the
    Winograd range: the fewest and most input and output channels; each
    spatial dimension at its smallest (3) or largest extent, the largest
    reduced until one image fits ``MAX_STAGE_INPUT_PIXELS``; one image and the
    most images that fit."""
    calls = []
    for channels, out_channels, extent in itertools.product(
        WINOGRAD_CHANNELS,
        WINOGRAD_CHANNELS,
        itertools.product((3, MAX_STAGE_EXTENT), repeat=2),
    ):
        extent = list(extent)
        while math.prod(extent) > MAX_STAGE_INPUT_PIXELS:
            i = max(range(2), key=lambda d: extent[d])
            extent[i] -= 1
        most = min(MAX_STAGE_BATCH, MAX_STAGE_INPUT_PIXELS // math.prod(extent))
        for batch in sorted({1, most}):
            calls.append(
                ((batch, channels, *extent), out_channels, (3, 3), (1, 1), (1, 1))
            )
    return calls


# Dense 3x3x3 stride-1 convolutions with padding 0 or 1 on tensor cores (FP16
# and BF16), with CONV3D_CHANNELS input and output channels and at least
# CONV3D_MIN_OUTPUT_PIXELS output pixels, at any larger batch and extent, with
# contiguous input, weight and output. ``igemm_tc_packed`` reads both GEMM
# operands along their contiguous dims, which Inductor's template does not for
# a channel-first input, and stays compute-bound as calls grow; fewer output
# pixels leave too few tiles to hide the operand copies. The benchmark's
# ``conv3d_boundary`` suite measures ``conv3d_boundary_calls()``.
CONV3D_GEOMETRIES = (
    ((3, 3, 3), (1, 1, 1), (0, 0, 0)),
    ((3, 3, 3), (1, 1, 1), (1, 1, 1)),
)
CONV3D_CHANNELS = (64, 1024)
CONV3D_MIN_OUTPUT_PIXELS = 64
CONV3D_DTYPES = frozenset({torch.float16, torch.bfloat16})
CONV3D_LAYOUTS = frozenset({("contiguous", "contiguous", "contiguous")})
# The large extent the boundary calls measure; the range has no upper extent.
CONV3D_LARGE_EXTENT = (18, 34, 34)


def _in_conv3d_geometry(call: ConvCall) -> bool:
    p = call.params
    geometry = (tuple(call.weight.shape[2:]), p.stride, p.padding_before)
    return (
        p.rank == 3
        and not p.transposed
        and p.padding_after == p.padding_before
        and geometry in CONV3D_GEOMETRIES
    )


def _conv3d_reason(call: ConvCall) -> Optional[str]:
    """Why ``call`` is outside the benchmarked 3-D range, or None."""
    x, w = call.input, call.weight
    if not _in_conv3d_geometry(call):
        return "implicit GEMM: not a benchmarked 3-D geometry"
    if x.dtype not in CONV3D_DTYPES:
        return f"implicit GEMM: {x.dtype} 3-D convolutions are not benchmarked"
    fewest, most = CONV3D_CHANNELS
    if not (fewest <= x.shape[1] <= most and fewest <= w.shape[0] <= most):
        return (
            f"implicit GEMM: {x.shape[1]} to {w.shape[0]} channels are not "
            "benchmarked faster in 3-D"
        )
    if x.shape[0] * math.prod(_output_size(call)) < CONV3D_MIN_OUTPUT_PIXELS:
        return "implicit GEMM: too few 3-D output pixels to be benchmarked faster"
    reason = unbenchmarked_layout(call, CONV3D_LAYOUTS)
    return None if reason is None else f"implicit GEMM: {reason}"


def conv3d_boundary_calls() -> (
    list[tuple[tuple[int, ...], int, tuple[int, ...], tuple[int, ...], tuple[int, ...]]]
):
    """(input shape, C_out, kernel, stride, padding) of each 3-D geometry with
    the fewest and most input and output channels, an input giving 4x4x4
    (``CONV3D_MIN_OUTPUT_PIXELS``) output pixels and ``CONV3D_LARGE_EXTENT``,
    and one and two images."""
    calls = []
    for (kernel, stride, padding), channels, out_channels in itertools.product(
        CONV3D_GEOMETRIES, CONV3D_CHANNELS, CONV3D_CHANNELS
    ):
        smallest = tuple(k - 2 * p + 3 for k, p in zip(kernel, padding))
        for extent, batch in itertools.product((smallest, CONV3D_LARGE_EXTENT), (1, 2)):
            calls.append(
                ((batch, channels, *extent), out_channels, kernel, stride, padding)
            )
    return calls


# Strict-FP32 1-D convolutions with few output rows (batch times output
# length): kernel 5 or 3 with stride 1 and "same" padding, kernel 3 with
# stride 2 and padding 1, and transposed kernel 4 with stride 2 and padding 1,
# with CONV1D_IN_CHANNELS input and CONV1D_OUT_CHANNELS output channels and
# CONV1D_ROWS output rows, in contiguous layouts. Such a call reads each weight
# for a few outputs, so it is bound by streaming the weights, which the
# split-K schedule spreads over the GPU and Inductor's template, with its few
# output tiles, does not; fewer input channels or rows leave too little weight
# traffic to hide the launch of the reduction. The benchmark's
# ``conv1d_boundary`` suite measures ``conv1d_boundary_calls()``.
CONV1D_GEOMETRIES = (
    # (kernel, stride, padding, transposed)
    (5, 1, 2, False),
    (3, 1, 1, False),
    (3, 2, 1, False),
    (4, 2, 1, True),
)
CONV1D_IN_CHANNELS = (512, 4096)
CONV1D_OUT_CHANNELS = (64, 2048)
CONV1D_ROWS = (4, 64)
CONV1D_LAYOUTS = frozenset({("contiguous", "contiguous", "contiguous")})


def _in_conv1d_geometry(call: ConvCall) -> bool:
    p = call.params
    return (
        p.rank == 1
        and p.padding_after == p.padding_before
        and p.output_padding == (0,)
        and (call.weight.shape[2], *p.stride, *p.padding_before, p.transposed)
        in CONV1D_GEOMETRIES
    )


def _conv1d_reason(call: ConvCall) -> Optional[str]:
    """Why ``call`` is outside the benchmarked 1-D range, or None."""
    x = call.input
    if not _in_conv1d_geometry(call):
        return "implicit GEMM: not a benchmarked 1-D geometry"
    if x.dtype != torch.float32 or conv_allows_tf32():
        return "implicit GEMM: only strict-FP32 1-D convolutions are benchmarked"
    out_channels = _out_channels(call)
    if not (
        CONV1D_IN_CHANNELS[0] <= x.shape[1] <= CONV1D_IN_CHANNELS[1]
        and CONV1D_OUT_CHANNELS[0] <= out_channels <= CONV1D_OUT_CHANNELS[1]
    ):
        return (
            f"implicit GEMM: {x.shape[1]} to {out_channels} channels are not "
            "benchmarked faster in 1-D"
        )
    rows = x.shape[0] * _output_size(call)[0]
    if not CONV1D_ROWS[0] <= rows <= CONV1D_ROWS[1]:
        return f"implicit GEMM: {rows} 1-D output rows are not benchmarked faster"
    reason = unbenchmarked_layout(call, CONV1D_LAYOUTS)
    return None if reason is None else f"implicit GEMM: {reason}"


def _conv1d_outputs(length, kernel, stride, padding, transposed) -> int:
    if transposed:
        return (length - 1) * stride - 2 * padding + kernel
    return (length + 2 * padding - kernel) // stride + 1


def conv1d_boundary_calls() -> list[
    tuple[
        tuple[int, ...],
        int,
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        bool,
    ]
]:
    """(input shape, C_out, kernel, stride, padding, transposed) of each 1-D
    geometry with the fewest and most input and output channels, and the fewest
    and most ``CONV1D_ROWS`` output rows: from one image of the longest input
    with at most that many outputs, and from images of length 1."""
    calls = []
    for geometry, channels, out_channels in itertools.product(
        CONV1D_GEOMETRIES, CONV1D_IN_CHANNELS, CONV1D_OUT_CHANNELS
    ):
        kernel, stride, padding, transposed = geometry
        for rows in CONV1D_ROWS:
            length = 1
            while _conv1d_outputs(length + 1, *geometry) <= rows:
                length += 1
            images = rows // _conv1d_outputs(1, *geometry)
            for shape in ((1, channels, length), (images, channels, 1)):
                calls.append(
                    (shape, out_channels, (kernel,), (stride,), (padding,), transposed)
                )
    return list(dict.fromkeys(calls))


def _family(call: ConvCall) -> Optional[RoutedFamily]:
    p = call.params
    if p.padding_before != p.padding_after:
        return None
    geometry = (tuple(call.weight.shape[2:]), p.stride, p.padding_before)
    return next((f for f in ROUTED_FAMILIES if geometry in f.geometries), None)


def worth_routing(call: ConvCall) -> Optional[str]:
    """Why a call ``supports`` accepts is left to the regular lowering, or None."""
    x, w, p = call.input, call.weight, call.params
    if _in_conv1d_geometry(call):
        return _conv1d_reason(call)
    if p.transposed:
        return "implicit GEMM: a transposed convolution is not benchmarked faster"
    if all(k == 1 for k in w.shape[2:]):
        return "implicit GEMM: a 1x1 convolution keeps the regular lowering"
    tensor_cores = x.dtype != torch.float32 or conv_allows_tf32()
    if tensor_cores and not device_supports(SCHEDULES[1].min_capability, x.device):
        return (
            f"implicit GEMM: {x.dtype} without tensor cores on the export GPU "
            "is not benchmarked"
        )
    if x.shape[1] != ROUTED_CHANNELS:
        if _in_conv3d_geometry(call):
            return _conv3d_reason(call)
        if _winograd_reason(call) is None:
            return None
        return _stage_reason(call)
    reason = unbenchmarked_layout(call, ROUTED_LAYOUTS)
    if reason is not None:
        return f"implicit GEMM: {reason}"
    family = _family(call)
    if family is None:
        return (
            f"implicit GEMM: kernel {list(w.shape[2:])}, stride {list(p.stride)}, "
            f"padding {list(p.padding_before)}/{list(p.padding_after)} is not a "
            "benchmarked geometry"
        )
    outputs = x.shape[0] * math.prod(_output_size(call))
    if (
        x.shape[0] > family.max_batch
        or any(e > m for e, m in zip(x.shape[2:], family.max_spatial))
        or x.shape[0] * math.prod(x.shape[2:]) > MAX_ROUTED_INPUT_PIXELS
        or not MIN_ROUTED_OUT_CHANNELS <= w.shape[0] <= MAX_ROUTED_OUT_CHANNELS
        or outputs < MIN_ROUTED_OUTPUT_PIXELS
        or outputs * w.shape[0] > MAX_ROUTED_OUTPUT_ELEMENTS
    ):
        return (
            f"implicit GEMM: input {list(x.shape)} with {w.shape[0]} output "
            f"channels is outside the benchmarked {family.name} range"
        )
    return None


def _outputs(batch, extent, kernel, stride, padding) -> int:
    return batch * math.prod(
        (e + 2 * p - k) // s + 1 for e, k, s, p in zip(extent, kernel, stride, padding)
    )


def boundary_calls() -> (
    list[tuple[tuple[int, ...], int, tuple[int, ...], tuple[int, ...], tuple[int, ...]]]
):
    """(input shape, C_out, kernel, stride, padding) of every routed geometry at
    every corner of its limits: each spatial dimension at its smallest (one
    output) or largest extent with the fewest images that give
    ``MIN_ROUTED_OUTPUT_PIXELS`` outputs, and the most images of the smallest
    extent; extents halved until the input fits ``MAX_ROUTED_INPUT_PIXELS`` and
    the output fits ``MIN_ROUTED_OUT_CHANNELS``; with the fewest and the most
    output channels the limits allow."""
    calls = []
    for family in ROUTED_FAMILIES:
        for kernel, stride, padding in family.geometries:
            smallest = tuple(k - 2 * p for k, p in zip(kernel, padding))
            corners = [
                (None, e) for e in itertools.product(*zip(smallest, family.max_spatial))
            ]
            corners.append((family.max_batch, smallest))
            for batch, extent in corners:
                extent = list(extent)
                while True:
                    images = batch
                    per_image = _outputs(1, extent, kernel, stride, padding)
                    if images is None:
                        images = -(-MIN_ROUTED_OUTPUT_PIXELS // per_image)
                    images = min(
                        images,
                        family.max_batch,
                        MAX_ROUTED_INPUT_PIXELS // math.prod(extent),
                    )
                    outputs = images * per_image
                    fits = (
                        math.prod(extent) <= MAX_ROUTED_INPUT_PIXELS
                        and MAX_ROUTED_OUTPUT_ELEMENTS // max(outputs, 1)
                        >= MIN_ROUTED_OUT_CHANNELS
                    )
                    if fits or list(extent) == list(smallest):
                        break
                    i = max(range(len(extent)), key=lambda d: extent[d] - smallest[d])
                    extent[i] = max(smallest[i], extent[i] // 2)
                batch = images
                if outputs < MIN_ROUTED_OUTPUT_PIXELS:
                    continue
                most = min(
                    MAX_ROUTED_OUT_CHANNELS, MAX_ROUTED_OUTPUT_ELEMENTS // outputs
                )
                for out_channels in (MIN_ROUTED_OUT_CHANNELS, most):
                    calls.append(
                        (
                            (batch, ROUTED_CHANNELS, *extent),
                            out_channels,
                            kernel,
                            stride,
                            padding,
                        )
                    )
    return list(dict.fromkeys(calls))


def schedule_supports(name: str, call: ConvCall) -> Optional[str]:
    """Why schedule ``name`` cannot run ``call`` (which ``supports``), or None."""
    schedule = next((s for s in SCHEDULES if s.name == name), None)
    if schedule is None:
        return f"unknown schedule {name}"
    if not device_supports(schedule.min_capability, call.input.device):
        return f"{name} needs sm_{schedule.min_capability}"
    if (
        name in TENSOR_CORE_SCHEDULES
        and call.input.dtype == torch.float32
        and not conv_allows_tf32()
    ):
        return f"{name} needs TF32 for FP32 operands, which is not allowed"
    if name == "igemm_tc_packed" and call.input.shape[1] < TAP_MAJOR_MIN_CHANNELS:
        return (
            "igemm_tc_packed: the flat K order of few input channels does not "
            "read the packed layout contiguously"
        )
    if name == "winograd_f4":
        return winograd.reason(call)
    if name == "igemm_ffma_splitk" and _split_k(call) == 1:
        return "igemm_ffma_splitk: the output tiles fill the GPU without splitting K"
    return None


def _split_k(call: ConvCall) -> int:
    """How many ways ``igemm_ffma_splitk`` splits the reduction of ``call``."""
    x, w = call.input, call.weight
    rows = x.shape[0] * math.prod(_output_size(call))
    tiles = triton.cdiv(rows, 64) * triton.cdiv(_out_channels(call), 64)
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    k_blocks = triton.cdiv(x.shape[1] * math.prod(w.shape[2:]), 32)
    split = min(
        SPLIT_K_WAVES * sms // tiles, MAX_SPLIT_K, k_blocks // SPLIT_K_MIN_BLOCKS
    )
    return max(split, 1)


@triton.jit
def _bound(ok, i, SIZE: tl.constexpr, LOW: tl.constexpr, HIGH: tl.constexpr):
    if LOW:
        ok = ok & (i >= 0)
    if HIGH:
        ok = ok & (i < SIZE)
    return ok


@triton.jit
def _source(o, k, PAD: tl.constexpr, STRIDE: tl.constexpr, SIZE: tl.constexpr):
    """The input index a transposed convolution reads for output index ``o`` and
    tap ``k``, and whether it exists: ``o + PAD - k`` must be a multiple of the
    stride inside the input."""
    q = o + PAD - k
    i = q // STRIDE
    return i, (q >= 0) & (i * STRIDE == q) & (i < SIZE)


@triton.jit
def _dot(a, b, acc, FFMA: tl.constexpr, CHANNEL_MAJOR: tl.constexpr):
    if CHANNEL_MAJOR:
        a, b = tl.trans(b), tl.trans(a)
    if FFMA:
        return tl.dot(a.to(tl.float32), b.to(tl.float32), acc, input_precision="ieee")
    if a.dtype == tl.float32:
        return tl.dot(a, b, acc, input_precision="tf32")
    return tl.dot(a, b, acc)


@triton.jit
def _store(
    out_ptr,
    acc,
    b_ptr,
    out_rows,
    m_mask,
    offs_n,
    n_mask,
    SO_C: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    CHANNEL_MAJOR: tl.constexpr,
):
    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    if CHANNEL_MAJOR:
        if HAS_BIAS:
            acc += bias[:, None]
        out_ptrs = out_ptr + offs_n[:, None] * SO_C + out_rows[None, :]
        mask = n_mask[:, None] & m_mask[None, :]
    else:
        if HAS_BIAS:
            acc += bias[None, :]
        out_ptrs = out_ptr + out_rows[:, None] + offs_n[None, :] * SO_C
        mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _k_range(BLOCKS: tl.constexpr, SPLIT_K: tl.constexpr):
    """The blocks of K this program reduces: all, or its share of a split."""
    if SPLIT_K == 1:
        return 0, BLOCKS
    per_split: tl.constexpr = (BLOCKS + SPLIT_K - 1) // SPLIT_K
    first = tl.program_id(1) * per_split
    return first, tl.minimum(first + per_split, BLOCKS)


@triton.jit
def _reduce_split_k(
    partial_ptr,
    b_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    CO: tl.constexpr,
    OD: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    SO_N: tl.constexpr,
    SO_C: tl.constexpr,
    SO_D: tl.constexpr,
    SO_H: tl.constexpr,
    SO_W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Sums the split-K partials in a fixed order, then stores like the kernel."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = rows < ROWS
    n_mask = offs_n < CO
    mask = m_mask[:, None] & n_mask[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for split in range(SPLIT_K):
        acc += tl.load(
            partial_ptr + (split * ROWS + rows)[:, None] * CO + offs_n[None, :],
            mask=mask,
            other=0.0,
        )
    PIXELS: tl.constexpr = OD * OH * OW
    n = rows // PIXELS
    pixel = rows % PIXELS
    ow = pixel % OW
    oh = pixel // OW % OH
    od = pixel // (OW * OH)
    out_rows = n * SO_N + od * SO_D + oh * SO_H + ow * SO_W
    _store(out_ptr, acc, b_ptr, out_rows, m_mask, offs_n, n_mask, SO_C, HAS_BIAS, False)


@triton.jit
def _kernel(  # noqa: C901
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    partial_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    CO: tl.constexpr,
    OD: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    SX_N: tl.constexpr,
    SX_C: tl.constexpr,
    SX_D: tl.constexpr,
    SX_H: tl.constexpr,
    SX_W: tl.constexpr,
    SW_O: tl.constexpr,
    SW_C: tl.constexpr,
    SW_D: tl.constexpr,
    SW_H: tl.constexpr,
    SW_W: tl.constexpr,
    SO_N: tl.constexpr,
    SO_C: tl.constexpr,
    SO_D: tl.constexpr,
    SO_H: tl.constexpr,
    SO_W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    FFMA: tl.constexpr,
    INDEX_64: tl.constexpr,
    PIXEL_TILES: tl.constexpr,
    CHANNEL_MAJOR: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    SPLIT_K: tl.constexpr,
    TAP_MAJOR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    PIXELS: tl.constexpr = OD * OH * OW
    TAPS: tl.constexpr = KD * KH * KW
    # Which bounds checks the geometry can fail.
    LOW_D: tl.constexpr = PAD_D > 0
    LOW_H: tl.constexpr = PAD_H > 0
    LOW_W: tl.constexpr = PAD_W > 0
    HIGH_D: tl.constexpr = (OD - 1) * STRIDE_D - PAD_D + KD > D
    HIGH_H: tl.constexpr = (OH - 1) * STRIDE_H - PAD_H + KH > H
    HIGH_W: tl.constexpr = (OW - 1) * STRIDE_W - PAD_W + KW > W
    # With pixel tiles, an M tile holds consecutive output pixels of one image,
    # so the store into a channel-first output is provably contiguous.
    PIXEL_BLOCKS: tl.constexpr = (PIXELS + BLOCK_M - 1) // BLOCK_M
    GRID_M: tl.constexpr = (
        N * PIXEL_BLOCKS if PIXEL_TILES else (N * PIXELS + BLOCK_M - 1) // BLOCK_M
    )
    GRID_N: tl.constexpr = (CO + BLOCK_N - 1) // BLOCK_N

    pid = tl.program_id(0)
    if INDEX_64:
        pid = pid.to(tl.int64)
    first_m = pid // (GROUP_M * GRID_N) * GROUP_M
    rows = tl.minimum(GRID_M - first_m, GROUP_M)
    pid_m = first_m + pid % (GROUP_M * GRID_N) % rows
    pid_n = pid % (GROUP_M * GRID_N) // rows
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < CO
    if PIXEL_TILES:
        n = pid_m // PIXEL_BLOCKS
        pixel = pid_m % PIXEL_BLOCKS * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = pixel < PIXELS
    else:
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = offs_m < N * PIXELS
        n = offs_m // PIXELS
        pixel = offs_m % PIXELS
    ow = pixel % OW
    oh = pixel // OW % OH
    od = pixel // (OW * OH)
    d0 = od * STRIDE_D - PAD_D
    h0 = oh * STRIDE_H - PAD_H
    w0 = ow * STRIDE_W - PAD_W
    x_images = x_ptr + n * SX_N
    x_rows = x_images + d0 * SX_D + h0 * SX_H + w0 * SX_W
    w_cols = w_ptr + offs_n * SW_O
    if CHANNEL_MAJOR:
        acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if TAP_MAJOR:
        C_BLOCKS: tl.constexpr = (C + BLOCK_K - 1) // BLOCK_K
        first, last = _k_range(TAPS * C_BLOCKS, SPLIT_K)
        for step in range(first, last):
            tap = step // C_BLOCKS
            c = (step % C_BLOCKS) * BLOCK_K + tl.arange(0, BLOCK_K)
            if INDEX_64:
                c = c.to(tl.int64)
            kd = tap // (KH * KW)
            kh = tap // KW % KH
            kw = tap % KW
            if TRANSPOSED:
                i_d, ok_d = _source(od, kd, PAD_D, STRIDE_D, D)
                i_h, ok_h = _source(oh, kh, PAD_H, STRIDE_H, H)
                i_w, ok_w = _source(ow, kw, PAD_W, STRIDE_W, W)
                ok = m_mask & ok_d & ok_h & ok_w
                taps = x_images + i_d * SX_D + i_h * SX_H + i_w * SX_W
            else:
                ok = _bound(m_mask, d0 + kd, D, LOW_D, HIGH_D)
                ok = _bound(ok, h0 + kh, H, LOW_H, HIGH_H)
                ok = _bound(ok, w0 + kw, W, LOW_W, HIGH_W)
                taps = x_rows + (kd * SX_D + kh * SX_H + kw * SX_W)
            c_ok = c < C
            a = tl.load(
                taps[:, None] + c[None, :] * SX_C,
                mask=ok[:, None] & c_ok[None, :],
                other=0.0,
            )
            b = tl.load(
                w_cols[None, :]
                + (kd * SW_D + kh * SW_H + kw * SW_W)
                + c[:, None] * SW_C,
                mask=c_ok[:, None] & n_mask[None, :],
                other=0.0,
            )
            acc = _dot(a, b, acc, FFMA, CHANNEL_MAJOR)
    else:
        K: tl.constexpr = C * TAPS
        first, last = _k_range((K + BLOCK_K - 1) // BLOCK_K, SPLIT_K)
        for k_block in range(first, last):
            k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
            if INDEX_64:
                k = k.to(tl.int64)
            c = k // TAPS
            tap = k % TAPS
            kd = tap // (KH * KW)
            kh = tap // KW % KH
            kw = tap % KW
            k_ok = k < K
            if TRANSPOSED:
                i_d, ok_d = _source(od[:, None], kd[None, :], PAD_D, STRIDE_D, D)
                i_h, ok_h = _source(oh[:, None], kh[None, :], PAD_H, STRIDE_H, H)
                i_w, ok_w = _source(ow[:, None], kw[None, :], PAD_W, STRIDE_W, W)
                ok = m_mask[:, None] & k_ok[None, :] & ok_d & ok_h & ok_w
                # With pixel tiles the image (and its base pointer) is a scalar.
                images = x_images if PIXEL_TILES else x_images[:, None]
                taps = images + i_d * SX_D + i_h * SX_H + i_w * SX_W
            else:
                ok = _bound(
                    m_mask[:, None] & k_ok[None, :],
                    d0[:, None] + kd[None, :],
                    D,
                    LOW_D,
                    HIGH_D,
                )
                ok = _bound(ok, h0[:, None] + kh[None, :], H, LOW_H, HIGH_H)
                ok = _bound(ok, w0[:, None] + kw[None, :], W, LOW_W, HIGH_W)
                taps = x_rows[:, None] + (kd * SX_D + kh * SX_H + kw * SX_W)[None, :]
            a = tl.load(taps + (c * SX_C)[None, :], mask=ok, other=0.0)
            b = tl.load(
                w_cols[None, :]
                + (c * SW_C + kd * SW_D + kh * SW_H + kw * SW_W)[:, None],
                mask=k_ok[:, None] & n_mask[None, :],
                other=0.0,
            )
            acc = _dot(a, b, acc, FFMA, CHANNEL_MAJOR)
    if SPLIT_K > 1:
        # FP32 partial sums [SPLIT_K, N * PIXELS, CO]; _reduce_split_k adds them.
        rows = (tl.program_id(1) * N + n) * PIXELS + pixel
        tl.store(
            partial_ptr + rows[:, None] * CO + offs_n[None, :],
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )
        return
    if PIXEL_TILES:
        out_rows = n * SO_N + pixel
    else:
        out_rows = n * SO_N + od * SO_D + oh * SO_H + ow * SO_W
    _store(
        out_ptr,
        acc,
        b_ptr,
        out_rows,
        m_mask,
        offs_n,
        n_mask,
        SO_C,
        HAS_BIAS,
        CHANNEL_MAJOR,
    )


def _configs(tiles, tap_major: bool) -> list[triton.Config]:
    return [
        triton.Config(
            {
                "TAP_MAJOR": tap_major,
                "BLOCK_M": m,
                "BLOCK_N": n,
                "BLOCK_K": k,
                "GROUP_M": 8,
            },
            num_warps=warps,
            num_stages=stages,
        )
        for m, n, k, warps, stages in tiles
    ]


# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
_FFMA_TILES = (
    (256, 32, 16, 4, 3),
    (128, 32, 16, 4, 2),
    (64, 32, 16, 4, 3),
    (64, 64, 16, 4, 3),
    (128, 64, 16, 8, 3),
    (64, 128, 16, 8, 3),
    (128, 128, 16, 8, 2),
    (64, 64, 32, 8, 3),
    (64, 32, 32, 4, 3),
    (128, 32, 16, 4, 3),
)
_TC_PACKED_TILES = (
    (128, 64, 32, 4, 4),
    (64, 128, 32, 4, 4),
    (128, 128, 32, 4, 3),
    (256, 64, 32, 8, 3),
    (64, 64, 64, 4, 3),
    (128, 64, 64, 4, 3),
    (128, 128, 64, 8, 3),
    (64, 32, 64, 4, 3),
)
_TC_TILES = (
    (64, 64, 32, 4, 4),
    (128, 64, 32, 4, 4),
    (64, 128, 32, 4, 4),
    (128, 128, 32, 8, 3),
    (128, 32, 32, 4, 4),
    (64, 32, 32, 4, 3),
    (64, 32, 64, 4, 3),
    (32, 64, 64, 4, 3),
)
CONFIGS = {
    (schedule, tap_major): _configs(tiles, tap_major)
    for schedule, tiles, k_orders in (
        ("igemm_ffma", _FFMA_TILES, (True, False)),
        ("igemm_tc", _TC_TILES, (True, False)),
        ("igemm_ffma_splitk", _FFMA_TILES, (True, False)),
        ("igemm_tc_packed", _TC_PACKED_TILES, (True,)),
    )
    for tap_major in k_orders
}
_TUNING_KEY = [
    "N",
    "C",
    "D",
    "H",
    "W",
    "CO",
    "KD",
    "KH",
    "KW",
    "STRIDE_D",
    "STRIDE_H",
    "STRIDE_W",
    "SX_C",
    "SO_C",
]
_KERNELS = {
    key: triton.autotune(configs=configs, key=_TUNING_KEY)(_kernel)
    for key, configs in CONFIGS.items()
}


def _as_3d(values, fill: int) -> tuple[int, ...]:
    return (fill,) * (3 - len(values)) + tuple(values)


def _in_memory_order(t: torch.Tensor, order: tuple[int, ...]) -> torch.Tensor:
    """``t`` stored with its dims ``order`` outermost first (copied unless it
    already is), with its logical dim order unchanged."""
    inverse = tuple(sorted(range(t.dim()), key=order.__getitem__))
    return t.permute(order).contiguous().permute(inverse)


def _packed(x: torch.Tensor, weight: torch.Tensor, transposed: bool):
    """The ``igemm_tc_packed`` operands: the input channels-last, and the weight
    as [taps, C_in, C_out] in memory."""
    taps = tuple(range(2, x.dim()))
    x = _in_memory_order(x, (0, *taps, 1))
    weight = _in_memory_order(weight, (*taps, 0, 1) if transposed else (*taps, 1, 0))
    return x, weight


def run_schedule(
    name: str,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
    params: ConvParams,
    config: Optional[triton.Config] = None,
) -> None:
    """Launches schedule ``name``; ``config`` replaces its autotuned config
    (and the K order, if it sets ``TAP_MAJOR``)."""
    if name == "winograd_f4":
        winograd.run(x, weight, bias, out, config)
        return
    if name == "igemm_tc_packed":
        x, weight = _packed(x, weight, params.transposed)
    n, c = x.shape[:2]
    co = out.shape[1]
    # A transposed weight is [C_in, C_out, ...]: the kernel indexes it by
    # (output channel, input channel) strides.
    w_strides = weight.stride()[:2]
    if params.transposed:
        w_strides = w_strides[::-1]
    out_size = _as_3d(out.shape[2:], 1)
    tap_major = c >= TAP_MAJOR_MIN_CHANNELS
    options = {}
    if config is not None:
        tap_major = config.kwargs.get("TAP_MAJOR", tap_major)
        options = {
            **config.kwargs,
            "TAP_MAJOR": tap_major,
            "num_warps": config.num_warps,
            "num_stages": config.num_stages,
        }
    kernel = _KERNELS[(name, tap_major)]
    if config is not None:
        kernel = kernel.fn
    pixels = out_size[0] * out_size[1] * out_size[2]
    out_strides = _as_3d(out.stride()[2:], 0)
    dense = (out_size[1] * out_size[2], out_size[2], 1)
    pixel_tiles = (n == 1 or pixels >= PIXEL_TILES_MIN_PIXELS) and all(
        size == 1 or s == d for size, s, d in zip(out_size, out_strides, dense)
    )
    split_k = 1
    partial = out
    if name == "igemm_ffma_splitk":
        split_k = _split_k(ConvCall(x, weight, bias, params))
        partial = out.new_empty((split_k, n * pixels, co), dtype=torch.float32)

    def grid(meta):
        if pixel_tiles:
            rows = n * triton.cdiv(pixels, meta["BLOCK_M"])
        else:
            rows = triton.cdiv(n * pixels, meta["BLOCK_M"])
        return (rows * triton.cdiv(co, meta["BLOCK_N"]), split_k)

    wrap_triton(kernel)[grid](
        x,
        weight,
        x if bias is None else bias,
        out,
        partial,
        n,
        c,
        *_as_3d(x.shape[2:], 1),
        co,
        *out_size,
        *_as_3d(weight.shape[2:], 1),
        *_as_3d(params.stride, 1),
        *_as_3d(params.padding_before, 0),
        *x.stride()[:2],
        *_as_3d(x.stride()[2:], 0),
        *w_strides,
        *_as_3d(weight.stride()[2:], 0),
        *out.stride()[:2],
        *out_strides,
        HAS_BIAS=bias is not None,
        FFMA=name not in TENSOR_CORE_SCHEDULES,
        INDEX_64=max(max_offset(x), max_offset(out), max_offset(weight)) > _INT32_MAX
        or split_k * n * pixels * co > _INT32_MAX,
        PIXEL_TILES=pixel_tiles,
        CHANNEL_MAJOR=name in TENSOR_CORE_SCHEDULES and out.stride(1) != 1,
        TRANSPOSED=params.transposed,
        SPLIT_K=split_k,
        **options,
    )
    if split_k > 1:
        wrap_triton(_reduce_split_k)[
            (triton.cdiv(n * pixels, 64), triton.cdiv(co, 64))
        ](
            partial,
            x if bias is None else bias,
            out,
            n * pixels,
            co,
            *out_size,
            out.stride(0),
            out.stride(1),
            *out_strides,
            HAS_BIAS=bias is not None,
            SPLIT_K=split_k,
            BLOCK_M=64,
            BLOCK_N=64,
        )


@autotune_launch_param("schedule", tuple(s.name for s in SCHEDULES))
def launch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
    params: ConvParams,
    *,
    schedule: str,
) -> None:
    """Runs the convolution into ``out``; during AOTI compile the schedule is timed."""
    reason = schedule_supports(schedule, ConvCall(x, weight, bias, params))
    if reason is not None:
        raise InvalidLaunchParam(reason)
    run_schedule(schedule, x, weight, bias, out, params)

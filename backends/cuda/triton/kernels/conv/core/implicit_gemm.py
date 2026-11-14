# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Implicit-GEMM template: dense (groups 1) forward convolution.

The convolution is a GEMM of the output pixels (``M = N * OD * OH * OW``) by the
output channels, reducing over ``K = C_in * KD * KH * KW``; input tiles are
gathered from the input as the GEMM walks K. The kernel is written once for
(D, H, W): a 2-D convolution has a unit depth, which, like every shape, stride
and geometry argument, is constexpr and compiles away; so does every bounds
check the geometry cannot fail (all of them when kernel == stride and
there is no padding). Tiles are autotuned. K is walked in
one of two orders:

- tap-major: for each kernel tap, blocks of input channels, sharing the bounds
  check of the tap. Used when the input has at least ``TAP_MAJOR_MIN_CHANNELS``
  channels.
- flat: one index over (channel, tap), decoded per element, which wastes no
  lanes on inputs with few channels, such as 3.

Two schedules write the same output layout, accumulating in FP32 and adding
the bias in FP32 before the single store:

- ``igemm_ffma``: FP32 FMA on operands converted to FP32. The base schedule,
  correct on every NVIDIA GPU.
- ``igemm_tc``: tensor cores on FP16/BF16 operands, or FP32 operands rounded to
  TF32 when TF32 is allowed (``conv_allows_tf32``). Needs sm_80 on the export
  GPU. For a channel-first output it accumulates the
  transposed tile (output channels by pixels), whose rows are contiguous in
  memory, so its store is coalesced.
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
from executorch.backends.cuda.triton.kernels.conv.core.common import (
    conv_allows_tf32,
    ConvCall,
    ConvParams,
    max_offset,
    Schedule,
    SUPPORTED_DTYPES,
)
from executorch.backends.cuda.triton.routing import device_supports
from torch.library import wrap_triton

SCHEDULES = (Schedule("igemm_ffma", 0), Schedule("igemm_tc", 80))
TAP_MAJOR_MIN_CHANNELS = 16
# Output pixels per image from which an output with dense spatial dims is
# tiled per image (always for one image): the partial last tile of each image
# wastes at most 1/8 of a 128-row tile.
PIXEL_TILES_MIN_PIXELS = 1024
_INT32_MAX = 2**31 - 1


def _output_size(call: ConvCall) -> list[int]:
    x, w, p = call.input, call.weight, call.params
    return [
        (i + b + a - k) // s + 1
        for i, k, s, b, a in zip(
            x.shape[2:], w.shape[2:], p.stride, p.padding_before, p.padding_after
        )
    ]


def _geometry_reason(call: ConvCall) -> Optional[str]:
    x, w, p = call.input, call.weight, call.params
    if p.rank not in (2, 3):
        return f"implicit GEMM: {p.rank}-D convolution, needs 2-D or 3-D"
    if p.transposed:
        return "implicit GEMM: transposed convolution"
    if any(d != 1 for d in p.dilation):
        return f"implicit GEMM: dilation {list(p.dilation)}"
    if p.groups != 1 or w.shape[1] != x.shape[1]:
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
    if b is not None and (tuple(b.shape) != (w.shape[0],) or not b.is_contiguous()):
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
# a dense NCHW or channels-last input and a contiguous weight (the layouts the
# benchmark measures). Its ``boundary`` suite measures ``boundary_calls()``:
# each geometry at every corner of these limits.
ROUTED_CHANNELS = 3
MIN_ROUTED_OUT_CHANNELS = 32
MAX_ROUTED_OUT_CHANNELS = 1280
MIN_ROUTED_OUTPUT_PIXELS = 256
MAX_ROUTED_INPUT_PIXELS = 1024 * 1024
MAX_ROUTED_OUTPUT_ELEMENTS = 1 << 24
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


def _benchmarked_layout(x: torch.Tensor, w: torch.Tensor) -> bool:
    """A dense, unsliced NCHW or channels-last input and a contiguous weight."""
    channels_last = torch.channels_last if x.dim() == 4 else torch.channels_last_3d
    dense_input = x.is_contiguous() or x.is_contiguous(memory_format=channels_last)
    return (
        dense_input
        and x.storage_offset() == 0
        and w.is_contiguous()
        and w.storage_offset() == 0
    )


def _family(call: ConvCall) -> Optional[RoutedFamily]:
    p = call.params
    if p.padding_before != p.padding_after:
        return None
    geometry = (tuple(call.weight.shape[2:]), p.stride, p.padding_before)
    return next((f for f in ROUTED_FAMILIES if geometry in f.geometries), None)


def worth_routing(call: ConvCall) -> Optional[str]:
    """Why a call ``supports`` accepts is left to the regular lowering, or None."""
    x, w, p = call.input, call.weight, call.params
    if all(k == 1 for k in w.shape[2:]):
        return "implicit GEMM: a 1x1 convolution keeps the regular lowering"
    tensor_cores = x.dtype != torch.float32 or conv_allows_tf32()
    if tensor_cores and not device_supports(SCHEDULES[1].min_capability, x.device):
        return (
            f"implicit GEMM: {x.dtype} without tensor cores on the export GPU "
            "is not benchmarked"
        )
    if x.shape[1] != ROUTED_CHANNELS:
        return f"implicit GEMM: {x.shape[1]} input channels are not benchmarked faster"
    if not _benchmarked_layout(x, w):
        return (
            f"implicit GEMM: input strides {list(x.stride())} (offset "
            f"{x.storage_offset()}) or weight strides {list(w.stride())} are not "
            "a benchmarked layout"
        )
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
        name == "igemm_tc"
        and call.input.dtype == torch.float32
        and not conv_allows_tf32()
    ):
        return "igemm_tc needs TF32 for FP32 operands, which is not allowed"
    return None


@triton.jit
def _bound(ok, i, SIZE: tl.constexpr, LOW: tl.constexpr, HIGH: tl.constexpr):
    if LOW:
        ok = ok & (i >= 0)
    if HIGH:
        ok = ok & (i < SIZE)
    return ok


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
def _kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
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
    x_rows = x_ptr + n * SX_N + d0 * SX_D + h0 * SX_H + w0 * SX_W
    w_cols = w_ptr + offs_n * SW_O
    if CHANNEL_MAJOR:
        acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if TAP_MAJOR:
        C_BLOCKS: tl.constexpr = (C + BLOCK_K - 1) // BLOCK_K
        for step in range(TAPS * C_BLOCKS):
            tap = step // C_BLOCKS
            c = (step % C_BLOCKS) * BLOCK_K + tl.arange(0, BLOCK_K)
            if INDEX_64:
                c = c.to(tl.int64)
            kd = tap // (KH * KW)
            kh = tap // KW % KH
            kw = tap % KW
            ok = _bound(m_mask, d0 + kd, D, LOW_D, HIGH_D)
            ok = _bound(ok, h0 + kh, H, LOW_H, HIGH_H)
            ok = _bound(ok, w0 + kw, W, LOW_W, HIGH_W)
            c_ok = c < C
            a = tl.load(
                x_rows[:, None]
                + (kd * SX_D + kh * SX_H + kw * SX_W)
                + c[None, :] * SX_C,
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
        for k0 in range(0, K, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            if INDEX_64:
                k = k.to(tl.int64)
            c = k // TAPS
            tap = k % TAPS
            kd = tap // (KH * KW)
            kh = tap // KW % KH
            kw = tap % KW
            k_ok = k < K
            ok = _bound(
                m_mask[:, None] & k_ok[None, :],
                d0[:, None] + kd[None, :],
                D,
                LOW_D,
                HIGH_D,
            )
            ok = _bound(ok, h0[:, None] + kh[None, :], H, LOW_H, HIGH_H)
            ok = _bound(ok, w0[:, None] + kw[None, :], W, LOW_W, HIGH_W)
            a = tl.load(
                x_rows[:, None]
                + (c * SX_C + kd * SX_D + kh * SX_H + kw * SX_W)[None, :],
                mask=ok,
                other=0.0,
            )
            b = tl.load(
                w_cols[None, :]
                + (c * SW_C + kd * SW_D + kh * SW_H + kw * SW_W)[:, None],
                mask=k_ok[:, None] & n_mask[None, :],
                other=0.0,
            )
            acc = _dot(a, b, acc, FFMA, CHANNEL_MAJOR)
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
    (64, 64, 16, 4, 3),
    (128, 64, 16, 8, 3),
    (64, 128, 16, 8, 3),
    (128, 128, 16, 8, 2),
    (64, 64, 32, 8, 3),
    (64, 32, 32, 4, 3),
    (128, 32, 16, 4, 3),
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
    for schedule, tiles in (("igemm_ffma", _FFMA_TILES), ("igemm_tc", _TC_TILES))
    for tap_major in (True, False)
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
    n, c = x.shape[:2]
    co = weight.shape[0]
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

    def grid(meta):
        if pixel_tiles:
            rows = n * triton.cdiv(pixels, meta["BLOCK_M"])
        else:
            rows = triton.cdiv(n * pixels, meta["BLOCK_M"])
        return (rows * triton.cdiv(co, meta["BLOCK_N"]),)

    wrap_triton(kernel)[grid](
        x,
        weight,
        x if bias is None else bias,
        out,
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
        *weight.stride()[:2],
        *_as_3d(weight.stride()[2:], 0),
        *out.stride()[:2],
        *out_strides,
        HAS_BIAS=bias is not None,
        FFMA=name == "igemm_ffma",
        INDEX_64=max(max_offset(x), max_offset(out), max_offset(weight)) > _INT32_MAX
        or n * pixels * co > _INT32_MAX,
        PIXEL_TILES=pixel_tiles,
        CHANNEL_MAJOR=name == "igemm_tc" and out.stride(1) != 1,
        **options,
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

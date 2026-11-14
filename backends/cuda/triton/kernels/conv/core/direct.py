# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Direct-stencil template: 2-D depthwise convolutions and grouped convolutions
with at most ``MAX_GROUP_CHANNELS`` input channels per group (any channel
multiplier).

Each output reads its ``KH x KW`` window of its group's input channels straight
from the input; the stencil is unrolled over constexpr taps (one kernel row at a
time for more than one channel per group, which keeps compile time flat in the
kernel height) and accumulated in FP32, and the bias is added in FP32 before the
single store. Shapes, strides and geometry are constexpr (static shapes), tiles
are autotuned. Five schedules write the same output layout:

- ``dw_flat``: one flat tile of outputs in the output's memory order. Legal for
  every supported call and correct on every NVIDIA GPU; it is the base schedule.
- ``dw_row8``: eight adjacent outputs of one row times a block of channels,
  sliding the input window through registers so each input vector is loaded
  once per kernel row (with unit W stride), and each weight vector once for
  the eight. Needs channel-contiguous input.
- ``dw_mult``: depthwise convolutions with a channel multiplier of at most
  ``MAX_MULT_MULTIPLIER``; a block of positions of one input channel computes
  all of its output channels, so each input value is loaded once for them.
- ``grp_plane``: grouped convolutions and channel multipliers; a block of
  positions of one output channel, so the weights are scalars shared by the
  block, loaded once per warp.
- ``grp_reduce``: grouped convolutions and channel multipliers with few
  outputs; a tile of outputs times a block of their taps summed across the
  tile, so the reduction runs in parallel.

The router sends a call here when ``supports()`` accepts it and it lies in the
range ``benchmarks/conv/benchmark_depthwise.py`` measures (``worth_routing``),
where the measured calls are faster than Inductor's Triton conv template; other
supported calls keep the regular lowering.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import triton
import triton.language as tl
from executorch.backends.cuda.autotune.launch_params import (
    autotune_launch_param,
    InvalidLaunchParam,
)
from executorch.backends.cuda.triton.kernels.conv.core.common import (
    ConvCall,
    ConvParams,
    max_offset,
    SUPPORTED_DTYPES,
)
from executorch.backends.cuda.triton.routing import device_supports
from torch.library import wrap_triton

MAX_KERNEL_SIZE = 9
MAX_GROUP_CHANNELS = 8
# dw_mult holds a (multiplier, BLOCK) FP32 accumulator per program: up to 16
# output channels it stays within 128 registers per thread in every config.
# Larger multipliers take the other schedules.
MAX_MULT_MULTIPLIER = 16
_INT32_MAX = 2**31 - 1


class Schedule(NamedTuple):
    name: str
    # The compute capability the export GPU needs; 0: every NVIDIA GPU.
    min_capability: int


SCHEDULES = (
    Schedule("dw_flat", 0),
    Schedule("dw_row8", 0),
    Schedule("dw_mult", 0),
    Schedule("grp_plane", 0),
    Schedule("grp_reduce", 0),
)


def _strides_ordered(t: torch.Tensor, order: tuple[int, ...]) -> bool:
    strides = [t.stride(d) for d in order if t.shape[d] > 1]
    return all(s > 0 for s in strides) and all(
        a > b for a, b in zip(strides, strides[1:])
    )


def _geometry_reason(call: ConvCall) -> Optional[str]:
    x, w, p = call.input, call.weight, call.params
    if p.rank != 2:
        return f"direct stencil: {p.rank}-D convolution, needs 2-D"
    if p.transposed:
        return "direct stencil: transposed convolution"
    if any(d != 1 for d in p.dilation):
        return f"direct stencil: dilation {list(p.dilation)}"
    if any(o != 0 for o in p.output_padding):
        return f"direct stencil: output_padding {list(p.output_padding)}"
    channels = x.shape[1]
    depthwise = p.groups == channels == w.shape[0]
    if not (depthwise or (p.groups > 1 and w.shape[1] <= MAX_GROUP_CHANNELS)):
        return (
            "direct stencil: not a depthwise convolution or a grouped convolution "
            f"with at most {MAX_GROUP_CHANNELS} input channels per group "
            f"(groups {p.groups}, input {list(x.shape)}, weight {list(w.shape)})"
        )
    if max(w.shape[2:]) > MAX_KERNEL_SIZE:
        return (
            f"direct stencil: kernel {list(w.shape[2:])} larger than {MAX_KERNEL_SIZE}"
        )
    out_hw = [
        (i + pb + pa - k) // s + 1
        for i, k, s, pb, pa in zip(
            x.shape[2:], w.shape[2:], p.stride, p.padding_before, p.padding_after
        )
    ]
    if min(*x.shape, *out_hw) <= 0:
        return f"direct stencil: empty input or output (input {list(x.shape)})"
    return None


def supports(call: ConvCall) -> Optional[str]:
    """Why this template cannot run ``call``, or None. Assumes valid canonical
    arguments (``check_params``: positive strides, no negative padding)."""
    reason = _geometry_reason(call)
    if reason is not None:
        return reason
    x, w, b = call.input, call.weight, call.bias
    if x.dtype not in SUPPORTED_DTYPES:
        return f"direct stencil: dtype {x.dtype}"
    if w.dtype != x.dtype or (b is not None and b.dtype != x.dtype):
        return "direct stencil: input, weight and bias dtypes differ"
    if not (_strides_ordered(x, (0, 1, 2, 3)) or _strides_ordered(x, (0, 2, 3, 1))):
        return f"direct stencil: input strides {list(x.stride())} are neither NCHW nor channels-last"
    if b is not None and (tuple(b.shape) != (w.shape[0],) or not b.is_contiguous()):
        return "direct stencil: bias is not a contiguous [C_out] tensor"
    return None


# The benchmarked range (the benchmark's envelope and extent suites cover every
# dtype and layout supports() accepts within it).
MAX_ROUTED_STRIDE = 3
MAX_ROUTED_CHANNELS = 4096
MAX_ROUTED_SPATIAL_SIZE = 2048
MAX_ROUTED_INPUT_ELEMENTS = 32 * 128 * 112 * 112
# Grouped convolutions (the grouped suite): kernels up to 7x7 and strides up to
# 2. Inductor's template runs each group as a matrix product (reduction over the
# group's input channels and taps, one column per output channel); it is as
# fast as the stencil with 4 or more input and output channels per group and a
# kernel larger than 3x3, and in fp16/bf16, where it uses tensor cores, with two
# or more output channels per group and a reduction of 4 x 7 x 7 or more. The
# suite runs fp32 and bf16.
MAX_ROUTED_GROUPED_KERNEL = 7
MAX_ROUTED_GROUPED_STRIDE = 2
MIN_TEMPLATE_GROUP_CHANNELS = 4
MIN_TEMPLATE_HALF_REDUCTION = 4 * 7 * 7


def worth_routing(call: ConvCall) -> Optional[str]:
    """Why a call ``supports`` accepts is left to the regular lowering, or None."""
    x, w, p = call.input, call.weight, call.params
    if max(p.stride) > MAX_ROUTED_STRIDE:
        return f"direct stencil: stride {list(p.stride)} is not benchmarked"
    if any(
        max(b, a) > k - 1
        for b, a, k in zip(p.padding_before, p.padding_after, w.shape[2:])
    ):
        return (
            f"direct stencil: padding {list(p.padding_before)}/"
            f"{list(p.padding_after)} beyond kernel - 1 is not benchmarked"
        )
    channels = max(x.shape[1], w.shape[0])
    if channels > MAX_ROUTED_CHANNELS:
        return f"direct stencil: {channels} channels are not benchmarked"
    if max(x.shape[2:]) > MAX_ROUTED_SPATIAL_SIZE:
        return f"direct stencil: spatial size {list(x.shape[2:])} is not benchmarked"
    if x.numel() > MAX_ROUTED_INPUT_ELEMENTS:
        return f"direct stencil: a {x.numel()}-element input is not benchmarked"
    if w.shape[1] == 1 and w.shape[0] == p.groups:
        return None
    group_channels, multiplier = w.shape[1], w.shape[0] // p.groups
    if (
        min(group_channels, multiplier) >= MIN_TEMPLATE_GROUP_CHANNELS
        and max(w.shape[2:]) > 3
    ):
        return (
            f"direct stencil: {group_channels} input and {multiplier} output "
            "channels per group, where Inductor's template is as fast"
        )
    if x.dtype == torch.float16:
        return "direct stencil: grouped float16 is not benchmarked"
    reduction = group_channels * w.shape[2] * w.shape[3]
    if (
        x.dtype != torch.float32
        and multiplier >= 2
        and reduction >= MIN_TEMPLATE_HALF_REDUCTION
    ):
        return (
            f"direct stencil: {x.dtype} with {multiplier} output channels and a "
            f"reduction of {reduction} per group, where Inductor's template is as fast"
        )
    if max(w.shape[2:]) > MAX_ROUTED_GROUPED_KERNEL:
        return f"direct stencil: grouped kernel {list(w.shape[2:])} is not benchmarked"
    if max(p.stride) > MAX_ROUTED_GROUPED_STRIDE:
        return f"direct stencil: grouped stride {list(p.stride)} is not benchmarked"
    return None


def schedule_supports(name: str, call: ConvCall) -> Optional[str]:
    """Why schedule ``name`` cannot run ``call`` (which ``supports``), or None."""
    schedule = next((s for s in SCHEDULES if s.name == name), None)
    if schedule is None:
        return f"unknown schedule {name}"
    if not device_supports(schedule.min_capability, call.input.device):
        return f"{name} needs sm_{schedule.min_capability}"
    if name == "dw_mult":
        w = call.weight
        if w.shape[1] != 1 or w.shape[0] == call.params.groups:
            return "dw_mult needs a depthwise convolution with a channel multiplier"
        multiplier = w.shape[0] // call.params.groups
        if multiplier > MAX_MULT_MULTIPLIER:
            return f"dw_mult: channel multiplier {multiplier} above {MAX_MULT_MULTIPLIER}"
    if name in ("grp_plane", "grp_reduce"):
        w = call.weight
        if w.shape[1] == 1 and w.shape[0] == call.params.groups:
            return f"{name} needs a grouped convolution or a channel multiplier"
    if name == "dw_row8":
        if call.input.stride(1) != 1:
            return "dw_row8 needs channel-contiguous input"
    return None


@triton.jit
def _group(c, M: tl.constexpr):
    if M == 1:
        return c
    return c // M


@triton.jit
def _flat_row(
    acc,
    x_base,
    w_base,
    mask,
    oh,
    ow,
    kh,
    CG: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    SX_C: tl.constexpr,
    SX_H: tl.constexpr,
    SX_W: tl.constexpr,
    SW_CI: tl.constexpr,
    SW_H: tl.constexpr,
    SW_W: tl.constexpr,
):
    """Adds kernel row ``kh`` of every input channel of the group to ``acc``."""
    ih = oh * STRIDE_H + kh - PAD_H
    row_mask = mask & (ih >= 0) & (ih < H)
    for kw in tl.static_range(KW):
        iw = ow * STRIDE_W + kw - PAD_W
        tap_mask = row_mask & (iw >= 0) & (iw < W)
        for ci in tl.static_range(CG):
            xv = tl.load(
                x_base + ci * SX_C + ih * SX_H + iw * SX_W, mask=tap_mask, other=0.0
            )
            wv = tl.load(w_base + ci * SW_CI + kh * SW_H + kw * SW_W, mask=mask, other=0.0)
            acc += xv.to(tl.float32) * wv.to(tl.float32)
    return acc


@triton.jit
def _flat_row_same_size(
    acc,
    x_pos,
    w_base,
    mask,
    oh,
    ow,
    kh,
    CG: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    KW: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    SX_C: tl.constexpr,
    SW_CI: tl.constexpr,
    SW_H: tl.constexpr,
    SW_W: tl.constexpr,
):
    """``_flat_row`` for unit strides, an output the size of the input and
    contiguous input rows: a tap reads the output's own flat spatial offset
    plus a constant, so adjacent outputs read adjacent inputs and their loads
    can be vectorized."""
    # The bounds compare the output index itself with constants, which lets
    # Triton prove a mask uniform over adjacent outputs and vectorize the load.
    row_mask = mask & (oh >= PAD_H - kh) & (oh < H + PAD_H - kh)
    row = x_pos + (kh - PAD_H) * W
    for kw in tl.static_range(KW):
        tap_mask = row_mask & (ow >= PAD_W - kw) & (ow < W + PAD_W - kw)
        for ci in tl.static_range(CG):
            xv = tl.load(row + ci * SX_C + (kw - PAD_W), mask=tap_mask, other=0.0)
            wv = tl.load(w_base + ci * SW_CI + kh * SW_H + kw * SW_W, mask=mask, other=0.0)
            acc += xv.to(tl.float32) * wv.to(tl.float32)
    return acc


@triton.jit
def _flat_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    CG: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    SX_N: tl.constexpr,
    SX_C: tl.constexpr,
    SX_H: tl.constexpr,
    SX_W: tl.constexpr,
    SW_C: tl.constexpr,
    SW_CI: tl.constexpr,
    SW_H: tl.constexpr,
    SW_W: tl.constexpr,
    SO_N: tl.constexpr,
    SO_C: tl.constexpr,
    SO_H: tl.constexpr,
    SO_W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    CHANNELS_LAST: tl.constexpr,
    OUT_DENSE: tl.constexpr,
    INDEX_64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    TOTAL: tl.constexpr = N * C * OH * OW
    pid = tl.program_id(0)
    if INDEX_64:
        pid = pid.to(tl.int64)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    if TOTAL % BLOCK == 0:
        mask = tl.full((BLOCK,), True, tl.int1)
    else:
        mask = idx < TOTAL
    if CHANNELS_LAST:
        c = idx % C
        rest = idx // C
        ow = rest % OW
        rest = rest // OW
        oh = rest % OH
        n = rest // OH
    else:
        ow = idx % OW
        rest = idx // OW
        oh = rest % OH
        rest = rest // OH
        c = rest % C
        n = rest // C
    x_base = x_ptr + n * SX_N + _group(c, M) * (CG * SX_C)
    w_base = w_ptr + c * SW_C
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    # Unit strides, an output the size of the input and contiguous input rows:
    # a tap reads the output's own flat offset plus a constant.
    SAME_SIZE: tl.constexpr = (
        not CHANNELS_LAST
        and STRIDE_H == 1
        and STRIDE_W == 1
        and OH == H
        and OW == W
        and SX_W == 1
        and SX_H == W
    )
    # Depthwise calls unroll every tap; grouped calls unroll one kernel row,
    # which keeps their compile time flat in the kernel height. Depthwise
    # kernels narrower than 5 have no tap the flat offset lets Triton vectorize
    # beyond the center one it vectorizes anyway.
    if SAME_SIZE and (CG > 1 or KW >= 5):
        x_pos = x_base + idx % (OH * OW)
        if CG == 1:
            for kh in tl.static_range(KH):
                acc = _flat_row_same_size(
                    acc, x_pos, w_base, mask, oh, ow, kh, CG, H, W, KW, PAD_H,
                    PAD_W, SX_C, SW_CI, SW_H, SW_W,
                )
        else:
            for kh in range(KH):
                acc = _flat_row_same_size(
                    acc, x_pos, w_base, mask, oh, ow, kh, CG, H, W, KW, PAD_H,
                    PAD_W, SX_C, SW_CI, SW_H, SW_W,
                )
    elif CG == 1:
        for kh in tl.static_range(KH):
            acc = _flat_row(
                acc, x_base, w_base, mask, oh, ow, kh, CG, H, W, KW, STRIDE_H,
                STRIDE_W, PAD_H, PAD_W, SX_C, SX_H, SX_W, SW_CI, SW_H, SW_W,
            )
    else:
        for kh in range(KH):
            acc = _flat_row(
                acc, x_base, w_base, mask, oh, ow, kh, CG, H, W, KW, STRIDE_H,
                STRIDE_W, PAD_H, PAD_W, SX_C, SX_H, SX_W, SW_CI, SW_H, SW_W,
            )
    if HAS_BIAS:
        acc += tl.load(b_ptr + c, mask=mask, other=0.0).to(tl.float32)
    if OUT_DENSE:
        out_ptrs = out_ptr + idx
    else:
        out_ptrs = out_ptr + n * SO_N + c * SO_C + oh * SO_H + ow * SO_W
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _load_row(ptrs, mask, iw, W: tl.constexpr, SX_W: tl.constexpr):
    v = tl.load(ptrs + iw * SX_W, mask=mask & (iw >= 0) & (iw < W), other=0.0)
    return v.to(tl.float32)


@triton.jit
def _store_row(out_ptrs, acc, mask, ow, OW: tl.constexpr, SO_W: tl.constexpr):
    if OW % 8 != 0:
        mask = mask & (ow < OW)
    tl.store(out_ptrs + ow * SO_W, acc.to(out_ptrs.dtype.element_ty), mask=mask)


@triton.jit
def _row8_input_row(
    acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7,
    ptrs,
    row_mask,
    w_row,
    c_mask,
    iw0,
    W: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_W: tl.constexpr,
    SX_W: tl.constexpr,
    SW_W: tl.constexpr,
):
    """Adds one input row times one kernel row to the eight outputs, sliding
    the input window through registers (with unit W stride; otherwise each
    output loads its own window, and the weights still serve all eight)."""
    if STRIDE_W != 1:
        for kw in tl.static_range(KW):
            wv = tl.load(w_row + kw * SW_W, mask=c_mask, other=0.0).to(tl.float32)
            acc0 += _load_row(ptrs, row_mask, iw0 + kw, W, SX_W) * wv
            acc1 += _load_row(ptrs, row_mask, iw0 + STRIDE_W + kw, W, SX_W) * wv
            acc2 += _load_row(ptrs, row_mask, iw0 + 2 * STRIDE_W + kw, W, SX_W) * wv
            acc3 += _load_row(ptrs, row_mask, iw0 + 3 * STRIDE_W + kw, W, SX_W) * wv
            acc4 += _load_row(ptrs, row_mask, iw0 + 4 * STRIDE_W + kw, W, SX_W) * wv
            acc5 += _load_row(ptrs, row_mask, iw0 + 5 * STRIDE_W + kw, W, SX_W) * wv
            acc6 += _load_row(ptrs, row_mask, iw0 + 6 * STRIDE_W + kw, W, SX_W) * wv
            acc7 += _load_row(ptrs, row_mask, iw0 + 7 * STRIDE_W + kw, W, SX_W) * wv
        return acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7
    x0 = _load_row(ptrs, row_mask, iw0, W, SX_W)
    x1 = _load_row(ptrs, row_mask, iw0 + 1, W, SX_W)
    x2 = _load_row(ptrs, row_mask, iw0 + 2, W, SX_W)
    x3 = _load_row(ptrs, row_mask, iw0 + 3, W, SX_W)
    x4 = _load_row(ptrs, row_mask, iw0 + 4, W, SX_W)
    x5 = _load_row(ptrs, row_mask, iw0 + 5, W, SX_W)
    x6 = _load_row(ptrs, row_mask, iw0 + 6, W, SX_W)
    x7 = _load_row(ptrs, row_mask, iw0 + 7, W, SX_W)
    for kw in tl.static_range(KW):
        wv = tl.load(w_row + kw * SW_W, mask=c_mask, other=0.0)
        wv = wv.to(tl.float32)
        acc0 += x0 * wv
        acc1 += x1 * wv
        acc2 += x2 * wv
        acc3 += x3 * wv
        acc4 += x4 * wv
        acc5 += x5 * wv
        acc6 += x6 * wv
        acc7 += x7 * wv
        if kw + 1 < KW:
            x0, x1, x2, x3, x4, x5, x6 = x1, x2, x3, x4, x5, x6, x7
            x7 = _load_row(ptrs, row_mask, iw0 + kw + 8, W, SX_W)
    return acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7


@triton.jit
def _row8_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    CG: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    SX_N: tl.constexpr,
    SX_C: tl.constexpr,
    SX_H: tl.constexpr,
    SX_W: tl.constexpr,
    SW_C: tl.constexpr,
    SW_CI: tl.constexpr,
    SW_H: tl.constexpr,
    SW_W: tl.constexpr,
    SO_N: tl.constexpr,
    SO_C: tl.constexpr,
    SO_H: tl.constexpr,
    SO_W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    CHANNELS_LAST: tl.constexpr,
    OUT_DENSE: tl.constexpr,
    INDEX_64: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    ow0 = tl.program_id(1) * 8
    c = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
    if INDEX_64:
        row = row.to(tl.int64)
        ow0 = ow0.to(tl.int64)
        c = c.to(tl.int64)
    n = row // OH
    oh = row % OH
    if C % BLOCK_C == 0:
        c_mask = tl.full((BLOCK_C,), True, tl.int1)
    else:
        c_mask = c < C
    x_base = x_ptr + n * SX_N + _group(c, M) * (CG * SX_C)
    w_base = w_ptr + c * SW_C
    iw0 = ow0 * STRIDE_W - PAD_W
    acc0 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc4 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc5 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc6 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc7 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    # Depthwise calls unroll every tap; grouped calls unroll one kernel row,
    # which keeps their compile time flat in the kernel height.
    if CG == 1:
        for kh in tl.static_range(KH):
            ih = oh * STRIDE_H + kh - PAD_H
            row_mask = c_mask & (ih >= 0) & (ih < H)
            acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7 = _row8_input_row(
                acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7,
                x_base + ih * SX_H,
                row_mask,
                w_base + kh * SW_H,
                c_mask,
                iw0,
                W,
                KW,
                STRIDE_W,
                SX_W,
                SW_W,
            )
    else:
        for kh in range(KH):
            ih = oh * STRIDE_H + kh - PAD_H
            row_mask = c_mask & (ih >= 0) & (ih < H)
            for ci in tl.static_range(CG):
                acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7 = _row8_input_row(
                    acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7,
                    x_base + ci * SX_C + ih * SX_H,
                    row_mask,
                    w_base + ci * SW_CI + kh * SW_H,
                    c_mask,
                    iw0,
                    W,
                    KW,
                    STRIDE_W,
                    SX_W,
                    SW_W,
                )
    if HAS_BIAS:
        bias = tl.load(b_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
        acc0 += bias
        acc1 += bias
        acc2 += bias
        acc3 += bias
        acc4 += bias
        acc5 += bias
        acc6 += bias
        acc7 += bias
    out_ptrs = out_ptr + n * SO_N + c * SO_C + oh * SO_H
    _store_row(out_ptrs, acc0, c_mask, ow0, OW, SO_W)
    _store_row(out_ptrs, acc1, c_mask, ow0 + 1, OW, SO_W)
    _store_row(out_ptrs, acc2, c_mask, ow0 + 2, OW, SO_W)
    _store_row(out_ptrs, acc3, c_mask, ow0 + 3, OW, SO_W)
    _store_row(out_ptrs, acc4, c_mask, ow0 + 4, OW, SO_W)
    _store_row(out_ptrs, acc5, c_mask, ow0 + 5, OW, SO_W)
    _store_row(out_ptrs, acc6, c_mask, ow0 + 6, OW, SO_W)
    _store_row(out_ptrs, acc7, c_mask, ow0 + 7, OW, SO_W)


@triton.jit
def _multiplier_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    CG: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    SX_N: tl.constexpr,
    SX_C: tl.constexpr,
    SX_H: tl.constexpr,
    SX_W: tl.constexpr,
    SW_C: tl.constexpr,
    SW_CI: tl.constexpr,
    SW_H: tl.constexpr,
    SW_W: tl.constexpr,
    SO_N: tl.constexpr,
    SO_C: tl.constexpr,
    SO_H: tl.constexpr,
    SO_W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    CHANNELS_LAST: tl.constexpr,
    OUT_DENSE: tl.constexpr,
    INDEX_64: tl.constexpr,
    MP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """A depthwise convolution with channel multiplier ``M``: a block of output
    positions of one image and input channel, all ``M`` output channels of the
    channel per position. Each input value is loaded once for the ``M``
    outputs, and the weights are the same across the block."""
    G: tl.constexpr = C // M
    pid = tl.program_id(0)
    ng = tl.program_id(1)
    if INDEX_64:
        pid = pid.to(tl.int64)
        ng = ng.to(tl.int64)
    n = ng // G
    g = ng % G
    pos = pid * BLOCK + tl.arange(0, BLOCK)
    if (OH * OW) % BLOCK == 0:
        mask = tl.full((BLOCK,), True, tl.int1)
    else:
        mask = pos < OH * OW
    oh = pos // OW
    ow = pos % OW
    m = tl.arange(0, MP)
    m_mask = m < M
    c = g * M + m
    # Output channels on the outer dimension: each thread holds every channel
    # of its positions, so an input value feeds them without being reloaded.
    acc = tl.zeros((MP, BLOCK), dtype=tl.float32)
    x_base = x_ptr + n * SX_N + g * SX_C
    w_base = w_ptr + c * SW_C
    for kh in tl.static_range(KH):
        ih = oh * STRIDE_H + kh - PAD_H
        row_mask = mask & (ih >= 0) & (ih < H)
        for kw in tl.static_range(KW):
            iw = ow * STRIDE_W + kw - PAD_W
            xv = tl.load(
                x_base + ih * SX_H + iw * SX_W,
                mask=row_mask & (iw >= 0) & (iw < W),
                other=0.0,
            )
            wv = tl.load(w_base + kh * SW_H + kw * SW_W, mask=m_mask, other=0.0)
            acc += wv.to(tl.float32)[:, None] * xv.to(tl.float32)[None, :]
    if HAS_BIAS:
        acc += tl.load(b_ptr + c, mask=m_mask, other=0.0).to(tl.float32)[:, None]
    out_ptrs = (
        out_ptr
        + n * SO_N
        + c[:, None] * SO_C
        + oh[None, :] * SO_H
        + ow[None, :] * SO_W
    )
    tl.store(
        out_ptrs,
        acc.to(out_ptr.dtype.element_ty),
        mask=m_mask[:, None] & mask[None, :],
    )


@triton.jit
def _reduce_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    CG: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    SX_N: tl.constexpr,
    SX_C: tl.constexpr,
    SX_H: tl.constexpr,
    SX_W: tl.constexpr,
    SW_C: tl.constexpr,
    SW_CI: tl.constexpr,
    SW_H: tl.constexpr,
    SW_W: tl.constexpr,
    SO_N: tl.constexpr,
    SO_C: tl.constexpr,
    SO_H: tl.constexpr,
    SO_W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    CHANNELS_LAST: tl.constexpr,
    OUT_DENSE: tl.constexpr,
    INDEX_64: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """A tile of outputs times a block of their taps (group channel, kernel
    row, kernel column), summed across the tile: the reduction runs in
    parallel, for calls with few outputs and many taps per output."""
    TOTAL: tl.constexpr = N * C * OH * OW
    TAPS: tl.constexpr = CG * KH * KW
    pid = tl.program_id(0)
    if INDEX_64:
        pid = pid.to(tl.int64)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < TOTAL
    if CHANNELS_LAST:
        c = idx % C
        rest = idx // C
        ow = rest % OW
        rest = rest // OW
        oh = rest % OH
        n = rest // OH
    else:
        ow = idx % OW
        rest = idx // OW
        oh = rest % OH
        rest = rest // OH
        c = rest % C
        n = rest // C
    x_base = x_ptr + n * SX_N + _group(c, M) * (CG * SX_C)
    w_base = w_ptr + c * SW_C
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for r0 in range(0, TAPS, BLOCK_R):
        r = r0 + tl.arange(0, BLOCK_R)
        r_mask = r < TAPS
        ci = r // (KH * KW)
        kh = (r // KW) % KH
        kw = r % KW
        ih = oh[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = ow[:, None] * STRIDE_W + kw[None, :] - PAD_W
        tap_mask = (
            mask[:, None]
            & r_mask[None, :]
            & (ih >= 0)
            & (ih < H)
            & (iw >= 0)
            & (iw < W)
        )
        xv = tl.load(
            x_base[:, None] + ci[None, :] * SX_C + ih * SX_H + iw * SX_W,
            mask=tap_mask,
            other=0.0,
        )
        wv = tl.load(
            w_base[:, None] + (ci * SW_CI + kh * SW_H + kw * SW_W)[None, :],
            mask=mask[:, None] & r_mask[None, :],
            other=0.0,
        )
        acc += tl.sum(xv.to(tl.float32) * wv.to(tl.float32), axis=1)
    if HAS_BIAS:
        acc += tl.load(b_ptr + c, mask=mask, other=0.0).to(tl.float32)
    if OUT_DENSE:
        out_ptrs = out_ptr + idx
    else:
        out_ptrs = out_ptr + n * SO_N + c * SO_C + oh * SO_H + ow * SO_W
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _plane_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    CG: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    SX_N: tl.constexpr,
    SX_C: tl.constexpr,
    SX_H: tl.constexpr,
    SX_W: tl.constexpr,
    SW_C: tl.constexpr,
    SW_CI: tl.constexpr,
    SW_H: tl.constexpr,
    SW_W: tl.constexpr,
    SO_N: tl.constexpr,
    SO_C: tl.constexpr,
    SO_H: tl.constexpr,
    SO_W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    CHANNELS_LAST: tl.constexpr,
    OUT_DENSE: tl.constexpr,
    INDEX_64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """A block of positions of one output channel of one image: the weights
    are scalars shared by the block, loaded once per warp."""
    pid = tl.program_id(0)
    nc = tl.program_id(1)
    if INDEX_64:
        pid = pid.to(tl.int64)
        nc = nc.to(tl.int64)
    n = nc // C
    c = nc % C
    pos = pid * BLOCK + tl.arange(0, BLOCK)
    mask = pos < OH * OW
    oh = pos // OW
    ow = pos % OW
    x_base = x_ptr + n * SX_N + (c // M) * (CG * SX_C)
    w_base = w_ptr + c * SW_C
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for kh in range(KH):
        ih = oh * STRIDE_H + kh - PAD_H
        row_mask = mask & (ih >= 0) & (ih < H)
        for kw in tl.static_range(KW):
            iw = ow * STRIDE_W + kw - PAD_W
            tap_mask = row_mask & (iw >= 0) & (iw < W)
            for ci in tl.static_range(CG):
                xv = tl.load(
                    x_base + ci * SX_C + ih * SX_H + iw * SX_W, mask=tap_mask, other=0.0
                )
                wv = tl.load(w_base + ci * SW_CI + kh * SW_H + kw * SW_W)
                acc += xv.to(tl.float32) * wv.to(tl.float32)
    if HAS_BIAS:
        acc += tl.load(b_ptr + c).to(tl.float32)
    out_ptrs = out_ptr + n * SO_N + c * SO_C + oh * SO_H + ow * SO_W
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask)


_TUNING_KEY = [
    "N",
    "C",
    "CG",
    "M",
    "H",
    "W",
    "KH",
    "KW",
    "STRIDE_H",
    "STRIDE_W",
    "CHANNELS_LAST",
]
FLAT_CONFIGS = [
    triton.Config({"BLOCK": block}, num_warps=warps)
    for block, warps in ((256, 4), (512, 4), (512, 8), (1024, 8))
]
ROW8_CONFIGS = [
    triton.Config({"BLOCK_C": block}, num_warps=warps)
    for block, warps in ((16, 1), (16, 2), (32, 2), (32, 4), (64, 4))
]
MULT_CONFIGS = [
    triton.Config({"BLOCK": block}, num_warps=warps)
    for block, warps in ((256, 2), (512, 4), (1024, 4), (1024, 8), (2048, 8))
]
PLANE_CONFIGS = [
    triton.Config({"BLOCK": block}, num_warps=warps)
    for block, warps in ((64, 2), (128, 4), (256, 4), (512, 4), (1024, 8))
]
REDUCE_CONFIGS = [
    triton.Config({"BLOCK": block, "BLOCK_R": r}, num_warps=warps)
    for block, r, warps in ((16, 64, 4), (32, 32, 4), (8, 128, 4), (64, 16, 4), (32, 64, 8))
]
_KERNELS = {
    "dw_flat": triton.autotune(configs=FLAT_CONFIGS, key=_TUNING_KEY)(_flat_kernel),
    "dw_row8": triton.autotune(configs=ROW8_CONFIGS, key=_TUNING_KEY)(_row8_kernel),
    "dw_mult": triton.autotune(configs=MULT_CONFIGS, key=_TUNING_KEY)(
        _multiplier_kernel
    ),
    "grp_plane": triton.autotune(configs=PLANE_CONFIGS, key=_TUNING_KEY)(_plane_kernel),
    "grp_reduce": triton.autotune(configs=REDUCE_CONFIGS, key=_TUNING_KEY)(
        _reduce_kernel
    ),
}


def _grid(name: str, n: int, c: int, oh: int, ow: int, groups: int):
    if name in ("dw_flat", "grp_reduce"):
        return lambda meta: (triton.cdiv(n * c * oh * ow, meta["BLOCK"]),)
    if name == "grp_plane":
        return lambda meta: (triton.cdiv(oh * ow, meta["BLOCK"]), n * c)
    if name == "dw_mult":
        return lambda meta: (triton.cdiv(oh * ow, meta["BLOCK"]), n * groups)
    return lambda meta: (n * oh, triton.cdiv(ow, 8), triton.cdiv(c, meta["BLOCK_C"]))


def run_schedule(
    name: str,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
    params: ConvParams,
    config: Optional[triton.Config] = None,
) -> None:
    """Launches schedule ``name``; ``config`` replaces its autotuned config."""
    n, _, h, w = x.shape
    _, c, oh, ow = out.shape
    channels_last = out.stride(1) == 1 and c > 1
    dense = (
        (c * ow * oh, 1, c * ow, c) if channels_last else (c * oh * ow, oh * ow, ow, 1)
    )
    kernel = _KERNELS[name]
    options = {}
    if config is not None:
        kernel = kernel.fn
        options = {**config.kwargs, "num_warps": config.num_warps}
    if name == "dw_mult":
        # The group's output channels, padded to a power of two.
        options["MP"] = triton.next_power_of_2(c // params.groups)
    wrap_triton(kernel)[_grid(name, n, c, oh, ow, params.groups)](
        x,
        weight,
        x if bias is None else bias,
        out,
        n,
        c,
        weight.shape[1],
        c // params.groups,
        h,
        w,
        oh,
        ow,
        weight.shape[2],
        weight.shape[3],
        *params.stride,
        *params.padding_before,
        *x.stride(),
        *weight.stride(),
        *out.stride(),
        HAS_BIAS=bias is not None,
        CHANNELS_LAST=channels_last,
        OUT_DENSE=all(
            size == 1 or s == d for size, s, d in zip(out.shape, out.stride(), dense)
        ),
        INDEX_64=max(max_offset(x), max_offset(out), max_offset(weight)) > _INT32_MAX
        or n * c * oh * ow > _INT32_MAX,
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

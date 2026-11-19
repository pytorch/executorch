# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Weight streaming for 1-D convolutions with few output rows.

A schedule of the implicit-GEMM template (``stream_ffma``). When the batch
times the output length is small, every weight is used for only a few
outputs and the convolution is bound by reading the weights. Each program
reads ``[BLOCK_N, BLOCK_K]`` blocks of the native ``[C_out, C_in * taps]``
weight, contiguous along K, multiplies them in FP32 with the matching im2col
blocks of all output rows, and accumulates ``[BLOCK_N, rows]`` in registers.
K is split over a second grid axis into ``SPLIT`` parts, an autotuned config
like the block sizes, so that enough programs stream the weights at once;
``_reduce`` adds the partial sums in split order, adds the bias and stores
through the output strides.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl
from executorch.backends.cuda.triton.kernels.conv.core.common import (
    ConvCall,
    max_offset,
    out_shape_and_stride,
)
from torch.library import wrap_triton

MAX_ROWS = 64
_INT32_MAX = 2**31 - 1


def _rows(call: ConvCall) -> int:
    x, w, p = call.input, call.weight, call.params
    padded = x.shape[2] + p.padding_before[0] + p.padding_after[0]
    length = (padded - w.shape[2]) // p.stride[0] + 1
    return x.shape[0] * length


def reason(call: ConvCall) -> Optional[str]:
    """Why weight streaming cannot run ``call``, or None."""
    x, w, p = call.input, call.weight, call.params
    if p.rank != 1 or p.transposed:
        return "stream_ffma needs a forward 1-D convolution"
    if not w.is_contiguous():
        return "stream_ffma reads the weight in its contiguous native layout"
    if _rows(call) > MAX_ROWS:
        return f"stream_ffma keeps at most {MAX_ROWS} output rows in registers"
    shape, strides = out_shape_and_stride(call)
    out_offset = sum((n - 1) * s for n, s in zip(shape, strides) if n > 0)
    partial = MAX_SPLIT * shape[1] * triton.next_power_of_2(_rows(call))
    if max(max_offset(x), max_offset(w), out_offset, partial) > _INT32_MAX:
        return "stream_ffma uses 32-bit indices"
    return None


CONFIGS = [
    triton.Config({"BLOCK_N": n, "BLOCK_K": k, "SPLIT": split}, num_warps=4)
    for n in (16, 32, 64)
    for k in (32, 128)
    for split in (4, 16, 32)
]
MAX_SPLIT = max(c.kwargs["SPLIT"] for c in CONFIGS)


def _prune(configs, named_args, **kwargs):
    """Drops configs whose ``[BLOCK_N, BLOCK_K, rows]`` product would not fit
    in registers, or whose split leaves parts without a block of K."""
    args = {**named_args, **kwargs}
    k = args["C"] * args["TAPS"]
    kept = [
        c
        for c in configs
        if c.kwargs["BLOCK_N"] * c.kwargs["BLOCK_K"] * args["ROWS_PAD"] <= 65536
        and c.kwargs["SPLIT"] * c.kwargs["BLOCK_K"] <= k
    ]
    return kept or configs[:1]


@triton.autotune(
    configs=CONFIGS,
    key=["C", "T_IN", "T_OUT", "CO", "TAPS", "STRIDE", "ROWS_PAD"],
    prune_configs_by={"early_config_prune": _prune},
)
@triton.jit
def _stream(
    x_ptr,
    w_ptr,
    partial_ptr,
    split_ptr,
    C: tl.constexpr,
    T_IN: tl.constexpr,
    T_OUT: tl.constexpr,
    CO: tl.constexpr,
    TAPS: tl.constexpr,
    STRIDE: tl.constexpr,
    PAD: tl.constexpr,
    SX_N: tl.constexpr,
    SX_C: tl.constexpr,
    SX_T: tl.constexpr,
    ROWS: tl.constexpr,
    ROWS_PAD: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT: tl.constexpr,
):
    K: tl.constexpr = C * TAPS
    K_PER_SPLIT: tl.constexpr = (
        ((K + SPLIT - 1) // SPLIT + BLOCK_K - 1) // BLOCK_K * BLOCK_K
    )
    if (tl.program_id(0) == 0) & (tl.program_id(1) == 0):
        tl.store(split_ptr, SPLIT)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < CO
    r = tl.arange(0, ROWS_PAD)
    r_mask = r < ROWS
    image = r // T_OUT
    t0 = r % T_OUT * STRIDE - PAD
    k_first = tl.program_id(1) * K_PER_SPLIT
    acc = tl.zeros((BLOCK_N, ROWS_PAD), dtype=tl.float32)
    for k0 in range(0, K_PER_SPLIT, BLOCK_K):
        k = k_first + k0 + tl.arange(0, BLOCK_K)
        k_mask = k < K
        w = tl.load(
            w_ptr + offs_n[:, None] * K + k[None, :],
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        c = k // TAPS
        src = t0[None, :] + (k % TAPS)[:, None]
        x = tl.load(
            x_ptr + image[None, :] * SX_N + c[:, None] * SX_C + src * SX_T,
            mask=k_mask[:, None] & r_mask[None, :] & (src >= 0) & (src < T_IN),
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w[:, :, None] * x[None, :, :], axis=1)
    tl.store(
        partial_ptr + (tl.program_id(1) * CO + offs_n[:, None]) * ROWS_PAD + r[None, :],
        acc,
        mask=n_mask[:, None] & r_mask[None, :],
    )


@triton.jit
def _reduce(
    partial_ptr,
    split_ptr,
    b_ptr,
    out_ptr,
    CO: tl.constexpr,
    T_OUT: tl.constexpr,
    ROWS: tl.constexpr,
    ROWS_PAD: tl.constexpr,
    SO_N: tl.constexpr,
    SO_C: tl.constexpr,
    SO_T: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Sums the ``SPLIT`` partials the streaming kernel wrote, in split order,
    adds the bias once and stores."""
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    r = tl.arange(0, ROWS_PAD)
    n_mask = offs_n < CO
    mask = n_mask[:, None] & (r < ROWS)[None, :]
    acc = tl.zeros((BLOCK_N, ROWS_PAD), dtype=tl.float32)
    for split in range(tl.load(split_ptr)):
        acc += tl.load(
            partial_ptr + (split * CO + offs_n[:, None]) * ROWS_PAD + r[None, :],
            mask=mask,
            other=0.0,
        )
    if HAS_BIAS:
        acc += tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)[:, None]
    out = out_ptr + (r // T_OUT * SO_N + r % T_OUT * SO_T)[None, :]
    tl.store(out + offs_n[:, None] * SO_C, acc.to(out_ptr.dtype.element_ty), mask=mask)


def run(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
    params,
    config: Optional[triton.Config] = None,
) -> None:
    """Computes the convolution into ``out``; ``config`` replaces the
    streaming kernel's autotuned config."""
    n, c, t_in = x.shape
    co, _, taps = weight.shape
    t_out = out.shape[2]
    rows = n * t_out
    rows_pad = triton.next_power_of_2(rows)
    partial = x.new_empty((MAX_SPLIT, co, rows_pad), dtype=torch.float32)
    split = x.new_empty((1,), dtype=torch.int32)
    kernel, options = _stream, {}
    if config is not None:
        kernel = _stream.fn
        options = {**config.kwargs, "num_warps": config.num_warps}
    wrap_triton(kernel)[lambda meta: (triton.cdiv(co, meta["BLOCK_N"]), meta["SPLIT"])](
        x,
        weight,
        partial,
        split,
        c,
        t_in,
        t_out,
        co,
        taps,
        params.stride[0],
        params.padding_before[0],
        *x.stride(),
        rows,
        rows_pad,
        **options,
    )
    wrap_triton(_reduce)[(triton.cdiv(co, 64),)](
        partial,
        split,
        x if bias is None else bias,
        out,
        co,
        t_out,
        rows,
        rows_pad,
        *out.stride(),
        HAS_BIAS=bias is not None,
        BLOCK_N=64,
    )

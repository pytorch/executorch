# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``gemv``: weight streaming for calls of at most ``MAX_ROWS`` rows.

With so few rows the product is bound by reading the weight once: each
program streams a ``BLOCK_N x BLOCK_K`` slab of the weight at a time and
multiplies it on CUDA cores, in FP32, with every row of ``x``, summing each
slab's products in a tree and the slabs in order. Nothing else is on the
critical path, so many programs in flight keep the memory busy. Strict FP32
costs nothing here. A launch is deterministic: the order depends only on the
config.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import torch
import triton
import triton.language as tl
from torch.library import wrap_triton

if TYPE_CHECKING:
    from executorch.backends.cuda.triton.kernels.gemm.linear import LinearLayout

MAX_ROWS = 4


@triton.jit
def _row(acc, w, x_ptr, row, M, SXK, rk, k_mask):
    x = tl.load(x_ptr + rk * SXK, mask=k_mask & (row < M), other=0.0)
    return acc + tl.sum(w * x[None, :], axis=1)


@triton.jit
def _store(out_ptr, acc, bias, row, M, SOM, SON, rn, n_mask):
    tl.store(out_ptr + row * SOM + rn * SON, acc + bias, mask=n_mask & (row < M))


@triton.jit
def _gemv_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    M,
    N,
    K,
    SXM,
    SXK,
    SWN,
    SWK,
    SOM,
    SON,
    ROWS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    INDEX_64: tl.constexpr,
    VEC: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # The row strides come divided by VEC: multiplied back by a constant, they
    # show Triton that every row starts on a VEC-element boundary, so with
    # aligned bases (``vector_width``) rows of any length that is a multiple
    # of VEC load as vectors.
    if INDEX_64:
        SXM = SXM.to(tl.int64)
        SWN = SWN.to(tl.int64)
    SXM = SXM * VEC
    SWN = SWN * VEC
    rn = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    if INDEX_64:
        rn = rn.to(tl.int64)
        rk = rk.to(tl.int64)
    n_mask = rn < N
    acc0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_ptrs = w_ptr + rn[:, None] * SWN + rk[None, :] * SWK
    x_ptrs = x_ptr
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = rk < K - k * BLOCK_K
        w = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        acc0 = _row(acc0, w, x_ptrs, 0, M, SXK, rk, k_mask)
        if ROWS > 1:
            acc1 = _row(acc1, w, x_ptrs + SXM, 1, M, SXK, rk, k_mask)
        if ROWS > 2:
            acc2 = _row(acc2, w, x_ptrs + 2 * SXM, 2, M, SXK, rk, k_mask)
            acc3 = _row(acc3, w, x_ptrs + 3 * SXM, 3, M, SXK, rk, k_mask)
        w_ptrs += BLOCK_K * SWK
        x_ptrs += BLOCK_K * SXK
    bias = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAS_BIAS:
        bias = tl.load(b_ptr + rn, mask=n_mask, other=0.0)
    _store(out_ptr, acc0, bias, 0, M, SOM, SON, rn, n_mask)
    if ROWS > 1:
        _store(out_ptr, acc1, bias, 1, M, SOM, SON, rn, n_mask)
    if ROWS > 2:
        _store(out_ptr, acc2, bias, 2, M, SOM, SON, rn, n_mask)
        _store(out_ptr, acc3, bias, 3, M, SOM, SON, rn, n_mask)


def _prune(configs, named_args, **kwargs):
    """Drops slabs longer than the reduction (keeps the shortest otherwise)."""
    k = named_args["K"]
    kept = [
        c for c in configs if c.kwargs["BLOCK_K"] <= max(128, triton.next_power_of_2(k))
    ]
    return kept or [configs[0]]


CONFIGS = [
    triton.Config({"BLOCK_N": block_n, "BLOCK_K": block_k}, num_warps=warps)
    for block_n, block_k, warps in (
        (16, 128, 4),
        (16, 256, 4),
        (16, 512, 8),
        (32, 128, 4),
        (32, 256, 8),
        (64, 128, 8),
    )
]
_KERNEL = triton.autotune(
    configs=CONFIGS,
    key=[
        "M",
        "N",
        "K",
        "SXM",
        "SXK",
        "SWN",
        "SWK",
        "SOM",
        "SON",
        "ROWS",
        "HAS_BIAS",
        "VEC",
    ],
    prune_configs_by={"early_config_prune": _prune},
)(_gemv_kernel)


def supports(m: int) -> Optional[str]:
    """Why the schedule cannot run a call of ``m`` rows, or None."""
    return None if m <= MAX_ROWS else f"gemv streams at most {MAX_ROWS} rows, not {m}"


def vector_width(x: torch.Tensor, weight: torch.Tensor, layout: LinearLayout) -> int:
    """4 when the rows and bases of both operands are 16-byte aligned, so the
    slabs load as vectors; 1 otherwise."""
    (sxm, sxk), (swn, swk) = layout.x, layout.weight
    aligned = (
        sxk == swk == 1
        and sxm % 4 == 0
        and swn % 4 == 0
        and x.storage_offset() % 4 == 0
        and weight.storage_offset() % 4 == 0
    )
    return 4 if aligned else 1


def run(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
    layout: LinearLayout,
    config: Optional[triton.Config] = None,
) -> None:
    """Launches the schedule; ``config`` replaces its autotuned config."""
    m, n, k = layout.m, layout.n, layout.k
    kernel, options = _KERNEL, {}
    if config is not None:
        kernel = _KERNEL.fn
        options = {**config.kwargs, "num_warps": config.num_warps}
    vec = vector_width(x, weight, layout)
    (sxm, sxk), (swn, swk) = layout.x, layout.weight
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_N"]),)  # noqa: E731
    wrap_triton(kernel)[grid](
        x,
        weight,
        x if bias is None else bias,
        out,
        m,
        n,
        k,
        sxm // vec,
        sxk,
        swn // vec,
        swk,
        *layout.out,
        ROWS=1 if m == 1 else (2 if m == 2 else 4),
        HAS_BIAS=bias is not None,
        INDEX_64=layout.index_64(),
        VEC=vec,
        **options,
    )

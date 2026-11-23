# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tile schedules: one output tile per program, in FP32.

Each program computes a ``BLOCK_M x BLOCK_N`` tile of ``x @ weight.T`` over
its slice of K with ``tl.dot`` in one of two precisions, both FP32-accurate:

- ``ieee``: FP32 FMAs (never TF32), on every NVIDIA GPU;
- ``tf32x3``: each operand split into a TF32 high and low part, with
  ``hi * hi + hi * lo + lo * hi`` accumulated in FP32 on tensor cores
  (sm_80 and newer).

Within a slice, every ``CHUNKS`` blocks of K are reduced into a partial sum
that is then added to the total, so each output's rounding chain is
``CHUNKS * BLOCK_K + K / (CHUNKS * BLOCK_K)`` adds instead of K and long
reductions stay as accurate as cuBLAS. Tiles are visited in groups of
``GROUP_M`` rows for L2 reuse.

With a split of 1, one program covers all of K and stores the tile with the
bias. With a split of S, S programs per tile each cover a slice of whole
blocks and store an FP32 partial into an ``[S, M, N]`` workspace, which
``_reduce_kernel`` sums in fixed order, adding the bias once. A launch is
deterministic: its summation order depends only on the config and the split.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import torch
import triton
import triton.language as tl
from torch.library import wrap_triton

if TYPE_CHECKING:
    from executorch.backends.cuda.triton.kernels.gemm.linear import LinearLayout

PRECISIONS = ("ieee", "tf32x3")


@triton.jit
def _tile_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    ws_ptr,
    M,
    N,
    K,
    SXM,
    SXK,
    SWN,
    SWK,
    SOM,
    SON,
    SPLIT,
    PRECISION: tl.constexpr,
    PARTIAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    INDEX_64: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    CHUNKS: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group = pid // width
    group_size = tl.minimum(grid_m - group * GROUP_M, GROUP_M)
    pid_m = group * GROUP_M + (pid % width) % group_size
    pid_n = (pid % width) // group_size
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    if INDEX_64:
        rm = rm.to(tl.int64)
        rn = rn.to(tl.int64)
        rk = rk.to(tl.int64)
    m_mask = rm < M
    n_mask = rn < N
    k_per_split = tl.cdiv(tl.cdiv(K, BLOCK_K), SPLIT) * BLOCK_K
    k0 = pid_k * k_per_split
    k_len = tl.minimum(k0 + k_per_split, K) - k0
    x_ptrs = x_ptr + rm[:, None] * SXM + (k0 + rk)[None, :] * SXK
    w_ptrs = w_ptr + (k0 + rk)[:, None] * SWK + rn[None, :] * SWN
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    part = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(k_len, BLOCK_K)):
        k_left = k_len - k * BLOCK_K
        a = tl.load(x_ptrs, mask=m_mask[:, None] & (rk[None, :] < k_left), other=0.0)
        b = tl.load(w_ptrs, mask=(rk[:, None] < k_left) & n_mask[None, :], other=0.0)
        part = tl.dot(a, b, part, input_precision=PRECISION)
        if (k + 1) % CHUNKS == 0:
            acc += part
            part = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        x_ptrs += BLOCK_K * SXK
        w_ptrs += BLOCK_K * SWK
    acc += part
    mask = m_mask[:, None] & n_mask[None, :]
    if PARTIAL:
        ws_ptrs = ws_ptr + pid_k * M * N + rm[:, None] * N + rn[None, :]
        tl.store(ws_ptrs, acc, mask=mask)
    else:
        if HAS_BIAS:
            acc += tl.load(b_ptr + rn, mask=n_mask, other=0.0)[None, :]
        tl.store(out_ptr + rm[:, None] * SOM + rn[None, :] * SON, acc, mask=mask)


@triton.jit
def _reduce_kernel(
    ws_ptr,
    b_ptr,
    out_ptr,
    M,
    N,
    SOM,
    SON,
    SPLIT,
    HAS_BIAS: tl.constexpr,
    INDEX_64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if INDEX_64:
        idx = idx.to(tl.int64)
    mask = idx < M * N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(0, SPLIT):
        acc += tl.load(ws_ptr + s * M * N + idx, mask=mask, other=0.0)
    m = idx // N
    n = idx % N
    if HAS_BIAS:
        acc += tl.load(b_ptr + n, mask=mask, other=0.0)
    tl.store(out_ptr + m * SOM + n * SON, acc, mask=mask)


def _prune(configs, named_args, **kwargs):
    """Drops tiles wider than the call needs (keeps the narrowest otherwise)."""
    m, n = named_args["M"], named_args["N"]
    kept = [
        c
        for c in configs
        if c.kwargs["BLOCK_M"] <= max(16, triton.next_power_of_2(m))
        and c.kwargs["BLOCK_N"] <= max(64, triton.next_power_of_2(n))
    ]
    return kept or [configs[0]]


CONFIGS = [
    triton.Config(
        {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": 32,
            "GROUP_M": 8,
            "CHUNKS": 4,
        },
        num_warps=warps,
        num_stages=stages,
    )
    for block_m, block_n, warps, stages in (
        (16, 64, 4, 3),
        (16, 128, 4, 3),
        (32, 64, 4, 3),
        (32, 128, 4, 3),
        (64, 64, 4, 3),
        (64, 128, 4, 3),
        (128, 64, 4, 3),
        (128, 128, 8, 2),
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
        "SPLIT",
        "PRECISION",
        "HAS_BIAS",
    ],
    prune_configs_by={"early_config_prune": _prune},
)(_tile_kernel)
_REDUCE_BLOCK = 1024


def run(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
    layout: LinearLayout,
    precision: str,
    split: int,
    config: Optional[triton.Config] = None,
) -> None:
    """Launches the schedule in ``precision`` with K split ``split`` ways;
    ``config`` replaces its autotuned config. A split above 1 allocates the
    ``[split, M, N]`` FP32 workspace."""
    m, n, k = layout.m, layout.n, layout.k
    kernel, options = _KERNEL, {}
    if config is not None:
        kernel = _KERNEL.fn
        options = {
            **config.kwargs,
            "num_warps": config.num_warps,
            "num_stages": config.num_stages,
        }
    partial = split > 1
    workspace = (
        torch.empty((split, m, n), dtype=torch.float32, device=x.device)
        if partial
        else out
    )
    grid = lambda meta: (  # noqa: E731
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
        split,
    )
    wrap_triton(kernel)[grid](
        x,
        weight,
        x if bias is None else bias,
        out,
        workspace,
        m,
        n,
        k,
        *layout.x,
        *layout.weight,
        *layout.out,
        split,
        PRECISION=precision,
        PARTIAL=partial,
        HAS_BIAS=bias is not None,
        INDEX_64=layout.index_64(),
        **options,
    )
    if partial:
        wrap_triton(_reduce_kernel)[(triton.cdiv(m * n, _REDUCE_BLOCK),)](
            workspace,
            x if bias is None else bias,
            out,
            m,
            n,
            *layout.out,
            split,
            HAS_BIAS=bias is not None,
            INDEX_64=layout.index_64(),
            BLOCK=_REDUCE_BLOCK,
            num_warps=4,
        )

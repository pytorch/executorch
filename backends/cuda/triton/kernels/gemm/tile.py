# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``ieee_tile``: one output tile per program, strict FP32.

Each program computes a ``BLOCK_M x BLOCK_N`` tile of ``x @ weight.T`` over
the whole of K with ``tl.dot(input_precision="ieee")`` (FP32 FMAs, never
TF32), adds the bias in FP32 and stores the tile once; tiles are visited in
groups of ``GROUP_M`` rows for L2 reuse. It needs no workspace and is legal
for every call ``triton::linear`` accepts on every NVIDIA GPU, which makes it
the base schedule. A launch is deterministic: the reduction order depends
only on the config.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import torch
import triton
import triton.language as tl
from torch.library import wrap_triton

if TYPE_CHECKING:
    from executorch.backends.cuda.triton.kernels.gemm.linear import LinearLayout


@triton.jit
def _ieee_tile_kernel(
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
    HAS_BIAS: tl.constexpr,
    INDEX_64: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    CHUNKS: tl.constexpr,
):
    pid = tl.program_id(0)
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
    x_ptrs = x_ptr + rm[:, None] * SXM + rk[None, :] * SXK
    w_ptrs = w_ptr + rk[:, None] * SWK + rn[None, :] * SWN
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    part = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Two-level summation: every CHUNKS blocks of K are reduced into a
    # partial sum, which is then added to the total, so each output's rounding
    # chain is CHUNKS * BLOCK_K + K / (CHUNKS * BLOCK_K) adds instead of K and
    # long reductions stay as accurate as cuBLAS.
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_left = K - k * BLOCK_K
        a = tl.load(x_ptrs, mask=m_mask[:, None] & (rk[None, :] < k_left), other=0.0)
        b = tl.load(w_ptrs, mask=(rk[:, None] < k_left) & n_mask[None, :], other=0.0)
        part = tl.dot(a, b, part, input_precision="ieee")
        if (k + 1) % CHUNKS == 0:
            acc += part
            part = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        x_ptrs += BLOCK_K * SXK
        w_ptrs += BLOCK_K * SWK
    acc += part
    if HAS_BIAS:
        acc += tl.load(b_ptr + rn, mask=n_mask, other=0.0)[None, :]
    out_ptrs = out_ptr + rm[:, None] * SOM + rn[None, :] * SON
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


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
    key=["M", "N", "K", "SXM", "SXK", "SWN", "SWK", "SOM", "SON", "HAS_BIAS"],
    prune_configs_by={"early_config_prune": _prune},
)(_ieee_tile_kernel)


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
        options = {
            **config.kwargs,
            "num_warps": config.num_warps,
            "num_stages": config.num_stages,
        }
    grid = lambda meta: (  # noqa: E731
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    wrap_triton(kernel)[grid](
        x,
        weight,
        x if bias is None else bias,
        out,
        m,
        n,
        k,
        *layout.x,
        *layout.weight,
        *layout.out,
        HAS_BIAS=bias is not None,
        INDEX_64=layout.index_64(),
        **options,
    )

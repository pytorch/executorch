# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""NVFP4 gate/up GEMM with the BF16 SiLU/mul epilogue."""

import torch
import triton
import triton.language as tl

from executorch.backends.cuda.autotune.launch_params import autotune_launch_param
from executorch.backends.cuda.triton.kernels.nvfp4_quantized_gemm import (
    _check_operands,
    _check_split,
    _has_native_fp4_conversion,
    _max_rows,
    _prune_w4a4,
    _w4a4,
    nvfp4_quantize_activation,
    SPLIT_K_CANDIDATES,
)
from executorch.backends.cuda.triton.kernels.nvfp4_tuning import (
    autotune_do_bench,
    launch_nvfp4_kernel,
    nvfp4_launch_bench,
)
from executorch.backends.cuda.triton.kernels.quantized_gemm_family import (
    _compile_every_candidate_at_once,
    _QUEUING_ONLY,
)
from torch.library import triton_op, wrap_triton
from triton.language.extra.cuda import libdevice


@triton.jit
def _silu_mul(gate, up, DTYPE: tl.constexpr):
    gate = gate.to(DTYPE).to(tl.float32)
    up = up.to(DTYPE).to(tl.float32)
    activated = (gate / (1.0 + libdevice.exp(-gate))).to(DTYPE).to(tl.float32)
    return activated * up


@triton.autotune(
    configs=_w4a4.configs,
    key=["M", "N", "K", "SPLIT_K", "ROW_PTS"],
    prune_configs_by={"early_config_prune": _prune_w4a4},
    do_bench=autotune_do_bench,
)
@triton.jit
def _gate_up(
    A,
    AS,
    AG,
    B,
    BS,
    PTS,
    C,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
    ROW_PTS: tl.constexpr,
):
    pid = tl.program_id(0)
    split = tl.program_id(1)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(N // 2, BN // 2)
    first = (pid // (GROUP_M * nn)) * GROUP_M
    gm = tl.minimum(nm - first, GROUP_M)
    m = ((first + pid % gm) * BM + tl.arange(0, BM)).to(tl.int64)
    tile = (pid % (GROUP_M * nn)) // gm
    lane = tl.arange(0, BN)
    column = tile * (BN // 2) + lane % (BN // 2)
    n = (column + (lane // (BN // 2)) * (N // 2)).to(tl.int64)
    kp = tl.arange(0, BK // 2)
    sp = tl.arange(0, BK // 16)
    acc = tl.zeros((BM, BN), tl.float32)
    chunk: tl.constexpr = tl.cdiv(K, BK * SPLIT_K)
    for offset in range(chunk):
        block = split * chunk + offset
        k = block * (BK // 2) + kp
        s = block * (BK // 16) + sp
        a = tl.load(
            A + m[:, None] * (K // 2) + k[None, :],
            (m[:, None] < M) & (k[None, :] < K // 2),
            0,
        )
        b = tl.load(
            B + n[None, :] * (K // 2) + k[:, None],
            (column[None, :] < N // 2) & (k[:, None] < K // 2),
            0,
        )
        sa = tl.load(
            AS + m[:, None] * (K // 16) + s[None, :],
            (m[:, None] < M) & (s[None, :] < K // 16),
            0,
        ).to(tl.float8e4nv, bitcast=True)
        sb = tl.load(
            BS + n[:, None] * (K // 16) + s[None, :],
            (column[:, None] < N // 2) & (s[None, :] < K // 16),
            0,
        ).to(tl.float8e4nv, bitcast=True)
        acc = tl.dot_scaled(a, sa, "e2m1", b, sb, "e2m1", acc)
    if SPLIT_K > 1:
        tl.store(
            C + split.to(tl.int64) * M * N + m[:, None] * N + n[None, :],
            acc,
            (m[:, None] < M) & (column[None, :] < N // 2),
        )
    else:
        g = tl.load(AG + m, m < M, 0)
        pts = tl.load(PTS + n, column < N // 2, 0)[None, :] if ROW_PTS else tl.load(PTS)
        value = acc * g[:, None] * pts
        gate, up = tl.split(tl.reshape(value, (BM, 2, BN // 2)).trans(0, 2, 1))
        result = _silu_mul(gate, up, C.dtype.element_ty)
        output_n = tile * (BN // 2) + tl.arange(0, BN // 2)
        tl.store(
            C + m[:, None] * (N // 2) + output_n[None, :],
            result,
            (m[:, None] < M) & (output_n[None, :] < N // 2),
        )


@triton.jit
def _reduce_gate_up(
    P,
    G,
    PTS,
    Y,
    M,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
    ROW_PTS: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row, column = index // (N // 2), index % (N // 2)
    mask = index < tl.cast(M, tl.int64) * (N // 2)
    split = tl.arange(0, SPLIT_K)
    offset = split[:, None].to(tl.int64) * M * N + row[None, :] * N + column[None, :]
    gate = tl.sum(tl.load(P + offset, mask[None, :], 0), 0)
    up = tl.sum(tl.load(P + offset + N // 2, mask[None, :], 0), 0)
    g = tl.load(G + row, mask, 0)
    gate_pts = tl.load(PTS + column, mask, 0) if ROW_PTS else tl.load(PTS)
    up_pts = tl.load(PTS + column + N // 2, mask, 0) if ROW_PTS else gate_pts
    result = _silu_mul(gate * (g * gate_pts), up * (g * up_pts), Y.dtype.element_ty)
    tl.store(Y + index, result, mask)


@triton.jit
def _gate_up_epilogue(X, Y, M, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    column, row = index % (N // 2), index // (N // 2)
    mask = index < tl.cast(M, tl.int64) * (N // 2)
    gate = tl.load(X + row * N + column, mask, 0)
    up = tl.load(X + row * N + column + N // 2, mask, 0)
    tl.store(Y + index, _silu_mul(gate, up, Y.dtype.element_ty), mask)


@triton_op("triton::nvfp4_gate_up_epilogue", mutates_args={})
def nvfp4_gate_up_epilogue(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    if n % 2 or not x.is_contiguous():
        raise ValueError("NVFP4 gate/up epilogue requires a contiguous, even-N input")
    out = torch.empty((m, n // 2), device=x.device, dtype=x.dtype)
    if not _QUEUING_ONLY.get():
        wrap_triton(_gate_up_epilogue)[(triton.cdiv(m * (n // 2), 1024),)](
            x, out, m, n, 1024, num_warps=4
        )
    return out


def _launch_w4a4_gate_up_impl(
    x, qdata, scale, pts, static_scale, max_rows, *, split_k, output_dtype=None
):
    m, k = x.shape
    n = qdata.shape[0]
    if n % 2:
        raise ValueError("NVFP4 gate/up requires an even output dimension")
    _check_split(k, split_k, m=max_rows, n=n)
    a, sa, g = nvfp4_quantize_activation(x, static_scale)
    out = torch.empty((m, n // 2), device=x.device, dtype=output_dtype or x.dtype)
    partial = (
        torch.empty((split_k, m, n), device=x.device, dtype=torch.float32)
        if split_k > 1
        else out
    )

    def grid(meta):
        return (
            triton.cdiv(m, meta["BM"]) * triton.cdiv(n // 2, meta["BN"] // 2),
            split_k,
        )

    launch_nvfp4_kernel(
        _gate_up,
        grid,
        (a, sa, g, qdata, scale, pts, partial, m, n, k, split_k),
        7,
        ROW_PTS=pts.numel() != 1,
    )
    if split_k > 1 and not _QUEUING_ONLY.get():
        wrap_triton(_reduce_gate_up)[(triton.cdiv(m * (n // 2), 256),)](
            partial, g, pts, out, m, n, split_k, 256, ROW_PTS=pts.numel() != 1
        )
    return out


_launch_w4a4_gate_up = autotune_launch_param(
    "split_k",
    SPLIT_K_CANDIDATES,
    candidate_prepass=_compile_every_candidate_at_once,
    do_bench=nvfp4_launch_bench,
)(_launch_w4a4_gate_up_impl)


@triton_op("triton::nvfp4_w4a4_gate_up", mutates_args={})
def nvfp4_w4a4_gate_up(
    x: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    static_scale: float = 0.0,
    split_k: int = -1,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    _check_operands(x, qdata, scale, per_tensor_scale)
    if not _has_native_fp4_conversion():
        raise RuntimeError("Native NVFP4 W4A4 requires an SM120 or newer CUDA target")
    launch = _launch_w4a4_gate_up if split_k < 0 else _launch_w4a4_gate_up_impl
    kwargs = {} if split_k < 0 else {"split_k": split_k}
    if output_dtype is not None:
        kwargs["output_dtype"] = output_dtype
    return launch(
        x,
        qdata,
        scale,
        per_tensor_scale,
        static_scale,
        _max_rows(x.shape[0]),
        **kwargs,
    )

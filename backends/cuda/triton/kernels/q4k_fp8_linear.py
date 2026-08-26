# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pure-Triton Q4_K -> FP8 prefill linear for SM90+ GPUs."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _cast_bf16_to_fp8_kernel(x, out, n_elements, XBLOCK: tl.constexpr):
    idx = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    mask = idx < n_elements
    value = tl.load(x + idx, mask=mask, other=0.0)
    tl.store(out + idx, value.to(tl.float8e4nv), mask=mask)


@triton.jit
def _dequant_q4k_fp8_kernel(
    qdata,
    scale_code,
    scale_step,
    zero_code,
    zero_step,
    out,
    n_elements,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_SUPER: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    mask = idx < n_elements
    n = idx // K
    k = idx - n * K
    packed = tl.load(qdata + n * (K // 2) + k // 2, mask=mask, other=0).to(tl.uint8)
    q = ((packed >> ((k & 1) * 4)) & 0xF).to(tl.float32)
    group = k // GROUP_SIZE
    super_group = group // GROUPS_PER_SUPER
    groups = K // GROUP_SIZE
    supers = groups // GROUPS_PER_SUPER
    sc = tl.load(scale_code + n * groups + group, mask=mask, other=0).to(tl.float32)
    ss = tl.load(scale_step + n * supers + super_group, mask=mask, other=0.0).to(
        tl.float32
    )
    zc = tl.load(zero_code + n * groups + group, mask=mask, other=0).to(tl.float32)
    zs = tl.load(zero_step + n * supers + super_group, mask=mask, other=0.0).to(
        tl.float32
    )
    value = (q - zc * zs) * (sc * ss)
    tl.store(out + idx, value.to(tl.float8e4nv), mask=mask)


_FP8_MM_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
        num_warps=8,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8},
        num_warps=8,
        num_stages=3,
    ),
]


def _validate_q4k_fp8_inputs(
    x: torch.Tensor, qdata: torch.Tensor, group_size: int
) -> None:
    if x.dtype != torch.bfloat16:
        raise RuntimeError(f"Expected bfloat16 activations, got {x.dtype}")
    if not x.is_contiguous():
        raise RuntimeError("Q4_K FP8 requires contiguous activations")
    if qdata.dtype != torch.uint8:
        raise RuntimeError(f"Expected uint8 packed weights, got {qdata.dtype}")
    if group_size != 32:
        raise RuntimeError(f"Q4_K FP8 requires group_size=32, got {group_size}")
    if x.shape[1] % 256 != 0:
        raise RuntimeError(f"Q4_K FP8 requires K divisible by 256, got {x.shape[1]}")


@triton.autotune(configs=_FP8_MM_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _fp8_mm_kernel(
    a,
    b,
    c,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    n_m = tl.cdiv(M, BLOCK_M)
    n_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * n_n
    group = pid // width
    first_m = group * GROUP_M
    group_m = tl.minimum(n_m - first_m, GROUP_M)
    pid_m = first_m + ((pid % width) % group_m)
    pid_n = (pid % width) // group_m
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for kt in range(0, tl.cdiv(K, BLOCK_K)):
        k = kt * BLOCK_K + offs_k
        av = tl.load(
            a + offs_m[:, None] * K + k[None, :],
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        bv = tl.load(
            b + offs_n[None, :] * K + k[:, None],
            mask=(offs_n[None, :] < N) & (k[:, None] < K),
            other=0.0,
        )
        acc += tl.dot(av, bv)
    tl.store(
        c + offs_m[:, None] * N + offs_n[None, :],
        acc.to(tl.bfloat16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton_op("triton::q4k_fp8_linear", mutates_args={})
def q4k_fp8_linear(
    x: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    scale_step: torch.Tensor,
    zero: torch.Tensor,
    zero_point_step: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Run Q4_K prefill linear through FP8 tensor cores on SM90+."""
    M, K = x.shape
    N = qdata.shape[0]
    _validate_q4k_fp8_inputs(x, qdata, group_size)
    groups_per_super = 256 // group_size
    x_fp8 = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x.device)
    w_fp8 = torch.empty((N, K), dtype=torch.float8_e4m3fn, device=x.device)
    output = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)

    wrap_triton(_cast_bf16_to_fp8_kernel)[
        lambda meta: (triton.cdiv(M * K, meta["XBLOCK"]),)
    ](x, x_fp8, M * K, XBLOCK=512, num_warps=4, num_stages=1)
    wrap_triton(_dequant_q4k_fp8_kernel)[
        lambda meta: (triton.cdiv(N * K, meta["XBLOCK"]),)
    ](
        qdata,
        scale,
        scale_step,
        zero,
        zero_point_step,
        w_fp8,
        N * K,
        K,
        group_size,
        groups_per_super,
        XBLOCK=512,
        num_warps=4,
        num_stages=1,
    )
    wrap_triton(_fp8_mm_kernel)[
        lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
        )
    ](x_fp8, w_fp8, output, M, N, K)
    return output


@q4k_fp8_linear.register_fake
def _q4k_fp8_linear_fake(
    x: torch.Tensor,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    scale_step: torch.Tensor,
    zero: torch.Tensor,
    zero_point_step: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    _validate_q4k_fp8_inputs(x, qdata, group_size)
    return torch.empty(
        (x.shape[0], qdata.shape[0]), dtype=torch.bfloat16, device=x.device
    )

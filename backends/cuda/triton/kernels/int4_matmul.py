# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Triton W4A16 matmul kernel for dense linear projections.

Replaces PyTorch's _weight_int4pack_mm (tinygemm) for prefill where large M
makes tinygemm's 16×8 tiles inefficient. Uses 128×128+ tiles for better
tensor core utilization.

Weight format: same as fused_moe experts:
  w_packed: [N, K//2] int8 — two INT4 values packed per byte
  w_scale:  [N, K//group_size] bf16 — symmetric dequant: (uint4 - 8) * scale

Registered as triton_op for AOTInductor export.
"""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


# -- Autotune configs ---------------------------------------------------------

_INT4_MATMUL_CONFIGS = [
    # Large-M prefill configs (tensor core saturated)
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128},
        num_warps=8,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64},
        num_warps=8,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128},
        num_warps=4,
        num_stages=4,
    ),
    # Small-M decode configs (bandwidth-bound, wide N tiles)
    triton.Config(
        {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128},
        num_warps=4,
        num_stages=5,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128},
        num_warps=4,
        num_stages=5,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64},
        num_warps=4,
        num_stages=5,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64},
        num_warps=4,
        num_stages=4,
    ),
]


# -- Triton kernel ------------------------------------------------------------


@triton.autotune(configs=_INT4_MATMUL_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _int4_matmul_kernel(
    # Pointers
    A,  # [M, K] bf16 activations
    B,  # [N, K//2] int8 packed INT4 weights
    C,  # [M, N] bf16 output
    B_scale,  # [N, K//group_size] bf16 per-group scales
    # Dimensions
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    # Strides
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_bsn,
    stride_bsk,
    # Config
    group_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """W4A16 matmul: C[M,N] = A[M,K] × dequant(B[N,K//2]).T

    Each program computes one (BLOCK_M, BLOCK_N) output tile.
    INT4 weights are unpacked and dequantized per-group inside the K-loop.
    """
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_SIZE_N)
    m_block = pid // num_n_blocks
    n_block = pid % num_n_blocks

    # M and N offsets for this block
    offs_m = m_block * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = n_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    m_mask = offs_m < M
    n_mask = offs_n < N

    # A pointers: [BLOCK_M, BLOCK_K] — rows of activations
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak

    # B pointers: [BLOCK_K, BLOCK_N] — weight is [N, K//2], need transposed access
    # B is stored as [N, K//2] with two INT4 per byte along K dim.
    # For the dot product A[M,K] @ B_dequant[K,N], we need B transposed:
    #   b_ptrs indexes as B[offs_n, offs_k//2], then we read [BLOCK_K//2, BLOCK_N]
    #   and reshape to [BLOCK_K, BLOCK_N] after unpacking.
    b_ptrs = B + offs_n[None, :] * stride_bn + (offs_k[:, None] // 2) * stride_bk
    b_shifter = (offs_k[:, None] % 2) * 4

    # Accumulator [BLOCK_M, BLOCK_N] in float32
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_step in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k_step * BLOCK_SIZE_K
        k_mask = offs_k < k_remaining

        # Load A tile [BLOCK_M, BLOCK_K]
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Load B tile [BLOCK_K, BLOCK_N] and unpack INT4
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0)
        b = (b >> b_shifter) & 0xF

        # Per-group scale dequantization
        if BLOCK_SIZE_K <= group_size:
            # One scale per column per tile — broadcast
            group_idx = (BLOCK_SIZE_K * k_step) // group_size
            scale_ptrs = B_scale + offs_n[None, :] * stride_bsn + group_idx * stride_bsk
            b_scale = tl.load(scale_ptrs, mask=n_mask[None, :], other=0.0).to(
                tl.float32
            )
        else:
            # Multiple groups per tile — per-element scale
            scale_ptrs = (
                B_scale
                + offs_n[None, :] * stride_bsn
                + ((offs_k[:, None] + BLOCK_SIZE_K * k_step) // group_size) * stride_bsk
            )
            b_scale = tl.load(
                scale_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0
            ).to(tl.float32)

        # Dequantize: (uint4 - 8) * scale → bf16
        b_dequant = ((b.to(tl.float32) - 8.0) * b_scale).to(tl.bfloat16)

        # Tensor core matmul: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] → f32 acc
        acc += tl.dot(a.to(tl.bfloat16), b_dequant)

        # Advance K pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk

    # Write output [BLOCK_M, BLOCK_N]
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


# -- triton_op wrapper --------------------------------------------------------


@triton_op("triton::int4_matmul", mutates_args={})
def int4_matmul(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    w_scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """W4A16 matmul: output = x @ dequant(w_packed).T

    Args:
        x:        [M, K] bf16 activations
        w_packed: [N, K//2] int8 — two INT4 values packed per byte
        w_scale:  [N, K//group_size] bf16 per-group scales
        group_size: quantization group size

    Returns:
        [M, N] bf16
    """
    M, K = x.shape
    N = w_packed.shape[0]

    assert x.dtype == torch.bfloat16
    assert w_packed.dtype == torch.int8
    assert w_scale.dtype == torch.bfloat16
    assert w_packed.shape == (
        N,
        K // 2,
    ), f"w_packed shape {w_packed.shape} != ({N}, {K // 2})"
    assert w_scale.shape == (
        N,
        K // group_size,
    ), f"w_scale shape {w_scale.shape} != ({N}, {K // group_size})"

    output = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)

    def _grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
        )

    wrap_triton(_int4_matmul_kernel)[_grid](
        x,
        w_packed,
        output,
        w_scale,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w_packed.stride(0),
        w_packed.stride(1),
        output.stride(0),
        output.stride(1),
        w_scale.stride(0),
        w_scale.stride(1),
        group_size,
    )
    return output


@int4_matmul.register_fake
def _int4_matmul_fake(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    w_scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    M, K = x.shape
    N = w_packed.shape[0]
    return torch.empty(M, N, dtype=torch.bfloat16, device=x.device)


# -- Dequant W4 → BF16 kernel ------------------------------------------------


# -- INT4 matvec kernel (M=1 decode) ------------------------------------------

_MATVEC_CONFIGS = [
    triton.Config({"BLOCK_N": 4, "BLOCK_K": 128}, num_warps=2, num_stages=3),
    triton.Config({"BLOCK_N": 8, "BLOCK_K": 128}, num_warps=2, num_stages=3),
    triton.Config({"BLOCK_N": 8, "BLOCK_K": 256}, num_warps=2, num_stages=3),
    triton.Config({"BLOCK_N": 4, "BLOCK_K": 256}, num_warps=2, num_stages=3),
]


@triton.autotune(configs=_MATVEC_CONFIGS, key=["N", "K"])
@triton.jit
def _int4_matvec_kernel(
    X,  # [K] bf16 input vector
    W,  # [N, K//2] int8 packed INT4 weights
    Out,  # [N] bf16 output
    W_scale,  # [N, K//group_size] bf16 per-group scales
    N: tl.constexpr,
    K: tl.constexpr,
    stride_wn,
    stride_wk,
    stride_sn,
    stride_sk,
    group_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    nb = tl.program_id(0)
    offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    nm = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for ks in range(tl.cdiv(K, BLOCK_K)):
        abs_k = ks * BLOCK_K + offs_k
        km = abs_k < K

        x_val = tl.load(X + abs_k, mask=km, other=0.0).to(tl.float32)

        w_ptrs = W + offs_n[:, None] * stride_wn + (abs_k[None, :] // 2) * stride_wk
        w_shift = (abs_k[None, :] % 2) * 4
        w_raw = tl.load(w_ptrs, mask=nm[:, None] & km[None, :], other=0)
        w_uint4 = (w_raw >> w_shift) & 0xF

        if BLOCK_K <= group_size:
            gi = (ks * BLOCK_K) // group_size
            scale = tl.load(
                W_scale + offs_n * stride_sn + gi * stride_sk, mask=nm, other=0.0
            ).to(tl.float32)
            w_dq = (w_uint4.to(tl.float32) - 8.0) * scale[:, None]
        else:
            scale_ptrs = (
                W_scale
                + offs_n[:, None] * stride_sn
                + (abs_k[None, :] // group_size) * stride_sk
            )
            scale = tl.load(scale_ptrs, mask=nm[:, None] & km[None, :], other=0.0).to(
                tl.float32
            )
            w_dq = (w_uint4.to(tl.float32) - 8.0) * scale

        acc += tl.sum(w_dq * x_val[None, :], axis=1)

    tl.store(Out + offs_n, acc.to(tl.bfloat16), mask=nm)


@triton_op("triton::int4_matvec", mutates_args={})
def int4_matvec(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    w_scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """W4A16 matvec for M=1 decode: out[1,N] = x[1,K] @ dequant(w[N,K]).T

    Args:
        x:        [1, K] bf16 input (M=1)
        w_packed: [N, K//2] int8 packed INT4 weights
        w_scale:  [N, K//group_size] bf16 per-group scales
        group_size: quantization group size

    Returns:
        [1, N] bf16
    """
    assert x.ndim == 2 and x.shape[0] == 1, f"int4_matvec requires [1, K] input, got {x.shape}"
    assert x.dtype == torch.bfloat16
    assert w_packed.dtype == torch.int8
    assert w_scale.dtype == torch.bfloat16
    K = x.shape[-1]
    N = w_packed.shape[0]
    assert w_packed.shape == (N, K // 2), f"w_packed shape {w_packed.shape} != ({N}, {K // 2})"
    assert w_scale.shape == (N, K // group_size), f"w_scale shape {w_scale.shape} != ({N}, {K // group_size})"
    assert K % 32 == 0, f"K={K} must be a multiple of 32 for vectorized loads"

    output = torch.empty(1, N, dtype=torch.bfloat16, device=x.device)

    def _grid(meta):
        return (triton.cdiv(N, meta["BLOCK_N"]),)

    wrap_triton(_int4_matvec_kernel)[_grid](
        x.reshape(-1),
        w_packed,
        output.reshape(-1),
        w_scale,
        N,
        K,
        w_packed.stride(0),
        w_packed.stride(1),
        w_scale.stride(0),
        w_scale.stride(1),
        group_size,
    )
    return output


@int4_matvec.register_fake
def _int4_matvec_fake(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    w_scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    N = w_packed.shape[0]
    return torch.empty(1, N, dtype=torch.bfloat16, device=x.device)


# -- Dequant W4 → BF16 kernel ------------------------------------------------

_DEQUANT_CONFIGS = [
    triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_N": 64, "BLOCK_K": 256}, num_warps=4, num_stages=3),
]


@triton.autotune(configs=_DEQUANT_CONFIGS, key=["N", "K"])
@triton.jit
def _dequant_w4_to_bf16_kernel(
    # Pointers
    W_packed,  # [N, K//2] int8 packed INT4 weights
    W_scale,  # [N, K//group_size] bf16 per-group scales
    Out,  # [N, K] bf16 output
    # Dimensions
    N: tl.constexpr,
    K: tl.constexpr,
    # Strides
    stride_wn,
    stride_wk,
    stride_sn,
    stride_sk,
    stride_on,
    stride_ok,
    # Config
    group_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Dequantize packed INT4 weights to BF16.

    Each program processes a (BLOCK_N, BLOCK_K) tile of the output [N, K].
    INT4 pairs are unpacked from [N, K//2] int8 and dequantized per-group.
    """
    pid = tl.program_id(0)
    num_k_blocks = tl.cdiv(K, BLOCK_K)
    n_block = pid // num_k_blocks
    k_block = pid % num_k_blocks

    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    k_mask = offs_k < K

    # Load packed bytes [BLOCK_N, BLOCK_K//2] — each byte has two int4 values
    packed_ptrs = (
        W_packed + offs_n[:, None] * stride_wn + (offs_k[None, :] // 2) * stride_wk
    )
    packed = tl.load(packed_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0)

    # Unpack: even k → low nibble, odd k → high nibble
    shift = (offs_k[None, :] % 2) * 4
    uint4 = (packed >> shift) & 0xF

    # Load per-group scales [BLOCK_N, ceil(BLOCK_K/group_size)]
    scale_ptrs = (
        W_scale
        + offs_n[:, None] * stride_sn
        + (offs_k[None, :] // group_size) * stride_sk
    )
    scale = tl.load(scale_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)

    # Dequantize: (uint4 - 8) * scale → bf16
    result = ((uint4.to(tl.float32) - 8.0) * scale.to(tl.float32)).to(tl.bfloat16)

    # Store [BLOCK_N, BLOCK_K]
    out_ptrs = Out + offs_n[:, None] * stride_on + offs_k[None, :] * stride_ok
    tl.store(out_ptrs, result, mask=n_mask[:, None] & k_mask[None, :])


# -- triton_op wrapper --------------------------------------------------------


@triton_op("triton::dequant_w4_to_bf16", mutates_args={})
def dequant_w4_to_bf16(
    w_packed: torch.Tensor,
    w_scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Dequantize packed INT4 weights to BF16.

    Args:
        w_packed: [N, K//2] int8 — two INT4 values packed per byte
        w_scale:  [N, K//group_size] bf16 per-group scales
        group_size: quantization group size

    Returns:
        [N, K] bf16 dequantized weight matrix
    """
    assert w_packed.ndim == 2, f"w_packed must be 2D, got {w_packed.ndim}D"
    assert w_packed.dtype == torch.int8
    assert w_scale.dtype == torch.bfloat16
    N, K_half = w_packed.shape
    K = K_half * 2
    assert w_scale.shape == (N, K // group_size), f"w_scale shape {w_scale.shape} != ({N}, {K // group_size})"

    output = torch.empty(N, K, dtype=torch.bfloat16, device=w_packed.device)

    def _grid(meta):
        return (triton.cdiv(N, meta["BLOCK_N"]) * triton.cdiv(K, meta["BLOCK_K"]),)

    wrap_triton(_dequant_w4_to_bf16_kernel)[_grid](
        w_packed,
        w_scale,
        output,
        N,
        K,
        w_packed.stride(0),
        w_packed.stride(1),
        w_scale.stride(0),
        w_scale.stride(1),
        output.stride(0),
        output.stride(1),
        group_size,
    )
    return output


@dequant_w4_to_bf16.register_fake
def _dequant_w4_to_bf16_fake(
    w_packed: torch.Tensor,
    w_scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    N, K_half = w_packed.shape
    K = K_half * 2
    return torch.empty(N, K, dtype=torch.bfloat16, device=w_packed.device)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# The INT4 dequantization logic is adapted from vLLM's fused_moe_kernel_gptq_awq:
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py
# Copyright contributors to the vLLM project. Licensed under Apache-2.0.
#
# Shared with vLLM: weight layout [E, N, K//2] packed INT4, scale layout
# [E, N, K//group_size], INT4 unpack via (b >> (k%2)*4) & 0xF, symmetric
# dequant (uint4 - 8) * scale, K-loop pointer advancement (BLOCK_SIZE_K // 2).
#
# Differences from vLLM: no token sorting (moe_align_block_size) — each
# program handles one (token-expert pair, N-block) directly via pair_idx.
# Uses vector-matrix multiply instead of tl.dot (no tensor cores). This is
# optimal for decode (M=1, memory-bandwidth-bound) but suboptimal for
# prefill (M >> 1, compute-bound). vLLM also supports INT8, asymmetric
# quantization, expert parallelism, and SPLIT_K — omitted here.

"""
Fused MoE Triton Kernel for ExecuTorch CUDA Backend.

Performs grouped GEMM for Mixture-of-Experts with INT4 weight-only
quantization (W4A16). Each Triton program handles one (token-expert pair,
N-block) — only active experts' weights are loaded from HBM.

Weight layout:
  B:       [E, N, K//2] int8 — two INT4 values packed per byte along K
  B_scale: [E, N, K//group_size] bf16 — per-group scales
  Dequant: weight = (uint4 - 8) * scale  (symmetric, no zero-point tensor)
"""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


# Autotune configs for GEMM1 (_fused_moe_kernel).
# Top performers from CI benchmark on A100-SXM4-80GB, Qwen3.5 MoE dimensions
# (M=1, N=1024, K=2048, 8 experts, group_size=128).
_GEMM1_CONFIGS = [
    triton.Config({"BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_N": 8, "BLOCK_SIZE_K": 256}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE_N": 8, "BLOCK_SIZE_K": 256}, num_warps=2, num_stages=4),
    triton.Config({"BLOCK_SIZE_N": 8, "BLOCK_SIZE_K": 256}, num_warps=2, num_stages=5),
    triton.Config({"BLOCK_SIZE_N": 8, "BLOCK_SIZE_K": 256}, num_warps=2, num_stages=3),
    triton.Config({"BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 256}, num_warps=2, num_stages=5),
]

# Autotune configs for GEMM2 (_fused_moe_silu_kernel).
# Top performers from CI benchmark on A100-SXM4-80GB, Qwen3.5 MoE dimensions
# (M=1, N=2048, K=512, 8 experts, group_size=128).
_GEMM2_CONFIGS = [
    triton.Config({"BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_N": 8, "BLOCK_SIZE_K": 128}, num_warps=2, num_stages=4),
    triton.Config({"BLOCK_SIZE_N": 8, "BLOCK_SIZE_K": 256}, num_warps=2, num_stages=4),
    triton.Config({"BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 256}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_SIZE_N": 8, "BLOCK_SIZE_K": 256}, num_warps=2, num_stages=3),
    triton.Config({"BLOCK_SIZE_N": 8, "BLOCK_SIZE_K": 256}, num_warps=4, num_stages=3),
]


@triton.autotune(configs=_GEMM1_CONFIGS, key=["N", "K"])
@triton.jit
def _fused_moe_kernel(
    # Pointers
    A,  # [M, K] bf16 activations
    B,  # [E, N, K//2] int8 packed INT4 weights
    C,  # [M * top_k, N] bf16 output
    B_scale,  # [E, N, K//group_size] bf16 scales
    topk_ids,  # [M * top_k] int64 expert indices
    topk_weights,  # [M * top_k] float32 router weights
    # Dimensions
    N: tl.constexpr,
    K: tl.constexpr,
    num_token_expert_pairs,
    # Strides
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsk,
    stride_bsn,
    # Config
    group_size: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    """One program per (token-expert pair, N-block).

    Grid = num_pairs * cdiv(N, BLOCK_SIZE_N). Only active experts'
    weights are loaded — optimal for decode where M=1.
    """
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_SIZE_N)
    pair_idx = pid // num_n_blocks
    n_block = pid % num_n_blocks

    if pair_idx >= num_token_expert_pairs:
        return

    # Which token and expert
    expert_id = tl.load(topk_ids + pair_idx).to(tl.int64)
    token_idx = pair_idx // top_k

    # Output column offsets
    offs_n = n_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    n_mask = offs_n < N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # A pointer: [token_idx, :]
    a_ptrs = A + token_idx * stride_am + offs_k * stride_ak

    # B pointer: [expert_id, offs_n, offs_k//2] — INT4 packed
    b_ptrs = (
        B
        + expert_id * stride_be
        + (offs_k[:, None] // 2) * stride_bk
        + offs_n[None, :] * stride_bn
    )
    b_shifter = (offs_k[:, None] % 2) * 4

    # Accumulate in fp32
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    for k_step in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k_step * BLOCK_SIZE_K
        k_mask = offs_k < k_remaining

        # Load A tile [BLOCK_SIZE_K]
        a = tl.load(a_ptrs, mask=k_mask, other=0.0)

        # Load B tile [BLOCK_SIZE_K, BLOCK_SIZE_N] and unpack INT4
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0)
        b = (b >> b_shifter) & 0xF

        # Load per-group scales [BLOCK_SIZE_K, BLOCK_SIZE_N]
        scale_ptrs = (
            B_scale
            + expert_id * stride_bse
            + offs_n[None, :] * stride_bsn
            + ((offs_k[:, None] + BLOCK_SIZE_K * k_step) // group_size) * stride_bsk
        )
        b_scale = tl.load(
            scale_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0
        ).to(tl.float32)

        # Dequantize and accumulate: vector-matrix multiply
        b_dequant = ((b.to(tl.float32) - 8.0) * b_scale).to(compute_type)
        acc += tl.sum(a[:, None].to(compute_type) * b_dequant, axis=0)

        # Advance K pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk

    # Multiply by router weight
    if MUL_ROUTED_WEIGHT:
        weight = tl.load(topk_weights + pair_idx)
        acc = acc * weight

    # Write output [BLOCK_SIZE_N]
    c_ptrs = C + pair_idx * stride_cm + offs_n * stride_cn
    tl.store(c_ptrs, acc.to(compute_type), mask=n_mask)


@triton.autotune(configs=_GEMM2_CONFIGS, key=["N", "K"])
@triton.jit
def _fused_moe_silu_kernel(
    # Pointers
    A,  # [M * top_k, 2*inter] bf16 GEMM1 output (gate | up)
    B,  # [E, N, K//2] int8 packed INT4 weights
    C,  # [M * top_k, N] bf16 output
    B_scale,  # [E, N, K//group_size] bf16 scales
    topk_ids,  # [M * top_k] int64 expert indices
    topk_weights,  # [M * top_k] float32 router weights
    # Dimensions
    N: tl.constexpr,
    K: tl.constexpr,  # intermediate_size
    num_token_expert_pairs,
    # Strides
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsk,
    stride_bsn,
    # Config
    group_size: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    compute_type: tl.constexpr,
):
    """GEMM2 with fused SiLU activation.

    Reads gate and up columns from GEMM1 output (A), applies SiLU(gate)*up
    on-the-fly, and multiplies by INT4 w2 weights. Router weights are applied
    to the output. Eliminates the intermediate activation buffer.
    """
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_SIZE_N)
    pair_idx = pid // num_n_blocks
    n_block = pid % num_n_blocks

    if pair_idx >= num_token_expert_pairs:
        return

    expert_id = tl.load(topk_ids + pair_idx).to(tl.int64)

    offs_n = n_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    n_mask = offs_n < N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # A pointers: gate at columns [0, K), up at columns [K, 2*K)
    a_gate_ptrs = A + pair_idx * stride_am + offs_k * stride_ak
    a_up_ptrs = a_gate_ptrs + K * stride_ak

    # B pointer: [expert_id, offs_n, offs_k//2]
    b_ptrs = (
        B
        + expert_id * stride_be
        + (offs_k[:, None] // 2) * stride_bk
        + offs_n[None, :] * stride_bn
    )
    b_shifter = (offs_k[:, None] % 2) * 4

    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    for k_step in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k_step * BLOCK_SIZE_K
        k_mask = offs_k < k_remaining

        # Load gate and up, apply SiLU(gate) * up
        gate = tl.load(a_gate_ptrs, mask=k_mask, other=0.0).to(tl.float32)
        up = tl.load(a_up_ptrs, mask=k_mask, other=0.0)
        a = (gate * tl.sigmoid(gate) * up).to(compute_type)

        # Load and dequantize INT4 weights
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0)
        b = (b >> b_shifter) & 0xF

        scale_ptrs = (
            B_scale
            + expert_id * stride_bse
            + offs_n[None, :] * stride_bsn
            + ((offs_k[:, None] + BLOCK_SIZE_K * k_step) // group_size) * stride_bsk
        )
        b_scale = tl.load(
            scale_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0
        ).to(tl.float32)

        b_dequant = ((b.to(tl.float32) - 8.0) * b_scale).to(compute_type)
        acc += tl.sum(a[:, None].to(compute_type) * b_dequant, axis=0)

        a_gate_ptrs += BLOCK_SIZE_K * stride_ak
        a_up_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk

    # Multiply by router weight
    weight = tl.load(topk_weights + pair_idx)
    acc = acc * weight

    c_ptrs = C + pair_idx * stride_cm + offs_n * stride_cn
    tl.store(c_ptrs, acc.to(compute_type), mask=n_mask)


# ---------------------------------------------------------------------------
# triton_op wrapper
# ---------------------------------------------------------------------------


@triton_op("triton::fused_moe", mutates_args={})
def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    num_experts: int,
    group_size: int,
) -> torch.Tensor:
    """Fused MoE with INT4 quantized expert weights.

    Performs: GEMM1 (gate+up) -> SiLU -> GEMM2 (down) -> weighted sum.
    Only loads active experts' weights from HBM.

    Args:
        hidden_states: [M, K] bf16 input activations
        w1: [E, 2*inter, K//2] int8 packed INT4 gate+up weights
        w1_scale: [E, 2*inter, K//group_size] bf16 scales
        w2: [E, K, inter//2] int8 packed INT4 down weights
        w2_scale: [E, K, inter//group_size] bf16 scales
        topk_weights: [M, top_k] float32 router weights (softmax)
        topk_ids: [M, top_k] int64 expert indices
        top_k: number of experts per token
        num_experts: total number of experts
        group_size: quantization group size

    Returns:
        [M, K] bf16 output
    """
    M, K = hidden_states.shape
    N1 = w1.shape[1]  # 2 * intermediate_size
    intermediate = N1 // 2
    N2 = w2.shape[1]  # hidden_size
    num_pairs = M * top_k

    # Flatten topk tensors
    topk_ids_flat = topk_ids.reshape(-1)
    topk_weights_flat = topk_weights.reshape(-1)

    # ---- GEMM1: gate + up projection ----
    # Grid is a lambda because BLOCK_SIZE_N is selected by autotune
    cache1 = torch.empty(
        num_pairs, N1, dtype=hidden_states.dtype, device=hidden_states.device
    )

    def grid1(meta):
        return (num_pairs * triton.cdiv(N1, meta["BLOCK_SIZE_N"]),)

    wrap_triton(_fused_moe_kernel)[grid1](
        hidden_states,
        w1,
        cache1,
        w1_scale,
        topk_ids_flat,
        topk_weights_flat,
        N=N1,
        K=K,
        num_token_expert_pairs=num_pairs,
        stride_am=hidden_states.stride(0),
        stride_ak=hidden_states.stride(1),
        stride_be=w1.stride(0),
        stride_bk=w1.stride(2),
        stride_bn=w1.stride(1),
        stride_cm=cache1.stride(0),
        stride_cn=cache1.stride(1),
        stride_bse=w1_scale.stride(0),
        stride_bsk=w1_scale.stride(2),
        stride_bsn=w1_scale.stride(1),
        group_size=group_size,
        MUL_ROUTED_WEIGHT=False,
        top_k=top_k,
        compute_type=tl.bfloat16,
    )

    # ---- GEMM2 with fused SiLU: reads gate+up from cache1, no intermediate buffer ----
    cache3 = torch.empty(
        num_pairs, N2, dtype=hidden_states.dtype, device=hidden_states.device
    )

    def grid2(meta):
        return (num_pairs * triton.cdiv(N2, meta["BLOCK_SIZE_N"]),)

    wrap_triton(_fused_moe_silu_kernel)[grid2](
        cache1,
        w2,
        cache3,
        w2_scale,
        topk_ids_flat,
        topk_weights_flat,
        N=N2,
        K=intermediate,
        num_token_expert_pairs=num_pairs,
        stride_am=cache1.stride(0),
        stride_ak=cache1.stride(1),
        stride_be=w2.stride(0),
        stride_bk=w2.stride(2),
        stride_bn=w2.stride(1),
        stride_cm=cache3.stride(0),
        stride_cn=cache3.stride(1),
        stride_bse=w2_scale.stride(0),
        stride_bsk=w2_scale.stride(2),
        stride_bsn=w2_scale.stride(1),
        group_size=group_size,
        compute_type=tl.bfloat16,
    )

    # ---- Sum across top-k experts ----
    return cache3.view(M, top_k, N2).sum(dim=1)


@fused_moe.register_fake
def _fused_moe_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    num_experts: int,
    group_size: int,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)

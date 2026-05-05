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
# Differences from vLLM: two kernel variants —
# 1. Decode (fused_moe): no token sorting, vector-matrix multiply (no tensor
#    cores). Each program handles one (token-expert pair, N-block). Optimal
#    for M=1, memory-bandwidth-bound.
# 2. Prefill (fused_moe_batched_gemm): token sorting via moe_align_block_size,
#    tl.dot tensor-core GEMMs. Optimal for M >> 1, compute-bound.
# vLLM also supports INT8, asymmetric quantization, expert parallelism,
# and SPLIT_K — omitted here.

"""
Fused MoE Triton Kernels for ExecuTorch CUDA Backend.

Performs grouped GEMM for Mixture-of-Experts with INT4 weight-only
quantization (W4A16) or INT4 weights + INT8 activations (W4A8).
Two kernel families (bf16 and int8), each with two variants:
  - fused_moe: vec-mat per-pair kernel for decode (M=1).
  - fused_moe_batched_gemm: token-sorted tensor-core kernel for prefill (M>>1).

Weight layout:
  B:       [E, N, K//2] int8 — two INT4 values packed per byte along K
  B_scale: [E, N, K//group_size] bf16 — per-group scales
  Dequant: weight = (uint4 - 8) * scale  (symmetric, no zero-point tensor)
"""

import functools

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@functools.lru_cache(maxsize=8)
def _num_sms(device_index: int) -> int:
    """Cache device SM count; queried once per device."""
    return torch.cuda.get_device_properties(device_index).multi_processor_count


# ---------------------------------------------------------------------------
# W4A8 batched MoE kernels (INT8 activations + INT4 weights).
#
# Activation INT8 quantization is HOISTED out of the GEMM K-loop into a
# dedicated pre-quantization kernel:
#   - _quantize_activations_int8_kernel writes [max_padded, K] INT8 +
#     [max_padded, num_k_tiles] float32 per-row-per-tile scales.
#   - _fused_moe_batched_int8_kernel (GEMM1) loads pre-quantized INT8 + scale.
#   - _silu_quantize_int8_kernel fuses SiLU(gate)*up with INT8 quantization
#     between GEMM1 and GEMM2.
#   - _fused_moe_silu_batched_int8_kernel (GEMM2) loads pre-quantized INT8.
#
# Hoisting eliminates ~256 redundant tl.max reductions per program
# (cdiv(K, BLOCK_SIZE_K) tiles * BLOCK_SIZE_M rows) and halves activation HBM
# bandwidth in the GEMM K-loop (bf16 -> int8).
#
# BLOCK_SIZE_K is fixed at PREQUANT_BLOCK_K (= 32, matches the llama.cpp
# group_size) so the per-tile activation scales line up with the GEMM K-loop.
# ---------------------------------------------------------------------------
PREQUANT_BLOCK_K = 32


@triton.jit
def _quantize_activations_int8_kernel(
    A,  # [M+1, K] bf16 input activations (with sentinel zero row)
    A_int8,  # [max_padded, K] int8 output (sorted order)
    A_scale,  # [max_padded, num_k_tiles] float32 per-row-per-tile scales
    sorted_token_ids,  # [max_padded] int64 pair indices
    K: tl.constexpr,
    NUM_K_TILES: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_qm,
    stride_qk,
    stride_sm,
    stride_sk,
):
    """Quantize one sorted M-row to INT8 with per-tile scales.

    Grid: (max_padded,) — one program per sorted row. Each program loops
    over K-tiles. Sentinel pair_ids map to the appended zero row in A.
    """
    row_id = tl.program_id(0)
    pair_id = tl.load(sorted_token_ids + row_id)
    token_id = pair_id // top_k

    offs_k = tl.arange(0, BLOCK_SIZE_K)

    for k_tile in range(NUM_K_TILES):
        k_offset = k_tile * BLOCK_SIZE_K
        k_full_offs = k_offset + offs_k
        k_mask = k_full_offs < K

        # Load bf16 activation slice [BLOCK_SIZE_K]
        a_ptrs = A + token_id * stride_am + k_full_offs * stride_ak
        a_bf16 = tl.load(a_ptrs, mask=k_mask, other=0.0)

        # Compute per-tile scale (scalar)
        a_f32 = a_bf16.to(tl.float32)
        a_absmax = tl.max(tl.abs(a_f32))
        a_scale_val = a_absmax / 127.0 + 1e-12

        # Quantize to INT8
        a_scaled = a_f32 / a_scale_val
        a_int8 = (a_scaled + tl.where(a_scaled >= 0, 0.5, -0.5)).to(tl.int8)

        # Store quantized activations
        q_ptrs = A_int8 + row_id * stride_qm + k_full_offs * stride_qk
        tl.store(q_ptrs, a_int8, mask=k_mask)

        # Store scale
        s_ptr = A_scale + row_id * stride_sm + k_tile * stride_sk
        tl.store(s_ptr, a_scale_val)


@triton.jit
def _silu_quantize_int8_kernel(
    A,  # [num_tokens_post_padded, 2*inter] bf16 GEMM1 output (sorted)
    A_int8,  # [num_tokens_post_padded, inter] int8 SiLU-quantized output
    A_scale,  # [num_tokens_post_padded, num_k_tiles] float32 per-tile scales
    K: tl.constexpr,  # intermediate_size
    NUM_K_TILES: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_qm,
    stride_qk,
    stride_sm,
    stride_sk,
):
    """SiLU(gate)*up + INT8 quantization for the batched GEMM2 input.

    Grid: (max_padded,). Reads gate at columns [0, K), up at [K, 2K),
    computes SiLU(gate)*up, quantizes to INT8 with per-tile scales.
    """
    row_id = tl.program_id(0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)

    for k_tile in range(NUM_K_TILES):
        k_offset = k_tile * BLOCK_SIZE_K
        k_full_offs = k_offset + offs_k
        k_mask = k_full_offs < K

        gate_ptrs = A + row_id * stride_am + k_full_offs * stride_ak
        up_ptrs = gate_ptrs + K * stride_ak

        gate = tl.load(gate_ptrs, mask=k_mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptrs, mask=k_mask, other=0.0).to(tl.float32)
        silu_out = gate * tl.sigmoid(gate) * up

        a_absmax = tl.max(tl.abs(silu_out))
        a_scale_val = a_absmax / 127.0 + 1e-12
        a_scaled = silu_out / a_scale_val
        a_int8 = (a_scaled + tl.where(a_scaled >= 0, 0.5, -0.5)).to(tl.int8)

        q_ptrs = A_int8 + row_id * stride_qm + k_full_offs * stride_qk
        tl.store(q_ptrs, a_int8, mask=k_mask)

        s_ptr = A_scale + row_id * stride_sm + k_tile * stride_sk
        tl.store(s_ptr, a_scale_val)


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

        # Load per-group scales and dequantize
        if BLOCK_SIZE_K <= group_size:
            # All K values in this tile share one scale group — load [1, N]
            group_idx = (BLOCK_SIZE_K * k_step) // group_size
            scale_ptrs = (
                B_scale
                + expert_id * stride_bse
                + offs_n[None, :] * stride_bsn
                + group_idx * stride_bsk
            )
            b_scale = tl.load(scale_ptrs, mask=n_mask[None, :], other=0.0).to(
                tl.float32
            )
        else:
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

        if BLOCK_SIZE_K <= group_size:
            group_idx = (BLOCK_SIZE_K * k_step) // group_size
            scale_ptrs = (
                B_scale
                + expert_id * stride_bse
                + offs_n[None, :] * stride_bsn
                + group_idx * stride_bsk
            )
            b_scale = tl.load(scale_ptrs, mask=n_mask[None, :], other=0.0).to(
                tl.float32
            )
        else:
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

    # Weight layout: [E, N, K//2].
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


# ---------------------------------------------------------------------------
# Batched prefill MoE — token sorting + tl.dot (tensor cores)
# ---------------------------------------------------------------------------

# Fixed BLOCK_M for the batched kernel. Not autotuned because the token
# sorting layout depends on it. Microbenchmarked on Qwen3.5 MoE prefill
# (M=1696, top_k=8, 256 experts) — BLOCK_M=64 is ~1.32x faster than 16
# despite the extra padding, because the per-expert M block (~30 tokens
# × 8 top_k = ~53 active rows/expert) saturates 64-row tensor-core MMAs
# and reduces total program count.
_BATCHED_BLOCK_M = 64


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Sort token-expert pairs by expert and pad to block_size boundaries.

    Given router output topk_ids [M, top_k], produces a flat array of pair
    indices grouped by expert with each expert's block padded to a multiple
    of block_size. Padding slots use sentinel value M*top_k which maps to
    a zero-row appended by the caller.

    All output shapes depend only on (M, top_k, num_experts, block_size) —
    no data-dependent shapes — so this is compatible with torch.export /
    symbolic tracing.

    Returns:
        sorted_token_ids: [max_num_tokens_padded] int64
        expert_ids: [max_num_expert_blocks] int64
        num_tokens_post_padded: scalar int64 tensor
    """
    M, top_k = topk_ids.shape
    num_pairs = M * top_k
    device = topk_ids.device
    sentinel = num_pairs  # out-of-bounds index -> zero padding row

    # Worst-case output size: every expert gets at least 1 token →
    # block_size padding each. With top_k routing, at most min(num_pairs,
    # num_experts) experts are active. Worst-case total slots:
    max_num_tokens_padded = num_pairs + num_experts * block_size
    max_num_expert_blocks = max_num_tokens_padded // block_size

    flat_ids = topk_ids.reshape(-1)  # [num_pairs] expert id per pair

    # Per-expert token counts via one_hot+sum (bincount lacks AOTI c-shim)
    tokens_per_expert = torch.nn.functional.one_hot(
        flat_ids, num_classes=num_experts
    ).sum(0)
    padded_per_expert = (
        (tokens_per_expert + block_size - 1) // block_size
    ) * block_size

    # Prefix sum for expert offsets in the output array
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    expert_offsets[1:] = padded_per_expert.cumsum(0)
    num_tokens_post_padded = expert_offsets[num_experts]  # scalar tensor

    # Pre-allocate at max size, filled with sentinel
    sorted_token_ids = torch.full(
        (max_num_tokens_padded,), sentinel, dtype=torch.int64, device=device
    )

    # Place each pair at its destination using counting sort.
    # For pair i with expert e = flat_ids[i], compute:
    #   dest = expert_offsets[e] + within_expert_rank[i]
    # within_expert_rank[i] = number of pairs j < i with flat_ids[j] == e.
    #
    # We compute this via exclusive prefix sum within each expert group:
    #   1) Create a key that sorts by (expert, pair_index):
    #      sort_key = flat_ids * num_pairs + arange(num_pairs)
    #   2) argsort gives indices sorted by expert then by original order.
    # To avoid argsort (needs sort_stable fallback in AOTI), we use a
    # scatter-based approach:
    #   For each pair i, within_expert_rank[i] = sum_{j<i} (flat_ids[j]==e)
    # This is a segmented exclusive prefix sum, computed as:
    #   within_expert_rank[i] = cumcount_of_expert[flat_ids[i]] before i
    # We iterate in pair order, incrementing per-expert counters.
    # Since this is a sequential scan, we implement it as a cumsum trick:
    #   For each expert e, the pairs assigned to it appear at positions
    #   where flat_ids == e. Their within-expert ranks are 0, 1, 2, ...
    #   in the order they appear.
    #
    # Vectorized: use argsort on (flat_ids * num_pairs + arange) which
    # gives stable expert-grouped ordering — same result as argsort(flat_ids).
    # The multiplication ensures expert grouping; adding arange breaks ties
    # by original order (equivalent to stable sort).
    sort_keys = flat_ids * num_pairs + torch.arange(
        num_pairs, device=device, dtype=torch.int64
    )
    sorted_order = sort_keys.argsort()  # no `stable` kwarg needed — keys are unique

    actual_offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    actual_offsets[1:] = tokens_per_expert.cumsum(0)

    sorted_experts = flat_ids[sorted_order]
    expert_starts_padded = expert_offsets[sorted_experts]
    expert_starts_actual = actual_offsets[sorted_experts]
    positions = torch.arange(num_pairs, device=device, dtype=torch.int64)
    within_expert_rank = positions - expert_starts_actual
    dest_positions = expert_starts_padded + within_expert_rank
    sorted_token_ids[dest_positions] = sorted_order

    # Build expert_ids: [max_num_expert_blocks].
    # For each block b, find which expert owns it via binary search on the
    # padded expert offsets. Blocks beyond num_tokens_post_padded are
    # all-sentinel (expert 0, harmless — kernel writes to sentinel row).
    block_starts = (
        torch.arange(max_num_expert_blocks, device=device, dtype=torch.int64)
        * block_size
    )
    # searchsorted on the full expert_offsets (not a slice — Inductor's
    # searchsorted lowering doesn't support SliceView inputs).
    # expert_offsets[0]=0, so searchsorted(..., right=True) returns 1 for
    # block_start=0 → subtract 1 to get expert 0.
    expert_ids = (
        torch.searchsorted(expert_offsets, block_starts, right=True) - 1
    ).clamp(min=0, max=num_experts - 1)

    return sorted_token_ids, expert_ids, num_tokens_post_padded


# Autotune configs for batched GEMM1 (gate+up projection).
# BLOCK_M is fixed at _BATCHED_BLOCK_M; only N and K are tuned.
_BATCHED_GEMM1_CONFIGS = [
    triton.Config({"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 16}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 16}, num_warps=4, num_stages=3),
    triton.Config(
        {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=2
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 16}, num_warps=4, num_stages=2
    ),
]

# Autotune configs for batched GEMM2 (down projection + SiLU).
_BATCHED_GEMM2_CONFIGS = [
    triton.Config({"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 16}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=2),
    triton.Config(
        {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=2
    ),
    triton.Config(
        {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 16}, num_warps=4, num_stages=2
    ),
]


@triton.autotune(configs=_BATCHED_GEMM1_CONFIGS, key=["N", "K"])
@triton.jit
def _fused_moe_batched_kernel(
    # Pointers
    A,  # [M+1, K] bf16 activations (row M is zero-padding sentinel)
    B,  # [E, N, K//2] int8 packed INT4 weights
    C,  # [num_tokens_post_padded, N] bf16 output (sorted order)
    B_scale,  # [E, N, K//group_size] bf16 scales
    sorted_token_ids,  # [num_tokens_post_padded] int64 pair indices
    expert_ids,  # [num_expert_blocks] int64
    # Dimensions
    N: tl.constexpr,
    K: tl.constexpr,
    num_expert_blocks,
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
    top_k: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    compute_type: tl.constexpr,
):
    """Batched GEMM1 (gate+up) with tensor cores — persistent + grouped tiles.

    Launches NUM_SMS programs and loops over the (expert_block, n_block)
    tile space. Uses Triton-style grouped (column-major within group) ordering
    so consecutive M-blocks of the same expert reuse B[expert, n_block, *]
    via L2.
    """
    start_pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_expert_blocks * num_n_blocks
    num_tiles_per_group = GROUP_SIZE_M * num_n_blocks

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
        group_id = tile_id // num_tiles_per_group
        first_eb = group_id * GROUP_SIZE_M
        group_size_m = tl.minimum(num_expert_blocks - first_eb, GROUP_SIZE_M)
        within = tile_id % num_tiles_per_group
        expert_block_idx = first_eb + (within % group_size_m)
        n_block = within // group_size_m

        expert_id = tl.load(expert_ids + expert_block_idx).to(tl.int64)

        offs_m = expert_block_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        pair_ids = tl.load(sorted_token_ids + offs_m)
        token_ids = pair_ids // top_k

        offs_n = n_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        n_mask = offs_n < N
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        a_ptrs = A + token_ids[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = (
            B
            + expert_id * stride_be
            + (offs_k[:, None] // 2) * stride_bk
            + offs_n[None, :] * stride_bn
        )
        b_shifter = (offs_k[:, None] % 2) * 4

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k_step in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            k_remaining = K - k_step * BLOCK_SIZE_K
            k_mask = offs_k < k_remaining

            a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)

            b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0)
            b = (b >> b_shifter) & 0xF

            if BLOCK_SIZE_K <= group_size:
                group_idx = (BLOCK_SIZE_K * k_step) // group_size
                scale_ptrs = (
                    B_scale
                    + expert_id * stride_bse
                    + offs_n[None, :] * stride_bsn
                    + group_idx * stride_bsk
                )
                b_scale = tl.load(scale_ptrs, mask=n_mask[None, :], other=0.0).to(
                    tl.float32
                )
            else:
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
            acc += tl.dot(a.to(compute_type), b_dequant)

            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk

        c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc.to(compute_type), mask=n_mask[None, :])


# Autotune configs for the prequant GEMM1 INT8 kernel.
# BLOCK_SIZE_K is FIXED at PREQUANT_BLOCK_K — only N/warps/stages/groups tunable.
_BATCHED_GEMM1_INT8_CONFIGS = [
    triton.Config({"BLOCK_SIZE_N": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_N": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_SIZE_N": 128, "GROUP_SIZE_M": 16}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_N": 128, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE_N": 128, "GROUP_SIZE_M": 16}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE_N": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_N": 64, "GROUP_SIZE_M": 16}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_N": 256, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE_N": 256, "GROUP_SIZE_M": 16}, num_warps=8, num_stages=2),
]


@triton.autotune(configs=_BATCHED_GEMM1_INT8_CONFIGS, key=["N", "K"])
@triton.jit
def _fused_moe_batched_int8_kernel(
    # Pointers — A is INT8 pre-quantized in sorted order, A_scale per-tile
    A_int8,  # [max_padded, K] int8 pre-quantized activations
    A_scale,  # [max_padded, num_k_tiles] float32 per-tile scales
    B,  # [E, N, K//2] int8 packed INT4 weights
    C,  # [num_tokens_post_padded, N] bf16 output (sorted order)
    B_scale,  # [E, N, K//group_size] bf16 scales
    expert_ids,  # [num_expert_blocks] int64
    # Dimensions
    N: tl.constexpr,
    K: tl.constexpr,
    num_expert_blocks,
    # Strides
    stride_qm,
    stride_qk,
    stride_sm,
    stride_sk,
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    compute_type: tl.constexpr,
):
    """Batched GEMM1 with INT8 tensor cores — persistent + grouped tiles.

    Consumes pre-quantized activations + per-row-per-tile scales.
    Persistent grid: NUM_SMS programs each loop over a slice of the
    (expert_block, n_block) tile space using grouped (column-major within
    group) ordering for L2 reuse of expert weights.
    """
    start_pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_expert_blocks * num_n_blocks
    num_tiles_per_group = GROUP_SIZE_M * num_n_blocks

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
        group_id = tile_id // num_tiles_per_group
        first_eb = group_id * GROUP_SIZE_M
        group_size_m = tl.minimum(num_expert_blocks - first_eb, GROUP_SIZE_M)
        within = tile_id % num_tiles_per_group
        expert_block_idx = first_eb + (within % group_size_m)
        n_block = within // group_size_m

        expert_id = tl.load(expert_ids + expert_block_idx).to(tl.int64)

        offs_m = expert_block_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

        offs_n = n_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        n_mask = offs_n < N
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        a_ptrs = A_int8 + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk

        b_ptrs = (
            B
            + expert_id * stride_be
            + (offs_k[:, None] // 2) * stride_bk
            + offs_n[None, :] * stride_bn
        )
        b_shifter = (offs_k[:, None] % 2) * 4

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k_step in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            k_remaining = K - k_step * BLOCK_SIZE_K
            k_mask = offs_k < k_remaining

            a_int8 = tl.load(a_ptrs, mask=k_mask[None, :], other=0)
            a_scale = tl.load(A_scale + offs_m * stride_sm + k_step * stride_sk)

            b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0)
            b = (b >> b_shifter) & 0xF
            b_int8 = (b - 8).to(tl.int8)

            if BLOCK_SIZE_K <= group_size:
                group_idx = (BLOCK_SIZE_K * k_step) // group_size
                scale_ptrs = (
                    B_scale
                    + expert_id * stride_bse
                    + offs_n[None, :] * stride_bsn
                    + group_idx * stride_bsk
                )
                b_scale = tl.load(scale_ptrs, mask=n_mask[None, :], other=0.0).to(
                    tl.float32
                )
                dot_i32 = tl.dot(a_int8, b_int8)
                acc += dot_i32.to(tl.float32) * a_scale[:, None] * b_scale
            else:
                scale_ptrs = (
                    B_scale
                    + expert_id * stride_bse
                    + offs_n[None, :] * stride_bsn
                    + ((offs_k[:, None] + BLOCK_SIZE_K * k_step) // group_size) * stride_bsk
                )
                b_scale = tl.load(
                    scale_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0
                ).to(tl.float32)
                b_dequant = (b_int8.to(tl.float32) * b_scale).to(compute_type)
                acc += (
                    tl.dot(a_int8.to(compute_type), b_dequant).to(tl.float32)
                    * a_scale[:, None]
                )

            a_ptrs += BLOCK_SIZE_K * stride_qk
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk

        c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc.to(compute_type), mask=n_mask[None, :])


@triton.autotune(configs=_BATCHED_GEMM2_CONFIGS, key=["N", "K"])
@triton.jit
def _fused_moe_silu_batched_kernel(
    # Pointers
    A,  # [num_tokens_post_padded, 2*inter] bf16 GEMM1 output (sorted order)
    B,  # [E, N, K//2] int8 packed INT4 weights
    C,  # [M*top_k + 1, N] bf16 output (scatter to original pair order)
    B_scale,  # [E, N, K//group_size] bf16 scales
    sorted_token_ids,  # [num_tokens_post_padded] int64 pair indices
    expert_ids,  # [num_expert_blocks] int64
    topk_weights,  # [M*top_k] float32 router weights (flat)
    # Dimensions
    N: tl.constexpr,
    K: tl.constexpr,  # intermediate_size
    num_pairs,  # M * top_k (for clamping sentinel weight lookups)
    num_expert_blocks,
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
    top_k: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    compute_type: tl.constexpr,
):
    """Batched GEMM2 with fused SiLU and scatter-back — persistent + grouped."""
    start_pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_expert_blocks * num_n_blocks
    num_tiles_per_group = GROUP_SIZE_M * num_n_blocks

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
        group_id = tile_id // num_tiles_per_group
        first_eb = group_id * GROUP_SIZE_M
        group_size_m = tl.minimum(num_expert_blocks - first_eb, GROUP_SIZE_M)
        within = tile_id % num_tiles_per_group
        expert_block_idx = first_eb + (within % group_size_m)
        n_block = within // group_size_m

        expert_id = tl.load(expert_ids + expert_block_idx).to(tl.int64)

        offs_m = expert_block_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        pair_ids = tl.load(sorted_token_ids + offs_m)

        offs_n = n_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        n_mask = offs_n < N
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        a_gate_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_up_ptrs = a_gate_ptrs + K * stride_ak

        b_ptrs = (
            B
            + expert_id * stride_be
            + (offs_k[:, None] // 2) * stride_bk
            + offs_n[None, :] * stride_bn
        )
        b_shifter = (offs_k[:, None] % 2) * 4

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k_step in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            k_remaining = K - k_step * BLOCK_SIZE_K
            k_mask = offs_k < k_remaining

            gate = tl.load(a_gate_ptrs, mask=k_mask[None, :], other=0.0).to(tl.float32)
            up = tl.load(a_up_ptrs, mask=k_mask[None, :], other=0.0)
            a = (gate * tl.sigmoid(gate) * up).to(compute_type)

            b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0)
            b = (b >> b_shifter) & 0xF

            if BLOCK_SIZE_K <= group_size:
                group_idx = (BLOCK_SIZE_K * k_step) // group_size
                scale_ptrs = (
                    B_scale
                    + expert_id * stride_bse
                    + offs_n[None, :] * stride_bsn
                    + group_idx * stride_bsk
                )
                b_scale = tl.load(scale_ptrs, mask=n_mask[None, :], other=0.0).to(
                    tl.float32
                )
            else:
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
            acc += tl.dot(a, b_dequant)

            a_gate_ptrs += BLOCK_SIZE_K * stride_ak
            a_up_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk

        safe_pair_ids = tl.minimum(pair_ids, num_pairs - 1)
        weights = tl.load(topk_weights + safe_pair_ids)
        is_valid = pair_ids < num_pairs
        weights = tl.where(is_valid, weights, 0.0)
        acc = acc * weights[:, None]

        scatter_ids = tl.where(is_valid, pair_ids, num_pairs)
        c_ptrs = C + scatter_ids[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc.to(compute_type), mask=n_mask[None, :])


_BATCHED_GEMM2_INT8_CONFIGS = [
    triton.Config({"BLOCK_SIZE_N": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_N": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_SIZE_N": 128, "GROUP_SIZE_M": 16}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_N": 128, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE_N": 128, "GROUP_SIZE_M": 16}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE_N": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_N": 64, "GROUP_SIZE_M": 16}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_N": 256, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE_N": 256, "GROUP_SIZE_M": 16}, num_warps=8, num_stages=2),
]


@triton.autotune(configs=_BATCHED_GEMM2_INT8_CONFIGS, key=["N", "K"])
@triton.jit
def _fused_moe_silu_batched_int8_kernel(
    A_int8,  # [max_padded, K] int8 pre-quantized SiLU output
    A_scale,  # [max_padded, num_k_tiles] float32 per-tile scales
    B,  # [E, N, K//2] int8 packed INT4 weights
    C,  # [M*top_k + 1, N] bf16 output (scatter to pair order)
    B_scale,  # [E, N, K//group_size] bf16 scales
    sorted_token_ids,  # [num_tokens_post_padded] int64 pair indices
    expert_ids,  # [num_expert_blocks] int64
    topk_weights,  # [M*top_k] float32 router weights
    # Dimensions
    N: tl.constexpr,
    K: tl.constexpr,
    num_pairs,
    num_expert_blocks,
    # Strides
    stride_qm,
    stride_qk,
    stride_sm,
    stride_sk,
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    compute_type: tl.constexpr,
):
    """GEMM2 with INT8 tensor cores, scatter-back — persistent + grouped tiles.

    Consumes pre-quantized SiLU(gate)*up activations + per-row-per-tile scales.
    """
    start_pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_expert_blocks * num_n_blocks
    num_tiles_per_group = GROUP_SIZE_M * num_n_blocks

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
        group_id = tile_id // num_tiles_per_group
        first_eb = group_id * GROUP_SIZE_M
        group_size_m = tl.minimum(num_expert_blocks - first_eb, GROUP_SIZE_M)
        within = tile_id % num_tiles_per_group
        expert_block_idx = first_eb + (within % group_size_m)
        n_block = within // group_size_m

        expert_id = tl.load(expert_ids + expert_block_idx).to(tl.int64)

        offs_m = expert_block_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        pair_ids = tl.load(sorted_token_ids + offs_m)

        offs_n = n_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        n_mask = offs_n < N
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        a_ptrs = A_int8 + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk

        b_ptrs = (
            B
            + expert_id * stride_be
            + (offs_k[:, None] // 2) * stride_bk
            + offs_n[None, :] * stride_bn
        )
        b_shifter = (offs_k[:, None] % 2) * 4

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k_step in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            k_remaining = K - k_step * BLOCK_SIZE_K
            k_mask = offs_k < k_remaining

            a_int8 = tl.load(a_ptrs, mask=k_mask[None, :], other=0)
            a_scale = tl.load(A_scale + offs_m * stride_sm + k_step * stride_sk)

            b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0)
            b = (b >> b_shifter) & 0xF
            b_int8 = (b - 8).to(tl.int8)

            if BLOCK_SIZE_K <= group_size:
                group_idx = (BLOCK_SIZE_K * k_step) // group_size
                scale_ptrs = (
                    B_scale
                    + expert_id * stride_bse
                    + offs_n[None, :] * stride_bsn
                    + group_idx * stride_bsk
                )
                b_scale = tl.load(scale_ptrs, mask=n_mask[None, :], other=0.0).to(
                    tl.float32
                )
                dot_i32 = tl.dot(a_int8, b_int8)
                acc += dot_i32.to(tl.float32) * a_scale[:, None] * b_scale
            else:
                scale_ptrs = (
                    B_scale
                    + expert_id * stride_bse
                    + offs_n[None, :] * stride_bsn
                    + ((offs_k[:, None] + BLOCK_SIZE_K * k_step) // group_size) * stride_bsk
                )
                b_scale = tl.load(
                    scale_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0
                ).to(tl.float32)
                b_dequant = (b_int8.to(tl.float32) * b_scale).to(compute_type)
                acc += (
                    tl.dot(a_int8.to(compute_type), b_dequant).to(tl.float32)
                    * a_scale[:, None]
                )

            a_ptrs += BLOCK_SIZE_K * stride_qk
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk

        safe_pair_ids = tl.minimum(pair_ids, num_pairs - 1)
        weights = tl.load(topk_weights + safe_pair_ids)
        is_valid = pair_ids < num_pairs
        weights = tl.where(is_valid, weights, 0.0)
        acc = acc * weights[:, None]

        scatter_ids = tl.where(is_valid, pair_ids, num_pairs)
        c_ptrs = C + scatter_ids[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc.to(compute_type), mask=n_mask[None, :])


# ---------------------------------------------------------------------------
# Batched triton_op wrapper
# ---------------------------------------------------------------------------


@triton_op("triton::fused_moe_batched_gemm", mutates_args={})
def fused_moe_batched_gemm(
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
    """Batched GEMM1 + GEMM2+SiLU with token sorting + tensor-core GEMMs."""
    M, K = hidden_states.shape
    N1 = w1.shape[1]  # 2 * intermediate_size
    intermediate = N1 // 2
    N2 = w2.shape[1]  # hidden_size
    num_pairs = M * top_k
    BLOCK_M = _BATCHED_BLOCK_M

    sorted_token_ids, expert_ids, _ = moe_align_block_size(
        topk_ids, BLOCK_M, num_experts
    )
    max_padded = sorted_token_ids.shape[0]
    num_expert_blocks = expert_ids.shape[0]

    hidden_padded = torch.cat(
        [
            hidden_states,
            torch.zeros(1, K, dtype=hidden_states.dtype, device=hidden_states.device),
        ],
        dim=0,
    )

    topk_weights_flat = topk_weights.reshape(-1)

    cache1 = torch.empty(
        max_padded,
        N1,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    NUM_SMS = _num_sms(hidden_states.device.index)

    def grid1(meta):
        n_blocks = triton.cdiv(N1, meta["BLOCK_SIZE_N"])
        total_tiles = num_expert_blocks * n_blocks
        return (min(NUM_SMS, total_tiles),)

    wrap_triton(_fused_moe_batched_kernel)[grid1](
        hidden_padded,
        w1,
        cache1,
        w1_scale,
        sorted_token_ids,
        expert_ids,
        N=N1,
        K=K,
        num_expert_blocks=num_expert_blocks,
        stride_am=hidden_padded.stride(0),
        stride_ak=hidden_padded.stride(1),
        stride_be=w1.stride(0),
        stride_bk=w1.stride(2),
        stride_bn=w1.stride(1),
        stride_cm=cache1.stride(0),
        stride_cn=cache1.stride(1),
        stride_bse=w1_scale.stride(0),
        stride_bsk=w1_scale.stride(2),
        stride_bsn=w1_scale.stride(1),
        top_k=top_k,
        group_size=group_size,
        BLOCK_SIZE_M=BLOCK_M,
        NUM_SMS=NUM_SMS,
        compute_type=tl.bfloat16,
    )

    out_buf = torch.zeros(
        num_pairs + 1,
        N2,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    def grid2(meta):
        n_blocks = triton.cdiv(N2, meta["BLOCK_SIZE_N"])
        total_tiles = num_expert_blocks * n_blocks
        return (min(NUM_SMS, total_tiles),)

    wrap_triton(_fused_moe_silu_batched_kernel)[grid2](
        cache1,
        w2,
        out_buf,
        w2_scale,
        sorted_token_ids,
        expert_ids,
        topk_weights_flat,
        N=N2,
        K=intermediate,
        num_pairs=num_pairs,
        num_expert_blocks=num_expert_blocks,
        stride_am=cache1.stride(0),
        stride_ak=cache1.stride(1),
        stride_be=w2.stride(0),
        stride_bk=w2.stride(2),
        stride_bn=w2.stride(1),
        stride_cm=out_buf.stride(0),
        stride_cn=out_buf.stride(1),
        stride_bse=w2_scale.stride(0),
        stride_bsk=w2_scale.stride(2),
        stride_bsn=w2_scale.stride(1),
        top_k=top_k,
        group_size=group_size,
        BLOCK_SIZE_M=BLOCK_M,
        NUM_SMS=NUM_SMS,
        compute_type=tl.bfloat16,
    )

    return out_buf[:num_pairs].view(M, top_k, N2).sum(dim=1)


@fused_moe_batched_gemm.register_fake
def _fused_moe_batched_gemm_fake(
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


@triton_op("triton::fused_moe_batched_gemm_int8", mutates_args={})
def fused_moe_batched_gemm_int8(
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
    """Batched W4A8 GEMM1 + GEMM2+SiLU with INT8 tensor cores.

    Pipeline:
      1. moe_align_block_size: sort pairs by expert.
      2. _quantize_activations_int8_kernel: quantize hidden_states to INT8
         in sorted order with per-row-per-tile scales.
      3. _fused_moe_batched_int8_kernel (GEMM1): consumes INT8 + scales.
      4. _silu_quantize_int8_kernel: fuse SiLU(gate)*up + INT8 quantization
         on the GEMM1 output.
      5. _fused_moe_silu_batched_int8_kernel (GEMM2): consumes INT8 + scales,
         scatter-back to original pair order.
    """
    M, K = hidden_states.shape
    N1 = w1.shape[1]
    intermediate = N1 // 2
    N2 = w2.shape[1]
    num_pairs = M * top_k
    BLOCK_M = _BATCHED_BLOCK_M

    sorted_token_ids, expert_ids, _ = moe_align_block_size(
        topk_ids, BLOCK_M, num_experts
    )
    max_padded = sorted_token_ids.shape[0]
    num_expert_blocks = expert_ids.shape[0]

    hidden_padded = torch.cat(
        [
            hidden_states,
            torch.zeros(1, K, dtype=hidden_states.dtype, device=hidden_states.device),
        ],
        dim=0,
    )

    topk_weights_flat = topk_weights.reshape(-1)

    # ---- Pre-quantize activations for GEMM1 ----
    BLOCK_K_QUANT = PREQUANT_BLOCK_K
    num_k_tiles_g1 = (K + BLOCK_K_QUANT - 1) // BLOCK_K_QUANT

    a_int8_g1 = torch.empty(
        max_padded, K, dtype=torch.int8, device=hidden_states.device
    )
    a_scale_g1 = torch.empty(
        max_padded, num_k_tiles_g1, dtype=torch.float32, device=hidden_states.device
    )

    grid_quant_g1 = (max_padded,)
    wrap_triton(_quantize_activations_int8_kernel)[grid_quant_g1](
        hidden_padded,
        a_int8_g1,
        a_scale_g1,
        sorted_token_ids,
        K=K,
        NUM_K_TILES=num_k_tiles_g1,
        top_k=top_k,
        BLOCK_SIZE_K=BLOCK_K_QUANT,
        stride_am=hidden_padded.stride(0),
        stride_ak=hidden_padded.stride(1),
        stride_qm=a_int8_g1.stride(0),
        stride_qk=a_int8_g1.stride(1),
        stride_sm=a_scale_g1.stride(0),
        stride_sk=a_scale_g1.stride(1),
    )

    cache1 = torch.empty(
        max_padded,
        N1,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    NUM_SMS = _num_sms(hidden_states.device.index)

    def grid1(meta):
        n_blocks = triton.cdiv(N1, meta["BLOCK_SIZE_N"])
        total_tiles = num_expert_blocks * n_blocks
        return (min(NUM_SMS, total_tiles),)

    wrap_triton(_fused_moe_batched_int8_kernel)[grid1](
        a_int8_g1,
        a_scale_g1,
        w1,
        cache1,
        w1_scale,
        expert_ids,
        N=N1,
        K=K,
        num_expert_blocks=num_expert_blocks,
        stride_qm=a_int8_g1.stride(0),
        stride_qk=a_int8_g1.stride(1),
        stride_sm=a_scale_g1.stride(0),
        stride_sk=a_scale_g1.stride(1),
        stride_be=w1.stride(0),
        stride_bk=w1.stride(2),
        stride_bn=w1.stride(1),
        stride_cm=cache1.stride(0),
        stride_cn=cache1.stride(1),
        stride_bse=w1_scale.stride(0),
        stride_bsk=w1_scale.stride(2),
        stride_bsn=w1_scale.stride(1),
        group_size=group_size,
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_K=BLOCK_K_QUANT,
        NUM_SMS=NUM_SMS,
        compute_type=tl.bfloat16,
    )

    # ---- SiLU + pre-quantize for GEMM2 ----
    num_k_tiles_g2 = (intermediate + BLOCK_K_QUANT - 1) // BLOCK_K_QUANT
    a_int8_g2 = torch.empty(
        max_padded, intermediate, dtype=torch.int8, device=hidden_states.device
    )
    a_scale_g2 = torch.empty(
        max_padded, num_k_tiles_g2, dtype=torch.float32, device=hidden_states.device
    )

    grid_silu = (max_padded,)
    wrap_triton(_silu_quantize_int8_kernel)[grid_silu](
        cache1,
        a_int8_g2,
        a_scale_g2,
        K=intermediate,
        NUM_K_TILES=num_k_tiles_g2,
        BLOCK_SIZE_K=BLOCK_K_QUANT,
        stride_am=cache1.stride(0),
        stride_ak=cache1.stride(1),
        stride_qm=a_int8_g2.stride(0),
        stride_qk=a_int8_g2.stride(1),
        stride_sm=a_scale_g2.stride(0),
        stride_sk=a_scale_g2.stride(1),
    )

    out_buf = torch.zeros(
        num_pairs + 1,
        N2,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    def grid2(meta):
        n_blocks = triton.cdiv(N2, meta["BLOCK_SIZE_N"])
        total_tiles = num_expert_blocks * n_blocks
        return (min(NUM_SMS, total_tiles),)

    wrap_triton(_fused_moe_silu_batched_int8_kernel)[grid2](
        a_int8_g2,
        a_scale_g2,
        w2,
        out_buf,
        w2_scale,
        sorted_token_ids,
        expert_ids,
        topk_weights_flat,
        N=N2,
        K=intermediate,
        num_pairs=num_pairs,
        num_expert_blocks=num_expert_blocks,
        stride_qm=a_int8_g2.stride(0),
        stride_qk=a_int8_g2.stride(1),
        stride_sm=a_scale_g2.stride(0),
        stride_sk=a_scale_g2.stride(1),
        stride_be=w2.stride(0),
        stride_bk=w2.stride(2),
        stride_bn=w2.stride(1),
        stride_cm=out_buf.stride(0),
        stride_cn=out_buf.stride(1),
        stride_bse=w2_scale.stride(0),
        stride_bsk=w2_scale.stride(2),
        stride_bsn=w2_scale.stride(1),
        group_size=group_size,
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_K=BLOCK_K_QUANT,
        NUM_SMS=NUM_SMS,
        compute_type=tl.bfloat16,
    )

    return out_buf[:num_pairs].view(M, top_k, N2).sum(dim=1)


@fused_moe_batched_gemm_int8.register_fake
def _fused_moe_batched_gemm_int8_fake(
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


def fused_moe_batched(
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
    activation_dtype: str = "bf16",
) -> torch.Tensor:
    """Convenience wrapper that dispatches to bf16 or int8 batched kernels."""
    if activation_dtype == "int8":
        return fused_moe_batched_gemm_int8(
            hidden_states,
            w1,
            w1_scale,
            w2,
            w2_scale,
            topk_weights,
            topk_ids,
            top_k,
            num_experts,
            group_size,
        )
    return fused_moe_batched_gemm(
        hidden_states,
        w1,
        w1_scale,
        w2,
        w2_scale,
        topk_weights,
        topk_ids,
        top_k,
        num_experts,
        group_size,
    )

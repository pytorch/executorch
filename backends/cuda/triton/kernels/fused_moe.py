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
quantization (W4A16). Two kernel variants:
  - fused_moe: vec-mat per-pair kernel for decode (M=1).
  - fused_moe_batched_gemm: token-sorted tensor-core kernel for prefill (M>>1).

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
# sorting layout depends on it. 16 is the minimum for tl.dot and wastes
# the least padding with typical Qwen3.5 expert load (~30 tokens/expert).
_BATCHED_BLOCK_M = 16


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
    triton.Config({"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=3),
    triton.Config(
        {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=2
    ),
]

# Autotune configs for batched GEMM2 (down projection + SiLU).
_BATCHED_GEMM2_CONFIGS = [
    triton.Config({"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=2),
    triton.Config(
        {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=2
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
    compute_type: tl.constexpr,
):
    """Batched GEMM1 (gate+up) with tensor cores.

    Each program handles one (expert-M-block, N-block). Tokens are
    gathered via sorted_token_ids for expert-grouped access.
    """
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_SIZE_N)
    expert_block_idx = pid // num_n_blocks
    n_block = pid % num_n_blocks

    expert_id = tl.load(expert_ids + expert_block_idx).to(tl.int64)

    # M-block: BLOCK_SIZE_M consecutive entries in sorted_token_ids
    offs_m = expert_block_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # Load pair indices; sentinel values map to the zero-padding row
    pair_ids = tl.load(sorted_token_ids + offs_m)
    token_ids = pair_ids // top_k  # map pair -> token for activation lookup

    # N offsets
    offs_n = n_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    n_mask = offs_n < N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # A pointers: gathered rows [BLOCK_M, K]
    a_ptrs = A + token_ids[:, None] * stride_am + offs_k[None, :] * stride_ak

    # B pointers: [expert_id, offs_n, offs_k//2]
    b_ptrs = (
        B
        + expert_id * stride_be
        + (offs_k[:, None] // 2) * stride_bk
        + offs_n[None, :] * stride_bn
    )
    b_shifter = (offs_k[:, None] % 2) * 4

    # 2D accumulator [BLOCK_M, BLOCK_N]
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_step in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k_step * BLOCK_SIZE_K
        k_mask = offs_k < k_remaining

        # Load A tile [BLOCK_M, BLOCK_K] — gathered via token_ids
        a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)

        # Load B tile [BLOCK_K, BLOCK_N] and unpack INT4
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0)
        b = (b >> b_shifter) & 0xF

        # Per-group scales
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

        # Dequantize: (uint4 - 8) * scale
        b_dequant = ((b.to(tl.float32) - 8.0) * b_scale).to(compute_type)

        # Tensor core matmul: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N]
        acc += tl.dot(a.to(compute_type), b_dequant)

        # Advance K pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk

    # Write output in sorted order [BLOCK_M, BLOCK_N]
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
    compute_type: tl.constexpr,
):
    """Batched GEMM2 with fused SiLU and scatter-back.

    Reads gate+up from GEMM1 output (sorted order), applies SiLU(gate)*up,
    multiplies by INT4 w2 weights, applies router weights, and scatters
    output to original pair positions.
    """
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_SIZE_N)
    expert_block_idx = pid // num_n_blocks
    n_block = pid % num_n_blocks

    expert_id = tl.load(expert_ids + expert_block_idx).to(tl.int64)

    # M-block in sorted order
    offs_m = expert_block_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    pair_ids = tl.load(sorted_token_ids + offs_m)

    # N offsets
    offs_n = n_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    n_mask = offs_n < N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # A pointers: gate at [0, K), up at [K, 2K) — contiguous in sorted order
    a_gate_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    a_up_ptrs = a_gate_ptrs + K * stride_ak

    # B pointers: [expert_id, offs_n, offs_k//2]
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

        # Load gate and up tiles [BLOCK_M, BLOCK_K], apply SiLU
        gate = tl.load(a_gate_ptrs, mask=k_mask[None, :], other=0.0).to(tl.float32)
        up = tl.load(a_up_ptrs, mask=k_mask[None, :], other=0.0)
        a = (gate * tl.sigmoid(gate) * up).to(compute_type)

        # Load and dequantize INT4 weights [BLOCK_K, BLOCK_N]
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

        # Tensor core matmul: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N]
        acc += tl.dot(a, b_dequant)

        a_gate_ptrs += BLOCK_SIZE_K * stride_ak
        a_up_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk

    # Apply router weights per row
    # Clamp sentinel pair_ids to a valid index for the weight load
    safe_pair_ids = tl.minimum(pair_ids, num_pairs - 1)
    weights = tl.load(topk_weights + safe_pair_ids)
    # Zero out sentinel rows (pair_ids >= num_pairs means padding)
    is_valid = pair_ids < num_pairs
    weights = tl.where(is_valid, weights, 0.0)
    acc = acc * weights[:, None]

    # Scatter to original pair order: write at pair_ids positions
    # Sentinel pair_ids write to the extra row at end (ignored)
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

    def grid1(meta):
        return (num_expert_blocks * triton.cdiv(N1, meta["BLOCK_SIZE_N"]),)

    wrap_triton(_fused_moe_batched_kernel)[grid1](
        hidden_padded,
        w1,
        cache1,
        w1_scale,
        sorted_token_ids,
        expert_ids,
        N=N1,
        K=K,
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
        compute_type=tl.bfloat16,
    )

    out_buf = torch.zeros(
        num_pairs + 1,
        N2,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    def grid2(meta):
        return (num_expert_blocks * triton.cdiv(N2, meta["BLOCK_SIZE_N"]),)

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
) -> torch.Tensor:
    """Convenience wrapper for benchmarking (same as fused_moe_batched_gemm)."""
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

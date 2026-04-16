# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# The GQA "pack GQA" optimization is adapted from FlashAttention
# (Tri Dao, 2023-2025):
#   https://github.com/Dao-AILab/flash-attention
#   flash_attn/cute/pack_gqa.py — PackGQA class
#   hopper/heuristics.h — should_pack_gqa() tile-utilization heuristic
# Licensed under BSD-3-Clause.
#
# Pack GQA folds multiple Q heads that share the same KV head into the M
# (sequence) dimension of a single tile, so K/V are loaded once per KV head
# instead of once per Q head. The tile-utilization heuristic decides when
# packing is beneficial (short seqlen_q, e.g. decode) vs. when simple head
# remapping suffices (long seqlen_q, e.g. prefill).

"""
Triton SDPA Kernel for ExecuTorch CUDA Backend.

This module provides a Triton-optimized implementation of scaled dot-product attention
that can replace the default ATen/Edge SDPA operator during graph transformation to allow
us export the model without decomposing the SDPA operator under libtorch free environment
and have better performance.

GQA support: when enable_gqa=True and H_q > H_kv, the kernel uses "pack GQA"
(adapted from FlashAttention) to fold multiple Q heads sharing the same KV head
into the M (sequence) dimension of a single tile. This avoids redundant K/V reads
and improves tile utilization, especially during decode (seqlen_q=1).
"""

import math
from typing import Optional

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


def _is_power_of_2(n: int) -> bool:
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def _next_power_of_2(x: int) -> int:
    """Get the next power of 2 >= x, clamped to [16, 256].

    Used for HEAD_DIM tiling where tile sizes below 16 waste warps
    and head dims above 256 are unsupported.
    """
    if x <= 16:
        return 16
    if x <= 32:
        return 32
    if x <= 64:
        return 64
    if x <= 128:
        return 128
    return 256


def _next_power_of_2_unclamped(x: int) -> int:
    """Get the next power of 2 >= x (no clamping).

    Used for GQA group-count tiling where num_groups can be small (1, 2, ...)
    and should not be inflated to 16.
    """
    if x <= 0:
        return 1
    return 1 << (x - 1).bit_length()


def _should_pack_gqa(L_q: int, num_groups: int, block_m: int) -> bool:
    """Decide whether to use pack GQA based on tile utilization.

    Pack GQA folds multiple Q heads into the M dimension so they share
    the same K/V loads. This helps when seqlen_q is small relative to
    BLOCK_M (e.g., decode with seqlen_q=1).

    Heuristic from FlashAttention (hopper/heuristics.h, should_pack_gqa):
    compare tile utilization with and without packing; pack if it
    improves efficiency by >10%.

    Reference: https://github.com/Dao-AILab/flash-attention/blob/main/hopper/heuristics.h
    """
    if num_groups <= 1:
        return False

    def round_up(a, b):
        return ((a + b - 1) // b) * b

    nopack_eff = L_q / round_up(L_q, block_m)
    pack_eff = (L_q * num_groups) / round_up(L_q * num_groups, block_m)
    return nopack_eff < 0.9 * pack_eff


def _compute_num_splits(L_kv: int, B: int, H_kv: int, device: torch.device) -> int:
    """Compute optimal KV-split count for flash-decoding on A100 / RTX 4090.

    Balances GPU occupancy against per-split work:
    * Targets >= 2 full SM waves (2 x SM-count CTAs) so the GPU stays
      saturated even with tail effects.
    * Enforces a minimum of 64 KV tokens per split to amortise
      kernel-launch and reduce overhead.
    * Caps at 128 splits to bound reduce-kernel cost.

    A100 -> 108 SMs, RTX 4090 -> 128 SMs.  The heuristic adapts to
    whatever GPU is present via ``torch.cuda.get_device_properties``.
    """
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    ctas_per_split = max(B * H_kv, 1)
    target = max(triton.cdiv(sm_count * 2, ctas_per_split), 1)
    max_by_work = max(L_kv // 64, 1)
    return min(target, max_by_work, 128)


def _validate_qkv_shapes(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    enable_gqa: bool = False,
) -> tuple[int, int, int, int, int, int, int]:
    """
    Validate dimensions and return shape info.
    Args:
        query: Query tensor [B, H_q, L_q, D]
        key: Key tensor [B, H_kv, L_kv, D]
        value: Value tensor [B, H_kv, L_kv, D]
        enable_gqa: If True, H_q must be a multiple of H_kv (GQA/MQA).
    Returns:
        Tuple of (B, H_q, H_kv, L_q, L_kv, D_q, D_kv)
    Raises:
        RuntimeError: If dimensions are incompatible
    """
    B_q, H_q, L_q, D_q = query.shape
    B_k, H_k, L_kv_k, D_k = key.shape
    B_v, H_v, L_kv_v, D_v = value.shape
    # Validate batch dimensions
    if not (B_q == B_k == B_v):
        raise RuntimeError(
            f"Batch dimension must match; got B_q={B_q}, B_k={B_k}, B_v={B_v}."
        )
    # Validate head dimensions
    if not (H_k == H_v):
        raise RuntimeError(f"K and V head counts must match; got H_k={H_k}, H_v={H_v}.")
    if enable_gqa:
        if H_q % H_k != 0:
            raise RuntimeError(
                f"GQA requires H_q divisible by H_kv; got H_q={H_q}, H_kv={H_k}."
            )
    else:
        if not (H_q == H_k):
            raise RuntimeError(
                f"Head counts must match (or use enable_gqa=True); "
                f"got H_q={H_q}, H_k={H_k}."
            )
    # Head dimension must match
    if not (D_q == D_k == D_v):
        raise RuntimeError(
            f"Head dimension must match across Q, K, V; got D_q={D_q}, D_k={D_k}, D_v={D_v}."
        )
    # Key and Value sequence lengths must match
    if L_kv_k != L_kv_v:
        raise RuntimeError(
            f"Key and Value must have the same sequence length; got L_k={L_kv_k}, L_v={L_kv_v}."
        )
    return B_q, H_q, H_k, L_q, L_kv_k, D_q, D_k


# ==============================================================================
# Non-power-of-2 HEAD_DIM kernel
# ==============================================================================
@triton.jit
def _sdpa_fwd_kernel_non_pow2(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    mask_ptr,
    B,
    H_grid,
    LQ,
    LK,
    HEAD_DIM,
    stride_qb,
    stride_qh,
    stride_ql,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kl,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vl,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_ol,
    stride_od,
    stride_mb,
    stride_mh,
    stride_mlq,
    stride_mlk,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_MASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    PACK_GQA: tl.constexpr,
):
    """
    SDPA forward kernel for non-power-of-2 HEAD_DIM.
    Uses dynamic masking to handle arbitrary head dimensions.

    PACK_GQA: when True, multiple Q heads sharing the same KV head are
    folded into the M dimension. The grid iterates over H_kv heads and
    each tile processes up to BLOCK_M rows from the packed (head, seq)
    space. K/V are loaded once per KV head.
    """
    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)

    b = pid_bh // H_grid
    h_grid = pid_bh % H_grid

    offs_packed = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < HEAD_DIM

    if PACK_GQA:
        seq_pos = offs_packed // NUM_GROUPS
        h_within = offs_packed % NUM_GROUPS
        h_q_rows = h_grid * NUM_GROUPS + h_within
        h_kv = h_grid
        row_valid = seq_pos < LQ
        q_ptrs = (
            q_ptr
            + b * stride_qb
            + h_q_rows[:, None] * stride_qh
            + seq_pos[:, None] * stride_ql
            + offs_d[None, :] * stride_qd
        )
    else:
        seq_pos = offs_packed
        h_kv = h_grid // NUM_GROUPS
        row_valid = offs_packed < LQ
        q_ptrs = (
            q_ptr
            + b * stride_qb
            + h_grid * stride_qh
            + offs_packed[:, None] * stride_ql
            + offs_d[None, :] * stride_qd
        )

    q = tl.load(q_ptrs, mask=row_valid[:, None] & d_mask[None, :], other=0.0)

    k_base = k_ptr + b * stride_kb + h_kv * stride_kh
    v_base = v_ptr + b * stride_vb + h_kv * stride_vh

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)

    qk_scale_log2 = scale * 1.4426950408889634

    if HAS_MASK:
        mask_b_base = mask_ptr + b * stride_mb

    NEG_INF: tl.constexpr = float("-inf")

    for start_n in tl.range(0, LK, BLOCK_N, num_stages=2):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_col_mask = offs_n < LK

        k_ptrs = k_base + (offs_n[:, None] * stride_kl + offs_d[None, :] * stride_kd)
        k = tl.load(k_ptrs, mask=kv_col_mask[:, None] & d_mask[None, :], other=0.0)

        qk = tl.dot(q, tl.trans(k))
        qk = (qk * qk_scale_log2).to(tl.float32)

        if IS_CAUSAL:
            causal_mask = offs_n[None, :] > seq_pos[:, None]
            qk = tl.where(causal_mask, tl.full(qk.shape, NEG_INF, dtype=tl.float32), qk)

        if HAS_MASK:
            m_ptrs = (
                mask_b_base
                + seq_pos[:, None] * stride_mlq
                + offs_n[None, :] * stride_mlk
            )
            tile_valid = row_valid[:, None] & kv_col_mask[None, :]
            keep = tl.load(m_ptrs, mask=tile_valid, other=False)
            qk = tl.where(keep, qk, tl.full(qk.shape, NEG_INF, dtype=tl.float32))

        qk = tl.where(
            kv_col_mask[None, :], qk, tl.full(qk.shape, NEG_INF, dtype=tl.float32)
        )

        m_ij = tl.maximum(m_i, tl.max(qk, 1).to(tl.float32))
        safe_diff = tl.where(
            m_ij[:, None] > -float("inf"), qk - m_ij[:, None], -float("inf")
        )
        p = tl.math.exp2(safe_diff).to(tl.float32)
        l_ij = tl.sum(p, 1).to(tl.float32)
        safe_alpha_diff = tl.where(m_ij > -float("inf"), m_i - m_ij, 0.0)
        alpha = tl.math.exp2(safe_alpha_diff).to(tl.float32)

        acc = (acc * alpha[:, None]).to(tl.float32)

        v_ptrs = v_base + (offs_n[:, None] * stride_vl + offs_d[None, :] * stride_vd)
        v = tl.load(v_ptrs, mask=kv_col_mask[:, None] & d_mask[None, :], other=0.0)

        acc = tl.dot(p.to(v.dtype), v, acc).to(tl.float32)

        l_i = (l_i * alpha + l_ij).to(tl.float32)
        m_i = m_ij

    out = acc / l_i[:, None]

    if PACK_GQA:
        o_ptrs = (
            o_ptr
            + b * stride_ob
            + h_q_rows[:, None] * stride_oh
            + seq_pos[:, None] * stride_ol
            + offs_d[None, :] * stride_od
        )
    else:
        o_ptrs = (
            o_ptr
            + b * stride_ob
            + h_grid * stride_oh
            + offs_packed[:, None] * stride_ol
            + offs_d[None, :] * stride_od
        )
    tl.store(o_ptrs, out.to(tl.bfloat16), mask=row_valid[:, None] & d_mask[None, :])


# ==============================================================================
# Power-of-2 HEAD_DIM kernels
# ==============================================================================
@triton.jit
def _sdpa_fwd_kernel_body(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    Mask_ptr,
    B,
    H_grid,
    Lq,
    Lk,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_mb,
    stride_mq,
    stride_mk,
    sm_scale: tl.float32,
    HAS_MASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    PACK_GQA: tl.constexpr,
):
    """
    Shared kernel body for SDPA forward pass.

    PACK_GQA: when True, multiple Q heads sharing the same KV head are
    folded into the M dimension (adapted from FlashAttention's pack_gqa).
    The grid iterates over H_kv heads; each tile processes rows from the
    packed (head, seq) space. K/V are loaded once per KV head, eliminating
    redundant HBM reads across Q heads in a group.

    When False, the grid iterates over H_q heads and each program handles
    one Q head with simple h_kv = h_q // NUM_GROUPS remapping.
    """
    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)
    b = pid_bh // H_grid
    h_grid = pid_bh % H_grid

    offs_packed = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    if PACK_GQA:
        # Decompose packed index: heads interleaved with positions
        # [h0_pos0, h1_pos0, ..., h(G-1)_pos0, h0_pos1, h1_pos1, ...]
        seq_pos = offs_packed // NUM_GROUPS
        h_within = offs_packed % NUM_GROUPS
        h_q_rows = h_grid * NUM_GROUPS + h_within  # [BLOCK_M] vector
        h_kv = h_grid
        row_valid = seq_pos < Lq

        # Scattered Q load: each row may be a different Q head
        q_ptrs = Q_ptr + (
            b * stride_qb
            + h_q_rows[:, None] * stride_qh
            + seq_pos[:, None] * stride_qm
            + offs_d[None, :] * stride_qd
        )
    else:
        seq_pos = offs_packed
        h_kv = h_grid // NUM_GROUPS
        row_valid = offs_packed < Lq

        # Uniform Q load: all rows are the same Q head
        q_ptrs = Q_ptr + (
            b * stride_qb
            + h_grid * stride_qh
            + offs_packed[:, None] * stride_qm
            + offs_d[None, :] * stride_qd
        )

    q_mask = row_valid[:, None] & (offs_d[None, :] < HEAD_DIM)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.bfloat16)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    offs_n_init = tl.arange(0, BLOCK_N)

    for start_n in tl.range(0, Lk, BLOCK_N):
        offs_n = start_n + offs_n_init

        # K load: uniform (single KV head, shared across all Q heads in tile)
        k_ptrs = K_ptr + (
            b * stride_kb
            + h_kv * stride_kh
            + (offs_n[:, None] * stride_kn)
            + (offs_d[None, :] * stride_kd)
        )
        k_mask = (offs_n[:, None] < Lk) & (offs_d[None, :] < HEAD_DIM)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.bfloat16)

        qk = (tl.dot(q, tl.trans(k)).to(tl.float32) * sm_scale).to(tl.float32)

        if HAS_MASK:
            mask_ptrs = Mask_ptr + (
                b * stride_mb
                + (seq_pos[:, None] * stride_mq)
                + (offs_n[None, :] * stride_mk)
            )
            mn_mask = row_valid[:, None] & (offs_n[None, :] < Lk)
            mask_block = tl.load(mask_ptrs, mask=mn_mask, other=False)
            qk = tl.where(
                mask_block, qk, tl.full(qk.shape, -float("inf"), dtype=tl.float32)
            )

        if IS_CAUSAL:
            causal = offs_n[None, :] > seq_pos[:, None]
            qk = tl.where(
                causal, tl.full(qk.shape, -float("inf"), dtype=tl.float32), qk
            )

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1).to(tl.float32))
        safe_diff = tl.where(
            m_ij[:, None] > -float("inf"), qk - m_ij[:, None], -float("inf")
        )
        p_f32 = tl.exp(safe_diff).to(tl.float32)
        l_ij = tl.sum(p_f32, axis=1).to(tl.float32)
        safe_alpha_diff = tl.where(m_ij > -float("inf"), m_i - m_ij, 0.0)
        alpha = tl.exp(safe_alpha_diff).to(tl.float32)

        # V load: uniform (single KV head)
        v_ptrs = V_ptr + (
            b * stride_vb
            + h_kv * stride_vh
            + (offs_n[:, None] * stride_vn)
            + (offs_d[None, :] * stride_vd)
        )
        v_mask = (offs_n[:, None] < Lk) & (offs_d[None, :] < HEAD_DIM)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.bfloat16)

        p_bf16 = p_f32.to(tl.bfloat16)
        acc = (acc * alpha[:, None] + tl.dot(p_bf16, v)).to(tl.float32)
        l_i = (l_i * alpha + l_ij).to(tl.float32)
        m_i = m_ij

    inv_l_i = tl.where(l_i > 0, 1.0 / l_i, 0.0)
    acc = acc * inv_l_i[:, None]

    # O store: scattered when PACK_GQA, uniform otherwise
    if PACK_GQA:
        o_ptrs = O_ptr + (
            b * stride_ob
            + h_q_rows[:, None] * stride_oh
            + seq_pos[:, None] * stride_om
            + offs_d[None, :] * stride_od
        )
    else:
        o_ptrs = O_ptr + (
            b * stride_ob
            + h_grid * stride_oh
            + offs_packed[:, None] * stride_om
            + offs_d[None, :] * stride_od
        )
    o_mask = row_valid[:, None] & (offs_d[None, :] < HEAD_DIM)
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=o_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    ],
    key=["Lq", "Lk", "HEAD_DIM", "HAS_MASK", "IS_CAUSAL", "NUM_GROUPS", "PACK_GQA"],
)
@triton.jit
def _sdpa_fwd_kernel_m64(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    Mask_ptr,
    B,
    H_grid,
    Lq,
    Lk,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_mb,
    stride_mq,
    stride_mk,
    sm_scale: tl.float32,
    HAS_MASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    PACK_GQA: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    _sdpa_fwd_kernel_body(
        Q_ptr,
        K_ptr,
        V_ptr,
        O_ptr,
        Mask_ptr,
        B,
        H_grid,
        Lq,
        Lk,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_om,
        stride_od,
        stride_mb,
        stride_mq,
        stride_mk,
        sm_scale,
        HAS_MASK=HAS_MASK,
        IS_CAUSAL=IS_CAUSAL,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=HEAD_DIM,
        NUM_GROUPS=NUM_GROUPS,
        PACK_GQA=PACK_GQA,
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    ],
    key=["Lq", "Lk", "HEAD_DIM", "HAS_MASK", "IS_CAUSAL", "NUM_GROUPS", "PACK_GQA"],
)
@triton.jit
def _sdpa_fwd_kernel_m32(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    Mask_ptr,
    B,
    H_grid,
    Lq,
    Lk,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_mb,
    stride_mq,
    stride_mk,
    sm_scale: tl.float32,
    HAS_MASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    PACK_GQA: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    _sdpa_fwd_kernel_body(
        Q_ptr,
        K_ptr,
        V_ptr,
        O_ptr,
        Mask_ptr,
        B,
        H_grid,
        Lq,
        Lk,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_om,
        stride_od,
        stride_mb,
        stride_mq,
        stride_mk,
        sm_scale,
        HAS_MASK=HAS_MASK,
        IS_CAUSAL=IS_CAUSAL,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=HEAD_DIM,
        NUM_GROUPS=NUM_GROUPS,
        PACK_GQA=PACK_GQA,
    )


def _validate_sdpa_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float,
    enable_gqa: bool,
) -> None:
    """Validate SDPA input tensors and unsupported feature flags."""
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        raise RuntimeError("Q, K, V must be CUDA tensors.")
    if (
        query.dtype != torch.bfloat16
        or key.dtype != torch.bfloat16
        or value.dtype != torch.bfloat16
    ):
        raise RuntimeError("Expected bfloat16 inputs")
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise RuntimeError(
            f"Expected 4D tensors shaped [B, H, L, D]; got "
            f"query.dim()={query.dim()}, key.dim()={key.dim()}, "
            f"value.dim()={value.dim()}."
        )
    if dropout_p != 0.0:
        raise RuntimeError(
            "dropout_p must be 0.0 (not supported in this implementation)."
        )


def _prepare_mask_params(
    attn_mask: Optional[torch.Tensor],
    B: int,
    L_q: int,
    L_kv: int,
) -> tuple[bool, torch.Tensor, int, int, int]:
    """Prepare attention mask parameters for kernel invocation."""
    if attn_mask is None:
        return False, 0, 0, 0, 0

    if attn_mask.dtype != torch.bool:
        raise RuntimeError("attn_mask must have dtype torch.bool")
    if not attn_mask.is_cuda:
        raise RuntimeError("attn_mask must be a CUDA tensor")
    if attn_mask.shape[1] != 1:
        raise RuntimeError(
            f"attn_mask head dimension must be 1 (broadcast over heads); "
            f"per-head masks are not supported. Got attn_mask.shape={attn_mask.shape}"
        )
    if (
        attn_mask.shape[0] != B
        or attn_mask.shape[2] != L_q
        or attn_mask.shape[3] != L_kv
    ):
        raise RuntimeError(
            f"attn_mask shape mismatch: expected [B={B}, 1, L_q={L_q}, L_kv={L_kv}], "
            f"got {attn_mask.shape}"
        )
    return (
        True,
        attn_mask,
        attn_mask.stride(0),
        attn_mask.stride(2),
        attn_mask.stride(3),
    )


def _launch_pow2_kernel(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    B: int,
    H_q: int,
    H_kv: int,
    L_q: int,
    L_kv: int,
    D: int,
    sm_scale: float,
    HAS_MASK: bool,
    Mask_ptr: torch.Tensor,
    stride_mb: int,
    stride_mq: int,
    stride_mk: int,
    is_causal: bool,
    num_groups: int,
    pack_gqa: bool,
) -> None:
    """Launch power-of-2 optimized SDPA kernel."""
    stride_qb, stride_qh, stride_qm, stride_qd = query.stride()
    stride_kb, stride_kh, stride_kn, stride_kd = key.stride()
    stride_vb, stride_vh, stride_vn, stride_vd = value.stride()
    stride_ob, stride_oh, stride_om, stride_od = out.stride()

    if pack_gqa:
        H_grid = H_kv
        Lq_packed = L_q * num_groups
    else:
        H_grid = H_q
        Lq_packed = L_q

    def grid(meta):
        return (triton.cdiv(Lq_packed, meta["BLOCK_M"]), B * H_grid)

    total_ctas_m64 = ((Lq_packed + 63) // 64) * (B * H_grid)
    threshold = 4 * 84
    kernel = (
        _sdpa_fwd_kernel_m32 if total_ctas_m64 < threshold else _sdpa_fwd_kernel_m64
    )

    wrap_triton(kernel)[grid](
        query,
        key,
        value,
        out,
        Mask_ptr if HAS_MASK else 0,
        B,
        H_grid,
        L_q,
        L_kv,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_om,
        stride_od,
        stride_mb,
        stride_mq,
        stride_mk,
        sm_scale,
        HAS_MASK=HAS_MASK,
        IS_CAUSAL=is_causal,
        HEAD_DIM=D,
        NUM_GROUPS=num_groups,
        PACK_GQA=pack_gqa,
    )


def _launch_non_pow2_kernel(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    B: int,
    H_q: int,
    H_kv: int,
    L_q: int,
    L_kv: int,
    D: int,
    sm_scale: float,
    HAS_MASK: bool,
    is_causal: bool,
    num_groups: int,
    pack_gqa: bool,
) -> None:
    """Launch non-power-of-2 SDPA kernel with dynamic HEAD_DIM masking."""
    stride_qb, stride_qh, stride_qm, stride_qd = query.stride()
    stride_kb, stride_kh, stride_kn, stride_kd = key.stride()
    stride_vb, stride_vh, stride_vn, stride_vd = value.stride()
    stride_ob, stride_oh, stride_om, stride_od = out.stride()

    BLOCK_D = _next_power_of_2(D)
    BLOCK_N = 64 if BLOCK_D >= 256 else 128
    BLOCK_M = 32
    num_warps = 4
    num_stages = 2

    if pack_gqa:
        H_grid = H_kv
        Lq_packed = L_q * num_groups
    else:
        H_grid = H_q
        Lq_packed = L_q

    if HAS_MASK:
        mask_ptr = attn_mask
        stride_mb_np2 = attn_mask.stride(0)
        stride_mh_np2 = attn_mask.stride(1)
        stride_mlq_np2 = attn_mask.stride(2)
        stride_mlk_np2 = attn_mask.stride(3)
    else:
        mask_ptr = 0
        stride_mb_np2 = stride_mh_np2 = stride_mlq_np2 = stride_mlk_np2 = 0

    def grid_non_pow2(meta):
        return (triton.cdiv(Lq_packed, meta["BLOCK_M"]), B * H_grid)

    wrap_triton(_sdpa_fwd_kernel_non_pow2)[grid_non_pow2](
        query,
        key,
        value,
        out,
        mask_ptr,
        B,
        H_grid,
        L_q,
        L_kv,
        D,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_om,
        stride_od,
        stride_mb_np2,
        stride_mh_np2,
        stride_mlq_np2,
        stride_mlk_np2,
        sm_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        HAS_MASK=HAS_MASK,
        IS_CAUSAL=is_causal,
        NUM_GROUPS=num_groups,
        PACK_GQA=pack_gqa,
        num_warps=num_warps,
        num_stages=num_stages,
    )


@triton_op("triton::sdpa", mutates_args={})
def sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float = 0.0,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """
    Triton fused Scaled Dot-Product Attention with GQA pack optimization.

    When enable_gqa=True and H_q > H_kv, this kernel automatically decides
    whether to use "pack GQA" (folding Q heads into the M dimension so they
    share K/V loads) based on a tile-utilization heuristic from FlashAttention.

    Args:
        query: Query tensor [B, H_q, L_q, D], dtype torch.bfloat16
        key: Key tensor [B, H_kv, L_kv, D], dtype torch.bfloat16
        value: Value tensor [B, H_kv, L_kv, D], dtype torch.bfloat16
        attn_mask: Optional bool mask [B, 1, L_q, L_kv] (broadcast over heads)
        dropout_p: must be 0.0
        is_causal: apply causal masking
        scale: attention scale (default: 1/sqrt(D))
        enable_gqa: allow H_q != H_kv (GQA/MQA)
    Returns:
        Output tensor [B, H_q, L_q, D], dtype torch.bfloat16
    """
    _validate_sdpa_inputs(query, key, value, dropout_p, enable_gqa)

    B, H_q, H_kv, L_q, L_kv, D_q, _ = _validate_qkv_shapes(
        query, key, value, enable_gqa
    )
    D = D_q
    num_groups = H_q // H_kv

    if is_causal and L_q != L_kv:
        raise RuntimeError(
            f"Causal masking requires L_q == L_kv; got L_q={L_q}, L_kv={L_kv}. "
            "For decode (L_q < L_kv), use an explicit bool mask instead."
        )

    # Decide whether to pack GQA based on tile utilization heuristic.
    # Use the actual BLOCK_M that the launched kernel will use:
    # - non-pow2 path always uses BLOCK_M=32
    # - pow2 path selects M32 or M64 based on CTA occupancy
    if not _is_power_of_2(D):
        block_m = 32
    else:
        total_ctas_m64 = ((L_q * num_groups + 63) // 64) * (B * H_kv)
        block_m = 32 if total_ctas_m64 < 4 * 84 else 64
    pack_gqa = _should_pack_gqa(L_q, num_groups, block_m)

    out = torch.empty((B, H_q, L_q, D), device=query.device, dtype=query.dtype)
    sm_scale = 1.0 / math.sqrt(D) if scale == 0.0 else scale
    HAS_MASK, Mask_ptr, stride_mb, stride_mq, stride_mk = _prepare_mask_params(
        attn_mask, B, L_q, L_kv
    )

    if _is_power_of_2(D):
        _launch_pow2_kernel(
            query,
            key,
            value,
            out,
            B,
            H_q,
            H_kv,
            L_q,
            L_kv,
            D,
            sm_scale,
            HAS_MASK,
            Mask_ptr,
            stride_mb,
            stride_mq,
            stride_mk,
            is_causal,
            num_groups,
            pack_gqa,
        )
    else:
        _launch_non_pow2_kernel(
            query,
            key,
            value,
            out,
            attn_mask,
            B,
            H_q,
            H_kv,
            L_q,
            L_kv,
            D,
            sm_scale,
            HAS_MASK,
            is_causal,
            num_groups,
            pack_gqa,
        )

    return out


# Register the abstract/fake implementation for torch.export
# This is critical to avoid accessing real tensor data during export
@sdpa.register_fake
def _sdpa_abstract(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float = 0.0,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """
    Abstract/fake implementation for torch.export.
    This just returns an empty tensor with the correct shape/dtype/device.
    """
    # Validate dtypes match
    assert query.dtype == key.dtype == value.dtype, "Q, K, V must have the same dtype"
    # Validate kqv's shape and get the output shape
    B, H_q, _H_kv, L_q, _, D_q, _ = _validate_qkv_shapes(query, key, value, enable_gqa)

    return torch.empty(B, H_q, L_q, D_q, dtype=query.dtype, device=query.device)


# ==============================================================================
# Split-K decode kernel (flash-decoding)
# ==============================================================================
# When L_q == 1 with GQA, the standard kernel launches only
# ceil(num_groups / BLOCK_M) * B * H_kv CTAs (e.g. 2 for Qwen3.5 MoE).
# Split-K partitions the KV sequence across many CTAs for better occupancy,
# then reduces partial results in a second kernel.


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_N": 64}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 256}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_N": 128}, num_warps=8, num_stages=3),
    ],
    key=["Lk", "HEAD_DIM", "NUM_GROUPS", "HAS_MASK"],
)
@triton.jit
def _sdpa_decode_splitk_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_partial_ptr,
    L_partial_ptr,
    Mask_ptr,
    B,
    H_kv,
    Lk,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_op_s,
    stride_op_b,
    stride_op_h,
    stride_op_d,
    stride_lp_s,
    stride_lp_b,
    stride_lp_h,
    stride_mb,
    stride_mq,
    stride_mk,
    sm_scale_log2: tl.float32,
    phi_log2: tl.float32,
    chunk_size,
    HAS_MASK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BATCH_ONE: tl.constexpr,
):
    split_id = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)
    if BATCH_ONE:
        b = 0
        h_kv = pid_bh
    else:
        b = pid_bh // H_kv
        h_kv = pid_bh % H_kv

    start_n = split_id * chunk_size
    end_n = tl.minimum(start_n + chunk_size, Lk)

    offs_d = tl.arange(0, HEAD_DIM)
    offs_g = tl.arange(0, BLOCK_G)
    g_valid = offs_g < NUM_GROUPS
    h_q_heads = h_kv * NUM_GROUPS + offs_g  # [BLOCK_G]

    # Load Q for all heads in this group: [BLOCK_G, HEAD_DIM]
    q_ptrs = Q_ptr + (
        b * stride_qb
        + h_q_heads[:, None] * stride_qh
        + 0 * stride_qm
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=g_valid[:, None], other=0.0)
    # Pre-scale Q so the inner loop avoids a per-element multiply on [G,N] QK
    q = (q.to(tl.float32) * sm_scale_log2).to(tl.bfloat16)

    # FlashDecoding++ async softmax with exp2: all scores in log2 space
    l_i = tl.zeros([BLOCK_G], dtype=tl.float32)
    acc = tl.zeros([BLOCK_G, HEAD_DIM], dtype=tl.float32)

    offs_n_init = tl.arange(0, BLOCK_N)

    for tile_start in tl.range(start_n, end_n, BLOCK_N):
        offs_n = tile_start + offs_n_init
        n_valid = offs_n < end_n

        k_ptrs = K_ptr + (
            b * stride_kb
            + h_kv * stride_kh
            + offs_n[:, None] * stride_kn
            + offs_d[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=n_valid[:, None], other=0.0).to(tl.bfloat16)

        # QK: [BLOCK_G, BLOCK_N] — Q already scaled, result in log2 space
        qk = tl.dot(q, tl.trans(k)).to(tl.float32)

        # Mask out-of-bounds KV positions
        qk = tl.where(
            n_valid[None, :],
            qk,
            tl.full(qk.shape, -float("inf"), dtype=tl.float32),
        )

        if HAS_MASK:
            mask_ptrs = Mask_ptr + (
                b * stride_mb + 0 * stride_mq + offs_n[None, :] * stride_mk
            )
            mask_block = tl.load(mask_ptrs, mask=n_valid[None, :], other=False)
            qk = tl.where(
                mask_block, qk, tl.full(qk.shape, -float("inf"), dtype=tl.float32)
            )

        # FlashDecoding++ async softmax: exp2 maps to single PTX ex2 instruction
        safe_diff = tl.where(qk > -float("inf"), qk - phi_log2, -float("inf"))
        p_f32 = tl.math.exp2(safe_diff).to(tl.float32)
        l_ij = tl.sum(p_f32, axis=1).to(tl.float32)

        v_ptrs = V_ptr + (
            b * stride_vb
            + h_kv * stride_vh
            + offs_n[:, None] * stride_vn
            + offs_d[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=n_valid[:, None], other=0.0).to(tl.bfloat16)

        p_bf16 = p_f32.to(tl.bfloat16)
        acc = (acc + tl.dot(p_bf16, v)).to(tl.float32)
        l_i = (l_i + l_ij).to(tl.float32)

    # Store partial results for valid groups only
    h_q_all = h_kv * NUM_GROUPS + offs_g  # [BLOCK_G]
    o_ptrs = O_partial_ptr + (
        split_id * stride_op_s
        + b * stride_op_b
        + h_q_all[:, None] * stride_op_h
        + offs_d[None, :] * stride_op_d
    )
    tl.store(o_ptrs, acc, mask=g_valid[:, None])

    ll_ptrs = L_partial_ptr + (
        split_id * stride_lp_s + b * stride_lp_b + h_q_all * stride_lp_h
    )
    tl.store(ll_ptrs, l_i, mask=g_valid)


@triton.jit
def _sdpa_decode_reduce_kernel(
    O_partial_ptr,
    L_partial_ptr,
    O_ptr,
    num_splits,
    stride_op_s,
    stride_op_b,
    stride_op_h,
    stride_op_d,
    stride_lp_s,
    stride_lp_b,
    stride_lp_h,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    HEAD_DIM: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_d = tl.arange(0, HEAD_DIM)

    # FlashDecoding++ async softmax: no rescaling needed, just sum partials
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    l_global = tl.zeros([1], dtype=tl.float32)

    for s in tl.range(0, num_splits, num_stages=2):
        l_ptr = L_partial_ptr + s * stride_lp_s + pid * stride_lp_h
        o_ptrs = O_partial_ptr + (
            s * stride_op_s + pid * stride_op_h + offs_d * stride_op_d
        )

        l_s = tl.load(l_ptr)
        o_s = tl.load(o_ptrs)

        acc += o_s
        l_global += l_s

    inv_l = tl.where(l_global > 0, 1.0 / l_global, 0.0)
    acc = acc * inv_l

    o_out_ptrs = O_ptr + pid * stride_oh + offs_d * stride_od
    tl.store(o_out_ptrs, acc.to(tl.bfloat16))


def _launch_decode_splitk(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    B: int,
    H_q: int,
    H_kv: int,
    L_kv: int,
    D: int,
    sm_scale: float,
    HAS_MASK: bool,
    Mask_ptr,
    stride_mb: int,
    stride_mq: int,
    stride_mk: int,
    num_groups: int,
    phi: float,
) -> None:
    num_splits = _compute_num_splits(L_kv, B, H_kv, query.device)
    chunk_size = triton.cdiv(L_kv, num_splits)

    _LOG2E = 1.4426950408889634
    sm_scale_log2 = sm_scale * _LOG2E
    phi_log2 = phi * _LOG2E

    O_partial = torch.empty(
        (num_splits, B, H_q, D), device=query.device, dtype=torch.float32
    )
    L_partial = torch.zeros(
        (num_splits, B, H_q), device=query.device, dtype=torch.float32
    )

    stride_qb, stride_qh, stride_qm, stride_qd = query.stride()
    stride_kb, stride_kh, stride_kn, stride_kd = key.stride()
    stride_vb, stride_vh, stride_vn, stride_vd = value.stride()
    stride_ob, stride_oh, stride_om, stride_od = out.stride()
    stride_op_s, stride_op_b, stride_op_h, stride_op_d = O_partial.stride()
    stride_lp_s, stride_lp_b, stride_lp_h = L_partial.stride()

    grid_split = (num_splits, B * H_kv)
    wrap_triton(_sdpa_decode_splitk_kernel)[grid_split](
        query,
        key,
        value,
        O_partial,
        L_partial,
        Mask_ptr if HAS_MASK else 0,
        B,
        H_kv,
        L_kv,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_op_s,
        stride_op_b,
        stride_op_h,
        stride_op_d,
        stride_lp_s,
        stride_lp_b,
        stride_lp_h,
        stride_mb,
        stride_mq,
        stride_mk,
        sm_scale_log2,
        phi_log2,
        chunk_size,
        HAS_MASK=HAS_MASK,
        HEAD_DIM=D,
        NUM_GROUPS=num_groups,
        BLOCK_G=_next_power_of_2_unclamped(num_groups),
        BATCH_ONE=B == 1,
    )

    grid_reduce = (B * H_q,)
    wrap_triton(_sdpa_decode_reduce_kernel)[grid_reduce](
        O_partial,
        L_partial,
        out,
        num_splits,
        stride_op_s,
        stride_op_b,
        stride_op_h,
        stride_op_d,
        stride_lp_s,
        stride_lp_b,
        stride_lp_h,
        stride_ob,
        stride_oh,
        stride_om,
        stride_od,
        HEAD_DIM=D,
        num_warps=4,
        num_stages=1,
    )


@triton_op("triton::sdpa_decode_splitk", mutates_args={})
def sdpa_decode_splitk(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float = 0.0,
    enable_gqa: bool = False,
    phi: float = 5.0,
) -> torch.Tensor:
    """Split-K flash-decoding SDPA for L_q=1 (decode step).

    Uses FlashDecoding++ async softmax with unified maximum value (phi)
    to eliminate per-split max tracking and cross-split rescaling.

    Signature mirrors sdpa() for drop-in use with torch.cond dispatch.
    enable_gqa is accepted but ignored — GQA is handled natively via
    H_q // H_kv grouping; no packed-GQA tradeoff exists at L_q=1.
    """
    _validate_sdpa_inputs(query, key, value, dropout_p, enable_gqa)

    B, H_q, L_q, D = query.shape
    _, H_kv, L_kv, _ = key.shape

    out = torch.empty((B, H_q, L_q, D), device=query.device, dtype=query.dtype)

    # is_causal is a no-op at L_q=1 (single query can't attend to future
    # positions), so we accept it silently for API compatibility with callers
    # that always pass is_causal=True for decode.

    # Validation — only check at runtime (concrete shapes), not during AOTI
    # tracing where shapes are symbolic. torch.cond traces both branches with
    # the same symbolic L_q, so L_q is not necessarily 1 during tracing.
    if isinstance(L_q, int):
        if L_q != 1:
            raise RuntimeError(
                f"sdpa_decode_splitk requires L_q == 1 (decode); got L_q={L_q}"
            )
        if H_q % H_kv != 0:
            raise RuntimeError(
                f"H_q must be divisible by H_kv; got H_q={H_q}, H_kv={H_kv}"
            )
        if not _is_power_of_2(D):
            raise RuntimeError(
                f"sdpa_decode_splitk requires power-of-2 head dim; got D={D}"
            )

    num_groups = H_q // H_kv
    sm_scale = 1.0 / math.sqrt(D) if scale == 0.0 else scale
    HAS_MASK, Mask_ptr, stride_mb, stride_mq, stride_mk = _prepare_mask_params(
        attn_mask, B, L_q, L_kv
    )

    _launch_decode_splitk(
        query,
        key,
        value,
        out,
        B,
        H_q,
        H_kv,
        L_kv,
        D,
        sm_scale,
        HAS_MASK,
        Mask_ptr,
        stride_mb,
        stride_mq,
        stride_mk,
        num_groups,
        phi,
    )
    return out


@sdpa_decode_splitk.register_fake
def _sdpa_decode_splitk_abstract(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float = 0.0,
    enable_gqa: bool = False,
    phi: float = 5.0,
) -> torch.Tensor:
    assert query.dtype == key.dtype == value.dtype, "Q, K, V must have the same dtype"
    B, H_q, L_q, D = query.shape
    return torch.empty(B, H_q, L_q, D, dtype=query.dtype, device=query.device)

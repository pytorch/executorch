# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# TQ4 fused Flash Attention kernel with Pack GQA optimization.
#
# Decompression logic adapted from turboquant-vllm v1.4.0
# (Alberto-Codes/turboquant-vllm, Apache 2.0).
# Pack GQA and kernel structure adapted from sdpa.py in this directory.
#
# Compatible with: turboquant-vllm 1.4.0
#
# Reference: arXiv 2504.19874 — "TurboQuant: Online Vector Quantization
# with Near-optimal Distortion Rate" (ICLR 2026).

"""
Fused TQ4 SDPA: attention over nibble-packed compressed K/V cache.

Both K and V tiles are decompressed inline from uint8 nibble-packed indices
in the attention inner loop. The full decompressed cache is never materialized.
Q is pre-rotated by Pi^T, output is post-rotated by Pi outside the kernel.

Pack GQA (from FlashAttention) folds multiple Q heads sharing one KV head
into the M dimension, loading K/V only once per KV head group. This gives
up to NUM_GROUPS x reduction in K/V HBM traffic for decode.
"""

import math
from typing import Optional

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _should_pack_gqa(L_q: int, num_groups: int, block_m: int) -> bool:
    """Decide whether to use Pack GQA based on tile utilization.

    Pack GQA folds multiple Q heads into the M dimension so they share
    the same K/V loads. This helps when seqlen_q is small relative to
    BLOCK_M (e.g., decode with seqlen_q=1).

    Heuristic from FlashAttention (hopper/heuristics.h, should_pack_gqa).
    """
    if num_groups <= 1:
        return False

    def round_up(a, b):
        return ((a + b - 1) // b) * b

    nopack_eff = L_q / round_up(L_q, block_m)
    pack_eff = (L_q * num_groups) / round_up(L_q * num_groups, block_m)
    return nopack_eff < 0.9 * pack_eff


# ---------------------------------------------------------------------------
# Kernel body
# ---------------------------------------------------------------------------


@triton.jit
def _tq4_sdpa_fwd_kernel_body(
    Q_ptr,
    KP_ptr,
    KN_ptr,
    VP_ptr,
    VN_ptr,
    LUT_hi_ptr,
    LUT_lo_ptr,
    Mask_ptr,
    O_ptr,
    B,
    H_grid,
    Lq,
    Lk,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kpb,
    stride_kph,
    stride_kpn,
    stride_kpd,
    stride_knb,
    stride_knh,
    stride_knn,
    stride_vpb,
    stride_vph,
    stride_vpn,
    stride_vpd,
    stride_vnb,
    stride_vnh,
    stride_vnn,
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
    HALF_D: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    PACK_GQA: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)
    b = pid_bh // H_grid
    h_grid = pid_bh % H_grid

    offs_packed = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_d_half = tl.arange(0, HALF_D)

    if PACK_GQA:
        seq_pos = offs_packed // NUM_GROUPS
        h_within = offs_packed % NUM_GROUPS
        h_q_rows = h_grid * NUM_GROUPS + h_within
        h_kv = h_grid
        row_valid = seq_pos < Lq

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

        q_ptrs = Q_ptr + (
            b * stride_qb
            + h_grid * stride_qh
            + offs_packed[:, None] * stride_qm
            + offs_d[None, :] * stride_qd
        )

    q = tl.load(q_ptrs, mask=row_valid[:, None], other=0.0).to(tl.bfloat16)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Prescale for exp2-based softmax (single PTX instruction)
    qk_scale = sm_scale * 1.44269504

    # TQ4 K/V base pointers (uniform: single KV head)
    kp_base = KP_ptr + b * stride_kpb + h_kv * stride_kph
    kn_base = KN_ptr + b * stride_knb + h_kv * stride_knh
    vp_base = VP_ptr + b * stride_vpb + h_kv * stride_vph
    vn_base = VN_ptr + b * stride_vnb + h_kv * stride_vnh

    offs_n_init = tl.arange(0, BLOCK_N)

    for start_n in tl.range(0, Lk, BLOCK_N):
        offs_n = start_n + offs_n_init
        kv_valid = offs_n < Lk

        # -- K decompression (LUT, no norm multiply on [N,D] tile) --
        kp_ptrs = (
            kp_base + offs_n[:, None] * stride_kpn + offs_d_half[None, :] * stride_kpd
        )
        k_packed_data = tl.load(kp_ptrs, mask=kv_valid[:, None], other=0).to(tl.int32)
        k = tl.join(
            tl.load(LUT_hi_ptr + k_packed_data),
            tl.load(LUT_lo_ptr + k_packed_data),
        ).reshape(BLOCK_N, HEAD_DIM)

        # Q @ K^T with norm factored out: Q @ (C*n)^T = (Q @ C^T) * n^T
        kn = tl.load(kn_base + offs_n * stride_knn, mask=kv_valid, other=0.0)
        qk = (tl.dot(q, tl.trans(k)) * qk_scale * kn[None, :]).to(tl.float32)

        if HAS_MASK:
            mask_ptrs = Mask_ptr + (
                b * stride_mb
                + seq_pos[:, None] * stride_mq
                + offs_n[None, :] * stride_mk
            )
            mn_mask = row_valid[:, None] & kv_valid[None, :]
            mask_block = tl.load(mask_ptrs, mask=mn_mask, other=False)
            qk = tl.where(mask_block, qk, float("-inf"))

        if IS_CAUSAL:
            causal = offs_n[None, :] > seq_pos[:, None]
            qk = tl.where(causal, float("-inf"), qk)

        qk = tl.where(kv_valid[None, :], qk, float("-inf"))

        # NaN-safe online softmax (exp2)
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        # Guard against -inf - (-inf) = NaN when all positions are masked
        safe_alpha = tl.where(m_new > -float("inf"), m_i - m_new, 0.0)
        alpha = tl.math.exp2(safe_alpha)
        safe_p = tl.where(
            m_new[:, None] > -float("inf"), qk - m_new[:, None], -float("inf")
        )
        p = tl.math.exp2(safe_p)
        l_ij = tl.sum(p, 1)

        # -- V decompression (LUT, norm factored into P) --
        vp_ptrs = (
            vp_base + offs_n[:, None] * stride_vpn + offs_d_half[None, :] * stride_vpd
        )
        v_packed_data = tl.load(vp_ptrs, mask=kv_valid[:, None], other=0).to(tl.int32)
        v = tl.join(
            tl.load(LUT_hi_ptr + v_packed_data),
            tl.load(LUT_lo_ptr + v_packed_data),
        ).reshape(BLOCK_N, HEAD_DIM)

        # P @ (C*n) = (P*n) @ C — factor norm into P instead of V
        vn = tl.load(vn_base + offs_n * stride_vnn, mask=kv_valid, other=0.0)
        p_scaled = (p * vn[None, :]).to(tl.bfloat16)
        acc = (acc * alpha[:, None] + tl.dot(p_scaled, v)).to(tl.float32)
        l_i = (l_i * alpha + l_ij).to(tl.float32)
        m_i = m_new

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
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=row_valid[:, None])


# ---------------------------------------------------------------------------
# Autotuned kernel wrappers (M64 and M32)
# ---------------------------------------------------------------------------


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
def _tq4_sdpa_fwd_kernel_m64(
    Q_ptr,
    KP_ptr,
    KN_ptr,
    VP_ptr,
    VN_ptr,
    LUT_hi_ptr,
    LUT_lo_ptr,
    Mask_ptr,
    O_ptr,
    B,
    H_grid,
    Lq,
    Lk,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kpb,
    stride_kph,
    stride_kpn,
    stride_kpd,
    stride_knb,
    stride_knh,
    stride_knn,
    stride_vpb,
    stride_vph,
    stride_vpn,
    stride_vpd,
    stride_vnb,
    stride_vnh,
    stride_vnn,
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
    HALF_D: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    PACK_GQA: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    _tq4_sdpa_fwd_kernel_body(
        Q_ptr,
        KP_ptr,
        KN_ptr,
        VP_ptr,
        VN_ptr,
        LUT_hi_ptr,
        LUT_lo_ptr,
        Mask_ptr,
        O_ptr,
        B,
        H_grid,
        Lq,
        Lk,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_kpb,
        stride_kph,
        stride_kpn,
        stride_kpd,
        stride_knb,
        stride_knh,
        stride_knn,
        stride_vpb,
        stride_vph,
        stride_vpn,
        stride_vpd,
        stride_vnb,
        stride_vnh,
        stride_vnn,
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
        HALF_D=HALF_D,
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
def _tq4_sdpa_fwd_kernel_m32(
    Q_ptr,
    KP_ptr,
    KN_ptr,
    VP_ptr,
    VN_ptr,
    LUT_hi_ptr,
    LUT_lo_ptr,
    Mask_ptr,
    O_ptr,
    B,
    H_grid,
    Lq,
    Lk,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kpb,
    stride_kph,
    stride_kpn,
    stride_kpd,
    stride_knb,
    stride_knh,
    stride_knn,
    stride_vpb,
    stride_vph,
    stride_vpn,
    stride_vpd,
    stride_vnb,
    stride_vnh,
    stride_vnn,
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
    HALF_D: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    PACK_GQA: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    _tq4_sdpa_fwd_kernel_body(
        Q_ptr,
        KP_ptr,
        KN_ptr,
        VP_ptr,
        VN_ptr,
        LUT_hi_ptr,
        LUT_lo_ptr,
        Mask_ptr,
        O_ptr,
        B,
        H_grid,
        Lq,
        Lk,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_kpb,
        stride_kph,
        stride_kpn,
        stride_kpd,
        stride_knb,
        stride_knh,
        stride_knn,
        stride_vpb,
        stride_vph,
        stride_vpn,
        stride_vpd,
        stride_vnb,
        stride_vnh,
        stride_vnn,
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
        HALF_D=HALF_D,
        NUM_GROUPS=NUM_GROUPS,
        PACK_GQA=PACK_GQA,
    )


# ---------------------------------------------------------------------------
# Host-side launcher
# ---------------------------------------------------------------------------


def _launch_tq4_kernel(
    q_rot,
    k_packed,
    k_norms,
    v_packed,
    v_norms,
    lut_hi,
    lut_lo,
    mask_ptr,
    out_rot,
    B,
    H_Q,
    H_KV,
    L_Q,
    L_KV,
    D,
    sm_scale,
    HAS_MASK,
    stride_mb,
    stride_mq,
    stride_mk,
    is_causal,
    num_groups,
    pack_gqa,
):
    HALF_D = D // 2

    if pack_gqa:
        H_grid = H_KV
        Lq_packed = L_Q * num_groups
    else:
        H_grid = H_Q
        Lq_packed = L_Q

    def grid(meta):
        return (triton.cdiv(Lq_packed, meta["BLOCK_M"]), B * H_grid)

    total_ctas_m64 = ((Lq_packed + 63) // 64) * (B * H_grid)
    threshold = 4 * 84
    kernel = (
        _tq4_sdpa_fwd_kernel_m32
        if total_ctas_m64 < threshold
        else _tq4_sdpa_fwd_kernel_m64
    )

    wrap_triton(kernel)[grid](
        q_rot,
        k_packed,
        k_norms,
        v_packed,
        v_norms,
        lut_hi,
        lut_lo,
        mask_ptr if HAS_MASK else 0,
        out_rot,
        B,
        H_grid,
        L_Q,
        L_KV,
        *q_rot.stride(),
        *k_packed.stride(),
        *k_norms.stride(),
        *v_packed.stride(),
        *v_norms.stride(),
        *out_rot.stride(),
        stride_mb,
        stride_mq,
        stride_mk,
        sm_scale,
        HAS_MASK=HAS_MASK,
        IS_CAUSAL=is_causal,
        HEAD_DIM=D,
        HALF_D=HALF_D,
        NUM_GROUPS=num_groups,
        PACK_GQA=pack_gqa,
    )


# ---------------------------------------------------------------------------
# @triton_op wrapper
# ---------------------------------------------------------------------------


def _validate_tq4_inputs(query, k_packed, v_packed):
    """Validate tensor shapes, dtypes, and device for tq4_sdpa."""
    B, H_Q, N_Q, D = query.shape
    B_kp, H_KV, N_KV, HALF_D = k_packed.shape

    if not query.is_cuda:
        raise RuntimeError("query must be a CUDA tensor")
    if query.dtype != torch.bfloat16:
        raise RuntimeError(f"query must be bfloat16, got {query.dtype}")
    if query.dim() != 4:
        raise RuntimeError(f"query must be 4D [B, H, L, D], got {query.dim()}D")
    if k_packed.dim() != 4 or v_packed.dim() != 4:
        raise RuntimeError("k_packed and v_packed must be 4D [B, H, L, D//2]")
    if k_packed.dtype != torch.uint8 or v_packed.dtype != torch.uint8:
        raise RuntimeError("k_packed and v_packed must be uint8")
    if B_kp != B:
        raise RuntimeError(
            f"Batch dim mismatch: query has B={B}, k_packed has B={B_kp}"
        )
    if H_Q % H_KV != 0:
        raise RuntimeError(
            f"H_Q must be a multiple of H_KV for GQA head mapping, "
            f"got H_Q={H_Q}, H_KV={H_KV}"
        )
    if HALF_D * 2 != D:
        raise RuntimeError(
            f"k_packed last dim ({HALF_D}) * 2 must equal query head_dim ({D})"
        )
    if D & (D - 1) != 0:
        raise RuntimeError(
            f"HEAD_DIM must be a power of 2, got {D}. "
            "Non-power-of-2 head dims are not supported."
        )


def _validate_tq4_mask(attn_mask, B, N_Q, N_KV):
    """Validate attention mask for tq4_sdpa."""
    if attn_mask is None:
        return
    if attn_mask.dtype != torch.bool:
        raise RuntimeError(
            f"attn_mask must be bool, got {attn_mask.dtype}. "
            "Additive float masks are not supported."
        )
    if not attn_mask.is_cuda:
        raise RuntimeError("attn_mask must be a CUDA tensor")
    if attn_mask.shape[1] != 1:
        raise RuntimeError(
            f"attn_mask head dimension must be 1 (broadcast over heads); "
            f"per-head masks are not supported. "
            f"Got attn_mask.shape={attn_mask.shape}"
        )
    if (
        attn_mask.shape[0] != B
        or attn_mask.shape[2] != N_Q
        or attn_mask.shape[3] != N_KV
    ):
        raise RuntimeError(
            f"attn_mask shape mismatch: expected "
            f"[B={B}, 1, L_Q={N_Q}, L_KV={N_KV}], "
            f"got {attn_mask.shape}"
        )


@triton_op("triton::tq4_sdpa", mutates_args={})
def tq4_sdpa(
    query: torch.Tensor,
    k_packed: torch.Tensor,
    k_norms: torch.Tensor,
    v_packed: torch.Tensor,
    v_norms: torch.Tensor,
    centroids: torch.Tensor,
    rotation: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Fused TQ4 SDPA over nibble-packed compressed K/V cache.

    Decompresses K/V per tile in the attention inner loop. The full
    decompressed cache is never materialized (3.8x memory savings).

    H_Q must be a multiple of H_KV (GQA/MQA). HEAD_DIM must be a
    power of 2. The kernel maps Q heads to KV heads internally via
    Pack GQA when beneficial.

    Args:
        query: [B, H_Q, L_Q, D] bf16
        k_packed: [B, H_KV, L_KV, D//2] uint8 nibble-packed key indices
        k_norms: [B, H_KV, L_KV, 1] key vector norms (float or bf16)
        v_packed: [B, H_KV, L_KV, D//2] uint8 nibble-packed value indices
        v_norms: [B, H_KV, L_KV, 1] value vector norms (float or bf16)
        centroids: [16] fp32 Lloyd-Max codebook
        rotation: [D, D] orthogonal rotation matrix
        attn_mask: Optional [B, 1, L_Q, L_KV] bool mask
        is_causal: apply causal masking (requires L_Q == L_KV)

    Returns:
        [B, H_Q, L_Q, D] bf16 attention output
    """
    _validate_tq4_inputs(query, k_packed, v_packed)

    B, H_Q, N_Q, D = query.shape
    _, H_KV, N_KV, HALF_D = k_packed.shape

    _validate_tq4_mask(attn_mask, B, N_Q, N_KV)

    sm_scale = 1.0 / math.sqrt(D)
    num_groups = H_Q // H_KV

    # Build [256] bf16 lookup tables from [16] centroids.
    # In the export path, inductor fuses this into the compiled graph.
    all_bytes = torch.arange(256, device=centroids.device)
    lut_hi = centroids[(all_bytes >> 4).long()].to(query.dtype).contiguous()
    lut_lo = centroids[(all_bytes & 0x0F).long()].to(query.dtype).contiguous()

    # Reshape norms: [B, H, S, 1] -> [B, H, S]
    k_n = k_norms.reshape(B, H_KV, N_KV).contiguous()
    v_n = v_norms.reshape(B, H_KV, N_KV).contiguous()

    # Pre-rotate Q: Q_rot = Q @ Pi^T (bf16 — TQ4 error dominates)
    q_rot = torch.matmul(query, rotation.T.to(query.dtype)).contiguous()

    out_rot = torch.empty_like(query)

    HAS_MASK = attn_mask is not None
    if is_causal and N_Q != N_KV:
        raise RuntimeError(
            f"is_causal requires L_Q == L_KV, got L_Q={N_Q}, L_KV={N_KV}. "
            "For decode (L_Q < L_KV), use an explicit bool mask instead."
        )
    if HAS_MASK:
        mask_ptr = attn_mask
        stride_mb = attn_mask.stride(0)
        stride_mq = attn_mask.stride(2)
        stride_mk = attn_mask.stride(3)
    else:
        mask_ptr = 0
        stride_mb = 0
        stride_mq = 0
        stride_mk = 0

    # Pack GQA decision
    total_ctas_m64 = ((N_Q * num_groups + 63) // 64) * (B * H_KV)
    block_m = 32 if total_ctas_m64 < 4 * 84 else 64
    pack_gqa = _should_pack_gqa(N_Q, num_groups, block_m)

    _launch_tq4_kernel(
        q_rot,
        k_packed,
        k_n,
        v_packed,
        v_n,
        lut_hi,
        lut_lo,
        mask_ptr,
        out_rot,
        B,
        H_Q,
        H_KV,
        N_Q,
        N_KV,
        D,
        sm_scale,
        HAS_MASK,
        stride_mb,
        stride_mq,
        stride_mk,
        is_causal,
        num_groups,
        pack_gqa,
    )

    # Post-rotate: convert from rotated space back to original space
    return torch.matmul(out_rot, rotation.to(query.dtype))


@tq4_sdpa.register_fake
def _tq4_sdpa_fake(
    query: torch.Tensor,
    k_packed: torch.Tensor,
    k_norms: torch.Tensor,
    v_packed: torch.Tensor,
    v_norms: torch.Tensor,
    centroids: torch.Tensor,
    rotation: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    return torch.empty_like(query)

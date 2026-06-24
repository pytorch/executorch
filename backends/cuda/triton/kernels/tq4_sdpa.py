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
    KV_LEN_ptr,
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
    HAS_KV_LEN: tl.constexpr,
    MASK_IS_CAUSAL: tl.constexpr,
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

    # Bound the KV loop to the number of valid (filled) positions instead of the
    # full pre-allocated buffer Lk. For decode this is input_pos+1; for a prefill
    # chunk it is chunk_end. This makes the global-layer attention O(context)
    # rather than O(max_seq_len) — the empty tail of the cache is never touched.
    # kv_len is read from a GPU scalar so the bound updates across CUDA-graph
    # replays (decode is graph-captured). When not provided (HAS_KV_LEN False)
    # it falls back to Lk, preserving the original behavior exactly.
    if HAS_KV_LEN:
        kv_len = tl.load(KV_LEN_ptr)
    else:
        kv_len = Lk

    # Per-tile causal upper bound (prefill). With a causal attn_mask, the rows in
    # this tile attend only up to their own absolute position; the largest such
    # position is (kv_len - Lq) + max(seq_pos) — seq_pos is the query-row index,
    # and the (kv_len - Lq) offset converts it to an absolute KV position (so it
    # is correct for chunked prefill, not just the first chunk). KV blocks that
    # start beyond it are fully masked, so we stop the loop there. This is the
    # prefill analogue of the kv_len decode clamp and ~halves the causal-triangle
    # work. For decode (Lq=1, max(seq_pos)=0) this evaluates to kv_len, so decode
    # is byte-identical. Applied only when MASK_IS_CAUSAL (the caller guarantees a
    # causal mask); otherwise the full kv_len bound is kept, which is safe for an
    # arbitrary mask.
    loop_end = kv_len
    if MASK_IS_CAUSAL or IS_CAUSAL:
        max_q_pos = (kv_len - Lq) + tl.max(seq_pos)
        loop_end = tl.minimum(kv_len, max_q_pos + 1)

    for start_n in tl.range(0, loop_end, BLOCK_N):
        offs_n = start_n + offs_n_init
        kv_valid = offs_n < kv_len

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
            # Absolute causal-offset: a query row's KV position is
            # (kv_len - Lq) + seq_pos, which is correct for chunked prefill
            # (Lq < kv_len). For the square is_causal case (kv_len == Lq) this
            # reduces to offs_n > seq_pos. This lets a caller that guarantees a
            # standard causal mask skip the explicit mask entirely.
            causal = offs_n[None, :] > (kv_len - Lq) + seq_pos[:, None]
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
# Autotuned prefill kernel (single, no-spill, GPU-portable configs)
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        # Single no-spill prefill config set. With small BLOCK_M the fp32
        # acc[BLOCK_M, HEAD_DIM] stays in registers, and the staged decompressed
        # K/V tile (num_stages*BLOCK_N*HEAD_DIM*2 bytes) fits A100 shared memory
        # at HEAD_DIM=512 (zero OutOfResources). The BLOCK_N=16/32 configs also
        # fit smaller-SMEM GPUs (e.g. RTX 5090); the autotuner prunes any config
        # that does not fit the running GPU.
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 16}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16}, num_warps=4, num_stages=4),
    ],
    key=["Lq", "Lk", "HEAD_DIM", "HAS_MASK", "IS_CAUSAL", "NUM_GROUPS", "PACK_GQA"],
)
@triton.jit
def _tq4_sdpa_prefill_kernel(
    Q_ptr,
    KP_ptr,
    KN_ptr,
    VP_ptr,
    VN_ptr,
    LUT_hi_ptr,
    LUT_lo_ptr,
    Mask_ptr,
    O_ptr,
    KV_LEN_ptr,
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
    HAS_KV_LEN: tl.constexpr,
    MASK_IS_CAUSAL: tl.constexpr,
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
        KV_LEN_ptr,
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
        HAS_KV_LEN=HAS_KV_LEN,
        MASK_IS_CAUSAL=MASK_IS_CAUSAL,
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


def _launch_tq4_prefill(
    q_rot,
    k_packed,
    k_norms,
    v_packed,
    v_norms,
    lut_hi,
    lut_lo,
    mask_ptr,
    out_rot,
    kv_len_ptr,
    B,
    H_Q,
    H_KV,
    L_Q,
    L_KV,
    D,
    sm_scale,
    HAS_MASK,
    HAS_KV_LEN,
    MASK_IS_CAUSAL,
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

    wrap_triton(_tq4_sdpa_prefill_kernel)[grid](
        q_rot,
        k_packed,
        k_norms,
        v_packed,
        v_norms,
        lut_hi,
        lut_lo,
        mask_ptr if HAS_MASK else 0,
        out_rot,
        kv_len_ptr if HAS_KV_LEN else 0,
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
        HAS_KV_LEN=HAS_KV_LEN,
        MASK_IS_CAUSAL=MASK_IS_CAUSAL,
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
    scale: Optional[float] = None,
    kv_len: Optional[torch.Tensor] = None,
    mask_is_causal: bool = False,
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
        scale: softmax scale applied to ``Q @ K^T``. Defaults to
            ``1/sqrt(HEAD_DIM)`` when ``None``. Models that handle their
            own normalization (e.g. QK-norm models that fold the
            ``1/sqrt(d)`` factor into the trained weights, which then use
            ``1.0``) should pass an explicit value.
        kv_len: Optional GPU int scalar = number of valid (filled) KV
            positions. When provided, the inner KV loop is bounded to
            ``kv_len`` instead of the full pre-allocated ``L_KV``, making
            attention O(context) instead of O(max_seq_len). It is read on
            the device (no host sync) so the bound updates correctly under
            CUDA-graph replay (decode). For decode pass ``input_pos + 1``;
            for a prefill chunk pass ``chunk_end``. When ``None`` the loop
            runs over the full ``L_KV`` (original behavior).
        mask_is_causal: Set True only when ``attn_mask`` is a standard
            causal mask (row at absolute position p attends to [0, p]).
            Enables a per-tile causal upper bound that skips KV blocks past
            the tile's last query position — the prefill analogue of the
            ``kv_len`` clamp, ~halving prefill (causal-triangle) work. It is
            a no-op for decode (L_Q=1) and byte-identical there. Leave False
            (default) for any non-causal mask; the kernel keeps the full
            ``kv_len`` bound, which is correct for an arbitrary mask.

    Returns:
        [B, H_Q, L_Q, D] bf16 attention output
    """
    _validate_tq4_inputs(query, k_packed, v_packed)

    B, H_Q, N_Q, D = query.shape
    _, H_KV, N_KV, HALF_D = k_packed.shape

    _validate_tq4_mask(attn_mask, B, N_Q, N_KV)

    sm_scale = float(1.0 / math.sqrt(D)) if scale is None else float(scale)
    num_groups = H_Q // H_KV

    HAS_KV_LEN = kv_len is not None
    if HAS_KV_LEN:
        # Device int32 scalar, clamped to the buffer size for OOB safety.
        # Reshaped to [1] so the kernel can ``tl.load`` element 0. No
        # ``.item()`` — keeps it CUDA-graph-safe (value updates on replay).
        kv_len_t = torch.clamp(
            kv_len.reshape(1).to(torch.int32), max=int(N_KV)
        ).contiguous()
    else:
        kv_len_t = None

    # The per-tile causal upper bound needs kv_len to convert query-row indices
    # to absolute KV positions, so it is only meaningful when kv_len is supplied.
    MASK_IS_CAUSAL = bool(mask_is_causal) and HAS_KV_LEN

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

    # Split-K decode dispatch: L_q == 1 AND kv_len >= threshold (256)
    # Uses flash-decoding to partition KV across many CTAs for better occupancy
    # Dispatch is static (based on buffer size N_KV, not runtime kv_len value)
    # to be export/AOTI traceable. The kernel still uses kv_len on-device
    # via tl.load for bounds checking (CUDA-graph safe).
    _SPLITK_LKV_THRESHOLD = 256
    use_splitk = (
        N_Q == 1
        and HAS_KV_LEN
        and kv_len_t is not None
        and N_KV >= _SPLITK_LKV_THRESHOLD
    )

    if use_splitk:
        _launch_tq4_decode_splitk(
            q_rot,
            k_packed,
            k_n,
            v_packed,
            v_n,
            lut_hi,
            lut_lo,
            mask_ptr,
            out_rot,
            kv_len_t,
            B,
            H_Q,
            H_KV,
            N_Q,
            N_KV,
            D,
            sm_scale,
            HAS_MASK,
            HAS_KV_LEN,
            stride_mb,
            stride_mq,
            stride_mk,
            num_groups,
            pack_gqa,
        )
    else:
        # Prefill path (N_Q > 1, plus the rare N_Q==1 && N_KV<256 fallthrough).
        # When the caller guarantees a standard causal mask AND kv_len is known
        # (MASK_IS_CAUSAL), use the kernel's built-in absolute causal-offset and
        # skip loading the explicit mask — identical result, less HBM traffic.
        # Otherwise honor the explicit mask / is_causal for an arbitrary mask.
        if MASK_IS_CAUSAL:
            prefill_has_mask = False
            prefill_is_causal = True
        else:
            prefill_has_mask = HAS_MASK
            prefill_is_causal = is_causal
        _launch_tq4_prefill(
            q_rot,
            k_packed,
            k_n,
            v_packed,
            v_n,
            lut_hi,
            lut_lo,
            mask_ptr,
            out_rot,
            kv_len_t,
            B,
            H_Q,
            H_KV,
            N_Q,
            N_KV,
            D,
            sm_scale,
            prefill_has_mask,
            HAS_KV_LEN,
            False,  # MASK_IS_CAUSAL: causal handled via is_causal (causal-offset)
            stride_mb,
            stride_mq,
            stride_mk,
            prefill_is_causal,
            num_groups,
            pack_gqa,
        )

    # Post-rotate: convert from rotated space back to original space
    return torch.matmul(out_rot, rotation.to(query.dtype))


# ==============================================================================
# Split-K decode kernel (flash-decoding) for TQ4
# ==============================================================================
# When L_q == 1 with GQA, the standard kernel launches only
# ceil(num_groups / BLOCK_M) * B * H_kv CTAs (e.g. 4 for a B=1, 8:4 GQA shape).
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
    ],
    key=["Lk", "HEAD_DIM", "NUM_GROUPS", "HAS_MASK", "PACK_GQA"],
)
@triton.jit
def _tq4_sdpa_decode_splitk_kernel(
    Q_ptr,
    KP_ptr,
    KN_ptr,
    VP_ptr,
    VN_ptr,
    LUT_hi_ptr,
    LUT_lo_ptr,
    O_partial_ptr,
    M_partial_ptr,
    L_partial_ptr,
    Mask_ptr,
    KV_LEN_ptr,
    B,
    H_grid,
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
    stride_op_s,
    stride_op_b,
    stride_op_h,
    stride_op_d,
    stride_mp_s,
    stride_mp_b,
    stride_mp_h,
    stride_lp_s,
    stride_lp_b,
    stride_lp_h,
    stride_mb,
    stride_mq,
    stride_mk,
    sm_scale: tl.float32,
    chunk_size,
    HAS_MASK: tl.constexpr,
    HAS_KV_LEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_D: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    PACK_GQA: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    split_id = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)
    b = pid_bh // H_grid
    h_grid = pid_bh % H_grid

    # Compute KV chunk bounds for this split
    start_n = split_id * chunk_size
    if HAS_KV_LEN:
        kv_len = tl.load(KV_LEN_ptr)
    else:
        kv_len = Lk
    end_n = tl.minimum(start_n + chunk_size, kv_len)

    offs_d = tl.arange(0, HEAD_DIM)
    offs_d_half = tl.arange(0, HALF_D)
    offs_m = tl.arange(0, BLOCK_M)

    if PACK_GQA:
        # Pack GQA: multiple Q heads folded into M dimension
        seq_pos = offs_m // NUM_GROUPS
        h_within = offs_m % NUM_GROUPS
        h_q_rows = h_grid * NUM_GROUPS + h_within
        h_kv = h_grid
        row_valid = seq_pos < 1  # Lq=1 for decode
        q_ptrs = Q_ptr + (
            b * stride_qb
            + h_q_rows[:, None] * stride_qh
            + seq_pos[:, None] * stride_qm
            + offs_d[None, :] * stride_qd
        )
    else:
        seq_pos = offs_m
        h_kv = h_grid // NUM_GROUPS
        row_valid = offs_m < 1
        q_ptrs = Q_ptr + (
            b * stride_qb
            + h_grid * stride_qh
            + offs_m[:, None] * stride_qm
            + offs_d[None, :] * stride_qd
        )

    q = tl.load(q_ptrs, mask=row_valid[:, None], other=0.0).to(tl.bfloat16)

    # Online softmax state
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Prescale for exp2-based softmax
    qk_scale = sm_scale * 1.44269504

    # TQ4 K/V base pointers (uniform: single KV head)
    kp_base = KP_ptr + b * stride_kpb + h_kv * stride_kph
    kn_base = KN_ptr + b * stride_knb + h_kv * stride_knh
    vp_base = VP_ptr + b * stride_vpb + h_kv * stride_vph
    vn_base = VN_ptr + b * stride_vnb + h_kv * stride_vnh

    offs_n_init = tl.arange(0, BLOCK_N)

    for tile_start in tl.range(start_n, end_n, BLOCK_N):
        offs_n = tile_start + offs_n_init
        kv_valid = offs_n < end_n

        # -- K decompression (LUT, no norm multiply on [N,D] tile) --
        kp_ptrs = (
            kp_base + offs_n[:, None] * stride_kpn + offs_d_half[None, :] * stride_kpd
        )
        k_packed_data = tl.load(kp_ptrs, mask=kv_valid[:, None], other=0).to(tl.int32)
        k = tl.join(
            tl.load(LUT_hi_ptr + k_packed_data),
            tl.load(LUT_lo_ptr + k_packed_data),
        ).reshape(BLOCK_N, HEAD_DIM)

        # Q @ K^T with norm factored out
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

        qk = tl.where(kv_valid[None, :], qk, float("-inf"))

        # NaN-safe online softmax (exp2)
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
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

    # Store partial results
    if PACK_GQA:
        h_q_all = h_grid * NUM_GROUPS + h_within
        o_ptrs = O_partial_ptr + (
            split_id * stride_op_s
            + b * stride_op_b
            + h_q_all[:, None] * stride_op_h
            + offs_d[None, :] * stride_op_d
        )
        m_ptrs = M_partial_ptr + (
            split_id * stride_mp_s + b * stride_mp_b + h_q_all * stride_mp_h
        )
        l_ptrs = L_partial_ptr + (
            split_id * stride_lp_s + b * stride_lp_b + h_q_all * stride_lp_h
        )
        tl.store(o_ptrs, acc, mask=row_valid[:, None])
        tl.store(m_ptrs, m_i, mask=row_valid)
        tl.store(l_ptrs, l_i, mask=row_valid)
    else:
        o_ptrs = O_partial_ptr + (
            split_id * stride_op_s
            + b * stride_op_b
            + h_grid * stride_op_h
            + offs_d * stride_op_d
        )
        m_ptrs = M_partial_ptr + (
            split_id * stride_mp_s + b * stride_mp_b + h_grid * stride_mp_h
        )
        l_ptrs = L_partial_ptr + (
            split_id * stride_lp_s + b * stride_lp_b + h_grid * stride_lp_h
        )
        tl.store(o_ptrs, acc, mask=row_valid[:, None])
        tl.store(m_ptrs, m_i, mask=row_valid)
        tl.store(l_ptrs, l_i, mask=row_valid)


@triton.jit
def _tq4_sdpa_decode_reduce_kernel(
    O_partial_ptr,
    M_partial_ptr,
    L_partial_ptr,
    O_ptr,
    num_splits,
    stride_op_s,
    stride_op_b,
    stride_op_h,
    stride_op_d,
    stride_mp_s,
    stride_mp_b,
    stride_mp_h,
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

    # Find global max across splits
    m_global = tl.full([1], -float("inf"), dtype=tl.float32)
    for s in tl.range(0, num_splits):
        m_ptr = M_partial_ptr + s * stride_mp_s + pid * stride_mp_h
        m_s = tl.load(m_ptr)
        m_global = tl.maximum(m_global, m_s)

    # Rescale and sum partials
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    l_global = tl.zeros([1], dtype=tl.float32)

    for s in tl.range(0, num_splits):
        m_ptr = M_partial_ptr + s * stride_mp_s + pid * stride_mp_h
        l_ptr = L_partial_ptr + s * stride_lp_s + pid * stride_lp_h
        o_ptrs = O_partial_ptr + (
            s * stride_op_s + pid * stride_op_h + offs_d * stride_op_d
        )

        m_s = tl.load(m_ptr)
        l_s = tl.load(l_ptr)
        o_s = tl.load(o_ptrs)

        # Rescale by exp(m_s - m_global)
        alpha = tl.where(m_global > -float("inf"), tl.math.exp2(m_s - m_global), 0.0)
        acc += o_s * alpha
        l_global += l_s * alpha

    inv_l = tl.where(l_global > 0, 1.0 / l_global, 0.0)
    acc = acc * inv_l

    o_out_ptrs = O_ptr + pid * stride_oh + offs_d * stride_od
    tl.store(o_out_ptrs, acc.to(tl.bfloat16))


def _launch_tq4_decode_splitk(
    q_rot,
    k_packed,
    k_norms,
    v_packed,
    v_norms,
    lut_hi,
    lut_lo,
    mask_ptr,
    out_rot,
    kv_len_ptr,
    B,
    H_Q,
    H_KV,
    L_Q,
    L_KV,
    D,
    sm_scale,
    HAS_MASK,
    HAS_KV_LEN,
    stride_mb,
    stride_mq,
    stride_mk,
    num_groups,
    pack_gqa,
):
    HALF_D = D // 2

    if pack_gqa:
        H_grid = H_KV
        # Size BLOCK_M to the actually-packed rows (L_q * num_groups; for decode
        # L_q=1 so == num_groups), rounded up to the bf16 tensor-core MMA minimum
        # of 16 -- instead of a fixed 64. For example with num_groups=8, the
        # old BLOCK_M=64 left 56/64 M-rows idle (the QK/PV MMAs still computed all
        # 64 rows) AND made acc[BLOCK_M, HEAD_DIM] fp32 = 64*512*4 = 128 KB/CTA,
        # which blows past the register file and spills to local memory. Matching
        # BLOCK_M to num_groups removes both the wasted MMA rows and the spills.
        _packed_m = L_Q * num_groups
        BLOCK_M = max(16, 1 << (_packed_m - 1).bit_length())
    else:
        H_grid = H_Q
        BLOCK_M = 32

    # Static num_splits for CUDA-graph compatibility
    # Derive from buffer size, not runtime kv_len value
    num_splits = min(max(triton.cdiv(L_KV, 256), 1), 128)
    chunk_size = triton.cdiv(L_KV, num_splits)

    # Allocate partial result buffers
    O_partial = torch.empty(
        (num_splits, B, H_Q, D), device=q_rot.device, dtype=torch.float32
    )
    M_partial = torch.full(
        (num_splits, B, H_Q), -float("inf"), device=q_rot.device, dtype=torch.float32
    )
    L_partial = torch.zeros(
        (num_splits, B, H_Q), device=q_rot.device, dtype=torch.float32
    )

    stride_op_s, stride_op_b, stride_op_h, stride_op_d = O_partial.stride()
    stride_mp_s, stride_mp_b, stride_mp_h = M_partial.stride()
    stride_lp_s, stride_lp_b, stride_lp_h = L_partial.stride()

    grid = (num_splits, B * H_grid)
    wrap_triton(_tq4_sdpa_decode_splitk_kernel)[grid](
        q_rot,
        k_packed,
        k_norms,
        v_packed,
        v_norms,
        lut_hi,
        lut_lo,
        O_partial,
        M_partial,
        L_partial,
        mask_ptr if HAS_MASK else 0,
        kv_len_ptr if HAS_KV_LEN else 0,
        B,
        H_grid,
        L_KV,
        *q_rot.stride(),
        *k_packed.stride(),
        *k_norms.stride(),
        *v_packed.stride(),
        *v_norms.stride(),
        stride_op_s,
        stride_op_b,
        stride_op_h,
        stride_op_d,
        stride_mp_s,
        stride_mp_b,
        stride_mp_h,
        stride_lp_s,
        stride_lp_b,
        stride_lp_h,
        stride_mb,
        stride_mq,
        stride_mk,
        sm_scale,
        chunk_size,
        HAS_MASK=HAS_MASK,
        HAS_KV_LEN=HAS_KV_LEN,
        HEAD_DIM=D,
        HALF_D=HALF_D,
        NUM_GROUPS=num_groups,
        PACK_GQA=pack_gqa,
        BLOCK_M=BLOCK_M,
    )

    # Reduce partials
    grid_reduce = (B * H_Q,)
    wrap_triton(_tq4_sdpa_decode_reduce_kernel)[grid_reduce](
        O_partial,
        M_partial,
        L_partial,
        out_rot,
        num_splits,
        stride_op_s,
        stride_op_b,
        stride_op_h,
        stride_op_d,
        stride_mp_s,
        stride_mp_b,
        stride_mp_h,
        stride_lp_s,
        stride_lp_b,
        stride_lp_h,
        *out_rot.stride(),
        HEAD_DIM=D,
        num_warps=4,
        num_stages=1,
    )


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
    scale: Optional[float] = None,
    kv_len: Optional[torch.Tensor] = None,
    mask_is_causal: bool = False,
) -> torch.Tensor:
    return torch.empty_like(query)

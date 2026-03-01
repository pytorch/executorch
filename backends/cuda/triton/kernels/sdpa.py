# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Triton SDPA Kernel for ExecuTorch CUDA Backend.

This module provides a Triton-optimized implementation of scaled dot-product attention
that can replace the default ATen/Edge SDPA operator during graph transformation to allow
us export the model without decomposing the SDPA operator under libtorch free environment
and have better performance.
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
    """Get the next power of 2 >= x, clamped to [16, 256]."""
    if x <= 16:
        return 16
    if x <= 32:
        return 32
    if x <= 64:
        return 64
    if x <= 128:
        return 128
    return 256


def _validate_qkv_shapes(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[int, int, int, int, int, int]:
    """
    Validate dimensions and return shape info.
    Args:
        query: Query tensor [B, H, L_q, D]
        key: Key tensor [B, H, L_kv, D]
        value: Value tensor [B, H, L_kv, D]
    Returns:
        Tuple of (B, H, L_q, L_kv, D_q, D_kv)
    Raises:
        RuntimeError: If dimensions are incompatible
    """
    B_q, H_q, L_q, D_q = query.shape
    B_k, H_k, L_kv_k, D_k = key.shape
    B_v, H_v, L_kv_v, D_v = value.shape
    # Validate batch and head dimensions
    if not (B_q == B_k == B_v):
        raise RuntimeError(
            f"Batch dimension must match; got B_q={B_q}, B_k={B_k}, B_v={B_v}."
        )

    if not (H_q == H_k == H_v):
        raise RuntimeError(
            f"Head dimension must match; got H_q={H_q}, H_k={H_k}, H_v={H_v}."
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
    return B_q, H_q, L_q, L_kv_k, D_q, D_k


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
    H,
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
):
    """
    SDPA forward kernel for non-power-of-2 HEAD_DIM.
    Uses dynamic masking to handle arbitrary head dimensions.
    """
    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)

    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    d_mask = offs_d < HEAD_DIM
    q_row_mask = offs_m < LQ

    q_base = q_ptr + b * stride_qb + h * stride_qh
    k_base = k_ptr + b * stride_kb + h * stride_kh
    v_base = v_ptr + b * stride_vb + h * stride_vh
    o_base = o_ptr + b * stride_ob + h * stride_oh

    q_ptrs = q_base + (offs_m[:, None] * stride_ql + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=q_row_mask[:, None] & d_mask[None, :], other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)

    qk_scale_log2 = scale * 1.4426950408889634

    if HAS_MASK:
        mask_b_base = mask_ptr + b * stride_mb

    for start_n in tl.range(0, LK, BLOCK_N, num_stages=2):
        kn = start_n + offs_n
        kv_col_mask = kn < LK

        k_ptrs = k_base + (kn[:, None] * stride_kl + offs_d[None, :] * stride_kd)
        k = tl.load(k_ptrs, mask=kv_col_mask[:, None] & d_mask[None, :], other=0.0)

        qk = tl.dot(q, tl.trans(k))
        qk = qk * qk_scale_log2

        if IS_CAUSAL:
            row_abs = offs_m[:, None]
            col_abs = kn[None, :]
            causal_mask = col_abs > row_abs
            qk = tl.where(causal_mask, -float("inf"), qk)

        if HAS_MASK:
            mask_ptrs = (
                mask_b_base + offs_m[:, None] * stride_mlq + kn[None, :] * stride_mlk
            )
            tile_valid = q_row_mask[:, None] & kv_col_mask[None, :]
            keep = tl.load(mask_ptrs, mask=tile_valid, other=True)
            qk = tl.where(keep, qk, -float("inf"))

        qk = tl.where(kv_col_mask[None, :], qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)

        acc = acc * alpha[:, None]

        v_ptrs = v_base + (kn[:, None] * stride_vl + offs_d[None, :] * stride_vd)
        v = tl.load(v_ptrs, mask=kv_col_mask[:, None] & d_mask[None, :], other=0.0)

        acc = tl.dot(p.to(v.dtype), v, acc)

        l_i = l_i * alpha + l_ij
        m_i = m_ij

    out = acc / l_i[:, None]
    o_ptrs = o_base + (offs_m[:, None] * stride_ol + offs_d[None, :] * stride_od)
    tl.store(o_ptrs, out.to(tl.bfloat16), mask=q_row_mask[:, None] & d_mask[None, :])


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
    H,
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
):
    """
    Shared kernel body for SDPA forward pass.
    """
    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)
    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_init = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = Q_ptr + (
        b * stride_qb
        + h * stride_qh
        + (offs_m[:, None] * stride_qm)
        + (offs_d[None, :] * stride_qd)
    )
    q_mask = (offs_m[:, None] < Lq) & (offs_d[None, :] < HEAD_DIM)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.bfloat16)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for start_n in tl.range(0, Lk, BLOCK_N):
        offs_n = start_n + offs_n_init

        k_ptrs = K_ptr + (
            b * stride_kb
            + h * stride_kh
            + (offs_n[:, None] * stride_kn)
            + (offs_d[None, :] * stride_kd)
        )
        k_mask = (offs_n[:, None] < Lk) & (offs_d[None, :] < HEAD_DIM)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.bfloat16)

        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * sm_scale

        if HAS_MASK:
            mask_ptrs = Mask_ptr + (
                b * stride_mb
                + (offs_m[:, None] * stride_mq)
                + (offs_n[None, :] * stride_mk)
            )
            mn_mask = (offs_m[:, None] < Lq) & (offs_n[None, :] < Lk)
            mask_block = tl.load(mask_ptrs, mask=mn_mask, other=False)
            qk = tl.where(mask_block, qk, -float("inf"))

        if IS_CAUSAL:
            abs_m = offs_m[:, None]
            abs_n = offs_n[None, :]
            causal = abs_n > abs_m
            qk = tl.where(causal, -float("inf"), qk)

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p_f32 = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p_f32, axis=1)
        alpha = tl.exp(m_i - m_ij)

        v_ptrs = V_ptr + (
            b * stride_vb
            + h * stride_vh
            + (offs_n[:, None] * stride_vn)
            + (offs_d[None, :] * stride_vd)
        )
        v_mask = (offs_n[:, None] < Lk) & (offs_d[None, :] < HEAD_DIM)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.bfloat16)

        p_bf16 = p_f32.to(tl.bfloat16)
        acc = acc * alpha[:, None] + tl.dot(p_bf16, v)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    inv_l_i = tl.where(l_i > 0, 1.0 / l_i, 0.0)
    acc = acc * inv_l_i[:, None]

    o_ptrs = O_ptr + (
        b * stride_ob
        + h * stride_oh
        + (offs_m[:, None] * stride_om)
        + (offs_d[None, :] * stride_od)
    )
    o_mask = (offs_m[:, None] < Lq) & (offs_d[None, :] < HEAD_DIM)
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=o_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    ],
    key=["Lq", "Lk", "HEAD_DIM", "HAS_MASK", "IS_CAUSAL"],
)
@triton.jit
def _sdpa_fwd_kernel_m64(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    Mask_ptr,
    B,
    H,
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    SDPA kernel with BLOCK_M=64 optimizations.
    """
    _sdpa_fwd_kernel_body(
        Q_ptr,
        K_ptr,
        V_ptr,
        O_ptr,
        Mask_ptr,
        B,
        H,
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
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    ],
    key=["Lq", "Lk", "HEAD_DIM", "HAS_MASK", "IS_CAUSAL"],
)
@triton.jit
def _sdpa_fwd_kernel_m32(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    Mask_ptr,
    B,
    H,
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    SDPA kernel with BLOCK_M=32 optimizations for small workloads.
    """
    _sdpa_fwd_kernel_body(
        Q_ptr,
        K_ptr,
        V_ptr,
        O_ptr,
        Mask_ptr,
        B,
        H,
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
    if enable_gqa is not False:
        raise RuntimeError(
            "enable_gqa must be False (not supported in this implementation)."
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
    if (
        attn_mask.shape[0] != B
        or attn_mask.shape[2] != L_q
        or attn_mask.shape[3] != L_kv
    ):
        raise RuntimeError(
            f"attn_mask shape mismatch: expected [B={B}, H, L_q={L_q}, L_kv={L_kv}], "
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
    H: int,
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
) -> None:
    """Launch power-of-2 optimized SDPA kernel."""
    stride_qb, stride_qh, stride_qm, stride_qd = query.stride()
    stride_kb, stride_kh, stride_kn, stride_kd = key.stride()
    stride_vb, stride_vh, stride_vn, stride_vd = value.stride()
    stride_ob, stride_oh, stride_om, stride_od = out.stride()

    def grid(meta):
        return (triton.cdiv(L_q, meta["BLOCK_M"]), B * H)

    total_ctas_m64 = ((L_q + 63) // 64) * (B * H)
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
        H,
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
    )


def _launch_non_pow2_kernel(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    B: int,
    H: int,
    L_q: int,
    L_kv: int,
    D: int,
    sm_scale: float,
    HAS_MASK: bool,
    is_causal: bool,
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

    if HAS_MASK:
        mask_ptr = attn_mask
        stride_mb_np2 = attn_mask.stride(0)
        stride_mh_np2 = attn_mask.stride(1)
        stride_mlq_np2 = attn_mask.stride(2)
        stride_mlk_np2 = attn_mask.stride(3)
    else:
        mask_ptr = torch.empty((1,), device=query.device, dtype=torch.bool)
        stride_mb_np2 = stride_mh_np2 = stride_mlq_np2 = stride_mlk_np2 = 0

    def grid_non_pow2(meta):
        return (triton.cdiv(L_q, meta["BLOCK_M"]), B * H)

    wrap_triton(_sdpa_fwd_kernel_non_pow2)[grid_non_pow2](
        query,
        key,
        value,
        out,
        mask_ptr,
        B,
        H,
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
    Triton fused Scaled Dot-Product Attention with optimized dual-kernel approach.

    Args:
        query: Query tensor with size [B, H, L_q, D] and dtype torch.bfloat16
        key: Key tensor [B, H, L_kv, D] and dtype torch.bfloat16
        value: Value tensor [B, H, L_kv, D] and dtype torch.bfloat16
        attn_mask: Optional attention mask [B, H, L_q, L_kv] with dtype torch.bool
        dropout_p: must be 0.0 (others are not supported)
        is_causal: whether to apply causal masking
        scale: attention scale (default: 1/sqrt(D))
        enable_gqa: must be False (True is not supported)
    Returns:
        Output tensor [B, H, L_q, D] with dtype torch.bfloat16
    """
    _validate_sdpa_inputs(query, key, value, dropout_p, enable_gqa)

    B, H, L_q, L_kv, D_q, _ = _validate_qkv_shapes(query, key, value)
    D = D_q

    if is_causal and L_q != L_kv:
        raise RuntimeError(
            f"Causal masking requires L_q == L_kv; got L_q={L_q}, L_kv={L_kv}."
        )

    out = torch.empty((B, H, L_q, D), device=query.device, dtype=query.dtype)
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
            H,
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
        )
    else:
        _launch_non_pow2_kernel(
            query,
            key,
            value,
            out,
            attn_mask,
            B,
            H,
            L_q,
            L_kv,
            D,
            sm_scale,
            HAS_MASK,
            is_causal,
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
    enable_gq: bool = False,
) -> torch.Tensor:
    """
    Abstract/fake implementation for torch.export.
    This just returns an empty tensor with the correct shape/dtype/device.
    """
    # Validate dtypes match
    assert query.dtype == key.dtype == value.dtype, "Q, K, V must have the same dtype"
    # Validate kqv's shape and get the output shape
    B, H, L_q, _, D_q, _ = _validate_qkv_shapes(query, key, value)

    return torch.empty(B, H, L_q, D_q, dtype=query.dtype, device=query.device)

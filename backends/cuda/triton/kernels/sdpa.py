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


AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["HEAD_DIM", "HAS_MASK", "MASK_IS_BOOL"])
@triton.jit
def _sdpa_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    B,
    H,
    LQ,
    LK,
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
    scale,
    mask_ptr,
    stride_mb,
    stride_mh,
    stride_mm,
    stride_mn,
    HAS_MASK: tl.constexpr,
    MASK_IS_BOOL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid_m_in = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)
    b = pid_bh // H
    h = pid_bh % H

    num_pid_m = tl.cdiv(LQ, BLOCK_M)
    group_id = pid_m_in // GROUP_M
    first_pid_m = group_id * GROUP_M
    pid_m = first_pid_m + (pid_m_in + pid_bh) % GROUP_M
    start_m = pid_m * BLOCK_M
    if start_m >= LQ:
        return

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_m = tl.multiple_of(offs_m, BLOCK_M)
    offs_d = tl.multiple_of(offs_d, 16)
    offs_d = tl.max_contiguous(offs_d, 16)

    q_ptrs = Q_ptr + (
        b * stride_qb
        + h * stride_qh
        + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    )
    q_mask = offs_m[:, None] < LQ
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    log2e = 1.4426950408889634
    scale_log2 = scale * log2e

    for start_n in tl.range(0, LK, BLOCK_N):
        n_ids = start_n + offs_n
        n_mask = n_ids < LK

        k_ptrs = K_ptr + (
            b * stride_kb
            + h * stride_kh
            + (offs_d[:, None] * stride_kd + n_ids[None, :] * stride_kn)
        )
        k = tl.load(k_ptrs, mask=n_mask[None, :], other=0.0)

        qk = tl.dot(q, k).to(tl.float32)
        qk = qk * scale_log2

        if HAS_MASK:
            m_ptrs = mask_ptr + (
                b * stride_mb
                + h * stride_mh
                + (offs_m[:, None] * stride_mm + n_ids[None, :] * stride_mn)
            )
            valid = (offs_m[:, None] < LQ) & n_mask[None, :]
            if MASK_IS_BOOL:
                m_bool = tl.load(m_ptrs, mask=valid, other=True)
                qk = tl.where(m_bool, qk, -float("inf"))
            else:
                m_add = tl.load(m_ptrs, mask=valid, other=0.0).to(tl.float32)
                qk = qk + m_add * log2e

        qk = tl.where(n_mask[None, :], qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        alpha = tl.math.exp2(m_i - m_ij)

        v_ptrs = V_ptr + (
            b * stride_vb
            + h * stride_vh
            + (n_ids[:, None] * stride_vn + offs_d[None, :] * stride_vd)
        )
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)

        l_i = l_i * alpha + l_ij
        m_i = m_ij

    row_mask = offs_m < LQ
    l_i = tl.where(row_mask, l_i, 1.0)
    out = acc / l_i[:, None]

    o_ptrs = O_ptr + (
        b * stride_ob
        + h * stride_oh
        + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
    )
    tl.store(o_ptrs, out.to(tl.bfloat16), mask=row_mask[:, None])


def _check_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    dropout_p: float,
    is_causal: bool,
    enable_gqa: bool,
):
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        raise ValueError("query, key, value must be CUDA tensors.")
    if (
        query.dtype != torch.bfloat16
        or key.dtype != torch.bfloat16
        or value.dtype != torch.bfloat16
    ):
        raise ValueError("This kernel expects bfloat16 inputs for query, key, value.")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, value must be 4D tensors [B, H, L, D].")
    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        raise ValueError("Batch dimension mismatch.")
    if query.shape[1] != key.shape[1] or query.shape[1] != value.shape[1]:
        raise ValueError("Heads dimension mismatch.")
    if query.shape[-1] != key.shape[-1] or query.shape[-1] != value.shape[-1]:
        raise ValueError("Head dimension (D) mismatch across query, key, value.")
    if attn_mask is not None:
        if attn_mask.ndim != 4:
            raise ValueError("attn_mask must be 4D [B, H, L_q, L_k].")
        if attn_mask.shape[0] != query.shape[0] or attn_mask.shape[1] != query.shape[1]:
            raise ValueError("attn_mask batch/head dims must match query.")
        if attn_mask.shape[2] != query.shape[2] or attn_mask.shape[3] != key.shape[2]:
            raise ValueError("attn_mask spatial dims must be [L_q, L_k].")
        if attn_mask.dtype not in (torch.bool, torch.bfloat16):
            raise ValueError("attn_mask must be dtype bool or bfloat16.")

    # Enforce unsupported features
    if dropout_p != 0.0:
        raise RuntimeError(
            "dropout_p must be 0.0 (not supported in this implementation)."
        )
    if enable_gqa is not False:
        raise RuntimeError(
            "enable_gqa must be False (not supported in this implementation)."
        )
    if is_causal:
        raise RuntimeError(
            "is_causal must be False (not supported in this implementation)."
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
    if attn_mask is not None and attn_mask.shape[1] == 1:
        attn_mask = attn_mask.expand(-1, query.shape[1], -1, -1).contiguous()

    _check_inputs(query, key, value, attn_mask, dropout_p, is_causal, enable_gqa)

    B, H, LQ, D = query.shape
    LK = key.shape[2]

    out = torch.empty_like(query)

    if scale == 0:
        scale = 1.0 / math.sqrt(D)
    scale = float(scale)

    stride_qb, stride_qh, stride_qm, stride_qd = query.stride()
    stride_kb, stride_kh, stride_kn, stride_kd = key.stride()
    stride_vb, stride_vh, stride_vn, stride_vd = value.stride()
    stride_ob, stride_oh, stride_om, stride_od = out.stride()

    HAS_MASK = 1 if attn_mask is not None else 0
    MASK_IS_BOOL = 1 if (attn_mask is not None and attn_mask.dtype == torch.bool) else 0
    if attn_mask is None:
        mask_ptr = query
        stride_mb = stride_mh = stride_mm = stride_mn = 0
    else:
        mask_ptr = attn_mask
        stride_mb, stride_mh, stride_mm, stride_mn = attn_mask.stride()

    def grid(meta):
        return (triton.cdiv(LQ, meta["BLOCK_M"]), B * H)

    wrap_triton(_sdpa_fwd_kernel)[grid](
        query,
        key,
        value,
        out,
        B,
        H,
        LQ,
        LK,
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
        scale,
        mask_ptr,
        stride_mb,
        stride_mh,
        stride_mm,
        stride_mn,
        HAS_MASK=HAS_MASK,
        MASK_IS_BOOL=MASK_IS_BOOL,
        HEAD_DIM=D,
        GROUP_M=8,
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
    assert (
        query.dtype == key.dtype == value.dtype
    ), "query, key, value must have the same dtype"
    # Validate kqv's shape and get the output shape
    B, H, LQ, D = query.shape

    return torch.empty(B, H, LQ, D, dtype=query.dtype, device=query.device)

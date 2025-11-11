import math
from typing import Any, Optional

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.autotune(
    configs=[
        # Favor configs tuned for HEAD_DIM=64 and L up to ~1500
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=3, num_warps=4),
    ],
    key=["L", "HEAD_DIM"],
)
@triton.jit
def _sdpa_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    B,
    H,
    L,
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
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM_CE: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(axis=0)  # along query length
    pid_hz = tl.program_id(axis=1)  # flattened batch*head

    off_b = pid_hz // H
    off_h = pid_hz % H

    # Compute ranges
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM_CE)
    mask_m = offs_m < L

    # Base pointers for this (b, h)
    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    k_base = k_ptr + off_b * stride_kb + off_h * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_h * stride_vh
    o_base = o_ptr + off_b * stride_ob + off_h * stride_oh

    # Make head-dim addresses compiler-friendly
    offs_d_ctg = tl.max_contiguous(tl.multiple_of(offs_d, 16), HEAD_DIM_CE)

    # Load Q tile [BLOCK_M, HEAD_DIM] - coalesced along HEAD_DIM
    q_ptrs = q_base + (offs_m[:, None] * stride_ql + offs_d_ctg[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    q = q.to(tl.bfloat16)

    # Initialize accumulators and softmax stats
    acc = tl.zeros((BLOCK_M, HEAD_DIM_CE), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Convert to base-2 scale for exp2
    qk_scale = sm_scale * 1.4426950408889634

    # Loop over keys/values along sequence length in tiles of BLOCK_N
    # Load K as [BLOCK_N, HEAD_DIM] for coalesced reads, then use tl.trans(K) in dot
    for start_n in tl.range(0, L, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < L

        # Load K tile [BLOCK_N, HEAD_DIM] (contiguous along HEAD_DIM)
        k_ptrs = k_base + (
            offs_n[:, None] * stride_kl + offs_d_ctg[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        k = k.to(tl.bfloat16)

        # Compute attention logits [BLOCK_M, BLOCK_N] = Q[BM,D] @ K[BN,D]^T
        qk = tl.dot(q, tl.trans(k)).to(tl.float32)  # accumulator in fp32
        qk = qk * qk_scale

        # Apply OOB masks for both rows and cols to keep stability
        qk = tl.where(mask_n[None, :], qk, -float("inf"))
        qk = tl.where(mask_m[:, None], qk, -float("inf"))

        # Online softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)

        # Load V tile [BLOCK_N, HEAD_DIM] (contiguous along HEAD_DIM)
        v_ptrs = v_base + (
            offs_n[:, None] * stride_vl + offs_d_ctg[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        v = v.to(tl.bfloat16)

        # Update accumulator
        acc = acc * alpha[:, None]
        # Cast p to bf16 to use tensor-cores in tl.dot; accumulate in fp32
        p_bf16 = p.to(tl.bfloat16)
        acc = tl.dot(p_bf16, v, acc)

        # Update softmax stats
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    # Normalize accumulator by softmax denominator
    acc = acc / l_i[:, None]

    # Store output [BLOCK_M, HEAD_DIM]
    o_ptrs = o_base + (offs_m[:, None] * stride_ol + offs_d_ctg[None, :] * stride_od)
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None])


@triton_op("custom::optimized_triton_scaled_dot_product_attention", mutates_args={})
def optimized_triton_scaled_dot_product_attention(
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
    Triton fused Scaled Dot-Product Attention (forward, no causal, no dropout).
    Expected shapes (tested): [B=1, H=20, L<=1500, D=64], dtype bfloat16.

    Args:
        query: Query tensor [B, H, L, D]
        key: Key tensor [B, H, L, D]
        value: Value tensor [B, H, L, D]
        attn_mask: must be None (not supported)
        dropout_p: must be 0.0 (not supported)
        is_causal: must be False (not supported)
        scale: must be 0.0 (not supported)
        enable_gqa: must be False (not supported)

    Returns:
        Output tensor [B, H, L, D]
    """
    # Validate inputs
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        raise RuntimeError("Q, K, V must be CUDA tensors.")
    if (
        query.dtype != torch.bfloat16
        or key.dtype != torch.bfloat16
        or value.dtype != torch.bfloat16
    ):
        raise RuntimeError("Expected bfloat16 inputs")
    if query.shape != key.shape or query.shape != value.shape:
        raise RuntimeError(
            f"Q, K, V must have identical shapes; got query={query.shape}, key={key.shape}, value={value.shape}."
        )
    if query.dim() != 4:
        raise RuntimeError(
            f"Expected 4D tensors shaped [B, H, L, D]; got {query.dim()}D."
        )

    # Enforce that only default values are accepted for these arguments
    if attn_mask is not None:
        raise RuntimeError(
            "attn_mask must be None (not supported in this implementation)."
        )
    if dropout_p != 0.0:
        raise RuntimeError(
            "dropout_p must be 0.0 (not supported in this implementation)."
        )
    if is_causal is not False:
        raise RuntimeError(
            "is_causal must be False (not supported in this implementation)."
        )
    if scale != 0.0:
        raise RuntimeError("scale must be 0.0 (not supported in this implementation).")
    if enable_gqa is not False:
        raise RuntimeError(
            "enable_gqa must be False (not supported in this implementation)."
        )

    B, H, L, D = query.shape
    # Allocate output
    out = torch.empty_like(query)

    # Element-wise strides (in elements)
    sqb, sqh, sql, sqd = query.stride()
    skb, skh, skl, skd = key.stride()
    svb, svh, svl, svd = value.stride()
    sob, soh, sol, sod = out.stride()

    # Grid: tile queries (M) and batch*heads axis
    def grid(META):
        return (
            triton.cdiv(L, META["BLOCK_M"]),
            B * H,
        )

    # Scale factor for SDPA
    sm_scale = 1.0 / math.sqrt(D)

    # Launch kernel using wrap_triton to avoid tracing issues during export/compile
    # Note: wrap_triton returns a callable that can be indexed with grid
    wrap_triton(_sdpa_fwd_kernel)[grid](
        query,
        key,
        value,
        out,
        B,
        H,
        L,
        D,
        sqb,
        sqh,
        sql,
        sqd,
        skb,
        skh,
        skl,
        skd,
        svb,
        svh,
        svl,
        svd,
        sob,
        soh,
        sol,
        sod,
        sm_scale,
        HEAD_DIM_CE=D,
    )

    return out


# Register the abstract/fake implementation for torch.export
# This is critical to avoid accessing real tensor data during export
@optimized_triton_scaled_dot_product_attention.register_fake
def _optimized_triton_sdpa_abstract(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale=None,
    enable_gqa=False,
) -> torch.Tensor:
    """
    Abstract/fake implementation for torch.export.
    This just returns an empty tensor with the correct shape/dtype/device.
    No actual computation happens here - this is only for shape inference during export.
    """
    # Validate shapes match
    assert query.shape == key.shape == value.shape, "Q, K, V must have the same shape"
    assert query.dtype == key.dtype == value.dtype, "Q, K, V must have the same dtype"

    # Output has the same shape and dtype as query
    # IMPORTANT: Use the exact same dtype to satisfy ExecuTorch validation
    return torch.empty_like(query, dtype=query.dtype, device=query.device)

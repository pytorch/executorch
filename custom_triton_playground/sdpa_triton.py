# kernel.py
import math
from typing import Any, Optional

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


"""
Fused Scaled Dot-Product Attention (SDPA) implemented in a single Triton kernel.

This module provides a transparent replacement for torch.nn.functional.scaled_dot_product_attention
using a custom Triton kernel. The replacement is automatic - no model code changes needed!

How it works:
1. We register a custom implementation using torch.library
2. When torch.nn.functional.scaled_dot_product_attention is called,
   PyTorch's dispatch mechanism routes it to our implementation during AOTI compilation
3. The model code remains unchanged

What is fused:
- We fuse QK^T matmul, numerically-stable online softmax, and the final
  multiplication by V into one streaming kernel. No intermediate attention
  matrix is materialized in memory.

Design notes:
- We tile along the query (sequence) dimension with BLOCK_M rows and iterate
  over the key/value sequence dimension in BLOCK_N columns.
- For each (batch, head) pair and query tile, we:
  * Load a tile of Q once and keep it in registers.
  * Stream over K/V in blocks: compute qk = Q @ K^T, update running row-wise
    softmax statistics (m_i, l_i) and the output accumulator acc = sum(p * V)
    using the "online softmax" algorithm:
       m_new = max(m_old, max(qk))
       p     = exp(qk - m_new)
       acc   = acc * exp(m_old - m_new) + p @ V
       l_new = l_old * exp(m_old - m_new) + sum(p)
       m_old = m_new
  * Finally, write O = acc / l_i.
- All accumulation is done in fp32 for numerical stability; inputs/outputs are fp16.
- Boundary conditions are handled with masks.
- The Python wrapper only validates inputs, allocates outputs, configures the grid,
  and launches the Triton kernel. All math is inside the Triton kernel.

Runtime constraints respected:
- No torch.nn or torch.nn.functional is used in the execution path.
- No PyTorch compute ops are used to implement the algorithm; all math happens
  in Triton via tl.load/tl.store/tl.dot/tl.exp/tl.max/tl.sum.
"""


@triton.jit
def _sdpa_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    B,
    H,
    S,
    D,  # shapes
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    scale,  # 1/sqrt(D)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)  # along sequence dimension (queries)
    pid_bh = tl.program_id(1)  # across batch*heads

    b = pid_bh // H
    h = pid_bh % H

    # Offsets for this block of queries
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # Base pointers for this (b, h)
    q_bh = q_ptr + b * stride_qb + h * stride_qh
    k_bh = k_ptr + b * stride_kb + h * stride_kh
    v_bh = v_ptr + b * stride_vb + h * stride_vh
    o_bh = o_ptr + b * stride_ob + h * stride_oh

    # Load Q tile: [BLOCK_M, HEAD_DIM]
    q_ptrs = q_bh + (offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd)
    q_mask = (offs_m[:, None] < S) & (offs_d[None, :] < D)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Initialize online-softmax stats and output accumulator
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Iterate over keys/values in blocks of BLOCK_N
    for start_n in tl.range(0, S, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_mask_cols = offs_n < S

        # Load K in a layout suitable for qk = q @ kT:
        # k_ptrs produces a tensor of shape [HEAD_DIM, BLOCK_N]
        k_ptrs = k_bh + (offs_n[None, :] * stride_ks + offs_d[:, None] * stride_kd)
        k = tl.load(
            k_ptrs, mask=(offs_d[:, None] < D) & (kv_mask_cols[None, :]), other=0.0
        )

        # qk = [BLOCK_M, BLOCK_N] in fp32
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, qk)
        qk = qk * scale  # scale by 1/sqrt(D)

        # Mask out-of-bounds columns so they don't affect max/sum
        qk = tl.where(kv_mask_cols[None, :], qk, -float("inf"))

        # Online softmax update
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])  # fp32
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_ij

        # Load V tile: [BLOCK_N, HEAD_DIM]
        v_ptrs = v_bh + (offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd)
        v = tl.load(
            v_ptrs, mask=(kv_mask_cols[:, None]) & (offs_d[None, :] < D), other=0.0
        )

        # Update output accumulator: acc = acc * alpha + p @ v
        acc = acc * alpha[:, None]
        # Use fp16 inputs for tl.dot with fp32 accumulation
        acc = tl.dot(p.to(tl.float16), v.to(tl.float16), acc)

    # Normalize: O = acc / l_i[:, None]
    o = acc / l_i[:, None]
    # Store O in fp16
    o_ptrs = o_bh + (offs_m[:, None] * stride_os + offs_d[None, :] * stride_od)
    o_mask = (offs_m[:, None] < S) & (offs_d[None, :] < D)
    tl.store(o_ptrs, o.to(tl.float16), mask=o_mask)


@triton_op("custom::scaled_dot_product_attention", mutates_args={})
def triton_scaled_dot_product_attention(
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
    Fused Scaled Dot-Product Attention registered as a custom op:
      O = softmax(Q @ K^T / sqrt(D)) @ V
    where Q, K, V are shaped [batch, heads, seq_len, head_dim].

    This function is registered with @triton_op so AOTI can discover and use it
    during compilation as a replacement for torch.nn.functional.scaled_dot_product_attention.

    Wrapper responsibilities:
    - Validate input tensors (dtype/device/shapes)
    - Allocate output tensor
    - Configure grid and launch the Triton kernel
    - No math is done here beyond basic scalar setup; all heavy compute runs in the Triton kernel.

    Fusion details:
    - This launches a single kernel that computes QK^T, performs online softmax,
      and multiplies by V to produce O, all in one pass over K/V blocks.
    - No intermediate attention matrix is written to global memory.

    Args:
        query: Query tensor [B, H, S, D]
        key: Key tensor [B, H, S, D]
        value: Value tensor [B, H, S, D]
        attn_mask: has to be None
        is_causal: has to be False
        scale: has to be None
        enable_gqa: has to be False

    Returns:
        Output tensor [B, H, S, D]
    """
    # Basic validation
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        raise RuntimeError("Q, K, V must be CUDA tensors.")
    if (
        query.dtype != torch.float16
        or key.dtype != torch.float16
        or value.dtype != torch.float16
    ):
        raise RuntimeError("This reference implementation expects float16 tensors.")
    if query.shape != key.shape or query.shape != value.shape:
        raise RuntimeError(
            f"Q, K, V must have identical shapes; got Q={query.shape}, K={key.shape}, V={value.shape}."
        )
    if query.dim() != 4:
        raise RuntimeError(
            f"Expected 4D tensors shaped [B, H, S, D]; got {query.dim()}D."
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
    if scale != 0:
        raise RuntimeError("scale must be None (not supported in this implementation).")
    if enable_gqa is not False:
        raise RuntimeError(
            "enable_gqa must be False (not supported in this implementation)."
        )

    B, H, S, D = query.shape

    # Allocate output
    O = torch.empty_like(query)

    # Choose tiling parameters (powers of two, coalesced-friendly)
    # Conservative sizes to keep register/SMEM pressure reasonable for D=1024
    BLOCK_M = 16
    BLOCK_N = 32

    # Compute softmax scale on host (scalar) - this is setup, not heavy math
    scale = 1.0 / math.sqrt(float(D))

    # Grid: one program per (query block, batch*head)
    grid = (triton.cdiv(S, BLOCK_M), B * H)

    # Launch kernel using wrap_triton to avoid tracing issues during export/compile
    # Note: wrap_triton returns a callable that can be indexed with grid
    wrap_triton(_sdpa_fwd_kernel)[grid](
        query,
        key,
        value,
        O,
        B,
        H,
        S,
        D,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        O.stride(0),
        O.stride(1),
        O.stride(2),
        O.stride(3),
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=D,
        num_warps=4,
        num_stages=2,
    )

    return O


# Register the abstract/fake implementation for torch.export
# This is critical to avoid accessing real tensor data during export
@triton_scaled_dot_product_attention.register_fake
def _triton_sdpa_abstract(
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

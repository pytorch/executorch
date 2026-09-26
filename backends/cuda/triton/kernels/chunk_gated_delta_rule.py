# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file wraps Triton kernels from flash-linear-attention (FLA),
# which is licensed under the MIT License:
#   Copyright (c) 2023-2025 Songlin Yang
#   https://github.com/fla-org/flash-linear-attention

"""
Register chunk_gated_delta_rule as a @triton_op for ExecuTorch CUDA backend.

Wraps Triton kernels from flash-linear-attention (FLA):
  https://github.com/fla-org/flash-linear-attention

Uses the same pattern as backends/cuda/triton/kernels/sdpa.py: all FLA Triton
kernels are launched via wrap_triton() so AOTInductor compiles them directly
into the generated .so (no C++ shim needed).

FLA kernels use @triton.heuristics which wrap_triton doesn't support directly.
We unwrap via kernel.fn to get the inner @triton.autotune kernel and pass the
heuristic-computed constexprs explicitly.

Requires: pip install flash-linear-attention
"""

import torch
import triton

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_kernel_h_blockdim64
from fla.ops.common.chunk_o import chunk_fwd_kernel_o
from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd_kernel
from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd_kernel
from fla.ops.utils.cumsum import chunk_local_cumsum_scalar_kernel
from fla.ops.utils.solve_tril import merge_16x16_to_64x64_inverse_kernel
from fla.utils import IS_TMA_SUPPORTED
from torch.library import triton_op, wrap_triton

CHUNK_SIZE = 64


def _unwrap(kernel):
    """Unwrap @triton.heuristics to get the inner Autotuner for wrap_triton."""
    if hasattr(kernel, "fn") and isinstance(
        kernel, triton.runtime.autotuner.Heuristics
    ):
        return kernel.fn
    return kernel


def _validate_inputs(q, k, v, g, beta, initial_state):
    B, T, H, K = q.shape
    V = v.shape[-1]
    if k.shape != (B, T, H, K):
        raise ValueError(f"k shape {k.shape} != expected {(B, T, H, K)}")
    if v.shape != (B, T, H, V):
        raise ValueError(f"v shape {v.shape} != expected {(B, T, H, V)}")
    if g.shape != (B, T, H):
        raise ValueError(f"g shape {g.shape} != expected {(B, T, H)}")
    if beta.shape != (B, T, H):
        raise ValueError(f"beta shape {beta.shape} != expected {(B, T, H)}")
    if initial_state.shape != (B, H, K, V):
        raise ValueError(
            f"initial_state shape {initial_state.shape} != expected {(B, H, K, V)}"
        )
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError("q, k, v must have the same dtype")
    if not (
        q.device
        == k.device
        == v.device
        == g.device
        == beta.device
        == initial_state.device
    ):
        raise ValueError("All tensors must be on the same device")
    if K > 256:
        raise ValueError(f"Head dim K={K} exceeds maximum 256")


@triton_op("triton::chunk_gated_delta_rule", mutates_args={})
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Chunked gated delta rule linear attention (forward only).

    Args:
        q: [B, T, H, K] queries (should be L2-normalized, K <= 256)
        k: [B, T, H, K] keys (should be L2-normalized)
        v: [B, T, H, V] values
        g: [B, T, H] gating in log space
        beta: [B, T, H] write strength
        initial_state: [B, H, K, V] initial hidden state

    Returns:
        o: [B, T, H, V] output
        final_state: [B, H, K, V] final hidden state
    """
    _validate_inputs(q, k, v, g, beta, initial_state)

    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = CHUNK_SIZE
    NT = triton.cdiv(T, BT)
    scale = K**-0.5

    # 1. chunk_local_cumsum: cumulative sum of g within each chunk
    g_cumsum = torch.empty(B, T, H, dtype=torch.float32, device=q.device)
    wrap_triton(_unwrap(chunk_local_cumsum_scalar_kernel))[(NT, B * H)](
        s=g,
        o=g_cumsum,
        scale=0,
        cu_seqlens=0,
        chunk_indices=0,
        T=T,
        B=B,
        H=H,
        BT=BT,
        HEAD_FIRST=False,
        REVERSE=False,
        HAS_SCALE=False,
        IS_VARLEN=False,
    )

    # 2. chunk_scaled_dot_kkt: compute beta * K * K^T with gating
    A = torch.empty(B, T, H, BT, device=q.device, dtype=torch.float32)
    wrap_triton(_unwrap(chunk_scaled_dot_kkt_fwd_kernel))[(NT, B * H)](
        k=k,
        g=g_cumsum,
        beta=beta,
        A=A,
        cu_seqlens=0,
        chunk_indices=0,
        T=T,
        H=H,
        K=K,
        BT=BT,
        USE_G=True,
        IS_VARLEN=False,
    )

    # 3. solve_tril: (I + A)^{-1} via block triangular solve
    # Output in k.dtype (not float32) to match FLA's solve_tril(output_dtype=k.dtype)
    Ai = torch.zeros_like(A, dtype=k.dtype)
    wrap_triton(_unwrap(merge_16x16_to_64x64_inverse_kernel))[NT, B * H](
        A=A,
        Ai=Ai,
        cu_seqlens=0,
        chunk_indices=0,
        T=T,
        H=H,
        BT=BT,
        USE_TMA=IS_TMA_SUPPORTED,
        IS_VARLEN=False,
    )

    # 4. recompute_w_u: WY representation
    w = torch.empty_like(k)
    u = torch.empty_like(v)
    wrap_triton(_unwrap(recompute_w_u_fwd_kernel))[(NT, B * H)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=Ai,
        g=g_cumsum,
        cu_seqlens=0,
        chunk_indices=0,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=64,
        BV=64,
        USE_G=True,
        IS_VARLEN=False,
    )

    # 5. chunk_gated_delta_rule_fwd_h: inter-chunk recurrence
    h = torch.empty(B, NT, H, K, V, dtype=q.dtype, device=q.device)
    final_state = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    v_new = torch.empty_like(v)

    def grid_h(meta):
        return (triton.cdiv(V, meta["BV"]), B * H)

    wrap_triton(_unwrap(chunk_gated_delta_rule_fwd_kernel_h_blockdim64))[grid_h](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g_cumsum,
        gk=0,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=0,
        chunk_offsets=0,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        USE_EXP2=False,
        TRANSPOSE_STATE=False,
        USE_G=True,
        USE_GK=False,
        USE_INITIAL_STATE=True,
        STORE_FINAL_STATE=True,
        SAVE_NEW_VALUE=True,
        IS_VARLEN=False,
    )

    # 6. chunk_fwd_o: output computation
    o = torch.empty_like(v)

    def grid_o(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * H)

    wrap_triton(_unwrap(chunk_fwd_kernel_o))[grid_o](
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g_cumsum,
        g_gamma=0,
        o=o,
        cu_seqlens=0,
        chunk_indices=0,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        TRANSPOSE_STATE=False,
        USE_G=True,
        USE_G_GAMMA=False,
        IS_VARLEN=False,
    )

    return o, final_state


@chunk_gated_delta_rule.register_fake
def _chunk_gated_delta_rule_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K = q.shape
    V = v.shape[-1]
    return (
        torch.empty(B, T, H, V, dtype=q.dtype, device=q.device),
        torch.empty(B, H, K, V, dtype=torch.float32, device=q.device),
    )

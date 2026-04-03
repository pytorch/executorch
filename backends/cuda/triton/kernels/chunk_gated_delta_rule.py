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

Runtime dispatch: when T=1 (decode), uses a fused recurrent kernel that is
~25% faster than the chunked path. When T>1 (prefill), uses the full chunked
FLA pipeline. Dispatch happens inside the triton_op Python wrapper, following
the same pattern as sdpa.py's pow2/non-pow2 dispatch.

Requires: pip install flash-linear-attention
"""

import torch
import triton
import triton.language as tl

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


# ---------------------------------------------------------------------------
# Recurrent kernel: fused single-step gated delta rule for T=1 decode.
#
# Each thread block handles one (batch, head) pair.
# Iterates over V-tiles: for each V-tile, loads the K-dim of state, q, k, v,
# computes the gated delta rule update and output dot product.
# ---------------------------------------------------------------------------


@triton.jit
def _recurrent_gated_delta_rule_kernel(
    # Pointers — all inputs [B, 1, H, *] squeezed to [B, H, *]
    q_ptr,      # [B, H, K]
    k_ptr,      # [B, H, K]
    v_ptr,      # [B, H, V]
    g_ptr,      # [B, H]
    beta_ptr,   # [B, H]
    state_ptr,  # [B, H, K, V] input state (read)
    o_ptr,      # [B, H, V] output
    new_state_ptr,  # [B, H, K, V] output state (write)
    # Dims
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,  # block size for K dimension
    BV: tl.constexpr,  # block size for V dimension
    scale: tl.constexpr,
):
    # One block per (batch, head)
    bh = tl.program_id(0)

    # Load scalar g and beta for this (b, h)
    g_val = tl.load(g_ptr + bh).to(tl.float32)
    beta_val = tl.load(beta_ptr + bh).to(tl.float32)
    decay = tl.exp(g_val)

    # Load full q and k vectors: [K]
    q_base = bh * K
    k_base = bh * K
    k_range = tl.arange(0, BK)
    k_mask = k_range < K

    q_vec = tl.load(q_ptr + q_base + k_range, mask=k_mask, other=0.0).to(tl.float32)
    k_vec = tl.load(k_ptr + k_base + k_range, mask=k_mask, other=0.0).to(tl.float32)

    # Process V in tiles
    v_base = bh * V
    state_base = bh * K * V  # state: [K, V] for this (b, h)

    for v_start in range(0, V, BV):
        v_range = v_start + tl.arange(0, BV)
        v_mask = v_range < V

        # Load v tile: [BV]
        v_tile = tl.load(v_ptr + v_base + v_range, mask=v_mask, other=0.0).to(
            tl.float32
        )

        # Load state tile: [BK, BV] — state[k, v] at state_base + k*V + v
        s_offsets = k_range[:, None] * V + v_range[None, :]
        s_mask = k_mask[:, None] & v_mask[None, :]
        state_tile = tl.load(
            state_ptr + state_base + s_offsets, mask=s_mask, other=0.0
        ).to(tl.float32)

        # Step 1: state *= exp(g)
        state_tile = state_tile * decay

        # Step 2: Sk = state^T @ k → [BV] (dot product along K)
        # Sk[v] = sum_k state[k, v] * k[k]
        Sk = tl.sum(state_tile * k_vec[:, None], axis=0)

        # Step 3: delta = v - Sk
        delta = v_tile - Sk

        # Step 4: state += beta * outer(k, delta)
        state_tile = state_tile + beta_val * (k_vec[:, None] * delta[None, :])

        # Step 5: o = state^T @ q → [BV]
        o_tile = tl.sum(state_tile * q_vec[:, None], axis=0) * scale

        # Store output tile
        tl.store(o_ptr + v_base + v_range, o_tile.to(o_ptr.dtype.element_ty), mask=v_mask)

        # Store new state tile
        tl.store(
            new_state_ptr + state_base + s_offsets,
            state_tile.to(new_state_ptr.dtype.element_ty),
            mask=s_mask,
        )


def _launch_recurrent(q, k, v, g, beta, initial_state, scale):
    """Launch the recurrent kernel for T=1 decode.

    Args:
        q, k, v: [B, 1, H, K/V] — single-step inputs
        g, beta: [B, 1, H] — gating and write strength
        initial_state: [B, H, K, V] — input hidden state

    Returns:
        o: [B, 1, H, V] — output
        final_state: [B, H, K, V] — updated hidden state (float32)
    """
    B, _, H, K = q.shape
    V = v.shape[-1]

    # Squeeze T=1 dim for kernel: [B, H, *]
    q_2d = q[:, 0].contiguous()  # [B, H, K]
    k_2d = k[:, 0].contiguous()  # [B, H, K]
    v_2d = v[:, 0].contiguous()  # [B, H, V]
    g_2d = g[:, 0].contiguous()  # [B, H]
    beta_2d = beta[:, 0].contiguous()  # [B, H]

    o_2d = torch.empty(B, H, V, device=q.device, dtype=q.dtype)
    final_state = torch.empty(B, H, K, V, device=q.device, dtype=torch.float32)

    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 128)  # cap tile size

    grid = (B * H,)
    wrap_triton(_recurrent_gated_delta_rule_kernel)[grid](
        q_ptr=q_2d,
        k_ptr=k_2d,
        v_ptr=v_2d,
        g_ptr=g_2d,
        beta_ptr=beta_2d,
        state_ptr=initial_state,
        o_ptr=o_2d,
        new_state_ptr=final_state,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        scale=scale,
    )

    # Unsqueeze back to [B, 1, H, V]
    o = o_2d.unsqueeze(1)
    return o, final_state


# ---------------------------------------------------------------------------
# Chunked kernel: full FLA pipeline for T>1 prefill.
# ---------------------------------------------------------------------------


def _launch_chunked(q, k, v, g, beta, initial_state, scale):
    """Launch the chunked FLA pipeline for T>1 prefill."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = CHUNK_SIZE
    NT = triton.cdiv(T, BT)

    # 1. chunk_local_cumsum
    g_cumsum = torch.empty(B, T, H, dtype=torch.float32, device=q.device)
    wrap_triton(_unwrap(chunk_local_cumsum_scalar_kernel))[(NT, B * H)](
        s=g, o=g_cumsum, scale=0, cu_seqlens=0, chunk_indices=0,
        T=T, B=B, H=H, BT=BT,
        HEAD_FIRST=False, REVERSE=False, HAS_SCALE=False, IS_VARLEN=False,
    )

    # 2. chunk_scaled_dot_kkt
    A = torch.empty(B, T, H, BT, device=q.device, dtype=torch.float32)
    wrap_triton(_unwrap(chunk_scaled_dot_kkt_fwd_kernel))[(NT, B * H)](
        k=k, g=g_cumsum, beta=beta, A=A,
        cu_seqlens=0, chunk_indices=0,
        T=T, H=H, K=K, BT=BT, USE_G=True, IS_VARLEN=False,
    )

    # 3. solve_tril
    Ai = torch.zeros_like(A, dtype=k.dtype)
    wrap_triton(_unwrap(merge_16x16_to_64x64_inverse_kernel))[NT, B * H](
        A=A, Ai=Ai, cu_seqlens=0, chunk_indices=0,
        T=T, H=H, BT=BT, USE_TMA=IS_TMA_SUPPORTED, IS_VARLEN=False,
    )

    # 4. recompute_w_u
    w = torch.empty_like(k)
    u = torch.empty_like(v)
    wrap_triton(_unwrap(recompute_w_u_fwd_kernel))[(NT, B * H)](
        k=k, v=v, beta=beta, w=w, u=u, A=Ai, g=g_cumsum,
        cu_seqlens=0, chunk_indices=0,
        T=T, H=H, K=K, V=V, BT=BT, BK=64, BV=64,
        USE_G=True, IS_VARLEN=False,
    )

    # 5. chunk_gated_delta_rule_fwd_h
    h = torch.empty(B, NT, H, K, V, dtype=q.dtype, device=q.device)
    final_state = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    v_new = torch.empty_like(v)

    def grid_h(meta):
        return (triton.cdiv(V, meta["BV"]), B * H)

    wrap_triton(_unwrap(chunk_gated_delta_rule_fwd_kernel_h_blockdim64))[grid_h](
        k=k, v=u, w=w, v_new=v_new, g=g_cumsum, gk=0,
        h=h, h0=initial_state, ht=final_state,
        cu_seqlens=0, chunk_offsets=0,
        T=T, H=H, K=K, V=V, BT=BT,
        USE_EXP2=False, TRANSPOSE_STATE=False, USE_G=True, USE_GK=False,
        USE_INITIAL_STATE=True, STORE_FINAL_STATE=True,
        SAVE_NEW_VALUE=True, IS_VARLEN=False,
    )

    # 6. chunk_fwd_o
    o = torch.empty_like(v)

    def grid_o(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * H)

    wrap_triton(_unwrap(chunk_fwd_kernel_o))[grid_o](
        q=q, k=k, v=v_new, h=h, g=g_cumsum, g_gamma=0, o=o,
        cu_seqlens=0, chunk_indices=0, scale=scale,
        T=T, H=H, K=K, V=V, BT=BT,
        TRANSPOSE_STATE=False, USE_G=True, USE_G_GAMMA=False, IS_VARLEN=False,
    )

    return o, final_state


# ---------------------------------------------------------------------------
# Public API: single triton_op with runtime dispatch.
# ---------------------------------------------------------------------------


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
        q.device == k.device == v.device
        == g.device == beta.device == initial_state.device
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
    Gated delta rule linear attention (forward only).

    Runtime dispatch: T=1 uses a fused recurrent kernel (faster for decode),
    T>1 uses the full chunked FLA pipeline (efficient for prefill).

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
    scale = K**-0.5

    if T == 1:
        return _launch_recurrent(q, k, v, g, beta, initial_state, scale)
    else:
        return _launch_chunked(q, k, v, g, beta, initial_state, scale)


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

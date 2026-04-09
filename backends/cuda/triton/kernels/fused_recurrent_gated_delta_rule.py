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
Register fused_recurrent_gated_delta_rule as a @triton_op for ExecuTorch CUDA backend.

Wraps the fused recurrent kernel from FLA for single-step (T=1) decode.
Uses wrap_triton() so AOTInductor compiles the Triton kernel directly
into the generated .so.

Requires: pip install flash-linear-attention
"""

import torch
import triton

from fla.ops.gated_delta_rule.fused_recurrent import (
    fused_recurrent_gated_delta_rule_fwd_kernel,
)
from torch.library import triton_op, wrap_triton


def _unwrap(kernel):
    """Unwrap @triton.heuristics to get the inner JIT kernel for wrap_triton."""
    if hasattr(kernel, "fn") and isinstance(
        kernel, triton.runtime.autotuner.Heuristics
    ):
        return kernel.fn
    return kernel


@triton_op("triton::fused_recurrent_gated_delta_rule", mutates_args={})
def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused recurrent gated delta rule linear attention (forward only).

    Optimized for single-step decode (T=1). For prefill (T>1), use
    chunk_gated_delta_rule instead.

    Args:
        q: [B, T, H, K] queries (should be L2-normalized)
        k: [B, T, H, K] keys (should be L2-normalized)
        v: [B, T, H, V] values
        g: [B, T, H] gating in log space
        beta: [B, T, H] write strength
        initial_state: [B, H, K, V] initial hidden state

    Returns:
        o: [B, T, H, V] output
        final_state: [B, H, K, V] final hidden state
    """
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    g, beta = g.contiguous(), beta.contiguous()
    initial_state = initial_state.contiguous()

    B, T, H, K = q.shape
    V = v.shape[-1]
    scale = K ** -0.5

    BK = triton.next_power_of_2(K)
    BV = min(8, triton.next_power_of_2(V))
    NV = triton.cdiv(V, BV)

    o = torch.empty_like(v)
    final_state = torch.empty(B, H, K, V, dtype=torch.float32, device=q.device)

    grid = (NV, B * H)
    wrap_triton(_unwrap(fused_recurrent_gated_delta_rule_fwd_kernel))[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        gk=0,
        gv=0,
        beta=beta,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=0,
        scale=scale,
        T=T,
        H=H,
        HV=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_G=True,
        USE_GK=False,
        USE_GV=False,
        USE_QK_L2NORM_IN_KERNEL=False,
        IS_BETA_HEADWISE=True,
        USE_INITIAL_STATE=True,
        STORE_FINAL_STATE=True,
        TRANSPOSE_STATE=False,
        IS_VARLEN=False,
        num_warps=1,
        num_stages=3,
    )

    return o, final_state


@fused_recurrent_gated_delta_rule.register_fake
def _fused_recurrent_gated_delta_rule_fake(
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

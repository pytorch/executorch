# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Correctness test: fully-fused Triton decode kernel vs reference PyTorch.

Verifies that torch.ops.triton.fused_deltanet_decode (which takes raw
qkv_conv, alpha, beta_raw, neg_A_exp, dt_bias) produces the same output
and state as the original GatedDeltaNet T=1 recurrence with manual
Q/K/V split, L2 norm, head repeat, and gating.
"""

import os
import sys

import torch
import torch.nn.functional as F

# Direct import from source tree
sys.path.insert(0, os.path.dirname(__file__))
from fused_deltanet_decode import fused_deltanet_decode  # noqa: F401, E402


def reference_deltanet_decode(
    qkv_conv,
    alpha,
    beta_raw,
    neg_A_exp,
    dt_bias,
    state,
    num_k_heads,
    num_v_heads,
    head_k_dim,
    head_v_dim,
):
    """Reference PyTorch implementation matching model.py's original T=1 path.

    Does Q/K/V split, L2 norm, head repeat, gating, then recurrent update.
    """
    B = qkv_conv.shape[0]
    key_dim = num_k_heads * head_k_dim

    # Q/K/V split from qkv_conv
    q = qkv_conv[:, :key_dim].reshape(B, num_k_heads, head_k_dim)
    k = qkv_conv[:, key_dim : 2 * key_dim].reshape(B, num_k_heads, head_k_dim)
    v = qkv_conv[:, 2 * key_dim :].reshape(B, num_v_heads, head_v_dim)

    # L2 normalize
    q = F.normalize(q.float(), p=2, dim=-1)
    k = F.normalize(k.float(), p=2, dim=-1)
    v = v.float()

    # Head repeat (16 K-heads -> 32 V-heads)
    head_repeat = num_v_heads // num_k_heads
    if head_repeat > 1:
        q = q.repeat_interleave(head_repeat, dim=1)
        k = k.repeat_interleave(head_repeat, dim=1)

    # Gating
    beta = torch.sigmoid(beta_raw.float())
    g = neg_A_exp.float() * F.softplus(alpha.float() + dt_bias.float())

    # Recurrent update (same math as original model.py T=1)
    scale = head_k_dim**-0.5
    state_f32 = state.float()

    decay = torch.exp(g).unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
    state_f32 = state_f32 * decay

    Sk = torch.einsum("bhkv,bhk->bhv", state_f32, k)
    delta = beta.unsqueeze(-1) * (v - Sk)
    state_f32 = state_f32 + torch.einsum("bhk,bhv->bhkv", k, delta)

    output = torch.einsum("bhkv,bhk->bhv", state_f32, q) * scale

    state.copy_(state_f32.to(state.dtype))
    return output, state


def test_fused_deltanet_decode():
    torch.manual_seed(42)
    device = "cuda"

    # Qwen3.5 MoE dimensions
    B = 1
    num_k_heads, num_v_heads = 16, 32
    head_k_dim, head_v_dim = 128, 128
    key_dim = num_k_heads * head_k_dim  # 2048
    value_dim = num_v_heads * head_v_dim  # 4096
    conv_dim = 2 * key_dim + value_dim  # 8192

    # Random qkv_conv (post-conv1d+silu output)
    qkv_conv = torch.randn(B, conv_dim, device=device, dtype=torch.bfloat16) * 0.1

    # Raw alpha and beta (pre-gating, pre-sigmoid)
    alpha = torch.randn(B, num_v_heads, device=device, dtype=torch.float32)
    beta_raw = torch.randn(B, num_v_heads, device=device, dtype=torch.float32)

    # Model parameters
    A_log = torch.log(torch.empty(num_v_heads, device=device).uniform_(0.5, 8))
    neg_A_exp = -torch.exp(A_log).float()
    dt_bias = torch.ones(num_v_heads, device=device, dtype=torch.float32)

    # Initial state
    state_init = (
        torch.randn(
            B, num_v_heads, head_k_dim, head_v_dim, device=device, dtype=torch.bfloat16
        )
        * 0.1
    )

    # --- Reference ---
    ref_state = state_init.clone()
    ref_output, ref_state = reference_deltanet_decode(
        qkv_conv.clone(),
        alpha.clone(),
        beta_raw.clone(),
        neg_A_exp.clone(),
        dt_bias.clone(),
        ref_state,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
    )

    # --- Fused kernel ---
    fused_state = state_init.clone()
    fused_output = torch.ops.triton.fused_deltanet_decode(
        qkv_conv.clone(),
        alpha.clone(),
        beta_raw.clone(),
        neg_A_exp.clone(),
        dt_bias.clone(),
        fused_state,
    )

    # Compare
    out_diff = (fused_output.float() - ref_output.float()).abs()
    out_max_err = out_diff.max().item()
    out_mean_err = out_diff.mean().item()

    state_diff = (fused_state.float() - ref_state.float()).abs()
    state_max_err = state_diff.max().item()
    state_mean_err = state_diff.mean().item()

    print(f"Output  — max err: {out_max_err:.6e}, mean err: {out_mean_err:.6e}")
    print(f"State   — max err: {state_max_err:.6e}, mean err: {state_mean_err:.6e}")

    assert out_max_err < 0.05, f"Output max error too large: {out_max_err}"
    assert state_max_err < 0.05, f"State max error too large: {state_max_err}"
    print("PASSED: fused kernel matches reference within tolerance.\n")

    # --- Batch test ---
    B2 = 4
    qkv2 = torch.randn(B2, conv_dim, device=device, dtype=torch.bfloat16) * 0.1
    alpha2 = torch.randn(B2, num_v_heads, device=device, dtype=torch.float32)
    beta_raw2 = torch.randn(B2, num_v_heads, device=device, dtype=torch.float32)
    state2_init = (
        torch.randn(
            B2, num_v_heads, head_k_dim, head_v_dim, device=device, dtype=torch.bfloat16
        )
        * 0.1
    )

    ref_state2 = state2_init.clone()
    ref_out2, ref_state2 = reference_deltanet_decode(
        qkv2.clone(),
        alpha2.clone(),
        beta_raw2.clone(),
        neg_A_exp.clone(),
        dt_bias.clone(),
        ref_state2,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
    )
    fused_state2 = state2_init.clone()
    fused_out2 = torch.ops.triton.fused_deltanet_decode(
        qkv2.clone(),
        alpha2.clone(),
        beta_raw2.clone(),
        neg_A_exp.clone(),
        dt_bias.clone(),
        fused_state2,
    )

    out_err2 = (fused_out2.float() - ref_out2.float()).abs().max().item()
    state_err2 = (fused_state2.float() - ref_state2.float()).abs().max().item()
    print(
        f"Batch={B2} — output max err: {out_err2:.6e}, state max err: {state_err2:.6e}"
    )
    assert out_err2 < 0.05, f"Batch output error too large: {out_err2}"
    assert state_err2 < 0.05, f"Batch state error too large: {state_err2}"
    print("PASSED: batch test.\n")

    # --- Multi-step sequential test (accumulation drift check) ---
    print("Testing multi-step sequential decode (10 steps)...")
    state_ref = (
        torch.randn(
            1, num_v_heads, head_k_dim, head_v_dim, device=device, dtype=torch.bfloat16
        )
        * 0.01
    )
    state_fused = state_ref.clone()

    for _ in range(10):
        qkv_step = torch.randn(1, conv_dim, device=device, dtype=torch.bfloat16) * 0.1
        a_step = torch.randn(1, num_v_heads, device=device, dtype=torch.float32)
        b_step = torch.randn(1, num_v_heads, device=device, dtype=torch.float32)

        ref_out, state_ref = reference_deltanet_decode(
            qkv_step.clone(),
            a_step.clone(),
            b_step.clone(),
            neg_A_exp.clone(),
            dt_bias.clone(),
            state_ref,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
        )
        fused_out = torch.ops.triton.fused_deltanet_decode(
            qkv_step.clone(),
            a_step.clone(),
            b_step.clone(),
            neg_A_exp.clone(),
            dt_bias.clone(),
            state_fused,
        )

    final_out_err = (fused_out.float() - ref_out.float()).abs().max().item()
    final_state_err = (state_fused.float() - state_ref.float()).abs().max().item()
    print(
        f"After 10 steps — output max err: {final_out_err:.6e}, state max err: {final_state_err:.6e}"
    )
    assert final_out_err < 0.1, f"Multi-step output error too large: {final_out_err}"
    assert final_state_err < 0.1, f"Multi-step state error too large: {final_state_err}"
    print("PASSED: multi-step test.\n")

    # --- CUDA Graph compatibility test ---
    print("Testing CUDA Graph compatibility...")
    qkv_cg = torch.randn(1, conv_dim, device=device, dtype=torch.bfloat16) * 0.1
    alpha_cg = torch.randn(1, num_v_heads, device=device, dtype=torch.float32)
    beta_cg = torch.randn(1, num_v_heads, device=device, dtype=torch.float32)
    state_cg = state_init[:1].clone()

    # Warmup
    for _ in range(3):
        state_cg.copy_(state_init[:1])
        _ = torch.ops.triton.fused_deltanet_decode(
            qkv_cg, alpha_cg, beta_cg, neg_A_exp, dt_bias, state_cg
        )

    # Capture
    state_cg.copy_(state_init[:1])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_cg = torch.ops.triton.fused_deltanet_decode(
            qkv_cg, alpha_cg, beta_cg, neg_A_exp, dt_bias, state_cg
        )

    # Replay
    state_cg.copy_(state_init[:1])
    graph.replay()

    # Compare with reference using same inputs
    ref_state_cg = state_init[:1].clone()
    ref_out_cg, _ = reference_deltanet_decode(
        qkv_cg.clone(),
        alpha_cg.clone(),
        beta_cg.clone(),
        neg_A_exp.clone(),
        dt_bias.clone(),
        ref_state_cg,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
    )
    cg_err = (out_cg.float() - ref_out_cg.float()).abs().max().item()
    print(f"CUDA Graph — output max err: {cg_err:.6e}")
    assert cg_err < 0.05, f"CUDA Graph output error too large: {cg_err}"
    print("PASSED: CUDA Graph compatible.\n")

    print("ALL TESTS PASSED.")


if __name__ == "__main__":
    test_fused_deltanet_decode()

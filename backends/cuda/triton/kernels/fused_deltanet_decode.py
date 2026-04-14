# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Fully-fused SRAM-resident GatedDeltaNet recurrent kernel for decode (T=1).

Fuses post-projection (Q/K/V split from conv1d output, L2 normalization,
head repeat, gating computation) AND the recurrent state update into a
single Triton kernel per layer.

This eliminates intermediate HBM reads/writes for q, k, v, g, beta tensors
and removes multiple small kernel launches (normalize, repeat_interleave,
sigmoid, softplus, exp) that the previous partial-fusion approach required.

For each (batch, v_head):
    k_head = v_head // V_PER_K          # shared K head
    q, k = L2_normalize(qkv_conv[Q/K])  # split + normalize
    v = qkv_conv[V]                      # split
    decay = exp(-exp(A_log) * softplus(alpha + dt_bias))
    beta = sigmoid(beta_raw)
    state = state * decay                # decay
    Sk = state @ k                       # [V]
    delta = beta * (v - Sk)              # [V]
    state = state + outer(k, delta)      # rank-1 update
    output = state @ (q * scale)         # [V]

The kernel tiles over the V dimension in blocks of BLOCK_V.
For each V-tile, it streams through K in blocks of BLOCK_K.

Registered as torch.ops.triton.fused_deltanet_decode for AOTI compilation.
"""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K": 32, "BLOCK_V": 32}),
        triton.Config({"BLOCK_K": 64, "BLOCK_V": 64}),
        triton.Config({"BLOCK_K": 128, "BLOCK_V": 128}),
        triton.Config({"BLOCK_K": 128, "BLOCK_V": 64}),
        triton.Config({"BLOCK_K": 64, "BLOCK_V": 128}),
    ],
    key=["K", "V_DIM"],
)
@triton.jit
def _fused_deltanet_decode_kernel(
    # Tensor pointers
    QKV_ptr,  # [B, conv_dim]  post-conv1d+silu output
    Alpha_ptr,  # [B, H]         raw gating input (a)
    BetaRaw_ptr,  # [B, H]         raw write strength (b, pre-sigmoid)
    NegAExp_ptr,  # [H]            -exp(A_log), precomputed
    DtBias_ptr,  # [H]            dt_bias parameter
    S_in_ptr,  # [B, H, K, V]   recurrent state input (read-only)
    S_out_ptr,  # [B, H, K, V]   recurrent state output (write-only)
    O_ptr,  # [B, H, V]      output
    # Dimension constants
    K: tl.constexpr,  # head_k_dim (128)
    V_DIM: tl.constexpr,  # head_v_dim (128)
    KEY_DIM: tl.constexpr,  # num_k_heads * K (2048)
    V_PER_K: tl.constexpr,  # num_v_heads // num_k_heads (2)
    SCALE: tl.constexpr,  # K^(-0.5)
    L2_EPS: tl.constexpr,  # 1e-6
    # Strides
    stride_qkv_b,  # qkv stride for batch dim
    stride_ab,  # alpha stride for batch dim
    stride_bb,  # beta_raw stride for batch dim
    stride_s_b,  # state stride: batch
    stride_s_h,  # state stride: head
    stride_s_k,  # state stride: K dim
    stride_s_v,  # state stride: V dim
    stride_ob,  # output stride: batch
    stride_oh,  # output stride: head
    stride_ov,  # output stride: V dim
    # Block sizes (autotuned)
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """One program per (batch, v_head, v_block)."""
    pid_bh = tl.program_id(0)  # batch * num_v_heads index
    pid_v = tl.program_id(1)  # V-tile index

    # Decompose pid_bh into batch and v_head
    H: tl.constexpr = KEY_DIM // K * V_PER_K  # num_v_heads
    bid = pid_bh // H
    h = pid_bh % H
    k_head = h // V_PER_K  # corresponding K head

    # V-tile range
    v_start = pid_v * BLOCK_V
    v_offs = v_start + tl.arange(0, BLOCK_V)
    v_mask = v_offs < V_DIM

    # ====== Phase 1: Load V slice from qkv_conv ======
    # Layout: qkv_conv = [Q(KEY_DIM) | K(KEY_DIM) | V(H * V_DIM)]
    qkv_base = QKV_ptr + bid * stride_qkv_b
    v_base = qkv_base + 2 * KEY_DIM + h * V_DIM
    v_vals = tl.load(v_base + v_offs, mask=v_mask, other=0.0).to(tl.float32)

    # ====== Phase 2: Compute gating and beta ======
    alpha_h = tl.load(Alpha_ptr + bid * stride_ab + h).to(tl.float32)
    neg_a_exp_h = tl.load(NegAExp_ptr + h).to(tl.float32)
    dt_bias_h = tl.load(DtBias_ptr + h).to(tl.float32)

    # softplus with numerical stability
    sp_input = alpha_h + dt_bias_h
    sp = tl.where(sp_input > 20.0, sp_input, tl.log(1.0 + tl.exp(sp_input)))
    gate = neg_a_exp_h * sp  # always negative
    decay = tl.exp(gate)

    beta_raw_h = tl.load(BetaRaw_ptr + bid * stride_bb + h).to(tl.float32)
    beta = tl.sigmoid(beta_raw_h)

    # ====== Phase 3: Compute K and Q L2 norms (full-vector reduction) ======
    # Each v_block program needs the full K-vector norms, so we compute them here.
    # This is redundant across v_blocks for the same (batch, head) but avoids
    # a separate kernel launch or shared memory coordination.
    q_base = qkv_base + k_head * K
    k_base = qkv_base + KEY_DIM + k_head * K

    q_sq_sum = tl.zeros([], dtype=tl.float32)
    k_sq_sum = tl.zeros([], dtype=tl.float32)
    for kk in range(0, K, BLOCK_K):
        kk_offs = kk + tl.arange(0, BLOCK_K)
        kk_mask = kk_offs < K
        q_chunk = tl.load(q_base + kk_offs, mask=kk_mask, other=0.0).to(tl.float32)
        k_chunk = tl.load(k_base + kk_offs, mask=kk_mask, other=0.0).to(tl.float32)
        q_sq_sum += tl.sum(q_chunk * q_chunk)
        k_sq_sum += tl.sum(k_chunk * k_chunk)

    q_norm = tl.maximum(tl.sqrt(q_sq_sum), L2_EPS)
    k_norm = tl.maximum(tl.sqrt(k_sq_sum), L2_EPS)

    # ====== Phase 4: Recurrent state update ======
    s_in_base = S_in_ptr + bid * stride_s_b + h * stride_s_h
    s_out_base = S_out_ptr + bid * stride_s_b + h * stride_s_h

    # --- Pass 1: Decay state, compute Sk = (decay*S)^T @ k_normalized ---
    sk_acc = tl.zeros([BLOCK_V], dtype=tl.float32)
    for kk in range(0, K, BLOCK_K):
        kk_offs = kk + tl.arange(0, BLOCK_K)
        kk_mask = kk_offs < K

        # Load normalized k slice
        k_vals = (
            tl.load(k_base + kk_offs, mask=kk_mask, other=0.0).to(tl.float32) / k_norm
        )

        # Load state tile [BLOCK_K, BLOCK_V]
        tile_offs = kk_offs[:, None] * stride_s_k + v_offs[None, :] * stride_s_v
        tile_mask = kk_mask[:, None] & v_mask[None, :]
        s_tile = tl.load(s_in_base + tile_offs, mask=tile_mask, other=0.0).to(
            tl.float32
        )

        # Decay
        s_tile = s_tile * decay

        # Sk[v] += sum_k(state[k,v] * k_normalized[k])
        sk_acc += tl.sum(s_tile * k_vals[:, None], axis=0)

    # delta = beta * (v - Sk)
    delta_v = beta * (v_vals - sk_acc)

    # --- Pass 2: Re-read input, decay + rank-1 update, write output state, compute output ---
    out_acc = tl.zeros([BLOCK_V], dtype=tl.float32)
    for kk in range(0, K, BLOCK_K):
        kk_offs = kk + tl.arange(0, BLOCK_K)
        kk_mask = kk_offs < K

        # Load normalized k and q slices
        k_vals = (
            tl.load(k_base + kk_offs, mask=kk_mask, other=0.0).to(tl.float32) / k_norm
        )
        q_vals = (
            tl.load(q_base + kk_offs, mask=kk_mask, other=0.0).to(tl.float32)
            / q_norm
            * SCALE
        )

        # Re-read input state and decay
        tile_offs = kk_offs[:, None] * stride_s_k + v_offs[None, :] * stride_s_v
        tile_mask = kk_mask[:, None] & v_mask[None, :]
        s_tile = tl.load(s_in_base + tile_offs, mask=tile_mask, other=0.0).to(
            tl.float32
        )
        s_tile = s_tile * decay

        # Rank-1 update: S += k ⊗ delta
        s_tile = s_tile + k_vals[:, None] * delta_v[None, :]

        # Store updated state
        tl.store(
            s_out_base + tile_offs,
            s_tile.to(S_out_ptr.dtype.element_ty),
            mask=tile_mask,
        )

        # Output: out[v] += sum_k(S_new[k,v] * q_scaled[k])
        out_acc += tl.sum(s_tile * q_vals[:, None], axis=0)

    # Store output
    o_offs = O_ptr + bid * stride_ob + h * stride_oh + v_offs * stride_ov
    tl.store(o_offs, out_acc.to(O_ptr.dtype.element_ty), mask=v_mask)


@triton_op("triton::fused_deltanet_decode", mutates_args={})
def fused_deltanet_decode(
    qkv: torch.Tensor,
    alpha: torch.Tensor,
    beta_raw: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fully-fused GatedDeltaNet decode (T=1) recurrent step.

    Fuses Q/K/V split, L2 normalization, head repeat, gating, and delta rule
    recurrence into a single kernel.

    Args:
        qkv: [B, conv_dim] post-conv1d+silu output (Q|K|V concatenated)
        alpha: [B, num_v_heads] raw gating input (pre-softplus)
        beta_raw: [B, num_v_heads] raw write strength (pre-sigmoid)
        A_log: [num_v_heads] log(A) parameter (negated exp computed inside)
        dt_bias: [num_v_heads] gating bias parameter
        state: [B, num_v_heads, K, V] recurrent state (read-only, not mutated)

    Returns:
        tuple of (output, new_state):
            output: [B, num_v_heads, V] decode output (same dtype as state)
            new_state: [B, num_v_heads, K, V] updated state (same dtype as state)
    """
    B = qkv.shape[0]
    H, K, V_DIM = state.shape[1], state.shape[2], state.shape[3]

    # Derive layout constants from tensor shapes
    # conv_dim = 2 * KEY_DIM + H * V_DIM, KEY_DIM = num_k_heads * K
    value_dim = H * V_DIM
    KEY_DIM = (qkv.shape[1] - value_dim) // 2
    num_k_heads = KEY_DIM // K
    V_PER_K = H // num_k_heads

    output = torch.empty(B, H, V_DIM, dtype=state.dtype, device=qkv.device)

    # Compute neg_A_exp from A_log parameter
    neg_A_exp = -torch.exp(A_log.float())

    # Separate input/output state buffers for autotuning safety
    # (autotuner may re-run the kernel; reading from a buffer we also write
    # would produce wrong results on the second run)
    state_in = state.float().contiguous()
    state_out = torch.empty_like(state_in)

    def grid(meta):
        return (B * H, triton.cdiv(V_DIM, meta["BLOCK_V"]))

    wrap_triton(_fused_deltanet_decode_kernel)[grid](
        qkv,
        alpha,
        beta_raw,
        neg_A_exp,
        dt_bias,
        state_in,
        state_out,
        output,
        # Dimensions
        K=K,
        V_DIM=V_DIM,
        KEY_DIM=KEY_DIM,
        V_PER_K=V_PER_K,
        SCALE=K**-0.5,
        L2_EPS=1e-6,
        # Strides
        stride_qkv_b=qkv.stride(0),
        stride_ab=alpha.stride(0),
        stride_bb=beta_raw.stride(0),
        stride_s_b=state_in.stride(0),
        stride_s_h=state_in.stride(1),
        stride_s_k=state_in.stride(2),
        stride_s_v=state_in.stride(3),
        stride_ob=output.stride(0),
        stride_oh=output.stride(1),
        stride_ov=output.stride(2),
    )

    return output, state_out.to(state.dtype)


@fused_deltanet_decode.register_fake
def _fused_deltanet_decode_fake(
    qkv: torch.Tensor,
    alpha: torch.Tensor,
    beta_raw: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    B = qkv.shape[0]
    H, K_DIM, V_DIM = state.shape[1], state.shape[2], state.shape[3]
    output = torch.empty(B, H, V_DIM, dtype=state.dtype, device=qkv.device)
    new_state = torch.empty(B, H, K_DIM, V_DIM, dtype=state.dtype, device=qkv.device)
    return output, new_state

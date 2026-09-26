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
When BLOCK_K >= K (e.g. BLOCK_K=128 for head_k_dim=128), the entire K
dimension fits in one tile, enabling a single-pass algorithm that reads
state only once from HBM.  Otherwise, a two-pass path is used with an
algebraic kq_dot trick to eliminate Q loads from the second pass.

State is kept in its original dtype (bf16/fp16) in HBM and converted to
fp32 only in registers, halving state memory traffic vs. a full fp32 copy.

Registered as torch.ops.triton.fused_deltanet_decode for AOTI compilation.
"""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.autotune(
    configs=[
        # BLOCK_K=128 enables single-pass for K=128 (typical head_k_dim).
        # Caches full state tile [K, BLOCK_V] in registers — only ONE state read.
        # Smaller BLOCK_V yields more programs for better occupancy at B=1.
        triton.Config({"BLOCK_K": 128, "BLOCK_V": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_K": 128, "BLOCK_V": 32}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_K": 128, "BLOCK_V": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_K": 128, "BLOCK_V": 64}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_K": 128, "BLOCK_V": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_K": 128, "BLOCK_V": 128}, num_warps=8, num_stages=1),
        # Fallback for larger K or high register pressure.
        triton.Config({"BLOCK_K": 64, "BLOCK_V": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_K": 64, "BLOCK_V": 64}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_K": 64, "BLOCK_V": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_K": 32, "BLOCK_V": 32}, num_warps=2, num_stages=1),
    ],
    key=["K", "V_DIM"],
)
@triton.jit
def _fused_deltanet_decode_kernel(
    # Tensor pointers
    QKV_ptr,  # [B, conv_dim]  post-conv1d+silu output
    Alpha_ptr,  # [B, H]         raw gating input (a)
    BetaRaw_ptr,  # [B, H]         raw write strength (b, pre-sigmoid)
    ALog_ptr,  # [H]            A_log parameter (-exp computed inside)
    DtBias_ptr,  # [H]            dt_bias parameter
    S_in_ptr,  # [B, H, K, V]   recurrent state input (original dtype)
    S_out_ptr,  # [B, H, K, V]   recurrent state output (original dtype)
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
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)

    H: tl.constexpr = KEY_DIM // K * V_PER_K
    bid = pid_bh // H
    h = pid_bh % H
    k_head = h // V_PER_K

    v_start = pid_v * BLOCK_V
    v_offs = v_start + tl.arange(0, BLOCK_V)
    v_mask = v_offs < V_DIM

    # ====== Load V slice from qkv_conv ======
    qkv_base = QKV_ptr + bid * stride_qkv_b
    v_vals = tl.load(
        qkv_base + 2 * KEY_DIM + h * V_DIM + v_offs,
        mask=v_mask,
        other=0.0,
    ).to(tl.float32)

    # ====== Gating (A_log → decay, fully fused) ======
    alpha_h = tl.load(Alpha_ptr + bid * stride_ab + h).to(tl.float32)
    a_log_h = tl.load(ALog_ptr + h).to(tl.float32)
    dt_bias_h = tl.load(DtBias_ptr + h).to(tl.float32)

    sp_input = alpha_h + dt_bias_h
    sp = tl.where(sp_input > 20.0, sp_input, tl.log(1.0 + tl.exp(sp_input)))
    decay = tl.exp(-tl.exp(a_log_h) * sp)

    beta = tl.sigmoid(tl.load(BetaRaw_ptr + bid * stride_bb + h).to(tl.float32))

    # ====== Pointer bases ======
    q_base = qkv_base + k_head * K
    k_base = qkv_base + KEY_DIM + k_head * K
    s_in_base = S_in_ptr + bid * stride_s_b + h * stride_s_h
    s_out_base = S_out_ptr + bid * stride_s_b + h * stride_s_h
    o_base = O_ptr + bid * stride_ob + h * stride_oh

    # ====== Main computation ======
    if BLOCK_K >= K:
        # ---- SINGLE-PASS: full K dim fits in one register tile ----
        # Only ONE state read from HBM.  The decayed state tile lives in
        # registers through Sk computation, rank-1 update, and output.
        kk_offs = tl.arange(0, BLOCK_K)
        kk_mask = kk_offs < K

        # Load K and Q once, compute norms in-place (no separate norm loop)
        k_raw = tl.load(k_base + kk_offs, mask=kk_mask, other=0.0).to(tl.float32)
        q_raw = tl.load(q_base + kk_offs, mask=kk_mask, other=0.0).to(tl.float32)

        k_inv_norm = 1.0 / tl.maximum(tl.sqrt(tl.sum(k_raw * k_raw)), L2_EPS)
        q_inv_norm = 1.0 / tl.maximum(tl.sqrt(tl.sum(q_raw * q_raw)), L2_EPS)

        k_vals = k_raw * k_inv_norm
        q_vals = q_raw * (q_inv_norm * SCALE)

        # Load & decay state tile [BLOCK_K, BLOCK_V]
        tile_offs = kk_offs[:, None] * stride_s_k + v_offs[None, :] * stride_s_v
        tile_mask = kk_mask[:, None] & v_mask[None, :]
        s_tile = tl.load(s_in_base + tile_offs, mask=tile_mask, other=0.0).to(
            tl.float32
        )
        s_tile = s_tile * decay

        # Sk = (decayed state)^T @ k_normalized → [BLOCK_V]
        sk = tl.sum(s_tile * k_vals[:, None], axis=0)

        # delta = beta * (v - Sk)
        delta_v = beta * (v_vals - sk)

        # Rank-1 update: S_new = S_decayed + outer(k, delta)
        s_tile = s_tile + k_vals[:, None] * delta_v[None, :]

        # Output: out = S_new^T @ q_scaled → [BLOCK_V]
        out_acc = tl.sum(s_tile * q_vals[:, None], axis=0)

        # Store state and output
        tl.store(
            s_out_base + tile_offs,
            s_tile.to(S_out_ptr.dtype.element_ty),
            mask=tile_mask,
        )
        tl.store(
            o_base + v_offs * stride_ov,
            out_acc.to(O_ptr.dtype.element_ty),
            mask=v_mask,
        )
    else:
        # ---- TWO-PASS with merged norm + kq_dot optimization ----
        # Pass 1 fuses norm accumulation, Sk, unnormalized output, and kq_dot
        # into one loop (eliminates separate norm loop + Q loads in Pass 2).
        #
        # Algebraic trick: output = (decayed_S @ q_norm)*SCALE + kq_dot*delta
        #   where kq_dot = dot(k_norm, q_norm)*SCALE is a scalar.
        # This lets Pass 2 only write state without touching Q or accumulating
        # output.
        sk_unnorm = tl.zeros([BLOCK_V], dtype=tl.float32)
        out_unnorm = tl.zeros([BLOCK_V], dtype=tl.float32)
        kq_unnorm = tl.zeros([], dtype=tl.float32)
        k_sq_sum = tl.zeros([], dtype=tl.float32)
        q_sq_sum = tl.zeros([], dtype=tl.float32)

        for kk in range(0, K, BLOCK_K):
            kk_offs = kk + tl.arange(0, BLOCK_K)
            kk_mask = kk_offs < K

            k_raw = tl.load(k_base + kk_offs, mask=kk_mask, other=0.0).to(tl.float32)
            q_raw = tl.load(q_base + kk_offs, mask=kk_mask, other=0.0).to(tl.float32)

            k_sq_sum += tl.sum(k_raw * k_raw)
            q_sq_sum += tl.sum(q_raw * q_raw)
            kq_unnorm += tl.sum(k_raw * q_raw)

            tile_offs = kk_offs[:, None] * stride_s_k + v_offs[None, :] * stride_s_v
            tile_mask = kk_mask[:, None] & v_mask[None, :]
            s_tile = tl.load(s_in_base + tile_offs, mask=tile_mask, other=0.0).to(
                tl.float32
            )
            s_tile = s_tile * decay

            sk_unnorm += tl.sum(s_tile * k_raw[:, None], axis=0)
            out_unnorm += tl.sum(s_tile * q_raw[:, None], axis=0)

        # Normalize and compute output
        k_inv_norm = 1.0 / tl.maximum(tl.sqrt(k_sq_sum), L2_EPS)
        q_inv_norm = 1.0 / tl.maximum(tl.sqrt(q_sq_sum), L2_EPS)

        sk = sk_unnorm * k_inv_norm
        delta_v = beta * (v_vals - sk)

        # out = (S_decayed @ q_norm)*SCALE + dot(k_norm, q_norm)*SCALE * delta
        norm_scale = q_inv_norm * SCALE
        kq_dot = kq_unnorm * k_inv_norm * norm_scale
        out_acc = out_unnorm * norm_scale + kq_dot * delta_v

        tl.store(
            o_base + v_offs * stride_ov,
            out_acc.to(O_ptr.dtype.element_ty),
            mask=v_mask,
        )

        # Pass 2: state update only (no output, no Q loads)
        for kk in range(0, K, BLOCK_K):
            kk_offs = kk + tl.arange(0, BLOCK_K)
            kk_mask = kk_offs < K

            k_vals = (
                tl.load(k_base + kk_offs, mask=kk_mask, other=0.0).to(tl.float32)
                * k_inv_norm
            )

            tile_offs = kk_offs[:, None] * stride_s_k + v_offs[None, :] * stride_s_v
            tile_mask = kk_mask[:, None] & v_mask[None, :]
            s_tile = tl.load(s_in_base + tile_offs, mask=tile_mask, other=0.0).to(
                tl.float32
            )
            s_tile = s_tile * decay + k_vals[:, None] * delta_v[None, :]

            tl.store(
                s_out_base + tile_offs,
                s_tile.to(S_out_ptr.dtype.element_ty),
                mask=tile_mask,
            )


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

    value_dim = H * V_DIM
    KEY_DIM = (qkv.shape[1] - value_dim) // 2
    num_k_heads = KEY_DIM // K
    V_PER_K = H // num_k_heads

    output = torch.empty(B, H, V_DIM, dtype=state.dtype, device=qkv.device)

    # State stays in original dtype (bf16/fp16/fp32). The kernel converts to
    # fp32 in registers for computation and writes back in the original dtype.
    # This halves HBM traffic for bf16/fp16 vs. the old .float() copy approach.
    state_in = state.contiguous()
    state_out = torch.empty_like(state_in)

    def grid(meta):
        return (B * H, triton.cdiv(V_DIM, meta["BLOCK_V"]))

    wrap_triton(_fused_deltanet_decode_kernel)[grid](
        qkv,
        alpha,
        beta_raw,
        A_log,
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

    return output, state_out


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

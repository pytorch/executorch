#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Metal source transformations for Qwen 3.5 MoE.

Replaces Triton-dependent modules (FusedMoEExperts, GatedDeltaNet) with
pure-PyTorch + Metal custom op equivalents that can be exported and lowered
to the Metal backend via AOTInductor.
"""

import logging
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.examples.models.qwen3_5_moe.model import (
    FullAttention,
    FusedMoEExperts,
    GatedDeltaNet,
    SparseMoE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MetalMoEExperts: replaces FusedMoEExperts
# ---------------------------------------------------------------------------


class MetalMoEExperts(nn.Module):
    """MoE experts using metal::gather_qmv for expert-indexed quantized matmul.

    Decomposes the fused MoE into two gather_qmv calls (gate+up, down) with
    SiLU gating in between. Expert weights are in MLX affine INT4 format.
    """

    def __init__(self, num_experts, intermediate_size, hidden_size, group_size=32):
        super().__init__()
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.group_size = group_size

    def forward(self, x, expert_weights, expert_indices, top_k):
        P = x.shape[0]
        # Flatten expert pairs: [P, top_k] -> [P*top_k]
        indices_flat = expert_indices.reshape(-1).to(torch.int32)
        x_expanded = x.unsqueeze(1).expand(-1, top_k, -1).reshape(P * top_k, -1)

        # GEMM1: gate+up projection  [P*top_k, K] @ [E, 2*inter, K].T -> [P*top_k, 2*inter]
        gate_up = torch.ops.metal.gather_qmv(
            x_expanded, self.w1, self.s1, self.b1, indices_flat, self.group_size
        )
        gate = gate_up[..., : self.intermediate_size]
        up = gate_up[..., self.intermediate_size :]
        activated = F.silu(gate) * up

        # GEMM2: down projection  [P*top_k, inter] @ [E, K, inter].T -> [P*top_k, K]
        down = torch.ops.metal.gather_qmv(
            activated, self.w2, self.s2, self.b2, indices_flat, self.group_size
        )

        # Weighted sum over top_k experts
        down = down.view(P, top_k, -1)
        return (down * expert_weights.unsqueeze(-1)).sum(dim=1)


# ---------------------------------------------------------------------------
# GatedDeltaNet replacement forward
# ---------------------------------------------------------------------------


def _metal_gated_delta_net_forward(self, x, input_pos):
    """Replacement forward for GatedDeltaNet using metal::gated_delta_rule.

    Same pre/post-processing as the original, but replaces both the T=1
    native path and the T>1 Triton kernel with a single custom op call
    that works for all T values.
    """
    B, T, _ = x.size()

    # Reset state at position 0
    reset = (input_pos[0] == 0).to(self.conv_state.dtype)
    keep = 1.0 - reset
    self.conv_state[:B].mul_(keep)
    self.recurrent_state[:B].mul_(keep)

    # Fused projection: split into qkv, z, b, a
    proj = self.in_proj(x)
    cd = self.conv_dim
    vd = self.value_dim
    nh = self.num_v_heads
    mixed_qkv = proj[..., :cd]
    z = proj[..., cd : cd + vd].reshape(B, T, self.num_v_heads, self.head_v_dim)
    b = proj[..., cd + vd : cd + vd + nh]
    a = proj[..., cd + vd + nh :]

    # Causal depthwise conv1d with state
    qkv_t = mixed_qkv.transpose(1, 2)
    conv_input = torch.cat([self.conv_state[:B], qkv_t], dim=-1)
    conv_len = conv_input.shape[-1]
    self.conv_state[:B].copy_(conv_input[:, :, conv_len - self.conv_kernel_size :])

    # Manual depthwise conv1d (avoids conv1d->conv2d decomposition)
    w = self.conv1d.weight.squeeze(1).float()
    T_conv = conv_input.shape[-1] - self.conv_kernel_size + 1
    acc = torch.zeros(
        B, conv_input.shape[1], T_conv, dtype=torch.float32, device=conv_input.device
    )
    for k in range(self.conv_kernel_size):
        acc = acc + conv_input[:, :, k : k + T_conv].float() * w[:, k : k + 1]
    qkv_conv = F.silu(acc[:, :, -T:]).to(conv_input.dtype).transpose(1, 2)

    # Split into Q, K, V
    kd = self.key_dim
    q = qkv_conv[..., :kd].reshape(B, T, self.num_k_heads, self.head_k_dim)
    k = qkv_conv[..., kd : 2 * kd].reshape(B, T, self.num_k_heads, self.head_k_dim)
    v = qkv_conv[..., 2 * kd :].reshape(B, T, self.num_v_heads, self.head_v_dim)

    # L2-normalize Q and K
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)

    # head_repeat for k_heads != v_heads
    if self.head_repeat > 1:
        q = q.repeat_interleave(self.head_repeat, dim=2)
        k = k.repeat_interleave(self.head_repeat, dim=2)

    # Mamba-style gating: g = exp(-A * softplus(a + dt_bias))
    beta = b.sigmoid()
    g = (-self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)).exp()

    # Metal custom op: handles both T=1 and T>1
    import executorch.backends.apple.metal.ops.gated_delta_rule as _  # noqa: F401

    output = torch.ops.metal.gated_delta_rule(
        q, k, v, g, beta, self.recurrent_state[:B]
    )

    # Output: RMSNorm(output) * silu(z)
    output = output.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    output = self.norm(output, z)
    output = output.reshape(B, T, -1)

    return self.out_proj(output)


# ---------------------------------------------------------------------------
# FullAttention: remove turboquant
# ---------------------------------------------------------------------------


def _metal_full_attention_forward(self, x, input_pos):
    """FullAttention forward without turboquant (CUDA-only)."""
    B, T, _ = x.size()
    dtype = x.dtype

    qkv = self.qkv_proj(x)
    q_and_gate = qkv[..., : self.q_dim].view(B, T, self.n_heads, self.head_dim * 2)
    q = q_and_gate[..., : self.head_dim]
    gate = q_and_gate[..., self.head_dim :]

    k = qkv[..., self.q_dim : self.q_dim + self.k_dim].view(
        B, T, self.n_kv_heads, self.head_dim
    )
    v = qkv[..., self.q_dim + self.k_dim :].view(B, T, self.n_kv_heads, self.head_dim)

    q = self.q_norm(q)
    k = self.k_norm(k)

    q, k = self.rotary_emb(input_pos, q, k)

    q = q.to(dtype).transpose(1, 2)
    k = k.to(dtype).transpose(1, 2)
    v = v.transpose(1, 2)

    attn_mask = (
        (self.cache_positions[None, :] <= input_pos[:, None]).unsqueeze(0).unsqueeze(0)
    )

    # Always use standard SDPA (no turboquant on Metal)
    k, v = self.kv_cache.update(input_pos, k, v)
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=True)

    y = y.transpose(1, 2).contiguous().view(B, T, -1)

    gate = gate.reshape(B, T, -1)
    y = y * torch.sigmoid(gate)

    return self.o_proj(y)


# ---------------------------------------------------------------------------
# Expert weight quantization (MLX affine INT4 format)
# ---------------------------------------------------------------------------


def quantize_experts_metal(model, config, group_size=32):
    """Quantize expert weights to MLX affine INT4 format for metal::gather_qmv.

    Produces unsigned INT4 with scale + bias (zero-point) per group:
      dequant(w) = w_uint4 * scale + bias

    Output layout per expert:
      w:      [N, K//2] uint8 (two 4-bit values packed per byte)
      scales: [N, K//group_size] same dtype as model
      biases: [N, K//group_size] same dtype as model
    """
    from torchao.quantization.quant_primitives import (
        choose_qparams_affine,
        MappingType,
        quantize_affine,
    )

    for i, layer in enumerate(model.layers):
        experts = layer.mlp.experts
        if not isinstance(experts, FusedMoEExperts):
            continue

        metal_experts = MetalMoEExperts(
            experts.num_experts,
            experts.intermediate_size,
            experts.hidden_size,
            group_size,
        )

        for name in ("w1_weight", "w2_weight"):
            w = getattr(experts, name).data.float()
            E, N, K = w.shape
            block_size = (1, 1, group_size)

            scale, zero_point = choose_qparams_affine(
                w,
                MappingType.ASYMMETRIC,
                block_size,
                target_dtype=torch.uint8,
                quant_min=0,
                quant_max=15,
            )

            int_data = quantize_affine(
                w,
                block_size,
                scale,
                zero_point,
                output_dtype=torch.uint8,
                quant_min=0,
                quant_max=15,
            )

            # Pack two uint4 values per byte: even -> low nibble, odd -> high nibble
            low = int_data[:, :, 0::2]
            high = int_data[:, :, 1::2]
            packed = (low | (high << 4)).to(torch.uint8)

            scale = scale.reshape(E, N, -1)
            # Compute bias: zero_point contribution -> -zero_point * scale
            bias = (-zero_point.reshape(E, N, -1).float() * scale.float()).to(
                scale.dtype
            )

            buf_prefix = "w1" if "w1" in name else "w2"
            metal_experts.register_buffer(f"{buf_prefix}", packed)
            metal_experts.register_buffer(f"s{buf_prefix[1]}", scale.to(w.dtype))
            metal_experts.register_buffer(f"b{buf_prefix[1]}", bias.to(w.dtype))

        # Replace in model
        parts = f"layers.{i}.mlp.experts".rsplit(".", 1)
        parent = model.get_submodule(parts[0])
        setattr(parent, parts[1], metal_experts)
        print(
            f"  Quantized experts (Metal INT4) layer {i + 1}/{config.num_hidden_layers}",
            end="\r",
        )
    print()


# ---------------------------------------------------------------------------
# Top-level transformation
# ---------------------------------------------------------------------------


def metal_source_transformations(model, config=None):
    """Replace all Triton-dependent modules with Metal-compatible equivalents.

    Transforms:
      1. GatedDeltaNet → metal::gated_delta_rule custom op
      2. FullAttention → remove turboquant, keep standard SDPA
      3. SparseMoE.experts already replaced by quantize_experts_metal()
    """
    count_gdn = 0
    for _name, module in model.named_modules():
        if isinstance(module, GatedDeltaNet):
            module.forward = types.MethodType(_metal_gated_delta_net_forward, module)
            count_gdn += 1

    count_attn = 0
    for _name, module in model.named_modules():
        if isinstance(module, FullAttention):
            module.turboquant = False
            module.forward = types.MethodType(_metal_full_attention_forward, module)
            count_attn += 1

    # Remove .float() cast on expert_weights in SparseMoE
    count_moe = 0
    for _name, module in model.named_modules():
        if isinstance(module, SparseMoE):

            def _sparse_moe_forward(self, x):
                B, T, C = x.size()
                x_flat = x.view(-1, C)
                scores = self.gate(x_flat)
                expert_weights, expert_indices = torch.topk(scores, self.top_k, dim=-1)
                expert_weights = expert_weights.softmax(dim=-1)
                routed_out = self.experts(
                    x_flat, expert_weights, expert_indices, self.top_k
                )
                shared_out = self.shared_expert(x_flat)
                shared_gate = torch.sigmoid(self.shared_expert_gate(x_flat))
                return (routed_out + shared_gate * shared_out).view(B, T, C)

            module.forward = types.MethodType(_sparse_moe_forward, module)
            count_moe += 1

    logger.info(f"Replaced {count_gdn} GatedDeltaNet → metal::gated_delta_rule")
    logger.info(f"Replaced {count_attn} FullAttention → standard SDPA (no turboquant)")
    logger.info(f"Replaced {count_moe} SparseMoE → no .float() cast")

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
MLX source transformations for Qwen 3.5 MoE.

Replaces Triton-dependent modules (FusedMoEExperts, GatedDeltaNet) with
pure-PyTorch + MLX custom op equivalents that can be exported and lowered
to the MLX delegate.
"""

import logging
import types

import torch
import torch.nn as nn

from executorch.examples.models.qwen3_5_moe.model import (
    FullAttention,
    FusedMoEExperts,
    GatedDeltaNet,
    GemmaRMSNorm,
    KVCache,
    SparseMoE,
)

logger = logging.getLogger(__name__)


def _rms_norm_gated_forward(self, x, z):
    """Export-friendly RMSNormGated: avoids explicit .float() / .type_as() casts.

    Uses F.rms_norm which maps to fast::rms_norm (handles precision internally)
    and F.silu which also handles bf16 natively in MLX.
    """
    return torch.nn.functional.rms_norm(
        x, (x.shape[-1],), self.weight, self.eps
    ) * torch.nn.functional.silu(z)


def _gemma_rms_norm_forward(self, x):
    """Export-friendly GemmaRMSNorm: avoids explicit .float() / .type_as() casts.

    The original does x.float() → normalize → (1+weight).float() → type_as,
    producing 2+ AsType nodes per norm. F.rms_norm handles precision internally.
    The (1+weight) offset is precomputed by the swap code below.
    """
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), self._rms_weight, self.eps)


def _sparse_moe_forward(self, x):
    """Export-friendly SparseMoE: removes .float() on expert_weights.

    The original passes expert_weights.float() to the experts, causing
    bf16→f32 casts. GatherMm/GatherQmm handle bf16 weights natively.
    """
    B, T, C = x.size()
    x_flat = x.view(-1, C)

    scores = self.gate(x_flat)
    expert_weights, expert_indices = torch.topk(scores, self.top_k, dim=-1)
    expert_weights = expert_weights.softmax(dim=-1)

    routed_out = self.experts(
        x_flat,
        expert_weights,
        expert_indices,
        self.top_k,
        sort_experts=getattr(self, "_sort_experts", False),
    )

    shared_out = self.shared_expert(x_flat)
    shared_gate = torch.sigmoid(self.shared_expert_gate(x_flat))
    return (routed_out + shared_gate * shared_out).view(B, T, C)


def _full_attention_forward(self, x, input_pos):
    """Export-friendly FullAttention: uses mlx::rope custom op.

    Replaces the decomposed RotaryEmbedding (~14 ops: outer, cos, sin, slice,
    multiply, subtract, cat, AsType) with 2 RopeNode ops that fuse to
    fast::rope. Also removes unnecessary .to(dtype) casts.
    """
    B, T, _ = x.size()

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

    # Transpose to BHSD before RoPE (mlx::rope expects B,H,T,D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Use mlx::rope custom op — fuses to a single RopeNode per tensor,
    # replacing ~14 decomposed ops (outer, cos, sin, slice, mul, cat, etc.)
    pos = input_pos[0].item()
    q = torch.ops.mlx.rope(q, self._rope_dims, pos, False, self._rope_base, 1.0, None)
    k = torch.ops.mlx.rope(k, self._rope_dims, pos, False, self._rope_base, 1.0, None)

    k, v = self.kv_cache.update(input_pos, k, v)

    if self.n_kv_groups > 1:
        k = k.repeat_interleave(self.n_kv_groups, dim=1)
        v = v.repeat_interleave(self.n_kv_groups, dim=1)

    attn_mask = self.mask[input_pos].unsqueeze(0).unsqueeze(0)
    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

    y = y.transpose(1, 2).contiguous().view(B, T, -1)

    gate = gate.reshape(B, T, -1)
    y = y * torch.sigmoid(gate)

    return self.o_proj(y)


def _exportable_gated_delta_net_forward(self, x, input_pos):
    """Pure PyTorch replacement for GatedDeltaNet.forward().

    Identical pre/post-processing to the original, but replaces
    torch.ops.triton.chunk_gated_delta_rule with a pure PyTorch
    recurrent implementation via mlx::gated_delta_rule custom op.
    """

    B, T, _ = x.size()

    # Reset state at position 0 (in-place to preserve buffer identity)
    reset = (input_pos[0] == 0).to(self.conv_state.dtype)
    keep = 1.0 - reset
    self.conv_state.mul_(keep)
    self.recurrent_state.mul_(keep)

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
    qkv_t = mixed_qkv.transpose(1, 2)  # [B, C, T]
    conv_input = torch.cat([self.conv_state[:B], qkv_t], dim=-1)
    conv_len = conv_input.shape[-1]
    # Update conv_state in-place to preserve buffer identity
    # (attribute reassignment would break mutation tracking)
    self.conv_state[:B].copy_(conv_input[:, :, conv_len - self.conv_kernel_size :])

    conv_out = torch.nn.functional.conv1d(
        conv_input, self.conv1d.weight, groups=self.conv_dim
    )
    conv_start = conv_out.shape[-1] - T
    qkv_conv = torch.nn.functional.silu(conv_out[:, :, conv_start:]).transpose(1, 2)

    # Split into Q, K, V
    kd = self.key_dim
    q = qkv_conv[..., :kd].reshape(B, T, self.num_k_heads, self.head_k_dim)
    k = qkv_conv[..., kd : 2 * kd].reshape(B, T, self.num_k_heads, self.head_k_dim)
    v = qkv_conv[..., 2 * kd :].reshape(B, T, self.num_v_heads, self.head_v_dim)

    # RMS-normalize Q and K with asymmetric scaling
    # Uses pre-registered _qk_rms_weight (bf16 ones) so rms_norm returns bf16
    inv_scale = torch.tensor(self.head_k_dim**-0.5, dtype=x.dtype)
    q = (inv_scale * inv_scale) * torch.nn.functional.rms_norm(
        q, (self.head_k_dim,), self._qk_rms_weight, eps=1e-6
    )
    k = inv_scale * torch.nn.functional.rms_norm(
        k, (self.head_k_dim,), self._qk_rms_weight, eps=1e-6
    )

    # head_repeat for k_heads != v_heads
    if self.head_repeat > 1:
        q = q.repeat_interleave(self.head_repeat, dim=2)
        k = k.repeat_interleave(self.head_repeat, dim=2)

    # Mamba-style gating
    beta = b.sigmoid()
    x = a + self.dt_bias
    g = (-self.A_log.exp() * torch.logaddexp(x, torch.zeros_like(x))).exp()

    import executorch.backends.mlx.model_ops.gated_delta_rule as _  # noqa: ensure op registered

    output = torch.ops.mlx.gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        self.recurrent_state[:B],
    )

    # RMSNorm(output) * silu(z) → out_proj
    # output shape from exportable fn: (B, T, num_v_heads * head_v_dim)
    output = output.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    output = self.norm(output, z)
    output = output.reshape(B, T, -1)

    return self.out_proj(output)


def _swap_moe_experts(model, fuse_gate_up):
    """FusedMoEExperts → SwitchMLP."""
    from executorch.backends.mlx.llm.switch import SwitchMLP

    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, FusedMoEExperts):
            continue

        switch_mlp = SwitchMLP(
            module.hidden_size,
            module.intermediate_size,
            module.num_experts,
            fuse_gate_up=fuse_gate_up,
        )
        switch_mlp.to(dtype=module.w1_weight.dtype)

        inter = module.intermediate_size
        with torch.no_grad():
            if fuse_gate_up:
                for e in range(module.num_experts):
                    switch_mlp.gate_up_proj.experts[e].weight.copy_(module.w1_weight[e])
                    switch_mlp.down_proj.experts[e].weight.copy_(module.w2_weight[e])
            else:
                for e in range(module.num_experts):
                    switch_mlp.gate_proj.experts[e].weight.copy_(
                        module.w1_weight[e, :inter, :]
                    )
                    switch_mlp.up_proj.experts[e].weight.copy_(
                        module.w1_weight[e, inter:, :]
                    )
                    switch_mlp.down_proj.experts[e].weight.copy_(module.w2_weight[e])

        parts = name.rsplit(".", 1)
        if len(parts) == 1:
            setattr(model, parts[0], switch_mlp)
        else:
            parent = model.get_submodule(parts[0])
            setattr(parent, parts[1], switch_mlp)
        count += 1
    return count


def _swap_gated_delta_net(model, model_dtype):
    """GatedDeltaNet → mlx::gated_delta_rule custom op."""
    count = 0
    for _name, module in model.named_modules():
        if isinstance(module, GatedDeltaNet):
            module.forward = types.MethodType(
                _exportable_gated_delta_net_forward, module
            )
            if module.recurrent_state.dtype != model_dtype:
                module.recurrent_state = module.recurrent_state.to(model_dtype)
            module.norm.forward = types.MethodType(_rms_norm_gated_forward, module.norm)
            module.register_buffer(
                "_qk_rms_weight",
                torch.ones(module.head_k_dim, dtype=model_dtype),
            )
            count += 1
    return count


def _swap_full_attention(model, config):
    """FullAttention → mlx::rope custom op + causal mask."""
    rope_theta = config.rope_theta if config else 10000.0
    max_seq_len = config.max_seq_len if config else 4096
    count = 0
    for _name, module in model.named_modules():
        if isinstance(module, FullAttention):
            module._rope_dims = module.rotary_emb.rotary_dim
            module._rope_base = rope_theta
            mask = torch.full((max_seq_len, max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            module.register_buffer("mask", mask)
            module.forward = types.MethodType(_full_attention_forward, module)
            count += 1
    return count


def _swap_kv_cache(model, model_dtype):
    """KVCache → MLX KVCache (mlx::kv_cache_update)."""
    from executorch.backends.mlx.llm.cache import KVCache as MLXKVCache

    count = 0
    for _name, module in model.named_modules():
        if hasattr(module, "kv_cache") and isinstance(module.kv_cache, KVCache):
            old = module.kv_cache
            module.kv_cache = MLXKVCache(
                max_batch_size=1,
                max_context_length=old.k_cache.shape[2],
                n_heads=old.k_cache.shape[1],
                head_dim=old.k_cache.shape[3],
                enable_dynamic_shape=True,
                dtype=model_dtype,
            )
            count += 1
    return count


def _swap_rms_norm(model):
    """GemmaRMSNorm → F.rms_norm (no .float() casts)."""
    count = 0
    for _name, module in model.named_modules():
        if isinstance(module, GemmaRMSNorm):
            module._rms_weight = nn.Parameter(1.0 + module.weight.data)
            module.forward = types.MethodType(_gemma_rms_norm_forward, module)
            count += 1
    return count


def _swap_sparse_moe(model, sort_experts):
    """SparseMoE → no .float() on expert_weights."""
    count = 0
    for _name, module in model.named_modules():
        if isinstance(module, SparseMoE):
            module._sort_experts = sort_experts
            module.forward = types.MethodType(_sparse_moe_forward, module)
            count += 1
    return count


def mlx_source_transformations(
    model,
    model_dtype=torch.bfloat16,
    config=None,
    sort_experts=False,
    fuse_gate_up=False,
):
    """Replace all Triton-dependent modules with MLX-compatible equivalents.

    Performs the following transformations:
      1. FusedMoEExperts → SwitchMLP (uses mlx::gather_mm / mlx::gather_qmm)
      2. GatedDeltaNet → mlx::gated_delta_rule custom op
      3. FullAttention → mlx::rope custom op
      4. KVCache → MLX KVCache (mlx::kv_cache_update)
      5. GemmaRMSNorm → F.rms_norm (no .float() casts)
      6. SparseMoE → no .float() on expert_weights

    Args:
        model: The Qwen 3.5 MoE model to transform.
        model_dtype: Target dtype for the model (default: bf16).
        config: Model config (Qwen35MoEConfig).
        sort_experts: Sort tokens by expert index for coalesced memory access.
        fuse_gate_up: Fuse gate+up into single SwitchLinear.
    """
    count_moe = _swap_moe_experts(model, fuse_gate_up)
    count_gdn = _swap_gated_delta_net(model, model_dtype)
    count_attn = _swap_full_attention(model, config)
    count_kv = _swap_kv_cache(model, model_dtype)
    count_norm = _swap_rms_norm(model)
    count_moe_fwd = _swap_sparse_moe(model, sort_experts)

    logger.info(f"Replaced {count_moe} FusedMoEExperts → SwitchMLP")
    logger.info(f"Replaced {count_gdn} GatedDeltaNet → exportable PyTorch forward")
    logger.info(f"Replaced {count_attn} FullAttention → mlx::rope + causal mask")
    logger.info(f"Replaced {count_kv} KVCache → MLX KVCache (mlx::kv_cache_update)")
    logger.info(f"Replaced {count_norm} GemmaRMSNorm → F.rms_norm (no .float() casts)")
    logger.info(f"Replaced {count_moe_fwd} SparseMoE → no .float() on expert_weights")

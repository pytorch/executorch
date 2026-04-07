#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export Qwen 3.5 MoE to MLX backend — dry-run test (no weight download).

Creates a tiny Qwen 3.5 MoE model with random weights and runs it through
the MLX export pipeline to validate op support. No HuggingFace download or
CUDA GPU is required.

The model's two Triton-dependent modules are replaced with pure PyTorch:
  - FusedMoEExperts  → unfused forward using index_select + bmm + silu
  - GatedDeltaNet    → exportable_recurrent_gated_delta_rule (pure PyTorch)

Usage:
    # Dry-run (validates export pipeline, prints op support summary):
    python -m executorch.backends.mlx.examples.qwen3_5_moe.export_qwen35_moe

    # Save .pte:
    python -m executorch.backends.mlx.examples.qwen3_5_moe.export_qwen35_moe \
        --output qwen35_moe_mlx.pte
"""

import argparse
import logging
import os
import types

import torch
import torch.nn as nn
from torch.nn import functional as F

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tiny config for testing (no download needed)
# ---------------------------------------------------------------------------

from executorch.examples.models.qwen3_5_moe.model import (
    FullAttention,
    FusedMoEExperts,
    GatedDeltaNet,
    GemmaRMSNorm,
    KVCache,
    Qwen35MoE,
    Qwen35MoEConfig,
    RotaryEmbedding,
    SparseMoE,
)

TINY_CONFIG = Qwen35MoEConfig(
    vocab_size=256,
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_kv_heads=2,
    head_dim=64,
    partial_rotary_factor=0.25,
    linear_num_key_heads=1,
    linear_num_value_heads=2,
    linear_key_head_dim=64,
    linear_value_head_dim=64,
    linear_conv_kernel_dim=4,
    num_experts=4,
    num_experts_per_tok=2,
    moe_intermediate_size=128,
    shared_expert_intermediate_size=128,
    full_attention_interval=2,
    rms_norm_eps=1e-6,
    rope_theta=10_000.0,
    max_seq_len=64,
)


# ---------------------------------------------------------------------------
# Source transformations: replace Triton ops with pure PyTorch
# ---------------------------------------------------------------------------


class UnfusedMoEExperts(nn.Module):
    """Export-friendly MoE experts using SwitchLinear for each projection.

    Replaces FusedMoEExperts (which uses Triton fused MoE kernel) with
    three SwitchLinear modules (gate_proj, up_proj, down_proj) that use
    mlx::gather_mm / mlx::gather_qmm custom ops.

    The original fused gate+up weight [E, 2*inter, D] is split into
    separate gate [E, inter, D] and up [E, inter, D] projections.
    """

    def __init__(self, num_experts, hidden_size, intermediate_size, sort_experts=False):
        super().__init__()
        from executorch.backends.mlx.llm.switch import SwitchLinear

        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.sort_experts = sort_experts

        self.gate_proj = SwitchLinear(hidden_size, intermediate_size, num_experts)
        self.up_proj = SwitchLinear(hidden_size, intermediate_size, num_experts)
        self.down_proj = SwitchLinear(intermediate_size, hidden_size, num_experts)

    def forward(self, x, expert_weights, expert_indices, top_k):
        # Pass all top_k indices at once (matches mlx-lm's SwitchGLU).
        # Instead of looping top_k=8 times with 3 gather_qmm each (24 total),
        # we do 3 gather_qmm calls with [N, top_k] indices.
        #
        # x: [N, D]
        # expert_weights: [N, top_k]  (already softmaxed)
        # expert_indices: [N, top_k]
        #
        # After unsqueeze: x → [N, 1, 1, D]
        # gather_qmm with indices [N, top_k] → [N, top_k, 1, out]

        N = x.shape[0]
        do_sort = self.sort_experts

        if do_sort:
            # Sort by expert index for coalesced memory access during prefill.
            # Matches mlx-lm's _gather_sort: flatten [N, top_k] → [N*top_k],
            # argsort, gather x in sorted order.
            # All index tensors must be int32 for MLX gather ops.
            flat_indices = expert_indices.flatten()  # [N*top_k]
            order = flat_indices.argsort().to(torch.int32)
            inv_order = order.argsort().to(torch.int32)
            sorted_idx = flat_indices[order].to(torch.int32)  # [N*top_k]
            # Gather x rows corresponding to each sorted index
            x_sorted = x[(order // top_k).to(torch.int64)]  # [N*top_k, D]
            x_input = x_sorted.unsqueeze(-2)  # [N*top_k, 1, D]
            idx = sorted_idx
        else:
            x_input = x.unsqueeze(-2).unsqueeze(-2)  # [N, 1, 1, D]
            idx = expert_indices

        gate = self.gate_proj.forward(x_input, idx, sorted_indices=do_sort)
        up = self.up_proj.forward(x_input, idx, sorted_indices=do_sort)
        h = torch.nn.functional.silu(gate) * up
        down = self.down_proj.forward(h, idx, sorted_indices=do_sort)

        if do_sort:
            # down: [N*top_k, 1, D] → squeeze → unsort → reshape to [N, top_k, D]
            down = down.squeeze(-2)  # [N*top_k, D]
            down = down[inv_order].reshape(N, top_k, -1)  # [N, top_k, D]
        else:
            down = down.squeeze(-2)  # [N, top_k, D]

        # Weighted sum across experts
        return (down * expert_weights.unsqueeze(-1)).sum(dim=-2)  # [N, D]


def _pack_all_experts(model):
    """Call pack() on all SwitchLinear modules in the model."""
    from executorch.backends.mlx.llm.switch import pack_all_switch_linears

    pack_all_switch_linears(model)


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
    recurrent implementation.
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
    # Use native F.conv1d — maps to fused Conv1DNode (1 op instead of 12)
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

    # RMS-normalize Q and K with asymmetric scaling (matches mlx-lm exactly)
    # mlx-lm: q = (1/Dk) * rms_norm(q), k = (1/√Dk) * rms_norm(k)
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
    # Use logaddexp(x, 0) for softplus — maps to a single LogAddExpNode.
    # torch.nn.functional.softplus decomposes into log(1+exp(x)) with f32
    # lifted constants, promoting bf16 tensors to f32 and generating ~90
    # unnecessary AsType nodes. logaddexp(x, 0) = log(exp(x) + 1) = softplus(x)
    # and stays in bf16. This matches mlx-lm's nn.softplus which is mx.logaddexp.
    x = a + self.dt_bias
    g = (-self.A_log.exp() * torch.logaddexp(x, torch.zeros_like(x))).exp()

    # Gated delta rule recurrence via custom op (replaces Python loop)
    # Use bf16 inputs directly — matches mlx-lm's decomposed ops path.
    # MLX auto-promotes when mixing bf16 activations with bf16 state.
    import executorch.backends.mlx.examples.qwen3_5_moe.gated_delta_rule as _  # noqa: ensure op registered

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


def replace_triton_ops(
    model,
    model_dtype=torch.bfloat16,
    config=None,
    sort_experts=False,
    fuse_gate_up=False,
):
    """Replace all Triton-dependent modules with pure PyTorch equivalents."""
    from executorch.backends.mlx.llm.cache import KVCache as MLXKVCache

    count_moe = 0
    count_gdn = 0
    count_kv = 0

    # Swap FusedMoEExperts → SwitchMLP
    for name, module in model.named_modules():
        if not isinstance(module, FusedMoEExperts):
            continue
        from executorch.backends.mlx.llm.switch import SwitchMLP

        switch_mlp = SwitchMLP(
            module.hidden_size,
            module.intermediate_size,
            module.num_experts,
            fuse_gate_up=fuse_gate_up,
        )
        # Match dtype of original weights
        switch_mlp.to(dtype=module.w1_weight.dtype)

        inter = module.intermediate_size
        with torch.no_grad():
            if fuse_gate_up:
                # w1_weight [E, 2*inter, D] goes directly into fused gate_up_proj
                for e in range(module.num_experts):
                    switch_mlp.gate_up_proj.experts[e].weight.copy_(module.w1_weight[e])
                    switch_mlp.down_proj.experts[e].weight.copy_(module.w2_weight[e])
            else:
                # Split fused w1_weight [E, 2*inter, D] into gate + up
                for e in range(module.num_experts):
                    switch_mlp.gate_proj.experts[e].weight.copy_(
                        module.w1_weight[e, :inter, :]
                    )
                    switch_mlp.up_proj.experts[e].weight.copy_(
                        module.w1_weight[e, inter:, :]
                    )
                    switch_mlp.down_proj.experts[e].weight.copy_(module.w2_weight[e])

        # Replace on parent
        parts = name.rsplit(".", 1)
        if len(parts) == 1:
            setattr(model, parts[0], switch_mlp)
        else:
            parent = model.get_submodule(parts[0])
            setattr(parent, parts[1], switch_mlp)
        count_moe += 1

    for _name, module in model.named_modules():
        if isinstance(module, GatedDeltaNet):
            module.forward = types.MethodType(
                _exportable_gated_delta_net_forward, module
            )
            # TODO: Test if A_log/dt_bias need fp32 precision with chat template fix
            # Use bf16 for recurrent_state — matches mlx-lm's decomposed ops path.
            # MLX element-wise ops handle bf16 natively.
            if module.recurrent_state.dtype != model_dtype:
                module.recurrent_state = module.recurrent_state.to(model_dtype)
            # if module.A_log.dtype != torch.float32:
            #     module.A_log.data = module.A_log.data.float()
            # if module.dt_bias.dtype != torch.float32:
            #     module.dt_bias.data = module.dt_bias.data.float()
            # Swap RMSNormGated.forward to avoid explicit .float()/.type_as() casts.
            # fast::rms_norm and silu handle bf16 natively.
            module.norm.forward = types.MethodType(_rms_norm_gated_forward, module.norm)
            # Register bf16 ones weight for Q/K rms_norm so it returns bf16
            # (weight=None causes rms_norm to return f32 from decomposition)
            module.register_buffer(
                "_qk_rms_weight",
                torch.ones(module.head_k_dim, dtype=model_dtype),
            )
            count_gdn += 1

    # Swap KVCache → MLX KVCache (uses mlx::kv_cache_update custom op)
    count_attn = 0
    rope_theta = config.rope_theta if config else 10000.0
    for _name, module in model.named_modules():
        if isinstance(module, FullAttention):
            # Store rope params for the swapped forward
            module._rope_dims = module.rotary_emb.rotary_dim
            module._rope_base = rope_theta
            # Swap FullAttention.forward to use mlx::rope custom op
            module.forward = types.MethodType(_full_attention_forward, module)
            count_attn += 1

    for _name, module in model.named_modules():
        if hasattr(module, "kv_cache") and isinstance(module.kv_cache, KVCache):
            old = module.kv_cache
            mlx_cache = MLXKVCache(
                max_batch_size=1,
                max_context_length=old.k_cache.shape[2],
                n_heads=old.k_cache.shape[1],
                head_dim=old.k_cache.shape[3],
                enable_dynamic_shape=True,
                dtype=model_dtype,  # match model dtype, not cache default (fp32)
            )
            module.kv_cache = mlx_cache
            count_kv += 1

    # Swap GemmaRMSNorm.forward to avoid .float()/.type_as() casts.
    # Precompute (1 + weight) so F.rms_norm can use it directly.
    count_norm = 0
    for _name, module in model.named_modules():
        if isinstance(module, GemmaRMSNorm):
            module._rms_weight = nn.Parameter(1.0 + module.weight.data)
            module.forward = types.MethodType(_gemma_rms_norm_forward, module)
            count_norm += 1

    # Swap SparseMoE.forward to remove .float() on expert_weights.
    # GatherMm/GatherQmm handle bf16 weights natively.
    count_moe_fwd = 0
    for _name, module in model.named_modules():
        if isinstance(module, SparseMoE):
            module._sort_experts = sort_experts
            module.forward = types.MethodType(_sparse_moe_forward, module)
            count_moe_fwd += 1

    logger.info(f"Replaced {count_moe} FusedMoEExperts → UnfusedMoEExperts (nn.Linear)")
    logger.info(f"Replaced {count_gdn} GatedDeltaNet → exportable PyTorch forward")
    logger.info(f"Replaced {count_kv} KVCache → MLX KVCache (mlx::kv_cache_update)")
    logger.info(f"Replaced {count_norm} GemmaRMSNorm → F.rms_norm (no .float() casts)")
    logger.info(f"Replaced {count_moe_fwd} SparseMoE → no .float() on expert_weights")


# ---------------------------------------------------------------------------
# Build and export
# ---------------------------------------------------------------------------


def build_tiny_model(config=None, dtype=torch.bfloat16):
    """Build a tiny Qwen 3.5 MoE model with random weights on CPU.

    No HuggingFace download needed — constructs from config with random init.
    """
    if config is None:
        config = TINY_CONFIG

    logger.info("Building tiny model with random weights...")
    torch.manual_seed(42)
    model = Qwen35MoE(config)
    model.to(dtype=dtype)

    for p in model.parameters():
        if p.device.type != "meta":
            p.data.normal_(0, 0.02)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Model: {config.num_hidden_layers} layers, {config.hidden_size}d, "
        f"{config.num_experts} experts top-{config.num_experts_per_tok}, "
        f"{param_count:,} params"
    )
    logger.info(f"Layer types: {config.layer_types}")

    return model, config


class GreedyArgmaxWrapper(nn.Module):
    """Wraps a model to append argmax on the last token's logits.

    Instead of returning full logits [B, T, vocab_size] (~500KB for 248K vocab),
    returns just the token id [B, 1] (8 bytes). This eliminates the expensive
    GPU→CPU logits copy during decode.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, input_pos):
        logits = self.model(tokens, input_pos)
        # Take last token's logits and argmax
        next_token = logits[:, -1:, :].argmax(dim=-1)
        return next_token


def export_to_mlx(model, config, output_path=None):
    """Export model through the full MLX pipeline.

    Steps:
      1. torch.export with fixed shapes (seq_len=2)
      2. Free the original model (exported_program captures all weights)
      3. to_edge_transform_and_lower with MLXPartitioner
      4. to_executorch
      5. Optionally save .pte + .ptd
    """
    import gc
    import time

    import executorch.exir as exir
    from executorch.backends.mlx import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.exir import EdgeCompileConfig
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes import MemoryPlanningPass

    timings = {}

    seq_len = 2
    example_tokens = torch.zeros((1, seq_len), dtype=torch.long)
    example_input_pos = torch.arange(seq_len, dtype=torch.long)

    # Dynamic shapes — seq_len is dynamic now that the GatedDeltaNet
    # recurrence loop is inside the mlx::gated_delta_rule custom op
    # (ScanNode handles dynamic T at runtime).
    max_seq_len = config.max_seq_len
    seq_len_dim = torch.export.Dim("seq_length", max=max_seq_len - 1)
    dynamic_shapes = {
        "tokens": {1: seq_len_dim},
        "input_pos": {0: seq_len_dim},
    }

    # --- Step 1: torch.export ---
    logger.info(f"Running torch.export (seq_len={seq_len}, dynamic shapes enabled)...")
    t0 = time.time()
    with torch.no_grad():
        exported = torch.export.export(
            model,
            (example_tokens, example_input_pos),
            dynamic_shapes=dynamic_shapes,
            strict=True,
        )
    timings["torch.export"] = time.time() - t0
    logger.info(f"torch.export succeeded! ({timings['torch.export']:.1f}s)")

    # Free the original model — exported_program captures all weight tensors.
    del model
    gc.collect()
    logger.info("Freed model memory")

    graph = exported.graph
    op_counts = {}
    for node in graph.nodes:
        if node.op == "call_function":
            target = str(node.target)
            op_counts[target] = op_counts.get(target, 0) + 1
    logger.info(
        f"Exported graph: {len(graph.nodes)} nodes, {len(op_counts)} unique ops"
    )
    for op, count in sorted(op_counts.items()):
        logger.info(f"  {op}: {count}")

    # --- Step 2: Lower to MLX ---
    # Importing gated_delta_rule registers both the custom op and pattern handler
    import executorch.backends.mlx.examples.qwen3_5_moe.gated_delta_rule as _gdr  # noqa: F401

    logger.info("Lowering to ExecuTorch with MLX backend...")
    try:
        import cProfile
        import io as _io
        import pstats

        profiler = cProfile.Profile()
        t0 = time.time()
        profiler.enable()
        edge_program = exir.to_edge_transform_and_lower(
            {"forward": exported},
            transform_passes=get_default_passes(),
            partitioner=[MLXPartitioner()],
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
                _skip_dim_order=True,
            ),
        )
        profiler.disable()
        timings["to_edge_transform_and_lower"] = time.time() - t0
        logger.info(
            f"Lowering complete ({timings['to_edge_transform_and_lower']:.1f}s)"
        )

        # Print profile results
        s = _io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats("cumulative").print_stats(40)
        logger.info(f"Profile (top 40 by cumulative time):\n{s.getvalue()}")

        s = _io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats("tottime").print_stats(40)
        logger.info(f"Profile (top 40 by internal time):\n{s.getvalue()}")
    except Exception as e:
        import traceback

        logger.error(f"Partitioning failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None

    # Free exported program — edge_program has its own copy of the graph.
    del exported
    gc.collect()
    logger.info("Freed exported program memory")

    # --- Step 3: Finalize ---
    logger.info("Generating ExecuTorch program...")
    t0 = time.time()
    executorch_program = edge_program.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        )
    )
    timings["to_executorch"] = time.time() - t0
    logger.info(f"to_executorch complete ({timings['to_executorch']:.1f}s)")

    # Free edge program before writing to disk.
    del edge_program
    gc.collect()

    if output_path:
        output_dir = os.path.dirname(output_path) or "."
        os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "wb") as f:
            executorch_program.write_to_file(f)
        pte_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"Saved {output_path} ({pte_mb:.2f} MB)")

        if executorch_program._tensor_data:
            executorch_program.write_tensor_data_to_file(output_dir)
            logger.info(f"Saved tensor data to {output_dir}/")

    logger.info("Export complete!")
    return executorch_program


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Export Qwen 3.5 MoE to MLX backend (dry-run, no downloads)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to HuggingFace checkpoint directory (e.g. ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/...)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="Maximum sequence length for KV cache and attention mask (default: 4096)",
    )
    parser.add_argument(
        "--sort-experts",
        action="store_true",
        default=False,
        help="Sort tokens by expert index before gather_qmm for coalesced memory "
        "access during prefill. Matches mlx-lm's presorting optimization. "
        "No effect on decode (single token).",
    )
    parser.add_argument(
        "--fuse-gate-up",
        action="store_true",
        default=False,
        help="Fuse gate and up projections into a single SwitchLinear with "
        "output dim 2*intermediate_size. Reduces gather_qmm calls from 3 "
        "to 2 per MoE layer. May not improve performance on all hardware.",
    )
    parser.add_argument(
        "--sampling",
        type=str,
        choices=["none", "greedy"],
        default="none",
        help="Include sampling in the model graph. 'greedy' appends argmax "
        "so the model returns a token id instead of full logits (eliminates "
        "large GPU→CPU logits copy during decode). Default: none.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .pte file path (optional, omit for dry-run)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp32", "bf16"],
        default="bf16",
        help="Model dtype (default: bf16). When using --qlinear, "
        "this is the dtype for non-quantized parameters only.",
    )
    parser.add_argument(
        "--load-dtype",
        type=str,
        choices=["fp32", "bf16"],
        default=None,
        help="Dtype for initial weight loading (default: same as --dtype). "
        "Useful for loading in bf16 to save RAM then casting to fp32 after quantization.",
    )

    from executorch.backends.mlx.llm.quantization import add_quantization_args

    add_quantization_args(parser)

    # Override default group size to 64 to match mlx-lm (better GPU tiling)
    for action in parser._actions:
        if hasattr(action, "dest") and action.dest == "qlinear_group_size":
            action.default = 64
            break

    args = parser.parse_args()

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16}
    model_dtype = dtype_map[args.dtype]
    load_dtype = dtype_map[args.load_dtype] if args.load_dtype else model_dtype

    if args.model_path:
        logger.info(f"Loading model from {args.model_path}...")
        model, config = Qwen35MoE.from_hf_checkpoint(
            args.model_path, max_seq_len=args.max_seq_len
        )
        model.to(dtype=load_dtype)
        logger.info(f"Loaded model in {args.load_dtype or args.dtype}")
        # Materialize any buffers still on meta device (conv_state, recurrent_state, etc.)
        for name, buf in model.named_buffers():
            if buf.device.type == "meta":
                model_part = model
                parts = name.split(".")
                for p in parts[:-1]:
                    model_part = getattr(model_part, p)
                setattr(model_part, parts[-1], torch.zeros_like(buf, device="cpu"))
    else:
        model, config = build_tiny_model(dtype=load_dtype)

    replace_triton_ops(
        model,
        model_dtype=model_dtype,
        config=config,
        sort_experts=args.sort_experts,
        fuse_gate_up=args.fuse_gate_up,
    )

    # Quantize if requested (before upcasting — quantization compresses weights)
    if args.qlinear or args.qembedding:
        from executorch.backends.mlx.llm.quantization import quantize_model_

        logger.info(
            f"Quantizing model (qlinear={args.qlinear}, qembedding={args.qembedding})..."
        )
        quantize_model_(
            model,
            qlinear_config=args.qlinear,
            qlinear_group_size=args.qlinear_group_size,
            qembedding_config=args.qembedding,
            qembedding_group_size=args.qembedding_group_size,
        )
        logger.info("Quantization done")

    # Upcast non-quantized parameters to target dtype if different from load dtype.
    # This lets us load in bf16 (saves RAM), quantize (compresses most weights),
    # then cast remaining params (norms, A_log, dt_bias, etc.) to fp32.
    # Quantized tensors support .to() — it changes dequant dtype and scale dtype.
    if model_dtype != load_dtype:
        logger.info(f"Casting model from {args.load_dtype} to {args.dtype}...")
        model.to(dtype=model_dtype)
        logger.info("Cast done")

    import gc

    # Free old parameter tensors from quantization before packing
    gc.collect()

    # Pack expert weights into stacked buffers (works for both quantized and unquantized)
    _pack_all_experts(model)

    # Free per-expert nn.Linear modules after stacking
    gc.collect()

    # Verify forward works before export
    logger.info("Verifying forward pass (eager, before export)...")
    with torch.no_grad():
        tokens = torch.zeros((1, 2), dtype=torch.long)
        input_pos = torch.arange(2, dtype=torch.long)
        output = model(tokens, input_pos)
        logger.info(
            f"Forward pass OK: output shape {output.shape}, dtype {output.dtype}"
        )

    # Wrap model with greedy argmax if requested
    if args.sampling == "greedy":
        logger.info("Wrapping model with GreedyArgmaxWrapper (argmax in graph)")
        model = GreedyArgmaxWrapper(model)

    # Pass model to export and delete our reference so it can be freed inside
    _model = model
    del model
    export_to_mlx(_model, config, output_path=args.output)


if __name__ == "__main__":
    main()

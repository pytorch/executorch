"""
Export-friendly Qwen 3.5 MoE (35B-A3B) model.

Self-contained model definition for torch.export(strict=True) with CUDA backend.
All stateful buffers (KV cache, conv_state, recurrent_state) are registered buffers.

Architecture: 40-layer hybrid transformer with 75% GatedDeltaNet (linear attention)
and 25% full softmax attention, with 256 routed experts (top-8) plus shared expert.

Reference implementations:
  - https://github.com/mergennachin/nano_qwen35_moe/ (export patterns)
  - transformers: models/qwen3_5_moe/modeling_qwen3_5_moe.py (HF reference)
  - vLLM: vllm/model_executor/models/qwen3_5.py (architecture details)
"""

import json
import os
import re
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Config


@dataclass
class Qwen35MoEConfig:
    vocab_size: int = 248320
    hidden_size: int = 2048
    num_hidden_layers: int = 40
    num_attention_heads: int = 16
    num_kv_heads: int = 2
    head_dim: int = 256
    partial_rotary_factor: float = 0.25
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    num_experts: int = 256
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    full_attention_interval: int = 4
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000_000.0
    max_seq_len: int = 4096
    layer_types: list = field(default_factory=list)

    def __post_init__(self):
        if not self.layer_types:
            self.layer_types = [
                (
                    "full_attention"
                    if (i + 1) % self.full_attention_interval == 0
                    else "linear_attention"
                )
                for i in range(self.num_hidden_layers)
            ]

    @staticmethod
    def from_hf_config(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
        # Handle multimodal config (text_config nested)
        if "text_config" in cfg:
            cfg = cfg["text_config"]
        # rope_theta and partial_rotary_factor may be nested in rope_parameters
        rope_params = cfg.get("rope_parameters", {})
        return Qwen35MoEConfig(
            vocab_size=cfg.get("vocab_size", 248320),
            hidden_size=cfg.get("hidden_size", 2048),
            num_hidden_layers=cfg.get("num_hidden_layers", 40),
            num_attention_heads=cfg.get("num_attention_heads", 16),
            num_kv_heads=cfg.get("num_key_value_heads", 2),
            head_dim=cfg.get("head_dim", 256),
            partial_rotary_factor=cfg.get(
                "partial_rotary_factor", rope_params.get("partial_rotary_factor", 0.25)
            ),
            linear_num_key_heads=cfg.get("linear_num_key_heads", 16),
            linear_num_value_heads=cfg.get("linear_num_value_heads", 32),
            linear_key_head_dim=cfg.get("linear_key_head_dim", 128),
            linear_value_head_dim=cfg.get("linear_value_head_dim", 128),
            linear_conv_kernel_dim=cfg.get("linear_conv_kernel_dim", 4),
            num_experts=cfg.get("num_experts", 256),
            num_experts_per_tok=cfg.get("num_experts_per_tok", 8),
            moe_intermediate_size=cfg.get("moe_intermediate_size", 512),
            shared_expert_intermediate_size=cfg.get(
                "shared_expert_intermediate_size", 512
            ),
            full_attention_interval=cfg.get("full_attention_interval", 4),
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
            rope_theta=cfg.get(
                "rope_theta", rope_params.get("rope_theta", 10_000_000.0)
            ),
            layer_types=cfg.get("layer_types"),
        )


# ---------------------------------------------------------------------------
# Normalization


class GemmaRMSNorm(nn.Module):
    """RMSNorm with unit offset: x * (1 + weight)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        x_float = x.float()
        normed = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * (1.0 + self.weight.float())).type_as(x)


class RMSNormGated(nn.Module):
    """RMSNorm(x) * silu(z) — used in GatedDeltaNet output."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x, z):
        x_float = x.float()
        normed = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        normed = self.weight * normed.type_as(x)
        return (normed * F.silu(z.float())).type_as(x)


# ---------------------------------------------------------------------------
# Rotary Position Embedding (partial)


class RotaryEmbedding(nn.Module):
    """Partial RoPE — only rotates first `rotary_dim` dimensions of each head."""

    def __init__(self, head_dim, partial_rotary_factor, rope_theta):
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            rope_theta
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
                / self.rotary_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, positions, q, k):
        # q: (B, T, n_heads, head_dim), k: (B, T, n_kv_heads, head_dim)
        freqs = torch.outer(positions.float(), self.inv_freq)
        cos = freqs.cos().unsqueeze(1)  # (T, 1, rotary_dim/2)
        sin = freqs.sin().unsqueeze(1)

        q_rot, q_pass = q[..., : self.rotary_dim], q[..., self.rotary_dim :]
        k_rot, k_pass = k[..., : self.rotary_dim], k[..., self.rotary_dim :]

        q_rot = self._apply_rotary(q_rot, cos, sin)
        k_rot = self._apply_rotary(k_rot, cos, sin)

        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)
        return q, k

    @staticmethod
    def _apply_rotary(x, cos, sin):
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ---------------------------------------------------------------------------
# KV Cache


class KVCache(nn.Module):

    def __init__(self, n_kv_heads, head_dim, max_seq_len):
        super().__init__()
        self.register_buffer(
            "k_cache", torch.zeros(1, n_kv_heads, max_seq_len, head_dim)
        )
        self.register_buffer(
            "v_cache", torch.zeros(1, n_kv_heads, max_seq_len, head_dim)
        )

    def update(self, input_pos, k_val, v_val):
        self.k_cache[:, :, input_pos] = k_val
        self.v_cache[:, :, input_pos] = v_val
        return self.k_cache, self.v_cache


# ---------------------------------------------------------------------------
# Full Attention with output gate, QK-norm, partial RoPE


class FullAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.n_kv_groups = self.n_heads // self.n_kv_heads

        # q_proj includes output gate: 2x heads
        self.q_proj = nn.Linear(
            config.hidden_size, self.n_heads * self.head_dim * 2, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, config.hidden_size, bias=False
        )

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            self.head_dim, config.partial_rotary_factor, config.rope_theta
        )

        self.kv_cache = KVCache(self.n_kv_heads, self.head_dim, config.max_seq_len)

        mask = torch.tril(
            torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool)
        )
        self.register_buffer("mask", mask)

    def forward(self, x, input_pos):
        B, T, _ = x.size()
        dtype = x.dtype

        # Q includes gate
        q_and_gate = self.q_proj(x).view(B, T, self.n_heads, self.head_dim * 2)
        q = q_and_gate[..., : self.head_dim]
        gate = q_and_gate[..., self.head_dim :]

        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        # QK-norm before RoPE
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Partial RoPE
        q, k = self.rotary_emb(input_pos, q, k)

        # Cast back after norm/rope may upcast
        q = q.to(dtype).transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = k.to(dtype).transpose(1, 2)
        v = v.transpose(1, 2)

        # KV cache
        k, v = self.kv_cache.update(input_pos, k, v)

        # GQA expansion
        if self.n_kv_groups > 1:
            k = k.repeat_interleave(self.n_kv_groups, dim=1)
            v = v.repeat_interleave(self.n_kv_groups, dim=1)

        # SDPA with bool mask
        attn_mask = self.mask[input_pos].unsqueeze(0).unsqueeze(0)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        y = y.transpose(1, 2).contiguous().view(B, T, -1)

        # Output gate
        gate = gate.reshape(B, T, -1)
        y = y * torch.sigmoid(gate)

        return self.o_proj(y)


# ---------------------------------------------------------------------------
# GatedDeltaNet — linear attention with delta rule recurrence


class GatedDeltaNet(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim

        # head_repeat for k_heads != v_heads
        assert self.num_v_heads % self.num_k_heads == 0
        self.head_repeat = self.num_v_heads // self.num_k_heads

        self.conv_dim = self.key_dim * 2 + self.value_dim

        # Separate projections (matching HF checkpoint structure)
        self.in_proj_qkv = nn.Linear(config.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(config.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)

        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            config.linear_conv_kernel_dim,
            groups=self.conv_dim,
            padding=0,
            bias=False,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, config.hidden_size, bias=False)

        # State buffers
        self.register_buffer(
            "conv_state", torch.zeros(1, self.conv_dim, config.linear_conv_kernel_dim)
        )
        self.register_buffer(
            "recurrent_state",
            torch.zeros(1, self.num_v_heads, self.head_k_dim, self.head_v_dim),
        )

    def forward(self, x, input_pos):
        B, T, _ = x.size()

        # Reset state at position 0
        reset = (input_pos[0] == 0).to(self.conv_state.dtype)
        keep = 1.0 - reset
        self.conv_state[:B].mul_(keep)
        self.recurrent_state[:B].mul_(keep)

        # Projections
        mixed_qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x).reshape(B, T, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)

        # Causal conv1d with state
        qkv_t = mixed_qkv.transpose(1, 2)
        conv_input = torch.cat([self.conv_state[:B], qkv_t], dim=-1)
        with torch.no_grad():
            self.conv_state[:B].copy_(conv_input[:, :, -self.conv_kernel_size :])
        qkv_conv = F.conv1d(
            conv_input, self.conv1d.weight, bias=None, padding=0, groups=self.conv_dim
        )
        qkv_conv = F.silu(qkv_conv[:, :, -T:]).transpose(1, 2)

        # Split via slicing (torch.split produces split_copy which lacks AOTI fallback)
        kd = self.key_dim
        q = qkv_conv[..., :kd].reshape(B, T, self.num_k_heads, self.head_k_dim)
        k = qkv_conv[..., kd : 2 * kd].reshape(B, T, self.num_k_heads, self.head_k_dim)
        v = qkv_conv[..., 2 * kd :].reshape(B, T, self.num_v_heads, self.head_v_dim)

        # L2-normalize Q and K (the FLA kernel expects pre-normalized inputs;
        # HF reference uses use_qk_l2norm_in_kernel=True which does this inside)
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # head_repeat for k_heads != v_heads
        if self.head_repeat > 1:
            q = q.repeat_interleave(self.head_repeat, dim=2)
            k = k.repeat_interleave(self.head_repeat, dim=2)

        # Mamba-style gating
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        # FLA Triton kernel
        state = self.recurrent_state[:B].clone()
        output, state = torch.ops.triton.chunk_gated_delta_rule(q, k, v, g, beta, state)

        with torch.no_grad():
            self.recurrent_state[:B].copy_(state)

        # Output: RMSNorm(output) * silu(z)
        output = output.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        output = self.norm(output, z)
        output = output.reshape(B, T, -1)

        return self.out_proj(output)


# ---------------------------------------------------------------------------
# MoE: stacked expert weights + index by top-k

# 16 experts per group keeps each nn.Linear under ~32K output features,
# within tinygemm int4 packing limits while keeping the graph small
# (32 matmul nodes per layer instead of 768 with per-expert linears).
_EXPERTS_PER_GROUP = 16


class ConditionalFeedForward(nn.Module):
    """Grouped expert weights as nn.Linear for quantization compatibility.

    Experts are split into groups of _EXPERTS_PER_GROUP. Each group has:
      gate_up_projs[g]: nn.Linear(hidden_size, G * intermediate_size * 2)
      down_projs[g]: nn.Linear(intermediate_size, G * hidden_size)
    This keeps each nn.Linear small enough for tinygemm int4 packing while
    allowing quantize_model_() to handle them automatically.
    """

    def __init__(self, hidden_size, intermediate_size, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        G = _EXPERTS_PER_GROUP
        assert num_experts % G == 0
        num_groups = num_experts // G

        self.gate_up_projs = nn.ModuleList(
            [
                nn.Linear(hidden_size, G * intermediate_size * 2, bias=False)
                for _ in range(num_groups)
            ]
        )
        self.down_projs = nn.ModuleList(
            [
                nn.Linear(intermediate_size, G * hidden_size, bias=False)
                for _ in range(num_groups)
            ]
        )

    def forward(self, x, expert_indices):
        # x: (T, D), expert_indices: (T, top_k)
        T = x.size(0)
        top_k = expert_indices.size(1)
        G = _EXPERTS_PER_GROUP
        H = self.intermediate_size
        D = self.hidden_size

        # Gate + Up: compute per-group, cat, gather top-k
        gate_up_parts = [proj(x).view(T, G, 2, H) for proj in self.gate_up_projs]
        gate_up = torch.cat(gate_up_parts, dim=1)  # (T, E, 2, H)

        idx = expert_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, H)
        gate_up_sel = gate_up.gather(1, idx)  # (T, top_k, 2, H)
        intermediate = F.silu(gate_up_sel[:, :, 0, :]) * gate_up_sel[:, :, 1, :]

        # Down: compute per-group, cat, gather correct expert per slot
        intermediate_flat = intermediate.reshape(T * top_k, H)
        down_parts = [
            proj(intermediate_flat).view(T, top_k, G, D) for proj in self.down_projs
        ]
        all_down = torch.cat(down_parts, dim=2)  # (T, top_k, E, D)

        eidx = expert_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, D)
        return all_down.gather(2, eidx).squeeze(2)  # (T, top_k, D)


class SwiGLU(nn.Module):
    """SwiGLU MLP for shared expert."""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoE(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.cond_ffn = ConditionalFeedForward(
            config.hidden_size,
            config.moe_intermediate_size,
            config.num_experts,
        )
        self.shared_expert = SwiGLU(
            config.hidden_size, config.shared_expert_intermediate_size
        )
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        x_flat = x.view(-1, C)

        scores = self.gate(x_flat)
        expert_weights, expert_indices = torch.topk(scores, self.top_k, dim=-1)
        expert_weights = expert_weights.softmax(dim=-1)

        expert_outs = self.cond_ffn(x_flat, expert_indices)
        routed_out = torch.einsum("tai,ta->ti", expert_outs, expert_weights)

        shared_out = self.shared_expert(x_flat)
        shared_gate = torch.sigmoid(self.shared_expert_gate(x_flat))
        return (routed_out + shared_gate * shared_out).view(B, T, C)


# ---------------------------------------------------------------------------
# Decoder block and full model


class Block(nn.Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.ln_1 = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ln_2 = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if self.layer_type == "full_attention":
            self.attn = FullAttention(config)
        else:
            self.attn = GatedDeltaNet(config)

        self.mlp = SparseMoE(config)

    def forward(self, x, input_pos):
        x = x + self.attn(self.ln_1(x), input_pos)
        x = x + self.mlp(self.ln_2(x))
        return x


class Qwen35MoE(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Block(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self, tokens: torch.LongTensor, input_pos: torch.LongTensor
    ) -> torch.Tensor:
        x = self.embed_tokens(tokens)
        for layer in self.layers:
            x = layer(x, input_pos)
        x = self.norm(x)
        return self.lm_head(x)

    @staticmethod
    def from_hf_checkpoint(model_dir, max_seq_len=4096):
        config_path = os.path.join(model_dir, "config.json")
        config = Qwen35MoEConfig.from_hf_config(config_path)
        config.max_seq_len = max_seq_len

        # Meta-device construction: no memory allocated for weights
        print("Building model on meta device...")
        with torch.device("meta"):
            model = Qwen35MoE(config)

        # Load and remap weights with lazy shard-by-shard loading
        print(f"Loading weights from {model_dir}...")
        state_dict = _load_and_remap_checkpoint(model_dir, config)

        # assign=True replaces meta tensors by reference (no copy)
        num_loaded = len(state_dict)
        missing, unexpected = model.load_state_dict(
            state_dict, strict=False, assign=True
        )
        del state_dict

        # Materialize remaining meta-device buffers (KV caches, conv/recurrent
        # state, causal masks, RoPE inv_freq) as zeros on CPU
        for fqn, buf in list(model.named_buffers()):
            if buf.device.type == "meta":
                parts = fqn.rsplit(".", 1)
                parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
                parent.register_buffer(
                    parts[-1],
                    torch.zeros(buf.shape, dtype=buf.dtype, device="cpu"),
                )

        # Recompute RoPE inv_freq (zero-fill above is wrong for these)
        for layer in model.layers:
            if hasattr(layer.attn, "rotary_emb"):
                rope = layer.attn.rotary_emb
                inv_freq = 1.0 / (
                    config.rope_theta
                    ** (
                        torch.arange(0, rope.rotary_dim, 2, dtype=torch.float32)
                        / rope.rotary_dim
                    )
                )
                rope.inv_freq = inv_freq

        # Recompute causal masks for full attention layers
        for layer in model.layers:
            if isinstance(layer.attn, FullAttention):
                mask = torch.tril(
                    torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool)
                )
                layer.attn.register_buffer("mask", mask)

        # Validate
        runtime_prefixes = (
            "lm_head.weight",
            ".kv_cache.",
            ".conv_state",
            ".recurrent_state",
            ".mask",
            ".inv_freq",
        )
        actual_missing = set(missing)
        expected_missing = {
            k for k in actual_missing if any(p in k for p in runtime_prefixes)
        }
        extra_missing = actual_missing - expected_missing
        if extra_missing:
            print(f"  WARNING: unexpected missing keys: {sorted(extra_missing)}")
        if unexpected:
            print(f"  WARNING: unexpected keys: {sorted(unexpected)}")
        loaded = num_loaded - len(unexpected)
        print(f"  Loaded {loaded} tensors")

        return model, config


# ---------------------------------------------------------------------------
# Weight loading utilities


def _load_and_remap_checkpoint(model_dir, config):
    """Load safetensors lazily shard-by-shard and remap keys.

    Uses safe_open for lazy tensor access — only one shard's worth of
    raw tensors is in memory at a time. Expert weights are accumulated
    and stacked at the end.
    """
    from safetensors import safe_open

    # Find safetensors files
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
    elif os.path.exists(os.path.join(model_dir, "model.safetensors")):
        shard_files = ["model.safetensors"]
    else:
        raise FileNotFoundError(f"No safetensors checkpoint in {model_dir}")

    state_dict = {}
    expert_weights = {}  # (layer_idx, proj, expert_idx) → tensor

    for shard_file in shard_files:
        shard_path = os.path.join(model_dir, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for ckpt_key in f.keys():
                _process_checkpoint_key(
                    ckpt_key,
                    f.get_tensor(ckpt_key),
                    state_dict,
                    expert_weights,
                )

    # Stack per-expert weights, split into groups, reshape for nn.Linear
    if expert_weights:
        G = _EXPERTS_PER_GROUP
        for layer_idx in range(config.num_hidden_layers):
            gate_list = [
                expert_weights.get((layer_idx, "gate", e))
                for e in range(config.num_experts)
            ]
            up_list = [
                expert_weights.get((layer_idx, "up", e))
                for e in range(config.num_experts)
            ]
            down_list = [
                expert_weights.get((layer_idx, "down", e))
                for e in range(config.num_experts)
            ]

            if gate_list[0] is not None:
                w_gate = torch.stack(gate_list, dim=0)  # (E, H, D)
                w_up = torch.stack(up_list, dim=0)
                fused = torch.cat([w_gate, w_up], dim=1)  # (E, 2*H, D)
                num_groups = config.num_experts // G
                for g in range(num_groups):
                    chunk = fused[g * G : (g + 1) * G]
                    state_dict[
                        f"layers.{layer_idx}.mlp.cond_ffn.gate_up_projs.{g}.weight"
                    ] = chunk.reshape(-1, chunk.size(-1))
            if down_list[0] is not None:
                w_down = torch.stack(down_list, dim=0)  # (E, D, H)
                num_groups = config.num_experts // G
                for g in range(num_groups):
                    chunk = w_down[g * G : (g + 1) * G]
                    state_dict[
                        f"layers.{layer_idx}.mlp.cond_ffn.down_projs.{g}.weight"
                    ] = chunk.reshape(-1, chunk.size(-1))
        del expert_weights

    # Handle tied embeddings
    if "lm_head.weight" not in state_dict and "embed_tokens.weight" in state_dict:
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    return state_dict


def _process_checkpoint_key(ckpt_key, tensor, state_dict, expert_weights):
    """Remap a single checkpoint key and store into state_dict or expert_weights."""
    norm_key = ckpt_key
    if norm_key.startswith("model.language_model."):
        norm_key = norm_key.replace("model.language_model.", "model.", 1)

    if not norm_key.startswith(("model.", "lm_head.")):
        return
    # Skip non-text keys (visual encoder, mtp)
    if norm_key.startswith(("model.visual.", "model.mtp_")):
        return

    # Fused expert weights: split into groups of _EXPERTS_PER_GROUP
    m = _FUSED_EXPERT_RE.match(norm_key)
    if m:
        layer_idx = int(m.group(1))
        proj_name = m.group(2)
        G = _EXPERTS_PER_GROUP
        num_groups = tensor.size(0) // G
        if proj_name == "gate_up_proj":
            # (E, 2*H, D) → groups of (G, 2*H, D) → each (G*2*H, D)
            for g in range(num_groups):
                chunk = tensor[g * G : (g + 1) * G]
                state_dict[
                    f"layers.{layer_idx}.mlp.cond_ffn.gate_up_projs.{g}.weight"
                ] = chunk.reshape(-1, chunk.size(-1)).contiguous()
        else:
            # down_proj: (E, D, H) → groups of (G, D, H) → each (G*D, H)
            for g in range(num_groups):
                chunk = tensor[g * G : (g + 1) * G]
                state_dict[f"layers.{layer_idx}.mlp.cond_ffn.down_projs.{g}.weight"] = (
                    chunk.reshape(-1, chunk.size(-1)).contiguous()
                )
        return

    # Per-expert weights
    m = _EXPERT_RE.match(norm_key)
    if m:
        layer_idx, expert_idx, proj = int(m.group(1)), int(m.group(2)), m.group(3)
        expert_weights[(layer_idx, proj, expert_idx)] = tensor
        return

    # Standard mapping
    mapped = _get_mapped_key(norm_key)
    if mapped is not None:
        state_dict[mapped] = tensor
    elif not norm_key.endswith(_IGNORED_SUFFIXES):
        print(f"Warning: unmapped key {ckpt_key}")


# HuggingFace key → export model key mapping
_HF_KEY_MAP = {
    # Embeddings
    "model.embed_tokens.weight": "embed_tokens.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "lm_head.weight",
    # Layer norms
    "model.layers.{}.input_layernorm.weight": "layers.{}.ln_1.weight",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ln_2.weight",
    # Full attention
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attn.q_proj.weight",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attn.k_proj.weight",
    "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attn.v_proj.weight",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attn.o_proj.weight",
    "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attn.q_norm.weight",
    "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attn.k_norm.weight",
    # GatedDeltaNet
    "model.layers.{}.linear_attn.in_proj_qkv.weight": "layers.{}.attn.in_proj_qkv.weight",
    "model.layers.{}.linear_attn.in_proj_z.weight": "layers.{}.attn.in_proj_z.weight",
    "model.layers.{}.linear_attn.in_proj_b.weight": "layers.{}.attn.in_proj_b.weight",
    "model.layers.{}.linear_attn.in_proj_a.weight": "layers.{}.attn.in_proj_a.weight",
    "model.layers.{}.linear_attn.conv1d.weight": "layers.{}.attn.conv1d.weight",
    "model.layers.{}.linear_attn.dt_bias": "layers.{}.attn.dt_bias",
    "model.layers.{}.linear_attn.A_log": "layers.{}.attn.A_log",
    "model.layers.{}.linear_attn.norm.weight": "layers.{}.attn.norm.weight",
    "model.layers.{}.linear_attn.out_proj.weight": "layers.{}.attn.out_proj.weight",
    # MoE (non-expert)
    "model.layers.{}.mlp.gate.weight": "layers.{}.mlp.gate.weight",
    "model.layers.{}.mlp.shared_expert_gate.weight": "layers.{}.mlp.shared_expert_gate.weight",
    "model.layers.{}.mlp.shared_expert.gate_proj.weight": "layers.{}.mlp.shared_expert.gate_proj.weight",
    "model.layers.{}.mlp.shared_expert.up_proj.weight": "layers.{}.mlp.shared_expert.up_proj.weight",
    "model.layers.{}.mlp.shared_expert.down_proj.weight": "layers.{}.mlp.shared_expert.down_proj.weight",
}

_IGNORED_SUFFIXES = (
    "rotary_emb.inv_freq",
    "linear_attn.conv1d.bias",
)

# Fused expert keys: experts.gate_up_proj [E, 2*H, D] and experts.down_proj [E, D, H]
_FUSED_EXPERT_RE = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)"
)

# Per-expert keys (alternative checkpoint format)
_EXPERT_RE = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight"
)


def _get_mapped_key(hf_key):
    """Map a HuggingFace key to export model key using _HF_KEY_MAP."""
    for hf_pattern, model_pattern in _HF_KEY_MAP.items():
        if "{}" not in hf_pattern:
            if hf_key == hf_pattern:
                return model_pattern
        else:
            # Build regex from pattern
            escaped = re.escape(hf_pattern).replace(r"\{\}", r"(\d+)")
            m = re.fullmatch(escaped, hf_key)
            if m:
                result = model_pattern
                for group in m.groups():
                    result = result.replace("{}", group, 1)
                return result

    return None

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""EAGLE-3 draft head for vLLM speculator checkpoints.

The draft model fuses target auxiliary hidden states with ``fc``, runs one
Llama-style decoder layer over token embeddings plus the fused feature, and
projects the midlayer output to reduced-vocabulary draft logits. The midlayer
output ``g`` is reused as the recurrent feature for drafting; ``fc`` is used
only for target auxiliary hidden states.

Draft ids map back to target ids with ``target_id = draft_id + d2t[draft_id]``.
Speculator checkpoints store the decoder layer under ``layers.0.*`` and may
include ``embed_tokens``, ``d2t``, and ``t2d``.
"""

import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class Eagle3Config:
    hidden_size: int = 5376
    target_hidden_size: int = 5376
    intermediate_size: int = 21504
    num_attention_heads: int = 32
    num_key_value_heads: int = 16
    head_dim: int = 256
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6
    draft_vocab_size: int = 32000
    target_vocab_size: int = 262144
    aux_hidden_state_layers: list = field(default_factory=lambda: [2, 30, 57])
    # norm_before_residual: store the attention residual after hidden_norm.
    # norm_before_fc: apply an RMSNorm over the concatenated aux features before
    #   fc (gpt-oss-style speculators checkpoints); not supported here.
    # has_own_embed: the head ships its own embed_tokens (set during load).
    norm_before_residual: bool = True
    norm_before_fc: bool = False
    has_own_embed: bool = False


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Eagle3Attention(nn.Module):
    """Llama GQA attention; q/k/v project from the doubled-width (2*hidden) input."""

    def __init__(self, config: Eagle3Config):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        in_dim = 2 * config.hidden_size

        self.q_proj = nn.Linear(in_dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(in_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(in_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, config.hidden_size, bias=False
        )

        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        freqs = torch.outer(positions.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(q.dtype)
        sin = emb.sin().to(q.dtype)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(y)


class Eagle3MLP(nn.Module):
    def __init__(self, config: Eagle3Config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Eagle3Midlayer(nn.Module):
    """Single EAGLE-3 decoder layer with dual input norms over two streams."""

    def __init__(self, config: Eagle3Config):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Eagle3Attention(config)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = Eagle3MLP(config)
        self.norm_before_residual = config.norm_before_residual

    def forward(
        self,
        input_embeds: torch.Tensor,
        feature: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        normed_embeds = self.input_layernorm(input_embeds)
        normed_feature = self.hidden_norm(feature)
        residual = normed_feature if self.norm_before_residual else feature
        x = torch.cat((normed_embeds, normed_feature), dim=-1)
        x = self.self_attn(x, positions)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return residual + x


class Eagle3Draft(nn.Module):
    def __init__(self, config: Eagle3Config):
        super().__init__()
        self.config = config
        self.fc = nn.Linear(
            len(config.aux_hidden_state_layers) * config.target_hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.midlayer = Eagle3Midlayer(config)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(
            config.hidden_size, config.draft_vocab_size, bias=False
        )
        if config.has_own_embed:
            self.embed_tokens = nn.Embedding(
                config.target_vocab_size, config.hidden_size
            )
        # d2t/t2d are loaded from the checkpoint (assign=True adopts their
        # shapes/dtypes): d2t[draft_id] is the offset to the target vocab id;
        # t2d masks which target ids are in the draft vocab.
        self.register_buffer(
            "d2t",
            torch.zeros(config.draft_vocab_size, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer("t2d", torch.zeros(1, dtype=torch.bool), persistent=False)

    def fuse(self, aux: torch.Tensor) -> torch.Tensor:
        """Fuse concatenated target aux hidden states (B,T,3*D) -> feature (B,T,D)."""
        return self.fc(aux)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """Embed token ids with the head's own table.

        Only valid when the checkpoint shipped its own ``embed_tokens``; heads
        that reuse the target embedding must source embeddings from the target.
        """
        if not self.config.has_own_embed:
            raise RuntimeError(
                "this draft head has no own embed_tokens (has_own_embed=False); "
                "provide token embeddings from the target model instead"
            )
        return self.embed_tokens(ids)

    def forward(
        self,
        input_embeds: torch.Tensor,
        feature: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the midlayer over a sequence.

        Returns (draft_logits, g):
          draft_logits: (B, T, draft_vocab_size) over the reduced vocab.
          g:            (B, T, hidden) midlayer output — the recurrent feature.
        """
        g = self.midlayer(input_embeds, feature, positions)
        draft_logits = self.lm_head(self.norm(g))
        return draft_logits, g

    def draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids + self.d2t[draft_ids]

    @staticmethod
    def from_checkpoint(
        model_dir: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16
    ) -> tuple["Eagle3Draft", Eagle3Config]:
        import json

        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)

        tlc = cfg["transformer_layer_config"]
        config = Eagle3Config(
            hidden_size=tlc["hidden_size"],
            target_hidden_size=cfg.get("target_hidden_size") or tlc["hidden_size"],
            intermediate_size=tlc["intermediate_size"],
            num_attention_heads=tlc["num_attention_heads"],
            num_key_value_heads=tlc["num_key_value_heads"],
            head_dim=tlc["head_dim"],
            rope_theta=tlc["rope_parameters"]["rope_theta"],
            rms_norm_eps=tlc["rms_norm_eps"],
            draft_vocab_size=cfg["draft_vocab_size"],
            target_vocab_size=tlc.get("vocab_size", 262144),
            aux_hidden_state_layers=cfg["eagle_aux_hidden_state_layer_ids"],
            norm_before_residual=cfg.get("norm_before_residual", False),
            norm_before_fc=cfg.get("norm_before_fc", False),
        )
        if config.norm_before_fc:
            # This checkpoint variant requires an input RMSNorm before fc.
            raise ValueError(
                "norm_before_fc=True checkpoints are not supported "
                "(would need an input RMSNorm before fc)"
            )

        raw = _load_safetensors(model_dir)
        config.has_own_embed = "embed_tokens.weight" in raw

        # Cast checkpoint weights after module construction so inv_freq stays fp32.
        model = Eagle3Draft(config)
        # The single decoder layer is stored as layers.0.* on disk.
        state_dict = {
            (k.replace("layers.0.", "midlayer.") if k.startswith("layers.0.") else k): (
                v.to(dtype) if v.is_floating_point() else v
            )
            for k, v in raw.items()
        }
        # d2t/t2d are index/mask tensors (their checkpoint shape differs from the
        # placeholder buffers); register them directly, load the rest strict.
        model.register_buffer("d2t", state_dict.pop("d2t"), persistent=False)
        model.register_buffer("t2d", state_dict.pop("t2d"), persistent=False)
        model.load_state_dict(state_dict, strict=True, assign=True)
        model = model.to(device)
        assert (
            model.midlayer.self_attn.inv_freq.dtype == torch.float32
        ), "RoPE inv_freq must remain float32"
        return model.eval(), config


def _load_safetensors(model_dir: str) -> dict:
    """Load a monolithic or sharded safetensors checkpoint to CPU tensors."""
    import json

    from safetensors import safe_open

    index = os.path.join(model_dir, "model.safetensors.index.json")
    mono = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(mono):
        shards = ["model.safetensors"]
    elif os.path.exists(index):
        with open(index) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
    else:
        raise FileNotFoundError(
            f"no model.safetensors or model.safetensors.index.json in {model_dir}"
        )
    raw = {}
    for shard in shards:
        with safe_open(
            os.path.join(model_dir, shard), framework="pt", device="cpu"
        ) as f:
            for k in f.keys():
                if k in raw:
                    raise ValueError(f"duplicate tensor {k!r} across shards ({shard})")
                raw[k] = f.get_tensor(k)
    return raw

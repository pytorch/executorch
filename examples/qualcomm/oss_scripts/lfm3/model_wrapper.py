# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Tuple

import torch
from executorch.examples.qualcomm.oss_scripts.llama.masking_utils import (
    AttentionMask,
    CausalAttentionMask,
    SlidingWindowAttentionMask,
)
from executorch.examples.qualcomm.oss_scripts.llama.model.apply_rope import (
    ROTARY_EMB_REGISTRY,
)
from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import repeat_kv
from torch import nn


@dataclass
class Lfm3Config:
    hidden_size: int = 2048
    intermediate_size: int = 7680
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 128
    vocab_size: int = 128000
    norm_eps: float = 1e-5
    conv_L_cache: int = 7
    sliding_window: int = 256
    layer_types: Optional[List[str]] = None
    rope_parameters: Optional[Dict[str, Dict]] = None
    bos_token_id: int = 124894
    eos_token_id: int = 124900
    # Runtime settings, filled in by the export flow.
    max_batch_size: int = 1
    max_seq_len: int = 1024
    max_context_len: int = 1024
    use_kv_cache: bool = True

    @classmethod
    def from_json(cls, path: str) -> "Lfm3Config":
        with open(path) as f:
            raw = json.load(f)
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in names})

    @property
    def attention_layers(self) -> List[int]:
        return [i for i, t in enumerate(self.layer_types) if t != "conv"]

    @property
    def conv_layers(self) -> List[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "conv"]


def _precompute_freqs(
    head_dim: int, max_context_len: int, rope_theta: float, partial_rotary_factor: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """cos/sin of shape [max_context_len, head_dim // 2] in LFM3's zero-tail layout.

    LFM3's partial RoPE keeps the full-head-dim frequency progression and zeroes the
    tail frequencies, so the tail is an identity rotation. That lets the regular
    rotate-half RoPE apply unchanged.
    """
    rope_angles = int(partial_rotary_factor * head_dim) // 2
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32) / head_dim)
    )
    inv_freq = torch.cat([inv_freq, torch.zeros(head_dim // 2 - rope_angles)])
    freqs = torch.outer(torch.arange(max_context_len, dtype=torch.float32), inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


class Lfm3Attention(nn.Module):
    """GQA with QK RMSNorm and a per-head sigmoid output gate."""

    def __init__(self, config: Lfm3Config):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads
        dim = config.hidden_size

        self.wq = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.gate = nn.Linear(dim, self.n_heads, bias=False)
        self.q_norm_fn = nn.RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_norm_fn = nn.RMSNorm(self.head_dim, eps=config.norm_eps)
        self.attn_softmax = nn.Softmax(dim=-1)
        self.scale = float(self.head_dim) ** 0.5
        self.apply_rope_emb = ROTARY_EMB_REGISTRY["default"]

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        atten_mask: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = x.shape

        q = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        q = self.q_norm_fn(q).transpose(1, 2)
        k = self.k_norm_fn(k).transpose(1, 2)
        v = v.transpose(1, 2)

        q = self.apply_rope_emb(q, freqs_cos, freqs_sin)
        k = self.apply_rope_emb(k, freqs_cos, freqs_sin)
        # Keys are cached transposed, [bsz, n_kv_heads, head_dim, seq].
        k = k.transpose(2, 3)

        kh = torch.cat([k_cache, k], dim=-1)
        vh = torch.cat([v_cache, v], dim=2)
        kh = repeat_kv(kh, self.n_rep)
        vh = repeat_kv(vh, self.n_rep)

        attn = q @ kh
        attn = attn / self.scale
        attn = attn + atten_mask
        attn = self.attn_softmax(attn)
        y = attn @ vh

        gate = torch.sigmoid(self.gate(x)).unsqueeze(-1)
        y = y.transpose(1, 2) * gate
        y = y.reshape(bsz, seq_len, -1)
        y = self.wo(y)
        return y, k, v


class Lfm3ShortConv(nn.Module):
    """Gated depthwise causal conv. The last `kernel - 1` inputs are carried as state."""

    def __init__(self, config: Lfm3Config):
        super().__init__()
        dim = config.hidden_size
        self.kernel_size = config.conv_L_cache
        self.in_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        # HF stores this as Conv1d weight [dim, 1, kernel]; kept as [kernel, dim]
        # so each tap broadcasts over [bsz, seq, dim].
        self.conv_weight = nn.Parameter(torch.zeros(self.kernel_size, dim))

    def forward(
        self, x: torch.Tensor, conv_state: torch.Tensor, window_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[1]
        b_gate, c_gate, xs = self.in_proj(x).chunk(3, dim=-1)
        conv_in = b_gate * xs

        # window_mask zeroes state rows that precede position 0, so a fresh
        # sequence needs no knowledge of the state's quantized zero point.
        window = torch.cat([conv_state, conv_in], dim=1) * window_mask
        y = window[:, 0:seq_len, :] * self.conv_weight[0]
        for k in range(1, self.kernel_size):
            y = y + window[:, k : k + seq_len, :] * self.conv_weight[k]

        y = self.out_proj(c_gate * y)
        return y, conv_in


class Lfm3FeedForward(nn.Module):
    def __init__(self, config: Lfm3Config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class Lfm3DecoderLayer(nn.Module):
    def __init__(self, config: Lfm3Config, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        # `attention` holds the token mixer of either kind, so code that walks
        # decoder.layers[i].attention keeps working.
        self.attention = (
            Lfm3ShortConv(config) if layer_type == "conv" else Lfm3Attention(config)
        )
        self.feed_forward = Lfm3FeedForward(config)
        self.attention_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.post_attention_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.post_ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(self, x, *mixer_args):
        out = self.attention(self.attention_norm(x), *mixer_args)
        h = x + self.post_attention_norm(out[0])
        h = h + self.post_ffn_norm(self.feed_forward(self.ffn_norm(h)))
        return (h, *out[1:])


class Lfm3Model(nn.Module):
    """QNN-export-ready LFM3 text decoder (static_llama.py I/O convention).

    forward(tokens, atten_mask, window_atten_mask, input_pos,
            *k_caches[n_attn], *v_caches[n_attn], *conv_states[n_conv])
        -> (logits, new_k[n_attn], new_v[n_attn] + new_conv_inputs[n_conv])

    Conv layers return the conv inputs of the current tokens; the runner keeps the
    last `kernel - 1` valid rows as the next conv state. They ride along with the
    V caches so the generic decoder plumbing (calibration, export flattening) sees
    the usual (tokens, masks, pos, k_caches, v_caches) signature.
    """

    def __init__(
        self,
        config: Lfm3Config,
        ar_len: int = 1,
        output_new_cache_only: bool = True,
        output_cache: bool = True,
        use_i64_token: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert output_new_cache_only, "LFM3 only supports outputting new cache entries"
        assert config.use_kv_cache, "LFM3 is only exported with KV cache I/O"
        self.config = config
        self.ar_len = ar_len
        self.output_cache = output_cache
        self.use_i64_token = use_i64_token
        self.use_kv_cache = True
        self.max_batch_size = config.max_batch_size
        self.vocab_size = config.vocab_size
        self.n_layers = config.num_hidden_layers
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.conv_state_len = config.conv_L_cache - 1
        self.kv_io_bit_width = kwargs.get("kv_io_bit_width") or 32
        self.attention_layers = config.attention_layers
        self.conv_layers = config.conv_layers

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Lfm3DecoderLayer(config, t) for t in config.layer_types]
        )
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # State row r holds position input_pos[0] - state_len + r. Kept independent
        # of ar_len: hybrid export copies buffers from the calibration graph into
        # the prefill/decode graphs.
        offsets = torch.arange(-self.conv_state_len, 0)
        positions = torch.arange(config.max_context_len)[:, None] + offsets[None, :]
        self.register_buffer(
            "conv_state_mask",
            (positions >= 0).float().unsqueeze(-1),
            persistent=False,
        )

        rope = config.rope_parameters
        for name, layer_type in (
            ("global", "full_attention"),
            ("local", "sliding_attention"),
        ):
            cos, sin = _precompute_freqs(
                config.head_dim,
                config.max_context_len,
                rope[layer_type]["rope_theta"],
                rope[layer_type].get("partial_rotary_factor", 1.0),
            )
            self.register_buffer(f"{name}_freqs_cos", cos, persistent=False)
            self.register_buffer(f"{name}_freqs_sin", sin, persistent=False)

    def forward(
        self,
        tokens: torch.Tensor,
        atten_mask: torch.Tensor,
        window_atten_mask: torch.Tensor,
        input_pos: torch.Tensor,
        *args,
    ):
        n_attn = len(self.attention_layers)
        k_caches = args[:n_attn]
        v_caches = args[n_attn : 2 * n_attn]
        conv_states = args[2 * n_attn :]

        global_cos = self.global_freqs_cos[input_pos][0]
        global_sin = self.global_freqs_sin[input_pos][0]
        local_cos = self.local_freqs_cos[input_pos][0]
        local_sin = self.local_freqs_sin[input_pos][0]

        hidden_states = self.tok_embeddings(tokens)

        # select, not slice: a [:, :1] slice is a no-op alias when ar_len == 1,
        # which breaks the graph matching used to share encodings across graphs.
        conv_window_mask = torch.cat(
            [
                self.conv_state_mask[input_pos[:, 0]],
                torch.ones_like(hidden_states[:, :, 0:1]),
            ],
            dim=1,
        )
        output_k_cache, output_v_cache, output_conv = [], [], []
        attn_idx, conv_idx = 0, 0
        for layer in self.layers:
            if layer.layer_type == "conv":
                hidden_states, conv_in = layer(
                    hidden_states, conv_states[conv_idx], conv_window_mask
                )
                output_conv.append(conv_in)
                conv_idx += 1
                continue
            if layer.layer_type == "sliding_attention":
                cos, sin, mask = local_cos, local_sin, window_atten_mask
            else:
                cos, sin, mask = global_cos, global_sin, atten_mask
            hidden_states, k, v = layer(
                hidden_states, cos, sin, mask, k_caches[attn_idx], v_caches[attn_idx]
            )
            output_k_cache.append(k)
            output_v_cache.append(v)
            attn_idx += 1

        hidden_states = self.norm(hidden_states)
        logits = self.output(hidden_states)
        if self.output_cache:
            return logits, output_k_cache, output_v_cache + output_conv
        return logits

    def get_example_inputs(self):
        cfg = self.config
        bsz = cfg.max_batch_size
        dtype = torch.int64 if self.use_i64_token else torch.int32
        tokens = torch.randint(cfg.vocab_size, (bsz, self.ar_len), dtype=dtype)
        attn_mask = AttentionMask(
            [
                CausalAttentionMask(bsz, self.ar_len, cfg.max_context_len),
                SlidingWindowAttentionMask(
                    bsz,
                    self.ar_len,
                    cfg.max_context_len,
                    sliding_window=cfg.sliding_window,
                ),
            ]
        )
        pos_ids = torch.zeros((bsz, self.ar_len), dtype=torch.int32)
        cache_len = cfg.max_context_len - self.ar_len
        k_caches = [
            torch.zeros(bsz, self.n_kv_heads, self.head_dim, cache_len)
            for _ in self.attention_layers
        ]
        v_caches = [
            torch.zeros(bsz, self.n_kv_heads, cache_len, self.head_dim)
            for _ in self.attention_layers
        ]
        conv_states = [
            torch.zeros(bsz, self.conv_state_len, cfg.hidden_size)
            for _ in self.conv_layers
        ]
        return (tokens, attn_mask, pos_ids, k_caches, v_caches + conv_states)

    def get_metadata(self):
        cfg = self.config
        return {
            "get_ar_len": self.ar_len,
            "get_bos_id": cfg.bos_token_id,
            "get_eos_id": cfg.eos_token_id,
            "get_dim": cfg.hidden_size,
            "get_head_dim": cfg.head_dim,
            "get_max_batch_size": cfg.max_batch_size,
            "get_max_seq_len": cfg.max_seq_len,
            "get_max_context_len": cfg.max_context_len,
            "get_n_bos": 1,
            "get_n_eos": 1,
            "get_n_kv_heads": cfg.num_key_value_heads,
            "get_n_layers": cfg.num_hidden_layers,
            "get_n_self_layers": len(self.attention_layers),
            "get_n_conv_layers": len(self.conv_layers),
            "get_conv_state_len": self.conv_state_len,
            "get_vocab_size": cfg.vocab_size,
            "get_use_kv_cache": self.use_kv_cache,
            "get_sliding_window": cfg.sliding_window,
            "get_kv_io_bit_width": self.kv_io_bit_width,
        }

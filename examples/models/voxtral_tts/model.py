# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Voxtral TTS eager model for ExecuTorch.

Two-stage TTS pipeline:
  Stage 1 (model.pte): LLM + acoustic transformer → audio codes
  Stage 2 (codec.pte): audio tokenizer decoder → waveform

Architecture follows voxtral_realtime patterns for Metal/XNNPACK compatibility.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.extension.llm.custom_ops import custom_ops as _custom_ops  # noqa: F401


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_AUDIO_SPECIAL_TOKENS = 2  # [EMPTY_AUDIO]=0, [END_AUDIO]=1
_EMPTY_AUDIO_TOKEN_ID = 0
_END_AUDIO_TOKEN_ID = 1
_ACOUSTIC_DECODE_ITERS = 8
_CFG_ALPHA = 1.2
_NOISE_SCALE = 1.0


def _round_up(n: int, multiple: int) -> int:
    return multiple * ((n + multiple - 1) // multiple)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class VoxtralTTSConfig:
    # LM decoder
    dim: int = 3072
    n_layers: int = 26
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128
    hidden_dim: int = 9216
    vocab_size: int = 131072
    rope_theta: float = 1_000_000.0
    norm_eps: float = 1e-5
    # Acoustic transformer
    at_input_dim: int = 3072
    at_dim: int = 768
    at_n_layers: int = 3
    at_n_heads: int = 6
    at_n_kv_heads: int = 2
    at_head_dim: int = 128
    at_hidden_dim: int = 2048
    at_use_biases: bool = False
    at_norm_eps: float = 1e-5
    # Audio codebooks
    semantic_codebook_size: int = 8192
    acoustic_codebook_size: int = 21
    n_acoustic_codebook: int = 36
    # Runtime
    max_seq_len: int = 4096
    backend: str = "metal"

    @property
    def padded_semantic_vocab_size(self) -> int:
        return _round_up(self.semantic_codebook_size + _N_AUDIO_SPECIAL_TOKENS, 128)

    @staticmethod
    def from_params_json(path: str) -> "VoxtralTTSConfig":
        with open(path) as f:
            p = json.load(f)
        audio = p.get("multimodal", {}).get("audio_model_args", {})
        at = audio.get("acoustic_transformer_args", {})

        codebook_str = audio.get("codebook_sizes", "")
        if codebook_str:
            sizes = [int(c) for c in codebook_str.split(",")]
            sem_size, acou_size, n_acou = sizes[0], sizes[1], len(sizes) - 1
        else:
            sem_size = audio.get("semantic_codebook_size", 8192)
            acou_size = audio.get("acoustic_codebook_size", 21)
            n_acou = audio.get("n_acoustic_codebook", 36)

        return VoxtralTTSConfig(
            dim=p["dim"],
            n_layers=p["n_layers"],
            n_heads=p["n_heads"],
            n_kv_heads=p["n_kv_heads"],
            head_dim=p["head_dim"],
            hidden_dim=p["hidden_dim"],
            vocab_size=p["vocab_size"],
            rope_theta=p["rope_theta"],
            norm_eps=p["norm_eps"],
            at_input_dim=at.get("input_dim", p["dim"]),
            at_dim=at.get("dim", 768),
            at_n_layers=at.get("n_layers", 3),
            at_n_heads=at.get("n_heads", 6),
            at_n_kv_heads=at.get("n_kv_heads", 2),
            at_head_dim=at.get("head_dim", 128),
            at_hidden_dim=at.get("hidden_dim", 2048),
            at_use_biases=at.get("use_biases", False),
            at_norm_eps=at.get("norm_eps", 1e-5),
            semantic_codebook_size=sem_size,
            acoustic_codebook_size=acou_size,
            n_acoustic_codebook=n_acou,
        )


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


def precompute_freqs_cis(
    head_dim: int, max_len: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len, dtype=torch.float)
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split-half rotation: swap first and second halves, negate first."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings using split-half rotation.

    Uses the HuggingFace/Mistral convention (rotate_half) where the first and
    second halves of the head dimension are swapped, NOT the Llama interleaved
    convention (reshape+unbind pairs). Both produce valid rotary embeddings
    but are NOT interchangeable — the model must be trained with one convention.
    Mistral models use split-half.
    """
    # freqs: (T, head_dim/2) → repeat to (T, head_dim) for split-half
    fc = freqs_cos.unsqueeze(0).unsqueeze(2)  # (1, T, 1, D/2)
    fs = freqs_sin.unsqueeze(0).unsqueeze(2)
    # Repeat cos/sin to match full head_dim
    fc = fc.repeat(1, 1, 1, 2)  # (1, T, 1, D)
    fs = fs.repeat(1, 1, 1, 2)

    q_f = q.float()
    k_f = k.float()
    q_out = q_f * fc + _rotate_half(q_f) * fs
    k_out = k_f * fc + _rotate_half(k_f) * fs
    return q_out.type_as(q), k_out.type_as(k)


# ---------------------------------------------------------------------------
# KV caches
# ---------------------------------------------------------------------------


class KVCache(nn.Module):
    """[B, S, H, D] layout for torch.ops.llama.update_cache (XNNPACK/Portable)."""

    def __init__(self, max_seq_len: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        cache_shape = (1, max_seq_len, n_kv_heads, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape))
        self.register_buffer("v_cache", torch.zeros(cache_shape))

    def update(
        self, input_pos: torch.Tensor, k_val: torch.Tensor, v_val: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start_pos = input_pos[0].item()
        torch._check_is_size(start_pos)
        torch._check(start_pos < self.max_seq_len)
        torch.ops.llama.update_cache(k_val, self.k_cache, start_pos)
        torch.ops.llama.update_cache(v_val, self.v_cache, start_pos)
        return self.k_cache, self.v_cache


class StaticKVCache(nn.Module):
    """[B, H, S, D] layout with index_copy_ (Metal/CUDA/AOTI)."""

    def __init__(self, max_seq_len: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        cache_shape = (1, n_kv_heads, max_seq_len, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape))
        self.register_buffer("v_cache", torch.zeros(cache_shape))

    def update(
        self, input_pos: torch.Tensor, k_val: torch.Tensor, v_val: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k_val = k_val.transpose(1, 2)
        v_val = v_val.transpose(1, 2)
        self.k_cache.index_copy_(2, input_pos, k_val)
        self.v_cache.index_copy_(2, input_pos, v_val)
        return self.k_cache, self.v_cache


# ---------------------------------------------------------------------------
# Attention masks
# ---------------------------------------------------------------------------


def _build_attn_mask(
    input_pos: torch.Tensor,
    max_seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Additive float mask for Metal. dtype must match Q/K/V."""
    k_pos = torch.arange(max_seq_len, device=device)
    # General path: works for both prefill (seq_len>1) and decode (seq_len=1)
    # without a shape-dependent branch that creates unprovable guards.
    diff = input_pos.unsqueeze(1) - k_pos.unsqueeze(0) + 1
    valid = torch.clamp(diff, min=0, max=1)
    return (valid.to(dtype) - 1.0) * 1e9


# ---------------------------------------------------------------------------
# SDPA variants
# ---------------------------------------------------------------------------


class SDPA(nn.Module):
    """torch.ops.llama.custom_sdpa for XNNPACK/Portable."""

    def __init__(self, n_heads: int, head_dim: int):
        super().__init__()
        self.dim = n_heads * head_dim

    def forward(
        self,
        input_pos: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bsz: int,
        seqlen: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_dtype = q.dtype
        q, k, v = q.float(), k.float(), v.float()
        start_pos = input_pos[0].item()
        torch._check_is_size(start_pos)
        if mask is not None:
            y = torch.ops.llama.custom_sdpa(q, k, v, start_pos, mask.float(), 0, False)
        else:
            y = torch.ops.llama.custom_sdpa(q, k, v, start_pos, None, 0, True)
        return y.view(bsz, seqlen, self.dim).to(dtype=input_dtype)


class MetalSDPA(nn.Module):
    """Native MPS SDPA kernel. Handles GQA natively."""

    def __init__(
        self, n_heads: int, n_kv_heads: int, head_dim: int, transpose_kv: bool = False
    ):
        super().__init__()
        self.dim = n_heads * head_dim
        self.transpose_kv = transpose_kv

    def forward(
        self,
        input_pos: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bsz: int,
        seqlen: int,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = q.transpose(1, 2)
        if self.transpose_kv:
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
        if attn_mask is None:
            attn_mask = _build_attn_mask(input_pos, k.shape[2], q.device, q.dtype)
        y, _ = torch.ops.aten._scaled_dot_product_attention_math_for_mps(
            q, k, v, attn_mask, 0.0, False, None
        )
        y = y.transpose(1, 2).contiguous()
        return y.view(bsz, seqlen, self.dim)


# ---------------------------------------------------------------------------
# LM decoder
# ---------------------------------------------------------------------------


class LMAttention(nn.Module):
    """GQA with RoPE, KV cache, SDPA. No biases."""

    def __init__(self, config: VoxtralTTSConfig, max_seq_len: int):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.backend = config.backend

        self.wq = nn.Linear(config.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, config.dim, bias=False)

        if self.backend == "metal":
            self.kv_cache = StaticKVCache(max_seq_len, self.n_kv_heads, self.head_dim)
            self.sdpa = MetalSDPA(self.n_heads, self.n_kv_heads, self.head_dim)
        else:  # xnnpack
            self.kv_cache = KVCache(max_seq_len, self.n_kv_heads, self.head_dim)
            self.sdpa = SDPA(self.n_heads, self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        input_pos: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
        q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)
        k, v = self.kv_cache.update(input_pos, k, v)
        return self.wo(self.sdpa(input_pos, q, k, v, B, T, attn_mask))


class LMMLP(nn.Module):
    """SwiGLU FFN. No biases."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MistralDecoderLayer(nn.Module):
    def __init__(self, config: VoxtralTTSConfig, max_seq_len: int):
        super().__init__()
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention = LMAttention(config, max_seq_len)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.feed_forward = LMMLP(config.dim, config.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        input_pos: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attention(
            self.attention_norm(x), freqs_cos, freqs_sin, input_pos, attn_mask
        )
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class MistralDecoder(nn.Module):
    """Mistral LM decoder. forward() returns normed hidden states (not logits)."""

    def __init__(self, config: VoxtralTTSConfig, max_seq_len: int):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            [MistralDecoderLayer(config, max_seq_len) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        freqs_cos, freqs_sin = precompute_freqs_cis(
            config.head_dim, max_seq_len, config.rope_theta
        )
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def forward(
        self, input_embeds: torch.Tensor, input_pos: torch.Tensor
    ) -> torch.Tensor:
        freqs_cos = self.freqs_cos[input_pos]
        freqs_sin = self.freqs_sin[input_pos]

        attn_mask: torch.Tensor | None = None
        if self.config.backend == "metal":
            attn_mask = _build_attn_mask(
                input_pos,
                self.freqs_cos.shape[0],
                input_embeds.device,
                input_embeds.dtype,
            )

        x = input_embeds
        for layer in self.layers:
            x = layer(x, freqs_cos, freqs_sin, input_pos, attn_mask)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Acoustic transformer (flow matching)
# ---------------------------------------------------------------------------


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for flow matching step."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = torch.exp(
            -math.log(theta) * torch.arange(dim // 2).float() / (dim // 2)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B, 1)
        emb = t * self.inv_freq  # (B, dim//2) via broadcast
        return torch.cat((emb.cos(), emb.sin()), dim=-1)


class BidirectionalAttention(nn.Module):
    """Non-causal attention for the acoustic transformer. No RoPE, no KV cache."""

    def __init__(self, config: VoxtralTTSConfig):
        super().__init__()
        self.n_heads = config.at_n_heads
        self.n_kv_heads = config.at_n_kv_heads
        self.head_dim = config.at_head_dim
        self.n_rep = self.n_heads // self.n_kv_heads
        dim = config.at_dim

        self.wq = nn.Linear(
            dim, self.n_heads * self.head_dim, bias=config.at_use_biases
        )
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=config.at_use_biases
        )
        self.wo = nn.Linear(
            self.n_heads * self.head_dim, dim, bias=config.at_use_biases
        )
        self.scale = self.head_dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        # Transpose to [B, H, T, D] for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # GQA expansion
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Manual attention (seq_len=3, no SDPA kernel needed)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(-1)
        y = attn @ v

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(y)


class AcousticTransformerBlock(nn.Module):
    def __init__(self, config: VoxtralTTSConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.at_dim, config.at_norm_eps)
        self.attention = BidirectionalAttention(config)
        self.ffn_norm = RMSNorm(config.at_dim, config.at_norm_eps)
        self.feed_forward = LMMLP(config.at_dim, config.at_hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class FlowMatchingAudioTransformer(nn.Module):
    """Generates audio codes from LLM hidden states via flow matching ODE."""

    def __init__(self, config: VoxtralTTSConfig):
        super().__init__()
        self.config = config
        self.n_acoustic_codebook = config.n_acoustic_codebook
        self.acoustic_levels = config.acoustic_codebook_size

        # Projections
        self.input_projection = nn.Linear(
            config.n_acoustic_codebook, config.at_dim, bias=False
        )
        self.time_projection = nn.Linear(config.at_dim, config.at_dim, bias=False)
        self.llm_projection = nn.Linear(config.at_input_dim, config.at_dim, bias=False)

        # Time embedding
        self.time_embedding = TimeEmbedding(config.at_dim)

        # Transformer layers
        self.layers = nn.ModuleList(
            [AcousticTransformerBlock(config) for _ in range(config.at_n_layers)]
        )
        self.norm = RMSNorm(config.at_dim, config.at_norm_eps)

        # Output heads
        # semantic_codebook_output operates on LLM hidden states directly
        # (at_input_dim), matching vLLM's FlowMatchingAudioTransformer.forward()
        self.semantic_codebook_output = nn.Linear(
            config.at_input_dim,
            config.padded_semantic_vocab_size,
            bias=config.at_use_biases,
        )
        self.acoustic_codebook_output = nn.Linear(
            config.at_dim, config.n_acoustic_codebook, bias=False
        )

        # Flow matching timesteps
        self.register_buffer(
            "timesteps",
            torch.linspace(0, 1, _ACOUSTIC_DECODE_ITERS),
            persistent=False,
        )

    def _predict_velocity(
        self,
        x_t: torch.Tensor,
        llm_output: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Single velocity prediction step.

        Args:
            x_t: (B, n_acoustic_codebook) current sample.
            llm_output: (B, llm_dim) hidden states from LLM.
            t_emb: (B, at_dim) time embedding.
        Returns:
            velocity: (B, n_acoustic_codebook).
        """
        x_t = x_t.to(llm_output.dtype)
        t_emb = self.time_projection(t_emb)
        llm_output = self.llm_projection(llm_output)

        # Build 3-token sequence: [acoustic_input, time, llm]
        h = torch.cat(
            [
                self.input_projection(x_t.unsqueeze(1)),
                t_emb.unsqueeze(1),
                llm_output.unsqueeze(1),
            ],
            dim=1,
        )  # (B, 3, at_dim)

        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        # Predict velocity from the first token
        return self.acoustic_codebook_output(h[:, 0, :].contiguous())

    def decode_one_frame(
        self, hidden_states: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Generate one frame of audio codes via flow matching.

        Args:
            hidden_states: (B, llm_dim) normed LLM hidden states.
            noise: (B, n_acoustic_codebook) initial noise from runner.
        Returns:
            audio_codes: (B, 1 + n_acoustic_codebook) semantic + acoustic codes.
        """
        B = hidden_states.shape[0]

        # Semantic code via greedy decoding
        semantic_logit = self.semantic_codebook_output(hidden_states).float()
        semantic_logit[:, _EMPTY_AUDIO_TOKEN_ID] = -float("inf")
        sem_vocab = self.config.semantic_codebook_size + _N_AUDIO_SPECIAL_TOKENS
        semantic_logit[:, sem_vocab:] = -float("inf")
        semantic_code = semantic_logit.argmax(dim=-1, keepdim=True)  # (B, 1)

        should_decode = semantic_code.squeeze(1) != _END_AUDIO_TOKEN_ID

        # Flow matching Euler ODE
        x = noise * _NOISE_SCALE
        hidden_zero = torch.zeros_like(hidden_states)
        timesteps = self.timesteps.to(dtype=hidden_states.dtype)

        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            dt = timesteps[i + 1] - timesteps[i]

            t_emb = self.time_embedding(t.view(-1, 1).repeat(B, 1)).to(
                hidden_states.dtype
            )

            # Batch conditional + unconditional for CFG
            x_batched = torch.cat([x, x], dim=0)
            llm_batched = torch.cat([hidden_states, hidden_zero], dim=0)
            t_emb_batched = torch.cat([t_emb, t_emb], dim=0)

            v_all = self._predict_velocity(x_batched, llm_batched, t_emb_batched)
            v_cond, v_uncond = v_all[:B], v_all[B:]
            v = _CFG_ALPHA * v_cond + (1 - _CFG_ALPHA) * v_uncond

            x = x + v * dt

        # Quantize to discrete codes
        x = torch.clamp(x, -1, 1)
        scaled = ((x + 1) / 2) * (self.acoustic_levels - 1)
        acoustic_codes = scaled.round().long()
        acoustic_codes[~should_decode] = _EMPTY_AUDIO_TOKEN_ID
        acoustic_codes = acoustic_codes + _N_AUDIO_SPECIAL_TOKENS

        return torch.cat([semantic_code, acoustic_codes], dim=1)


# ---------------------------------------------------------------------------
# Audio token embedding
# ---------------------------------------------------------------------------


class AudioTokenEmbedding(nn.Module):
    """Multi-codebook embedding with per-codebook offsets, summed across codebooks."""

    def __init__(self, config: VoxtralTTSConfig):
        super().__init__()
        # Codebook sizes with special tokens, no padding
        sizes = [config.semantic_codebook_size + _N_AUDIO_SPECIAL_TOKENS] + [
            config.acoustic_codebook_size + _N_AUDIO_SPECIAL_TOKENS
        ] * config.n_acoustic_codebook
        total = sum(sizes)
        padded = _round_up(total, 128)

        offsets = [0]
        for s in sizes[:-1]:
            offsets.append(offsets[-1] + s)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        self.embeddings = nn.Embedding(padded, config.dim)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            codes: (B, n_codebooks, seq_len) integer audio codes.
        Returns:
            embeds: (B, seq_len, dim) summed across codebooks.
        """
        # Offset each codebook into the shared vocabulary.
        # Explicit long cast ensures indices stay int64 (not promoted to bf16).
        offset_codes = codes + self.offsets.long().unsqueeze(0).unsqueeze(2)
        # (B, n_codebooks, seq_len, dim)
        emb = self.embeddings(offset_codes)
        # Sum across codebooks → (B, seq_len, dim)
        return emb.sum(dim=1)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class VoxtralTTSModel(nn.Module):
    def __init__(self, config: VoxtralTTSConfig, max_seq_len: int):
        super().__init__()
        self.config = config
        self.decoder = MistralDecoder(config, max_seq_len)
        self.acoustic_transformer = FlowMatchingAudioTransformer(config)
        self.audio_token_embedding = AudioTokenEmbedding(config)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


_KEY_MAP = {
    # LM decoder
    "layers.": "decoder.layers.",
    "norm.weight": "decoder.norm.weight",
    "output.weight": "decoder.output.weight",
    # Token embeddings
    "mm_audio_embeddings.tok_embeddings.weight": "decoder.tok_embeddings.weight",
    # Audio token embedding
    "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight": "audio_token_embedding.embeddings.weight",
    "mm_audio_embeddings.audio_codebook_embeddings.embeddings.bias": "audio_token_embedding.embeddings.bias",
    # Acoustic transformer (prefix stripped by load_weights)
    "acoustic_transformer.": "acoustic_transformer.",
}


def _map_checkpoint_key(ckpt_key: str) -> str | None:
    """Map a checkpoint key to the model state dict key."""
    # Audio tokenizer weights (codec) — skip for model.pte
    if ckpt_key.startswith("audio_tokenizer."):
        return None

    # Acoustic transformer: strip prefix
    if ckpt_key.startswith("acoustic_transformer."):
        return ckpt_key

    # Audio embeddings
    if ckpt_key.startswith("mm_audio_embeddings."):
        for src, dst in _KEY_MAP.items():
            if ckpt_key == src or ckpt_key.startswith(src):
                return ckpt_key.replace(src, dst, 1)
        return None

    # LM decoder
    if ckpt_key.startswith("layers."):
        return "decoder." + ckpt_key

    if ckpt_key == "norm.weight":
        return "decoder.norm.weight"

    if ckpt_key == "output.weight":
        return "decoder.output.weight"

    # Try direct mapping
    return None


def load_model(
    model_path: str,
    max_seq_len: int = 4096,
    dtype: torch.dtype = torch.float32,
    backend: str = "metal",
) -> VoxtralTTSModel:
    from safetensors import safe_open

    model_dir = Path(model_path)
    config = VoxtralTTSConfig.from_params_json(str(model_dir / "params.json"))
    config.max_seq_len = max_seq_len
    config.backend = backend

    print(
        f"VoxtralTTS config: dim={config.dim}, n_layers={config.n_layers}, "
        f"at_dim={config.at_dim}, at_n_layers={config.at_n_layers}, "
        f"n_acoustic_codebook={config.n_acoustic_codebook}, backend={backend}"
    )

    with torch.device("meta"):
        model = VoxtralTTSModel(config, max_seq_len)

    ckpt_path = str(model_dir / "consolidated.safetensors")
    state_dict = {}
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        for ckpt_key in f.keys():
            model_key = _map_checkpoint_key(ckpt_key)
            if model_key is None:
                continue
            state_dict[model_key] = f.get_tensor(ckpt_key).to(dtype)

    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)

    # Re-tie output weights
    model.decoder.output.weight = model.decoder.tok_embeddings.weight

    # Materialize remaining meta buffers (KV caches, etc.)
    # Preserve original dtype — int64 index buffers must not become bf16.
    for fqn, buf in list(model.named_buffers()):
        if buf.device.type == "meta":
            parts = fqn.rsplit(".", 1)
            parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
            buf_dtype = buf.dtype if not buf.dtype.is_floating_point else dtype
            parent.register_buffer(
                parts[-1], torch.zeros(buf.shape, dtype=buf_dtype, device="cpu")
            )

    # Recompute RoPE
    dec_cos, dec_sin = precompute_freqs_cis(
        config.head_dim, max_seq_len, config.rope_theta
    )
    model.decoder.register_buffer("freqs_cos", dec_cos)
    model.decoder.register_buffer("freqs_sin", dec_sin)

    # Recompute flow matching timesteps (persistent=False, lost during meta construction)
    model.acoustic_transformer.register_buffer(
        "timesteps",
        torch.linspace(0, 1, _ACOUSTIC_DECODE_ITERS),
        persistent=False,
    )

    # Recompute audio token embedding offsets (lost during meta construction)
    sizes = [config.semantic_codebook_size + _N_AUDIO_SPECIAL_TOKENS] + [
        config.acoustic_codebook_size + _N_AUDIO_SPECIAL_TOKENS
    ] * config.n_acoustic_codebook
    offsets = [0]
    for s in sizes[:-1]:
        offsets.append(offsets[-1] + s)
    model.audio_token_embedding.register_buffer(
        "offsets", torch.tensor(offsets, dtype=torch.long)
    )

    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    model.eval()
    return model

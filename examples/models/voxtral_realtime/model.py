# Voxtral-Mini-4B-Realtime-2602 reference implementation for ExecuTorch.
# Based on the Mistral model released under the Apache-2.0 license.
# See https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602

"""Voxtral-Mini-4B-Realtime-2602 eager model for ExecuTorch.

See model.md for architecture details and design choices.
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
# Config
# ---------------------------------------------------------------------------


@dataclass
class VoxtralRealtimeConfig:
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
    ada_rms_norm_t_cond_dim: int = 32
    # Encoder
    enc_dim: int = 1280
    enc_n_layers: int = 32
    enc_n_heads: int = 32
    enc_head_dim: int = 64
    enc_hidden_dim: int = 5120
    enc_rope_theta: float = 1_000_000.0
    enc_norm_eps: float = 1e-5
    # Audio
    num_mel_bins: int = 128
    downsample_factor: int = 4
    # Runtime
    max_seq_len: int = 4096

    @staticmethod
    def from_params_json(path: str) -> "VoxtralRealtimeConfig":
        with open(path) as f:
            p = json.load(f)
        enc = p["multimodal"]["whisper_model_args"]["encoder_args"]
        ds = p["multimodal"]["whisper_model_args"]["downsample_args"]
        audio = enc.get("audio_encoding_args", {})
        return VoxtralRealtimeConfig(
            dim=p["dim"],
            n_layers=p["n_layers"],
            n_heads=p["n_heads"],
            n_kv_heads=p["n_kv_heads"],
            head_dim=p["head_dim"],
            hidden_dim=p["hidden_dim"],
            vocab_size=p["vocab_size"],
            rope_theta=p["rope_theta"],
            norm_eps=p["norm_eps"],
            ada_rms_norm_t_cond_dim=p["ada_rms_norm_t_cond_dim"],
            enc_dim=enc["dim"],
            enc_n_layers=enc["n_layers"],
            enc_n_heads=enc["n_heads"],
            enc_head_dim=enc["head_dim"],
            enc_hidden_dim=enc["hidden_dim"],
            enc_rope_theta=enc["rope_theta"],
            enc_norm_eps=enc["norm_eps"],
            num_mel_bins=audio.get("num_mel_bins", 128),
            downsample_factor=ds["downsample_factor"],
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


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings.

    q, k: (B, T, n_heads, head_dim).  freqs: (T, head_dim//2).

    Uses reshape+unbind (export-friendly, avoids stride-2 slicing) and
    float32 upcast for numerical stability (matches Llama pattern).
    """
    q_r, q_i = q.float().reshape(q.shape[:-1] + (-1, 2)).unbind(-1)
    k_r, k_i = k.float().reshape(k.shape[:-1] + (-1, 2)).unbind(-1)

    fc = freqs_cos.unsqueeze(0).unsqueeze(2)  # (1, T, 1, D/2)
    fs = freqs_sin.unsqueeze(0).unsqueeze(2)

    q_out = torch.stack([q_r * fc - q_i * fs, q_r * fs + q_i * fc], dim=-1).flatten(-2)
    k_out = torch.stack([k_r * fc - k_i * fs, k_r * fs + k_i * fc], dim=-1).flatten(-2)

    return q_out.type_as(q), k_out.type_as(k)


# ---------------------------------------------------------------------------
# Encoder components
# ---------------------------------------------------------------------------


class CausalConv1d(nn.Module):
    """Conv1d with left-only (causal) padding."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, bias=True)
        self.pad_length = kernel_size - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad_length, 0)))


class EncoderAttention(nn.Module):
    """Multi-head attention with RoPE for the causal whisper encoder.

    Biases: wq yes, wk no, wv yes, wo yes.
    """

    def __init__(self, dim: int, n_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        attn_dim = n_heads * head_dim
        self.wq = nn.Linear(dim, attn_dim, bias=True)
        self.wk = nn.Linear(dim, attn_dim, bias=False)
        self.wv = nn.Linear(dim, attn_dim, bias=True)
        self.wo = nn.Linear(attn_dim, dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim)
        q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(y.transpose(1, 2).contiguous().view(B, T, -1))


class EncoderSwiGLU(nn.Module):
    """SwiGLU FFN for the encoder. Biases: w1 no, w2 yes, w3 no."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=True)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CausalEncoderLayer(nn.Module):
    def __init__(self, config: VoxtralRealtimeConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.enc_dim, config.enc_norm_eps)
        self.attention = EncoderAttention(
            config.enc_dim, config.enc_n_heads, config.enc_head_dim
        )
        self.ffn_norm = RMSNorm(config.enc_dim, config.enc_norm_eps)
        self.feed_forward = EncoderSwiGLU(config.enc_dim, config.enc_hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class CausalWhisperEncoder(nn.Module):
    """Causal whisper encoder: 2 causal Conv1d + 32 transformer layers + RMSNorm.

    Input: (B, n_mels, T_mel).  Output: (B, T_mel//2, enc_dim).
    """

    def __init__(self, config: VoxtralRealtimeConfig, max_enc_len: int = 16384):
        super().__init__()
        self.conv_layers = nn.ModuleList(
            [
                CausalConv1d(config.num_mel_bins, config.enc_dim, 3, stride=1),
                CausalConv1d(config.enc_dim, config.enc_dim, 3, stride=2),
            ]
        )
        self.layers = nn.ModuleList(
            [CausalEncoderLayer(config) for _ in range(config.enc_n_layers)]
        )
        self.norm = RMSNorm(config.enc_dim, config.enc_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(
            config.enc_head_dim, max_enc_len, config.enc_rope_theta
        )
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.conv_layers[0](mel))
        x = F.gelu(self.conv_layers[1](x))
        x = x.transpose(1, 2)  # (B, T', enc_dim)

        T = x.shape[1]
        freqs_cos = self.freqs_cos[:T]
        freqs_sin = self.freqs_sin[:T]

        for layer in self.layers:
            x = layer(x, freqs_cos, freqs_sin)

        return self.norm(x)


# ---------------------------------------------------------------------------
# Audio-language adapter
# ---------------------------------------------------------------------------


class AudioLanguageAdapter(nn.Module):
    """Linear(5120, 3072) -> GELU -> Linear(3072, 3072). No biases."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # Named to match checkpoint: audio_language_projection.{0,2}
        self.w_in = nn.Linear(in_dim, out_dim, bias=False)
        self.w_out = nn.Linear(out_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(F.gelu(self.w_in(x)))


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------


def compute_time_embedding(
    n_delay_tokens: int, dim: int, theta: float = 10000.0
) -> torch.Tensor:
    """Sinusoidal time embedding. Returns (1, dim)."""
    inv_freq = torch.exp(-math.log(theta) * torch.arange(dim // 2).float() / (dim // 2))
    t = torch.tensor([n_delay_tokens], dtype=torch.float)
    emb = t.unsqueeze(-1) * inv_freq  # (1, dim//2)
    return torch.cat([emb.cos(), emb.sin()], dim=-1)  # (1, dim)


# ---------------------------------------------------------------------------
# LM decoder components
# ---------------------------------------------------------------------------


class KVCache(nn.Module):
    """KV cache in [B, S, H, D] layout for torch.ops.llama.update_cache."""

    def __init__(self, max_seq_len: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        cache_shape = (1, max_seq_len, n_kv_heads, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape))
        self.register_buffer("v_cache", torch.zeros(cache_shape))

    def update(
        self, input_pos: torch.Tensor, k_val: torch.Tensor, v_val: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write k_val/v_val into cache and return full cache.

        Args:
            input_pos: (seq_len,) position indices.
            k_val, v_val: (B, seq_len, n_kv_heads, head_dim).
        Returns:
            k_cache, v_cache: (B, max_seq_len, n_kv_heads, head_dim).
        """
        start_pos = input_pos[0].item()
        torch.ops.llama.update_cache(k_val, self.k_cache, start_pos)
        torch.ops.llama.update_cache(v_val, self.v_cache, start_pos)
        return self.k_cache, self.v_cache


class SDPA(nn.Module):
    """Scaled dot-product attention using torch.ops.llama.custom_sdpa.

    Handles GQA expansion and causal masking internally via the fused kernel.
    All tensors in [B, S, H, D] layout — no transposes needed.
    """

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
    ) -> torch.Tensor:
        """
        Args:
            input_pos: (seq_len,) position indices.
            q: (B, seq_len, n_heads, head_dim).
            k, v: (B, max_seq_len, n_kv_heads, head_dim) — full KV cache.
            bsz, seqlen: batch size and query sequence length.
        Returns:
            output: (B, seq_len, n_heads * head_dim).
        """
        input_dtype = q.dtype
        q = q.to(dtype=torch.float32)
        k = k.to(dtype=torch.float32)
        v = v.to(dtype=torch.float32)
        start_pos = input_pos[0].item()
        y = torch.ops.llama.custom_sdpa(
            q,
            k,
            v,
            start_pos,
            None,
            0,
            True,
        )
        return y.view(bsz, seqlen, self.dim).to(dtype=input_dtype)


class LMAttention(nn.Module):
    """GQA with RoPE, KV cache, and fused SDPA. No biases.

    Data flows in [B, T, H, D] throughout — no transposes in the hot path.
    GQA expansion is handled inside the custom_sdpa kernel.
    Causal masking is handled via start_pos — no pre-built mask buffer.
    """

    def __init__(self, config: VoxtralRealtimeConfig, max_seq_len: int):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.dim = config.dim

        self.wq = nn.Linear(config.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, config.dim, bias=False)

        self.kv_cache = KVCache(max_seq_len, self.n_kv_heads, self.head_dim)
        self.sdpa = SDPA(self.n_heads, self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        input_pos: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)

        k, v = self.kv_cache.update(input_pos, k, v)

        y = self.sdpa(input_pos, q, k, v, B, T)

        return self.wo(y)


class LMMLP(nn.Module):
    """SwiGLU FFN for the LM. No biases."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MistralDecoderLayer(nn.Module):
    """Decoder layer: attention_norm -> attention -> residual ->
    adaptive_ffn_norm -> feed_forward -> residual.

    The adaptive RMSNorm applies a time-conditioned scale after the base norm:
      scale = 1 + Sequential(Linear, GELU, Linear)(t_cond)
    """

    def __init__(self, config: VoxtralRealtimeConfig, max_seq_len: int):
        super().__init__()
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention = LMAttention(config, max_seq_len)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        # nn.Sequential indices 0, 1, 2 match checkpoint keys .0.weight, .2.weight
        self.ada_rms_norm_t_cond = nn.Sequential(
            nn.Linear(config.dim, config.ada_rms_norm_t_cond_dim, bias=False),
            nn.GELU(),
            nn.Linear(config.ada_rms_norm_t_cond_dim, config.dim, bias=False),
        )
        self.feed_forward = LMMLP(config.dim, config.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        input_pos: torch.Tensor,
        t_cond: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin, input_pos)
        normed = self.ffn_norm(x)
        scale = 1.0 + self.ada_rms_norm_t_cond(t_cond)
        x = x + self.feed_forward(normed * scale)
        return x


class MistralDecoder(nn.Module):
    """Mistral LM decoder with tied embeddings."""

    def __init__(self, config: VoxtralRealtimeConfig, max_seq_len: int):
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
        self,
        input_embeds: torch.Tensor,
        input_pos: torch.Tensor,
        t_cond: torch.Tensor,
    ) -> torch.Tensor:
        freqs_cos = self.freqs_cos[input_pos]
        freqs_sin = self.freqs_sin[input_pos]

        x = input_embeds
        for layer in self.layers:
            x = layer(x, freqs_cos, freqs_sin, input_pos, t_cond)

        return self.output(self.norm(x))


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class VoxtralRealtimeModel(nn.Module):
    def __init__(self, config: VoxtralRealtimeConfig, max_seq_len: int | None = None):
        super().__init__()
        if max_seq_len is None:
            max_seq_len = config.max_seq_len
        self.config = config

        self.encoder = CausalWhisperEncoder(config)
        self.adapter = AudioLanguageAdapter(
            config.enc_dim * config.downsample_factor, config.dim
        )
        self.decoder = MistralDecoder(config, max_seq_len)

        # Tie output and embedding weights
        self.decoder.output.weight = self.decoder.tok_embeddings.weight

    def encode_audio(self, mel: torch.Tensor) -> torch.Tensor:
        """Encode mel spectrogram to audio embeddings.

        Args:
            mel: (B, n_mels, T_mel) mel spectrogram in channels-first format.
                 T_mel must be a multiple of 8 (conv stride 2 halves it, then
                 downsample by 4).
        Returns:
            audio_embeds: (B, T_mel // 8, dim).
        """
        x = self.encoder(mel)  # (B, T_enc, enc_dim)
        B, T, D = x.shape
        ds = self.config.downsample_factor
        x = x.reshape(B, T // ds, D * ds)
        return self.adapter(x)

    def text_decoder(
        self,
        input_embeds: torch.Tensor,
        input_pos: torch.Tensor,
        t_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Run LM decoder.

        Args:
            input_embeds: (B, seq_len, dim) combined audio+text embeddings.
            input_pos: (seq_len,) position indices for RoPE and KV cache.
            t_cond: (1, dim) precomputed time embedding.
        Returns:
            logits: (B, seq_len, vocab_size).
        """
        return self.decoder(input_embeds, input_pos, t_cond)

    def token_embedding(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up token embeddings.

        Args:
            token_ids: (B, seq_len) token indices.
        Returns:
            embeds: (B, seq_len, dim).
        """
        return self.decoder.tok_embeddings(token_ids)


# ---------------------------------------------------------------------------
# Streaming encoder
# ---------------------------------------------------------------------------


class StreamingAudioEncoderExport(nn.Module):
    """Streaming encoder: processes one 8-mel-frame chunk at a time.

    Shares conv/transformer/adapter weights with the offline encoder.
    Owns separate KV caches and SDPA for incremental KV-cached attention.

    Forward:
        mel_chunk(1,128,8) + conv1_state(1,128,2) + conv2_state(1,1280,2)
        + enc_input_pos(4,)
        -> audio_embeds(1,1,3072), new_conv1_state(1,128,2), new_conv2_state(1,1280,2)
    """

    def __init__(self, model: VoxtralRealtimeModel, max_enc_len: int = 750):
        super().__init__()
        config = model.config

        # Shared encoder weights (read-only references, never mutated)
        self.conv1 = model.encoder.conv_layers[0].conv
        self.conv2 = model.encoder.conv_layers[1].conv
        self.layers = model.encoder.layers
        self.enc_norm = model.encoder.norm
        self.adapter = model.adapter

        self.downsample_factor = config.downsample_factor
        self.n_heads = config.enc_n_heads
        self.head_dim = config.enc_head_dim

        # Streaming-specific: encoder KV caches (one per layer)
        self.kv_caches = nn.ModuleList(
            [
                KVCache(max_enc_len, config.enc_n_heads, config.enc_head_dim)
                for _ in range(config.enc_n_layers)
            ]
        )

        # SDPA for encoder MHA (n_heads=32, head_dim=64 -> attn_dim=2048)
        self.sdpa = SDPA(config.enc_n_heads, config.enc_head_dim)

        # RoPE for encoder dimensions
        freqs_cos, freqs_sin = precompute_freqs_cis(
            config.enc_head_dim, max_enc_len, config.enc_rope_theta
        )
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def _streaming_encoder_layer(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        input_pos: torch.Tensor,
        layer: CausalEncoderLayer,
        layer_idx: int,
    ) -> torch.Tensor:
        """One encoder layer with streaming attention (KV cache + custom_sdpa)."""
        h = layer.attention_norm(x)

        B, T, _ = h.shape
        attn = layer.attention
        q = attn.wq(h).view(B, T, self.n_heads, self.head_dim)
        k = attn.wk(h).view(B, T, self.n_heads, self.head_dim)
        v = attn.wv(h).view(B, T, self.n_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)
        k, v = self.kv_caches[layer_idx].update(input_pos, k, v)
        y = self.sdpa(input_pos, q, k, v, B, T)
        y = attn.wo(y)

        x = x + y
        x = x + layer.feed_forward(layer.ffn_norm(x))
        return x

    def forward(
        self,
        mel_chunk: torch.Tensor,
        conv1_state: torch.Tensor,
        conv2_state: torch.Tensor,
        enc_input_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Conv1: cat state + chunk, raw Conv1d (no CausalConv1d padding)
        # (1, 128, 2+8=10) -> conv1(k=3, s=1) -> (1, 1280, 8)
        conv1_input = torch.cat([conv1_state, mel_chunk], dim=2)
        conv1_out = F.gelu(self.conv1(conv1_input))
        new_conv1_state = mel_chunk[:, :, -2:]

        # Conv2: cat state + conv1_out, raw Conv1d
        # (1, 1280, 2+8=10) -> conv2(k=3, s=2) -> (1, 1280, 4)
        conv2_input = torch.cat([conv2_state, conv1_out], dim=2)
        conv2_out = F.gelu(self.conv2(conv2_input))
        new_conv2_state = conv1_out[:, :, -2:]

        x = conv2_out.transpose(1, 2)  # (1, 4, 1280)

        # Encoder transformer with KV cache
        freqs_cos = self.freqs_cos[enc_input_pos]
        freqs_sin = self.freqs_sin[enc_input_pos]

        for i, layer in enumerate(self.layers):
            x = self._streaming_encoder_layer(
                x, freqs_cos, freqs_sin, enc_input_pos, layer, i
            )

        x = self.enc_norm(x)  # (1, 4, 1280)

        # Downsample: concat 4 consecutive frames -> (1, 1, 5120)
        B, T, D = x.shape
        x = x.reshape(B, T // self.downsample_factor, D * self.downsample_factor)

        audio_embeds = self.adapter(x)  # (1, 1, 3072)

        return audio_embeds, new_conv1_state, new_conv2_state


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

_MM_PREFIX = "mm_streams_embeddings.embedding_module."


def _map_checkpoint_key(ckpt_key: str) -> str | None:
    """Map Mistral consolidated checkpoint key to model state_dict key."""
    if ckpt_key.startswith(_MM_PREFIX):
        suffix = ckpt_key[len(_MM_PREFIX) :]

        # Encoder convolutions
        if suffix.startswith("whisper_encoder.conv_layers."):
            return "encoder." + suffix.replace("whisper_encoder.", "")

        # Encoder transformer layers
        if suffix.startswith("whisper_encoder.transformer.layers."):
            return "encoder." + suffix.replace("whisper_encoder.transformer.", "")

        # Encoder final norm
        if suffix.startswith("whisper_encoder.transformer.norm."):
            return "encoder." + suffix.replace("whisper_encoder.transformer.", "")

        # Audio-language adapter
        if suffix == "audio_language_projection.0.weight":
            return "adapter.w_in.weight"
        if suffix == "audio_language_projection.2.weight":
            return "adapter.w_out.weight"

        # Token embeddings (tied with output)
        if suffix == "tok_embeddings.weight":
            return "decoder.tok_embeddings.weight"

        return None

    # LM decoder layers
    if ckpt_key.startswith("layers."):
        return "decoder." + ckpt_key

    # LM final norm
    if ckpt_key == "norm.weight":
        return "decoder.norm.weight"

    return None


def load_model(
    model_path: str,
    max_seq_len: int = 4096,
    n_delay_tokens: int = 6,
    dtype: torch.dtype = torch.float32,
) -> VoxtralRealtimeModel:
    """Load VoxtralRealtimeModel from a Mistral consolidated checkpoint.

    Uses meta-device construction + assign-based loading to minimize peak
    memory (avoids allocating ~17 GB of random weights before overwriting
    them with checkpoint data).

    Args:
        model_path: Directory containing params.json and consolidated.safetensors.
        max_seq_len: Maximum sequence length for KV cache.
        n_delay_tokens: Transcription delay in tokens (default 6 = 480ms).
        dtype: Weight dtype (default: float32).
    """
    from safetensors import safe_open

    model_dir = Path(model_path)
    config = VoxtralRealtimeConfig.from_params_json(str(model_dir / "params.json"))
    config.max_seq_len = max_seq_len

    print(
        f"Building model on meta device (dim={config.dim}, enc_dim={config.enc_dim}, "
        f"layers={config.n_layers}, enc_layers={config.enc_n_layers})..."
    )
    with torch.device("meta"):
        model = VoxtralRealtimeModel(config, max_seq_len)

    print(f"Loading weights from {model_dir / 'consolidated.safetensors'}...")
    ckpt_path = str(model_dir / "consolidated.safetensors")
    state_dict = {}
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        for ckpt_key in f.keys():
            model_key = _map_checkpoint_key(ckpt_key)
            if model_key is None:
                print(f"  Skipping unmapped key: {ckpt_key}")
                continue
            state_dict[model_key] = f.get_tensor(ckpt_key).to(dtype)

    # assign=True replaces meta tensors by reference instead of copying,
    # avoiding a full duplication of all weights in memory.
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)

    # Re-tie output weights (assign=True breaks the tie established in __init__)
    model.decoder.output.weight = model.decoder.tok_embeddings.weight

    # Materialize remaining meta-device buffers (KV caches, RoPE freqs)
    # that weren't in the checkpoint. Use model dtype for KV caches so they
    # match the K/V values from the model (update_cache requires same dtype).
    # RoPE freqs are overwritten below so their dtype here doesn't matter.
    for fqn, buf in list(model.named_buffers()):
        if buf.device.type == "meta":
            parts = fqn.rsplit(".", 1)
            parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
            parent.register_buffer(
                parts[-1],
                torch.zeros(buf.shape, dtype=dtype, device="cpu"),
            )

    # Recompute RoPE frequency tables (the zero-fill above is wrong for these)
    enc_cos, enc_sin = precompute_freqs_cis(
        config.enc_head_dim, 16384, config.enc_rope_theta
    )
    model.encoder.register_buffer("freqs_cos", enc_cos)
    model.encoder.register_buffer("freqs_sin", enc_sin)
    dec_cos, dec_sin = precompute_freqs_cis(
        config.head_dim, max_seq_len, config.rope_theta
    )
    model.decoder.register_buffer("freqs_cos", dec_cos)
    model.decoder.register_buffer("freqs_sin", dec_sin)

    # Validate
    runtime_prefixes = (
        "decoder.output.weight",
        "decoder.freqs_",
        "encoder.freqs_",
        ".kv_cache.",
    )
    actual_missing = set(missing)
    expected_missing = {
        k for k in actual_missing if any(p in k for p in runtime_prefixes)
    }
    extra_missing = actual_missing - expected_missing
    if extra_missing:
        print(f"  WARNING: missing keys: {sorted(extra_missing)}")
    if unexpected:
        print(f"  WARNING: unexpected keys: {sorted(unexpected)}")

    loaded = len(state_dict) - len(unexpected)
    print(
        f"  Loaded {loaded} tensors ({len(expected_missing)} runtime buffers OK, "
        f"{len(extra_missing)} unexpected missing)"
    )

    # Precompute time embedding as a constant buffer (must match model dtype)
    t_cond = compute_time_embedding(n_delay_tokens, config.dim).to(dtype)
    model.register_buffer("t_cond", t_cond)

    model.eval()
    return model

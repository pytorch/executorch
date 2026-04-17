# Voxtral-4B-TTS-2603 reference implementation for ExecuTorch.
# Based on the Mistral model released under the CC-BY-NC-4.0 license.
# See https://huggingface.co/mistralai/Voxtral-4B-TTS-2603

"""Voxtral-4B-TTS-2603 eager model for ExecuTorch.

Three-component architecture:
  1. Mistral LLM backbone (~4B params) — autoregressive text-to-hidden-states
  2. FlowMatchingHead — hidden states to 37 audio codebook tokens per frame
  3. CodecDecoder — codebook tokens to 24kHz waveform

See the plan document for architecture details.
"""

import json
import math
from copy import deepcopy
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
class VoxtralTTSConfig:
    # LLM (Mistral backbone)
    dim: int = 3072
    n_layers: int = 26
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128
    hidden_dim: int = 9216
    vocab_size: int = 131072
    rope_theta: float = 1_000_000.0
    norm_eps: float = 1e-5
    # Acoustic transformer (flow matching head) — defaults match 4B checkpoint
    at_dim: int = 3072
    at_n_layers: int = 3
    at_n_heads: int = 32
    at_n_kv_heads: int = 8
    at_head_dim: int = 128
    at_hidden_dim: int = 9216
    at_norm_eps: float = 1e-5
    at_use_biases: bool = False
    n_decoding_steps: int = 7
    cfg_alpha: float = 1.2
    noise_scale: float = 1.0
    audio_token_id: int = 24
    begin_audio_token_id: int = 25
    text_to_audio_token_id: int = 36
    repeat_audio_text_token_id: int = 35
    # Codebooks
    semantic_codebook_size: int = 8192
    semantic_dim: int = 256
    acoustic_levels: int = 21
    acoustic_dim: int = 36
    # Codec decoder
    codec_dim: int = 1024
    codec_hidden_dim: int = 4096
    codec_n_heads: int = 8
    codec_n_kv_heads: int = 8
    codec_head_dim: int = 128
    codec_norm_eps: float = 1e-2
    codec_qk_norm_eps: float = 1e-6
    codec_sliding_window: int = 16
    codec_patch_size: int = 240
    codec_use_biases: bool = False
    codec_layer_scale: bool = True
    codec_conv_weight_norm: bool = True
    codec_causal: bool = True
    codec_half_attn_window_upon_downsampling: bool = True
    codec_decoder_transformer_lengths: tuple[int, ...] = (2, 2, 2, 2)
    codec_decoder_convs_kernels: tuple[int, ...] = (3, 4, 4, 4)
    codec_decoder_convs_strides: tuple[int, ...] = (1, 2, 2, 2)
    sampling_rate: int = 24000
    # Runtime
    max_seq_len: int = 4096
    backend: str = "xnnpack"

    @staticmethod
    def from_params_json(path: str) -> "VoxtralTTSConfig":
        with open(path) as f:
            p = json.load(f)

        mm = p.get("multimodal", {})
        audio_model = mm.get("audio_model_args", {})
        at_args = audio_model.get("acoustic_transformer_args", {})
        tokenizer_args = mm.get("audio_tokenizer_args", {})
        audio_enc = audio_model.get("audio_encoding_args", {})

        # Parse codebook sizes from comma-separated string or individual fields
        if "codebook_sizes" in audio_model:
            cb_sizes = [int(c) for c in audio_model["codebook_sizes"].split(",")]
            semantic_cb_size = cb_sizes[0]
            acoustic_cb_size = cb_sizes[1] if len(cb_sizes) > 1 else 21
            n_acoustic = len(cb_sizes) - 1
        else:
            semantic_cb_size = audio_model.get("semantic_codebook_size", 8192)
            acoustic_cb_size = audio_model.get("acoustic_codebook_size", 21)
            n_acoustic = audio_model.get("n_acoustic_codebook", 36)

        def _str2tuple(s: str) -> tuple[int, ...]:
            return tuple(int(x) for x in s.split(","))

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
            at_dim=at_args.get("dim", 3072),
            at_n_layers=at_args.get("n_layers", 3),
            at_n_heads=at_args.get("n_heads", 32),
            at_n_kv_heads=at_args.get("n_kv_heads", 8),
            at_head_dim=at_args.get("head_dim", 128),
            at_hidden_dim=at_args.get("hidden_dim", 9216),
            at_norm_eps=at_args.get("norm_eps", 1e-5),
            at_use_biases=at_args.get("use_biases", False),
            n_decoding_steps=at_args.get("n_decoding_steps", 7),
            audio_token_id=audio_model.get("audio_token_id", 24),
            begin_audio_token_id=audio_model.get("begin_audio_token_id", 25),
            text_to_audio_token_id=audio_model.get("text_to_audio_token_id", 36),
            semantic_codebook_size=semantic_cb_size,
            acoustic_levels=acoustic_cb_size,
            acoustic_dim=n_acoustic,
            codec_dim=tokenizer_args.get("dim", 1024),
            codec_hidden_dim=tokenizer_args.get("hidden_dim", 4096),
            codec_n_heads=tokenizer_args.get("n_heads", 8),
            codec_n_kv_heads=tokenizer_args.get("n_kv_heads", 8),
            codec_head_dim=tokenizer_args.get("head_dim", 128),
            codec_norm_eps=tokenizer_args.get("norm_eps", 1e-2),
            codec_qk_norm_eps=tokenizer_args.get("qk_norm_eps", 1e-6),
            codec_sliding_window=tokenizer_args.get("attn_sliding_window_size", 16),
            codec_patch_size=tokenizer_args.get("pretransform_patch_size", 240),
            codec_use_biases=tokenizer_args.get("use_biases", False),
            codec_layer_scale=tokenizer_args.get("layer_scale", True),
            codec_conv_weight_norm=tokenizer_args.get("conv_weight_norm", True),
            codec_causal=tokenizer_args.get("causal", True),
            codec_half_attn_window_upon_downsampling=tokenizer_args.get(
                "half_attn_window_upon_downsampling", True
            ),
            codec_decoder_transformer_lengths=_str2tuple(
                tokenizer_args.get("decoder_transformer_lengths_str", "2,2,2,2")
            ),
            codec_decoder_convs_kernels=_str2tuple(
                tokenizer_args.get("decoder_convs_kernels_str", "3,4,4,4")
            ),
            codec_decoder_convs_strides=_str2tuple(
                tokenizer_args.get("decoder_convs_strides_str", "1,2,2,2")
            ),
            sampling_rate=audio_enc.get("sampling_rate", 24000),
            semantic_dim=tokenizer_args.get("semantic_dim", 256),
        )

    @property
    def n_codebooks(self) -> int:
        return 1 + self.acoustic_dim  # 1 semantic + N acoustic

    @property
    def downsample_factor(self) -> int:
        return self.codec_patch_size * math.prod(self.codec_decoder_convs_strides)

    @property
    def frame_rate(self) -> float:
        return self.sampling_rate / self.downsample_factor


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
    """Pairwise interleaved RoPE matching mistral_inference's complex convention."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len, dtype=torch.float)
    emb = torch.outer(t, freqs)  # (max_len, head_dim/2)
    cos = emb.cos().repeat_interleave(2, dim=-1)  # (max_len, head_dim)
    sin = emb.sin().repeat_interleave(2, dim=-1)
    return cos, sin


def _rotate_interleave(x: torch.Tensor) -> torch.Tensor:
    """Pairwise rotation on adjacent pairs: (-x1, x0, -x3, x2, ...)."""
    x = x.unflatten(-1, (-1, 2))
    x = torch.stack((-x[..., 1], x[..., 0]), dim=-1)
    return x.flatten(-2)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    fc = freqs_cos.unsqueeze(0).unsqueeze(2)
    fs = freqs_sin.unsqueeze(0).unsqueeze(2)
    q_float = q.float()
    k_float = k.float()
    q_out = q_float * fc + _rotate_interleave(q_float) * fs
    k_out = k_float * fc + _rotate_interleave(k_float) * fs
    return q_out.type_as(q), k_out.type_as(k)


# ---------------------------------------------------------------------------
# LLM decoder components
# ---------------------------------------------------------------------------


class KVCache(nn.Module):
    """KV cache in [B, S, H, D] layout for torch.ops.llama.update_cache."""

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


class SDPA(nn.Module):
    """Scaled dot-product attention using torch.ops.llama.custom_sdpa."""

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
        q = q.to(dtype=torch.float32)
        k = k.to(dtype=torch.float32)
        v = v.to(dtype=torch.float32)
        start_pos = input_pos[0].item()
        torch._check_is_size(start_pos)
        if mask is not None:
            y = torch.ops.llama.custom_sdpa(
                q, k, v, start_pos, mask.to(dtype=torch.float32), 0, False,
            )
        else:
            y = torch.ops.llama.custom_sdpa(q, k, v, start_pos, None, 0, True)
        return y.view(bsz, seqlen, self.dim).to(dtype=input_dtype)


class LMAttention(nn.Module):
    """GQA with RoPE, KV cache, and SDPA. No biases."""

    def __init__(self, config: VoxtralTTSConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.dim = config.dim

        self.wq = nn.Linear(config.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, config.dim, bias=False)

        self.kv_cache = KVCache(config.max_seq_len, self.n_kv_heads, self.head_dim)
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
        y = self.sdpa(input_pos, q, k, v, B, T, attn_mask)
        return self.wo(y)


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
    """Decoder layer with standard pre-norm (no adaptive RMSNorm for TTS)."""

    def __init__(self, config: VoxtralTTSConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention = LMAttention(config)
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
    """Mistral LM decoder. Returns hidden states (no lm_head projection)."""

    def __init__(self, config: VoxtralTTSConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            [MistralDecoderLayer(config) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.dim, config.norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(
            config.head_dim, config.max_seq_len, config.rope_theta
        )
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def forward(
        self,
        input_embeds: torch.Tensor,
        input_pos: torch.Tensor,
    ) -> torch.Tensor:
        freqs_cos = self.freqs_cos[input_pos]
        freqs_sin = self.freqs_sin[input_pos]

        x = input_embeds
        for layer in self.layers:
            x = layer(x, freqs_cos, freqs_sin, input_pos)

        return self.norm(x)


# ---------------------------------------------------------------------------
# Flow matching head (acoustic transformer)
# ---------------------------------------------------------------------------

# Special token IDs for audio codebooks (0-indexed)
EMPTY_AUDIO_ID = 0
END_AUDIO_ID = 1
N_SPECIAL_TOKENS = 2


class AudioTokenEmbedding(nn.Module):
    """Embed one semantic+acoustic frame back into the LLM hidden space."""

    def __init__(self, config: VoxtralTTSConfig):
        super().__init__()
        self.codebook_sizes = [
            config.semantic_codebook_size + N_SPECIAL_TOKENS,
            *[config.acoustic_levels + N_SPECIAL_TOKENS for _ in range(config.acoustic_dim)],
        ]
        total_vocab_size = sum(self.codebook_sizes)
        padded_vocab_size = 128 * ((total_vocab_size + 127) // 128)
        self.embeddings = nn.Embedding(padded_vocab_size, config.dim)
        self.register_buffer("offsets", self.make_offsets(), persistent=False)

    def make_offsets(self) -> torch.Tensor:
        offsets = []
        offset = 0
        for size in self.codebook_sizes:
            offsets.append(offset)
            offset += size
        return torch.tensor(offsets, dtype=torch.long)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        offsets = self.offsets.view(1, -1, 1)
        return self.embeddings(codes + offsets).sum(dim=1)


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding for flow matching timestep."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = torch.exp(
            -math.log(theta) * torch.arange(dim // 2).float() / (dim // 2)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = torch.einsum("bi, j -> bj", t, self.inv_freq)
        return torch.cat((emb.cos(), emb.sin()), dim=-1)


class BidirectionalAttention(nn.Module):
    """Full (non-causal) attention with GQA. No positional encoding."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        use_biases: bool = False,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.repeats = n_heads // n_kv_heads

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=use_biases)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=use_biases)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=use_biases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            bsz, seqlen = 1, x.shape[0]
        else:
            bsz, seqlen, _ = x.shape

        q = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        k = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # GQA expansion
        if self.repeats > 1:
            k = k.unsqueeze(3).expand(-1, -1, -1, self.repeats, -1).flatten(2, 3)
            v = v.unsqueeze(3).expand(-1, -1, -1, self.repeats, -1).flatten(2, 3)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scale and compute attention (bidirectional, no mask)
        scale = self.head_dim**-0.5
        attn = (q * scale) @ k.transpose(-2, -1)
        attn = attn.softmax(-1)
        y = attn @ v

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(y)


class AcousticFeedForward(nn.Module):
    """SwiGLU FFN for the acoustic transformer."""

    def __init__(self, dim: int, hidden_dim: int, use_biases: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=use_biases)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class AcousticTransformerBlock(nn.Module):
    def __init__(self, config: VoxtralTTSConfig, layer_id: int):
        super().__init__()
        self.attention = BidirectionalAttention(
            config.at_dim,
            config.at_n_heads,
            config.at_n_kv_heads,
            config.at_head_dim,
            config.at_use_biases,
        )
        self.feed_forward = AcousticFeedForward(
            config.at_dim, config.at_hidden_dim, config.at_use_biases
        )
        self.attention_norm = RMSNorm(config.at_dim, config.at_norm_eps)
        self.ffn_norm = RMSNorm(config.at_dim, config.at_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class FlowMatchingHead(nn.Module):
    """Generates audio codebook tokens from LLM hidden states via flow matching ODE.

    Per frame: produces 1 semantic code (argmax) + N acoustic codes (7-step Euler ODE).
    The predict_velocity method is exported separately for the C++ runner to call
    in a loop.
    """

    def __init__(self, config: VoxtralTTSConfig):
        super().__init__()
        self.config = config

        # Projections
        self.input_projection = nn.Linear(
            config.acoustic_dim, config.at_dim, bias=False
        )
        self.time_projection = nn.Linear(config.at_dim, config.at_dim, bias=False)
        self.llm_projection = nn.Linear(config.dim, config.at_dim, bias=False)

        # Semantic codebook head
        padded_semantic_size = 128 * (
            (config.semantic_codebook_size + N_SPECIAL_TOKENS + 127) // 128
        )
        self.semantic_codebook_output = nn.Linear(
            config.at_dim, padded_semantic_size, bias=config.at_use_biases
        )
        self.padded_semantic_size = padded_semantic_size

        # Acoustic codebook head (predicts velocity vector)
        self.acoustic_codebook_output = nn.Linear(
            config.at_dim, config.acoustic_dim, bias=False
        )

        # Transformer layers
        self.layers = nn.ModuleDict(
            {
                str(i): AcousticTransformerBlock(config, i)
                for i in range(config.at_n_layers)
            }
        )
        self.norm = RMSNorm(config.at_dim, config.at_norm_eps)

        # Time embedding
        self.time_embedding = TimeEmbedding(config.at_dim)

        # Pre-compute timestep table for export
        self.register_buffer(
            "_timesteps",
            torch.linspace(0, 1, config.n_decoding_steps + 1),
            persistent=False,
        )

    def forward_layers(self, h: torch.Tensor) -> torch.Tensor:
        for i in range(self.config.at_n_layers):
            h = self.layers[str(i)](h)
        return h

    def predict_velocity(
        self,
        x_t: torch.Tensor,
        t_idx: torch.Tensor,
        llm_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Single velocity prediction step for the flow matching ODE.

        Args:
            x_t: (B, acoustic_dim) current noisy state
            t_idx: (B,) timestep index into self._timesteps
            llm_hidden: (B, llm_dim) hidden state from LLM
        Returns:
            v_t: (B, acoustic_dim) predicted velocity
        """
        t = self._timesteps[t_idx].unsqueeze(-1)  # (B, 1)
        t_emb = self.time_embedding(t).to(llm_hidden.dtype)
        t_emb = self.time_projection(t_emb)
        llm_proj = self.llm_projection(llm_hidden)

        inp = self.input_projection(x_t.to(llm_hidden.dtype)).unsqueeze(1)
        t_tok = t_emb.unsqueeze(1)
        ctx_tok = llm_proj.unsqueeze(1)
        h = torch.cat([inp, t_tok, ctx_tok], dim=1)  # (B, 3, at_dim)

        h = self.forward_layers(h)
        h = self.norm(h)
        return self.acoustic_codebook_output(h[:, 0, :])

    def semantic_head(self, llm_hidden: torch.Tensor) -> torch.Tensor:
        """Predict semantic codebook token (greedy argmax).

        Args:
            llm_hidden: (B, llm_dim) hidden state from LLM
        Returns:
            code: (B,) semantic codebook index
        """
        logit = self.semantic_logits(llm_hidden)
        return logit.argmax(dim=-1)

    def semantic_logits(self, llm_hidden: torch.Tensor) -> torch.Tensor:
        """Raw masked logits for semantic code prediction."""
        logit = self.semantic_codebook_output(llm_hidden).float()
        logit[:, EMPTY_AUDIO_ID] = float("-inf")
        logit[:, (N_SPECIAL_TOKENS + self.config.semantic_codebook_size) :] = float(
            "-inf"
        )
        return logit

    def forward(self, llm_hidden: torch.Tensor) -> torch.Tensor:
        """Full forward: semantic code + flow matching ODE -> all codes.

        Used for eager validation. The C++ runner calls predict_velocity
        and semantic_head separately.

        Args:
            llm_hidden: (B, llm_dim)
        Returns:
            codes: (B, 1 + acoustic_dim) = (B, 37) per frame
        """
        B = llm_hidden.shape[0]
        semantic_code = self.semantic_head(llm_hidden).unsqueeze(1)  # (B, 1)

        should_decode = semantic_code.squeeze(1) != END_AUDIO_ID

        # Flow matching ODE
        x = torch.randn(B, self.config.acoustic_dim, device=llm_hidden.device)
        x = x.to(llm_hidden.dtype) * self.config.noise_scale

        timesteps = self._timesteps.to(llm_hidden.dtype)
        llm_zero = torch.zeros_like(llm_hidden)

        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            dt = timesteps[i + 1] - timesteps[i]

            t_emb = self.time_embedding(
                t.view(-1, 1).repeat(B, 1)
            ).to(llm_hidden.dtype)
            t_emb = self.time_projection(t_emb)

            # CFG: batch cond + uncond
            x_batched = torch.cat([x, x], dim=0)
            llm_batched = torch.cat([llm_hidden, llm_zero], dim=0)
            t_emb_batched = torch.cat([t_emb, t_emb], dim=0)
            llm_proj = self.llm_projection(llm_batched)

            inp = self.input_projection(x_batched.to(llm_hidden.dtype)).unsqueeze(1)
            t_tok = t_emb_batched.unsqueeze(1)
            ctx_tok = llm_proj.unsqueeze(1)
            h = torch.cat([inp, t_tok, ctx_tok], dim=1)

            h = self.forward_layers(h)
            h = self.norm(h)
            v_all = self.acoustic_codebook_output(h[:, 0, :])

            v_cond, v_uncond = v_all[:B], v_all[B:]
            v = self.config.cfg_alpha * v_cond + (1 - self.config.cfg_alpha) * v_uncond
            x = x + v * dt

        # Quantize
        x = torch.clamp(x, -1, 1)
        scaled = ((x + 1) / 2) * (self.config.acoustic_levels - 1)
        acoustic_codes = scaled.round().long()
        acoustic_codes[~should_decode] = EMPTY_AUDIO_ID
        acoustic_codes = acoustic_codes + N_SPECIAL_TOKENS

        return torch.cat([semantic_code, acoustic_codes], dim=1)


# ---------------------------------------------------------------------------
# Codec decoder components
# ---------------------------------------------------------------------------


def _pad1d(
    x: torch.Tensor,
    paddings: tuple[int, int],
    mode: str = "constant",
    value: float = 0.0,
) -> torch.Tensor:
    length = x.shape[-1]
    padding_left, padding_right = paddings
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    return F.pad(x, paddings, mode, value)


class CodecCausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "reflect",
        use_weight_norm: bool = True,
        use_bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=0, dilation=dilation, bias=use_bias,
        )
        if use_weight_norm:
            self.conv = torch.nn.utils.parametrizations.weight_norm(self.conv)
        self.pad_mode = pad_mode
        self._stride = stride
        self._effective_kernel_size = (kernel_size - 1) * dilation + 1
        self._padding_total = self._effective_kernel_size - self._stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_frames = (
            x.shape[-1] - self._effective_kernel_size + self._padding_total
        ) / self._stride + 1
        target_length = (
            (math.ceil(n_frames) - 1) * self._stride
            + (self._effective_kernel_size - self._padding_total)
        )
        extra_padding = target_length - x.shape[-1]
        x = _pad1d(x, (self._padding_total, extra_padding), mode=self.pad_mode)
        return self.conv(x)


class CodecCausalConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        use_weight_norm: bool = True,
        use_bias: bool = True,
        trim_ratio: float = 1.0,
    ):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, bias=use_bias,
        )
        if use_weight_norm:
            self.conv = torch.nn.utils.parametrizations.weight_norm(self.conv)
        self.trim_ratio = trim_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        total_padding = kernel_size - stride
        out = self.conv(x)
        right_padding = math.ceil(total_padding * self.trim_ratio)
        left_padding = total_padding - right_padding
        return out[..., left_padding : out.shape[-1] - right_padding]


def _get_alibi_slopes(n_heads: int) -> torch.Tensor:
    def _slopes_power_of_2(n: int) -> torch.Tensor:
        r = 2.0 ** (-8.0 / n)
        return torch.tensor([r ** (i + 1) for i in range(n)], dtype=torch.float32)

    if math.log2(n_heads).is_integer():
        return _slopes_power_of_2(n_heads)
    m = 2 ** math.floor(math.log2(n_heads))
    return torch.cat([_slopes_power_of_2(m), _slopes_power_of_2(2 * m)[::2][: n_heads - m]])


class CodecAttention(nn.Module):
    """Causal attention with ALiBi + sliding window for the codec decoder."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        sliding_window: int,
        qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
        use_biases: bool = False,
        causal: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.repeats = n_heads // n_kv_heads
        self.sliding_window = sliding_window
        self.causal = causal

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=use_biases)

        if qk_norm:
            self.q_norm = RMSNorm(n_heads * head_dim, qk_norm_eps)
            self.k_norm = RMSNorm(n_kv_heads * head_dim, qk_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        self.register_buffer(
            "alibi_slopes", _get_alibi_slopes(n_heads), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            bsz, seqlen = 1, x.shape[0]
        else:
            bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        if self.q_norm is not None:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # Transpose to (B, H, S, D)
        q = xq.transpose(1, 2)
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)

        # GQA expansion
        if self.repeats > 1:
            k = k.repeat_interleave(self.repeats, dim=1)
            v = v.repeat_interleave(self.repeats, dim=1)

        # Build ALiBi + causal + sliding window bias
        positions = torch.arange(seqlen, device=x.device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)
        alibi_slopes = self.alibi_slopes.to(dtype=x.dtype, device=x.device)
        attn_bias = alibi_slopes.view(self.n_heads, 1, 1) * rel_pos.unsqueeze(0).to(
            x.dtype
        )

        if self.causal:
            attn_bias = attn_bias.masked_fill(rel_pos.unsqueeze(0) > 0, float("-inf"))

        window_right = 0 if self.causal else self.sliding_window
        outside_window = (rel_pos < -self.sliding_window) | (rel_pos > window_right)
        attn_bias = attn_bias.masked_fill(outside_window.unsqueeze(0), float("-inf"))

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias.unsqueeze(0)
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(y)


class CodecFeedForward(nn.Module):
    """SwiGLU FFN for the codec."""

    def __init__(self, dim: int, hidden_dim: int, use_biases: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=use_biases)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CodecTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        hidden_dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        sliding_window: int,
        norm_eps: float,
        qk_norm: bool,
        qk_norm_eps: float,
        use_biases: bool,
        layer_scale: bool,
        causal: bool,
    ):
        super().__init__()
        self.attention = CodecAttention(
            dim, n_heads, n_kv_heads, head_dim, sliding_window,
            qk_norm, qk_norm_eps, use_biases, causal,
        )
        self.feed_forward = CodecFeedForward(dim, hidden_dim, use_biases)
        self.attention_norm = RMSNorm(dim, norm_eps)
        self.ffn_norm = RMSNorm(dim, norm_eps)

        self.use_layer_scale = layer_scale
        if layer_scale:
            if layer_id < 18:
                init_scale = 0.1
            elif layer_id <= 24:
                init_scale = 1e-5
            else:
                init_scale = 1e-6
            self.attention_scale = nn.Parameter(
                torch.full((dim,), init_scale, requires_grad=True)
            )
            self.ffn_scale = nn.Parameter(
                torch.full((dim,), init_scale, requires_grad=True)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.attention(self.attention_norm(x))
        if self.use_layer_scale:
            r = self.attention_scale * r
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        if self.use_layer_scale:
            r = self.ffn_scale * r
        return h + r


class CodecTransformer(nn.Module):
    """Stack of codec transformer blocks with specified sliding window."""

    def __init__(self, config: VoxtralTTSConfig, n_layers: int, sliding_window: int):
        super().__init__()
        self.layers = nn.ModuleDict()
        for i in range(n_layers):
            self.layers[str(i)] = CodecTransformerBlock(
                layer_id=i,
                dim=config.codec_dim,
                hidden_dim=config.codec_hidden_dim,
                n_heads=config.codec_n_heads,
                n_kv_heads=config.codec_n_kv_heads,
                head_dim=config.codec_head_dim,
                sliding_window=sliding_window,
                norm_eps=config.codec_norm_eps,
                qk_norm=True,
                qk_norm_eps=config.codec_qk_norm_eps,
                use_biases=config.codec_use_biases,
                layer_scale=config.codec_layer_scale,
                causal=config.codec_causal,
            )
        self.n_layers = n_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.n_layers):
            x = self.layers[str(i)](x)
        return x


# ---------------------------------------------------------------------------
# Codebook quantizer (decode only)
# ---------------------------------------------------------------------------


class SemanticCodebook(nn.Module):
    """Euclidean distance VQ codebook — decode is just an embedding lookup."""

    def __init__(self, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.register_buffer("cluster_usage", torch.ones(codebook_size))
        self.register_buffer("embedding_sum", torch.zeros(codebook_size, codebook_dim))

    @property
    def embedding(self) -> torch.Tensor:
        return self.embedding_sum / self.cluster_usage.clamp(min=1e-5)[:, None]

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (B, 1, T) -> (B, semantic_dim, T)"""
        codes = codes.squeeze(1)  # (B, T)
        quantized = F.embedding(codes, self.embedding)  # (B, T, D)
        return quantized.transpose(1, 2)  # (B, D, T)


class AcousticCodebook(nn.Module):
    """Finite Scalar Quantization — decode rescales integers to [-1, 1]."""

    def __init__(self, n_levels: int, dim: int):
        super().__init__()
        self.n_levels = n_levels
        self.dim = dim

    def decode(self, codes: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """codes: (B, dim, T) long -> (B, dim, T) float in [-1, 1]"""
        return ((codes.to(dtype) * 2) / (self.n_levels - 1) - 1)


class AudioCodebook(nn.Module):
    """Combined semantic + acoustic codebook for decode."""

    def __init__(self, config: VoxtralTTSConfig):
        super().__init__()
        self.semantic_codebook = SemanticCodebook(
            config.semantic_codebook_size, config.semantic_dim
        )
        self.acoustic_codebook = AcousticCodebook(config.acoustic_levels, config.acoustic_dim)
        self.semantic_dim = config.semantic_dim
        self.acoustic_dim = config.acoustic_dim

    def decode(self, codes: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """codes: (B, 1+acoustic_dim, T) -> (B, semantic_dim+acoustic_dim, T)"""
        semantic_codes = codes[:, :1, :]
        acoustic_codes = codes[:, 1:, :]
        sem_emb = self.semantic_codebook.decode(semantic_codes).to(dtype)
        aco_emb = self.acoustic_codebook.decode(acoustic_codes, dtype)
        return torch.cat([sem_emb, aco_emb], dim=1)


# ---------------------------------------------------------------------------
# Full codec decoder
# ---------------------------------------------------------------------------


class CodecDecoder(nn.Module):
    """Converts codebook tokens to waveform via VQ/FSQ decode + upsampling."""

    def __init__(self, config: VoxtralTTSConfig):
        super().__init__()
        self.config = config
        self.quantizer = AudioCodebook(config)
        latent_dim = config.semantic_dim + config.acoustic_dim

        decoder_blocks: list[nn.Module] = []
        # The encoder starts at codec_sliding_window and halves at each
        # downsample.  The decoder mirrors this: start at the most-compressed
        # window and double at each upsample.
        n_upsample = sum(
            1 for s in config.codec_decoder_convs_strides if s > 1
        )
        if config.codec_half_attn_window_upon_downsampling and n_upsample > 0:
            cur_window_size = config.codec_sliding_window // (2 ** n_upsample)
        else:
            cur_window_size = config.codec_sliding_window

        # First projection: latent_dim -> codec_dim
        decoder_blocks.append(
            CodecCausalConv1d(
                latent_dim,
                config.codec_dim,
                kernel_size=config.codec_decoder_convs_kernels[0],
                stride=config.codec_decoder_convs_strides[0],
                pad_mode="replicate",
                use_weight_norm=config.codec_conv_weight_norm,
                use_bias=False,
            )
        )
        if (
            config.codec_half_attn_window_upon_downsampling
            and config.codec_decoder_convs_strides[0] > 1
        ):
            cur_window_size *= 2

        for idx, n_layers in enumerate(config.codec_decoder_transformer_lengths):
            decoder_blocks.append(
                CodecTransformer(config, n_layers, cur_window_size)
            )
            if (idx + 1 < len(config.codec_decoder_transformer_lengths)):
                next_k = config.codec_decoder_convs_kernels[idx + 1]
                next_s = config.codec_decoder_convs_strides[idx + 1]
                if next_k != 1 or next_s != 1:
                    decoder_blocks.append(
                        CodecCausalConvTranspose1d(
                            config.codec_dim,
                            config.codec_dim,
                            kernel_size=next_k,
                            stride=next_s,
                            use_weight_norm=config.codec_conv_weight_norm,
                            use_bias=False,
                        )
                    )
                    if config.codec_half_attn_window_upon_downsampling and next_s > 1:
                        cur_window_size *= 2

        self.decoder_blocks = nn.ModuleList(decoder_blocks)

        self.output_proj = CodecCausalConv1d(
            config.codec_dim,
            config.codec_patch_size,
            kernel_size=7,
            use_weight_norm=config.codec_conv_weight_norm,
            use_bias=False,
        )

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode codebook tokens to waveform.

        Args:
            codes: (B, n_codebooks, T) integer codes
        Returns:
            waveform: (B, 1, T * downsample_factor)
        """
        # The generator emits all 37 codebooks in the shifted token space where
        # 0/1 are EMPTY/END special tokens and normal codes start at +2.
        # Match the reference tokenizer path by unshifting every codebook while
        # mapping specials/padding back to 0 for decode.
        codes_stripped = torch.where(
            codes >= N_SPECIAL_TOKENS,
            codes - N_SPECIAL_TOKENS,
            torch.zeros_like(codes),
        )

        latent = self.quantizer.decode(codes_stripped, dtype=codes.dtype if codes.is_floating_point() else torch.float32)

        x = latent  # (B, D, T) channels-first
        for block in self.decoder_blocks:
            if isinstance(block, CodecTransformer):
                x = x.transpose(1, 2)  # (B, D, T) -> (B, T, D)
                x = block(x)
                x = x.transpose(1, 2)  # (B, T, D) -> (B, D, T)
            else:
                x = block(x)  # Conv1d / ConvTranspose1d: stays (B, D, T)

        waveform = self.output_proj(x)  # (B, patch_size=240, T')
        B, P, T = waveform.shape
        # Audio samples are produced frame-by-frame: for each frame t we emit
        # P contiguous samples. Interleave time-outer / patch-inner to match
        # the reference C codec (`samples[t*P + h] = out_proj[h*T + t]`).
        return waveform.transpose(1, 2).reshape(B, 1, T * P)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class VoxtralTTSModel(nn.Module):
    def __init__(self, config: VoxtralTTSConfig):
        super().__init__()
        self.config = config
        self.decoder = MistralDecoder(config)
        self.audio_token_embedding = AudioTokenEmbedding(config)
        self.flow_head = FlowMatchingHead(config)
        self.codec_decoder = CodecDecoder(config)


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def _map_checkpoint_key(ckpt_key: str) -> str | None:
    """Map Mistral consolidated checkpoint key to model state_dict key.

    Checkpoint structure:
      - layers.N.* -> decoder.layers.N.*
      - norm.weight -> decoder.norm.weight
      - mm_audio_embeddings.tok_embeddings.weight -> decoder.tok_embeddings.weight
      - mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight
        -> audio_token_embedding.embeddings.weight
      - acoustic_transformer.* -> flow_head.*
      - audio_tokenizer.* -> codec_decoder.*
    """
    # LLM decoder layers
    if ckpt_key.startswith("layers."):
        return "decoder." + ckpt_key

    if ckpt_key == "norm.weight":
        return "decoder.norm.weight"

    # Token embeddings
    if ckpt_key == "mm_audio_embeddings.tok_embeddings.weight":
        return "decoder.tok_embeddings.weight"

    if ckpt_key == "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight":
        return "audio_token_embedding.embeddings.weight"

    # Flow matching head (acoustic transformer)
    if ckpt_key.startswith("acoustic_transformer."):
        suffix = ckpt_key[len("acoustic_transformer."):]
        return "flow_head." + suffix

    # Codec decoder
    if ckpt_key.startswith("audio_tokenizer."):
        suffix = ckpt_key[len("audio_tokenizer."):]
        return "codec_decoder." + suffix

    # Skip voice embeddings (loaded separately)
    if ckpt_key.startswith("mm_audio_embeddings.audio_codebook"):
        return None

    return None


def _fold_weight_norm(model: nn.Module) -> None:
    """Remove weight_norm parametrizations, fusing weight_v + weight_g into weight."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            if hasattr(module, "parametrizations"):
                torch.nn.utils.parametrize.remove_parametrizations(
                    module, "weight"
                )


def load_model(
    model_path: str,
    max_seq_len: int = 4096,
    dtype: torch.dtype = torch.float32,
    backend: str = "xnnpack",
) -> VoxtralTTSModel:
    """Load VoxtralTTSModel from a Mistral checkpoint.

    Uses meta-device construction + assign-based loading to minimize peak memory.
    """
    from safetensors import safe_open

    model_dir = Path(model_path)
    config = VoxtralTTSConfig.from_params_json(str(model_dir / "params.json"))
    config.max_seq_len = max_seq_len
    config.backend = backend

    print(
        f"Building model on meta device (dim={config.dim}, layers={config.n_layers}, "
        f"at_dim={config.at_dim}, at_layers={config.at_n_layers}, "
        f"codec_dim={config.codec_dim}, backend={backend})..."
    )
    with torch.device("meta"):
        model = VoxtralTTSModel(config)

    # Load weights
    ckpt_path = str(model_dir / "consolidated.safetensors")
    print(f"Loading weights from {ckpt_path}...")
    state_dict = {}
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        for ckpt_key in f.keys():
            model_key = _map_checkpoint_key(ckpt_key)
            if model_key is None:
                continue
            state_dict[model_key] = f.get_tensor(ckpt_key).to(dtype)

    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)

    # Materialize meta-device buffers (KV caches, RoPE, timesteps, etc.)
    for fqn, buf in list(model.named_buffers()):
        if buf.device.type == "meta":
            parts = fqn.rsplit(".", 1)
            parent = model.get_submodule(parts[0]) if len(parts) > 1 else model
            parent.register_buffer(
                parts[-1],
                torch.zeros(buf.shape, dtype=dtype, device="cpu"),
            )

    # Recompute RoPE
    dec_cos, dec_sin = precompute_freqs_cis(
        config.head_dim, max_seq_len, config.rope_theta
    )
    model.decoder.register_buffer("freqs_cos", dec_cos)
    model.decoder.register_buffer("freqs_sin", dec_sin)

    # Recompute audio-token embedding offsets
    model.audio_token_embedding.register_buffer(
        "offsets",
        model.audio_token_embedding.make_offsets(),
        persistent=False,
    )

    # Recompute flow-matching timestep embedding buffers
    model.flow_head.time_embedding.register_buffer(
        "inv_freq",
        torch.exp(
            -math.log(10000.0)
            * torch.arange(config.at_dim // 2, dtype=torch.float32)
            / (config.at_dim // 2)
        ),
        persistent=True,
    )

    # Recompute timesteps
    model.flow_head.register_buffer(
        "_timesteps",
        torch.linspace(0, 1, config.n_decoding_steps + 1),
    )

    # Recompute ALiBi slopes for codec attention
    for module in model.codec_decoder.modules():
        if isinstance(module, CodecAttention):
            slopes = _get_alibi_slopes(module.n_heads)
            module.register_buffer("alibi_slopes", slopes, persistent=False)

    # Recompute semantic codebook embedding
    sem = model.codec_decoder.quantizer.semantic_codebook
    if sem.embedding_sum.device.type != "meta":
        sem.register_buffer(
            "_embedding",
            sem.embedding_sum / sem.cluster_usage.clamp(min=1e-5)[:, None],
            persistent=False,
        )

    # Fold weight_norm in codec decoder
    _fold_weight_norm(model.codec_decoder)

    # Validate loading
    runtime_prefixes = (
        "decoder.freqs_",
        "audio_token_embedding.offsets",
        ".kv_cache.",
        "flow_head.time_embedding.inv_freq",
        "flow_head._timesteps",
        ".alibi_slopes",
        "._embedding",
    )
    actual_missing = set(missing)
    expected_missing = {
        k for k in actual_missing if any(p in k for p in runtime_prefixes)
    }
    extra_missing = actual_missing - expected_missing
    if extra_missing:
        print(f"  WARNING: {len(extra_missing)} unexpected missing keys")
        for k in sorted(extra_missing)[:20]:
            print(f"    {k}")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys")

    loaded = len(state_dict) - len(unexpected)
    print(
        f"  Loaded {loaded} tensors ({len(expected_missing)} runtime buffers OK, "
        f"{len(extra_missing)} unexpected missing)"
    )

    model.eval()
    return model

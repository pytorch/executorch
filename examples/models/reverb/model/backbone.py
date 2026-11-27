# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The Reverb backbone: a bidirectional Qwen3 decoder over text and codec frames.

Reverb fills a grid of masked codec tokens over several passes. The prompt
(text tokens, then the reference voice's codec frames) runs once and leaves a
key/value cache; every later pass runs only the target frames against it.
Batch size is 1 and nothing is padded, so attention needs no mask; the polish
passes add a constant bias towards the reference frames.

Module names follow the released checkpoint, so its state dict loads as is.
"""

import torch
import torch.nn.functional as F
from torch import nn

from .config import CODEBOOKS, CODES, MAIN_CODEBOOKS, MASK, ReverbConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


class FactorizedEmbedding(nn.Module):
    """The text embedding as a rank-r product: a [vocab, r] table, then r -> hidden."""

    def __init__(self, vocab: int, hidden: int, rank: int) -> None:
        super().__init__()
        self.table = nn.Embedding(vocab, rank)
        self.project = nn.Linear(rank, hidden, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.project(self.table(ids))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Attention(nn.Module):
    def __init__(self, config: ReverbConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        hidden, q_dim = config.hidden_size, config.num_heads * config.head_dim
        kv_dim = config.num_kv_heads * config.head_dim
        self.q_proj = nn.Linear(hidden, q_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, hidden, bias=False)
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        repeats = self.num_heads // self.num_kv_heads
        if repeats == 1:
            return x
        length = x.shape[2]
        x = x[:, :, None].expand(1, self.num_kv_heads, repeats, length, self.head_dim)
        return x.reshape(1, self.num_heads, length, self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key: torch.Tensor | None = None,
        past_value: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """[1, T, hidden] -> output [1, T, hidden], this pass's keys and values [1, kv, T, D]."""
        length = x.shape[1]
        q = self.q_norm(self.q_proj(x).view(1, length, self.num_heads, self.head_dim))
        k = self.k_norm(
            self.k_proj(x).view(1, length, self.num_kv_heads, self.head_dim)
        )
        v = self.v_proj(x).view(1, length, self.num_kv_heads, self.head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        keys, values = k, v
        if past_key is not None and past_value is not None:
            keys = torch.cat([past_key, k], dim=2)
            values = torch.cat([past_value, v], dim=2)
        out = F.scaled_dot_product_attention(
            q, self._repeat_kv(keys), self._repeat_kv(values), attn_mask=bias
        )
        out = out.transpose(1, 2).reshape(1, length, self.num_heads * self.head_dim)
        return self.o_proj(out), k, v


class MLP(nn.Module):
    def __init__(self, config: ReverbConfig) -> None:
        super().__init__()
        hidden, inner = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(hidden, inner, bias=False)
        self.up_proj = nn.Linear(hidden, inner, bias=False)
        self.down_proj = nn.Linear(inner, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: ReverbConfig) -> None:
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x, cos, sin, past_key=None, past_value=None, bias=None):
        attended, k, v = self.self_attn(
            self.input_layernorm(x), cos, sin, past_key, past_value, bias
        )
        x = x + attended
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, k, v


class LanguageModel(nn.Module):
    """The Qwen3 decoder, named as in the checkpoint's `llm.*` tensors."""

    def __init__(self, config: ReverbConfig) -> None:
        super().__init__()
        if config.embedding_rank:
            self.embed_tokens = FactorizedEmbedding(
                config.vocab_size, config.hidden_size, config.embedding_rank
            )
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(config) for _ in range(config.num_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)


class AudioHeads(nn.Module):
    """The audio output heads, split by the codebooks each pass reads.

    The checkpoint stores one [8 * 1025, hidden] matrix; codebook c owns rows
    c * 1025 + code. The main passes read codebooks 0-1, the first polish pass
    2-7 and the second 4-7, so the matrix is kept as three linears.
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.main = nn.Linear(hidden, MAIN_CODEBOOKS * CODES, bias=False)
        self.middle = nn.Linear(hidden, 2 * CODES, bias=False)
        self.late = nn.Linear(hidden, 4 * CODES, bias=False)

    def load_full(self, weight: torch.Tensor) -> None:
        rows = torch.split(weight, [2 * CODES, 2 * CODES, 4 * CODES])
        for linear, part in zip((self.main, self.middle, self.late), rows):
            linear.weight.data.copy_(part)


def sum_codebooks(embedded: torch.Tensor) -> torch.Tensor:
    """[8, T, C] -> [T, C] as explicit adds, which every delegate lowers."""
    total = embedded[0]
    for codebook in range(1, embedded.shape[0]):
        total = total + embedded[codebook]
    return total


def _log_probs(logits: torch.Tensor, codebooks: int) -> torch.Tensor:
    """[T, codebooks * 1025] logits -> [codebooks, T, 1024] fp32 log-probabilities."""
    length = logits.shape[0]
    logits = logits.float().view(length, codebooks, CODES).transpose(0, 1)
    return logits.log_softmax(-1)[..., :MASK]  # a masked slot is never a prediction


class ReverbBackbone(nn.Module):
    def __init__(self, config: ReverbConfig) -> None:
        super().__init__()
        self.config = config
        self.llm = LanguageModel(config)
        self.audio_embeddings = nn.Embedding(CODEBOOKS * CODES, config.hidden_size)
        self.audio_heads = AudioHeads(config.hidden_size)
        self.register_buffer(
            "inv_freq", torch.empty(config.head_dim // 2), persistent=False
        )
        self.register_buffer(
            "codebook_offsets",
            torch.empty(CODEBOOKS, 1, dtype=torch.int64),
            persistent=False,
        )
        self.init_buffers()

    def _apply(self, fn, recurse=True):
        # The RoPE frequencies stay fp32 whatever dtype the weights take.
        super()._apply(fn, recurse)
        if self.inv_freq.dtype != torch.float32:
            self.inv_freq = self.inv_freq.float()
            self.init_buffers()
        return self

    def init_buffers(self) -> None:
        """Fills the derived buffers; call again after `to_empty`."""
        dim = self.config.head_dim
        exponents = torch.arange(0, dim, 2, dtype=torch.float32) / dim
        self.inv_freq.copy_(1.0 / self.config.rope_theta**exponents)
        self.codebook_offsets.copy_(CODES * torch.arange(CODEBOOKS)[:, None])

    def _rope(
        self, start: int | torch.SymInt, length: int | torch.SymInt, dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(length, dtype=torch.float32) + start
        freqs = positions[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def _frames(self, codes: torch.Tensor) -> torch.Tensor:
        """[8, T] codes -> [T, hidden]: the sum of a frame's codebook embeddings."""
        return sum_codebooks(self.audio_embeddings(codes + self.codebook_offsets))

    def prefill(self, ids: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """Prompt ids [L] and reference codes [8, R] -> cache [2, layers, kv, L + R, D]."""
        x = torch.cat([self.llm.embed_tokens(ids), self._frames(codes)])[None]
        cos, sin = self._rope(0, x.shape[1], x.dtype)
        keys, values = [], []
        for layer in self.llm.layers:
            x, k, v = layer(x, cos, sin)
            keys.append(k[0])
            values.append(v[0])
        return torch.stack([torch.stack(keys), torch.stack(values)])

    def target_hidden(
        self,
        tokens: torch.Tensor,
        cache: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One pass over the target: tokens [8, T] -> normed hidden states [T, hidden]."""
        x = self._frames(tokens)[None]
        cos, sin = self._rope(cache.shape[3], x.shape[1], x.dtype)
        for i, layer in enumerate(self.llm.layers):
            x, _, _ = layer(x, cos, sin, cache[0, i][None], cache[1, i][None], bias)
        return self.llm.norm(x)[0]

    def reference_bias(
        self,
        prefix: int | torch.SymInt,
        length: int | torch.SymInt,
        reference_frames: torch.Tensor,
        strength: float,
        dtype,
    ) -> torch.Tensor:
        """An additive [1, 1, 1, prefix + length] attention bias on the reference frames' keys."""
        keys = torch.arange(prefix + length)
        on_reference = (keys >= prefix - reference_frames) & (keys < prefix)
        return (on_reference.to(dtype) * strength).view(1, 1, 1, -1)

    def step(
        self, tokens: torch.Tensor, cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One main pass: confidence, prediction and held log-probability, each [2, T]."""
        lp = _log_probs(
            self.audio_heads.main(self.target_hidden(tokens, cache)), MAIN_CODEBOOKS
        )
        confidence, prediction = lp.amax(-1), lp.argmax(-1)
        # The log-probability of each held code, as a masked sum rather than a
        # gather and squeeze, which not every delegate resizes correctly.
        held = tokens[:MAIN_CODEBOOKS].clamp(max=MASK - 1).unsqueeze(-1)
        codes = torch.arange(MASK, dtype=held.dtype)
        likely = torch.where(codes == held, lp, torch.zeros_like(lp)).sum(-1)
        return confidence, prediction, likely

    def polish(
        self,
        tokens: torch.Tensor,
        reference_frames: torch.Tensor,
        cache: torch.Tensor,
        bias_strength: float,
        guidance: float,
    ) -> torch.Tensor:
        """Both polish passes: tokens [8, T] with codebooks 0-1 decided -> codebooks 2-7 [6, T].

        Codebooks 2-7 of `tokens` are ignored: the first pass sees them masked,
        the second sees its own codebooks 2-3 and 4-7 masked.
        """
        bias = self.reference_bias(
            cache.shape[3],
            tokens.shape[1],
            reference_frames,
            bias_strength,
            self.audio_embeddings.weight.dtype,
        )
        main = tokens[:MAIN_CODEBOOKS]
        masked = torch.full_like(tokens[MAIN_CODEBOOKS:], MASK)
        hidden = self.target_hidden(torch.cat([main, masked]), cache, bias)
        first = _log_probs(
            torch.cat(
                [self.audio_heads.middle(hidden), self.audio_heads.late(hidden)], -1
            ),
            6,
        )
        middle = first[:2].argmax(-1)
        tokens = torch.cat([main, middle, masked[2:]])
        second = _log_probs(
            self.audio_heads.late(self.target_hidden(tokens, cache, bias)), 4
        )
        late = (second + guidance * (second - first[2:])).argmax(-1)
        return torch.cat([middle, late])

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Static Liquid codec decoder with exact overlap-window synthesis."""

import math
from collections.abc import Iterator
from typing import Dict, List

import torch
from torch import nn
from torch.nn import functional as F


def codec_window_plan(
    total_frames: int, window: int, left: int, right: int
) -> Iterator[tuple[int, int, int, int]]:
    """Yield (input start, valid input frames, output offset, output frames)."""
    if total_frames < 0 or min(left, right) < 0 or window <= left + right:
        raise ValueError("Codec window must leave room for context and new frames")
    emitted = 0
    while emitted < total_frames:
        start = max(0, emitted - left)
        valid = min(window, total_frames - start)
        end = total_frames if total_frames - start <= window else start + window - right
        yield start, valid, emitted - start, end - emitted
        emitted = end


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class _Attention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, head_dim, eps):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, head_dim
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, n_heads, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=eps)

    def forward(self, x, cos, sin, mask):
        n = x.shape[0]
        q = self.q_norm(self.q_proj(x).view(n, self.n_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).view(n, self.n_kv_heads, self.head_dim))
        v = self.v_proj(x).view(n, self.n_kv_heads, self.head_dim)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        q, k, v = (t.transpose(0, 1) for t in (q, k, v))
        rep = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(rep, dim=0)
        v = v.repeat_interleave(rep, dim=0)
        attn = torch.softmax(
            (q @ k.transpose(1, 2)) * self.head_dim**-0.5 + mask, dim=-1
        )
        y = (attn @ v).transpose(0, 1) * torch.sigmoid(self.gate_proj(x)).unsqueeze(-1)
        return self.o_proj(y.reshape(n, -1))


class _ShortConv(nn.Module):
    def __init__(self, dim, kernel):
        super().__init__()
        self.kernel = kernel
        self.in_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.conv_weight = nn.Parameter(torch.zeros(kernel, dim))
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        n = x.shape[0]
        b_gate, c_gate, xs = self.in_proj(x).chunk(3, dim=-1)
        window = F.pad(b_gate * xs, (0, 0, self.kernel - 1, 0))
        y = window[0:n] * self.conv_weight[0]
        for k in range(1, self.kernel):
            y = y + window[k : k + n] * self.conv_weight[k]
        return self.out_proj(c_gate * y)


class _Layer(nn.Module):
    def __init__(self, layer_type, dim, ffn, n_heads, n_kv, head_dim, kernel, eps):
        super().__init__()
        self.is_attention = layer_type != "conv"
        if self.is_attention:
            self.self_attn = _Attention(dim, n_heads, n_kv, head_dim, eps)
        else:
            self.conv = _ShortConv(dim, kernel)
        self.feed_forward = nn.ModuleDict(
            {
                "w1": nn.Linear(dim, ffn, bias=False),
                "w2": nn.Linear(ffn, dim, bias=False),
                "w3": nn.Linear(dim, ffn, bias=False),
            }
        )
        for name in (
            "operator_norm",
            "post_operator_norm",
            "ffn_norm",
            "post_ffn_norm",
        ):
            setattr(self, name, nn.RMSNorm(dim, eps=eps))

    def forward(self, x, cos, sin, mask):
        h = self.operator_norm(x)
        h = self.self_attn(h, cos, sin, mask) if self.is_attention else self.conv(h)
        x = x + self.post_operator_norm(h)
        f, y = self.feed_forward, self.ffn_norm(x)
        return x + self.post_ffn_norm(f["w2"](F.silu(f["w1"](y)) * f["w3"](y)))


class StaticCodecDecoder(nn.Module):
    """codes [n_q, frames], valid_frames [1] -> fixed-size float32 audio.

    Only the first valid_frames * frame_samples output samples are meaningful.
    valid_frames excludes padding from BOTH overlap-add and its envelope.
    """

    def __init__(
        self,
        codec_cfg: Dict,
        frames: int,
        window: torch.Tensor,
        *,
        hidden_ffn: int = 2304,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        head_dim: int = 64,
    ):
        super().__init__()
        levels: List[int] = codec_cfg["fsq_levels"]
        n_q = codec_cfg["n_q"]
        self.frames, self.n_q = frames, n_q
        self.stack = codec_cfg["stft_frame_stack"]
        n_fft, hop = codec_cfg["stft_n_fft"], codec_cfg["stft_hop_length"]
        win = codec_cfg["stft_win_length"]
        self.decoder_window = codec_cfg["decoder_window"]
        if (
            min(frames, n_q, self.stack, hop, self.decoder_window) <= 0
            or win != n_fft
            or n_fft % hop != 0
            or tuple(window.shape) != (win,)
            or min(levels) < 2
        ):
            raise ValueError("Unsupported codec geometry")
        self.hop, self.n_fft = hop, n_fft
        self.frame_samples = self.stack * hop
        steps = frames * self.stack
        self.steps = steps
        self.trim = (win - hop) // 2

        basis = [math.prod(levels[:i]) for i in range(len(levels))]
        size = math.prod(levels)
        self.audio_eos_code, self.codebook_size = size, size + 1
        idx = torch.arange(size + 1)[:, None]
        lv, bs = torch.tensor(levels), torch.tensor(basis)
        half = lv // 2
        table = ((idx // bs) % lv - half).float() / half
        table[size] = 0
        self.register_buffer("fsq_table", table, persistent=False)

        dim = codec_cfg["encoder_dim"]
        self.project_out = nn.Linear(n_q * len(levels), codec_cfg["quantizer_dim"])
        self.from_quantizer = nn.Linear(codec_cfg["quantizer_dim"], dim)

        layer_types = ["attn" if i in (2, 4, 6) else "conv" for i in range(8)]
        self.layers = nn.ModuleList(
            [
                _Layer(t, dim, hidden_ffn, n_heads, n_kv_heads, head_dim, 7, 1e-5)
                for t in layer_types
            ]
        )
        self.history_steps = sum(
            self.decoder_window - 1 if layer.is_attention else layer.conv.kernel - 1
            for layer in self.layers
        )
        # At sample x after trimming, contributing spectra span
        # floor((x + trim - n_fft) / hop) + 1 through floor((x + trim) / hop).
        synthesis_left = math.ceil((n_fft - self.trim) / hop) - 1
        synthesis_right = math.ceil(self.trim / hop)
        self.left_context = math.ceil(
            (self.history_steps + synthesis_left) / self.stack
        )
        self.right_context = math.ceil(synthesis_right / self.stack)
        self.embedding_norm = nn.RMSNorm(dim, eps=1e-5)
        self.lin = nn.Linear(dim, 2 * (n_fft // 2 + 1))

        inv_freq = 1.0 / (1e6 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs = torch.outer(torch.arange(steps).float(), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)[:, None, :]
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)
        q, k = torch.arange(steps)[:, None], torch.arange(steps)[None, :]
        allowed = (k <= q) & (q - k < self.decoder_window)
        self.register_buffer(
            "mask", torch.where(allowed, 0.0, -1e4)[None], persistent=False
        )
        self.register_buffer(
            "step_index", torch.arange(steps, dtype=torch.int32), persistent=False
        )

        bins = n_fft // 2 + 1
        kk = torch.arange(bins).float()[:, None]
        nn_ = torch.arange(n_fft).float()[None, :]
        weight = torch.full((bins, 1), 2.0)
        weight[0] = weight[-1] = 1.0
        ang = 2 * math.pi * kk * nn_ / n_fft
        self.register_buffer(
            "irfft_cos", weight * torch.cos(ang) / n_fft, persistent=False
        )
        self.register_buffer(
            "irfft_sin", -weight * torch.sin(ang) / n_fft, persistent=False
        )
        self.register_buffer("window", window.float(), persistent=False)
        self.chunks = n_fft // hop
        self.register_buffer(
            "window_square",
            window.float().square().view(self.chunks, hop),
            persistent=False,
        )

    def get_metadata(self) -> dict[str, int]:
        return {
            "get_speech_contract_version": 1,
            "get_num_codebooks": self.n_q,
            "get_codebook_size": self.codebook_size,
            "get_audio_eos_code": self.audio_eos_code,
            "get_codec_frames": self.frames,
            "get_codec_left_context": self.left_context,
            "get_codec_right_context": self.right_context,
            "get_frame_samples": self.frame_samples,
        }

    def forward(
        self, codes: torch.Tensor, valid_frames: torch.Tensor | None = None
    ) -> torch.Tensor:
        latents = self.fsq_table[codes.transpose(0, 1)].reshape(self.frames, -1)
        x = self.from_quantizer(self.project_out(latents))
        x = x.repeat_interleave(self.stack, dim=0)
        for layer in self.layers:
            x = layer(x, self.cos, self.sin, self.mask)
        x = self.lin(self.embedding_norm(x))
        log_mag, angle = x.chunk(2, dim=-1)
        mag = torch.exp(log_mag)
        frames = (mag * torch.cos(angle)) @ self.irfft_cos + (
            mag * torch.sin(angle)
        ) @ self.irfft_sin
        frames = (frames * self.window).view(self.steps, self.chunks, self.hop)
        valid = (
            torch.ones(self.steps, dtype=torch.bool, device=codes.device)
            if valid_frames is None
            else self.step_index < valid_frames.reshape(1) * self.stack
        )
        # Padding must not synthesize future spectra at the true utterance end.
        frames = torch.where(valid[:, None, None], frames, 0.0)
        envelope = valid[:, None, None].to(frames.dtype) * self.window_square[None]
        audio, env = None, None
        for j in range(self.chunks):
            padding = (0, 0, j, self.chunks - 1 - j)
            term = F.pad(frames[:, j], padding)
            env_term = F.pad(envelope[:, j], padding)
            audio = term if audio is None else audio + term
            env = env_term if env is None else env + env_term
        audio = audio.reshape(-1)[self.trim : self.trim + self.steps * self.hop]
        env = env.reshape(-1)[self.trim : self.trim + self.steps * self.hop]
        return audio / env.clamp_min(1e-8)


def codec_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {
        "project_out.weight": sd["quantizer.project_out.weight"],
        "project_out.bias": sd["quantizer.project_out.bias"],
        "from_quantizer.weight": sd["from_quantizer.weight"],
        "from_quantizer.bias": sd["from_quantizer.bias"],
        "lin.weight": sd["decoder.lin.weight"],
        "lin.bias": sd["decoder.lin.bias"],
        "embedding_norm.weight": sd["decoder.lfm.embedding_norm.weight"],
    }
    for k, v in sd.items():
        if k.startswith("decoder.lfm.layers."):
            name = k[len("decoder.lfm.") :]
            if name.endswith("conv.conv.weight"):
                name = name.replace("conv.conv.weight", "conv.conv_weight")
                v = v.squeeze(1).transpose(0, 1)
            out[name] = v
    return {k: v.float() for k, v in out.items()}

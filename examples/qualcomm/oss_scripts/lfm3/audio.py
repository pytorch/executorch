# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Static-shape LFM3-Omni speech encoder (downsampler + LFM2 stack + projector).

Input [1, n_mels + 1, max_frames]: the log-mel features plus a 0/1 row marking the
valid frames. Padding is masked exactly like HF; the LFM2 stack is causal, so
outputs at valid positions do not depend on the padding.
"""

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class _Downsampler(nn.Module):
    def __init__(self, n_mels: int, channels: int, hidden: int):
        super().__init__()
        conv = {"kernel_size": 3, "stride": 2, "padding": 1}
        self.conv_in = nn.Conv2d(1, channels, **conv)
        self.conv_mid = nn.Conv2d(channels, channels, groups=channels, **conv)
        self.lin_mid = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_end = nn.Conv2d(channels, channels, groups=channels, **conv)
        self.lin_end = nn.Conv2d(channels, channels, kernel_size=1)
        self.lin_out = nn.Conv1d(n_mels * channels // 8, hidden, kernel_size=1)

    def forward(self, mel, valid):
        # After k stride-2 convs, position i is valid iff frame i * 2**k is.
        def mask(h, stride):
            return h * valid[::stride]

        h = mask(mel.unsqueeze(1), 1)
        h = F.silu(self.conv_in(h))
        h = mask(h, 2)
        h = F.silu(self.lin_mid(self.conv_mid(h)))
        h = mask(h, 4)
        h = F.silu(self.lin_end(self.conv_end(h)))
        return self.lin_out(h.flatten(1, 2))  # [1, hidden, tokens]


class _Attention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, eps):
        super().__init__()
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        self.q_layernorm = nn.RMSNorm(self.head_dim, eps=eps)
        self.k_layernorm = nn.RMSNorm(self.head_dim, eps=eps)

    def forward(self, x, cos, sin, mask):
        n = x.shape[0]
        q = self.q_layernorm(self.q_proj(x).view(n, self.n_heads, self.head_dim))
        k = self.k_layernorm(self.k_proj(x).view(n, self.n_kv_heads, self.head_dim))
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
        return self.out_proj((attn @ v).transpose(0, 1).reshape(n, -1))


class _ShortConv(nn.Module):
    """LFM2 gated causal conv (width k) over a whole clip."""

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
    def __init__(self, cfg, layer_type, hidden_ffn):
        super().__init__()
        dim, eps = cfg["hidden_size"], cfg["norm_eps"]
        self.is_attention = layer_type == "full_attention"
        if self.is_attention:
            self.self_attn = _Attention(
                dim, cfg["num_attention_heads"], cfg["num_key_value_heads"], eps
            )
        else:
            self.conv = _ShortConv(dim, cfg["conv_L_cache"])
        self.feed_forward = nn.ModuleDict(
            {
                "w1": nn.Linear(dim, hidden_ffn, bias=False),
                "w2": nn.Linear(hidden_ffn, dim, bias=False),
                "w3": nn.Linear(dim, hidden_ffn, bias=False),
            }
        )
        self.operator_norm = nn.RMSNorm(dim, eps=eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=eps)

    def forward(self, x, cos, sin, mask):
        h = self.operator_norm(x)
        h = self.self_attn(h, cos, sin, mask) if self.is_attention else self.conv(h)
        x = x + h
        f = self.feed_forward
        y = self.ffn_norm(x)
        return x + f["w2"](F.silu(f["w1"](y)) * f["w3"](y))


class StaticAudioEncoder(nn.Module):
    """features [1, n_mels + 1, max_frames] -> audio embeds [1, max_frames / 8,
    text_hidden]; rows past ceil(valid_frames / 8) are padding."""

    def __init__(
        self, enc_cfg: Dict, text_hidden: int, max_frames: int, hidden_ffn: int
    ):
        super().__init__()
        lfm = enc_cfg["text_config"]
        dim = lfm["hidden_size"]
        self.max_tokens = max_frames // 8
        self.n_mels = enc_cfg["num_mel_bins"]
        self.downsample = _Downsampler(
            enc_cfg["num_mel_bins"], enc_cfg["conv_channels"], enc_cfg["hidden_size"]
        )
        self.layers = nn.ModuleList(
            [_Layer(lfm, t, hidden_ffn) for t in lfm["layer_types"]]
        )
        self.embedding_norm = nn.RMSNorm(dim, eps=lfm["norm_eps"])

        head_dim = dim // lfm["num_attention_heads"]
        theta = lfm["rope_parameters"]["rope_theta"]
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs = torch.outer(torch.arange(self.max_tokens).float(), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)[:, None, :]
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)
        q = torch.arange(self.max_tokens)[:, None]
        k = torch.arange(self.max_tokens)[None, :]
        window = enc_cfg["sliding_window"]
        allowed = (k <= q) & (q - k < window)
        self.register_buffer(
            "mask", torch.where(allowed, 0.0, -1e4)[None], persistent=False
        )

        self.proj_layer_norm = nn.LayerNorm(dim)
        self.proj_linear_1 = nn.Linear(dim, text_hidden)
        self.proj_linear_2 = nn.Linear(text_hidden, text_hidden)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        mel, valid = features[:, : self.n_mels], features[0, self.n_mels]
        x = self.downsample(mel, valid)[0].transpose(0, 1)  # [tokens, dim]
        for layer in self.layers:
            x = layer(x, self.cos, self.sin, self.mask)
        x = self.embedding_norm(x)
        x = self.proj_linear_2(F.gelu(self.proj_linear_1(self.proj_layer_norm(x))))
        return x.unsqueeze(0)


def audio_state_dict(hf_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in hf_sd.items():
        if k.startswith("model.audio_encoder.downsample."):
            out[k[len("model.audio_encoder.") :]] = v.float()
        elif k.startswith("model.audio_encoder.lfm."):
            name = k[len("model.audio_encoder.lfm.") :]
            if name.endswith("conv.conv.weight"):
                name = name.replace("conv.conv.weight", "conv.conv_weight")
                v = v.squeeze(1).transpose(0, 1)
            out[name] = v.float()
        elif k.startswith("model.audio_projector."):
            out["proj_" + k[len("model.audio_projector.") :]] = v.float()
    return out


class Lfm3OmniAudioEncoder(nn.Module):
    """StaticAudioEncoder with weights from the HF checkpoint, for the QNN
    multimodal export flow (no HF modeling code needed)."""

    def __init__(self, repo_id: str, max_frames: int):
        super().__init__()
        import json

        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file

        snap = snapshot_download(
            repo_id, allow_patterns=["config.json", "model.safetensors"]
        )
        cfg = json.load(open(f"{snap}/config.json"))
        sd = audio_state_dict(load_file(f"{snap}/model.safetensors"))
        self.encoder = StaticAudioEncoder(
            cfg["audio_encoder_config"],
            cfg["text_config"]["hidden_size"],
            max_frames,
            sd["layers.0.feed_forward.w1.weight"].shape[0],
        )
        self.encoder.load_state_dict(sd, strict=True)
        self.n_rows = cfg["audio_encoder_config"]["num_mel_bins"] + 1
        self.max_frames = max_frames

    def forward(self, features):
        return self.encoder(features)

    def get_example_inputs(self):
        return (torch.zeros(1, self.n_rows, self.max_frames),)

    def get_metadata(self):
        return {"get_n_layers": len(self.encoder.layers)}

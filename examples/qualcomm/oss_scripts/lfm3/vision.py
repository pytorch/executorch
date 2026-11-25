# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Static-shape LFM3 vision tower + multimodal projector for QNN export.

One image at a fixed patch grid. HF's per-frame bookkeeping (cu_seqlens, frame
shapes, 6-D merge-block reshapes) becomes constants: RoPE tables and a token
permutation between merge-block order and raster order.
"""

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


class _MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class _Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, eps: float):
        super().__init__()
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        self.q_layernorm = nn.RMSNorm(self.head_dim, eps=eps)
        self.k_layernorm = nn.RMSNorm(self.head_dim, eps=eps)

    def forward(self, x, cos, sin):
        n = x.shape[0]
        q = self.q_layernorm(self.q_proj(x).view(n, self.n_heads, self.head_dim))
        k = self.k_layernorm(self.k_proj(x).view(n, self.n_kv_heads, self.head_dim))
        v = self.v_proj(x).view(n, self.n_kv_heads, self.head_dim)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        q, k, v = (t.transpose(0, 1) for t in (q, k, v))  # [heads, n, head_dim]
        rep = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(rep, dim=0)
        v = v.repeat_interleave(rep, dim=0)
        attn = torch.softmax((q @ k.transpose(1, 2)) * self.head_dim**-0.5, dim=-1)
        y = (attn @ v).transpose(0, 1).reshape(n, -1)
        return self.out_proj(y)


class _Conv(nn.Module):
    """Gated 3x3 depthwise conv over the patch grid (raster order)."""

    def __init__(self, dim: int, kernel: int):
        super().__init__()
        self.in_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.spatial_conv = nn.Conv2d(
            dim, dim, kernel, padding=kernel // 2, groups=dim, bias=False
        )
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, height, width):
        b_gate, c_gate, xs = self.in_proj(x).chunk(3, dim=-1)
        gated = (b_gate * xs)[self.to_raster]
        # Depthwise conv as shifted multiply-adds on a channels-last grid: no
        # NCHW permutes for the QNN layout pass to misread.
        k = self.spatial_conv.kernel_size[0]
        pad = k // 2
        grid = F.pad(gated.view(height, width, -1), (0, 0, pad, pad, pad, pad))
        weight = self.spatial_conv.weight[:, 0]  # [dim, k, k]
        conv = None
        for dy in range(k):
            for dx in range(k):
                term = grid[dy : dy + height, dx : dx + width] * weight[:, dy, dx]
                conv = term if conv is None else conv + term
        conv = conv.reshape(height * width, -1)
        return self.out_proj(c_gate * conv[self.from_raster])


class _Layer(nn.Module):
    def __init__(self, cfg: Dict, layer_type: str):
        super().__init__()
        dim, eps = cfg["hidden_size"], cfg["norm_eps"]
        self.is_attention = layer_type == "full_attention"
        if self.is_attention:
            self.self_attn = _Attention(
                dim, cfg["num_attention_heads"], cfg["num_key_value_heads"], eps
            )
        else:
            self.conv = _Conv(dim, cfg["conv_kernel_size"])
        self.feed_forward = _MLP(dim, cfg["intermediate_size"])
        self.operator_norm = nn.RMSNorm(dim, eps=eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=eps)

    def forward(self, x, cos, sin, height, width):
        h = self.operator_norm(x)
        if self.is_attention:
            h = self.self_attn(h, cos, sin)
        else:
            h = self.conv(h, height, width)
        x = x + h
        return x + self.feed_forward(self.ffn_norm(x))


class StaticVisionEncoder(nn.Module):
    """pixel_values [num_patches, C*T*P*P] (merge-block order) -> image embeds
    [num_patches / merge**2, text_hidden]."""

    def __init__(
        self,
        vision_cfg: Dict,
        text_hidden: int,
        projector_hidden: int,
        downsample: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        height: int,
        width: int,
    ):
        super().__init__()
        dim = vision_cfg["hidden_size"]
        patch_dim = (
            vision_cfg["in_channels"]
            * vision_cfg["temporal_patch_size"]
            * vision_cfg["patch_size"] ** 2
        )
        merge = vision_cfg["spatial_merge_size"]
        self.height, self.width = height, width
        self.patchifier = nn.Linear(patch_dim, dim, bias=False)
        self.embedding_norm = nn.RMSNorm(dim, eps=vision_cfg["norm_eps"])
        self.layers = nn.ModuleList(
            [_Layer(vision_cfg, t) for t in vision_cfg["layer_types"]]
        )
        # [n, 1, head_dim] so they broadcast over heads.
        self.register_buffer("cos", cos[:, None, :].float(), persistent=False)
        self.register_buffer("sin", sin[:, None, :].float(), persistent=False)

        # Token n in merge-block order sits at raster position perm[n].
        hb, wb, mh, mw = torch.meshgrid(
            torch.arange(height // merge),
            torch.arange(width // merge),
            torch.arange(merge),
            torch.arange(merge),
            indexing="ij",
        )
        raster = ((hb * merge + mh) * width + (wb * merge + mw)).reshape(-1)
        # Per-layer copies: a constant shared by several layers ends up shared by
        # several QNN partitions, which lowering rejects.
        for layer in self.layers:
            if not layer.is_attention:
                layer.conv.register_buffer(
                    "from_raster", raster.clone(), persistent=False
                )
                layer.conv.register_buffer(
                    "to_raster", torch.argsort(raster), persistent=False
                )

        self.proj_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.proj_linear_1 = nn.Linear(dim * downsample**2, projector_hidden)
        self.proj_linear_2 = nn.Linear(projector_hidden, text_hidden)
        self.proj_in = dim * downsample**2

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.embedding_norm(self.patchifier(pixel_values))
        for layer in self.layers:
            x = layer(x, self.cos, self.sin, self.height, self.width)
        x = self.proj_layer_norm(x).reshape(-1, self.proj_in)
        return self.proj_linear_2(F.gelu(self.proj_linear_1(x)))


def vision_state_dict(hf_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in hf_sd.items():
        if k.startswith("model.vision_tower."):
            out[k[len("model.vision_tower.") :]] = v.float()
        elif k.startswith("model.multi_modal_projector."):
            name = k[len("model.multi_modal_projector.") :]
            out["proj_" + name] = v.float()
    return out


class Lfm3OmniVisionEncoder(nn.Module):
    """StaticVisionEncoder with weights and RoPE tables built from the HF checkpoint,
    for the QNN multimodal export flow (no HF modeling code needed)."""

    def __init__(
        self, repo_id: str, img_resized_h: int = 448, img_resized_w: int = 448
    ):
        super().__init__()
        import json

        from executorch.examples.qualcomm.oss_scripts.lfm3.preprocess import (
            PATCH,
            vision_rope,
        )
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file

        assert img_resized_h == img_resized_w, "square inputs only"
        snap = snapshot_download(
            repo_id, allow_patterns=["config.json", "model.safetensors"]
        )
        cfg = json.load(open(f"{snap}/config.json"))
        vc = cfg["vision_config"]
        head_dim = vc["hidden_size"] // vc["num_attention_heads"]
        cos, sin = vision_rope(
            img_resized_h, head_dim, vc["rope_parameters"]["rope_theta"]
        )
        grid = img_resized_h // PATCH
        self.encoder = StaticVisionEncoder(
            vc,
            cfg["text_config"]["hidden_size"],
            cfg["projector_hidden_size"],
            cfg["downsample_factor"],
            cos,
            sin,
            grid,
            grid,
        )
        self.encoder.load_state_dict(
            vision_state_dict(load_file(f"{snap}/model.safetensors")), strict=True
        )
        self.num_patches = grid * grid
        self.patch_dim = self.encoder.patchifier.in_features
        self.n_layers = len(vc["layer_types"])

    def forward(self, pixel_values):
        # The runner's image path adds batch/channel dims on the way in and expects
        # [1, tokens, dim] embeddings out.
        return self.encoder(pixel_values[0, 0]).unsqueeze(0)

    def get_example_inputs(self):
        return (torch.randn(1, 1, self.num_patches, self.patch_dim),)

    def get_metadata(self):
        return {"get_n_layers": self.n_layers}

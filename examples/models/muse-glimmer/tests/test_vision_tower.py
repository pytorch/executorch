# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the export-friendly Muse Glimmer vision encoder (vision_tower.py).

Checks (CPU):
  * ``MuseGlimmerVisionEncoder`` reproduces an inline eager reference (the Muse
    Glimmer vision math) to bf16 tolerance, given shared random weights.
  * The forward runs on a single-tile image (identity sparse perm) and a
    multi-tile image (non-trivial sparse perm), producing the right shapes.
  * ``torch.export(strict=True)`` traces the encoder with a dynamic num_patches.
"""

import unittest

import torch
import torch.nn.functional as F
from executorch.examples.models.muse_glimmer.model.vision_tower import (
    MuseGlimmerVisionEncoder,
)

from executorch.examples.models.muse_glimmer.vision.precompute import (
    MuseGlimmerVisionConfig,
    precompute_vision_inputs,
)


def _tiny_config() -> MuseGlimmerVisionConfig:
    """A small but structurally-faithful vision config (fast on CPU)."""
    return MuseGlimmerVisionConfig(
        patch_size=14,
        patch_temporal=2,
        downsample_factor=2,
        sparse_attention_factor=4,
        pos_emb_grid_h=8,
        pos_emb_grid_w=8,
        latent_dim=32,
        n_heads=4,
        n_layers=6,
        rope_theta=10000.0,
        mlp_hidden=64,
        adapter_dim=48,
        encoder_output_dim=32 * 4,  # latent * downsample^2
        hidden_size=40,
        rms_norm_eps=1e-5,
    )


def _init_random(model: torch.nn.Module, seed: int = 0) -> None:
    torch.manual_seed(seed)
    for p in model.parameters():
        if p.device.type != "meta":
            p.data.normal_(0, 0.05)
    for b in model.buffers():
        if b.device.type != "meta" and b.dtype.is_floating_point:
            b.data.normal_(0, 0.05)


# ---------------------------------------------------------------------------
# Inline eager reference (mirrors MuseGlimmerVisionEncoder math on float32).


def _rotate_interleaved_complex(x, cos, sin):
    """Adjacent-pair rotation via complex mul (eager Muse Glimmer formulation)."""
    freqs = torch.complex(cos, sin)  # [P, d/2]
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    out = torch.view_as_real(xc * freqs.unsqueeze(0).unsqueeze(2)).flatten(-2)
    return out.to(x.dtype)


def _eager_reference(
    model: MuseGlimmerVisionEncoder, inputs, cfg: MuseGlimmerVisionConfig
):
    """Recompute the encoder output in float32 straight from the module weights.

    Uses the same submodule weights as ``model`` but a plain float32 path (no
    quantization, complex RoPE) so it is an independent reference for parity.
    """
    patches = inputs.patches.float()
    pos_emb = inputs.pos_emb.float()
    cos = inputs.cos_2d.float()
    sin = inputs.sin_2d.float()
    perm = inputs.sparse_perm
    inv_perm = inputs.inv_perm
    gmask = inputs.global_mask
    smask = inputs.sparse_mask
    pixel_perm = inputs.pixel_perm

    x = F.linear(patches, model.conv1_linear.weight.float())
    x = x + pos_emb
    x = F.layer_norm(
        x, (cfg.latent_dim,), model.ln_pre.weight.float(), model.ln_pre.bias.float()
    )

    x = x.index_select(1, perm)
    cos_p = cos.index_select(0, perm)
    sin_p = sin.index_select(0, perm)

    n_heads, head_dim = cfg.n_heads, cfg.head_dim
    for i, blk in enumerate(model.blocks):
        mask = gmask if model._is_global[i] else smask
        h = F.layer_norm(
            x, (cfg.latent_dim,), blk.ln_1.weight.float(), blk.ln_1.bias.float()
        )
        B, P, _ = h.shape
        q = F.linear(h, blk.attn.q_proj.weight.float(), blk.attn.q_proj.bias.float())
        k = F.linear(h, blk.attn.k_proj.weight.float(), blk.attn.k_proj.bias.float())
        v = F.linear(h, blk.attn.v_proj.weight.float(), blk.attn.v_proj.bias.float())
        q = q.view(B, P, n_heads, head_dim)
        k = k.view(B, P, n_heads, head_dim)
        v = v.view(B, P, n_heads, head_dim)
        q = _rotate_interleaved_complex(q, cos_p, sin_p).transpose(1, 2)
        k = _rotate_interleaved_complex(k, cos_p, sin_p).transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        attn = attn.transpose(1, 2).contiguous().view(B, P, n_heads * head_dim)
        attn = F.linear(
            attn, blk.attn.o_proj.weight.float(), blk.attn.o_proj.bias.float()
        )
        x = x + attn

        h = F.layer_norm(
            x, (cfg.latent_dim,), blk.ln_2.weight.float(), blk.ln_2.bias.float()
        )
        h = F.linear(h, blk.mlp.c_fc.weight.float(), blk.mlp.c_fc.bias.float())
        h = F.gelu(h)
        h = F.linear(h, blk.mlp.c_proj.weight.float(), blk.mlp.c_proj.bias.float())
        x = x + h

    x = x.index_select(1, inv_perm)
    x = F.layer_norm(
        x, (cfg.latent_dim,), model.ln_post.weight.float(), model.ln_post.bias.float()
    )

    f = cfg.downsample_factor
    d = cfg.latent_dim
    x_flat = x[0].index_select(0, pixel_perm)
    x_ds = (
        x_flat.view(-1, f * f, d).permute(0, 2, 1).contiguous().view(1, -1, d * f * f)
    )

    x_ds = F.gelu(F.linear(x_ds, model.adapter_fc.weight.float()))
    x_ds = F.gelu(F.linear(x_ds, model.adapter_proj.weight.float()))
    x_ds = F.linear(x_ds, model.vision_proj.weight.float())
    x_ds = F.rms_norm(x_ds.float(), (cfg.hidden_size,), None, cfg.rms_norm_eps)
    return x_ds


class VisionTowerParityTest(unittest.TestCase):
    def _run(self, grid_side: int, dtype: torch.dtype = torch.float32):
        cfg = _tiny_config()
        model = MuseGlimmerVisionEncoder(cfg)
        _init_random(model)
        model.to(dtype)
        model.eval()

        px = grid_side * cfg.patch_size
        image = torch.randn(3, px, px)
        table = torch.randn(
            cfg.pos_emb_grid_h * cfg.pos_emb_grid_w,
            cfg.latent_dim,
            dtype=torch.bfloat16,
        )
        inputs = precompute_vision_inputs(image, table, cfg)
        with torch.no_grad():
            out = model(*inputs.as_args())
            ref = _eager_reference(model, inputs, cfg)
        return out, ref, grid_side, cfg

    def test_single_tile_parity(self):
        out, ref, grid_side, cfg = self._run(grid_side=6)  # 6x6 <= 8x8 tile
        n_out = (grid_side // cfg.downsample_factor) ** 2
        self.assertEqual(out.shape, (1, n_out, cfg.hidden_size))
        self.assertEqual(out.dtype, torch.float32)
        torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)

    def test_multi_tile_parity(self):
        # 16x16 grid > 8x8 tile -> 4 tiles -> non-trivial sparse permutation.
        out, ref, grid_side, cfg = self._run(grid_side=16)
        n_out = (grid_side // cfg.downsample_factor) ** 2
        self.assertEqual(out.shape, (1, n_out, cfg.hidden_size))
        torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)

    def test_output_dtype_follows_compute_dtype(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                out, _, _, _ = self._run(grid_side=6, dtype=dtype)
                self.assertEqual(out.dtype, dtype)


class VisionTowerExportTest(unittest.TestCase):
    def test_strict_export_dynamic_patches(self):
        from torch.export import Dim, export

        cfg = _tiny_config()
        model = MuseGlimmerVisionEncoder(cfg)
        _init_random(model)
        model.eval()

        px = 8 * cfg.patch_size
        image = torch.randn(3, px, px)
        table = torch.randn(
            cfg.pos_emb_grid_h * cfg.pos_emb_grid_w,
            cfg.latent_dim,
            dtype=torch.bfloat16,
        )
        inputs = precompute_vision_inputs(image, table, cfg)

        ds2 = cfg.downsample_factor**2
        groups = Dim("vis_groups", min=1, max=64)
        p = ds2 * groups
        dyn = (
            {1: p},
            {1: p},
            {0: p},
            {0: p},
            {0: p},
            {0: p},
            {2: p, 3: p},
            {2: p, 3: p},
            {0: p},
        )
        with torch.no_grad():
            ep = export(model, inputs.as_args(), dynamic_shapes=dyn, strict=True)
        self.assertIsNotNone(ep)
        # Re-run the exported program on the same inputs.
        out = ep.module()(*inputs.as_args())
        self.assertEqual(out.shape[-1], cfg.hidden_size)


if __name__ == "__main__":
    unittest.main()

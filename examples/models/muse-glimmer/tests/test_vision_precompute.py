# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the Muse Glimmer vision host-precompute (vision_precompute.py).

Verifies the host-side precomputed tensors (patchify, positional-embedding
interpolation, 2D-RoPE, sparse permutation, block-diagonal masks, pixel-shuffle
permutation) match the eager ``MuseGlimmerVisionEncoder`` math. Runs on CPU.
"""

import unittest

import torch
from executorch.examples.models.muse_glimmer.vision.precompute import (
    compute_grid_size,
    interpolate_pos_emb,
    make_2d_rope,
    MuseGlimmerVisionConfig,
    patchify_image,
    pixel_shuffle_perm,
    precompute_vision_inputs,
    sparse_perm_and_slens,
)


def _small_config() -> MuseGlimmerVisionConfig:
    # Real vision hyperparams; only the token count varies with image size.
    return MuseGlimmerVisionConfig()


class PatchifyTest(unittest.TestCase):
    def test_shapes_and_grid(self):
        cfg = _small_config()
        # 28-px cell (patch 14 * downsample 2); use a 4x4-cell image (112px).
        h = w = cfg.patch_size * 4  # 56 -> grid 4x4
        image = torch.randn(3, h, w)
        patches, grid_h, grid_w = patchify_image(image, cfg)
        self.assertEqual((grid_h, grid_w), (4, 4))
        self.assertEqual(patches.shape, (1, 16, cfg.patch_dim))
        self.assertEqual(patches.dtype, torch.float32)

    def test_temporal_replication(self):
        """The 2 temporal copies of an image are identical per patch."""
        cfg = _small_config()
        image = torch.randn(3, cfg.patch_size, cfg.patch_size)  # 1 patch
        patches, _, _ = patchify_image(image, cfg)
        row = patches[0, 0]  # [patch_dim = pt*3*ps*ps]
        half = row.numel() // cfg.patch_temporal
        self.assertTrue(torch.equal(row[:half], row[half:]))


class PosEmbInterpTest(unittest.TestCase):
    def test_identity_grid_matches_table(self):
        """Interpolating to the native 32x32 grid ~ returns the table itself."""
        cfg = _small_config()
        gh, gw = cfg.pos_emb_grid_h, cfg.pos_emb_grid_w
        table = torch.randn(gh * gw, cfg.latent_dim, dtype=torch.bfloat16)
        pos = interpolate_pos_emb(table, gh, gw, cfg)
        self.assertEqual(pos.shape, (1, gh * gw, cfg.latent_dim))
        self.assertEqual(pos.dtype, torch.bfloat16)

    def test_downscaled_shape(self):
        cfg = _small_config()
        table = torch.randn(
            cfg.pos_emb_grid_h * cfg.pos_emb_grid_w,
            cfg.latent_dim,
            dtype=torch.bfloat16,
        )
        pos = interpolate_pos_emb(table, 8, 8, cfg)
        self.assertEqual(pos.shape, (1, 64, cfg.latent_dim))


class Rope2DTest(unittest.TestCase):
    def test_shape_and_range(self):
        cfg = _small_config()
        cos, sin = make_2d_rope(6, 6, cfg)
        self.assertEqual(cos.shape, (36, cfg.head_dim // 2))
        self.assertEqual(sin.shape, (36, cfg.head_dim // 2))
        self.assertTrue((cos.abs() <= 1.0 + 1e-6).all())
        self.assertTrue((sin.abs() <= 1.0 + 1e-6).all())

    def test_matches_eager_complex_form(self):
        """Real (cos,sin) reproduce the eager complex freqs_cis element-wise."""
        cfg = _small_config()
        grid_h = grid_w = 5
        cos, sin = make_2d_rope(grid_h, grid_w, cfg)

        # Eager reference (MuseGlimmerVisionEncoder._make_2d_rope), inline.
        head_dim = cfg.head_dim
        half_dim = head_dim // 2
        quarter = half_dim // 2
        theta = cfg.rope_theta
        inv_freq = 1.0 / (
            theta
            ** (torch.arange(0, half_dim, 2, dtype=torch.float32)[:quarter] / half_dim)
        )
        idx_h = torch.arange(1, grid_h + 1, dtype=torch.float32)
        idx_w = torch.arange(1, grid_w + 1, dtype=torch.float32)
        idx_ij_h = idx_h.unsqueeze(1).expand(-1, grid_w).reshape(-1)
        idx_ij_w = idx_w.unsqueeze(0).expand(grid_h, -1).reshape(-1)
        freq_h = torch.outer(idx_ij_h, inv_freq)
        freq_w = torch.outer(idx_ij_w, inv_freq)
        freq = torch.cat([freq_w, freq_h], dim=-1)
        freqs_cis = torch.view_as_complex(
            torch.stack([torch.cos(freq), torch.sin(freq)], dim=-1)
        )
        torch.testing.assert_close(cos, freqs_cis.real, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(sin, freqs_cis.imag, atol=1e-6, rtol=1e-6)


class SparsePermTest(unittest.TestCase):
    def test_identity_when_grid_fits_one_tile(self):
        """A grid <= 32x32 fits in one window: perm is identity, one group."""
        cfg = _small_config()
        perm, slens = sparse_perm_and_slens(8, 8, cfg)
        self.assertTrue(torch.equal(perm, torch.arange(64)))
        self.assertEqual(slens, [64])

    def test_multi_tile_partition(self):
        """A 64x64 grid tiles into four 32x32 blocks of 1024 tokens each."""
        cfg = _small_config()
        grid = 2 * cfg.pos_emb_grid_h  # 64
        perm, slens = sparse_perm_and_slens(grid, grid, cfg)
        self.assertEqual(slens, [1024, 1024, 1024, 1024])
        # A valid permutation of all tokens.
        self.assertEqual(sorted(perm.tolist()), list(range(grid * grid)))

    def test_inv_perm_roundtrip(self):
        cfg = _small_config()
        perm, _ = sparse_perm_and_slens(40, 48, cfg)
        n = perm.numel()
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(n, dtype=perm.dtype)
        x = torch.randn(n, 3)
        torch.testing.assert_close(x[perm][inv], x)


class PixelShufflePermTest(unittest.TestCase):
    def test_matches_eager(self):
        cfg = _small_config()
        grid_h, grid_w, f = 6, 8, cfg.downsample_factor
        perm = pixel_shuffle_perm(grid_h, grid_w, cfg)
        ds = torch.arange(grid_h * grid_w).view(grid_h // f, f, grid_w // f, f)
        ds = ds.permute(0, 2, 1, 3).reshape(-1)
        self.assertTrue(torch.equal(perm, ds.to(torch.int64)))


class GridSizeTest(unittest.TestCase):
    def test_divisible_by_cell(self):
        cfg = _small_config()
        cell = cfg.patch_size * cfg.downsample_factor
        for w, h in [(640, 480), (100, 800), (1024, 1024)]:
            th, tw, n = compute_grid_size(w, h, cfg)
            self.assertEqual(th % cell, 0)
            self.assertEqual(tw % cell, 0)
            self.assertEqual(n, (th // cell) * (tw // cell))
            self.assertLessEqual(n, 4096)


# Grids this function is expected to produce, mirrored verbatim by
# tests/vision_grid_parity_test.cpp so the runner and the tokenizer agree.
#
# Many are exact cost ties that ``min`` breaks by CPython set-iteration order,
# which follows no rule: 1440x1440 resolves to the larger grid and 225x225 to
# the smaller one (both from P2447507824). That order is stable -- it ignores
# PYTHONHASHSEED and has not changed since CPython 3.8 -- but it is not a
# guarantee CPython makes. If a future runtime reorders it, THIS test is what
# fails, and the C++ table has to be regenerated to match.
_REFERENCE_GRIDS = [
    # (img_w, img_h, nph, npw)
    (1440, 1440, 52, 52),
    (225, 225, 8, 8),
    (512, 512, 19, 19),
    (1024, 1024, 37, 37),
    (200, 203, 8, 8),
    (24, 60, 3, 1),
    (700, 700, 25, 25),
    (336, 336, 12, 12),
    (1000, 2500, 90, 36),
    (6000, 6000, 64, 64),
    (1, 1, 1, 1),
    (27, 27, 1, 1),
    (28, 28, 1, 1),
    (1, 10000, 358, 1),
]


class GridReferenceParityTest(unittest.TestCase):
    def test_grids_match_pinned_table(self):
        cfg = _small_config()
        cell = cfg.patch_size * cfg.downsample_factor
        actual = []
        expected = []
        for w, h, nph, npw in _REFERENCE_GRIDS:
            th, tw, _ = compute_grid_size(w, h, cfg)
            actual.append((w, h, th // cell, tw // cell))
            expected.append((w, h, nph, npw))
        self.assertEqual(actual, expected)

    def test_ties_resolve_in_opposite_directions(self):
        # The pair from P2447507824 that rules out any fixed tie-break: both
        # are exact ties, resolved in opposite directions.
        cfg = _small_config()
        cell = cfg.patch_size * cfg.downsample_factor
        th_ceil, _, _ = compute_grid_size(1440, 1440, cfg)
        th_floor, _, _ = compute_grid_size(225, 225, cfg)
        self.assertEqual((th_ceil // cell, th_floor // cell), (52, 8))


class ResizeReferenceParityTest(unittest.TestCase):
    """Pillow's LANCZOS output, pinned so a Pillow upgrade cannot drift silently.

    tests/vision_resize_parity_test.cpp asserts the C++ ``resize_rgb_lanczos``
    reproduces these same bytes. Regenerate both with
    ``python gen_resize_golden.py``.
    """

    def test_matches_pinned_golden(self):
        from executorch.examples.models.muse_glimmer.tests.gen_resize_golden import (
            make_image,
        )
        from PIL import Image

        # Spot values sampled across each output; the C++ test pins every byte.
        expectations = [
            (9, 7, 5, 4, [19, 255, 1], [68, 16, 153]),
            (4, 3, 7, 6, [0, 255, 0], [137, 16, 22]),
            (16, 16, 8, 8, [25, 250, 1], [184, 16, 99]),
        ]
        actual = []
        expected = []
        for sw, sh, dw, dh, head, tail in expectations:
            out = list(make_image(sw, sh).resize((dw, dh), Image.LANCZOS).tobytes())
            self.assertEqual(len(out), dw * dh * 3)
            actual.append((out[:3], out[-3:]))
            expected.append((head, tail))
        self.assertEqual(actual, expected)


class PrecomputeBundleTest(unittest.TestCase):
    def test_all_inputs_consistent(self):
        cfg = _small_config()
        # 8x8 grid image (fits one tile -> identity sparse perm).
        px = 8 * cfg.patch_size
        image = torch.randn(3, px, px)
        table = torch.randn(
            cfg.pos_emb_grid_h * cfg.pos_emb_grid_w,
            cfg.latent_dim,
            dtype=torch.bfloat16,
        )
        out = precompute_vision_inputs(image, table, cfg)
        p = 64  # 8*8 patches
        self.assertEqual(out.patches.shape, (1, p, cfg.patch_dim))
        self.assertEqual(out.pos_emb.shape, (1, p, cfg.latent_dim))
        self.assertEqual(out.cos_2d.shape, (p, cfg.head_dim // 2))
        self.assertEqual(out.sin_2d.shape, (p, cfg.head_dim // 2))
        self.assertEqual(out.sparse_perm.shape, (p,))
        self.assertEqual(out.inv_perm.shape, (p,))
        self.assertEqual(out.global_mask.shape, (1, 1, p, p))
        self.assertEqual(out.sparse_mask.shape, (1, 1, p, p))
        self.assertEqual(out.pixel_perm.shape, (p,))
        # Single tile: global == sparse (all-True) and both are full attention.
        self.assertTrue(out.global_mask.all())
        self.assertTrue(out.sparse_mask.all())
        self.assertEqual(len(out.as_args()), 9)


if __name__ == "__main__":
    unittest.main()

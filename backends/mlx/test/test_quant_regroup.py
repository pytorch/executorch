# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for MLX affine group_size regrouping (op_helpers). No hardware needed."""

import unittest

import torch
from executorch.backends.mlx.builder.op_helpers import (
    mlx_affine_group_size,
    regroup_affine_scales,
)


class TestMlxAffineGroupSize(unittest.TestCase):
    def test_supported_passthrough(self):
        for gs in (16, 32, 64, 128):
            self.assertEqual(mlx_affine_group_size(gs), gs)

    def test_coarse_maps_to_largest_divisor(self):
        self.assertEqual(mlx_affine_group_size(256), 128)  # int4 common case
        self.assertEqual(mlx_affine_group_size(5376), 128)  # per-axis lm_head
        self.assertEqual(mlx_affine_group_size(48), 16)  # only 16 divides

    def test_no_legal_divisor_returns_none(self):
        self.assertIsNone(mlx_affine_group_size(5))
        self.assertIsNone(mlx_affine_group_size(24))  # 24%16!=0, 24%32.. none


class TestRegroupAffineScales(unittest.TestCase):
    def test_noop_when_already_target(self):
        scale = torch.randn(64, 8)  # K=256, gs=32
        zp = torch.zeros(64, 8, dtype=torch.int8)
        s2, z2, changed = regroup_affine_scales(scale, zp, 256, 32)
        self.assertFalse(changed)
        self.assertIs(s2, scale)
        self.assertIs(z2, zp)

    def test_per_axis_regroups_shape(self):
        scale = torch.randn(64, 1)  # per-axis, K=256 -> gs=256
        zp = torch.zeros(64, 1, dtype=torch.int8)
        s2, z2, changed = regroup_affine_scales(scale, zp, 256, 128)
        self.assertTrue(changed)
        self.assertEqual(s2.shape, (64, 2))
        self.assertEqual(z2.shape, (64, 2))

    def test_regroup_preserves_dequant(self):
        # A coarse group is a stack of finer groups sharing one scale, so
        # regrouping must not change the dequantized values.
        torch.manual_seed(0)
        N, K = 4, 256
        qdata = torch.randint(-8, 7, (N, K), dtype=torch.int8)
        scale = torch.rand(N, 1) + 0.1  # per-axis (group_size = K)
        zp = torch.randint(-4, 4, (N, 1), dtype=torch.int8)

        coarse = (qdata.float() - zp.float()) * scale  # one scale per row

        s2, z2, changed = regroup_affine_scales(scale, zp, K, 128)
        self.assertTrue(changed)
        fine = (qdata.float() - z2.repeat_interleave(128, -1).float()) * (
            s2.repeat_interleave(128, -1)
        )
        self.assertTrue(torch.allclose(coarse, fine))


if __name__ == "__main__":
    unittest.main()

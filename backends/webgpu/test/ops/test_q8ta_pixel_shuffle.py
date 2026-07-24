# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`et_vk.q8ta_pixel_shuffle` module + configs: int8 pixel_shuffle.

The int8 input `[N, C*r*r, H, W]` is a baked constant, so the op is exercised
alone with a byte-exact int8 golden vs the CPU eager op (gather then dequant ->
requant). Note the schema's 4th arg is `output_inv_scale` (already 1/scale).
"""

import unittest

import executorch.backends.vulkan.custom_ops_lib  # noqa: F401 registers et_vk ops

import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower


class Q8taPixelShuffleModule(torch.nn.Module):
    def __init__(
        self,
        x_vals,
        shape,
        input_scale=0.05,
        input_zp=-3,
        output_scale=0.05,
        output_zp=-3,
        upscale_factor=2,
    ):
        super().__init__()
        self.register_buffer("x", torch.tensor(x_vals, dtype=torch.int8).reshape(shape))
        self.input_scale, self.input_zp = input_scale, input_zp
        self.output_inv_scale = 1.0 / output_scale
        self.output_zp, self.upscale_factor = output_zp, upscale_factor

    def forward(self) -> torch.Tensor:
        return torch.ops.et_vk.q8ta_pixel_shuffle(
            self.x,
            self.input_scale,
            self.input_zp,
            self.output_inv_scale,
            self.output_zp,
            self.upscale_factor,
        )


class Q8taPixelShuffleTest(unittest.TestCase):
    def test_delegates(self) -> None:
        m = Q8taPixelShuffleModule(list(range(-8, 8)), (1, 4, 2, 2))
        et = to_edge_transform_and_lower(
            torch.export.export(m, ()), partitioner=[VulkanPartitioner()]
        ).to_executorch()
        self.assertTrue(
            any(
                d.id == "VulkanBackend"
                for plan in et.executorch_program.execution_plan
                for d in plan.delegates
            )
        )

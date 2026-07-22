# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`et_vk.q8ta_relu` module + configs: int8 relu.

The int8 input is a baked constant, so the op is exercised alone with a byte-exact
int8 golden vs the CPU eager op. Inputs whose dequantized value is negative are
clamped to 0 by the relu, pinning the `max(x, 0)` term.
"""

import unittest

import torch

import executorch.backends.vulkan.custom_ops_lib  # noqa: F401 registers et_vk ops

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower


class Q8taReluModule(torch.nn.Module):
    def __init__(
        self,
        x_vals,
        input_scale=0.05,
        input_zp=-3,
        output_scale=0.05,
        output_zp=-3,
    ):
        super().__init__()
        self.register_buffer("x", torch.tensor(x_vals, dtype=torch.int8))
        self.input_scale, self.input_zp = input_scale, input_zp
        self.output_scale, self.output_zp = output_scale, output_zp

    def forward(self) -> torch.Tensor:
        return torch.ops.et_vk.q8ta_relu(
            self.x,
            self.input_scale,
            self.input_zp,
            self.output_scale,
            self.output_zp,
        )


class Q8taReluTest(unittest.TestCase):
    def test_delegates(self) -> None:
        m = Q8taReluModule(list(range(-8, 8)))
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

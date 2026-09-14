# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.vulkan.quantizer.vulkan_quantizer import (
    get_symmetric_quantization_config,
    VulkanQuantizer,
)
from torch.export import export
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e


class TestVulkanQuantizer(unittest.TestCase):
    def test_int64_scalar_add_used_as_index(self):
        """Scalars lifted to attrs must keep the op's output dtype; an int64
        add chain used as an index must not be promoted to float32."""

        class M(torch.nn.Module):
            def forward(self, x):
                return x[:, torch.arange(4) + 0]

        quantizer = VulkanQuantizer()
        quantization_config = get_symmetric_quantization_config()
        quantizer.set_global(quantization_config)
        example_inputs = (torch.randn(1, 4, 5),)
        m = export(M(), example_inputs, strict=True).module()
        m = quantizer.transform_for_annotation(m)
        lifted_constants = [
            m.get_buffer(n.target)
            for n in m.graph.nodes
            if n.op == "get_attr" and n.target.startswith("_tensor_constant_")
        ]
        self.assertEqual(len(lifted_constants), 1)
        self.assertEqual(lifted_constants[0].dtype, torch.int64)
        m = prepare_pt2e(m, quantizer)
        m(*example_inputs)

    def test_int64_scalar_lifted_without_set_global(self):
        """Passing through VulkanQuantizer without a config still lifts scalars
        and must preserve dtype."""

        class M(torch.nn.Module):
            def forward(self, x):
                return x[:, torch.arange(4) + 0]

        example_inputs = (torch.randn(1, 4, 5),)
        m = export(M(), example_inputs, strict=True).module()
        m = VulkanQuantizer().transform_for_annotation(m)
        lifted_constants = [
            m.get_buffer(n.target)
            for n in m.graph.nodes
            if n.op == "get_attr" and n.target.startswith("_tensor_constant_")
        ]
        self.assertEqual(len(lifted_constants), 1)
        self.assertEqual(lifted_constants[0].dtype, torch.int64)
        m(*example_inputs)

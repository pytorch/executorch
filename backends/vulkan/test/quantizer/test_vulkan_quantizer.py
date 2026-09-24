# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
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
        class M(torch.nn.Module):
            def forward(self, x):
                return x[:, torch.arange(4) + 0]

        quantizer = VulkanQuantizer()
        quantization_config = get_symmetric_quantization_config()
        quantizer.set_global(quantization_config)
        example_inputs = (torch.randn(1, 4, 5),)
        m = export(M(), example_inputs, strict=True).module()
        m = quantizer.transform_for_annotation(m)
        m = prepare_pt2e(m, quantizer)
        torch.testing.assert_close(m(*example_inputs), M()(*example_inputs))

    def test_int64_scalar_add_without_set_global(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x[:, torch.arange(4) + 0]

        example_inputs = (torch.randn(1, 4, 5),)
        m = export(M(), example_inputs, strict=True).module()
        m = VulkanQuantizer().transform_for_annotation(m)
        torch.testing.assert_close(m(*example_inputs), M()(*example_inputs))

    def test_scalar_type_promotion(self):
        class M(torch.nn.Module):
            def __init__(self, op, scalar):
                super().__init__()
                self.op = op
                self.scalar = scalar

            def forward(self, x):
                return self.op(x, self.scalar)

        cases = [
            (torch.float16, 1e-4, 100000.0),
            (torch.float16, 1e-4, 100000),
            (torch.float16, 10000.0, 1e-8),
            (torch.bfloat16, 100.0, 1.0039),
            (torch.float64, 1.0, 1.0 + 2**-30),
            (torch.int64, 2**54 + 1, 1),
            (torch.int32, 1, 1),
            (torch.int32, 1, 2**31),
            (torch.int8, 1, 256),
            (torch.bool, True, False),
            (torch.int32, 1, 0.5),
            (torch.float32, 1.0, 2),
            (torch.complex64, 1j, 1 + 2j),
        ]
        for op, (dtype, value, scalar), shape, configured in itertools.product(
            (torch.add, torch.mul), cases, ((), (2,)), (False, True)
        ):
            with self.subTest(
                op=op, dtype=dtype, scalar=scalar, shape=shape, configured=configured
            ):
                model = M(op, scalar)
                example_inputs = (torch.full(shape, value, dtype=dtype),)
                expected = model(*example_inputs)
                quantizer = VulkanQuantizer()
                if configured:
                    quantizer.set_global(get_symmetric_quantization_config())
                m = export(model, example_inputs, strict=True).module()
                m = prepare_pt2e(m, quantizer)
                torch.testing.assert_close(m(*example_inputs), expected, rtol=0, atol=0)

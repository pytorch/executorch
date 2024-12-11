# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestMaxPool2d(unittest.TestCase):
    class MaxPool2d(torch.nn.Module):
        def __init__(self, kernel_size=3, stride=1, padding=0, dilation=1):
            super().__init__()
            self.max_pool2d_module = torch.nn.MaxPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )

        def forward(self, x):
            return self.max_pool2d_module(x)

    class MaxPool2dUnsupported(torch.nn.Module):
        def __init__(self, kernel_size=3, stride=1, padding=0, dilation=1):
            super().__init__()
            self.max_pool2d_module = torch.nn.MaxPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                return_indices=True,
            )

        def forward(self, x):
            return self.max_pool2d_module(x)[1]

    class MaxPool2dUnsupportedCeilMode(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.max_pool2d_module = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)

        def forward(self, x):
            return self.max_pool2d_module(x)

    def _test_maxpool2d(self, inputs):
        """
        Note that the export process generates aten.max_pool2d_with_indices. The remove_getitem_op
        pass transforms it into aten.max_pool2d (if supported).
        """
        (
            Tester(self.MaxPool2d(3, 1, 0, 1), inputs)
            .export()
            .check_count({"torch.ops.aten.max_pool2d.default": 1})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(
                [
                    "executorch_exir_dialects_edge__ops_aten_max_pool2d_with_indices_default"
                ]
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp16_maxpool2d(self):
        inputs = (torch.randn(4, 3, 24, 24).to(torch.float16),)
        self._test_maxpool2d(inputs)

    def test_fp32_maxpool2d(self):
        inputs = (torch.randn(4, 3, 24, 24),)
        self._test_maxpool2d(inputs)

    def test_fp32_maxpool2d_unsupported(self):
        """
        MaxPool2d with return_indices is not generally supported (see maxpool2d_with_indices constraint).
        """
        inputs = (torch.randn(4, 3, 24, 24),)
        (
            Tester(self.MaxPool2dUnsupported(), inputs)
            .export()
            .check_count({"torch.ops.aten.max_pool2d_with_indices.default": 1})
            .to_edge_transform_and_lower()
            # We expect it not be be delegated.
            .check_count(
                {
                    "executorch_exir_dialects_edge__ops_aten_max_pool2d_with_indices_default": 1
                }
            )
        )

    def test_fp32_maxpool2d_unsupported_ceilmode(self):
        """
        MaxPool2d with ceil mode is not generally supported (see maxpool2d constraint).
        """
        inputs = (torch.randn(1, 32, 23, 23),)
        (
            Tester(self.MaxPool2dUnsupportedCeilMode(), inputs)
            .export()
            .check_count({"torch.ops.aten.max_pool2d.default": 1})
            .to_edge_transform_and_lower()
            # We expect it not be be delegated.
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 0})
            .check_count(
                {
                    "executorch_exir_dialects_edge__ops_aten_max_pool2d_with_indices_default": 1
                }
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_qs8_maxpool2d(self):
        class MaxPool(torch.nn.Module):
            def __init__(self, maxpool_params):
                super().__init__()
                self.max = torch.nn.MaxPool2d(*maxpool_params)

            def forward(self, x):
                z = x + x
                return self.max(z)

        # Parameter order is kernel_size, stride, padding.
        for maxpool_params in [(4,), (4, 2), (4, 2, 2)]:
            inputs = (torch.randn(1, 2, 8, 8),)
            (
                Tester(MaxPool(maxpool_params), inputs)
                .quantize()
                .export()
                .check_count({"torch.ops.aten.max_pool2d.default": 1})
                .check(["torch.ops.quantized_decomposed"])
                .to_edge_transform_and_lower()
                .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
                .check_not(
                    [
                        "executorch_exir_dialects_edge__ops_aten_max_pool2d_with_indices_default",
                        "torch.ops.quantized_decomposed",
                    ]
                )
                .to_executorch()
                .serialize()
                .run_method_and_compare_outputs()
            )

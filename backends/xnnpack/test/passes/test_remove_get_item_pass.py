# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack._passes.remove_getitem_op import RemoveGetItemPass
from executorch.backends.xnnpack.test.tester import RunPasses, Tester


class TestRemoveGetItemPass(unittest.TestCase):
    PassStage = RunPasses([RemoveGetItemPass])
    max_pool2d_name = "executorch_exir_dialects_edge__ops_aten_max_pool2d_default"
    amax_name = "executorch_exir_dialects_edge__ops_aten_amax_default"

    class MaxPool2dModule(torch.nn.Module):
        def __init__(
            self,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
        ):
            super().__init__()
            self.max_pool2d_module = torch.nn.MaxPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )

        def forward(self, x):
            return self.max_pool2d_module(x)

    def test_fp32_max_pool2d_remove_getitem(self):
        (
            Tester(self.MaxPool2dModule(), (torch.randn(4, 3, 24, 24),))
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check_count({self.max_pool2d_name: 1})
            .run_method_and_compare_outputs()
        )

    def test_q8_max_pool2d_remove_getitem(self):
        (
            Tester(self.MaxPool2dModule(), (torch.randn(4, 3, 24, 24),))
            .quantize()
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check_count({self.max_pool2d_name: 1})
            .run_method_and_compare_outputs()
        )

    class MaxModule(torch.nn.Module):
        def __init__(
            self,
        ):
            super().__init__()

        def forward(self, x):
            max_vals, indices = torch.max(x, dim=2, keepdim=True)
            return max_vals

    def test_fp32_max_remove_getitem(self):
        (
            Tester(self.MaxModule(), (torch.randn(4, 3, 24, 24),))
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check_count(
                {
                    self.amax_name: 1,
                }
            )
            .run_method_and_compare_outputs()
        )

    def test_q8_max_remove_getitem(self):
        (
            Tester(self.MaxModule(), (torch.randn(4, 3, 24, 24),))
            .quantize()
            .export()
            .to_edge()
            .run_passes(self.PassStage)
            .check_count(
                {
                    self.amax_name: 1,
                }
            )
            .run_method_and_compare_outputs()
        )

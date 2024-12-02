# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Tester


class TestMaxDim(unittest.TestCase):
    class Max(torch.nn.Module):
        def forward(self, x):
            max_values_1, max_indices_1 = torch.max(x, dim=2, keepdim=True)
            max_values_2, max_indices_2 = torch.max(x, dim=3, keepdim=True)
            return (max_values_1, max_indices_1, max_values_2, max_indices_2)

    class MaxNoIndices(torch.nn.Module):
        def forward(self, x):
            max_values_1, _ = torch.max(x, dim=2, keepdim=True)
            max_values_2, _ = torch.max(x, dim=3, keepdim=True)
            return (max_values_1, max_values_2)

    def _test_max_dim(self, inputs):
        (
            Tester(self.Max(), inputs)
            .export()
            .check_count({"torch.ops.aten.max.dim": 2})
            .to_edge_transform_and_lower()
            .check_not(["torch.ops.higher_order.executorch_call_delegate"])
            .check_count({"executorch_exir_dialects_edge__ops_aten_max_dim": 2})
        )

    def _test_max_dim_no_indicies(self, inputs):
        (
            Tester(self.MaxNoIndices(), inputs)
            .export()
            .check_count({"torch.ops.aten.max.dim": 2})
            .to_edge_transform_and_lower()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .check_not(["executorch_exir_dialects_edge__ops_aten_max_dim"])
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp16_max_dim_with_indicies(self):
        inputs = (torch.randn(16, 3, 12, 12).to(torch.float16),)
        self._test_max_dim(inputs)

    def test_fp32_max_dim_with_indices(self):
        inputs = (torch.randn(16, 3, 12, 12),)
        self._test_max_dim(inputs)

    def test_fp32_max_dim_no_indices(self):
        inputs = (torch.randn(16, 3, 12, 12),)
        self._test_max_dim_no_indicies(inputs)

    def test_fp16_max_dim_no_indices(self):
        inputs = (torch.randn(16, 3, 12, 12).to(torch.float16),)
        self._test_max_dim_no_indicies(inputs)

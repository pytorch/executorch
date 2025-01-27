# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Export, Tester
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import Dim


class TestUnsqueeze(unittest.TestCase):
    class Unsqueeze(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, x):
            return torch.unsqueeze(x, self.dim)

    def test_fp32_unsqueeze(self):
        inputs = (torch.randn(1, 2, 4),)
        for dim in range(len(inputs[0].shape)):
            (
                Tester(self.Unsqueeze(dim), inputs)
                .export()
                .check_node_count(
                    {
                        torch.ops.aten.unsqueeze.default: 1,
                    }
                )
                .to_edge_transform_and_lower()
                .check_node_count(
                    {
                        exir_ops.edge.aten.unsqueeze_copy.default: 0,
                        exir_ops.edge.aten.view_copy.default: 0,
                        torch.ops.higher_order.executorch_call_delegate: 1,
                    }
                )
                .run_method_and_compare_outputs()
            )

    def test_fp16_unsqueeze(self):
        inputs = (torch.randn(1, 2, 4).to(torch.float16),)
        for dim in range(len(inputs[0].shape)):
            (
                Tester(self.Unsqueeze(dim), inputs)
                .export()
                .check_node_count(
                    {
                        torch.ops.aten.unsqueeze.default: 1,
                    }
                )
                .to_edge_transform_and_lower()
                .check_node_count(
                    {
                        exir_ops.edge.aten.unsqueeze_copy.default: 0,
                        exir_ops.edge.aten.view_copy.default: 0,
                        torch.ops.higher_order.executorch_call_delegate: 1,
                    }
                )
                .run_method_and_compare_outputs()
            )

    def test_fp32_unsqueeze_dynamic(self):
        inputs = (torch.randn(1, 2, 4),)
        dynamic_shapes = {"x": {1: Dim("x_1", min=1, max=10)}}

        for dim in range(len(inputs[0].shape)):
            (
                Tester(self.Unsqueeze(dim), inputs)
                .export(Export(dynamic_shapes=dynamic_shapes))
                .check_node_count(
                    {
                        torch.ops.aten.unsqueeze.default: 1,
                    }
                )
                .to_edge_transform_and_lower()
                .check_node_count(
                    {
                        exir_ops.edge.aten.unsqueeze_copy.default: 0,
                        exir_ops.edge.aten.view_copy.default: 0,
                        torch.ops.higher_order.executorch_call_delegate: 1,
                    }
                )
                .run_method_and_compare_outputs()
            )

    def test_fp32_unsqueeze_unsupported_dynamism(self):
        inputs = (torch.randn(1, 2, 4),)
        # Only one dynamic dimension is supported.
        dynamic_shapes = {
            "x": {
                1: Dim("x_1", min=1, max=10),
                2: Dim("x_2", min=1, max=10),
            }
        }

        for dim in range(len(inputs[0].shape)):
            (
                Tester(self.Unsqueeze(dim), inputs)
                .export(Export(dynamic_shapes=dynamic_shapes))
                .check_node_count(
                    {
                        torch.ops.aten.unsqueeze.default: 1,
                    }
                )
                .to_edge_transform_and_lower()
                .check_node_count(
                    {
                        exir_ops.edge.aten.unsqueeze_copy.default: 1,
                        torch.ops.higher_order.executorch_call_delegate: 0,
                    }
                )
                .run_method_and_compare_outputs()
            )

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


class TestSqueeze(unittest.TestCase):
    class Squeeze(torch.nn.Module):
        def __init__(self, dims):
            super().__init__()
            self.dims = dims

        def forward(self, x):
            return torch.squeeze(x, self.dims)

    def test_fp32_squeeze(self):
        inputs = (torch.randn(1, 2, 1, 4, 1),)
        squeeze_dims = (0, 2, 4)

        for dims in squeeze_dims:
            (
                Tester(self.Squeeze(dims), inputs)
                .export()
                .check_node_count(
                    {
                        torch.ops.aten.squeeze.dim: 1,
                    }
                )
                .to_edge_transform_and_lower()
                .check_node_count(
                    {
                        exir_ops.edge.aten.squeeze_copy.dim: 0,
                        exir_ops.edge.aten.view_copy.default: 0,
                        torch.ops.higher_order.executorch_call_delegate: 1,
                    }
                )
                .run_method_and_compare_outputs()
            )

    def test_fp16_squeeze(self):
        inputs = (torch.randn(1, 2, 1, 4, 1).to(torch.float16),)
        squeeze_dims = (0, 2, 4)

        for dims in squeeze_dims:
            (
                Tester(self.Squeeze(dims), inputs)
                .export()
                .check_node_count(
                    {
                        torch.ops.aten.squeeze.dim: 1,
                    }
                )
                .to_edge_transform_and_lower()
                .check_node_count(
                    {
                        exir_ops.edge.aten.squeeze_copy.dim: 0,
                        exir_ops.edge.aten.view_copy.default: 0,
                        torch.ops.higher_order.executorch_call_delegate: 1,
                    }
                )
                .run_method_and_compare_outputs()
            )

    def test_fp32_squeeze_dynamic(self):
        inputs = (torch.randn(1, 2, 1, 4, 1),)
        squeeze_dims = (0, 2, 4)
        dynamic_shapes = {"x": {1: Dim("x_1", min=1, max=10)}}

        for dims in squeeze_dims:
            (
                Tester(self.Squeeze(dims), inputs)
                .export(Export(dynamic_shapes=dynamic_shapes))
                .check_node_count(
                    {
                        torch.ops.aten.squeeze.dim: 1,
                    }
                )
                .to_edge_transform_and_lower()
                .check_node_count(
                    {
                        exir_ops.edge.aten.squeeze_copy.dim: 0,
                        exir_ops.edge.aten.view_copy.default: 0,
                        torch.ops.higher_order.executorch_call_delegate: 1,
                    }
                )
                .run_method_and_compare_outputs()
            )

    def test_fp32_squeeze_unsupported_dynamism(self):
        inputs = (torch.randn(1, 2, 1, 4, 1),)
        squeeze_dims = (0, 2, 4)
        # Only one dynamic dimension is supported.
        dynamic_shapes = {
            "x": {
                1: Dim("x_1", min=1, max=10),
                3: Dim("x_3", min=1, max=10),
            }
        }

        for dims in squeeze_dims:
            (
                Tester(self.Squeeze(dims), inputs)
                .export(Export(dynamic_shapes=dynamic_shapes))
                .check_node_count(
                    {
                        torch.ops.aten.squeeze.dim: 1,
                    }
                )
                .to_edge_transform_and_lower()
                .check_node_count(
                    {
                        exir_ops.edge.aten.squeeze_copy.dims: 1,
                        torch.ops.higher_order.executorch_call_delegate: 0,
                    }
                )
                .run_method_and_compare_outputs()
            )

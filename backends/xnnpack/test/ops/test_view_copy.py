# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

import torch
from executorch.backends.xnnpack.test.tester import Export, Tester
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import Dim


class TestViewCopy(unittest.TestCase):
    class View(torch.nn.Module):
        def __init__(self, new_shape):
            super().__init__()
            self.new_shape = new_shape

        def forward(self, x):
            z = x.view(self.new_shape)
            return z

    def test_fp16_view_copy(self):
        inputs = (torch.randn(4, 4).to(torch.float16),)
        (
            Tester(self.View((2, 8)), inputs)
            .export()
            .check_node_count({torch.ops.aten.view.default: 1})
            .to_edge_transform_and_lower()
            .check_node_count(
                {
                    torch.ops.higher_order.executorch_call_delegate: 1,
                    exir_ops.edge.aten.view_copy.default: 0,
                }
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp32_view_copy(self):
        inputs = (torch.randn(4, 4),)
        (
            Tester(self.View((2, 8)), inputs)
            .export()
            .check_node_count({torch.ops.aten.view.default: 1})
            .to_edge_transform_and_lower()
            .check_node_count(
                {
                    torch.ops.higher_order.executorch_call_delegate: 1,
                    exir_ops.edge.aten.view_copy.default: 0,
                }
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp32_view_copy_inferred_dim(self):
        inputs = (torch.randn(4, 4),)
        (
            Tester(self.View((-1, 8)), inputs)
            .export()
            .check_node_count({torch.ops.aten.view.default: 1})
            .to_edge_transform_and_lower()
            .check_node_count(
                {
                    torch.ops.higher_order.executorch_call_delegate: 1,
                    exir_ops.edge.aten.view_copy.default: 0,
                }
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp32_view_copy_dynamic_shape(self):
        inputs = (torch.randn(4, 4, 6),)
        for dynamic_dim_index in range(len(inputs[0].shape)):
            dynamic_shapes = {
                "x": {dynamic_dim_index: Dim("x", min=1, max=10) * 2},
            }

            # Test as min and max bounds.
            test_inputs = [
                (inputs[0].clone(),),
                (inputs[0].clone(),),
            ]
            test_inputs[0][0][dynamic_dim_index] = 2
            test_inputs[1][0][dynamic_dim_index] = 20

            # Non-dynamic dimensions are halved in the view.
            view_size = [n // 2 for n in inputs[0].shape]
            view_size[dynamic_dim_index] = -1

            tester = (
                Tester(self.View(view_size), inputs)
                .export(Export(dynamic_shapes=dynamic_shapes))
                .check_node_count({torch.ops.aten.view.default: 1})
                .to_edge_transform_and_lower()
                .check_node_count(
                    {
                        torch.ops.higher_order.executorch_call_delegate: 1,
                        exir_ops.edge.aten.view_copy.default: 0,
                    }
                )
                .to_executorch()
                .serialize()
                .run_method_and_compare_outputs()
            )

            for test_input in test_inputs:
                tester.run_method_and_compare_outputs(inputs=test_input)

    def test_fp32_view_copy_unsupported_dynamism(self):
        class SymbolicView(torch.nn.Module):
            def forward(self, x):
                return x.view(x.shape[0] // 2, x.shape[1] * 2)

        inputs = (torch.randn(4, 4),)
        dynamic_shapes = {
            "x": {1: Dim("x", min=1, max=10) * 2},
        }
        (
            Tester(SymbolicView(), inputs)
            .export(Export(dynamic_shapes=dynamic_shapes))
            .check_node_count({torch.ops.aten.view.default: 1})
            .to_edge_transform_and_lower()
            .check_node_count(
                {  # Expect no delegation as the view has two dynamic dimensions.
                    torch.ops.higher_order.executorch_call_delegate: 0,
                    exir_ops.edge.aten.view_copy.default: 1,
                }
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp32_view_copy_static_symbolic_arg(self):
        class SymbolicView(torch.nn.Module):
            def forward(self, x):
                return x.view(x.shape[0] // 2, x.shape[1] * 2)

        inputs = (torch.randn(4, 4),)
        (
            Tester(SymbolicView(), inputs)
            .export()
            .check_node_count({torch.ops.aten.view.default: 1})
            .to_edge_transform_and_lower()
            .check_node_count(
                {
                    # Expect delegatation, as the the symbolic shape expressions will
                    # be resolved to static values in the absense of dynamic shapes.
                    torch.ops.higher_order.executorch_call_delegate: 1,
                    exir_ops.edge.aten.view_copy.default: 0,
                }
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp32_view_copy_increase_rank(self):
        inputs = (torch.randn(4, 4),)
        (
            Tester(self.View((1, 2, 4, 2)), inputs)
            .export()
            .check_node_count({torch.ops.aten.view.default: 1})
            .to_edge_transform_and_lower()
            .check_node_count(
                {
                    torch.ops.higher_order.executorch_call_delegate: 1,
                    exir_ops.edge.aten.view_copy.default: 0,
                }
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp32_view_copy_increase_rank_dynamic(self):
        test_inputs = (
            (torch.randn(2, 4),),
            (torch.randn(10, 4),),
        )
        dynamic_shapes = {
            "x": {0: Dim("x", min=1, max=10) * 2},
        }
        inputs = (torch.randn(4, 4),)
        tester = (
            Tester(self.View((1, 2, 4, -1)), inputs)
            .export(Export(dynamic_shapes=dynamic_shapes))
            .check_node_count({torch.ops.aten.view.default: 1})
            .to_edge_transform_and_lower()
            .check_node_count(
                {
                    torch.ops.higher_order.executorch_call_delegate: 1,
                    exir_ops.edge.aten.view_copy.default: 0,
                }
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

        for test_input in test_inputs:
            tester.run_method_and_compare_outputs(inputs=test_input)

    def test_fp32_view_copy_decrease_rank(self):
        inputs = (torch.randn(4, 4),)
        (
            Tester(self.View((-1)), inputs)
            .export()
            .check_node_count({torch.ops.aten.view.default: 1})
            .to_edge_transform_and_lower()
            .check_node_count(
                {
                    torch.ops.higher_order.executorch_call_delegate: 1,
                    exir_ops.edge.aten.view_copy.default: 0,
                }
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

    def test_fp32_view_copy_decrease_rank_dynamic(self):
        test_inputs = (
            (torch.randn(2, 2, 4),),
            (torch.randn(2, 10, 4),),
        )
        dynamic_shapes = {
            "x": {1: Dim("x", min=1, max=10) * 2},
        }
        inputs = (torch.randn(2, 4, 4),)
        tester = (
            Tester(self.View((-1)), inputs)
            .export(Export(dynamic_shapes=dynamic_shapes))
            .check_node_count({torch.ops.aten.view.default: 1})
            .to_edge_transform_and_lower()
            .check_node_count(
                {
                    torch.ops.higher_order.executorch_call_delegate: 1,
                    exir_ops.edge.aten.view_copy.default: 0,
                }
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

        for test_input in test_inputs:
            tester.run_method_and_compare_outputs(inputs=test_input)

    def test_fp32_view_copy_nhwc(self):
        class ViewNHWC(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3)
                self.conv2 = torch.nn.Conv2d(3, 3, 3)

            def forward(self, x):
                y = self.conv1(x)
                y = y.view(1, 3, 3, -1)
                y = self.conv2(y)
                return y.view(1, 3, 2, -1)

        inputs = (torch.randn(1, 3, 8, 8),)
        (
            Tester(ViewNHWC(), inputs)
            .export()
            .dump_artifact()
            .check_node_count({torch.ops.aten.view.default: 2})
            .to_edge_transform_and_lower()
            .dump_artifact()
            .check_node_count(
                {
                    torch.ops.higher_order.executorch_call_delegate: 1,
                    exir_ops.edge.aten.view_copy.default: 0,
                }
            )
            .to_executorch()
            .serialize()
            .run_method_and_compare_outputs()
        )

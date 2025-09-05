# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.transforms.remove_clone_ops import RemoveCloneOpsTransform
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dim_order_utils import is_channel_last_dim_order
from executorch.exir.tests.test_memory_format_ops_pass_utils import (
    SimpleCloneChannelsLastModule,
)
from torch.export import export
from torch.fx import GraphModule
from torch.testing import FileCheck
from torch.testing._internal.common_utils import TestCase


class TestRemoveCloneOpsTransform(TestCase):
    # Clone ops can appear as either aten.clone or _clone_dim_order depending on the _skip_dim_order flag.
    # _skip_dim_order=True tests aten.clone
    # _skip_dim_order=False tests _clone_dim_order.
    CLONE_OP_CASES = [
        (True, "executorch_exir_dialects_edge__ops_aten_clone_default"),
        (
            False,
            "executorch_exir_dialects_edge__ops_dim_order_ops__clone_dim_order_default",
        ),
    ]

    def test_dq_clone_q_linear(self):
        """
        Test RemoveCloneOpsTransform on a graph with d/q -> clone -> q -> linear pattern

        Before: Should contain all nodes
        After: Should only have the linear operation
        """

        # Create a graph module directly with the pattern: quant -> clone -> dequant -> fp linear
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)

            def forward(self, x):
                # This will be replaced with our custom graph
                return self.linear(x)

        # Create a module instance
        module = TestModule()

        # Create a new graph with our desired pattern
        graph = torch.fx.Graph()

        # Add placeholders
        input_node = graph.placeholder("x")

        # Create nodes for our pattern: quant -> clone -> dequant -> fp linear
        # Constants for quantization parameters
        scale = graph.create_node(
            "call_function", torch.tensor, args=([0.1],), kwargs={}
        )
        zero_point = graph.create_node(
            "call_function", torch.tensor, args=([0],), kwargs={}
        )

        # Dequantize node
        dequant_node = graph.create_node(
            "call_function",
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            args=(input_node, scale, zero_point, torch.int8),
            kwargs={},
        )

        # Clone node.
        # Use Edge op as this is an executorch pass
        clone_node = graph.create_node(
            "call_function",
            exir_ops.edge.aten.clone.default,
            args=(dequant_node,),
            kwargs={},
        )

        # Quantize node
        quant_node = graph.create_node(
            "call_function",
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            args=(clone_node, scale, zero_point, torch.int8),
            kwargs={},
        )

        # Linear node (using the module's linear layer)
        # Technically, should use quantized weight and bias
        # but we are just inspecting graph patterns in this test
        weight = graph.create_node("get_attr", "linear.weight")
        bias = graph.create_node("get_attr", "linear.bias")
        linear_node = graph.create_node(
            "call_function",
            torch.nn.functional.linear,
            args=(quant_node, weight, bias),
            kwargs={},
        )

        # Output
        graph.output(linear_node)

        # Create a GraphModule with our custom graph
        gm = GraphModule(module, graph)

        # Verify we have the expected nodes before transformation using FileCheck
        FileCheck().check(
            "torch.ops.quantized_decomposed.dequantize_per_tensor.default",
        ).check(
            "executorch_exir_dialects_edge__ops_aten_clone_default",
        ).check(
            "torch.ops.quantized_decomposed.quantize_per_tensor.default",
        ).check(
            "torch._C._nn.linear",
        ).run(
            gm.code
        )

        # Apply the transform
        transformed_gm = RemoveCloneOpsTransform()(gm).graph_module

        # Verify the dq -> clone -> q pattern is removed and linear op is still present using FileCheck
        FileCheck().check_not(
            "executorch_exir_dialects_edge__ops_aten_clone_default"
        ).check_not("quantized_decomposed.dequantize_per_tensor.default").check_not(
            "quantized_decomposed.quantize_per_tensor.default"
        ).check_count(
            "torch._C._nn.linear",
            1,
            exactly=True,
        ).run(
            transformed_gm.code
        )

    def test_clone_channels_last_survives(self):
        """Verify clone ops that modify memory_format are preserved by RemoveCloneOpsTransform."""

        for skip_dim_order, clone_op_str in self.CLONE_OP_CASES:
            model = SimpleCloneChannelsLastModule()
            x = torch.randn(3, 4, 5, 6).to(memory_format=torch.contiguous_format)

            exported = export(model.eval(), (x,), strict=True)
            before_epm = to_edge(
                exported,
                compile_config=EdgeCompileConfig(_skip_dim_order=skip_dim_order),
            )

            updated_epm = before_epm.transform([RemoveCloneOpsTransform()])

            FileCheck().check_count(clone_op_str, 1, exactly=True).run(
                updated_epm.exported_program().graph_module.code
            )

            expected = before_epm.exported_program().module()(x)
            actual = updated_epm.exported_program().module()(x)
            assert torch.allclose(actual, expected)
            assert is_channel_last_dim_order(actual)

    def test_clone_identity_removed(self):
        """Verify identity clone ops are removed by RemoveCloneOpsTransform."""

        for skip_dim_order, clone_op_str in self.CLONE_OP_CASES:
            model = SimpleCloneChannelsLastModule()
            x = torch.randn(3, 4, 5, 6).to(memory_format=torch.channels_last)

            exported = export(model.eval(), (x,), strict=True)
            before_epm = to_edge(
                exported,
                compile_config=EdgeCompileConfig(_skip_dim_order=skip_dim_order),
            )

            FileCheck().check_count(clone_op_str, 1, exactly=True).run(
                before_epm.exported_program().graph_module.code
            )

            updated_epm = before_epm.transform([RemoveCloneOpsTransform()])

            FileCheck().check_not(clone_op_str).run(
                updated_epm.exported_program().graph_module.code
            )

            expected = before_epm.exported_program().module()(x)
            actual = updated_epm.exported_program().module()(x)
            assert torch.allclose(actual, expected)
            assert is_channel_last_dim_order(actual)


if __name__ == "__main__":
    unittest.main()

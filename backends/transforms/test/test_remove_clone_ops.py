# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.transforms.remove_clone_ops import RemoveCloneOpsTransform
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx import GraphModule
from torch.testing import FileCheck
from torch.testing._internal.common_utils import TestCase


class TestRemoveCloneOpsTransform(TestCase):
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


if __name__ == "__main__":
    unittest.main()

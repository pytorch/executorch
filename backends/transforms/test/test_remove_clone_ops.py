# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.transforms.remove_clone_ops import RemoveCloneOpsTransform
from torch.fx import GraphModule
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

        # Clone node
        clone_node = graph.create_node(
            "call_function",
            torch.ops.aten.clone.default,
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

        # Print the graph before transformation
        print("Before transformation:")
        print(gm.graph)

        # Check node counts before transformation
        node_counts_before = {}
        for node in gm.graph.nodes:
            if node.op == "call_function":
                target_name = str(node.target)
                if target_name not in node_counts_before:
                    node_counts_before[target_name] = 0
                node_counts_before[target_name] += 1

        # Verify we have the expected nodes before transformation
        self.assertIn(str(torch.ops.aten.clone.default), node_counts_before)
        self.assertIn(
            str(torch.ops.quantized_decomposed.quantize_per_tensor.default),
            node_counts_before,
        )
        self.assertIn(
            str(torch.ops.quantized_decomposed.dequantize_per_tensor.default),
            node_counts_before,
        )
        self.assertIn(str(torch.nn.functional.linear), node_counts_before)

        # Apply the transform
        transformed_gm = RemoveCloneOpsTransform()(gm).graph_module

        # Print the graph after transformation
        print("After transformation:")
        print(transformed_gm.graph)

        # Check node counts after transformation
        node_counts_after = {}
        for node in transformed_gm.graph.nodes:
            if node.op == "call_function":
                target_name = str(node.target)
                if target_name not in node_counts_after:
                    node_counts_after[target_name] = 0
                node_counts_after[target_name] += 1

        # Verify the dq -> clone -> q pattern is removed
        self.assertNotIn(str(torch.ops.aten.clone.default), node_counts_after)
        self.assertNotIn(
            str(torch.ops.quantized_decomposed.dequantize_per_tensor.default),
            node_counts_after,
        )
        self.assertNotIn(
            str(torch.ops.quantized_decomposed.quantize_per_tensor.default),
            node_counts_after,
        )

        # Verify the linear op is still present
        self.assertIn(str(torch.nn.functional.linear), node_counts_after)


if __name__ == "__main__":
    unittest.main()

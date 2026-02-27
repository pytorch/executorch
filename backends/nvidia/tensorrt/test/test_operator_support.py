# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for TensorRT operator support functionality."""

import unittest

import torch


class OperatorSupportTest(unittest.TestCase):
    """Tests for TensorRTOperatorSupport functionality."""

    def test_get_op_name_for_add(self) -> None:
        """Test that TensorRTOperatorSupport correctly identifies add.Tensor."""
        from executorch.backends.nvidia.tensorrt.partitioner.operator_support import (
            TensorRTOperatorSupport,
        )
        from executorch.exir import to_edge
        from torch.export import export

        class AddModel(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

        model = AddModel()
        example_inputs = (torch.randn(2, 3), torch.randn(2, 3))
        exported = export(model, example_inputs)
        edge_program = to_edge(exported).exported_program()

        # Create an instance to test the methods
        op_support = TensorRTOperatorSupport()

        for node in edge_program.graph_module.graph.nodes:
            if node.op == "call_function" and "add" in node.name:
                full_op_name = op_support._get_op_name(node)
                formatted_name = op_support._remove_namespace(full_op_name)
                self.assertEqual(formatted_name, "add.Tensor")
                break
        else:
            self.fail("Could not find add node in graph")

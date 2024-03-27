# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.exir import to_edge
from executorch.exir.print_program import inspect_node
from torch.export import export


class TestPrintProgram(unittest.TestCase):
    def test_inspect_node(self) -> None:
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(32, 32, 1)
                self.conv2 = torch.nn.Conv2d(32, 32, 1)
                self.conv3 = torch.nn.Conv2d(32, 32, 1)
                self.gelu = torch.nn.GELU()

            def forward(self, x: torch.Tensor):
                a = self.conv1(x)
                b = self.conv2(a)
                c = self.conv3(a + b)
                return self.gelu(c)

        class WrapModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.test_model = TestModel()

            def forward(self, x):
                return self.test_model(x)

        warp_model = WrapModule()
        example_inputs = (torch.rand(1, 32, 16, 16),)

        exir_exported_program = to_edge(export(warp_model, example_inputs))
        number_of_stack_trace = 0
        for node in exir_exported_program.exported_program().graph.nodes:
            node_info = inspect_node(
                exir_exported_program.exported_program().graph, node
            )
            self.assertRegex(node_info, r".*-->.*")
            if "stack_trace" in node.meta:
                self.assertRegex(
                    node_info, r".*Traceback \(most recent call last\)\:.*"
                )
                number_of_stack_trace = number_of_stack_trace + 1
        self.assertGreaterEqual(number_of_stack_trace, 1)

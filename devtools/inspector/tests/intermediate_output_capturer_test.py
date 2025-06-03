# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from executorch.devtools.inspector._intermediate_output_capturer import (
    IntermediateOutputCapturer,
)

from executorch.exir import EdgeCompileConfig, EdgeProgramManager, to_edge
from torch.export import export, ExportedProgram
from torch.fx import GraphModule


class TestIntermediateOutputCapturer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class TestModule(nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()
                self.conv = nn.Conv2d(
                    in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1
                )
                self.conv.weight = nn.Parameter(
                    torch.tensor(
                        [[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]]]
                    )
                )
                self.conv.bias = nn.Parameter(torch.tensor([0.0]))

                self.linear = nn.Linear(in_features=4, out_features=2)
                self.linear.weight = nn.Parameter(
                    torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
                )
                self.linear.bias = nn.Parameter(torch.tensor([0.0, 0.0]))
                self.bias = nn.Parameter(torch.tensor([0.5, -0.5]), requires_grad=False)
                self.scale = nn.Parameter(torch.tensor([2.0, 0.5]), requires_grad=False)

            def forward(self, x):
                x = self.conv(x)
                x = x.view(x.size(0), -1)
                x = self.linear(x)
                x = x + self.bias
                x = x - 0.1
                x = x * self.scale
                x = x / (self.scale + 1.0)
                x = F.relu(x)
                x = torch.sigmoid(x)
                x1, x2 = torch.split(x, 1, dim=1)
                return x1, x2

        cls.model = TestModule()
        cls.input = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)
        cls.aten_model: ExportedProgram = export(cls.model, (cls.input,), strict=True)
        cls.edge_program_manager: EdgeProgramManager = to_edge(
            cls.aten_model, compile_config=EdgeCompileConfig(_check_ir_validity=True)
        )
        cls.graph_module: GraphModule = cls.edge_program_manager._edge_programs[
            "forward"
        ].module()
        cls.capturer = IntermediateOutputCapturer(cls.graph_module)
        cls.intermediate_outputs = cls.capturer.run_and_capture(cls.input)

    def test_keying_with_debug_handle_tuple(self):
        for key in self.intermediate_outputs.keys():
            self.assertIsInstance(key, tuple)

    def test_tensor_cloning_and_detaching(self):
        for output in self.intermediate_outputs.values():
            if isinstance(output, torch.Tensor):
                self.assertFalse(output.requires_grad)
                self.assertTrue(output.is_leaf)

    def test_placeholder_nodes_are_skipped(self):
        for node in self.graph_module.graph.nodes:
            if node.op == "placeholder":
                self.assertNotIn(
                    node.meta.get("debug_handle"), self.intermediate_outputs
                )

    def test_multiple_outputs_capture(self):
        outputs = self.capturer.run_and_capture(self.input)
        for output in outputs.values():
            if isinstance(output, tuple):
                self.assertEqual(len(output), 2)
                for part in output:
                    self.assertIsInstance(part, torch.Tensor)

    def test_capture_correct_outputs(self):
        expected_outputs_with_handles = {
            (10,): torch.tensor([[[[7.7000, 6.7000], [4.7000, 3.7000]]]]),
            (11,): torch.tensor([[7.7000, 6.7000, 4.7000, 3.7000]]),
            (12,): torch.tensor(
                [[0.1000, 0.5000], [0.2000, 0.6000], [0.3000, 0.7000], [0.4000, 0.8000]]
            ),
            (13,): torch.tensor([[5.0000, 14.1200]]),
            (14,): torch.tensor([[5.5000, 13.6200]]),
            (15,): torch.tensor([[5.4000, 13.5200]]),
            (16,): torch.tensor([[10.8000, 6.7600]]),
            (17,): torch.tensor([3.0000, 1.5000]),
            (18,): torch.tensor([[3.6000, 4.5067]]),
            (19,): torch.tensor([[3.6000, 4.5067]]),
            (20,): torch.tensor([[0.9734, 0.9891]]),
            (21,): [torch.tensor([[0.9734]]), torch.tensor([[0.9891]])],
        }
        self.assertEqual(
            len(self.intermediate_outputs), len(expected_outputs_with_handles)
        )

        for debug_handle, expected_output in expected_outputs_with_handles.items():
            actual_output = self.intermediate_outputs.get(debug_handle)
            self.assertIsNotNone(actual_output)
            if isinstance(expected_output, list):
                self.assertIsInstance(actual_output, list)
                self.assertEqual(len(actual_output), len(expected_output))
                for actual, expected in zip(actual_output, expected_output):
                    self.assertTrue(
                        torch.allclose(actual, expected, rtol=1e-4, atol=1e-5)
                    )
            else:
                self.assertTrue(
                    torch.allclose(actual_output, expected_output, rtol=1e-4, atol=1e-5)
                )

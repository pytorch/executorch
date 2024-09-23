# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import unittest

import torch
import torch._dynamo

from executorch.exir import to_edge

from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)
from torch.export._trace import _export
from torch.export.experimental import _export_forward_backward
from torch.export.exported_program import OutputKind


class TestJointGraph(unittest.TestCase):
    def test_joint_graph(self) -> None:
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)
                self.linear_no_train = torch.nn.Linear(3, 3)
                for param in self.linear_no_train.parameters():
                    param.requires_grad = False
                self.loss = torch.nn.CrossEntropyLoss()

            def forward(self, x, y):
                return self.loss(self.linear_no_train(self.linear(x)).softmax(dim=0), y)

        m = Module()
        example_inputs = (torch.ones(3), torch.tensor([1.0, 0.0, 0.0]))
        m(*example_inputs)
        ep = _export(m, example_inputs, pre_dispatch=True)
        joint_ep = _export_forward_backward(ep)
        edge = to_edge(joint_ep)

        output_node = None
        for node in edge.exported_program().graph.nodes:
            if node.op == "output":
                output_node = node
                break

        orig_outputs = len(output_node.args[0])

        et = edge.to_executorch()

        weight_output_specs = [
            spec
            for spec in et.exported_program().graph_signature.output_specs
            if spec.kind == OutputKind.TOKEN
        ]

        output_node = None
        for node in et.exported_program().graph.nodes:
            if node.op == "output":
                output_node = node
                break

        weight_outputs = len(output_node.args[0])

        # make sure 2 new outputs are added to both the node and the spec
        self.assertEqual(len(weight_output_specs), 2)  # linear layer weight and bias
        self.assertEqual(
            weight_outputs - orig_outputs, 2
        )  # linear layer weight and bias

        # assert that the weight and bias have proper data_buffer_idx and allocation_info
        self.assertEqual(
            et.executorch_program.execution_plan[0]  # pyre-ignore
            .values[0]
            .val.data_buffer_idx,
            1,
        )
        self.assertEqual(
            et.executorch_program.execution_plan[0]  # pyre-ignore
            .values[1]
            .val.data_buffer_idx,
            2,
        )
        self.assertEqual(
            et.executorch_program.execution_plan[0]  # pyre-ignore
            .values[0]
            .val.allocation_info.memory_offset_low,
            0,
        )
        self.assertEqual(
            et.executorch_program.execution_plan[0]  # pyre-ignore
            .values[1]
            .val.allocation_info.memory_offset_low,
            48,
        )

        loss = m(*example_inputs)
        loss.backward()
        et_mod = _load_for_executorch_from_buffer(et.buffer)
        et_outputs = et_mod.forward(
            example_inputs
        )  # ET outputs are [loss, grads, weights]

        self.assertTrue(torch.allclose(loss, et_outputs[0]))
        self.assertTrue(
            torch.allclose(m.linear.weight.grad, et_outputs[1])  # pyre-ignore[6]
        )
        self.assertTrue(torch.allclose(m.linear.bias.grad, et_outputs[2]))
        self.assertTrue(torch.allclose(m.linear.weight, et_outputs[3]))
        self.assertTrue(torch.allclose(m.linear.bias, et_outputs[4]))

        self.assertEqual(
            len(et.executorch_program.execution_plan), 4
        )  # forward + 2 training metadata functions

        # gradient outputs start at index 1
        self.assertEqual(
            et.executorch_program.execution_plan[1]  # pyre-ignore
            .values[0]
            .val.int_val,
            1,
        )

        self.assertEqual(
            et.executorch_program.execution_plan[2]  # pyre-ignore
            .values[0]
            .val.string_val,
            "linear.weight",
        )

        # parameter outputs start at index 3
        self.assertEqual(
            et.executorch_program.execution_plan[3]  # pyre-ignore
            .values[0]
            .val.int_val,
            3,
        )

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn
from executorch.exir import to_edge
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.passes import MemoryPlanningPass


class TestCat(nn.Module):
    def forward(self, x, y, z):
        empty = torch.empty((0, 6))
        return torch.cat([empty, x, empty, y, z, empty])

    def get_example_inputs(self):
        return (torch.rand(5, 6), torch.rand(5, 6), torch.rand(5, 6))


class TestPruneEmptyTensors(unittest.TestCase):
    def test_empty_tensor_removed_from_cat(self) -> None:
        model = TestCat()
        model.eval()
        example_inputs = model.get_example_inputs()
        ep = torch.export.export(model, example_inputs, strict=True)
        etpm = to_edge(ep).to_executorch(
            config=ExecutorchBackendConfig(
                remove_view_copy=False,
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            ),
        )

        for node in etpm.exported_program().graph_module.graph.nodes:
            if node.target in [
                exir_ops.edge.aten.cat.default,
                torch.ops.aten.cat.default,
            ]:
                self.assertTrue(len(node.all_input_nodes) == 3)
                for input_arg in node.all_input_nodes:
                    tensor_val = input_arg.meta["val"]
                    self.assertTrue(tensor_val.numel() != 0)

        actual = etpm.exported_program().module()(*example_inputs)

        reference = model(*example_inputs)

        self.assertTrue(torch.allclose(actual, reference))

    def test_cat_removed_all_empty(self) -> None:
        model = TestCat()
        model.eval()
        example_inputs = (torch.empty((0, 6)), torch.empty((0, 6)), torch.empty((0, 6)))
        ep = torch.export.export(model, example_inputs, strict=True)
        etpm = to_edge(ep).to_executorch(
            config=ExecutorchBackendConfig(
                remove_view_copy=False,
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            ),
        )

        for node in etpm.exported_program().graph_module.graph.nodes:
            self.assertFalse(
                node.target
                in [exir_ops.edge.aten.cat.default, torch.ops.aten.cat.default]
            )

        actual = etpm.exported_program().module()(*example_inputs)

        reference = model(*example_inputs)

        self.assertTrue(torch.allclose(actual, reference))

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
from executorch.exir.passes.prune_empty_tensors_pass import PruneEmptyTensorsPass


class TestCat(nn.Module):
    __test__ = False

    def forward(self, x, y, z):
        empty = torch.empty((0, 6))
        return torch.cat([empty, x, empty, y, z, empty])

    def get_example_inputs(self):
        return (torch.rand(5, 6), torch.rand(5, 6), torch.rand(5, 6))


class UnbackedCatModel(nn.Module):
    __test__ = False

    def forward(self, x, routing_weights):
        expert_mask = routing_weights > 0.5
        indices = torch.nonzero(expert_mask)
        selected = x[indices[:, 0]]
        return torch.cat([x[:0], selected], dim=0)


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

    def test_unbacked_symint_numel_does_not_crash(self) -> None:
        """PruneEmptyTensorsPass must not crash on tensors whose numel() is an
        unbacked SymInt (e.g. from torch.nonzero in MoE expert routing).
        The pass should conservatively keep such tensors."""
        model = UnbackedCatModel()
        model.eval()
        ep = torch.export.export(
            model,
            (torch.randn(4, 8), torch.randn(4, 2)),
            strict=False,
        )
        result = PruneEmptyTensorsPass()(ep.graph_module)
        found_cat = False
        for node in result.graph_module.graph.nodes:
            if node.target == torch.ops.aten.cat.default:
                found_cat = True
                cat_inputs = node.args[0]
                self.assertEqual(len(cat_inputs), 1)
                val = cat_inputs[0].meta["val"]
                self.assertGreater(len(val.shape), 0)
        self.assertTrue(found_cat)

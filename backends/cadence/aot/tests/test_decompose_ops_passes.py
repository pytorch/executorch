# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Union

import torch
from executorch.backends.cadence.aot.decompose_ops import DecomposeAtenApproxGeluPass
from executorch.backends.cadence.aot.graph_builder import single_op_builder
from executorch.backends.cadence.aot.pass_utils import count_node
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass


class TestDecomposeOpsPasses(unittest.TestCase):
    def assertTargetCountEqual(
        self,
        graph_module: torch.fx.GraphModule,
        target: Union[EdgeOpOverload, str],
        expected_count: int,
    ) -> None:
        """Helper function to check the number of nodes with a given target."""
        actual_count = count_node(graph_module, target)
        self.assertEqual(
            actual_count,
            expected_count,
            f"{target} count mismatch for graph {graph_module}",
        )

    def assertTargetCountsEqual(
        self,
        graph_module: torch.fx.GraphModule,
        targets_and_counts: list[tuple[Union[EdgeOpOverload, str], int]],
    ) -> None:
        """Helper function to check the number of nodes of all types for a given target."""
        for target, expected_count in targets_and_counts:
            self.assertTargetCountEqual(graph_module, target, expected_count)

    def test_decompose_aten_approximate_gelu(self) -> None:
        inputs = torch.randn(2, 1, 64)

        gm = single_op_builder(
            placeholders=(inputs,),
            op=exir_ops.edge.aten.gelu.default,
            args=(inputs,),
            kwargs={"approximate": "tanh"},
        )
        gm = ExportPass().call(gm).graph_module

        p = DecomposeAtenApproxGeluPass()
        graph_after_passes = p.call(gm).graph_module

        # Assert that aten.gelu op was decomposed
        self.assertEqual(
            count_node(
                graph_after_passes,
                exir_ops.edge.aten.gelu.default,
            ),
            0,
        )

        # The decomposition should have one tanh, 2 add and 6 mul
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.tanh.default),
            1,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.add.Tensor),
            2,
        )
        self.assertEqual(
            count_node(graph_after_passes, exir_ops.edge.aten.mul.Tensor),
            6,
        )

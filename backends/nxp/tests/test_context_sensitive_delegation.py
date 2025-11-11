# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch

from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters import (
    ViewCopyConverter,
)
from executorch.backends.nxp.tests.executorch_pipeline import to_quantized_edge_program
from executorch.backends.nxp.tests.executors import graph_contains_any_of_ops
from executorch.exir.dialects._ops import ops as exir_ops


class SingleViewCopyModule(torch.nn.Module):
    def __init__(self, new_shape: list[int]):
        super().__init__()
        self.new_shape = new_shape

    def forward(self, x):
        return torch.reshape(x, self.new_shape)


class TestContextSensitiveDelegation(unittest.TestCase):
    __test__ = False  # Prevent interfering with PyTest tests.

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(23)
        np.random.seed(42)

    def test_single_view_copy_partition(self):
        input_shape = (2, 10)
        module = SingleViewCopyModule([1, 20])

        ep = to_quantized_edge_program(module, input_shape).exported_program()

        # Make sure the `view_copy` was not delegated.
        assert graph_contains_any_of_ops(
            ep.graph, [exir_ops.edge.aten.view_copy.default]
        )
        assert not any("delegate" in n.name for n in ep.graph.nodes)

    def test_single_view_copy_partition__forced_delegation(self):
        input_shape = (2, 10)
        module = SingleViewCopyModule([1, 20])

        def _supported_partitioning(*_):
            return True

        # Replace the partition support check function, to accept anything.
        original_supports_partitioning_result = (
            ViewCopyConverter.supports_partitioning_result
        )
        ViewCopyConverter.supports_partitioning_result = _supported_partitioning

        with self.assertRaises(RuntimeError) as e:
            to_quantized_edge_program(module, input_shape).exported_program()
        assert (
            str(e.exception)
            == "Model converted with neutron-converter does not contain a NeutronGraph node."
        )

        # Return to the original partition support check function.
        ViewCopyConverter.supports_partitioning_result = (
            original_supports_partitioning_result
        )

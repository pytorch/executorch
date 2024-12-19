# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


import executorch.backends.cadence.aot.ops_registrations  # noqa
import torch
from executorch.backends.cadence.aot.graph_builder import (
    GraphBuilder,
    single_op_builder,
)
from executorch.backends.cadence.aot.pass_utils import count_node
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass
from later.unittest import TestCase


class TestGraphBuilder(TestCase):
    def test_graph_with_single_im2row(self) -> None:
        # Create a graph with a single im2row node.
        builder = GraphBuilder()
        x = builder.placeholder("x", torch.randn(1, 3, 224, 224))
        pad_value = builder.placeholder("pad", torch.randn(1))
        channels_last = False
        im2row = builder.call_operator(
            exir_ops.edge.cadence.im2row.default,
            # pyre-ignore
            (
                x,
                (2, 2),
                (1, 1),
                (0, 0),
                (1, 1),
                pad_value,
                channels_last,
            ),
        )
        builder.output([im2row])
        gm = builder.get_graph_module()
        # Check if graph module is valid by running exportpass on it.
        gm = ExportPass().call(gm).graph_module

        # Check graph has a single im2row node.
        self.assertEqual(len([gm.graph.nodes]), 1)
        self.assertEqual(count_node(gm, exir_ops.edge.cadence.im2row.default), 1)


class TestSingleOpBuilderUtility(TestCase):
    def test_graph_with_single_im2row(self) -> None:
        # Create a graph with a single im2row node.
        x = torch.randn(1, 3, 224, 224)
        pad_value = torch.randn(1)
        channels_last = False
        gm = single_op_builder(
            (x, pad_value),
            exir_ops.edge.cadence.im2row.default,
            (
                x,
                (2, 2),
                (1, 1),
                (0, 0),
                (1, 1),
                pad_value,
                channels_last,
            ),
        )
        # Check if graph module is valid by running exportpass on it.
        gm = ExportPass().call(gm).graph_module

        # Check graph has a single im2row node.
        self.assertEqual(len([gm.graph.nodes]), 1)
        self.assertEqual(count_node(gm, exir_ops.edge.cadence.im2row.default), 1)

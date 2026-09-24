# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import executorch.backends.cadence.aot.ops_registrations  # noqa
import executorch.backends.fused_quant.ops  # noqa
import torch
from executorch.backends.fused_quant.optimization_passes.hoist_activations import (
    HoistActivations,
)
from executorch.backends.fused_quant.test.helpers import create_per_tensor_qparams
from executorch.backends.test.program_builder import ProgramBuilder
from executorch.exir.dialects._ops import ops as exir_ops


def _build_relu_after_permutes_graph() -> torch.fx.GraphModule:
    """Build a graph: inp -> permute -> permute -> relu -> output.

    Quant params are lifted front placeholders (matching production), so they
    are graph inputs rather than inline body nodes fed to the relu.
    """
    inp = torch.randn(1, 4, 8, 8).to(torch.int8)

    builder = ProgramBuilder()
    node = builder.placeholder("inp", inp)

    # Quant params are lifted constants and thus front placeholders.
    inp_qparams = create_per_tensor_qparams(builder)
    out_qparams = create_per_tensor_qparams(builder)

    node = builder.call_operator(
        op=exir_ops.edge.aten.permute_copy.default,
        args=(node, [0, 2, 3, 1]),
    )
    node = builder.call_operator(
        op=exir_ops.edge.aten.permute_copy.default,
        args=(node, [0, 3, 1, 2]),
    )

    relu = builder.call_operator(
        op=exir_ops.edge.fused_quant.relu.default,
        args=(
            node,
            *inp_qparams,
            *out_qparams,
        ),
    )

    builder.output([relu])
    graph_module = builder.get_program().graph_module

    return graph_module


class HoistActivationsTest(unittest.TestCase):
    """Tests for the HoistActivations optimization pass."""

    def test_relu_moves_above_permutes(self) -> None:
        """Relu is hoisted above a chain of permute_copy ops.

        Before: inp -> permute -> permute -> relu -> output
        After:  inp -> relu -> permute -> permute -> output

        The relu never moves down; each single-user permute is moved forward to
        sit after the relu. The relu's qparams are lifted front placeholders,
        so they are trivially already before the relu and do not move.
        """
        graph_module = _build_relu_after_permutes_graph()

        result = HoistActivations().call(graph_module)

        self.assertTrue(result.modified)

        graph = result.graph_module.graph
        relu_nodes = graph.find_nodes(
            op="call_function", target=exir_ops.edge.fused_quant.relu.default
        )
        permute_nodes = graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.permute_copy.default
        )

        self.assertEqual(len(relu_nodes), 1, "Expected 1 relu node")
        self.assertEqual(len(permute_nodes), 2, "Expected 2 permute nodes")

        all_nodes = list(graph.nodes)
        relu_idx = all_nodes.index(relu_nodes[0])

        # Relu should come before both permutes
        for permute_node in permute_nodes:
            permute_idx = all_nodes.index(permute_node)
            self.assertLess(
                relu_idx, permute_idx, "Relu should come before permute nodes"
            )

    def test_convergence(self) -> None:
        """First invocation modifies the graph; second invocation is a no-op."""
        graph_module = _build_relu_after_permutes_graph()

        opt_pass = HoistActivations()

        first_result = opt_pass.call(graph_module)
        self.assertTrue(first_result.modified)

        second_result = opt_pass.call(first_result.graph_module)
        self.assertFalse(second_result.modified)

    def test_no_hoist_when_data_movement_has_multiple_users(self) -> None:
        """Should not hoist when the data-movement op feeding the activation
        has more than one user.

        If permute feeds both relu and another consumer, hoisting would make
        the other consumer see relu(permute(inp)) instead of permute(inp),
        silently changing its semantics.
        """
        inp = torch.randn(1, 4, 8, 8).to(torch.int8)

        builder = ProgramBuilder()
        node = builder.placeholder("inp", inp)
        inp_qparams = create_per_tensor_qparams(builder)
        out_qparams = create_per_tensor_qparams(builder)
        permute = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(node, [0, 2, 3, 1]),
        )
        # A second consumer of permute (besides relu below).
        other = builder.call_operator(
            op=exir_ops.edge.aten.abs.default,
            args=(permute,),
        )
        relu = builder.call_operator(
            op=exir_ops.edge.fused_quant.relu.default,
            args=(permute, *inp_qparams, *out_qparams),
        )
        builder.output([relu, other])
        graph_module = builder.get_program().graph_module

        result = HoistActivations().call(graph_module)

        self.assertFalse(
            result.modified,
            "Should not hoist when the data-movement op has multiple users",
        )

    def test_requantizing_relu_keeps_output_dtype(self) -> None:
        """A requantizing relu (int8 -> int16) keeps its output dtype on hoist.

        The relu's meta['val'] dtype is int16 (its output), not int8 (its
        input). After hoisting, the relu must still report int16, and each
        permute it now feeds must be retyped from int8 to int16 since it now
        carries the relu's requantized output.
        """
        inp = torch.randn(1, 4, 8, 8).to(torch.int8)

        builder = ProgramBuilder()
        node = builder.placeholder("inp", inp)
        inp_qparams = create_per_tensor_qparams(builder, dtype=torch.int8)
        out_qparams = create_per_tensor_qparams(builder, dtype=torch.int16)
        node = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(node, [0, 2, 3, 1]),
        )
        node = builder.call_operator(
            op=exir_ops.edge.aten.permute_copy.default,
            args=(node, [0, 3, 1, 2]),
        )
        relu = builder.call_operator(
            op=exir_ops.edge.fused_quant.relu.default,
            args=(node, *inp_qparams, *out_qparams),
        )
        builder.output([relu])
        graph_module = builder.get_program().graph_module

        relu_before = graph_module.graph.find_nodes(
            op="call_function", target=exir_ops.edge.fused_quant.relu.default
        )[0]
        self.assertEqual(relu_before.meta["val"].dtype, torch.int16)

        result = HoistActivations().call(graph_module)
        self.assertTrue(result.modified)

        graph = result.graph_module.graph
        relu_after = graph.find_nodes(
            op="call_function", target=exir_ops.edge.fused_quant.relu.default
        )[0]
        self.assertEqual(
            relu_after.meta["val"].dtype,
            torch.int16,
            "Hoisted relu must keep its requantized output dtype (int16), not "
            "the int8 dtype of its new input",
        )

        # The permutes now carry the relu's requantized output, so their
        # meta['val'] dtype must follow (int16), not the stale int8 from before.
        for permute_node in graph.find_nodes(
            op="call_function", target=exir_ops.edge.aten.permute_copy.default
        ):
            self.assertEqual(
                permute_node.meta["val"].dtype,
                torch.int16,
                "Permute carrying the hoisted relu's output must be retyped to "
                "the relu's output dtype",
            )

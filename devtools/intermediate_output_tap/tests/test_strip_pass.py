# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

import torch
from executorch.devtools.intermediate_output_tap._reducers import FULL_TENSOR, STATS
from executorch.devtools.intermediate_output_tap._selectors import select_by_op_type
from executorch.devtools.intermediate_output_tap._strip_pass import strip_taps_
from executorch.devtools.intermediate_output_tap._tap_pass import (
    find_tap_nodes,
    tap_intermediate_outputs_,
)
from executorch.exir import to_edge
from torch.export import export


class _MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(8, 8)
        self.l2 = torch.nn.Linear(8, 4)

    def forward(self, x):
        return self.l2(self.l1(x).relu())


def _tapped_edge(reducer):
    ep = export(_MLP(), (torch.randn(2, 8),), strict=True)
    ep_t, specs = tap_intermediate_outputs_(
        ep, (select_by_op_type("aten.linear.default"), reducer)
    )
    return to_edge(ep_t), specs


class StripPassTest(unittest.TestCase):
    def test_strip_removes_all_tap_nodes_full_tensor(self):
        edge, _ = _tapped_edge(FULL_TENSOR)
        # Pre-strip: tap nodes present.
        for method_name in edge.methods:
            ep = edge.exported_program(method_name)
            self.assertGreater(len(find_tap_nodes(ep.graph_module)), 0)

        strip_taps_(edge)

        # Post-strip: no tap nodes.
        for method_name in edge.methods:
            ep = edge.exported_program(method_name)
            self.assertEqual(len(find_tap_nodes(ep.graph_module)), 0)

    def test_strip_full_tensor_routes_source_to_output(self):
        edge, specs = _tapped_edge(FULL_TENSOR)
        strip_taps_(edge)
        # Output node should still have all the user outputs + tap outputs.
        for method_name in edge.methods:
            ep = edge.exported_program(method_name)
            outs = list(ep.graph_module.graph.output_node().args[0])
            # Original outputs + 2 linears tapped.
            self.assertGreaterEqual(len(outs), len(specs))

    def test_strip_stats_emits_subgraph(self):
        edge, _ = _tapped_edge(STATS)
        strip_taps_(edge)
        for method_name in edge.methods:
            ep = edge.exported_program(method_name)
            self.assertEqual(len(find_tap_nodes(ep.graph_module)), 0)
            # Some reduction op (amin/amax/mean) should now be in the graph.
            # Substring match because EdgeOpOverload's str() looks like
            # "<EdgeOpOverload: aten.amin.default>: schema = ..." (no clean
            # endswith).
            targets = {str(n.target) for n in ep.graph_module.graph.nodes}
            self.assertTrue(
                any(
                    "aten.amin" in t or "aten.amax" in t or "aten.mean" in t
                    for t in targets
                ),
                f"expected reducer ops in graph, got {targets}",
            )

    def test_strip_full_tensor_no_clone_added(self):
        """For FULL_TENSOR taps, strip should route the source intermediate
        directly to the output node — no clone op should sit between the
        source and the output."""
        edge, specs = _tapped_edge(FULL_TENSOR)

        # Capture each tap's source node name before strip.
        pre_strip_sources = {}
        for method_name in edge.methods:
            ep = edge.exported_program(method_name)
            pre_strip_sources[method_name] = {
                tap.args[0].name for tap in find_tap_nodes(ep.graph_module)
            }
            self.assertEqual(len(pre_strip_sources[method_name]), len(specs))

        strip_taps_(edge)

        for method_name in edge.methods:
            ep = edge.exported_program(method_name)
            print(ep)
            graph = ep.graph_module.graph
            output_args = graph.output_node().args[0]
            out_names = {n.name for n in output_args if hasattr(n, "name")}
            # Every tapped source should appear directly as an output, i.e.,
            # the strip pass routed it through with no clone
            self.assertTrue(
                pre_strip_sources[method_name].issubset(out_names),
                f"Expected tap sources {pre_strip_sources[method_name]} to appear "
                f"directly in output args {out_names}",
            )
            # And no clone ops should have been inserted by the strip pass.
            for n in graph.nodes:
                self.assertNotIn(
                    "clone",
                    str(n.target),
                    f"strip_taps_ inserted an unexpected clone op: {n.target}",
                )

    def test_strip_idempotent(self):
        edge, _ = _tapped_edge(FULL_TENSOR)
        strip_taps_(edge)
        # Second call should be a no-op.
        strip_taps_(edge)
        for method_name in edge.methods:
            ep = edge.exported_program(method_name)
            self.assertEqual(len(find_tap_nodes(ep.graph_module)), 0)

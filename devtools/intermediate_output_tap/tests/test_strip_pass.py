# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

import torch
from executorch.devtools.intermediate_output_tap._reducers import (
    DEFAULT_STATS,
    FULL_TENSOR,
    MIN_MAX_MEAN,
)
from executorch.devtools.intermediate_output_tap._selectors import (
    select_by_op_type,
)
from executorch.devtools.intermediate_output_tap._strip_pass import strip_taps_
from executorch.devtools.intermediate_output_tap._tap_pass import (
    find_tap_nodes,
    tap_intermediate_outputs,
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
    ep_t, specs = tap_intermediate_outputs(
        ep,
        selector=select_by_op_type("aten.linear.default"),
        reducer=reducer,
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

    def test_strip_min_max_mean_emits_subgraph(self):
        edge, specs = _tapped_edge(MIN_MAX_MEAN)
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

    def test_strip_default_stats_preserves_debug_handle(self):
        edge, specs = _tapped_edge(DEFAULT_STATS)
        # Take a known debug_handle from one of the tap specs.
        known_handles = {s.debug_handle for s in specs if s.debug_handle is not None}
        if not known_handles:
            self.skipTest("Test model produced no debug_handle on tap sources")

        strip_taps_(edge)

        post_handles: set = set()
        for method_name in edge.methods:
            ep = edge.exported_program(method_name)
            for n in ep.graph_module.graph.nodes:
                if n.meta.get("is_tap"):
                    post_handles.add(n.meta.get("debug_handle"))
        # At least one tapped debug handle should still be present.
        self.assertTrue(known_handles & post_handles)

    def test_strip_idempotent(self):
        edge, _ = _tapped_edge(FULL_TENSOR)
        strip_taps_(edge)
        # Second call should be a no-op.
        strip_taps_(edge)
        for method_name in edge.methods:
            ep = edge.exported_program(method_name)
            self.assertEqual(len(find_tap_nodes(ep.graph_module)), 0)

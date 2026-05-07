# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import copy
import unittest

import torch
from executorch.devtools.intermediate_output_tap._reducers import (
    DEFAULT_STATS,
    FULL_TENSOR,
)
from executorch.devtools.intermediate_output_tap._selectors import select_by_op_type
from executorch.devtools.intermediate_output_tap._tap_pass import (
    is_tap_node,
    tap_intermediate_outputs,
)
from torch.export import export
from torch.export.exported_program import OutputKind


class _MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(8, 16)
        self.l2 = torch.nn.Linear(16, 8)
        self.l3 = torch.nn.Linear(8, 4)

    def forward(self, x):
        return self.l3(self.l2(self.l1(x).relu()).relu())


def _export():
    return export(_MLP(), (torch.randn(2, 8),), strict=True)


class TapPassTest(unittest.TestCase):
    def test_inserts_tap_per_selected_node(self):
        ep = _export()
        ep_t, specs = tap_intermediate_outputs(
            ep,
            selector=select_by_op_type("aten.linear.default"),
            reducer=FULL_TENSOR,
        )
        # MLP has 3 linears.
        self.assertEqual(len(specs), 3)
        tap_nodes = [n for n in ep_t.graph_module.graph.nodes if is_tap_node(n)]
        self.assertEqual(len(tap_nodes), 3)

    def test_appends_user_outputs(self):
        ep = _export()
        original_user_outs = sum(
            1
            for s in ep.graph_signature.output_specs
            if s.kind == OutputKind.USER_OUTPUT
        )
        ep_t, specs = tap_intermediate_outputs(
            ep,
            selector=select_by_op_type("aten.linear.default"),
            reducer=FULL_TENSOR,
        )
        new_user_outs = sum(
            1
            for s in ep_t.graph_signature.output_specs
            if s.kind == OutputKind.USER_OUTPUT
        )
        self.assertEqual(new_user_outs, original_user_outs + len(specs))

    def test_output_indices_contiguous_after_user_outputs(self):
        ep = _export()
        original_user_outs = sum(
            1
            for s in ep.graph_signature.output_specs
            if s.kind == OutputKind.USER_OUTPUT
        )
        _, specs = tap_intermediate_outputs(
            ep,
            selector=select_by_op_type("aten.linear.default"),
            reducer=FULL_TENSOR,
        )
        for i, spec in enumerate(specs):
            self.assertEqual(spec.output_index, original_user_outs + i)

    def test_default_reducer_is_default_stats(self):
        ep = _export()
        _, specs = tap_intermediate_outputs(
            ep, selector=select_by_op_type("aten.linear.default")
        )
        for s in specs:
            self.assertEqual(s.reducer_name, DEFAULT_STATS.name)
            self.assertEqual(s.fields, DEFAULT_STATS.fields)

    def test_inplace_false_does_not_mutate_original(self):
        ep = _export()
        before_outs = len(list(ep.graph_module.graph.output_node().args[0]))
        before_specs = len(ep.graph_signature.output_specs)
        _ = tap_intermediate_outputs(
            ep, selector=select_by_op_type("aten.linear.default"), reducer=FULL_TENSOR
        )
        after_outs = len(list(ep.graph_module.graph.output_node().args[0]))
        after_specs = len(ep.graph_signature.output_specs)
        self.assertEqual(before_outs, after_outs)
        self.assertEqual(before_specs, after_specs)

    def test_max_taps(self):
        ep = _export()
        _, specs = tap_intermediate_outputs(
            ep,
            selector=select_by_op_type("aten.linear.default"),
            reducer=FULL_TENSOR,
            max_taps=2,
        )
        self.assertEqual(len(specs), 2)

    def test_idempotent_does_not_tap_taps(self):
        ep = _export()
        ep_once, specs1 = tap_intermediate_outputs(
            ep,
            selector=select_by_op_type("aten.linear.default"),
            reducer=FULL_TENSOR,
        )
        # Running again should not add NEW taps for our existing tap nodes.
        ep_twice, specs2 = tap_intermediate_outputs(
            ep_once,
            selector=select_by_op_type("aten.linear.default"),
            reducer=FULL_TENSOR,
        )
        # Same number of linears matched; tap.Tensor itself is excluded.
        self.assertEqual(len(specs2), len(specs1))

    def test_no_match_returns_empty_specs(self):
        ep = _export()
        ep_t, specs = tap_intermediate_outputs(
            ep,
            selector=select_by_op_type("aten.does.not.exist"),
            reducer=FULL_TENSOR,
        )
        self.assertEqual(specs, [])
        # Original graph signature is unchanged.
        self.assertEqual(
            len(ep_t.graph_signature.output_specs),
            len(ep.graph_signature.output_specs),
        )

    def test_skip_if_no_debug_handle(self):
        ep = _export()
        # Strip all debug handles to simulate a graph without them.
        ep_clean = copy.deepcopy(ep)
        for n in ep_clean.graph_module.graph.nodes:
            n.meta.pop("debug_handle", None)
        _, specs = tap_intermediate_outputs(
            ep_clean,
            selector=select_by_op_type("aten.linear.default"),
            reducer=FULL_TENSOR,
            skip_if_no_debug_handle=True,
        )
        self.assertEqual(specs, [])

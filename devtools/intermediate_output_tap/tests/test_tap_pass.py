# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

import torch
from executorch.devtools.intermediate_output_tap._reducers import FULL_TENSOR, STATS
from executorch.devtools.intermediate_output_tap._selectors import (
    select_all_call_function,
    select_by_op_type,
)
from executorch.devtools.intermediate_output_tap._tap_pass import (
    is_tap_node,
    tap_intermediate_outputs_,
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


class _RichModel(torch.nn.Module):
    """Model with multiple inputs/outputs AND a mutable buffer.

    Surfaces non-USER_OUTPUT entries (BUFFER_MUTATION) and a multi-leaf
    forward result, so we can check the tap pass preserves the full EP
    structure rather than just the trivial single-input/output case.
    """

    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(4, 4)
        self.l2 = torch.nn.Linear(4, 2)
        self.register_buffer("running", torch.zeros(4))

    def forward(self, x, y):
        z = self.l1(x + y)
        # In-place buffer mutation BUFFER_MUTATION OutputSpec.
        self.running.add_(z.sum(dim=0))
        # Tuple return multiple USER_OUTPUT OutputSpecs.
        return self.l2(z), z.sum()


def _export_rich():
    return export(
        _RichModel(),
        (torch.randn(3, 4), torch.randn(3, 4)),
        strict=True,
    )


_LINEAR_FULL_TENSOR_RULE = (select_by_op_type("aten.linear.default"), FULL_TENSOR)


class TapPassTest(unittest.TestCase):
    def test_inserts_tap_per_selected_node(self):
        ep = _export()
        ep_t, specs = tap_intermediate_outputs_(ep, _LINEAR_FULL_TENSOR_RULE)
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
        ep_t, specs = tap_intermediate_outputs_(ep, _LINEAR_FULL_TENSOR_RULE)
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
        _, specs = tap_intermediate_outputs_(ep, _LINEAR_FULL_TENSOR_RULE)
        for i, spec in enumerate(specs):
            self.assertEqual(spec.output_index, original_user_outs + i)

    def test_default_rules_use_stats_on_call_functions(self):
        ep = _export()
        _, specs = tap_intermediate_outputs_(ep)
        self.assertGreater(len(specs), 0)
        for s in specs:
            self.assertEqual(s.reducer_name, STATS.name)
            self.assertEqual(s.fields, STATS.fields)

    def test_mutates_original_and_returns_same_ep(self):
        ep = _export()
        before_outs = len(list(ep.graph_module.graph.output_node().args[0]))
        ep_returned, specs = tap_intermediate_outputs_(ep, _LINEAR_FULL_TENSOR_RULE)
        # Same object — in-place ops return self.
        self.assertIs(ep_returned, ep)
        # And the original was mutated.
        after_outs = len(list(ep.graph_module.graph.output_node().args[0]))
        self.assertEqual(after_outs, before_outs + len(specs))

    def test_max_taps(self):
        ep = _export()
        _, specs = tap_intermediate_outputs_(ep, _LINEAR_FULL_TENSOR_RULE, max_taps=2)
        self.assertEqual(len(specs), 2)

    def test_idempotent_does_not_tap_taps(self):
        ep = _export()
        _, specs1 = tap_intermediate_outputs_(ep, _LINEAR_FULL_TENSOR_RULE)
        # Running again should not add NEW taps for our existing tap nodes.
        _, specs2 = tap_intermediate_outputs_(ep, _LINEAR_FULL_TENSOR_RULE)
        # Same number of linears matched; tap.Tensor itself is excluded.
        self.assertEqual(len(specs2), len(specs1))

    def test_no_match_returns_empty_specs(self):
        ep = _export()
        before_specs = len(ep.graph_signature.output_specs)
        _, specs = tap_intermediate_outputs_(
            ep,
            (select_by_op_type("aten.does.not.exist"), FULL_TENSOR),
            error_on_empty=False,
        )
        self.assertEqual(specs, [])
        # Graph signature is unchanged.
        self.assertEqual(len(ep.graph_signature.output_specs), before_specs)

    def test_error_on_empty_raises(self):
        ep = _export()
        with self.assertRaises(ValueError):
            tap_intermediate_outputs_(
                ep,
                (select_by_op_type("aten.does.not.exist"), FULL_TENSOR),
            )

    def test_first_matching_rule_wins(self):
        # Both rules match aten.linear.default. The first rule is FULL_TENSOR;
        # the second is STATS. Every linear should land on FULL_TENSOR because
        # the loop in tap_intermediate_outputs_ stops at the first match.
        ep = _export()
        linear_sel = select_by_op_type("aten.linear.default")
        # Second rule selects everything (superset of linears).
        broad_sel = select_all_call_function()
        _, specs = tap_intermediate_outputs_(
            ep,
            [(linear_sel, FULL_TENSOR), (broad_sel, STATS)],
        )
        # Each linear must have FULL_TENSOR (first rule), not STATS.
        linear_specs = [s for s in specs if s.op_target.endswith("aten.linear.default")]
        self.assertEqual(len(linear_specs), 3)
        for s in linear_specs:
            self.assertEqual(s.reducer_name, FULL_TENSOR.name)
        # Non-linear nodes (relu, etc.) should have STATS from the second rule.
        non_linear_specs = [
            s for s in specs if not s.op_target.endswith("aten.linear.default")
        ]
        self.assertGreater(len(non_linear_specs), 0)
        for s in non_linear_specs:
            self.assertEqual(s.reducer_name, STATS.name)

    def test_tapspec_module_metadata_populated(self):
        ep = _export()
        _, specs = tap_intermediate_outputs_(ep, _LINEAR_FULL_TENSOR_RULE)
        # _MLP has self.l1, self.l2, self.l3 — all torch.nn.Linear instances.
        self.assertEqual(len(specs), 3)
        for s in specs:
            self.assertEqual(s.module_class, "Linear")
            self.assertIn(s.module_path, {"l1", "l2", "l3"})
        # All three module paths show up exactly once.
        self.assertEqual({s.module_path for s in specs}, {"l1", "l2", "l3"})

    def test_preserves_full_ep_structure_with_buffers_and_multi_output(self):
        # Rich EP: 2 user inputs, 1 buffer (input-side), several params,
        # 2 user outputs (tuple return). The pass should append USER_OUTPUTs
        # without disturbing anything else.
        ep = _export_rich()
        ep = ep.run_decompositions({})

        # Snapshot the input-side and pre-existing output-side specs.
        before_input_kinds = [s.kind for s in ep.graph_signature.input_specs]
        before_input_targets = [s.target for s in ep.graph_signature.input_specs]
        before_output_specs = list(ep.graph_signature.output_specs)
        before_n_outputs = len(before_output_specs)

        # Sanity: setup is non-trivial (multi-input, multi-output, params,
        # at least one buffer on the input side, AND a BUFFER_MUTATION
        # surfaced by run_decompositions).
        self.assertGreater(len(before_input_kinds), 2)
        self.assertGreater(before_n_outputs, 1)
        self.assertIn(
            OutputKind.BUFFER_MUTATION,
            {s.kind for s in before_output_specs},
            "test setup: _RichModel must produce a BUFFER_MUTATION OutputSpec",
        )

        ep_t, specs = tap_intermediate_outputs_(ep, _LINEAR_FULL_TENSOR_RULE)
        # _RichModel has 2 linears (l1, l2).
        self.assertEqual(len(specs), 2)

        # Input-side: completely untouched.
        after_input_kinds = [s.kind for s in ep_t.graph_signature.input_specs]
        after_input_targets = [s.target for s in ep_t.graph_signature.input_specs]
        self.assertEqual(after_input_kinds, before_input_kinds)
        self.assertEqual(after_input_targets, before_input_targets)

        # Output-side: original entries preserved in order; new tap entries
        # appended as USER_OUTPUTs at the end.
        after_output_specs = list(ep_t.graph_signature.output_specs)
        self.assertEqual(len(after_output_specs), before_n_outputs + len(specs))
        # Original specs unchanged, in the same order.
        for before, after in zip(before_output_specs, after_output_specs):
            self.assertEqual(before.kind, after.kind)
            self.assertEqual(before.target, after.target)
        # Appended specs are all USER_OUTPUT.
        for s in after_output_specs[before_n_outputs:]:
            self.assertEqual(s.kind, OutputKind.USER_OUTPUT)

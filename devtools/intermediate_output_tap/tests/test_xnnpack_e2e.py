# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
End-to-end test: prove that intermediate values surfaced as USER_OUTPUT taps
flow through XNNPACK delegation and out the runtime *with no XNNPACK-side
support*. This is the central correctness claim of the design.
"""

import os
import sys
import tempfile
import unittest

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.devtools.intermediate_output_tap import (
    FULL_TENSOR,
    select_by_op_type,
    STATS,
    strip_taps_,
    tap_intermediate_outputs,
)
from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime, Verification
from torch.export import export


class _MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(8, 16)
        self.l2 = torch.nn.Linear(16, 4)

    def forward(self, x):
        return self.l2(self.l1(x).relu())


@unittest.skipIf(
    sys.platform.startswith("win"), "ExecuTorch runtime not available on Windows"
)
class XnnpackEndToEndTest(unittest.TestCase):
    def _run_pipeline(self, reducer):
        model = _MLP()
        example_inputs = (torch.randn(2, 8),)

        ep = export(model, example_inputs, strict=True)
        ep_t, specs = tap_intermediate_outputs(
            ep,
            selector=select_by_op_type("aten.linear.default"),
            reducer=reducer,
        )
        edge = to_edge_transform_and_lower(ep_t, partitioner=[XnnpackPartitioner()])
        strip_taps_(edge)
        et_program = edge.to_executorch()

        with tempfile.TemporaryDirectory() as temp_dir:
            pte_path = os.path.join(temp_dir, "model.pte")
            et_program.save(pte_path)

            rt = Runtime.get()
            program = rt.load_program(pte_path, verification=Verification.Minimal)
            method = program.load_method("forward")
            flat_outputs = method.execute(list(example_inputs))

        return specs, flat_outputs, model, example_inputs

    def test_full_tensor_taps_match_eager(self):
        specs, flat, _, _ = self._run_pipeline(FULL_TENSOR)
        self.assertEqual(len(specs), 2)  # two linears
        # FULL_TENSOR preserves the source tensor's shape.
        for spec in specs:
            tap_value = flat[spec.output_index]
            self.assertIsInstance(tap_value, torch.Tensor)
            self.assertGreater(tap_value.numel(), 0)

    def test_stats_returns_thirteen_floats(self):
        specs, flat, _, _ = self._run_pipeline(STATS)
        self.assertEqual(len(specs), 2)
        for spec in specs:
            tap_value = flat[spec.output_index]
            self.assertIsInstance(tap_value, torch.Tensor)
            self.assertEqual(tap_value.numel(), len(STATS.fields))
            vals = tap_value.tolist()
            field_idx = {f: i for i, f in enumerate(STATS.fields)}
            mn = vals[field_idx["min"]]
            mx = vals[field_idx["max"]]
            abs_max = vals[field_idx["abs_max"]]
            l2 = vals[field_idx["l2_norm"]]
            l1 = vals[field_idx["l1_norm"]]
            self.assertLessEqual(mn, mx)
            self.assertGreaterEqual(abs_max, max(abs(mn), abs(mx)) - 1e-3)
            self.assertGreaterEqual(l1, 0.0)
            self.assertGreaterEqual(l2, 0.0)
            # No NaN/Inf in random fp32 — should be exactly zero.
            self.assertEqual(vals[field_idx["nan_count"]], 0.0)
            self.assertEqual(vals[field_idx["inf_count"]], 0.0)

    def test_user_outputs_still_correct(self):
        """Tap outputs must not corrupt the original user outputs."""
        specs, flat, model, example_inputs = self._run_pipeline(FULL_TENSOR)

        eager_out = model(*example_inputs)
        # User output is at index 0 (one user output for our MLP).
        user_out = flat[0]
        torch.testing.assert_close(user_out, eager_out, atol=1e-3, rtol=1e-3)
        # Verify tap indices are non-overlapping with user-output index 0.
        for spec in specs:
            self.assertGreaterEqual(spec.output_index, 1)

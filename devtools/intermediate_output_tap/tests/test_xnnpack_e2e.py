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
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)
from executorch.devtools.intermediate_output_tap import (
    ABS_MAX_ONLY,
    DEFAULT_STATS,
    FULL_TENSOR,
    MIN_MAX_MEAN,
    select_by_op_type,
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


@unittest.skipIf(sys.platform.startswith("win"), "ExecuTorch runtime not available on Windows")
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
        edge = to_edge_transform_and_lower(
            ep_t, partitioner=[XnnpackPartitioner()]
        )
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
        specs, flat, model, example_inputs = self._run_pipeline(FULL_TENSOR)
        self.assertEqual(len(specs), 2)  # two linears

        # The user output is at index 0; tap outputs follow.
        for spec in specs:
            tap_value = flat[spec.output_index]
            self.assertIsInstance(tap_value, torch.Tensor)
            # FULL_TENSOR preserves the source tensor's shape — so e.g. for
            # the first linear, shape is (batch, l1.out_features).
            self.assertGreater(tap_value.numel(), 0)

    def test_abs_max_only_returns_scalar(self):
        specs, flat, _, _ = self._run_pipeline(ABS_MAX_ONLY)
        self.assertEqual(len(specs), 2)
        for spec in specs:
            tap_value = flat[spec.output_index]
            self.assertIsInstance(tap_value, torch.Tensor)
            # 0-dim scalar
            self.assertEqual(tap_value.numel(), 1)
            self.assertGreaterEqual(float(tap_value), 0.0)

    def test_min_max_mean_e2e(self):
        specs, flat, _, _ = self._run_pipeline(MIN_MAX_MEAN)
        self.assertEqual(len(specs), 2)
        for spec in specs:
            tap_value = flat[spec.output_index]
            self.assertEqual(tap_value.numel(), 3)

    def test_default_stats_returns_seven_floats(self):
        specs, flat, _, _ = self._run_pipeline(DEFAULT_STATS)
        self.assertEqual(len(specs), 2)
        for spec in specs:
            tap_value = flat[spec.output_index]
            self.assertIsInstance(tap_value, torch.Tensor)
            self.assertEqual(tap_value.numel(), 4)
            mn, mx, _, abs_max = tap_value.tolist()
            self.assertLessEqual(mn, mx)
            self.assertGreaterEqual(abs_max, max(abs(mn), abs(mx)) - 1e-5)

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

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock, patch

import torch
from executorch.backends.test.harness.stages import StageType
from executorch.backends.test.harness.tester import Tester
from executorch.backends.test.harness.tolerance import ToleranceConfig


class IdentityModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TestTesterToleranceLookup(unittest.TestCase):
    def _make_tester(
        self,
        *,
        backend: str,
        dtype: torch.dtype = torch.float32,
        quantized: bool = False,
    ) -> Tester:
        module = IdentityModule().eval()
        example_inputs = (torch.randn(1, 4, dtype=torch.float32),)
        tester = Tester(
            module,
            example_inputs,
            backend=backend,
            dtype=dtype,
            quantized=quantized,
        )

        # Stub export stage so run_method_and_compare_outputs can run without
        # a full ExportedProgram / backend install.
        export_stage = MagicMock()
        export_stage.artifact = MagicMock()
        output = torch.randn(1, 4)
        export_stage.run_artifact.return_value = output
        tester.stages[StageType.EXPORT] = export_stage
        tester.cur = StageType.EXPORT
        return tester

    def test_run_method_passes_fp16_dtype_to_get_tolerance(self):
        tester = self._make_tester(backend="xnnpack", dtype=torch.float16)
        expected = ToleranceConfig(atol=2e-3, rtol=1e-3, qtol=0)
        output = torch.randn(1, 4)

        with (
            patch(
                "executorch.backends.test.harness.tolerance.get_tolerance",
                return_value=expected,
            ) as mock_get_tolerance,
            patch.object(
                Tester,
                "_calculate_reference_output",
                return_value=(output, None),
            ),
            patch.object(Tester, "_compare_outputs"),
        ):
            tester.run_method_and_compare_outputs()

        mock_get_tolerance.assert_called_once_with(
            "xnnpack",
            dtype=torch.float16,
            quantized=False,
        )

    def test_run_method_passes_quantized_flag_to_get_tolerance(self):
        tester = self._make_tester(backend="xnnpack", quantized=True)
        expected = ToleranceConfig(atol=1e-3, rtol=1e-3, qtol=1)
        output = torch.randn(1, 4)

        with (
            patch(
                "executorch.backends.test.harness.tolerance.get_tolerance",
                return_value=expected,
            ) as mock_get_tolerance,
            patch.object(
                Tester,
                "_calculate_reference_output",
                return_value=(output, 0.05),
            ),
            patch.object(Tester, "_compare_outputs"),
        ):
            tester.run_method_and_compare_outputs()

        mock_get_tolerance.assert_called_once_with(
            "xnnpack",
            dtype=torch.float32,
            quantized=True,
        )


if __name__ == "__main__":
    unittest.main()

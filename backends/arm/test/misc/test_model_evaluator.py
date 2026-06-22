# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest

import torch
from executorch.backends.arm.util.arm_model_evaluator import (
    FileCompressionEvaluator,
    NumericalModelEvaluator,
)

# Create an input that is hard to compress
COMPRESSION_RATIO_TEST = torch.rand([1024, 1024])


def mocked_model_1(input: torch.Tensor) -> torch.Tensor:
    return torch.tensor([1.0, 2.0, 3.0, 4.0])


def mocked_model_2(input: torch.Tensor) -> torch.Tensor:
    return torch.tensor([1.0, 2.0, 3.0, 3.0])


class TestModelEvaluator(unittest.TestCase):
    """Tests the Arm model evaluators."""

    def test_get_model_error(self):
        example_input = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        evaluator = NumericalModelEvaluator(
            "dummy_model",
            mocked_model_1,
            mocked_model_2,
            (example_input,),
            None,
        )

        metrics = evaluator.evaluate()

        self.assertEqual(metrics["max_error"], 1.0)
        self.assertEqual(metrics["max_absolute_error"], 1.0)
        self.assertEqual(metrics["max_percentage_error"], 25.0)
        self.assertEqual(metrics["mean_absolute_error"], 0.25)

    def test_get_compression_ratio(self):
        with tempfile.NamedTemporaryFile(delete=True) as temp_bin:
            torch.save(COMPRESSION_RATIO_TEST, temp_bin)

            evaluator = FileCompressionEvaluator("dummy_model", temp_bin.name)

            ratio = evaluator.evaluate()["compression_ratio"]
            self.assertAlmostEqual(ratio, 1.1, places=1)

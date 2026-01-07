# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest

import torch
from executorch.backends.arm.util.arm_model_evaluator import GenericModelEvaluator

# Create an input that is hard to compress
COMPRESSION_RATIO_TEST = torch.rand([1024, 1024])


def mocked_model_1(input: torch.Tensor) -> torch.Tensor:
    return torch.tensor([1.0, 2.0, 3.0, 4.0])


def mocked_model_2(input: torch.Tensor) -> torch.Tensor:
    return torch.tensor([1.0, 2.0, 3.0, 3.0])


class TestGenericModelEvaluator(unittest.TestCase):
    """Tests the GenericModelEvaluator class."""

    def test_get_model_error_no_target(self):
        example_input = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        evaluator = GenericModelEvaluator(
            "dummy_model",
            mocked_model_1,
            mocked_model_2,
            example_input,
            "tmp/output_tag0.tosa",
        )

        model_error_dict = evaluator.get_model_error()

        self.assertEqual(model_error_dict["max_error"], [1.0])
        self.assertEqual(model_error_dict["max_absolute_error"], [1.0])
        self.assertEqual(model_error_dict["max_percentage_error"], [25.0])
        self.assertEqual(model_error_dict["mean_absolute_error"], [0.25])

    def test_get_compression_ratio_no_target(self):
        with tempfile.NamedTemporaryFile(delete=True) as temp_bin:
            torch.save(COMPRESSION_RATIO_TEST, temp_bin)

            example_input = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
            evaluator = GenericModelEvaluator(
                "dummy_model",
                mocked_model_1,
                mocked_model_2,
                example_input,
                temp_bin.name,
            )

            ratio = evaluator.get_compression_ratio()
            self.assertAlmostEqual(ratio, 1.1, places=1)

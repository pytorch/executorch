# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
import tempfile
import unittest

import torch
from executorch.backends.arm.scripts.arm_model_evaluator import GenericModelEvaluator

random.seed(0)

# Create an input that is hard to compress
COMPRESSION_RATIO_TEST = bytearray(random.getrandbits(8) for _ in range(1000000))


def mocked_model_1(input: torch.Tensor) -> torch.Tensor:
    return torch.tensor([1.0, 2.0, 3.0, 4.0])


def mocked_model_2(input: torch.Tensor) -> torch.Tensor:
    return torch.tensor([1.0, 2.0, 3.0, 3.0])


class TestGenericModelEvaluator(unittest.TestCase):
    """Tests the GenericModelEvaluator class."""

    def test_get_model_error(self):
        example_input = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        evaluator = GenericModelEvaluator(
            "dummy_model",
            mocked_model_1,
            mocked_model_2,
            example_input,
            "tmp/output_tag0.tosa",
        )
        max_error, max_absolute_error, max_percentage_error, mae = (
            evaluator.get_model_error()
        )

        self.assertEqual(max_error, 1.0)
        self.assertEqual(max_absolute_error, 1.0)
        self.assertEqual(max_percentage_error, 25.0)
        self.assertEqual(mae, 0.25)

    def test_get_compression_ratio(self):
        with tempfile.NamedTemporaryFile(delete=True) as temp_bin:
            temp_bin.write(COMPRESSION_RATIO_TEST)

            # As the size of the file is quite small we need to call flush()
            temp_bin.flush()
            temp_bin_name = temp_bin.name

            example_input = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
            evaluator = GenericModelEvaluator(
                "dummy_model",
                mocked_model_1,
                mocked_model_2,
                example_input,
                temp_bin_name,
            )

            ratio = evaluator.get_compression_ratio()
            self.assertAlmostEqual(ratio, 1.0, places=2)

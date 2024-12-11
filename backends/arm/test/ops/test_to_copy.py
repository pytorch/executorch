# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Tests the _to_copy op which is interpreted as a cast for our purposes.
#

import unittest

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import ArmTester

from parameterized import parameterized


class Cast(torch.nn.Module):
    def __init__(self, target_dtype):
        super().__init__()
        self.target_dtype = target_dtype

    def forward(self, x: torch.Tensor):
        return x.to(dtype=self.target_dtype)


class TestToCopy(unittest.TestCase):
    """
    Tests the _to_copy operation.

    Only test unquantized graphs as explicit casting of dtypes messes with the
    quantization.

    Note: This is also covered by test_scalars.py.
    """

    _TO_COPY_TEST_DATA = (
        (torch.rand((1, 2, 3, 4), dtype=torch.float16), torch.float32),
        (torch.rand((1, 2, 3, 4), dtype=torch.float32), torch.float16),
        (torch.randint(-127, 128, (1, 2, 3, 4), dtype=torch.int8), torch.float32),
        (torch.randint(-127, 128, (1, 2, 3, 4), dtype=torch.int8), torch.int32),
        (torch.randint(-127, 128, (1, 2, 3, 4), dtype=torch.int32), torch.int8),
    )

    def _test_to_copy_tosa_MI_pipeline(
        self, module: torch.nn.Module, test_data: torch.Tensor
    ):
        (
            ArmTester(
                module,
                example_inputs=test_data,
                compile_spec=common.get_tosa_compile_spec("TOSA-0.80.0+MI"),
            )
            .export()
            .dump_artifact()
            .to_edge()
            .dump_artifact()
            .partition()
            .check_count({"torch.ops.higher_order.executorch_call_delegate": 1})
            .to_executorch()
            .run_method_and_compare_outputs(inputs=test_data)
        )

    @parameterized.expand(_TO_COPY_TEST_DATA)
    def test_view_tosa_MI(self, test_tensor: torch.Tensor, new_dtype):
        self._test_to_copy_tosa_MI_pipeline(Cast(new_dtype), (test_tensor,))

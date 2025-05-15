# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Tests the _to_copy op which is interpreted as a cast for our purposes.
#

from typing import Tuple

import torch

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import TosaPipelineMI

input_t1 = Tuple[torch.Tensor]  # Input x


class Cast(torch.nn.Module):
    def __init__(self, target_dtype):
        super().__init__()
        self.target_dtype = target_dtype

    def forward(self, x: torch.Tensor):
        return x.to(dtype=self.target_dtype)


"""
Tests the _to_copy operation.

Only test unquantized graphs as explicit casting of dtypes messes with the
quantization.

Note: This is also covered by test_scalars.py.
"""

_TO_COPY_TEST_DATA = {
    "rand_fp16": lambda: (torch.rand((1, 2, 3, 4), dtype=torch.float16), torch.float32),
    "rand_fp32": lambda: (torch.rand((1, 2, 3, 4), dtype=torch.float32), torch.float16),
    "rand_int8": lambda: (
        torch.randint(-127, 128, (1, 2, 3, 4), dtype=torch.int8),
        torch.float32,
    ),
    "rand_int8_int32": lambda: (
        torch.randint(-127, 128, (1, 2, 3, 4), dtype=torch.int8),
        torch.int32,
    ),
    "rand_int32": lambda: (
        torch.randint(-127, 128, (1, 2, 3, 4), dtype=torch.int32),
        torch.int8,
    ),
}


@common.parametrize("test_data", _TO_COPY_TEST_DATA)
def test_copy_tosa_MI(test_data: Tuple):
    test_tensor, new_dtype = test_data()

    pipeline = TosaPipelineMI[input_t1](
        Cast(new_dtype),
        (test_tensor,),
        aten_op=[],
        exir_op=[],
    )
    pipeline.run()

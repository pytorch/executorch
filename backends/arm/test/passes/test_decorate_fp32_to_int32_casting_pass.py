# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm.test import common, conftest

from executorch.backends.arm.test.tester.test_pipeline import (
    OpNotSupportedPipeline,
    TosaPipelineMI,
)

input_t1 = Tuple[torch.Tensor]  # Input x


class FP32ToINT32Casting(torch.nn.Module):
    def __init__(self, target_dtype):
        super().__init__()
        self.target_dtype = target_dtype

    def forward(self, x: torch.Tensor):
        return x.to(self.target_dtype)


test_data_fp32_input = {
    "fp32_input_rank1": lambda: (
        torch.rand((4), dtype=torch.float32),
        torch.int32,
    ),
    "fp32_input_rank2": lambda: (
        torch.rand((3, 4), dtype=torch.float32),
        torch.int32,
    ),
    "fp32_input_rank3": lambda: (
        torch.rand((2, 3, 4), dtype=torch.float32),
        torch.int32,
    ),
    "fp32_input_rank4": lambda: (
        torch.rand((1, 2, 3, 4), dtype=torch.float32),
        torch.int32,
    ),
}


@common.parametrize("test_data", test_data_fp32_input)
def test_decorate_fp32_to_int32_casting_tosa_MI(test_data: Tuple):
    test_tensor, target_dtype = test_data()
    module = FP32ToINT32Casting(target_dtype)

    pipeline = TosaPipelineMI[input_t1](
        module,
        (test_tensor,),
        aten_op=[],
        exir_op=[],
        run_on_tosa_ref_model=conftest.is_option_enabled("tosa_ref_model"),
    )
    pipeline.run()


@common.parametrize("test_data", test_data_fp32_input)
def test_decorate_fp32_to_int32_casting_tosa_BI(test_data: Tuple):
    """
    Casting operation involving floating-point dtypes will be rejected in BI/INT profile.
    Therefore, the DecorateFp32toInt32CastingPass is not required in this profile.
    Add a BI test to ensure that such casting is rejected as expected.
    """
    test_tensor, target_dtype = test_data()
    module = FP32ToINT32Casting(target_dtype)

    pipeline = OpNotSupportedPipeline[input_t1](
        module,
        (test_tensor,),
        {
            "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 1
        },
        quantize=True,
    )
    pipeline.run()

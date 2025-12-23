# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, Tuple

import torch
from executorch.backends.arm.test import common, conftest

from executorch.backends.arm.test.tester.test_pipeline import (
    OpNotSupportedPipeline,
    TosaPipelineFP,
)

input_t1 = Tuple[torch.Tensor]  # Input x


class FP32ToINT32Casting(torch.nn.Module):
    def __init__(self, target_dtype: torch.dtype) -> None:
        super().__init__()
        self.target_dtype = target_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.target_dtype)


test_data_fp32_input: Dict[str, Callable[[], Tuple[torch.Tensor, torch.dtype]]] = {
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
def test_decorate_fp32_to_int32_casting_tosa_FP(
    test_data: Callable[[], Tuple[torch.Tensor, torch.dtype]]
) -> None:
    test_tensor, target_dtype = test_data()
    module = FP32ToINT32Casting(target_dtype)

    pipeline = TosaPipelineFP[input_t1](
        module,
        (test_tensor,),
        aten_op=[],
        exir_op=[],
        run_on_tosa_ref_model=conftest.is_option_enabled("tosa_ref_model"),
    )
    pipeline.run()


@common.parametrize("test_data", test_data_fp32_input)
def test_decorate_fp32_to_int32_casting_tosa_INT(
    test_data: Callable[[], Tuple[torch.Tensor, torch.dtype]]
) -> None:
    """
    Casting operation involving floating-point dtypes will be rejected in INT/INT profile.
    Therefore, the DecorateFp32toInt32CastingPass is not required in this profile.
    Add a INT test to ensure that such casting is rejected as expected.
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

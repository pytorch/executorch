# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, Tuple

import torch
from executorch.backends.arm._passes import ConvertInt64OutputOpsToInt32Pass

from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import TosaPipelineFP

input_t1 = Tuple[torch.Tensor]  # Input x


#########################################
## Test [int32 | other types] -> int64 ##
#########################################


class CastingToInt64Model(torch.nn.Module):
    def __init__(self, target_dtype: torch.dtype) -> None:
        super().__init__()
        self.target_dtype = target_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(dtype=self.target_dtype)


test_data_suite_convert: Dict[str, Callable[[], Tuple[torch.Tensor, torch.dtype]]] = {
    "fp32_input": lambda: (torch.rand((1, 2, 3, 4), dtype=torch.float32), torch.int64),
    "fp16_input": lambda: (torch.rand((1, 2, 3, 4), dtype=torch.float16), torch.int64),
}

test_data_suite_remove: Dict[str, Callable[[], Tuple[torch.Tensor, torch.dtype]]] = {
    "int32_input": lambda: (
        torch.randint(-127, 128, (1, 2, 3, 4), dtype=torch.int32),
        torch.int64,
    ),
}


TestDataFactory = Callable[[], Tuple[torch.Tensor, torch.dtype]]


@common.parametrize("test_data", test_data_suite_convert)
def test_convert_or_remove_casting_to_int64_convert_tosa_FP(
    test_data: TestDataFactory,
) -> None:
    test_tensor, target_dtype = test_data()
    module = CastingToInt64Model(target_dtype)

    pipeline = TosaPipelineFP[input_t1](
        module,
        (test_tensor,),
        aten_op="torch.ops.aten.to.dtype",
        exir_op=[],
        transform_passes=[ConvertInt64OutputOpsToInt32Pass()],
    )
    pipeline.pop_stage(
        "run_method_and_compare_outputs"
    )  # As expected: RuntimeError: Int did not match Long
    pipeline.run()


@common.parametrize("test_data", test_data_suite_remove)
def test_convert_or_remove_casting_to_int64_remove_tosa_FP(
    test_data: TestDataFactory,
) -> None:
    test_tensor, target_dtype = test_data()
    module = CastingToInt64Model(target_dtype)

    pipeline = TosaPipelineFP[input_t1](
        module,
        (test_tensor,),
        aten_op=[],
        exir_op=[],
        transform_passes=[ConvertInt64OutputOpsToInt32Pass()],
    )
    pipeline.change_args(
        "check_count.exir", {"torch.ops.higher_order.executorch_call_delegate": 0}
    )  # Empty graph without nodes
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()


#####################################################
## Test arange(dtype=int64) -> arange(dtype=int32) ##
#####################################################


class Int64OutputModel(torch.nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return torch.argmax(x)  # RuntimeError: Int did not match Long; But this is expected as we expect _argmax_i32 to generate int32 output
        # return (10 * torch.argmax(x) + 10).to(dtype=torch.int32)  #  [1]. This behavior is deprecated, and in a future PyTorch release outputs will not be resized unless they have zero elements. You can explicitly reuse an out tensor t by resizing it, inplace, to zero elements with t.resize_(0). (function _resize_output_check)
        return (10 * torch.argmax(x, dim=-1) + 10) + 1.5

    def get_inputs(self) -> input_t1:
        return (
            torch.randint(
                0,
                10,
                (2, 4, 6, 8),
            ),
        )


def test_insert_int64_output_to_int32_cast_tosa_FP():
    module = Int64OutputModel()
    aten_ops_checks = [
        "torch.ops.aten.argmax.default",
        "torch.ops.aten.mul.Tensor",
        "torch.ops.aten.add.Tensor",
    ]
    exir_ops_checks = [
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
        "executorch_exir_dialects_edge__ops_aten_add_Tensor",
    ]
    pipeline = TosaPipelineFP[input_t1](
        module,
        module.get_inputs(),
        aten_op=aten_ops_checks,
        exir_op=exir_ops_checks,
        transform_passes=[ConvertInt64OutputOpsToInt32Pass()],
    )
    pipeline.run()

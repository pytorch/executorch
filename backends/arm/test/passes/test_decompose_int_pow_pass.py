# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict, Protocol, Tuple

import torch
from executorch.backends.arm._passes import DecomposeIntPowPass

from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

input_t = Tuple[torch.Tensor]  # Inputs to the module


class ModuleWithInputs(Protocol):
    def get_inputs(self) -> input_t: ...


TestParam = Tuple[ModuleWithInputs, int]


class Square(torch.nn.Module):
    """
    Basic squaring
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.square()

    def get_inputs(self) -> input_t:
        return (torch.rand(4, 4),)


class Pow(torch.nn.Module):
    """
    Basic squaring
    """

    def __init__(self, exponent: int) -> None:
        super().__init__()
        self.exponent = exponent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.pow(self.exponent)

    def get_inputs(self) -> input_t:
        return (torch.rand(4, 4),)


test_data: Dict[str, TestParam] = {
    "square": (Square(), 1),
    "pow_2": (Pow(2), 1),
    "pow_3": (Pow(3), 2),
    "pow_0": (Pow(0), 0),
    "pow_neg_2": (Pow(-2), 1),
}


@common.parametrize("data", test_data)
def test_decompose_int_pow(data: TestParam) -> None:
    module_with_inputs, nbr_muls = data
    module = cast(torch.nn.Module, module_with_inputs)
    pipeline = PassPipeline[input_t](
        module,
        module_with_inputs.get_inputs(),
        quantize=False,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_pow_Tensor_Scalar": 1,
        },
        ops_not_before_pass=[],
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_mul_Tensor": nbr_muls,
        },
        ops_not_after_pass=["executorch_exir_dialects_edge__ops_pow_Tensor_Scalar"],
        pass_list=[DecomposeIntPowPass],
    )
    pipeline.run()

# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from executorch.backends.arm._passes import ConvertIntPowToMuls

from executorch.backends.arm.test import common

from executorch.backends.arm.test.tester.test_pipeline import PassPipeline

input_t = Tuple[torch.nn.Module, int]  # Input x


class Square(torch.nn.Module):
    """
    Basic squaring
    """

    def forward(self, x):
        return x.square()

    def get_inputs(self) -> input_t:
        return (torch.rand(4, 4),)


class Pow(torch.nn.Module):
    """
    Basic squaring
    """

    def __init__(self, exponent):
        super().__init__()
        self.exponent = exponent

    def forward(self, x):
        return x.pow(self.exponent)

    def get_inputs(self) -> input_t:
        return (torch.rand(4, 4),)


test_data = {
    "square": (Square(), 1),
    "pow_2": (Pow(2), 1),
    "pow_3": (Pow(3), 2),
    "pow_0": (Pow(0), 0),
    "pow_neg_2": (Pow(-2), 1),
}


@common.parametrize("data", test_data)
def test_convert_pow_to_muls(data):
    module = data[0]
    nbr_muls = data[1]
    pipeline = PassPipeline[input_t](
        module,
        module.get_inputs(),
        quantize=False,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_pow_Tensor_Scalar": 1,
        },
        ops_not_before_pass=[],
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_mul_Tensor": nbr_muls,
        },
        ops_not_after_pass=["executorch_exir_dialects_edge__ops_pow_Tensor_Scalar"],
        pass_list=[ConvertIntPowToMuls],
    )
    pipeline.run()

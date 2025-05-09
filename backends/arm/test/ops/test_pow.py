# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
    TosaPipelineMI,
)


class Pow_TensorTensor(torch.nn.Module):
    aten_op = "torch.ops.aten.pow.Tensor_Tensor"
    exir_op = "executorch_exir_dialects_edge__ops_aten_pow_Tensor_Tensor"

    input_t = Tuple[torch.Tensor | float, torch.Tensor | float]

    # The sign of the operands are important w.r.t. TOSA's spec of pow
    test_data = {
        "zero_base_pos_exp": lambda: (
            torch.zeros(1, 8, 3, 7),
            torch.abs(torch.randn((1, 8, 1, 7))) + 1e5,
        ),
        "pos_base": lambda: (
            torch.abs(torch.randn((3, 2, 4, 2))) + 1e5,
            torch.randn((1, 2, 4, 1)),
        ),
        "zero_base_zero_exp": lambda: (torch.zeros(2, 3), torch.zeros(2, 3)),
        "pos_base_zero_exp": lambda: (
            torch.abs(torch.randn((1, 7, 2, 3))) + 1e5,
            torch.zeros(1, 1, 2, 3),
        ),
        "neg_base_zero_exp": lambda: (
            -torch.abs(torch.randn((1, 2, 3, 4))) - 1e5,
            torch.zeros(1, 2, 3, 4),
        ),
        "base_has_lower_rank": lambda: (torch.ones(3, 4), torch.ones(1, 2, 3, 4)),
        "exp_has_lower_rank": lambda: (torch.ones(1, 2, 3, 4), torch.ones(3, 4)),
        "f16_tensors": lambda: (
            torch.HalfTensor([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]]),
            torch.HalfTensor([[1.0, 2.0, 0.0]]),
        ),
    }

    def forward(self, x: torch.Tensor | float, y: torch.Tensor | float):
        return torch.pow(x, y)


class Pow_TensorScalar(torch.nn.Module):
    aten_op = "torch.ops.aten.pow.Tensor_Scalar"
    exir_op = "executorch_exir_dialects_edge__ops_aten_pow_Tensor_Scalar"

    input_t = Tuple[torch.Tensor]

    test_data = {
        # Test whole number exponents
        "exp_minus_three": lambda: (torch.randn((10, 5)), -3.0),
        "exp_minus_one": lambda: (torch.randn((42,)), -1.0),
        "exp_zero": lambda: (torch.randn((1, 2, 3, 7)), 0.0),
        "exp_one": lambda: (torch.randn((1, 4, 6, 2)), 1.0),
        "exp_two": lambda: (torch.randn((1, 2, 3, 6)), 2.0),
        # Test decimal exponent (base must be non-negative)
        "non_neg_base_exp_pos_decimal": lambda: (
            torch.abs(torch.randn((1, 2, 3, 6))),
            6.789,
        ),
    }

    def __init__(self, exp):
        super().__init__()
        self.exp = exp

    def forward(self, x: torch.Tensor):
        return torch.pow(x, self.exp)


x_fail = {
    "zero_base_zero_exp": "TOSA constraints: If x == 0 and y ⇐ 0, the result is undefined.",
    "neg_base_zero_exp": "TOSA constraints: If x == 0 and y ⇐ 0, the result is undefined.",
}


@common.parametrize("test_data", Pow_TensorTensor.test_data, x_fail, strict=False)
def test_pow_tensor_tensor_tosa_MI(test_data: Pow_TensorTensor.input_t):
    pipeline = TosaPipelineMI[Pow_TensorTensor.input_t](
        Pow_TensorTensor(),
        test_data(),
        Pow_TensorTensor.aten_op,
        Pow_TensorTensor.exir_op,
    )
    pipeline.run()


x_fail = {
    "exp_minus_three": "TOSA constraints: If x == 0 and y ⇐ 0, the result is undefined.",
    "exp_minus_one": "TOSA constraints: If x == 0 and y ⇐ 0, the result is undefined.",
    "exp_zero": "TOSA constraints: If x == 0 and y ⇐ 0, the result is undefined.",
    "exp_one": "TOSA constraints: If x == 0 and y ⇐ 0, the result is undefined.",
    "exp_two": "TOSA constraints: If x == 0 and y ⇐ 0, the result is undefined.",
    "non_neg_base_exp_pos_decimal": "TOSA constraints: If x == 0 and y ⇐ 0, the result is undefined.",
}


@common.parametrize("test_data", Pow_TensorScalar.test_data, x_fail, strict=False)
def test_pow_tensor_scalar_tosa_MI(test_data: Pow_TensorScalar.input_t):
    base, exp = test_data()
    pipeline = TosaPipelineMI[Pow_TensorScalar.input_t](
        Pow_TensorScalar(exp),
        (base,),
        Pow_TensorScalar.aten_op,
        Pow_TensorScalar.exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", Pow_TensorScalar.test_data, x_fail, strict=False)
def test_pow_tensor_scalar_tosa_BI(test_data: Pow_TensorScalar.input_t):
    base, exp = test_data()
    pipeline = TosaPipelineBI[Pow_TensorScalar.input_t](
        Pow_TensorScalar(exp),
        (base,),
        Pow_TensorScalar.aten_op,
        Pow_TensorScalar.exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", Pow_TensorScalar.test_data)
@common.XfailIfNoCorstone300
def test_pow_tensor_scalar_u55_BI(test_data: Pow_TensorScalar.input_t):
    base, exp = test_data()
    pipeline = EthosU55PipelineBI[Pow_TensorScalar.input_t](
        Pow_TensorScalar(exp),
        (base,),
        Pow_TensorScalar.aten_op,
        Pow_TensorScalar.exir_op,
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", Pow_TensorScalar.test_data)
@common.XfailIfNoCorstone320
def test_pow_tensor_scalar_u85_BI(test_data: Pow_TensorScalar.input_t):
    base, exp = test_data()
    pipeline = EthosU85PipelineBI[Pow_TensorScalar.input_t](
        Pow_TensorScalar(exp),
        (base,),
        Pow_TensorScalar.aten_op,
        Pow_TensorScalar.exir_op,
        run_on_fvp=True,
    )
    pipeline.run()

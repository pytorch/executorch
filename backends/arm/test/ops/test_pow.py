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


def get_aten_op(base, exp):
    if isinstance(base, torch.Tensor):
        if isinstance(exp, torch.Tensor):
            return "torch.ops.aten.pow.Tensor_Tensor"
        else:
            return "torch.ops.aten.pow.Tensor_Scalar"
    else:
        return "torch.ops.aten.pow.Scalar"


def get_exir_op(base, exp):
    if isinstance(base, torch.Tensor):
        if isinstance(exp, torch.Tensor):
            return "executorch_exir_dialects_edge__ops_aten_pow_Tensor_Tensor"
        else:
            return "executorch_exir_dialects_edge__ops_aten_pow_Tensor_Scalar"
    else:
        return "executorch_exir_dialects_edge__ops_aten_pow_Scalar"


class Pow(torch.nn.Module):
    def forward(self, x: torch.Tensor | float, y: torch.Tensor | float):
        return torch.pow(x, y)

    input_t = Tuple[torch.Tensor | float, torch.Tensor | float]

    # The sign of the operands are important w.r.t. TOSA's spec of pow
    test_data = {
        "zero_base_pos_exp": (
            torch.zeros(1, 8, 3, 7),
            torch.abs(torch.randn((1, 8, 1, 7))) + 1e5,
        ),
        "pos_base": (
            torch.abs(torch.randn((3, 2, 4, 2))) + 1e5,
            torch.randn((1, 2, 4, 1)),
        ),
        "zero_base_zero_exp": (torch.zeros(2, 3), torch.zeros(2, 3)),
        "pos_base_zero_exp": (
            torch.abs(torch.randn((1, 7, 2, 3))) + 1e5,
            torch.zeros(1, 1, 2, 3),
        ),
        "neg_base_zero_exp": (
            -torch.abs(torch.randn((1, 2, 3, 4))) - 1e5,
            torch.zeros(1, 2, 3, 4),
        ),
        "base_has_lower_rank": (torch.ones(3, 4), torch.ones(1, 2, 3, 4)),
        "exp_has_lower_rank": (torch.ones(1, 2, 3, 4), torch.ones(3, 4)),
        "scalar_exp": (torch.randn((3, 4, 6, 2)), 2.0),
        "f16_tensors": (
            torch.HalfTensor([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]]),
            torch.HalfTensor([[1.0, 2.0, 0.0]]),
        ),
        # TODO: Unsupported for now. Enable this test case when MLETORCH-408 is resolved.
        # "scalar_base": (2.0, torch.tensor([1.0, 2.0, 3.0])),
    }


class PowConstExp(torch.nn.Module):
    def __init__(self, exp):
        super().__init__()
        self.exp = exp

    def forward(self, x: torch.Tensor):
        return torch.pow(x, self.exp)

    input_t = Tuple[torch.Tensor]

    test_data = {
        # Test whole number exponents
        "exp_minus_three": (torch.randn((10, 5)), -3.0),
        "exp_minus_one": (torch.randn((42,)), -1.0),
        "exp_zero": (torch.randn((1, 2, 3, 7)), 0.0),
        "exp_one": (torch.randn((1, 4, 6, 2)), 1.0),
        "exp_two": (torch.randn((1, 2, 3, 6)), 2.0),
        # Test decimal exponent (base must be non-negative)
        "non_neg_base_exp_pos_decimal": (torch.abs(torch.randn((1, 2, 3, 6))), 6.789),
    }


@common.parametrize("test_data", Pow.test_data)
def test_pow_tosa_MI(test_data: Pow.input_t):
    pipeline = TosaPipelineMI[Pow.input_t](
        Pow(),
        test_data,
        get_aten_op(*test_data),
        get_exir_op(*test_data),
    )
    pipeline.run()


@common.parametrize("test_data", PowConstExp.test_data)
def test_pow_const_exp_tosa_MI(test_data: PowConstExp.input_t):
    base, exp = test_data
    pipeline = TosaPipelineMI[PowConstExp.input_t](
        PowConstExp(exp),
        (base,),
        get_aten_op(*test_data),
        get_exir_op(*test_data),
    )
    pipeline.run()


@common.parametrize("test_data", PowConstExp.test_data)
def test_pow_const_exp_tosa_BI(test_data: PowConstExp.input_t):
    base, exp = test_data
    pipeline = TosaPipelineBI[PowConstExp.input_t](
        PowConstExp(exp),
        (base,),
        "torch.ops.aten.pow.Tensor_Tensor",  # Operator is converted to Tensor_Tensor variant during a pass
        get_exir_op(*test_data),
    )
    pipeline.run()


@common.parametrize("test_data", PowConstExp.test_data)
def test_pow_const_exp_u55_BI(test_data: PowConstExp.input_t):
    base, exp = test_data
    pipeline = EthosU55PipelineBI[PowConstExp.input_t](
        PowConstExp(exp),
        (base,),
        "torch.ops.aten.pow.Tensor_Tensor",  # Operator is converted to Tensor_Tensor variant during a pass
        "executorch_exir_dialects_edge__ops_aten_pow_Tensor_Tensor",
        run_on_fvp=False,
    )
    pipeline.run()


@common.parametrize("test_data", PowConstExp.test_data)
@common.SkipIfNoCorstone300
def test_pow_const_exp_u55_on_fvp(test_data: PowConstExp.input_t):
    base, exp = test_data
    pipeline = EthosU55PipelineBI[PowConstExp.input_t](
        PowConstExp(exp),
        (base,),
        "torch.ops.aten.pow.Tensor_Tensor",  # Operator is converted to Tensor_Tensor variant during a pass
        "executorch_exir_dialects_edge__ops_aten_pow_Tensor_Tensor",
        run_on_fvp=True,
    )
    pipeline.run()


@common.parametrize("test_data", PowConstExp.test_data)
def test_pow_const_exp_u85_BI(test_data: PowConstExp.input_t):
    base, exp = test_data
    pipeline = EthosU85PipelineBI[PowConstExp.input_t](
        PowConstExp(exp),
        (base,),
        "torch.ops.aten.pow.Tensor_Tensor",  # Operator is converted to Tensor_Tensor variant during a pass
        "executorch_exir_dialects_edge__ops_aten_pow_Tensor_Tensor",
        run_on_fvp=False,
    )
    pipeline.run()


@common.parametrize("test_data", PowConstExp.test_data)
@common.SkipIfNoCorstone320
def test_pow_const_exp_u85_on_fvp(test_data: PowConstExp.input_t):
    base, exp = test_data
    pipeline = EthosU85PipelineBI[PowConstExp.input_t](
        PowConstExp(exp),
        (base,),
        "torch.ops.aten.pow.Tensor_Tensor",  # Operator is converted to Tensor_Tensor variant during a pass
        "executorch_exir_dialects_edge__ops_aten_pow_Tensor_Tensor",
        run_on_fvp=True,
    )
    pipeline.run()

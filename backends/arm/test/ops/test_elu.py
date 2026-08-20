# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Callable, Tuple

import torch
import torch.nn as nn

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)


class Elu(nn.Module):
    aten_op = "torch.ops.aten.elu.default"
    exir_op = "executorch_exir_dialects_edge__ops_aten__elu_default"
    quantized_aten_op = aten_op
    quantized_exir_op = exir_op

    def __init__(
        self,
        input_alpha: float = 1.0,
        scale: float = 1.0,
        input_scale: float = 1.0,
    ):
        super().__init__()
        self.input_alpha = input_alpha
        self.scale = scale
        self.input_scale = input_scale

    def forward(self, input_: torch.Tensor):
        return torch.ops.aten.elu.default(
            input_, self.input_alpha, self.scale, self.input_scale
        )


class Selu(nn.Module):
    aten_op = "torch.ops.aten.selu.default"
    exir_op = "executorch_exir_dialects_edge__ops_aten_selu_default"
    quantized_aten_op = Elu.aten_op
    quantized_exir_op = Elu.exir_op

    def __init__(self):
        super().__init__()
        self.selu = torch.nn.SELU()

    def forward(self, input_: torch.Tensor):
        return self.selu(input_)


class Celu(nn.Module):
    aten_op = "torch.ops.aten.celu.default"
    exir_op = "executorch_exir_dialects_edge__ops_aten_celu_default"
    quantized_aten_op = Elu.aten_op
    quantized_exir_op = Elu.exir_op

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.celu = torch.nn.CELU(alpha=alpha)

    def forward(self, input_: torch.Tensor):
        return self.celu(input_)


input_t1 = Tuple[torch.Tensor]


@dataclass
class EluTestCase:
    model: nn.Module
    example_inputs: input_t1 | Callable[[], input_t1]

    def get_example_inputs(self) -> input_t1:
        if callable(self.example_inputs):
            return self.example_inputs()
        return self.example_inputs

    def aten_op(self, quantized: bool = False) -> str:
        attr = "quantized_aten_op" if quantized else "aten_op"
        return getattr(self.model, attr)

    def exir_op(self, quantized: bool = False) -> str:
        attr = "quantized_exir_op" if quantized else "exir_op"
        return getattr(self.model, attr)


def _input(input_: torch.Tensor) -> input_t1:
    return (input_,)


test_suite = {
    "elu_rand_default": lambda: EluTestCase(Elu(), _input(torch.rand(10, 10) - 0.5)),
    "elu_randn_pos_default": lambda: EluTestCase(
        Elu(), _input(torch.randn(1, 2, 3, 3) + 10)
    ),
    "elu_randn_neg_default": lambda: EluTestCase(
        Elu(1.0), _input(torch.randn(2, 4, 3) - 10)
    ),
    "elu_large_pos_default": lambda: EluTestCase(
        Elu(1.0), _input(torch.randn(3, 3) * 1e6 + 1e7)
    ),
    "elu_large_neg_default": lambda: EluTestCase(
        Elu(1.0), _input(-torch.empty(5).uniform_(1e5, 1e8))
    ),
    "elu_small_pos_default": lambda: EluTestCase(
        Elu(1), _input(torch.empty(5).uniform_(1e-8, 1e-5))
    ),
    "elu_small_neg_default": lambda: EluTestCase(
        Elu(1), _input(-torch.empty(5).uniform_(1e-8, 1e-5))
    ),
    "elu_rand_custom": lambda: EluTestCase(Elu(2.5), _input(torch.rand(10, 10) - 0.5)),
    "elu_randn_pos_custom": lambda: EluTestCase(
        Elu(2.0), _input(torch.randn(1, 3, 3) + 10)
    ),
    "elu_ramp_custom": lambda: EluTestCase(
        Elu(10.0), _input(torch.arange(-16, 16, 0.2))
    ),
    "elu_large_pos_custom": lambda: EluTestCase(
        Elu(2.0), _input(torch.randn(3, 3) * 1e6 + 1e7)
    ),
    "elu_large_neg_custom": lambda: EluTestCase(
        Elu(2.0), _input(-torch.empty(5).uniform_(1e5, 1e8))
    ),
    "elu_small_pos_custom": lambda: EluTestCase(
        Elu(2.0), _input(torch.empty(5).uniform_(1e-8, 1e-5))
    ),
    "elu_small_neg_custom": lambda: EluTestCase(
        Elu(2.0), _input(-torch.empty(5).uniform_(1e-8, 1e-5))
    ),
    "elu_rand_zero": lambda: EluTestCase(Elu(0.0), _input(torch.rand(10, 10) - 0.5)),
    "elu_ramp_zero": lambda: EluTestCase(Elu(0.0), _input(torch.arange(-16, 16, 0.2))),
    "elu_large_pos_zero": lambda: EluTestCase(
        Elu(0.0), _input(torch.randn(3, 3) * 1e6 + 1e7)
    ),
    "elu_large_neg_zero": lambda: EluTestCase(
        Elu(0.0), _input(-torch.empty(5).uniform_(1e5, 1e8))
    ),
    "elu_selu_params_ramp": lambda: EluTestCase(
        Elu(1.6732632423543772, 1.0507009873554805, 1.0),
        _input(torch.arange(-16, 16, 0.2)),
    ),
    "elu_celu_alpha_0_5_params_rand": lambda: EluTestCase(
        Elu(0.5, 1.0, 2.0), _input(torch.rand(10, 10) - 0.5)
    ),
    "elu_celu_alpha_2_params_ramp": lambda: EluTestCase(
        Elu(2.0, 1.0, 0.5), _input(torch.arange(-16, 16, 0.2))
    ),
    "nn_selu_ramp": lambda: EluTestCase(Selu(), _input(torch.arange(-16, 16, 0.2))),
    "nn_celu_alpha_0_5_rand": lambda: EluTestCase(
        Celu(0.5), _input(torch.rand(10, 10) - 0.5)
    ),
    "nn_celu_alpha_2_ramp": lambda: EluTestCase(
        Celu(2.0), _input(torch.arange(-16, 16, 0.2))
    ),
}


@common.parametrize("test_case", test_suite)
def test_elu_tosa_FP(test_case: Callable[[], EluTestCase]):
    test_case = test_case()
    pipeline = TosaPipelineFP[input_t1](
        test_case.model,
        test_case.get_example_inputs(),
        aten_op=test_case.aten_op(),
        exir_op=test_case.exir_op(),
    )
    pipeline.run()


@common.parametrize("test_case", test_suite)
def test_elu_tosa_INT(test_case: Callable[[], EluTestCase]):
    test_case = test_case()
    pipeline = TosaPipelineINT[input_t1](
        test_case.model,
        test_case.get_example_inputs(),
        aten_op=test_case.aten_op(quantized=True),
        exir_op=test_case.exir_op(quantized=True),
    )
    pipeline.run()


@common.XfailIfNoCorstone300
@common.parametrize("test_case", test_suite)
def test_elu_u55_INT(test_case: Callable[[], EluTestCase]):
    test_case = test_case()
    pipeline = EthosU55PipelineINT[input_t1](
        test_case.model,
        test_case.get_example_inputs(),
        aten_ops=test_case.aten_op(quantized=True),
        exir_ops=test_case.exir_op(quantized=True),
    )
    pipeline.run()


@common.XfailIfNoCorstone320
@common.parametrize("test_case", test_suite)
def test_elu_u85_INT(test_case: Callable[[], EluTestCase]):
    test_case = test_case()
    pipeline = EthosU85PipelineINT[input_t1](
        test_case.model,
        test_case.get_example_inputs(),
        aten_ops=test_case.aten_op(quantized=True),
        exir_ops=test_case.exir_op(quantized=True),
    )
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("test_case", test_suite)
def test_elu_vgf_no_quant(test_case: Callable[[], EluTestCase]):
    test_case = test_case()
    pipeline = VgfPipeline[input_t1](
        test_case.model,
        test_case.get_example_inputs(),
        aten_op=test_case.aten_op(),
        exir_op=test_case.exir_op(),
        quantize=False,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("test_case", test_suite)
def test_elu_vgf_quant(test_case: Callable[[], EluTestCase]):
    test_case = test_case()
    pipeline = VgfPipeline[input_t1](
        test_case.model,
        test_case.get_example_inputs(),
        aten_op=test_case.aten_op(quantized=True),
        exir_op=test_case.exir_op(quantized=True),
        quantize=True,
    )
    pipeline.run()

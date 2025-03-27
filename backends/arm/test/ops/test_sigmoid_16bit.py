# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineBI,
    EthosU85PipelineBI,
    TosaPipelineBI,
)
from executorch.backends.xnnpack.test.tester import Quantize
from torch.ao.quantization.observer import HistogramObserver
from torch.ao.quantization.quantizer import QuantizationSpec


def _get_16_bit_quant_config():
    int16_spec = QuantizationSpec(
        dtype=torch.int16,
        observer_or_fake_quant_ctr=HistogramObserver,
        qscheme=torch.per_tensor_symmetric,
    )
    qconfig = QuantizationConfig(
        input_activation=int16_spec,
        output_activation=int16_spec,
        weight=None,
        bias=None,
    )
    return qconfig


def get_16bit_sigmoid_quantizer(tosa_str: str):
    tosa_spec = common.TosaSpecification.create_from_string(tosa_str)
    quantizer = TOSAQuantizer(tosa_spec)
    quantizer.set_global(get_symmetric_quantization_config())
    quantizer.set_module_type(
        torch.nn.modules.activation.Sigmoid, _get_16_bit_quant_config()
    )

    return Quantize(quantizer, get_symmetric_quantization_config())


input_t = tuple[torch.Tensor]
test_data_suite = {
    "ones": lambda: torch.ones(10, 10, 10),
    "rand": lambda: torch.rand(10, 10) - 0.5,
    "rand_4d": lambda: torch.rand(1, 1, 5, 10),
    "randn_pos": lambda: torch.randn(10) + 10,
    "randn_neg": lambda: torch.randn(10) - 10,
    "ramp": lambda: torch.arange(-16, 16, 0.02),
}


class Sigmoid(torch.nn.Module):
    aten_op = "torch.ops.aten.sigmoid.default"
    exir_op = "executorch_exir_dialects_edge__ops_aten_sigmoid_default"

    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(x)


class SigmoidAddSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid((self.sigmoid(x) + self.sigmoid(x)))


@common.parametrize("test_data", test_data_suite)
@pytest.mark.flaky(reruns=5)
def test_sigmoid_tosa_BI(test_data):
    pipeline = TosaPipelineBI(
        Sigmoid(), (test_data(),), Sigmoid.aten_op, Sigmoid.exir_op
    )
    pipeline.change_args("quantize", get_16bit_sigmoid_quantizer("TOSA-0.80+BI"))
    pipeline.run()


@common.parametrize(
    "test_data",
    test_data_suite,
    xfails={
        "ramp": "AssertionError: Output 0 does not match reference output. MLETORCH-787"
    },
)
@pytest.mark.flaky(reruns=5)
def test_sigmoid_add_sigmoid_tosa_BI(test_data):
    pipeline = TosaPipelineBI(
        SigmoidAddSigmoid(), (test_data(),), Sigmoid.aten_op, Sigmoid.exir_op
    )
    pipeline.change_args("quantize", get_16bit_sigmoid_quantizer("TOSA-0.80+BI"))
    pipeline.run()


@common.parametrize(
    "test_data",
    test_data_suite,
    xfails={
        "ones": "AssertionError: Output 0 does not match reference output. MLBEDSW-9770",
        "rand": "AssertionError: Output 0 does not match reference output. MLBEDSW-9770",
        "rand_4d": "AssertionError: Output 0 does not match reference output. MLBEDSW-9770",
        "randn_pos": "AssertionError: Output 0 does not match reference output. MLBEDSW-9770",
        "randn_neg": "AssertionError: Output 0 does not match reference output. MLBEDSW-9770",
        "ramp": "AssertionError: Output 0 does not match reference output. MLBEDSW-9770",
    },
    # int16 tables are not supported, but some tests happen to pass regardless.
    # Set them to xfail but strict=False -> ok if they pass.
    strict=False,
)
@common.XfailIfNoCorstone300
def test_sigmoid_tosa_u55(test_data):
    pipeline = EthosU55PipelineBI(
        Sigmoid(), (test_data(),), Sigmoid.aten_op, Sigmoid.exir_op, run_on_fvp=True
    )
    pipeline.change_args("quantize", get_16bit_sigmoid_quantizer("TOSA-0.80+BI+u55"))
    pipeline.run()


@common.parametrize(
    "test_data",
    test_data_suite,
    xfails={
        "ones": "AssertionError: Output 0 does not match reference output. MLBEDSW-9770",
        "rand": "AssertionError: Output 0 does not match reference output. MLBEDSW-9770",
        "rand_4d": "AssertionError: Output 0 does not match reference output. MLBEDSW-9770",
        "randn_neg": "AssertionError: Output 0 does not match reference output. MLBEDSW-9770",
        "randn_pos": "AssertionError: Output 0 does not match reference output. MLBEDSW-9770",
        "ramp": "AsssertionError: Output 0 does not match reference output. MLBEDSW-9770",
    },
    # int16 tables are not supported, but some tests happen to pass regardless.
    # Set them to xfail but strict=False -> ok if they pass.
    strict=False,
)
@common.XfailIfNoCorstone300
def test_sigmoid_add_sigmoid_tosa_u55(test_data):
    pipeline = EthosU55PipelineBI(
        SigmoidAddSigmoid(),
        (test_data(),),
        Sigmoid.aten_op,
        Sigmoid.exir_op,
        run_on_fvp=True,
    )
    pipeline.change_args("quantize", get_16bit_sigmoid_quantizer("TOSA-0.80+BI+u55"))
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@pytest.mark.flaky(reruns=5)
@common.XfailIfNoCorstone320
def test_sigmoid_tosa_u85(test_data):
    pipeline = EthosU85PipelineBI(
        Sigmoid(), (test_data(),), Sigmoid.aten_op, Sigmoid.exir_op, run_on_fvp=True
    )
    pipeline.change_args("quantize", get_16bit_sigmoid_quantizer("TOSA-0.80+BI"))
    pipeline.run()


@common.parametrize(
    "test_data",
    test_data_suite,
    xfails={
        "ramp": "AssertionError: Output 0 does not match reference output.",
    },
)
@pytest.mark.flaky(reruns=5)
@common.XfailIfNoCorstone320
def test_sigmoid_add_sigmoid_tosa_u85(test_data):
    pipeline = EthosU85PipelineBI(
        SigmoidAddSigmoid(),
        (test_data(),),
        Sigmoid.aten_op,
        Sigmoid.exir_op,
        run_on_fvp=True,
    )
    pipeline.change_args("quantize", get_16bit_sigmoid_quantizer("TOSA-0.80+BI"))
    pipeline.run()

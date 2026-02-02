# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU85PipelineINT,
    OpNotSupportedPipeline,
    TosaPipelineINT,
)
from torchao.quantization.pt2e import HistogramObserver
from torchao.quantization.pt2e.quantizer import QuantizationSpec


def _get_16_bit_quant_config():
    int16_spec = QuantizationSpec(
        dtype=torch.int16,
        observer_or_fake_quant_ctr=HistogramObserver,
        qscheme=torch.per_tensor_symmetric,
    )
    int32_spec = QuantizationSpec(
        dtype=torch.int32,
        observer_or_fake_quant_ctr=HistogramObserver,
        qscheme=torch.per_tensor_symmetric,
    )
    qconfig = QuantizationConfig(
        input_activation=int16_spec,
        output_activation=int32_spec,
        weight=None,
        bias=None,
    )
    return qconfig


def _get_32_bit_quant_config():
    int32_spec = QuantizationSpec(
        dtype=torch.int32,
        observer_or_fake_quant_ctr=HistogramObserver,
        qscheme=torch.per_tensor_symmetric,
    )
    qconfig = QuantizationConfig(
        input_activation=int32_spec,
        output_activation=int32_spec,
        weight=None,
        bias=None,
    )
    return qconfig


def configure_32bit_sigmoid_quantizer(pipeline):
    pipeline.quantizer.set_global(_get_32_bit_quant_config())
    pipeline.quantizer.set_module_type(
        torch.nn.modules.activation.Sigmoid, _get_16_bit_quant_config()
    )


input_t = tuple[torch.Tensor]
test_data_suite = {
    "ones": lambda: torch.ones(10, 10, 10),
    "rand": lambda: torch.rand(10, 10) - 0.5,
    "rand_4d": lambda: torch.rand(1, 10, 10, 10),
    "randn_pos": lambda: torch.randn(10) + 10,
    "randn_neg": lambda: torch.randn(10) - 10,
    "ramp": lambda: torch.arange(-16, 16, 0.2),
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
def test_sigmoid_tosa_INT(test_data):
    pipeline = TosaPipelineINT(
        Sigmoid(),
        (test_data(),),
        Sigmoid.aten_op,
        Sigmoid.exir_op,
        qtol=1,
        tosa_extensions=["int16"],
    )
    configure_32bit_sigmoid_quantizer(pipeline)
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
def test_sigmoid_tosa_INT_add_sigmoid(test_data):
    pipeline = TosaPipelineINT(
        SigmoidAddSigmoid(),
        (test_data(),),
        Sigmoid.aten_op,
        Sigmoid.exir_op,
        qtol=1,
        tosa_extensions=["int16"],
    )
    configure_32bit_sigmoid_quantizer(pipeline)
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_sigmoid_u55_INT(test_data):
    pipeline = OpNotSupportedPipeline(
        Sigmoid(),
        (test_data(),),
        {Sigmoid.exir_op: 1},
        quantize=True,
        u55_subset=True,
        tosa_extensions=["int16"],
    )
    configure_32bit_sigmoid_quantizer(pipeline)
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_sigmoid_u55_INT_add_sigmoid(test_data):
    pipeline = OpNotSupportedPipeline(
        SigmoidAddSigmoid(),
        (test_data(),),
        {Sigmoid.exir_op: 3},
        n_expected_delegates=1,
        quantize=True,
        u55_subset=True,
        tosa_extensions=["int16"],
    )
    configure_32bit_sigmoid_quantizer(pipeline)
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_sigmoid_u85_INT(test_data):
    pipeline = EthosU85PipelineINT(
        Sigmoid(),
        (test_data(),),
        Sigmoid.aten_op,
        Sigmoid.exir_op,
    )
    configure_32bit_sigmoid_quantizer(pipeline)
    pipeline.run()


@common.parametrize(
    "test_data",
    test_data_suite,
)
@common.XfailIfNoCorstone320
def test_sigmoid_u85_INT_add_sigmoid(test_data):
    pipeline = EthosU85PipelineINT(
        SigmoidAddSigmoid(),
        (test_data(),),
        Sigmoid.aten_op,
        Sigmoid.exir_op,
    )
    configure_32bit_sigmoid_quantizer(pipeline)
    pipeline.run()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch
from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_a16w8_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.test import common, conftest
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)
from executorch.backends.arm.tosa.specification import TosaSpecification
from executorch.backends.xnnpack.test.tester import Quantize

aten_op = "torch.ops.aten.sigmoid.default"  # Used for checking that we do not have softmax in the graph after decompose
exir_op = "executorch_exir_dialects_edge__ops_aten_sigmoid_default"
input_t1 = Tuple[torch.Tensor]  # Input x

test_data_suite = {
    # (test_name, test_data)
    "zeros": lambda: torch.zeros(10, 10, 10, 10),
    "ones": lambda: torch.ones(10, 10, 10),
    "rand": lambda: torch.rand(10, 10) - 0.5,
    "rand_4d": lambda: torch.rand(1, 1, 5, 10),
    "randn_pos": lambda: torch.randn(10) + 10,
    "randn_neg": lambda: torch.randn(10) - 10,
    "ramp": lambda: torch.arange(-16, 16, 0.2),
}


class Sigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(x)


class AddSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(x + x)


class SigmoidAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return x + self.sigmoid(x)


class SigmoidAddSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, y):
        return self.sigmoid((self.sigmoid(y) + self.sigmoid(x)))


@common.parametrize("test_data", test_data_suite)
def test_sigmoid_tosa_FP(test_data: torch.Tensor):
    TosaPipelineFP[input_t1](Sigmoid(), (test_data(),), aten_op, exir_op).run()


@common.parametrize("test_data", test_data_suite)
def test_sigmoid_tosa_INT(test_data: torch.Tensor):
    TosaPipelineINT[input_t1](Sigmoid(), (test_data(),), aten_op, exir_op).run()


def test_sigmoid_tosa_FP_add():
    TosaPipelineFP[input_t1](
        AddSigmoid(),
        (test_data_suite["zeros"](),),
        aten_op,
        exir_op,
    ).run()


def test_sigmoid_tosa_INT_add():
    TosaPipelineINT[input_t1](
        AddSigmoid(),
        (test_data_suite["ramp"](),),
        aten_op,
        exir_op,
    ).run()


def test_sigmoid_tosa_FP_add_2():
    TosaPipelineFP[input_t1](
        SigmoidAdd(),
        (test_data_suite["zeros"](),),
        aten_op,
        exir_op,
    ).run()


def test_sigmoid_tosa_INT_add_2():
    TosaPipelineINT[input_t1](
        SigmoidAdd(),
        (test_data_suite["zeros"](),),
        aten_op,
        exir_op,
    ).run()


def test_sigmoid_tosa_FP_add_3():
    TosaPipelineFP[input_t1](
        SigmoidAddSigmoid(),
        (test_data_suite["randn_neg"](), test_data_suite["randn_pos"]()),
        aten_op,
        exir_op,
    ).run()


def test_sigmoid_tosa_INT_3():
    TosaPipelineINT[input_t1](
        SigmoidAddSigmoid(),
        (test_data_suite["randn_neg"](), test_data_suite["randn_pos"]()),
        aten_op,
        exir_op,
    ).run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_sigmoid_u55_INT(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t1](
        Sigmoid(),
        (test_data(),),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_sigmoid_u85_INT(test_data: Tuple):
    pipeline = EthosU85PipelineINT[input_t1](
        Sigmoid(),
        (test_data(),),
        aten_op,
        exir_op,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_sigmoid_vgf_no_quant(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Sigmoid(),
        (test_data(),),
        aten_op,
        exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_sigmoid_vgf_quant(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Sigmoid(),
        (test_data(),),
        aten_op,
        exir_op,
        quantize=True,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
def test_sigmoid_vgf_no_quant_add():
    pipeline = VgfPipeline[input_t1](
        AddSigmoid(),
        (test_data_suite["zeros"](),),
        aten_op,
        exir_op,
        quantize=False,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
def test_sigmoid_vgf_quant_add():
    pipeline = VgfPipeline[input_t1](
        AddSigmoid(),
        (test_data_suite["ramp"](),),
        aten_op,
        exir_op,
        quantize=True,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
def test_sigmoid_vgf_no_quant_add_2():
    pipeline = VgfPipeline[input_t1](
        SigmoidAdd(),
        (test_data_suite["zeros"](),),
        aten_op,
        exir_op,
        quantize=False,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
def test_sigmoid_vgf_quant_add_2():
    pipeline = VgfPipeline[input_t1](
        SigmoidAdd(),
        (test_data_suite["zeros"](),),
        aten_op,
        exir_op,
        quantize=True,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
def test_sigmoid_vgf_no_quant_add_3():
    pipeline = VgfPipeline[input_t1](
        SigmoidAddSigmoid(),
        (test_data_suite["randn_neg"](), test_data_suite["randn_pos"]()),
        aten_op,
        exir_op,
        quantize=False,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
def test_sigmoid_vgf_quant_add_3():
    pipeline = VgfPipeline[input_t1](
        SigmoidAddSigmoid(),
        (test_data_suite["randn_neg"](), test_data_suite["randn_pos"]()),
        aten_op,
        exir_op,
        quantize=True,
    )
    pipeline.run()


def get_symmetric_a16w8_sigmoid_quantizer(per_channel_quantization=False):
    tosa_version = conftest.get_option("tosa_version")
    tosa_profiles = {
        "1.0": TosaSpecification.create_from_string("TOSA-1.0+INT+int16"),
    }

    quantizer = TOSAQuantizer(tosa_profiles[tosa_version])

    # Use a smaller episilon value to not greatly inflate [qmin, qmax]
    quantizer.set_global(
        get_symmetric_a16w8_quantization_config(
            is_per_channel=per_channel_quantization, epsilon=2**-16
        )
    )

    return Quantize(
        quantizer,
        get_symmetric_a16w8_quantization_config(
            is_per_channel=per_channel_quantization, epsilon=2**-16
        ),
    )


@common.parametrize("test_data", test_data_suite)
def test_sigmoid_16a8w_tosa_INT(test_data: torch.Tensor):
    """Test sigmoid operation with 16A8W quantization (16-bit activations, 8-bit weights)"""
    per_channel_quantization = False

    pipeline = TosaPipelineINT[input_t1](
        Sigmoid(),
        (test_data(),),
        aten_op,
        exir_op=[],
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
        tosa_extensions=["int16"],
    )

    pipeline.change_args(
        "quantize",
        get_symmetric_a16w8_sigmoid_quantizer(
            per_channel_quantization=per_channel_quantization
        ),
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_sigmoid_16a8w_u55_INT16(test_data: torch.Tensor):
    """Test sigmoid operation with 16A8W quantization on U55 (16-bit activations, 8-bit weights)"""
    per_channel_quantization = False

    pipeline = EthosU55PipelineINT[input_t1](
        Sigmoid(),
        (test_data(),),
        aten_op,
        exir_op,
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
    )

    pipeline.change_args(
        "quantize",
        get_symmetric_a16w8_sigmoid_quantizer(
            per_channel_quantization=per_channel_quantization
        ),
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_sigmoid_16a8w_u85_INT(test_data: torch.Tensor):
    """Test sigmoid operation with 16A8W quantization on U85 (16-bit activations, 8-bit weights)"""
    per_channel_quantization = False

    pipeline = EthosU85PipelineINT[input_t1](
        Sigmoid(),
        (test_data(),),
        aten_op,
        exir_op,
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
    )

    pipeline.change_args(
        "quantize",
        get_symmetric_a16w8_sigmoid_quantizer(
            per_channel_quantization=per_channel_quantization
        ),
    )
    pipeline.run()

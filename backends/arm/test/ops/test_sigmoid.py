# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import pytest

import torch
from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_a16w8_quantization_config,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

aten_op = "torch.ops.aten.sigmoid.default"  # Used for checking that we do not have sigmoid in the graph after decompose
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
test_data_suite_fp16 = {
    "rand_fp16": lambda: torch.rand(4, 4, dtype=torch.float16) - 0.2,
}

test_data_suite_bf16 = {
    "rand_bf16": lambda: torch.rand(4, 4, dtype=torch.bfloat16) - 0.2,
}

# Sigmoid is decomposed to neg→exp→add→reciprocal. The decomposed exp(-x)
# overflows the quantization range for large |x|, causing numerical errors in
# quantized pipelines. bf16 precision loss also compounds through the chain.
_SIGMOID_DECOMPOSE_INT8_XFAIL = (
    "Decomposed exp(-x) overflows int8 quantization for |x|>~5, "
    "known limitation of sigmoid decomposition"
)
_SIGMOID_DECOMPOSE_INT16_XFAIL = (
    "Decomposed sigmoid accumulates quantization error across "
    "exp/add/reciprocal in int16"
)


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


@common.parametrize(
    "test_data",
    test_data_suite | test_data_suite_fp16 | test_data_suite_bf16,
)
def test_sigmoid_tosa_FP(test_data: torch.Tensor):
    pipeline = TosaPipelineFP[input_t1](
        Sigmoid(),
        (test_data(),),
        [],
        tosa_extensions=["bf16"],
    )
    pipeline.add_stage_after(
        "to_edge_transform_and_lower", pipeline.tester.check_not, [exir_op]
    )
    pipeline.run()


@common.parametrize(
    "test_data",
    test_data_suite,
    xfails={"ramp": _SIGMOID_DECOMPOSE_INT8_XFAIL},
)
def test_sigmoid_tosa_INT(test_data: torch.Tensor):
    pipeline = TosaPipelineINT[input_t1](Sigmoid(), (test_data(),), [])
    pipeline.add_stage_after("quantize", pipeline.tester.check_not, [aten_op])
    pipeline.run()


def test_sigmoid_tosa_FP_add():
    pipeline = TosaPipelineFP[input_t1](
        AddSigmoid(),
        (test_data_suite["zeros"](),),
        [],
    )
    pipeline.add_stage_after(
        "to_edge_transform_and_lower", pipeline.tester.check_not, [exir_op]
    )
    pipeline.run()


@pytest.mark.xfail(reason=_SIGMOID_DECOMPOSE_INT8_XFAIL, strict=True)
def test_sigmoid_tosa_INT_add():
    pipeline = TosaPipelineINT[input_t1](
        AddSigmoid(),
        (test_data_suite["ramp"](),),
        [],
    )
    pipeline.add_stage_after("quantize", pipeline.tester.check_not, [aten_op])
    pipeline.run()


def test_sigmoid_tosa_FP_add_2():
    pipeline = TosaPipelineFP[input_t1](
        SigmoidAdd(),
        (test_data_suite["zeros"](),),
        [],
    )
    pipeline.add_stage_after(
        "to_edge_transform_and_lower", pipeline.tester.check_not, [exir_op]
    )
    pipeline.run()


def test_sigmoid_tosa_INT_add_2():
    pipeline = TosaPipelineINT[input_t1](
        SigmoidAdd(),
        (test_data_suite["zeros"](),),
        [],
    )
    pipeline.add_stage_after("quantize", pipeline.tester.check_not, [aten_op])
    pipeline.run()


def test_sigmoid_tosa_FP_add_3():
    pipeline = TosaPipelineFP[input_t1](
        SigmoidAddSigmoid(),
        (test_data_suite["randn_neg"](), test_data_suite["randn_pos"]()),
        [],
    )
    pipeline.add_stage_after(
        "to_edge_transform_and_lower", pipeline.tester.check_not, [exir_op]
    )
    pipeline.run()


def test_sigmoid_tosa_INT_3():
    pipeline = TosaPipelineINT[input_t1](
        SigmoidAddSigmoid(),
        (test_data_suite["randn_neg"](), test_data_suite["randn_pos"]()),
        [],
    )
    pipeline.add_stage_after("quantize", pipeline.tester.check_not, [aten_op])
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone300
def test_sigmoid_u55_INT(test_data: Tuple):
    pipeline = EthosU55PipelineINT[input_t1](
        Sigmoid(),
        (test_data(),),
        [],
    )
    pipeline.add_stage_after("quantize", pipeline.tester.check_not, [aten_op])
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.XfailIfNoCorstone320
def test_sigmoid_u85_INT(test_data: Tuple):
    pipeline = EthosU85PipelineINT[input_t1](
        Sigmoid(),
        (test_data(),),
        [],
    )
    pipeline.add_stage_after("quantize", pipeline.tester.check_not, [aten_op])
    pipeline.run()


@common.parametrize("test_data", test_data_suite | test_data_suite_fp16)
@common.SkipIfNoModelConverter
def test_sigmoid_vgf_no_quant(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Sigmoid(),
        (test_data(),),
        [],
        quantize=False,
    )
    pipeline.add_stage_after(
        "to_edge_transform_and_lower", pipeline.tester.check_not, [exir_op]
    )
    pipeline.run()


@common.parametrize("test_data", test_data_suite)
@common.SkipIfNoModelConverter
def test_sigmoid_vgf_quant(test_data: Tuple):
    pipeline = VgfPipeline[input_t1](
        Sigmoid(),
        (test_data(),),
        [],
        quantize=True,
    )
    pipeline.add_stage_after("quantize", pipeline.tester.check_not, [aten_op])
    pipeline.run()


@common.SkipIfNoModelConverter
def test_sigmoid_vgf_no_quant_add():
    pipeline = VgfPipeline[input_t1](
        AddSigmoid(),
        (test_data_suite["zeros"](),),
        [],
        quantize=False,
    )
    pipeline.add_stage_after(
        "to_edge_transform_and_lower", pipeline.tester.check_not, [exir_op]
    )
    pipeline.run()


@common.SkipIfNoModelConverter
def test_sigmoid_vgf_quant_add():
    pipeline = VgfPipeline[input_t1](
        AddSigmoid(),
        (test_data_suite["ramp"](),),
        [],
        quantize=True,
    )
    pipeline.add_stage_after("quantize", pipeline.tester.check_not, [aten_op])
    pipeline.run()


@common.SkipIfNoModelConverter
def test_sigmoid_vgf_no_quant_add_2():
    pipeline = VgfPipeline[input_t1](
        SigmoidAdd(),
        (test_data_suite["zeros"](),),
        [],
        quantize=False,
    )
    pipeline.add_stage_after(
        "to_edge_transform_and_lower", pipeline.tester.check_not, [exir_op]
    )
    pipeline.run()


@common.SkipIfNoModelConverter
def test_sigmoid_vgf_quant_add_2():
    pipeline = VgfPipeline[input_t1](
        SigmoidAdd(),
        (test_data_suite["zeros"](),),
        [],
        quantize=True,
    )
    pipeline.add_stage_after("quantize", pipeline.tester.check_not, [aten_op])
    pipeline.run()


@common.SkipIfNoModelConverter
def test_sigmoid_vgf_no_quant_add_3():
    pipeline = VgfPipeline[input_t1](
        SigmoidAddSigmoid(),
        (test_data_suite["randn_neg"](), test_data_suite["randn_pos"]()),
        [],
        quantize=False,
    )
    pipeline.add_stage_after(
        "to_edge_transform_and_lower", pipeline.tester.check_not, [exir_op]
    )
    pipeline.run()


@common.SkipIfNoModelConverter
def test_sigmoid_vgf_quant_add_3():
    pipeline = VgfPipeline[input_t1](
        SigmoidAddSigmoid(),
        (test_data_suite["randn_neg"](), test_data_suite["randn_pos"]()),
        [],
        quantize=True,
    )
    pipeline.add_stage_after("quantize", pipeline.tester.check_not, [aten_op])
    pipeline.run()


_A16W8_XFAILS = {
    "rand": _SIGMOID_DECOMPOSE_INT16_XFAIL,
    "rand_4d": _SIGMOID_DECOMPOSE_INT16_XFAIL,
    "ramp": _SIGMOID_DECOMPOSE_INT16_XFAIL,
}

# Use skips (not xfails) for EthosU tests to avoid conflict with
# @XfailIfNoCorstone which specifies raises=FileNotFoundError.
_A16W8_U55_SKIPS = {
    "rand": _SIGMOID_DECOMPOSE_INT16_XFAIL,
    "rand_4d": _SIGMOID_DECOMPOSE_INT16_XFAIL,
    "ramp": _SIGMOID_DECOMPOSE_INT16_XFAIL,
}
_A16W8_U85_SKIPS = {
    "rand": _SIGMOID_DECOMPOSE_INT16_XFAIL,
    "rand_4d": _SIGMOID_DECOMPOSE_INT16_XFAIL,
    "randn_neg": _SIGMOID_DECOMPOSE_INT16_XFAIL,
    "ramp": _SIGMOID_DECOMPOSE_INT16_XFAIL,
}


@common.parametrize("test_data", test_data_suite, xfails=_A16W8_XFAILS)
def test_sigmoid_16a8w_tosa_INT(test_data: torch.Tensor):
    """Test sigmoid operation with 16A8W quantization (16-bit activations, 8-bit
    weights)
    """
    per_channel_quantization = False

    pipeline = TosaPipelineINT[input_t1](
        Sigmoid(),
        (test_data(),),
        [],
        exir_op=[],
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
        tosa_extensions=["int16"],
    )
    pipeline.quantizer.set_global(
        get_symmetric_a16w8_quantization_config(
            is_per_channel=per_channel_quantization, epsilon=2**-16
        )
    )
    pipeline.add_stage_after("quantize", pipeline.tester.check_not, [aten_op])
    pipeline.run()


@common.parametrize("test_data", test_data_suite, skips=_A16W8_U55_SKIPS)
@common.XfailIfNoCorstone300
def test_sigmoid_16a8w_u55_INT16(test_data: torch.Tensor):
    """Test sigmoid operation with 16A8W quantization on U55 (16-bit
    activations, 8-bit weights)
    """
    per_channel_quantization = False

    pipeline = EthosU55PipelineINT[input_t1](
        Sigmoid(),
        (test_data(),),
        [],
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.quantizer.set_global(
        get_symmetric_a16w8_quantization_config(
            is_per_channel=per_channel_quantization, epsilon=2**-16
        )
    )
    pipeline.add_stage_after("quantize", pipeline.tester.check_not, [aten_op])
    pipeline.run()


@common.parametrize("test_data", test_data_suite, skips=_A16W8_U85_SKIPS)
@common.XfailIfNoCorstone320
def test_sigmoid_16a8w_u85_INT(test_data: torch.Tensor):
    """Test sigmoid operation with 16A8W quantization on U85 (16-bit
    activations, 8-bit weights)
    """
    per_channel_quantization = False

    pipeline = EthosU85PipelineINT[input_t1](
        Sigmoid(),
        (test_data(),),
        [],
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
    )
    pipeline.quantizer.set_global(
        get_symmetric_a16w8_quantization_config(
            is_per_channel=per_channel_quantization, epsilon=2**-16
        )
    )
    pipeline.add_stage_after("quantize", pipeline.tester.check_not, [aten_op])
    pipeline.run()

# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import conftest
import torch

from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_a16w8_quantization_config,
    get_symmetric_a8w4_quantization_config,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    OpNotSupportedPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

aten_op = "torch.ops.aten.conv_transpose2d.input"
exir_op = "executorch_exir_dialects_edge__ops_aten_convolution_default"  # No edge transpoe conv

input_t = Tuple[torch.Tensor]


class TransposeConv2d(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.deconv = torch.nn.ConvTranspose2d(**kwargs)

    def get_inputs(self):
        return (torch.randn(1, self.deconv.in_channels, 10, 10),)

    def forward(self, x):
        return self.deconv(x)


test_data_FP = {
    "basic": lambda: TransposeConv2d(
        in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1
    ),
    "output_padding": lambda: TransposeConv2d(
        in_channels=8,
        out_channels=4,
        kernel_size=3,
        stride=2,
        padding=1,
        output_padding=1,
    ),
    "nonsquare_kernel": lambda: TransposeConv2d(
        in_channels=4,
        out_channels=6,
        kernel_size=(2, 3),
        stride=(1, 2),
        padding=(0, 1),
    ),
    "non_equal_strides": lambda: TransposeConv2d(
        in_channels=4,
        out_channels=6,
        kernel_size=3,
        stride=(1, 2),
        padding=1,
    ),
    "no_bias": lambda: TransposeConv2d(
        in_channels=3,
        out_channels=5,
        kernel_size=5,
        stride=1,
        padding=2,
        bias=False,
    ),
}

test_data_INT = {
    f"{k},per_channel_quant={q}": (lambda v=v, q=q: (v(), q))
    for (k, v) in test_data_FP.items()
    for q in [True, False]
}

u55_supported_test_data_INT = {
    k: v
    for k, v in test_data_INT.items()
    if not (k.startswith("nonsquare_kernel,") or k.startswith("non_equal_strides,"))
}

reject_suite = {
    k: v
    for k, v in test_data_INT.items()
    if k.startswith("nonsquare_kernel,") or k.startswith("non_equal_strides,")
}
test_data_INT16 = {
    "basic": test_data_FP["basic"],
}


@common.parametrize("test_data", test_data_FP)
def test_transpose_conv2d_tosa_FP(test_data):
    model = test_data()
    pipeline = TosaPipelineFP[input_t](
        model,
        model.get_inputs(),
        aten_op,
        exir_op,
        run_on_tosa_ref_model=conftest.is_option_enabled("tosa_ref_model"),
    )
    pipeline.run()


@common.parametrize("test_data", test_data_INT)
def test_transpose_conv2d_tosa_INT(test_data):
    model, per_channel_quantization = test_data()
    pipeline = TosaPipelineINT[input_t](
        model,
        model.get_inputs(),
        aten_op,
        exir_op,
        per_channel_quantization=per_channel_quantization,
        qtol=1,
        run_on_tosa_ref_model=conftest.is_option_enabled("tosa_ref_model"),
    )
    pipeline.run()


_a8w4_transpose_conv_xfails = {
    k: "per-channel int4 weight quantization is not supported for transpose conv yet."
    for k in test_data_INT
    if k.endswith("per_channel_quant=True")
}


@common.parametrize("test_data", test_data_INT, xfails=_a8w4_transpose_conv_xfails)
def test_transpose_conv2d_tosa_INT_a8w4(test_data):
    model, per_channel_quantization = test_data()
    pipeline = TosaPipelineINT[input_t](
        model,
        model.get_inputs(),
        aten_op,
        exir_op,
        tosa_extensions=["int4"],
    )
    pipeline.quantizer.set_global(
        get_symmetric_a8w4_quantization_config(is_per_channel=per_channel_quantization)
    )
    pipeline.run()


@common.parametrize("test_data", test_data_INT16)
def test_transpose_conv2d_tosa_INT_a16w8(test_data):
    model = test_data()
    per_channel_quantization = False
    pipeline = TosaPipelineINT[input_t](
        model,
        model.get_inputs(),
        aten_op,
        exir_op,
        per_channel_quantization=per_channel_quantization,
        use_to_edge_transform_and_lower=True,
        tosa_extensions=["int16"],
        qtol=1,
    )
    pipeline.quantizer.set_global(
        get_symmetric_a16w8_quantization_config(is_per_channel=per_channel_quantization)
    )
    pipeline.run()


@common.parametrize("test_data", test_data_FP)
@common.SkipIfNoModelConverter
def test_transpose_conv2d_vgf_no_quant(test_data):
    model = test_data()
    pipeline = VgfPipeline[input_t](
        model,
        model.get_inputs(),
        aten_op,
        exir_op,
        quantize=False,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_INT)
@common.SkipIfNoModelConverter
def test_transpose_conv2d_vgf_quant(test_data):
    model, per_channel_quantization = test_data()
    pipeline = VgfPipeline[input_t](
        model,
        model.get_inputs(),
        aten_op,
        exir_op,
        per_channel_quantization=per_channel_quantization,
        quantize=True,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_INT16)
@common.SkipIfNoModelConverter
def test_transpose_conv2d_vgf_int16(test_data):
    model = test_data()
    per_channel_quantization = False
    pipeline = VgfPipeline[input_t](
        model,
        model.get_inputs(),
        aten_op,
        exir_op,
        per_channel_quantization=per_channel_quantization,
        quantize=True,
        use_to_edge_transform_and_lower=True,
        tosa_extensions=["int16"],
        qtol=1,
    )
    pipeline.quantizer.set_global(
        get_symmetric_a16w8_quantization_config(is_per_channel=per_channel_quantization)
    )
    pipeline.run()


@common.parametrize("test_data", test_data_INT)
@common.XfailIfNoCorstone320
def test_transpose_conv2d_u85_INT(test_data):
    model, per_channel_quantization = test_data()
    pipeline = EthosU85PipelineINT[input_t](
        model,
        model.get_inputs(),
        aten_op,
        exir_op,
        per_channel_quantization=per_channel_quantization,
    )
    pipeline.run()


@common.parametrize("test_data", u55_supported_test_data_INT)
@common.XfailIfNoCorstone300
def test_transpose_conv2d_u55_INT(test_data):
    model, per_channel_quantization = test_data()
    pipeline = EthosU55PipelineINT[input_t](
        model,
        model.get_inputs(),
        aten_op,
        exir_op,
        per_channel_quantization=per_channel_quantization,
    )
    pipeline.run()


@common.parametrize("test_data", reject_suite)
def test_transpose_conv2d_u55_INT_not_delegated(test_data):
    model, per_channel_quantization = test_data()
    OpNotSupportedPipeline(
        model,
        model.get_inputs(),
        {"executorch_exir_dialects_edge__ops_aten_convolution_default": 1},
        quantize=True,
        u55_subset=True,
    ).run()

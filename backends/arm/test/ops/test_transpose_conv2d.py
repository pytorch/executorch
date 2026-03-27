# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import conftest
import torch

from executorch.backends.arm.quantizer import QuantizationConfig
from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_a16w8_quantization_config,
    get_symmetric_a8w4_quantization_config,
    get_symmetric_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    OpNotSupportedPipeline,
    QuantizationPipeline,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)
from executorch.backends.arm.tosa.specification import TosaSpecification
from executorch.backends.test.harness.stages.quantize import Quantize
from torchao.quantization.pt2e import (
    FakeQuantize,
    FusedMovingAvgObsFakeQuantize,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, prepare_qat_pt2e
from torchao.quantization.pt2e.quantizer import QuantizationSpec

aten_op = "torch.ops.aten.conv_transpose2d.input"
exir_op = "executorch_exir_dialects_edge__ops_aten_convolution_default"  # No edge transpoe conv

input_t = Tuple[torch.Tensor]


class TransposeConv2d(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.dtype = kwargs.get("dtype", torch.float32)
        self.deconv = torch.nn.ConvTranspose2d(**kwargs)

    def get_inputs(self):
        return (torch.randn(1, self.deconv.in_channels, 10, 10, dtype=self.dtype),)

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
    "grouped": lambda: TransposeConv2d(
        in_channels=4,
        out_channels=6,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=2,
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

test_data_FP_fp16 = {
    "basic_fp16": lambda: TransposeConv2d(
        in_channels=16,
        out_channels=8,
        kernel_size=4,
        stride=2,
        padding=1,
        dtype=torch.float16,
    ),
}

test_data_INT = {
    f"{k},per_channel_quant={q}": (lambda v=v, q=q: (v(), q))
    for (k, v) in test_data_FP.items()
    for q in [True, False]
}
_grouped_per_channel_xfails = {
    k: "per-channel quantization for grouped transpose conv is not supported yet."
    for k in test_data_INT
    if k.startswith("grouped,") and k.endswith("per_channel_quant=True")
}

test_data_QAT_MODEL = {
    "qat_basic": lambda: TransposeConv2d(
        in_channels=16,
        out_channels=4,
        kernel_size=4,
        stride=2,
        padding=1,
        groups=1,
    ),
    "non_grouped": lambda: TransposeConv2d(
        in_channels=12,
        out_channels=16,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1,
    ),
    "grouped": lambda: TransposeConv2d(
        in_channels=4,
        out_channels=6,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=2,
    ),
}


def _get_per_channel_fake_quants(module: torch.nn.Module):
    result = []
    for mod in module.modules():
        if isinstance(mod, (FakeQuantize, FusedMovingAvgObsFakeQuantize)):
            observer = getattr(mod, "activation_post_process", None)
            if observer is not None and hasattr(observer, "ch_axis"):
                result.append((mod, observer))
    return result


def _get_per_channel_observers(module: torch.nn.Module):
    result = []
    for mod in module.modules():
        if isinstance(mod, MovingAveragePerChannelMinMaxObserver):
            result.append(mod)
    return result


u55_supported_test_data_INT = {
    k: v
    for k, v in test_data_INT.items()
    if not (
        k.startswith("nonsquare_kernel,")
        or k.startswith("non_equal_strides,")
        or k.startswith("grouped,")
    )
}

u55_TransposeConv_limitations = {
    "stride_3": lambda: (
        TransposeConv2d(
            in_channels=3,
            out_channels=1,
            kernel_size=(5, 5),
            stride=(3, 3),
            padding=(2, 2),
        ),
        True,
    ),
    "stride_1_2": lambda: (
        TransposeConv2d(
            in_channels=3,
            out_channels=1,
            kernel_size=(5, 5),
            stride=(1, 2),
            padding=(2, 2),
        ),
        True,
    ),
    "stride_2_1": lambda: (
        TransposeConv2d(
            in_channels=3,
            out_channels=1,
            kernel_size=(5, 5),
            stride=(2, 1),
            padding=(2, 2),
        ),
        True,
    ),
}

reject_suite = {
    k: v
    for k, v in test_data_INT.items()
    if (
        k.startswith("nonsquare_kernel,")
        or k.startswith("non_equal_strides,")
        or k.startswith("grouped,")
    )
}
test_data_INT16 = {
    "basic": test_data_FP["basic"],
}
test_data_BF16 = {
    "basic_bf16": lambda: TransposeConv2d(
        in_channels=16,
        out_channels=8,
        kernel_size=4,
        stride=2,
        padding=1,
        dtype=torch.bfloat16,
    ),
}


@common.parametrize("test_data", test_data_FP | test_data_FP_fp16 | test_data_BF16)
def test_conv_transpose2d_tosa_FP(test_data):
    model = test_data()
    inputs = model.get_inputs()
    pipeline = TosaPipelineFP[input_t](
        model,
        inputs,
        aten_op,
        exir_op,
        run_on_tosa_ref_model=conftest.is_option_enabled("tosa_ref_model"),
        tosa_extensions=["bf16"],
    )
    pipeline.run()


@common.parametrize("test_data", test_data_INT, xfails=_grouped_per_channel_xfails)
def test_conv_transpose2d_tosa_INT(test_data):
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


@common.parametrize("test_data", {"qat_basic": test_data_QAT_MODEL["qat_basic"]})
def test_conv_transpose2d_tosa_INT_qat_per_channel_quantization_pipeline(test_data):
    model = test_data()
    inputs = model.get_inputs()
    is_per_channel = True
    is_qat = True
    quantizer = TOSAQuantizer(TosaSpecification.create_from_string("TOSA-1.0+INT"))
    quantizer.set_global(
        get_symmetric_quantization_config(
            is_per_channel=is_per_channel,
            is_qat=is_qat,
        )
    )
    pipeline = QuantizationPipeline[input_t](model, inputs, quantizer)
    pipeline.change_args(
        "quantize",
        Quantize(
            quantizer,
            quantization_config=quantizer.global_config,
            is_qat=is_qat,
        ),
    )
    pipeline.run()


@common.parametrize("test_data", {"non_grouped": test_data_QAT_MODEL["non_grouped"]})
def test_conv_transpose2d_tosa_INT_qat_axis1_uses_non_fused_fake_quant(test_data):
    model = test_data()
    inputs = model.get_inputs()
    quantizer = TOSAQuantizer(TosaSpecification.create_from_string("TOSA-1.0+INT"))

    activation_qspec = QuantizationSpec(
        dtype=torch.int8,
        qscheme=torch.per_tensor_affine,
        observer_or_fake_quant_ctr=FusedMovingAvgObsFakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver
        ),
    )
    weight_qspec = QuantizationSpec(
        dtype=torch.int8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,
        observer_or_fake_quant_ctr=FusedMovingAvgObsFakeQuantize.with_args(
            observer=MovingAveragePerChannelMinMaxObserver, ch_axis=0
        ),
    )
    quantizer.set_global(
        QuantizationConfig(
            input_activation=activation_qspec,
            output_activation=activation_qspec,
            weight=weight_qspec,
            bias=None,
        )
    )

    prepared = prepare_qat_pt2e(
        torch.export.export(model, inputs, strict=True).module(), quantizer
    )
    per_channel_fqs = _get_per_channel_fake_quants(prepared)
    assert per_channel_fqs
    assert all(isinstance(mod, FakeQuantize) for mod, _ in per_channel_fqs)
    assert all(obs.ch_axis == 1 for _, obs in per_channel_fqs)


@common.parametrize("test_data", {"axis0_grouped": test_data_QAT_MODEL["grouped"]})
def test_conv_transpose2d_tosa_INT_grouped_qat_axis0_keeps_fused_fake_quant(test_data):
    model = test_data()
    inputs = model.get_inputs()
    quantizer = TOSAQuantizer(TosaSpecification.create_from_string("TOSA-1.0+INT"))

    activation_qspec = QuantizationSpec(
        dtype=torch.int8,
        qscheme=torch.per_tensor_affine,
        observer_or_fake_quant_ctr=FusedMovingAvgObsFakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver
        ),
    )
    weight_qspec = QuantizationSpec(
        dtype=torch.int8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,
        observer_or_fake_quant_ctr=FusedMovingAvgObsFakeQuantize.with_args(
            observer=MovingAveragePerChannelMinMaxObserver, ch_axis=0
        ),
    )
    quantizer.set_global(
        QuantizationConfig(
            input_activation=activation_qspec,
            output_activation=activation_qspec,
            weight=weight_qspec,
            bias=None,
        )
    )

    prepared = prepare_qat_pt2e(
        torch.export.export(model, inputs, strict=True).module(), quantizer
    )
    per_channel_fqs = _get_per_channel_fake_quants(prepared)
    assert per_channel_fqs
    assert all(
        isinstance(mod, FusedMovingAvgObsFakeQuantize) for mod, _ in per_channel_fqs
    )
    assert all(obs.ch_axis == 0 for _, obs in per_channel_fqs)


@common.parametrize("test_data", {"non_grouped": test_data_QAT_MODEL["non_grouped"]})
def test_conv_transpose2d_tosa_INT_ptq_observer_updates_axis(test_data):
    model = test_data()
    inputs = model.get_inputs()
    quantizer = TOSAQuantizer(TosaSpecification.create_from_string("TOSA-1.0+INT"))

    activation_qspec = QuantizationSpec(
        dtype=torch.int8,
        qscheme=torch.per_tensor_affine,
        observer_or_fake_quant_ctr=MovingAverageMinMaxObserver.with_args(),
    )
    weight_qspec = QuantizationSpec(
        dtype=torch.int8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,
        observer_or_fake_quant_ctr=MovingAveragePerChannelMinMaxObserver.with_args(
            ch_axis=0
        ),
    )
    quantizer.set_global(
        QuantizationConfig(
            input_activation=activation_qspec,
            output_activation=activation_qspec,
            weight=weight_qspec,
            bias=None,
        )
    )

    prepared = prepare_pt2e(
        torch.export.export(model, inputs, strict=True).module(), quantizer
    )
    per_channel_obs = _get_per_channel_observers(prepared)
    assert per_channel_obs
    assert all(obs.ch_axis == 1 for obs in per_channel_obs)


@common.parametrize("test_data", {"non_grouped": test_data_QAT_MODEL["non_grouped"]})
def test_conv_transpose2d_tosa_INT_qat_correct_qspec_wrong_ctor_axis(test_data):
    model = test_data()
    inputs = model.get_inputs()
    quantizer = TOSAQuantizer(TosaSpecification.create_from_string("TOSA-1.0+INT"))

    activation_qspec = QuantizationSpec(
        dtype=torch.int8,
        qscheme=torch.per_tensor_affine,
        observer_or_fake_quant_ctr=FusedMovingAvgObsFakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver
        ),
    )
    weight_qspec = QuantizationSpec(
        dtype=torch.int8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=1,
        observer_or_fake_quant_ctr=FusedMovingAvgObsFakeQuantize.with_args(
            observer=MovingAveragePerChannelMinMaxObserver, ch_axis=0
        ),
    )
    quantizer.set_global(
        QuantizationConfig(
            input_activation=activation_qspec,
            output_activation=activation_qspec,
            weight=weight_qspec,
            bias=None,
        )
    )

    prepared = prepare_qat_pt2e(
        torch.export.export(model, inputs, strict=True).module(), quantizer
    )
    per_channel_fqs = _get_per_channel_fake_quants(prepared)
    assert per_channel_fqs
    assert all(isinstance(mod, FakeQuantize) for mod, _ in per_channel_fqs)
    assert all(obs.ch_axis == 1 for _, obs in per_channel_fqs)


_a8w4_transpose_conv_xfails = {
    k: "per-channel int4 weight quantization is not supported for transpose conv yet."
    for k in test_data_INT
    if k.endswith("per_channel_quant=True")
}


@common.parametrize("test_data", test_data_INT, xfails=_a8w4_transpose_conv_xfails)
def test_conv_transpose2d_tosa_INT_a8w4(test_data):
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
def test_conv_transpose2d_tosa_INT_a16w8(test_data):
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


@common.parametrize("test_data", test_data_FP | test_data_FP_fp16)
@common.SkipIfNoModelConverter
def test_conv_transpose2d_vgf_no_quant(test_data):
    model = test_data()
    inputs = model.get_inputs()
    match inputs[0].dtype:
        case torch.float16:
            atol = 5e-3
            rtol = 5e-3
        case _:
            atol = 1e-3
            rtol = 1e-3
    pipeline = VgfPipeline[input_t](
        model,
        inputs,
        aten_op,
        exir_op,
        quantize=False,
        atol=atol,
        rtol=rtol,
    )
    pipeline.run()


@common.parametrize("test_data", test_data_INT, xfails=_grouped_per_channel_xfails)
@common.SkipIfNoModelConverter
def test_conv_transpose2d_vgf_quant(test_data):
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


@common.parametrize("test_data", test_data_INT, xfails=_a8w4_transpose_conv_xfails)
@common.SkipIfNoModelConverter
def test_conv_transpose2d_vgf_quant_a8w4(test_data):
    model, per_channel_quantization = test_data()
    pipeline = VgfPipeline[input_t](
        model,
        model.get_inputs(),
        aten_op,
        exir_op,
    )
    pipeline.quantizer.set_global(
        get_symmetric_a8w4_quantization_config(is_per_channel=per_channel_quantization)
    )
    pipeline.run()


@common.parametrize("test_data", test_data_INT16)
@common.SkipIfNoModelConverter
def test_conv_transpose2d_vgf_quant_a16w8(test_data):
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


@common.parametrize("test_data", test_data_INT, xfails=_grouped_per_channel_xfails)
@common.XfailIfNoCorstone320
def test_conv_transpose2d_u85_INT(test_data):
    model, per_channel_quantization = test_data()
    pipeline = EthosU85PipelineINT[input_t](
        model,
        model.get_inputs(),
        aten_op,
        exir_op,
        per_channel_quantization=per_channel_quantization,
    )
    pipeline.run()


@common.parametrize(
    "test_data", u55_supported_test_data_INT, xfails=_grouped_per_channel_xfails
)
@common.XfailIfNoCorstone300
def test_conv_transpose2d_u55_INT(test_data):
    model, per_channel_quantization = test_data()
    pipeline = EthosU55PipelineINT[input_t](
        model,
        model.get_inputs(),
        aten_op,
        exir_op,
        per_channel_quantization=per_channel_quantization,
    )
    pipeline.run()


_u55_grouped_not_delegated_xfails = {
    k: "grouped transpose conv quantization mismatch on U55 not-delegated path."
    for k in reject_suite
    if k.startswith("grouped,")
}


@common.parametrize(
    "test_data",
    reject_suite | u55_TransposeConv_limitations,
    xfails=_u55_grouped_not_delegated_xfails,
)
def test_conv_transpose2d_u55_INT_not_delegated(test_data):
    model, per_channel_quantization = test_data()
    OpNotSupportedPipeline(
        model,
        model.get_inputs(),
        {"executorch_exir_dialects_edge__ops_aten_convolution_default": 1},
        quantize=True,
        u55_subset=True,
    ).run()

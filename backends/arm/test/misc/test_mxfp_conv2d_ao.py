# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.ao_ext import MXFPOpConfig, to_mxfp
from executorch.backends.arm.ao_ext.mxfp import mxfp_dtype_to_str, MXFPDType
from executorch.backends.arm.ao_ext.ops import MXFPConv2dOp
from torch.export import export
from torchao.prototype.mx_formats.mx_tensor import DTYPE_FP6_E2M3, DTYPE_FP6_E3M2


IN_CHANNELS = 64
OUT_CHANNELS = 8


class Conv2dModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            IN_CHANNELS,
            OUT_CHANNELS,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        with torch.no_grad():
            weight = torch.linspace(
                -1.0,
                1.0,
                self.conv.weight.numel(),
            ).reshape_as(self.conv.weight)
            channel_scales = torch.cat(
                (
                    torch.linspace(0.125, 0.5, IN_CHANNELS // 2),
                    torch.linspace(0.75, 1.5, IN_CHANNELS // 2),
                )
            ).view(1, IN_CHANNELS, 1, 1)
            output_scales = torch.linspace(0.5, 1.5, OUT_CHANNELS).view(
                OUT_CHANNELS,
                1,
                1,
                1,
            )
            self.conv.weight.copy_(weight * channel_scales * output_scales)
            assert self.conv.bias is not None
            self.conv.bias.copy_(torch.linspace(-0.25, 0.25, OUT_CHANNELS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _test_mxfp_conv2d_quantize_swaps_module(
    weight_dtype: MXFPDType,
    expected_weight_qdata_dtype: torch.dtype,
    expected_weight_qdata_shape: tuple[int, ...],
) -> None:
    model = Conv2dModule().eval()

    to_mxfp(
        model,
        MXFPOpConfig(weight_dtype=weight_dtype),
    )

    assert isinstance(model.conv, MXFPConv2dOp)
    assert model.conv.weight_qdata.dtype == expected_weight_qdata_dtype
    assert model.conv.weight_dtype == mxfp_dtype_to_str(weight_dtype)
    assert model.conv.weight_scale.dtype == torch.float8_e8m0fnu
    assert tuple(model.conv.weight_qdata.shape) == expected_weight_qdata_shape
    assert tuple(model.conv.weight_scale.shape) == (
        OUT_CHANNELS,
        3,
        3,
        IN_CHANNELS // 32,
    )
    assert torch.unique(model.conv.weight_scale.to(torch.float32)).numel() > 1


def test_mxfp8_e4m3_conv2d_quantize_swaps_module() -> None:
    _test_mxfp_conv2d_quantize_swaps_module(
        torch.float8_e4m3fn,
        torch.float8_e4m3fn,
        (OUT_CHANNELS, 3, 3, IN_CHANNELS),
    )


def test_mxfp4_conv2d_quantize_swaps_module() -> None:
    _test_mxfp_conv2d_quantize_swaps_module(
        torch.float4_e2m1fn_x2,
        torch.uint8,
        (OUT_CHANNELS, 3, 3, IN_CHANNELS // 2),
    )


def test_mxfp6_e2m3_conv2d_quantize_swaps_module() -> None:
    _test_mxfp_conv2d_quantize_swaps_module(
        DTYPE_FP6_E2M3,
        torch.uint8,
        (OUT_CHANNELS, 3, 3, IN_CHANNELS),
    )


def test_mxfp6_e3m2_conv2d_quantize_swaps_module() -> None:
    _test_mxfp_conv2d_quantize_swaps_module(
        DTYPE_FP6_E3M2,
        torch.uint8,
        (OUT_CHANNELS, 3, 3, IN_CHANNELS),
    )


def test_mxfp_conv2d_quantize_filter_fn_selects_modules() -> None:
    class TwoConv2dModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.selected = torch.nn.Conv2d(IN_CHANNELS, OUT_CHANNELS, 3, padding=1)
            self.skipped = torch.nn.Conv2d(IN_CHANNELS, OUT_CHANNELS, 3, padding=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.selected(x) + self.skipped(x)

    def _is_selected_conv2d(module: torch.nn.Module, fqn: str) -> bool:
        return isinstance(module, torch.nn.Conv2d) and fqn == "selected"

    model = TwoConv2dModule().eval()

    to_mxfp(
        model,
        MXFPOpConfig(weight_dtype=torch.float8_e4m3fn),
        filter_fn=_is_selected_conv2d,
    )

    assert isinstance(model.selected, MXFPConv2dOp)
    assert isinstance(model.skipped, torch.nn.Conv2d)


def test_mxfp_conv2d_quantize_supports_fp4_weights() -> None:
    model = Conv2dModule().eval()

    to_mxfp(
        model,
        MXFPOpConfig(weight_dtype=torch.float4_e2m1fn_x2),
    )

    assert isinstance(model.conv, MXFPConv2dOp)
    assert model.conv.weight_qdata.dtype == torch.uint8
    assert model.conv.weight_scale.dtype == torch.float8_e8m0fnu
    assert tuple(model.conv.weight_qdata.shape) == (
        OUT_CHANNELS,
        3,
        3,
        IN_CHANNELS // 2,
    )
    assert tuple(model.conv.weight_scale.shape) == (
        OUT_CHANNELS,
        3,
        3,
        IN_CHANNELS // 32,
    )


def test_mxfp_conv2d_preserves_bfloat16_output_dtype() -> None:
    model = Conv2dModule().eval().to(torch.bfloat16)
    to_mxfp(
        model,
        MXFPOpConfig(weight_dtype=torch.float8_e4m3fn),
    )

    output = model(torch.randn(1, IN_CHANNELS, 8, 8, dtype=torch.bfloat16))

    assert isinstance(model.conv, MXFPConv2dOp)
    assert model.conv.output_dtype == torch.bfloat16
    assert output.dtype == torch.bfloat16


def test_mxfp_conv2d_op_output_dtype_constructor_arg() -> None:
    model = Conv2dModule().eval()
    config = MXFPOpConfig(weight_dtype=torch.float8_e4m3fn)
    to_mxfp(
        model,
        config,
    )
    assert isinstance(model.conv, MXFPConv2dOp)

    fp32_conv = MXFPConv2dOp(
        model.conv.weight_qdata,
        model.conv.weight_scale,
        model.conv.bias,
        model.conv.stride,
        model.conv.padding,
        model.conv.dilation,
        model.conv.groups,
        config.weight_dtype,
        config.block_size,
    )
    bf16_conv = MXFPConv2dOp(
        model.conv.weight_qdata,
        model.conv.weight_scale,
        model.conv.bias,
        model.conv.stride,
        model.conv.padding,
        model.conv.dilation,
        model.conv.groups,
        config.weight_dtype,
        config.block_size,
        output_dtype=torch.bfloat16,
    )

    test_input = torch.randn(1, IN_CHANNELS, 8, 8)

    assert fp32_conv.output_dtype == torch.float32
    assert fp32_conv(test_input).dtype == torch.float32
    assert bf16_conv.output_dtype == torch.bfloat16
    assert bf16_conv(test_input).dtype == torch.bfloat16


def _test_mxfp_conv2d_export_preserves_custom_op(config: MXFPOpConfig) -> None:
    model = Conv2dModule().eval()
    to_mxfp(model, config)

    exported = export(model, (torch.randn(1, IN_CHANNELS, 8, 8),), strict=False)

    targets = [
        node.target
        for node in exported.graph_module.graph.nodes
        if node.op == "call_function"
    ]

    assert torch.ops.tosa_mxfp.conv2d.default in targets


def test_mxfp8_e4m3_conv2d_export_preserves_custom_op() -> None:
    _test_mxfp_conv2d_export_preserves_custom_op(
        MXFPOpConfig(weight_dtype=torch.float8_e4m3fn)
    )


def test_mxfp4_conv2d_export_preserves_custom_op() -> None:
    _test_mxfp_conv2d_export_preserves_custom_op(
        MXFPOpConfig(weight_dtype=torch.float4_e2m1fn_x2)
    )


def test_mxfp6_e2m3_conv2d_export_preserves_custom_op() -> None:
    _test_mxfp_conv2d_export_preserves_custom_op(
        MXFPOpConfig(weight_dtype=DTYPE_FP6_E2M3)
    )


def test_mxfp6_e3m2_conv2d_export_preserves_custom_op() -> None:
    _test_mxfp_conv2d_export_preserves_custom_op(
        MXFPOpConfig(weight_dtype=DTYPE_FP6_E3M2)
    )


def test_mxfp_conv2d_export_preserves_inferred_bfloat16_output_dtype() -> None:
    model = Conv2dModule().eval().to(torch.bfloat16)
    to_mxfp(
        model,
        MXFPOpConfig(weight_dtype=torch.float8_e4m3fn),
    )

    exported = export(
        model,
        (torch.randn(1, IN_CHANNELS, 8, 8, dtype=torch.bfloat16),),
        strict=False,
    )

    cast_nodes = [
        node
        for node in exported.graph_module.graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten.to.dtype
    ]

    assert len(cast_nodes) == 1
    assert cast_nodes[0].args[1] == torch.bfloat16
    assert cast_nodes[0].meta["val"].dtype == torch.bfloat16
    cast_input = cast_nodes[0].args[0]
    assert isinstance(cast_input, torch.fx.Node)
    assert cast_input.target == torch.ops.tosa_mxfp.conv2d.default


def test_mxfp_conv2d_cpu_impl_matches_ref() -> None:
    ref_model = Conv2dModule().eval()
    test_model = Conv2dModule().eval()
    test_model.load_state_dict(ref_model.state_dict())

    to_mxfp(test_model, MXFPOpConfig())

    x = torch.linspace(-0.03, 0.03, IN_CHANNELS * 8 * 8).reshape(
        1,
        IN_CHANNELS,
        8,
        8,
    )
    test_output = test_model(x)
    ref_output = ref_model(x)

    torch.testing.assert_close(test_output, ref_output, rtol=0.1, atol=0.1)


def test_mxfp4_conv2d_cpu_impl_matches_ref() -> None:
    ref_model = Conv2dModule().eval()
    test_model = Conv2dModule().eval()
    test_model.load_state_dict(ref_model.state_dict())

    to_mxfp(test_model, MXFPOpConfig(weight_dtype=torch.float4_e2m1fn_x2))

    x = torch.linspace(-0.03, 0.03, IN_CHANNELS * 8 * 8).reshape(
        1,
        IN_CHANNELS,
        8,
        8,
    )
    test_output = test_model(x)
    ref_output = ref_model(x)

    torch.testing.assert_close(test_output, ref_output, rtol=0.5, atol=0.5)

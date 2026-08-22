# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.ao_ext import MXFPOpConfig, to_mxfp
from executorch.backends.arm.ao_ext.mxfp import mxfp_dtype_to_str, MXFPDType
from executorch.backends.arm.ao_ext.ops import MXFPLinearOp

from torch.export import export
from torchao.prototype.mx_formats.mx_tensor import DTYPE_FP6_E2M3, DTYPE_FP6_E3M2


class LinearModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(32, 8, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _test_mxfp_linear_quantize_swaps_module(
    weight_dtype: MXFPDType,
    expected_weight_qdata_dtype: torch.dtype,
    expected_weight_qdata_shape: tuple[int, ...],
) -> None:
    model = LinearModule().eval()

    to_mxfp(
        model,
        MXFPOpConfig(weight_dtype=weight_dtype),
    )

    assert isinstance(model.linear, MXFPLinearOp)
    assert model.linear.weight_qdata.dtype == expected_weight_qdata_dtype
    assert model.linear.weight_dtype == mxfp_dtype_to_str(weight_dtype)
    assert model.linear.weight_scale.dtype == torch.float8_e8m0fnu
    assert tuple(model.linear.weight_qdata.shape) == expected_weight_qdata_shape
    assert tuple(model.linear.weight_scale.shape) == (1, 8, 1)


def test_mxfp8_e4m3_linear_quantize_swaps_module() -> None:
    _test_mxfp_linear_quantize_swaps_module(
        torch.float8_e4m3fn,
        torch.float8_e4m3fn,
        (1, 8, 32),
    )


def test_mxfp4_linear_quantize_swaps_module() -> None:
    _test_mxfp_linear_quantize_swaps_module(
        torch.float4_e2m1fn_x2,
        torch.uint8,
        (1, 8, 16),
    )


def test_mxfp6_e2m3_linear_quantize_swaps_module() -> None:
    _test_mxfp_linear_quantize_swaps_module(
        DTYPE_FP6_E2M3,
        torch.uint8,
        (1, 8, 32),
    )


def test_mxfp6_e3m2_linear_quantize_swaps_module() -> None:
    _test_mxfp_linear_quantize_swaps_module(
        DTYPE_FP6_E3M2,
        torch.uint8,
        (1, 8, 32),
    )


def test_mxfp_linear_quantize_filter_fn_selects_modules() -> None:
    class TwoLinearModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.selected = torch.nn.Linear(32, 8)
            self.skipped = torch.nn.Linear(32, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.selected(x) + self.skipped(x)

    def _is_selected_linear(module: torch.nn.Module, fqn: str) -> bool:
        return isinstance(module, torch.nn.Linear) and fqn == "selected"

    model = TwoLinearModule().eval()

    to_mxfp(
        model,
        MXFPOpConfig(weight_dtype=torch.float8_e4m3fn),
        filter_fn=_is_selected_linear,
    )

    assert isinstance(model.selected, MXFPLinearOp)
    assert isinstance(model.skipped, torch.nn.Linear)


def test_mxfp_linear_preserves_bfloat16_output_dtype() -> None:
    model = LinearModule().eval().to(torch.bfloat16)
    to_mxfp(
        model,
        MXFPOpConfig(weight_dtype=torch.float8_e4m3fn),
    )

    output = model(torch.randn(4, 32, dtype=torch.bfloat16))

    assert isinstance(model.linear, MXFPLinearOp)
    assert model.linear.output_dtype == torch.bfloat16
    assert output.dtype == torch.bfloat16


def test_mxfp_linear_op_output_dtype_constructor_arg() -> None:
    model = LinearModule().eval()
    config = MXFPOpConfig(weight_dtype=torch.float8_e4m3fn)
    to_mxfp(
        model,
        config,
    )
    assert isinstance(model.linear, MXFPLinearOp)

    fp32_linear = MXFPLinearOp(
        model.linear.weight_qdata,
        model.linear.weight_scale,
        model.linear.bias,
        config.weight_dtype,
        config.block_size,
    )
    bf16_linear = MXFPLinearOp(
        model.linear.weight_qdata,
        model.linear.weight_scale,
        model.linear.bias,
        config.weight_dtype,
        config.block_size,
        output_dtype=torch.bfloat16,
    )

    test_input = torch.randn(4, 32)

    assert fp32_linear.output_dtype == torch.float32
    assert fp32_linear(test_input).dtype == torch.float32
    assert bf16_linear.output_dtype == torch.bfloat16
    assert bf16_linear(test_input).dtype == torch.bfloat16


def _test_mxfp_linear_export_preserves_custom_op(config: MXFPOpConfig) -> None:
    model = LinearModule().eval()
    to_mxfp(model, config)

    exported = export(model, (torch.randn(4, 32),), strict=False)

    targets = [
        node.target
        for node in exported.graph_module.graph.nodes
        if node.op == "call_function"
    ]

    assert torch.ops.tosa_mxfp.linear.default in targets


def test_mxfp8_e4m3_linear_export_preserves_custom_op() -> None:
    _test_mxfp_linear_export_preserves_custom_op(
        MXFPOpConfig(weight_dtype=torch.float8_e4m3fn)
    )


def test_mxfp4_linear_export_preserves_custom_op() -> None:
    _test_mxfp_linear_export_preserves_custom_op(
        MXFPOpConfig(weight_dtype=torch.float4_e2m1fn_x2)
    )


def test_mxfp6_e2m3_linear_export_preserves_custom_op() -> None:
    _test_mxfp_linear_export_preserves_custom_op(
        MXFPOpConfig(weight_dtype=DTYPE_FP6_E2M3)
    )


def test_mxfp6_e3m2_linear_export_preserves_custom_op() -> None:
    _test_mxfp_linear_export_preserves_custom_op(
        MXFPOpConfig(weight_dtype=DTYPE_FP6_E3M2)
    )


def test_mxfp_linear_export_preserves_inferred_bfloat16_output_dtype() -> None:
    model = LinearModule().eval().to(torch.bfloat16)
    to_mxfp(
        model,
        MXFPOpConfig(weight_dtype=torch.float8_e4m3fn),
    )

    exported = export(model, (torch.randn(4, 32, dtype=torch.bfloat16),), strict=False)

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
    assert cast_input.target == torch.ops.tosa_mxfp.linear.default

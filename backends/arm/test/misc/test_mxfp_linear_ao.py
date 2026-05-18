# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.ao_ext import MXFPOpConfig, to_mxfp
from executorch.backends.arm.ao_ext.ops import MXFPLinearOp

from torch.export import export


class LinearModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(32, 8, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_mxfp_linear_quantize_swaps_module() -> None:
    model = LinearModule().eval()

    to_mxfp(model, MXFPOpConfig())

    assert isinstance(model.linear, MXFPLinearOp)
    assert model.linear.weight_qdata.dtype == torch.float8_e4m3fn
    assert model.linear.weight_scale.dtype == torch.float8_e8m0fnu
    assert tuple(model.linear.weight_qdata.shape) == (1, 8, 32)
    assert tuple(model.linear.weight_scale.shape) == (1, 8, 1)


def test_mxfp4_linear_quantize_swaps_module() -> None:
    model = LinearModule().eval()

    to_mxfp(
        model,
        MXFPOpConfig(weight_dtype=torch.float4_e2m1fn_x2),
    )

    assert isinstance(model.linear, MXFPLinearOp)
    assert model.linear.weight_qdata.dtype == torch.uint8
    assert model.linear.weight_scale.dtype == torch.float8_e8m0fnu
    assert tuple(model.linear.weight_qdata.shape) == (1, 8, 16)
    assert tuple(model.linear.weight_scale.shape) == (1, 8, 1)


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

    to_mxfp(model, MXFPOpConfig(), filter_fn=_is_selected_linear)

    assert isinstance(model.selected, MXFPLinearOp)
    assert isinstance(model.skipped, torch.nn.Linear)


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


def test_mxfp_linear_export_preserves_custom_op() -> None:
    _test_mxfp_linear_export_preserves_custom_op(MXFPOpConfig())


def test_mxfp4_linear_export_preserves_custom_op() -> None:
    _test_mxfp_linear_export_preserves_custom_op(
        MXFPOpConfig(weight_dtype=torch.float4_e2m1fn_x2)
    )

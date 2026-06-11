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


def test_mxfp_linear_export_preserves_custom_op() -> None:
    model = LinearModule().eval()
    to_mxfp(model, MXFPOpConfig())

    exported = export(model, (torch.randn(4, 32),), strict=False)

    targets = [
        node.target
        for node in exported.graph_module.graph.nodes
        if node.op == "call_function"
    ]

    assert torch.ops.tosa_mxfp.linear.default in targets

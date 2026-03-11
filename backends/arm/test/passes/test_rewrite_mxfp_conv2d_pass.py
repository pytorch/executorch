# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

import executorch.backends.arm.tosa.dialect  # noqa: F401
import torch
from executorch.backends.arm._passes.rewrite_mxfp_conv2d import RewriteMXFPConv2dPass
from executorch.backends.arm.ao_ext import MXFPOpConfig, to_mxfp
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export


class _Conv2dModule(torch.nn.Module):
    def __init__(self, bias: bool = True) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            32,
            8,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _is_conv2d(module: torch.nn.Module, _fqn: str) -> bool:
    return isinstance(module, torch.nn.Conv2d)


def _targets(graph_module: torch.fx.GraphModule) -> list[object]:
    return [
        node.target for node in graph_module.graph.nodes if node.op == "call_function"
    ]


def test_rewrite_mxfp_conv2d_replaces_custom_op() -> None:
    model = _Conv2dModule(bias=True).eval()
    to_mxfp(model, MXFPOpConfig(), filter_fn=_is_conv2d)
    exported = export(model, (torch.randn(1, 32, 10, 12),), strict=False)
    tosa_spec = TosaSpecification.create_from_string("TOSA-1.1+FP+mxfp")

    with TosaLoweringContext(tosa_spec):
        graph_module = (
            RewriteMXFPConv2dPass(exported).call(exported.graph_module).graph_module
        )

    targets = _targets(graph_module)

    assert torch.ops.tosa_mxfp.conv2d.default not in targets
    assert exir_ops.backend.tosa.CAST_TO_BLOCK_SCALED.default in targets
    assert exir_ops.backend.tosa.CONV2D_BLOCK_SCALED.default in targets
    assert exir_ops.edge.aten.permute_copy.default in targets
    assert operator.getitem in targets


def test_rewrite_mxfp_conv2d_restores_output_shape() -> None:
    model = _Conv2dModule(bias=False).eval()
    to_mxfp(model, MXFPOpConfig(), filter_fn=_is_conv2d)
    exported = export(model, (torch.randn(1, 32, 10, 12),), strict=False)
    tosa_spec = TosaSpecification.create_from_string("TOSA-1.1+FP+mxfp")

    with TosaLoweringContext(tosa_spec):
        graph_module = (
            RewriteMXFPConv2dPass(exported).call(exported.graph_module).graph_module
        )

    conv_node = next(
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function"
        and node.target == exir_ops.backend.tosa.CONV2D_BLOCK_SCALED.default
    )
    output_node = next(
        node
        for node in reversed(tuple(graph_module.graph.nodes))
        if node.op == "call_function"
        and node.target == exir_ops.edge.aten.permute_copy.default
    )

    assert tuple(conv_node.meta["val"].shape) == (1, 5, 6, 8)
    assert tuple(output_node.meta["val"].shape) == (1, 8, 5, 6)
